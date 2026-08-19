import argparse
import contextlib
import logging
import multiprocessing as mp
import os
import time
from copy import deepcopy
from functools import partial

import ale_py
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.atari_wrappers import ClipRewardEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from sb3_contrib.rainbow.rainbow import Rainbow
from sb3_contrib.rainbow.rainbow_policy import FactorizedNoisyLinear, NatureC51, RainbowPolicy

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def choose_eval_action(observation, eval_net, device):
    # evaluation protocol: greedy actions, noisy-net noise disabled
    # noise is zeroed once on the eval net after loading, do not reset it here
    with torch.no_grad():
        state = torch.tensor(observation, dtype=torch.float32).to(device)
        qvals = eval_net.qvals(state, advantages_only=True)
        action = torch.argmax(qvals, dim=1).cpu()
    return action


def make_env(envs_create, game, framestack, repeat_probs, terminal_on_life_loss=True, clip_rewards=True):
    """Build the vectorized Atari env.

    Wrapper order matters: Monitor sits below ClipRewardEnv so logged episode
    returns info["episode"]["r"] are raw game scores, while the agent trains on
    clipped rewards. Evaluation envs disable clipping so scores read directly
    from step() are raw.
    """
    logger.debug("Creating environments.")
    def make_single_env():
        env = gym.make("ALE/" + game + "-v5", frameskip=1, repeat_action_probability=repeat_probs)
        env = gym.wrappers.AtariPreprocessing(env, terminal_on_life_loss=terminal_on_life_loss)
        env = Monitor(env)
        if clip_rewards:
            env = ClipRewardEnv(env)
        env = gym.wrappers.FrameStackObservation(env, framestack)
        return env

    all_envs = SubprocVecEnv([make_single_env for _ in range(envs_create)])
    logger.debug("Environments created.")
    return all_envs


def create_network(framestack, n_actions, device, linear_size):
    return NatureC51(framestack, n_actions, device=device, linear_size=linear_size)


def non_default_args(args, parser):
    result = []
    for arg in vars(args):
        user_val = getattr(args, arg)
        default_val = parser.get_default(arg)
        if user_val != default_val and default_val != "NameThisGame" and arg != "include_evals" and arg != "eval_envs"\
                and arg != "num_eval_episodes":

            result.append(f"{arg}={user_val}")
    return ', '.join(result)


def format_arguments(arg_string):
    arg_string = arg_string.replace('=', '')
    arg_string = arg_string.replace('True', '1')
    arg_string = arg_string.replace('False', '0')
    arg_string = arg_string.replace(', ', '_')
    return arg_string


def evaluate_agent(net_state_dict, network_creator, eval_envs, num_eval_episodes, agent_name, testing, game,
                   n_actions, device, index, framestack, repeat_probs):

    # paper evaluates on full episodes (life loss is NOT terminal during eval)
    eval_env = make_env(
        eval_envs,
        game,
        framestack,
        repeat_probs,
        terminal_on_life_loss=False,
        clip_rewards=False,
    )
    evals = []
    eval_episodes = 0
    eval_scores = np.array([0 for i in range(eval_envs)])
    eval_observation = eval_env.reset()

    eval_net = network_creator()

    # move state dict to gpu - pytorch doesn't allow sharing across threads on gpu
    state_dict_gpu = {k: v.to(device) for k, v in net_state_dict.items()}

    eval_net.load_state_dict(state_dict_gpu)

    for m in eval_net.modules():
        if isinstance(m, FactorizedNoisyLinear):
            m.disable_noise()

    logger.debug(f"Evaluating for {num_eval_episodes} episodes.")
    while eval_episodes < num_eval_episodes:

        eval_action = choose_eval_action(eval_observation, eval_net, device)
        eval_observation_, eval_reward, eval_done_, eval_info = eval_env.step(eval_action)

        for i in range(eval_envs):
            eval_scores[i] += eval_reward[i]
            if eval_done_[i]:
                eval_episodes += 1
                evals.append(eval_scores[i])
                eval_scores[i] = 0
                if eval_episodes >= num_eval_episodes:
                    break

        eval_observation = eval_observation_

    if not testing:
        fname = f"{agent_name}_Evaluation.npy"
        logger.debug(f"Loading {fname}")
        data = np.load(fname)

        # Update the specified index in the 0th dimension
        data[index] = evals
        logger.info(f"Evaluation {index + 1} M Complete, average score:")
        logger.info(f"{np.mean(evals)}")

        # Save the updated array back to the file
        np.save(fname, data)
    eval_env.close()


class RainbowLoopCallback(BaseCallback):
    """Replaces the old hand-rolled training loop.

    The env must only be stepped by SB3's learn(). This callback reproduces
    the old loop's responsibilities: raw-score tracking via Monitor,
    progress printing, periodic evaluation in a background process and
    model checkpoints.
    """

    def __init__(
        self,
        agent_name,
        game,
        testing,
        include_evals,
        eval_every,
        eval_envs,
        num_eval_episodes,
        framestack,
        repeat_probs,
        n_actions,
        device,
        linear_size,
        total_steps,
        print_every=10_000,
    ):
        super().__init__()
        self.agent_name = agent_name
        self.game = game
        self.testing = testing
        self.include_evals = include_evals
        self.eval_every = eval_every
        self.eval_envs = eval_envs
        self.num_eval_episodes = num_eval_episodes
        self.framestack = framestack
        self.repeat_probs = repeat_probs
        self.n_actions = n_actions
        self.eval_device = device
        self.linear_size = linear_size
        self.total_steps = total_steps
        self.print_every = print_every
        self.next_eval = eval_every

        self.scores = []
        self.scores_temp = []
        self.episodes = 0
        self.current_eval = 0
        self.next_print = print_every
        self.last_steps = 0
        self.last_time = time.time()
        self.last_eval_step = -1
        self.processes = []

    def _on_step(self):
        for info in self.locals["infos"]:
            ep = info.get("episode")
            if ep is not None:
                self.episodes += 1
                self.scores.append([ep["r"], self.num_timesteps])
                self.scores_temp.append(ep["r"])

        if self.num_timesteps >= self.next_print and len(self.scores) > 0:
            avg_score = np.mean(self.scores_temp[-50:])
            now = time.time()
            fps = (self.num_timesteps - self.last_steps) / (now - self.last_time)

            logger.info(
                "{} {} avg score {:.2f} total_steps {:.0f} fps {:.2f} games {}".format(
                    self.agent_name,
                    self.game,
                    avg_score,
                    self.num_timesteps,
                    fps,
                    self.episodes,
                ),
            )

            self.last_steps = self.num_timesteps
            self.last_time = now
            self.next_print += self.print_every

        if self.num_timesteps >= self.next_eval:
            self._run_eval()
            self.next_eval += self.eval_every

        return True

    def _run_eval(self):
        logger.info(f"Evaluating: {self.current_eval}. (Testing: {self.testing})")
        self.last_eval_step = self.num_timesteps

        if not self.testing and (self.current_eval + 1) in (1, 10, 50, 100, 150, 200):
            self.model.q_net.save_checkpoint(
                self.agent_name + "_" + str(int(self.num_timesteps // 250000)) + "M"
            )

        if not self.testing:
            np.save(f"{self.agent_name}_Experiment.npy", np.array(self.scores))

        if self.include_evals:
            logger.debug(f"Joining {len(self.processes)} previous eval processes")
            for process in self.processes:
                logger.debug(f"Waiting for PID {process.pid}")
                process.join()
                logger.debug(f"PID {process.pid} completed")
            self.processes = []
            logger.debug("All joins completed")

            self.model.disable_noise(self.model.q_net)
            logger.debug("Copying model to CPU")
            net_state_dict = deepcopy({k: v.cpu() for k, v in self.model.q_net.state_dict().items()})
            logger.debug("CPU copy complete")
            network_creator = partial(
                create_network,
                self.framestack,
                self.n_actions,
                self.eval_device,
                self.linear_size,
            )

            eval_process = mp.Process(
                target=evaluate_agent,
                args=(
                    net_state_dict,
                    network_creator,
                    self.eval_envs,
                    self.num_eval_episodes,
                    self.agent_name,
                    self.testing,
                    self.game,
                    self.n_actions,
                    self.eval_device,
                    self.current_eval,
                    self.framestack,
                    self.repeat_probs,
                ),
            )
            eval_process.start()
            self.processes.append(eval_process)

        self.current_eval += 1

    def _on_training_end(self):
        if self.last_eval_step < self.total_steps:
            self._run_eval()

        if not self.testing:
            np.save(f"{self.agent_name}_Experiment.npy", np.array(self.scores))

        for process in self.processes:
            process.join()
        self.processes = []


def main():
    parser = argparse.ArgumentParser()

    # environment setup
    parser.add_argument('--game', type=str, default="NameThisGame")

    parser.add_argument('--envs', type=int, default=64) # parallel envs
    parser.add_argument('--frames', type=int, default=200_000_000) # total frames (frames / 4 = steps due to frameskip)
    parser.add_argument('--eval_envs', type=int, default=5)

    parser.add_argument('--bs', type=int, default=32)  # Rainbow paper batch size

    parser.add_argument('--repeat', type=int, default=0)  # this is just for repeating experiments (multiple seeds)
    parser.add_argument('--include_evals', type=int, default=1)  # use the evaluation protocol where every 250k steps, we evaluate the agent

    parser.add_argument('--num_eval_episodes', type=int, default=20)
    parser.add_argument('--framestack', type=int, default=4)
    parser.add_argument('--sticky', type=int, default=1)  # sticky actions

    # agent setup
    parser.add_argument('--nstep', type=int, default=3)  # n-step Q-learning
    parser.add_argument('--lr', type=float, default=6.25e-5)  # learning rate
    parser.add_argument('--testing', type=bool, default=False)  # testing mode
    parser.add_argument('--grad_clip', type=int, default=10)  # gradient clipping - not mentioned in Rainbow DQN, but used in DQN and was likely kept

    parser.add_argument('--discount', type=float, default=0.99)  # discount factor
    parser.add_argument('--target_replace_frames', type=int, default=32_000)  # target network update frequency in frames
    parser.add_argument('--linear_size', type=int, default=512)  # linear size of the network
    parser.add_argument('--per_alpha', type=float, default=0.5)  # priority exponent for PER
    # gradient steps per env-transition. Rainbow paper: one batch-32 update every
    # 4 agent steps -> 0.25. With 64 envs this is 16 gradient steps per vector step.
    parser.add_argument('--replay_ratio', type=float, default=0.25)
    # torch.compile with mode="max-autotune" (significant speedup; CUDA + Linux only,
    # silently skipped elsewhere). 0 disables.
    parser.add_argument("--compile", type=int, default=1)

    args = parser.parse_args()

    arg_string = non_default_args(args, parser)
    logger.debug(f"Args: {args}")
    formatted_string = format_arguments(arg_string)
    logger.info(f"Formatted args: {formatted_string}")

    compile_mode = "max-autotune" if args.compile else None

    game = args.game
    envs = args.envs
    bs = args.bs
    # convert target-replace period from environment frames to gradient steps.
    # frames / 4 -> env-steps; * replay_ratio -> gradient steps.
    # 32k frames at replay_ratio=0.25 gives 2000 gradient steps, independent of env count.
    c = int((args.target_replace_frames / 4) * args.replay_ratio)
    lr = args.lr

    num_eval_episodes = args.num_eval_episodes
    framestack = args.framestack
    sticky = args.sticky
    repeat_probs = 0 if not sticky else 0.25

    nstep = args.nstep
    grad_clip = args.grad_clip
    discount = args.discount
    linear_size = args.linear_size
    total_steps = args.frames // 4
    per_alpha = args.per_alpha
    replay_ratio = args.replay_ratio

    lr_str = "{:e}".format(lr)
    lr_str = str(lr_str).replace(".", "").replace("0", "")
    frame_name = str(int(args.frames / 1000000)) + "M"

    include_evals = bool(args.include_evals)
    agent_name = "Rainbow_" + game + frame_name

    if len(formatted_string) > 2:
        agent_name += '_' + formatted_string

    logger.info(f"Agent Name: {agent_name}")
    testing = args.testing

    # creates new directory for results and models
    if not testing:
        counter = 0
        while True:
            if counter == 0:
                new_dir_name = agent_name
            else:
                new_dir_name = f"{agent_name}_{counter}"
            if not os.path.exists(new_dir_name):
                break
            counter += 1
        os.mkdir(new_dir_name)
        logger.info(f"Created directory: {new_dir_name}")
        os.chdir(new_dir_name)

    if testing:
        # goes easy on the PC when debugging
        num_envs = 8
        eval_envs = 2
        eval_every = 11580000
        num_eval_episodes = 5
        n_steps = 11560000
        bs = 64
    else:
        num_envs = envs
        eval_envs = args.eval_envs
        n_steps = total_steps
        eval_every = 200000
    next_eval = eval_every

    # create blank evaluation file — size off the actual eval cadence so we never overflow.
    # +2 covers the end-of-training eval and rounding from num_envs overshooting next_eval.
    fname = f"{agent_name}_Evaluation.npy"
    if not testing:
        num_eval_slots = n_steps // eval_every + 2
        np.save(fname, np.zeros((num_eval_slots, num_eval_episodes)))

    logger.info("Currently Playing Game: " + str(game))

    gpu = "0"
    device = torch.device('cuda:' + gpu if torch.cuda.is_available() else 'cpu')
    logger.info("Device: " + str(device))

    env = make_env(num_envs, game, framestack, repeat_probs)
    logger.info(f"Observation Space: {env.observation_space}")
    logger.info(f"Action Space: {env.action_space}")
    if hasattr(env.action_space, "n"):
        n_actions = env.action_space.n
    else:
        # Take first if multiple discrete
        n_actions = env.action_space.nvec[0]

    agent = Rainbow(
        RainbowPolicy,
        env,
        total_timesteps=n_steps,
        target_replace=c,
        gamma=discount,
        per_alpha=per_alpha,
        n=nstep,
        grad_clip=grad_clip,
        replay_ratio=replay_ratio,
        compile_mode=compile_mode,
        learning_starts=20000,
        buffer_size=1048576,
        batch_size=bs,
        learning_rate=lr,
        device=device,
        policy_kwargs=dict(linear_size=linear_size),
    )

    callback = RainbowLoopCallback(
        agent_name=agent_name,
        game=game,
        testing=testing,
        include_evals=include_evals,
        eval_every=eval_every,
        eval_envs=eval_envs,
        num_eval_episodes=num_eval_episodes,
        framestack=framestack,
        repeat_probs=repeat_probs,
        n_actions=n_actions,
        device=device,
        linear_size=linear_size,
        total_steps=n_steps,
    )

    # Single learn() call: SB3 owns the env-stepping loop.
    agent.learn(total_timesteps=n_steps, callback=callback)

    env.close()

    logger.info("Evaluations finished, job completed successfully!")


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
