import argparse
import multiprocessing as mp
import os
import time
from copy import deepcopy

import gymnasium as gym
import numpy as np
import torch
import ale_py

from sb3_contrib.rainbow.rainbow import Rainbow
from sb3_contrib.rainbow.rainbow_policy import RainbowPolicy, FactorizedNoisyLinear
from stable_baselines3.common.vec_env import SubprocVecEnv


def choose_eval_action(observation, eval_net, device):
    with torch.no_grad():
        state = torch.tensor(observation, dtype=torch.float32).to(device)

        # IMPORTANT: reset noisy layers for stochasticity if needed
        for m in eval_net.modules():
            if isinstance(m, FactorizedNoisyLinear):
                m.reset_noise()

        qvals = eval_net.qvals(state, advantages_only=True)
        action = torch.argmax(qvals, dim=1).cpu()

    return action


def make_env(envs_create, game, framestack, repeat_probs, terminal_on_life_loss=True):
    def make_single_env():
        return gym.wrappers.FrameStackObservation(
            gym.wrappers.AtariPreprocessing(
                gym.make("ALE/" + game + "-v5", frameskip=1, repeat_action_probability=repeat_probs),
                terminal_on_life_loss=terminal_on_life_loss,
            ),
            framestack,
        )
    return SubprocVecEnv([make_single_env for _ in range(envs_create)])


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
    eval_env = make_env(eval_envs, game, framestack, repeat_probs, terminal_on_life_loss=False)
    evals = []
    eval_episodes = 0
    eval_scores = np.array([0 for i in range(eval_envs)])
    eval_observation, eval_info = eval_env.reset()

    eval_net = network_creator()

    # move state dict to gpu - pytorch doesn't allow sharing across threads on gpu
    state_dict_gpu = {k: v.to(device) for k, v in net_state_dict.items()}

    eval_net.load_state_dict(state_dict_gpu)

    while eval_episodes < num_eval_episodes:

        eval_action = choose_eval_action(eval_observation, eval_net, device)
        eval_observation_, eval_reward, eval_done_, eval_trun_, eval_info = eval_env.step(eval_action)
        eval_done_ = np.logical_or(eval_done_, eval_trun_)

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
        fname = agent_name + "Evaluation.npy"
        data = np.load(fname)

        # Update the specified index in the 0th dimension
        data[index] = evals
        print("Evaluation " + str(index + 1) + "M Complete, average score:")
        print(np.mean(evals))

        # Save the updated array back to the file
        np.save(fname, data)


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
    parser.add_argument('--spi', type=int, default=16)  # Samples per insert ratio (SPI). This is the same as Rainbow DQN.

    args = parser.parse_args()

    arg_string = non_default_args(args, parser)
    formatted_string = format_arguments(arg_string)
    print(formatted_string)

    game = args.game
    envs = args.envs
    bs = args.bs
    # convert target-replace period from environment frames to gradient steps.
    # frames / 4 → env-steps; / envs → main-loop iterations; * spi → gradient steps.
    c = int((args.target_replace_frames / 4) * args.spi / envs)
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
    spi = args.spi

    lr_str = "{:e}".format(lr)
    lr_str = str(lr_str).replace(".", "").replace("0", "")
    frame_name = str(int(args.frames / 1000000)) + "M"

    include_evals = bool(args.include_evals)
    agent_name = "Rainbow_" + game + frame_name

    if len(formatted_string) > 2:
        agent_name += '_' + formatted_string

    print("Agent Name:" + str(agent_name))
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
        print(f"Created directory: {new_dir_name}")
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
    fname = agent_name + "Evaluation.npy"
    if not testing:
        num_eval_slots = n_steps // eval_every + 2
        np.save(fname, np.zeros((num_eval_slots, num_eval_episodes)))

    print("Currently Playing Game: " + str(game))

    gpu = "0"
    device = torch.device('cuda:' + gpu if torch.cuda.is_available() else 'cpu')
    print("Device: " + str(device))

    env = make_env(num_envs, game, framestack, repeat_probs)
    print(f"Observation Space: {env.observation_space}")
    print(f"Action Space: {env.action_space}")
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
        spi=spi,
        learning_starts=20000,
        buffer_size=1048576,
        batch_size=bs,
        learning_rate=lr,
        device=device,
        policy_kwargs=dict(
            linear_size=linear_size
        ),
    )

    scores_temp = []
    steps = 0
    last_steps = 0
    last_time = time.time()
    episodes = 0
    current_eval = 0
    scores_count = [0 for _ in range(num_envs)]
    scores = []
    observation = env.reset()
    processes = []

    while steps < n_steps:
        steps += num_envs
        action, _ = agent.predict(observation, deterministic=False)

        # sync vector env: step then learn (no overlap possible)
        observation_, reward, done_, info = env.step(action)
        agent.learn(total_timesteps=num_envs, reset_num_timesteps=False)

        # this just tracks the score for each environment
        for i in range(num_envs):
            trun_i = info[i].get("TimeLimit.truncated", False)
            scores_count[i] += reward[i]
            if done_[i] or trun_i:
                episodes += 1
                scores.append([scores_count[i], steps])
                scores_temp.append(scores_count[i])
                scores_count[i] = 0

        # reward clipping
        reward = np.clip(reward, -1., 1.)

        observation = observation_

        # print progress
        if steps % 1200 == 0 and len(scores) > 0:
            avg_score = np.mean(scores_temp[-50:])
            if episodes % 1 == 0:
                print('{} {} avg score {:.2f} total_steps {:.0f} fps {:.2f} games {}'
                      .format(agent_name, game, avg_score, steps, (steps - last_steps) / (time.time() - last_time), episodes),
                      flush=True)
                last_steps = steps
                last_time = time.time()

        # Evaluation
        if steps >= next_eval or steps >= n_steps:
            print("Evaluating")

            # Save model
            if not testing and (current_eval + 1) == 1 or (current_eval + 1) == 10 or (current_eval + 1) == 50\
                    or (current_eval + 1) == 100 or (current_eval + 1) == 150 or (current_eval + 1) == 200:
                agent.save_model()

            fname = agent_name + "Experiment.npy"
            if not testing:
                np.save(fname, np.array(scores))

            if include_evals:

                # wait for our evaluations to finish before we start the next evaluation
                for process in processes:
                    process.join()

                agent.disable_noise(agent.net)
                net_state_dict = deepcopy({k: v.cpu() for k, v in agent.net.state_dict().items()})
                network_creator = deepcopy(agent.network_creator_fn)

                # Start evaluation in a separate process
                eval_process = mp.Process(target=evaluate_agent,
                                          args=(net_state_dict, network_creator, eval_envs, num_eval_episodes, agent_name, testing, game,
                                                n_actions, device, current_eval, framestack, repeat_probs))
                eval_process.start()
                processes.append(eval_process)

            current_eval += 1

            next_eval += eval_every

    # wait for our evaluations to finish before we quit the program
    for process in processes:
        process.join()

    print("Evaluations finished, job completed successfully!")


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
