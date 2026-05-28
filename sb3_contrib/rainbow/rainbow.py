import argparse
import math
import multiprocessing as mp
import os
import time
from copy import deepcopy
from functools import partial
from math import sqrt

import gymnasium as gym
import numpy as np
import torch
import torch as T
import torch.nn.functional as F
import torch.optim as optim
from sb3_contrib.rainbow.rainbow_policy import RainbowPolicy, FactorizedNoisyLinear
from torch import Tensor
from torch import nn as nn
from torch.nn import init
import ale_py
from sb3_contrib.rainbow.rainbow_buffer import PERBufferWrapper
from sb3_contrib.rainbow.rainbow_buffer import PER

from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm

############################################## Networks Section




################# Now Entering the Prioritized Experience Replay Section



def choose_eval_action(observation, eval_net, device):
    with torch.no_grad():
        state = T.tensor(observation, dtype=T.float).to(device)
        qvals = eval_net.qvals(state, advantages_only=True)
        x = T.argmax(qvals, dim=1).cpu()

    return x

# def create_network(input_dims, n_actions, device, linear_size):
#
#     return NatureC51(input_dims[0], n_actions, device=device, linear_size=linear_size)


#################### The big ol agent class, be prepared

class Rainbow(OffPolicyAlgorithm):
    """
    Transitional Rainbow implementation.
    Wraps existing Agent logic but exposes SB3 structure.
    """

    def __init__(
        self,
        policy,
        env,
        learning_rate=6.25e-5,
        buffer_size=1_000_000,
        batch_size=32,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=8000,
        verbose=0,
        device="auto",
        _init_setup_model=True,
        **kwargs,
    ):
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            batch_size=batch_size,
            tau=1.0,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=None,
            replay_buffer_class=None,  # we replace later
            replay_buffer_kwargs=None,
            policy_kwargs=None,
            verbose=verbose,
            device=device,
            support_multi_env=True,
        )

        # Store Rainbow-specific params
        self.target_update_interval = target_update_interval

        # Temporary: attach your old logic
        self._agent = None

        if _init_setup_model:
            self._setup_model()

    # =========================
    # REQUIRED SB3 METHODS
    # =========================

    def _setup_model(self):
        """
        Called by SB3 to initialise networks, optimizer, etc.
        """

        # For now, reuse your existing Agent

        obs_shape = self.observation_space.shape
        n_actions = self.action_space.n

        self._agent = Agent(
            n_actions=n_actions,
            input_dims=list(obs_shape),
            device=self.device,
            num_envs=1,
            agent_name="sb3_rainbow",
            total_steps=100000,
            batch_size=self.batch_size,
        )

        self.policy = RainbowPolicy(
            observation_space_shape=obs_shape,
            n_actions=n_actions,
            device=self.device,
        )
        self.q_net = self.policy.q_net
        self.target_policy = RainbowPolicy(
            observation_space_shape=obs_shape,
            n_actions=n_actions,
            device=self.device,
        )
        self.target_q_net = self.target_policy.q_net

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.learning_rate)

        self._raw_per = PER(
            self.buffer_size,
            self.device,
            n=3,
            envs=1,
            gamma=self.gamma,
        )

        self.replay_buffer = PERBufferWrapper(self._raw_per)

        self.grad_steps = 0
        self.env_steps = 0
        self.n = 3

        self.Vmin = -10
        self.Vmax = 10
        self.N_ATOMS = 51

        self.grad_clip = 10
        self.replace_target_cnt = self.target_update_interval
        self.min_sampling_size = 1000

        self.policy.to(self.device)

    def train(self, gradient_steps: int, batch_size: int):
        """SB3 training step
        """
        for _ in range(gradient_steps):
            self.train_call()

    def train_call(self):

        if self.env_steps < self.min_sampling_size:
            return

        # NoisyNet: resample noise on both networks per gradient step
        self.policy.reset_noise()
        self.target_policy.reset_noise()

        if self.grad_steps % self.replace_target_cnt == 0:
            self.replace_target_network()

        samples, idxs, weights = self.replay_buffer.sample(self.batch_size)

        states = samples.observations
        actions = samples.actions.squeeze(-1)
        rewards = samples.rewards.squeeze(-1)
        next_states = samples.next_observations
        dones = samples.dones.squeeze(-1)

        # use this code to check your states are correct if applying to a custom env
        # If you apply Rainbow to a custom env and don't check your states first, you are killing both
        # trees and your own time

        # plt.imshow(states[0][0].unsqueeze(dim=0).cpu().permute(1, 2, 0))
        # plt.show()
        #
        # plt.imshow(states[0][1].unsqueeze(dim=0).cpu().permute(1, 2, 0))
        # plt.show()
        #
        # plt.imshow(states[0][2].unsqueeze(dim=0).cpu().permute(1, 2, 0))
        # plt.show()
        #
        # plt.imshow(states[1][0].unsqueeze(dim=0).cpu().permute(1, 2, 0))
        # plt.show()
        #
        # plt.imshow(states[2][0].unsqueeze(dim=0).cpu().permute(1, 2, 0))
        # plt.show()

        self.optimizer.zero_grad()
        distr_v, qvals_v = self.net.both(states)
        state_action_values = distr_v[range(self.batch_size), actions.data]
        state_log_sm_v = F.log_softmax(state_action_values, dim=1)

        with torch.no_grad():
            # this is using Double DQN
            next_distr_v, next_qvals_v = self.tgt_net.both(next_states)
            action_distr_v, action_qvals_v = self.net.both(next_states)

            next_actions_v = action_qvals_v.max(1)[1]

            next_best_distr_v = next_distr_v[range(self.batch_size), next_actions_v.data]
            next_best_distr_v = self.tgt_net.apply_softmax(next_best_distr_v)
            next_best_distr = next_best_distr_v.data.cpu()

            proj_distr = distr_projection(next_best_distr, rewards.cpu(), dones.cpu(), self.Vmin, self.Vmax,
                                          self.N_ATOMS, self.gamma ** self.n)

            proj_distr_v = proj_distr.to(self.net.device)

        # per-sample KL (cross-entropy form); used as priority before IS-weighting
        kl_per_sample = (-state_log_sm_v * proj_distr_v).sum(dim=1)

        # update PER priorities with the raw (unweighted) KL
        self.replay_buffer.update_priorities(idxs, kl_per_sample.detach().cpu().numpy())

        weights = T.squeeze(weights).to(self.net.device)
        loss = (weights * kl_per_sample).mean()

        loss.backward()

        # this wasn't explicitly mentioned in the Rainbow DQN paper, but was used in DQN and was likely kept
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
        self.optimizer.step()

        self.grad_steps += 1
        if self.grad_steps % 10000 == 0:
            print("Completed " + str(self.grad_steps) + " gradient steps")

    def predict(self, observation, deterministic=False):
        """
        Used by SB3 for inference
        """

        with torch.no_grad():
            qvals = self.policy.get_q_values(observation)
            action = torch.argmax(qvals, dim=1).cpu().numpy()
            # If single environment → return scalar
            if action.shape[0] == 1:
                return action[0], None
            self.env_steps += 1

        return action,

    @torch.no_grad()
    def reset_noise(self):
        self.policy.reset_noise()
        for m in self.target_q_net.modules():
            if hasattr(m, "reset_noise"):
                m.reset_noise()

    @torch.no_grad()
    def replace_target_network(self):
        self.target_q_net.load_state_dict(self.q_net.state_dict())


class Agent:
    def __init__(self, n_actions, input_dims, device, num_envs, agent_name, total_steps, testing=False, batch_size=32,
                 rr=1, lr=6.25e-5, target_replace=2000, discount=0.99, linear_size=512,
                 framestack=4, rgb=False, imagex=84, imagey=84, per_alpha=0.5, max_mem_size=1048576,
                 n=3, grad_clip=10, spi=16):

        self.per_alpha = per_alpha

        self.Vmin = -10
        self.Vmax = 10
        self.N_ATOMS = 51

        self.spi = spi

        self.procgen = True if input_dims[1] == 64 else False
        self.grad_clip = grad_clip

        self.n_actions = n_actions
        self.input_dims = input_dims
        self.device = device
        self.agent_name = agent_name
        self.testing = testing

        self.loading_checkpoint = False

        self.per_beta = 0.4

        self.total_steps = total_steps
        self.num_envs = num_envs

        if self.testing:
            self.min_sampling_size = 1000
        else:
            # paper: 80K frames; frameskip=4 → 20K env-steps
            self.min_sampling_size = 20000

        self.lr = lr

        self.priority_weight_increase = (1 - self.per_beta) / self.total_steps

        self.action_space = [i for i in range(self.n_actions)]
        self.learn_step_counter = 0

        self.chkpt_dir = ""

        self.n = n
        self.gamma = discount
        self.batch_size = batch_size

        # 1 Million rounded to the nearest power of 2 for tree implementation
        self.max_mem_size = max_mem_size

        self.replace_target_cnt = target_replace  # This is the number of grad steps - could be a little jank
        # when changing num_envs/batch size/replay ratio

        # Best used value is 32000 frames per replace. For bs 256, this is 500. For bs 16, this is every 8000!

        self.linear_size = linear_size

        self.framestack = framestack
        self.rgb = rgb
        self.memory = PER(self.max_mem_size, device, self.n, num_envs, self.gamma, alpha=self.per_alpha,
                          beta=self.per_beta, framestack=self.framestack, rgb=self.rgb, imagex=imagex, imagey=imagey)

        

        self.policy = RainbowPolicy(
            observation_space_shape=self.input_dims,
            n_actions=self.n_actions,
            device=self.device,
            linear_size=self.linear_size,
        )

        self.target_policy = RainbowPolicy(
            observation_space_shape=self.input_dims,
            n_actions=self.n_actions,
            device=self.device,
            linear_size=self.linear_size,
        )

        self.net = self.policy.q_net
        self.tgt_net = self.target_policy.q_net

        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr, eps=1.5e-4)

        self.net.train()

        self.eval_net = None

        # disable gradients for the target network
        for param in self.tgt_net.parameters():
            param.requires_grad = False

        self.env_steps = 0
        self.grad_steps = 0

        self.eval_mode = False

    def prep_evaluation(self):
        self.eval_net = deepcopy(self.net)
        self.policy.disable_noise()

    @torch.no_grad()
    def reset_noise(self, net):
        for m in net.modules():
            if isinstance(m, FactorizedNoisyLinear):
                m.reset_noise()

    @torch.no_grad()
    def disable_noise(self, net):
        for m in net.modules():
            if isinstance(m, FactorizedNoisyLinear):
                m.disable_noise()

    def choose_action(self, observation):
        # this chooses an action for a batch. Can be used with a batch of 1 if needed though
        with T.no_grad():
            if not self.eval_mode:
                self.reset_noise(self.net)

            state = T.tensor(observation, dtype=T.float).to(self.net.device)

            qvals = self.policy.get_q_values(state)
            x = T.argmax(qvals, dim=1).cpu()

            return x

    def store_transition(self, state, action, reward, next_state, done, trun, stream, prio=True):

        if self.rgb:
            # expand dims to create "framestack" dim, so it works with my replay buffer
            state = np.expand_dims(state, axis=0)
            next_state = np.expand_dims(next_state, axis=0)

        self.memory.append(state, action, reward, next_state, done, trun, stream, prio=prio)

        self.env_steps += 1
        # anneal PER's beta
        self.memory.beta = min(self.memory.beta + self.priority_weight_increase, 1.0)

    def replace_target_network(self):
        self.tgt_net.load_state_dict(self.net.state_dict())

    def save_model(self):
        self.net.save_checkpoint(self.agent_name + "_" + str(int((self.env_steps // 250000))) + "M")

    def load_models(self, name):
        self.net.load_checkpoint(name)
        self.tgt_net.load_checkpoint(name)

    def learn(self):
        for i in range(self.spi):
            self.learn_call()

    def learn_call(self):

        if self.env_steps < self.min_sampling_size:
            return

        # NoisyNet: resample noise on both networks per gradient step
        self.policy.reset_noise()
        self.target_policy.reset_noise()

        if self.grad_steps % self.replace_target_cnt == 0:
            self.replace_target_network()

        idxs, states, actions, rewards, next_states, dones, weights = self.memory.sample(self.batch_size)

        # use this code to check your states are correct if applying to a custom env
        # If you apply Rainbow to a custom env and don't check your states first, you are killing both
        # trees and your own time

        # plt.imshow(states[0][0].unsqueeze(dim=0).cpu().permute(1, 2, 0))
        # plt.show()
        #
        # plt.imshow(states[0][1].unsqueeze(dim=0).cpu().permute(1, 2, 0))
        # plt.show()
        #
        # plt.imshow(states[0][2].unsqueeze(dim=0).cpu().permute(1, 2, 0))
        # plt.show()
        #
        # plt.imshow(states[1][0].unsqueeze(dim=0).cpu().permute(1, 2, 0))
        # plt.show()
        #
        # plt.imshow(states[2][0].unsqueeze(dim=0).cpu().permute(1, 2, 0))
        # plt.show()

        self.optimizer.zero_grad()
        distr_v, qvals_v = self.net.both(states)
        state_action_values = distr_v[range(self.batch_size), actions.data]
        state_log_sm_v = F.log_softmax(state_action_values, dim=1)

        with torch.no_grad():
            # this is using Double DQN
            next_distr_v, next_qvals_v = self.tgt_net.both(next_states)
            action_distr_v, action_qvals_v = self.net.both(next_states)

            next_actions_v = action_qvals_v.max(1)[1]

            next_best_distr_v = next_distr_v[range(self.batch_size), next_actions_v.data]
            next_best_distr_v = self.tgt_net.apply_softmax(next_best_distr_v)
            next_best_distr = next_best_distr_v.data.cpu()

            proj_distr = distr_projection(next_best_distr, rewards.cpu(), dones.cpu(), self.Vmin, self.Vmax,
                                          self.N_ATOMS, self.gamma ** self.n)

            proj_distr_v = proj_distr.to(self.net.device)

        # per-sample KL (cross-entropy form); used as priority before IS-weighting
        kl_per_sample = (-state_log_sm_v * proj_distr_v).sum(dim=1)

        # update PER priorities with the raw (unweighted) KL
        self.memory.update_priorities(idxs, kl_per_sample.detach().cpu().numpy())

        weights = T.squeeze(weights).to(self.net.device)
        loss = (weights * kl_per_sample).mean()

        loss.backward()

        # this wasn't explicitly mentioned in the Rainbow DQN paper, but was used in DQN and was likely kept
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
        self.optimizer.step()

        self.grad_steps += 1
        if self.grad_steps % 10000 == 0:
            print("Completed " + str(self.grad_steps) + " gradient steps")

def distr_projection(next_distr, rewards, dones, Vmin, Vmax, n_atoms, gamma):
    """
    Perform distribution projection aka Catergorical Algorithm from the
    "A Distributional Perspective on RL" paper
    """
    batch_size = len(rewards)
    proj_distr = T.zeros((batch_size, n_atoms), dtype=T.float32)
    delta_z = (Vmax - Vmin) / (n_atoms - 1)
    for atom in range(n_atoms):
        tz_j = T.clamp(rewards + (Vmin + atom * delta_z) * gamma, Vmin, Vmax)
        b_j = (tz_j - Vmin) / delta_z
        l = T.floor(b_j).type(T.int64)
        u = T.ceil(b_j).type(T.int64)
        eq_mask = u == l
        proj_distr[eq_mask, l[eq_mask]] += next_distr[eq_mask, atom]
        ne_mask = u != l
        proj_distr[ne_mask, l[ne_mask]] += next_distr[ne_mask, atom] * (u - b_j)[ne_mask]
        proj_distr[ne_mask, u[ne_mask]] += next_distr[ne_mask, atom] * (b_j - l)[ne_mask]
    if dones.any():
        proj_distr[dones] = 0.0
        tz_j = T.clamp(rewards[dones], Vmin, Vmax)
        b_j = (tz_j - Vmin) / delta_z
        l = T.floor(b_j).type(T.int64)
        u = T.ceil(b_j).type(T.int64)
        eq_mask = u == l
        eq_dones = T.clone(dones)
        eq_dones[dones] = eq_mask
        if eq_dones.any():
            proj_distr[eq_dones, l[eq_mask]] = 1.0
        ne_mask = u != l
        ne_dones = T.clone(dones)
        ne_dones[dones] = ne_mask
        if ne_dones.any():
            proj_distr[ne_dones, l[ne_mask]] = (u - b_j)[ne_mask]
            proj_distr[ne_dones, u[ne_mask]] = (b_j - l)[ne_mask]
    return proj_distr

def make_env(envs_create, game, framestack, repeat_probs, terminal_on_life_loss=True):
    return gym.vector.SyncVectorEnv([lambda: gym.wrappers.FrameStackObservation(
        gym.wrappers.AtariPreprocessing(
            gym.make(
                "ALE/" + game + "-v5",
                frameskip=1,
                repeat_action_probability=repeat_probs),
            terminal_on_life_loss=terminal_on_life_loss),
        framestack) for _ in range(envs_create)])


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
    print(env.observation_space)
    print(env.action_space[0])
    n_actions = env.action_space[0].n

    agent = Agent(n_actions=env.action_space[0].n, input_dims=[framestack, 84, 84], device=device, num_envs=num_envs,
                  agent_name=agent_name, total_steps=n_steps, testing=testing, batch_size=bs, lr=lr,
                  target_replace=c, discount=discount, linear_size=linear_size,
                  framestack=framestack, per_alpha=per_alpha, n=nstep, grad_clip=grad_clip, spi=spi)

    scores_temp = []
    steps = 0
    last_steps = 0
    last_time = time.time()
    episodes = 0
    current_eval = 0
    scores_count = [0 for i in range(num_envs)]
    scores = []
    observation, info = env.reset()
    processes = []

    while steps < n_steps:
        steps += num_envs
        action = agent.choose_action(observation)

        # sync vector env: step then learn (no overlap possible)
        observation_, reward, done_, trun_, info = env.step(action)
        agent.learn()

        # this just tracks the score for each environment
        for i in range(num_envs):
            scores_count[i] += reward[i]
            if done_[i] or trun_[i]:
                episodes += 1
                scores.append([scores_count[i], steps])
                scores_temp.append(scores_count[i])
                scores_count[i] = 0

        # reward clipping
        reward = np.clip(reward, -1., 1.)

        # add transitions to the replay buffer
        for stream in range(num_envs):
            terminal_in_buffer = done_[stream]

            next_obs = observation_[stream] if not trun_[stream] else np.array(info["final_observation"][stream])
            agent.store_transition(observation[stream], action[stream], reward[stream], next_obs,
                                   terminal_in_buffer, trun_[stream], stream=stream)

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
