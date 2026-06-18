"""
TODO Remove before PR into SB3
"""

from copy import deepcopy
from functools import partial

import numpy as np
import torch
import torch as T
import torch.nn.functional as F
import torch.optim as optim

from sb3_contrib.rainbow.rainbow import distr_projection
from sb3_contrib.rainbow.rainbow_buffer import PER
from sb3_contrib.rainbow.rainbow_policy import FactorizedNoisyLinear
from sb3_contrib.rainbow.rainbow_policy import NatureC51


def create_network(input_dims, n_actions, device, linear_size):

    return NatureC51(input_dims[0], n_actions, device=device, linear_size=linear_size)


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

        self.network_creator_fn = partial(create_network, self.input_dims, self.n_actions, self.device, self.linear_size)

        self.net = self.network_creator_fn()
        self.tgt_net = self.network_creator_fn()

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
        self.disable_noise(self.eval_net)

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

            qvals = self.net.qvals(state, advantages_only=True)
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
        self.reset_noise(self.net)
        self.reset_noise(self.tgt_net)

        if self.grad_steps % self.replace_target_cnt == 0:
            self.replace_target_network()

        batch = self.memory.sample(self.batch_size)
        idxs = batch.idxs
        states = batch.observations
        actions = batch.actions
        rewards = batch.rewards
        next_states = batch.next_observations
        dones = batch.dones
        weights = batch.weights


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
        if self.grad_steps % 10000 == 0:
            print(f"Beta: {self.memory.beta}")
