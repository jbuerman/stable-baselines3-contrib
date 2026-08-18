import logging
import platform
import time

import torch
import torch as T
import torch.nn.functional as F
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm

from sb3_contrib.rainbow.rainbow_buffer import PER
from sb3_contrib.rainbow.rainbow_policy import FactorizedNoisyLinear

logger = logging.getLogger(__name__)


class Rainbow(OffPolicyAlgorithm):
    def __init__(
        self,
        policy,
        env,
        total_timesteps=None,
        target_replace=2000,
        per_alpha=0.5,
        gamma=0.99,
        max_mem_size=None,
        n=3,
        grad_clip=10,
        replay_ratio=0.25,
        learning_rate=6.25e-5,
        buffer_size=1_000_000,
        learning_starts=20000,
        batch_size=32,
        rgb=False,
        framestack=4,
        imagex=84,
        imagey=84,
        init_setup_model=True,
        policy_kwargs=None,
        compile_mode="max-autotune",
        **kwargs,
    ):

        policy_kwargs = policy_kwargs or {}
        if not "linear_size" in policy_kwargs:
            policy_kwargs["linear_size"] = 512

        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            gamma=gamma,
            train_freq=(1, "step"),
            gradient_steps=1,
            policy_kwargs=policy_kwargs,
            support_multi_env=True,
            **kwargs,
        )

        self.compile_mode = compile_mode

        assert replay_ratio > 0, "replay_ratio must be positive"
        self.replay_ratio = replay_ratio

        grads_per_vec_step = replay_ratio * self.n_envs

        if grads_per_vec_step >= 1:
            self.train_freq = (1, "step")
            self.gradient_steps = round(grads_per_vec_step)
        else:
            self.train_freq = (round(1 / grads_per_vec_step), "step")
            self.gradient_steps = 1

        effective = self.gradient_steps / (self.train_freq[0] * self.n_envs)

        if abs(effective - replay_ratio) / replay_ratio > 0.01:
            import warnings

            warnings.warn(
                f"replay_ratio={replay_ratio} is not achievable exactly with "
                f"{self.n_envs} envs; using {effective:.4g} "
                f"(train_freq={self.train_freq[0]} steps, gradient_steps={self.gradient_steps})"
            )
        self.grad_steps = 0
        self.replace_target_cnt = target_replace
        self.Vmin = -10
        self.Vmax = 10
        self.N_ATOMS = 51
        self.n = n
        self.grad_clip = grad_clip

        self.rgb = rgb

        # buffer_size is the standard SB3 name; max_mem_size kept as an explicit override
        self.max_mem_size = max_mem_size if max_mem_size is not None else buffer_size
        self.per_alpha = per_alpha
        self.per_beta = 0.4
        self.framestack = framestack
        self.imagex = imagex
        self.imagey = imagey

        self.total_timesteps = total_timesteps

        if init_setup_model:
            self._setup_model()

    def _setup_model(self):
        self.per_buffer = PER(
            size=self.max_mem_size,
            device=self.device,
            rgb=self.rgb,
            n=self.n,
            envs=self.env.num_envs,
            gamma=self.gamma,
            alpha=self.per_alpha,
            beta=self.per_beta,
            framestack=self.framestack,
            imagex=self.imagex,
            imagey=self.imagey,
        )
        self.replay_buffer = self.per_buffer

        super()._setup_model()

        self.q_net = self.policy.q_net
        self.q_net_target = self.policy.q_net_target

        if self.compile_mode is not None and self.device.type == "cuda" and platform.system() == "Linux":
            self.q_net.forward = torch.compile(self.q_net.forward, mode=self.compile_mode)
            self.q_net_target.forward = torch.compile(self.q_net_target.forward, mode=self.compile_mode)
        elif self.compile_mode is not None:
            logger.info(f"torch.compile skipped (needs CUDA + Linux; got {self.device.type} + {platform.system()})")

    def _setup_learn(self, total_timesteps, *args, **kwargs):
        effective_total = self.total_timesteps if self.total_timesteps is not None else total_timesteps
        self.per_buffer.beta_increment = (1.0 - self.per_buffer.beta) * self.n_envs / effective_total
        return super()._setup_learn(total_timesteps, *args, **kwargs)

    def train(self, gradient_steps, batch_size):        
        for _ in range(gradient_steps):
            self._train_call()

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

    def replace_target_network(self):
        self.q_net_target.load_state_dict(self.q_net.state_dict())

    def _sample_buffer(self):
        return self.replay_buffer.sample(self.batch_size)

    def _train_call(self):
        if self.grad_steps % 10000 == 0:
            logger.debug(
                f"train_call start: "
                f"timesteps={self.num_timesteps} "
                f"grad_steps={self.grad_steps} "
                f"buffer_size={self.replay_buffer.size}"
            )

        if self.num_timesteps < self.learning_starts:
            logger.debug("Skipping training: learning_starts not reached")
            return

        # NoisyNet: resample noise on both networks per gradient step
        self.reset_noise(self.q_net)
        self.reset_noise(self.q_net_target)

        if self.grad_steps % self.replace_target_cnt == 0:
            self.replace_target_network()

        if self.grad_steps % 10000 == 0:
            t0 = time.perf_counter()
            logger.debug("Sampling replay buffer")
        batch = self._sample_buffer()
        if self.grad_steps % 10000 == 0:
            t1 = time.perf_counter()
            logger.debug(f"Replay buffer sample completed in {round(t1 - t0, 3)}")
        obs = batch.observations
        actions = batch.actions
        rewards = batch.rewards
        next_obs = batch.next_observations
        dones = batch.dones
        weights = batch.weights
        idxs = batch.idxs
        discounts = batch.discounts
        device = self.q_net.device

        obs = obs.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        next_obs = next_obs.to(device)
        dones = dones.to(device)
        weights = weights.to(device)
        discounts = discounts.to(device)

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

        self.policy.optimizer.zero_grad()
        distr_v, qvals_v = self.q_net.both(obs)
        state_action_values = distr_v[torch.arange(actions.shape[0]), actions]
        state_log_sm_v = F.log_softmax(state_action_values, dim=1)

        with torch.no_grad():
            # this is using Double DQN
            next_distr_v, next_qvals_v = self.q_net_target.both(next_obs)
            action_distr_v, action_qvals_v = self.q_net.both(next_obs)

            next_actions_v = action_qvals_v.max(1)[1]

            next_best_distr_v = next_distr_v[range(self.batch_size), next_actions_v.data]
            next_best_distr_v = self.q_net_target.apply_softmax(next_best_distr_v)
            next_best_distr = next_best_distr_v.detach()

            proj_distr = distr_projection(
                next_best_distr,
                rewards,
                dones,
                self.Vmin,
                self.Vmax,
                self.N_ATOMS,
                discounts,
            )

            proj_distr_v = proj_distr.to(self.q_net.device)

        state_log_sm_v = state_log_sm_v.to(self.q_net.device)
        kl_per_sample = (-state_log_sm_v * proj_distr_v).sum(dim=1)

        # update PER priorities with the raw (unweighted) KL
        if hasattr(self.replay_buffer, "update_priorities") and idxs is not None:
            self.replay_buffer.update_priorities(idxs, kl_per_sample.detach().cpu().numpy())

        weights = weights.squeeze().to(self.q_net.device)
        loss = (weights * kl_per_sample).mean()
        self.last_loss = loss.item()
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/beta", self.replay_buffer.beta)
        self.logger.record("train/grad_steps", self.grad_steps)

        loss.backward()

        # this wasn't explicitly mentioned in the Rainbow DQN paper, but was used in DQN and was likely kept
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.grad_clip)
        self.policy.optimizer.step()

        self.grad_steps += 1
        if self.grad_steps % 10000 == 0:
            logger.info(f"Completed {self.grad_steps} gradient steps")
        if self.grad_steps % 10000 == 0:
            logger.info(f"Beta: {self.replay_buffer.beta}")


def distr_projection(next_distr, rewards, dones, Vmin, Vmax, n_atoms, gamma):
    """
    Perform distribution projection aka Categorical Algorithm from the
    "A Distributional Perspective on RL" paper.

    gamma may be a scalar or a per-sample (batch,) tensor of bootstrap discounts
    (gamma^k for k-step windows cut short by truncation).

    Fully vectorized: a Python loop over atoms launches hundreds of small CUDA
    kernels per gradient step, which dominates training time on GPU.
    """
    device = next_distr.device
    batch_size = len(rewards)

    rewards = rewards.to(device).float()
    dones = dones.to(device).bool()

    if not torch.is_tensor(gamma):
        gamma = torch.full((batch_size,), float(gamma), device=device)
    else:
        gamma = gamma.to(device).float()

    delta_z = (Vmax - Vmin) / (n_atoms - 1)
    support = torch.linspace(Vmin, Vmax, n_atoms, device=device)

    # Tz = r + gamma^k * z; terminal transitions have no bootstrap term,
    # so their whole distribution collapses onto the clamped reward.
    tz = rewards.unsqueeze(1) + gamma.unsqueeze(1) * support.unsqueeze(0)
    tz[dones] = rewards[dones].unsqueeze(1)
    tz = tz.clamp(Vmin, Vmax)

    b = (tz - Vmin) / delta_z
    l = b.floor().long()
    u = b.ceil().long()

    # When b lands exactly on an atom, shift the pair so interpolation weights
    # still sum to 1 and no probability mass is dropped.
    l[(u > 0) & (l == u)] -= 1
    u[(l < (n_atoms - 1)) & (l == u)] += 1

    proj_distr = T.zeros((batch_size, n_atoms), dtype=T.float32, device=device)
    offset = (torch.arange(batch_size, device=device) * n_atoms).unsqueeze(1)

    proj_distr.view(-1).index_add_(
        0,
        (l + offset).view(-1),
        (next_distr * (u.float() - b)).view(-1),
    )
    proj_distr.view(-1).index_add_(
        0,
        (u + offset).view(-1),
        (next_distr * (b - l.float())).view(-1),
    )

    return proj_distr
