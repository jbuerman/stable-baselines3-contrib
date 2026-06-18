import torch
import torch as T
import torch.nn.functional as F
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.type_aliases import TrainFreq, TrainFrequencyUnit

from sb3_contrib.rainbow.rainbow_buffer import PER
from sb3_contrib.rainbow.rainbow_policy import FactorizedNoisyLinear

class Rainbow(OffPolicyAlgorithm):
    def __init__(
        self,
        policy,
        env,
        total_timesteps=None,
        target_replace=2000,
        per_alpha=0.5,
        gamma=0.99,
        max_mem_size=1048576,
        n=3,
        grad_clip=10,
        spi=16,
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
        **kwargs
    ):
        train_freq = (spi, "step")
        gradient_steps = spi

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
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            policy_kwargs=policy_kwargs,
            support_multi_env = True,
            **kwargs,
        )

        self.spi = spi
        self.grad_steps = 0
        self.replace_target_cnt = target_replace
        self.Vmin = -10
        self.Vmax = 10
        self.N_ATOMS = 51
        self.n = n
        self.grad_clip = grad_clip

        self.rgb = rgb

        self.max_mem_size = max_mem_size
        self.per_alpha = per_alpha
        self.per_beta = 0.4
        self.framestack = framestack
        self.imagex = imagex
        self.imagey = imagey

        self.total_timesteps = total_timesteps
        self._beta_initialized = False

        if init_setup_model:
            self._setup_model()

    def _setup_model(self):
        super()._setup_model()

        self.q_net = self.policy.q_net
        self.q_net_target = self.policy.q_net_target

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

    def _setup_learn(self, total_timesteps, *args, **kwargs):
        self.priority_weight_increase = (
            1 - self.per_beta
        ) / total_timesteps
        self.per_buffer.beta_increment = self.priority_weight_increase
        return super()._setup_learn(total_timesteps, *args, **kwargs)

    def train(self, gradient_steps, batch_size):        
        for _ in range(gradient_steps):
            self._train_call()

    def learn(self, total_timesteps, *args, **kwargs):
        if not self._beta_initialized:
            effective_total = (
                self.total_timesteps
                if self.total_timesteps is not None
                else total_timesteps
            )
            beta_start = self.replay_buffer.beta
            self.replay_buffer.per_beta_increment = (
                    (1.0 - beta_start) / (effective_total * self.n_envs)
            )
            self._beta_initialized = True

        return super().learn(total_timesteps, *args, **kwargs)

    @torch.no_grad()
    def reset_noise(self, net):
        for m in net.modules():
            if isinstance(m, FactorizedNoisyLinear):
                m.reset_noise()

    def replace_target_network(self):
        self.q_net_target.load_state_dict(self.q_net.state_dict())

    def _sample_buffer(self):
        return self.replay_buffer.sample(self.batch_size)

    def _train_call(self):
        if self.num_timesteps < self.learning_starts:
            return

        # NoisyNet: resample noise on both networks per gradient step
        self.reset_noise(self.q_net)
        self.reset_noise(self.q_net_target)

        if self.grad_steps % self.replace_target_cnt == 0:
            self.replace_target_network()

        batch = self._sample_buffer()
        obs = batch.observations
        actions = batch.actions
        rewards = batch.rewards
        next_obs = batch.next_observations
        dones = batch.dones
        weights = batch.weights
        idxs = batch.idxs
        device = self.q_net.device

        obs = obs.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        next_obs = next_obs.to(device)
        dones = dones.to(device)
        weights = weights.to(device)

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
                next_best_distr, rewards, dones, self.Vmin, self.Vmax, self.N_ATOMS, self.gamma**self.n
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

        loss.backward()

        # this wasn't explicitly mentioned in the Rainbow DQN paper, but was used in DQN and was likely kept
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.grad_clip)
        self.policy.optimizer.step()

        self.grad_steps += 1
        if self.grad_steps % 10000 == 0:
            print("Completed " + str(self.grad_steps) + " gradient steps")
        if self.grad_steps % 10000 == 0:
            print(f"Beta: {self.replay_buffer.beta}")


def distr_projection(next_distr, rewards, dones, Vmin, Vmax, n_atoms, gamma):
    """
    Perform distribution projection aka Catergorical Algorithm from the
    "A Distributional Perspective on RL" paper
    """
    batch_size = len(rewards)
    device = next_distr.device
    rewards = rewards.to(device)
    dones = dones.to(device)
    proj_distr = T.zeros((batch_size, n_atoms), dtype=T.float32, device=device)
    delta_z = (Vmax - Vmin) / (n_atoms - 1)
    for atom in range(n_atoms):
        tz_j = T.clamp(rewards + (Vmin + atom * delta_z) * gamma, Vmin, Vmax).to(device)
        b_j = ((tz_j - Vmin) / delta_z).to(device)
        l = T.floor(b_j).long()
        u = T.ceil(b_j).long()
        eq_mask = u == l
        proj_distr[eq_mask, l[eq_mask]] += next_distr[eq_mask, atom]
        ne_mask = u != l
        proj_distr[ne_mask, l[ne_mask]] += next_distr[ne_mask, atom] * (u - b_j)[ne_mask]
        proj_distr[ne_mask, u[ne_mask]] += next_distr[ne_mask, atom] * (b_j - l)[ne_mask]
    if dones.any():
        dones = dones.bool()
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
