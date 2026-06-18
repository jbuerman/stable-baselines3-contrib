import math
from math import sqrt

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn, optim
from torch.nn import init as torch_init


from stable_baselines3.common.policies import BasePolicy


class RainbowPolicy(BasePolicy):
    def __init__(self, observation_space, action_space, lr_schedule, linear_size=512, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule)

        obs_shape = observation_space.shape
        n_actions = action_space.n

        self.linear_size = linear_size

        self.q_net = NatureC51(
            in_depth=obs_shape[0],
            actions=n_actions,
            device=self.device,
            linear_size=linear_size,
        )

        self.q_net_target = NatureC51(
            in_depth=obs_shape[0],
            actions=n_actions,
            device=self.device,
            linear_size=linear_size,
        )

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr_schedule(1), eps=1.5e-4)

    def forward(self, obs):
        return self.q_net.qvals(obs)

    def _predict(self, obs, deterministic=True):
        self.reset_noise()
        qvals = self.forward(obs)
        return qvals.argmax(dim=1)

    def reset_noise(self):
        for module in self.q_net.modules():
            if isinstance(module, FactorizedNoisyLinear):
                module.reset_noise()


class FactorizedNoisyLinear(nn.Module):
    """ The factorized Gaussian noise layer for noisy-nets dqn. """
    def __init__(self, in_features: int, out_features: int, sigma_0=0.5, self_norm=False) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_0 = sigma_0

        # weight: w = \mu^w + \sigma^w . \epsilon^w
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))

        # bias: b = \mu^b + \sigma^b . \epsilon^b
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        if self_norm:
            self.reset_parameters_self_norm()
        else:
            self.reset_parameters()
        self.reset_noise()

        self.disable_noise()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        # initialization is similar to Kaiming uniform (He. initialization) with fan_mode=fan_in
        scale = 1 / sqrt(self.in_features)

        torch_init.uniform_(self.weight_mu, -scale, scale)
        torch_init.uniform_(self.bias_mu, -scale, scale)

        torch_init.constant_(self.weight_sigma, self.sigma_0 * scale)
        torch_init.constant_(self.bias_sigma, self.sigma_0 * scale)

    @torch.no_grad()
    def reset_parameters_self_norm(self) -> None:
        # initialization is similar to Kaiming uniform (He. initialization) with fan_mode=fan_in

        nn.init.normal_(self.weight_mu, std=1 / math.sqrt(self.out_features))
        if self.bias_mu is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_mu)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_mu, -bound, bound)

    @torch.no_grad()
    def _get_noise(self, size: int) -> Tensor:
        noise = torch.randn(size, device=self.weight_mu.device)
        # f(x) = sgn(x)sqrt(|x|)
        return noise.sign().mul_(noise.abs().sqrt_())

    @torch.no_grad()
    def reset_noise(self) -> None:
        # like in eq 10 and 11 of the paper
        epsilon_in = self._get_noise(self.in_features)
        epsilon_out = self._get_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    @torch.no_grad()
    def disable_noise(self) -> None:
        self.weight_epsilon[:] = 0
        self.bias_epsilon[:] = 0

    def forward(self, input: Tensor) -> Tensor:
        # y = wx + d, where
        # w = \mu^w + \sigma^w * \epsilon^w
        # b = \mu^b + \sigma^b * \epsilon^b
        return F.linear(input,
                        self.weight_mu + self.weight_sigma*self.weight_epsilon,
                        self.bias_mu + self.bias_sigma*self.bias_epsilon)

class NatureC51(nn.Module):
    """
    Implementation of the Nature CNN, with the Categorical heads used for C51.
    """
    def __init__(self, in_depth, actions, atoms=51, Vmin=-10, Vmax=10, device='cuda:0', linear_size=512):
        super().__init__()

        self.actions = actions
        self.atoms = atoms
        self.device = device
        self.linear_size = linear_size

        DELTA_Z = (Vmax - Vmin) / (atoms - 1)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_depth, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        conv_out_size = 3136

        # Noisy Linear Layers, with both value and advantage functions for dueling DQN
        self.fc1V = FactorizedNoisyLinear(conv_out_size, self.linear_size)
        self.fc1A = FactorizedNoisyLinear(conv_out_size, self.linear_size)
        self.fcV2 = FactorizedNoisyLinear(self.linear_size, self.atoms)
        self.fcA2 = FactorizedNoisyLinear(self.linear_size, actions * self.atoms)

        self.register_buffer("supports", torch.arange(Vmin, Vmax+DELTA_Z, DELTA_Z))
        self.softmax = nn.Softmax(dim=1)

        self.to(device)

    def reset_noise(self):
        for name, module in self.named_children():
            if 'fc' in name:
                module.reset_noise()

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def fc_val(self, x):
        x = F.relu(self.fc1V(x))
        x = self.fcV2(x)

        return x

    def fc_adv(self, x):
        x = F.relu(self.fc1A(x))
        x = self.fcA2(x)

        return x

    def forward(self, x):
        batch_size = x.size()[0]
        device = next(self.parameters()).device
        fx = x.to(device).float() / 255
        conv_out = self.conv(fx)

        conv_out = conv_out.view(batch_size, -1)

        val_out = self.fc_val(conv_out).view(batch_size, 1, self.atoms)
        adv_out = self.fc_adv(conv_out).view(batch_size, -1, self.atoms)
        adv_mean = adv_out.mean(dim=1, keepdim=True)
        return val_out + (adv_out - adv_mean)

    def both(self, x):
        cat_out = self(x)
        probs = self.apply_softmax(cat_out)
        weights = probs * self.supports
        res = weights.sum(dim=2)
        return cat_out, res

    def qvals(self, x, advantages_only=False):
        return self.both(x)[1]

    def apply_softmax(self, t):
        return self.softmax(t.view(-1, self.atoms)).view(t.size())

    def save_checkpoint(self, name):
        #print('... saving checkpoint ...')
        torch.save(self.state_dict(), name + ".model")

    def load_checkpoint(self, name):
        #print('... loading checkpoint ...')
        self.load_state_dict(torch.load(name))