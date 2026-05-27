import numpy as np
import torch
import gymnasium as gym

from sb3_contrib.rainbow.rainbow import Agent, Rainbow
from sb3_contrib.rainbow.rainbow_policy import RainbowPolicy


def make_dummy_obs(n_envs, framestack=4, H=84, W=84):
    return np.random.randint(
        0, 256, size=(n_envs, framestack, H, W), dtype=np.uint8
    )

class DummyImageEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(4, 84, 84), dtype=np.uint8
        )
        self.action_space = gym.spaces.Discrete(6)

    def reset(self, seed=None, options=None):
        return self._obs(), {}

    def step(self, action):
        return self._obs(), 0.0, False, False, {}

    def _obs(self):
        return np.random.randint(0, 256, (4, 84, 84), dtype=np.uint8)


def test_sb3_wrapper_runs():
    env = DummyImageEnv()

    model = Rainbow(
        policy=RainbowPolicy,
        env=env,
        batch_size=8,
        gradient_steps=1,
    )

    obs, _ = env.reset()

    for _ in range(10):
        action, _ = model.predict(obs)
        obs, _, _, _, _ = env.step(int(action))

        model.train(gradient_steps=1, batch_size=8)

    assert True


def test_rainbow_smoke():
    device = torch.device("cpu")

    n_envs = 4
    n_actions = 6
    framestack = 4

    agent = Agent(
        n_actions=n_actions,
        input_dims=[framestack, 84, 84],
        device=device,
        num_envs=n_envs,
        agent_name="test",
        total_steps=10_000,
        testing=True,
        batch_size=8,
        max_mem_size=1000,
        n=3,
    )

    obs = make_dummy_obs(n_envs)

    # Fill buffer with random data
    for step in range(200):
        actions = np.random.randint(0, n_actions, size=n_envs)
        next_obs = make_dummy_obs(n_envs)

        rewards = np.random.randn(n_envs)
        dones = np.random.rand(n_envs) < 0.1
        truncs = np.zeros_like(dones)

        for i in range(n_envs):
            agent.store_transition(
                obs[i],
                actions[i],
                rewards[i],
                next_obs[i],
                dones[i],
                truncs[i],
                stream=i,
            )

        obs = next_obs

    # Should not crash
    agent.learn()

    # Basic sanity checks
    assert agent.grad_steps >= 0


def test_forward_consistency():
    device = torch.device("cpu")

    n_envs = 2
    agent = Agent(
        n_actions=4,
        input_dims=[4, 84, 84],
        device=device,
        num_envs=n_envs,
        agent_name="test",
        total_steps=1000,
        testing=True,
    )

    obs = np.ones((n_envs, 4, 84, 84), dtype=np.uint8) * 128

    with torch.no_grad():
        q1 = agent.net.qvals(torch.tensor(obs, dtype=torch.float32))
        q2 = agent.net.qvals(torch.tensor(obs, dtype=torch.float32))

    # NoisyNet means they may differ unless noise disabled
    agent.disable_noise(agent.net)

    with torch.no_grad():
        q1 = agent.net.qvals(torch.tensor(obs, dtype=torch.float32))
        q2 = agent.net.qvals(torch.tensor(obs, dtype=torch.float32))

    assert torch.allclose(q1, q2), "Deterministic forward pass broken"

# def test_sb3_wrapper_runs():
#     env = gym.make("CartPole-v1")
#
#     model = Rainbow(
#         policy=RainbowPolicy,
#         env=env,
#         batch_size=8,
#         gradient_steps=1,
#     )
#
#     obs, _ = env.reset()
#
#     for _ in range(10):
#         action, _ = model.predict(obs)
#         obs, reward, done, trunc, _ = env.step(int(action))
#
#         if done or trunc:
#             obs, _ = env.reset()
#
#     # No crash = pass
#     assert True
