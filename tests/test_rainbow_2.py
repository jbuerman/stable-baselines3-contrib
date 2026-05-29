import torch
import numpy as np

from gymnasium import spaces

from sb3_contrib.rainbow2.rainbow import Agent, make_env, NatureC51, PER
from sb3_contrib.rainbow2.rainbow_policy import RainbowPolicy

def test_smoke_test():
    device = torch.device("cpu")

    # minimal env
    env = make_env(2, "Pong", framestack=4, repeat_probs=0.0)
    obs, _ = env.reset()

    n_actions = env.action_space[0].n

    agent = Agent(
        n_actions=n_actions,
        input_dims=[4, 84, 84],
        device=device,
        num_envs=2,
        agent_name="test",
        total_steps=1000,
        testing=True,
        batch_size=8
    )

    # one interaction step
    action = agent.choose_action(obs)
    obs_, reward, done_, trun_, _ = env.step(action)

    # push to replay buffer
    for i in range(2):
        agent.store_transition(
            obs[i], action[i], reward[i], obs_[i],
            done_[i], trun_[i], stream=i
        )

    # force learning call
    for _ in range(5):
        agent.learn()

    print("Smoke test passed")

def test_network():
    net = NatureC51(4, 6, device="cpu")
    x = torch.randn(2, 4, 84, 84)
    out = net.qvals(x)

    assert out.shape == (2, 6)


def test_replay():
    buffer = PER(size=100, device="cpu", n=3, envs=1, gamma=0.99)

    dummy_state = np.zeros((4, 84, 84), dtype=np.uint8)

    for _ in range(50):
        buffer.append(dummy_state, 0, 1.0, dummy_state, False, False, stream=0)

    batch = buffer.sample(8)
    print("Replay test passed")

def test_agent_action():
    agent = Agent(
        n_actions=4,
        input_dims=[4, 84, 84],
        device="cpu",
        num_envs=1,
        agent_name="test",
        total_steps=1000,
        testing=True
    )

    obs = np.random.randint(0, 255, (1, 4, 84, 84), dtype=np.uint8)
    action = agent.choose_action(obs)

    assert action.shape == (1,)

def test_policy():
    obs_space = spaces.Box(low=0, high=255, shape=(4, 84, 84), dtype=np.uint8)
    action_space = spaces.Discrete(6)

    policy = RainbowPolicy(obs_space, action_space, lr_schedule=lambda x: 1e-4)

    obs = torch.randn(2, 4, 84, 84)
    action = policy._predict(obs)

    assert action.shape == (2,)
