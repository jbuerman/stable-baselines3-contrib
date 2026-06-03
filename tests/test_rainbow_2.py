import torch
import numpy as np

from gymnasium import spaces

from sb3_contrib.rainbow2.rainbow import Agent, make_env, NatureC51, PER, Rainbow
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


class TestRainbowBufferInterface:

    def test_replay_sample_structure(self):
        env = make_env(1, "Pong", framestack=4, repeat_probs=0.0).envs[0]

        model = Rainbow(
            RainbowPolicy,
            env,
            learning_starts=10,
            batch_size=8,
            buffer_size=1000,
        )

        model.learn(200)

        batch = model.replay_buffer.sample(8)

        assert hasattr(batch, "observations")
        assert hasattr(batch, "actions")
        assert hasattr(batch, "next_observations")
        assert hasattr(batch, "dones")
        assert hasattr(batch, "rewards")
        assert hasattr(batch, "idxs")
        assert hasattr(batch, "weights")


    def test_replay_sample_shapes_and_types(self):
        env = make_env(1, "Pong", framestack=4, repeat_probs=0.0).envs[0]

        model = Rainbow(
            RainbowPolicy,
            env,
            learning_starts=10,
            batch_size=8,
            buffer_size=1000,
        )

        model.learn(200)

        batch = model.replay_buffer.sample(8)

        assert batch.observations.shape[0] == 8
        assert batch.actions.shape[0] == 8
        assert batch.next_observations.shape[0] == 8
        assert batch.rewards.shape[0] == 8
        assert batch.dones.shape[0] == 8

        assert batch.observations.dtype == torch.float32
        assert batch.actions.dtype == torch.int64
        assert batch.rewards.dtype == torch.float32


    def test_replay_sample_device(self):
        env = make_env(1, "Pong", framestack=4, repeat_probs=0.0).envs[0]

        model = Rainbow(
            RainbowPolicy,
            env,
            learning_starts=10,
            batch_size=8,
            buffer_size=1000,
        )

        model.learn(200)

        batch = model.replay_buffer.sample(8)

        assert batch.observations.device == model.device
        assert batch.next_observations.device == model.device
        assert batch.weights.device == model.device


class TestRainbowBufferPER:
    pass # TODO fill in next