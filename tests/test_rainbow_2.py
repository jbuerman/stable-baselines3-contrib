import numpy as np
import torch
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

        assert batch.observations.device.type == model.device.type
        assert batch.next_observations.device.type == model.device.type
        assert batch.weights.device.type == model.device.type


class TestRainbowBufferPER:

    def test_per_priority_update_changes_sampling(self):
        env = make_env(1, "Pong", framestack=4, repeat_probs=0.0).envs[0]

        model = Rainbow(
            RainbowPolicy,
            env,
            learning_starts=10,
            batch_size=32,
            buffer_size=2000,
        )

        model.learn(500)

        batch1 = model.replay_buffer.sample(32)
        idxs = batch1.idxs.copy()

        # Set very high priorities
        high_priorities = np.ones_like(idxs, dtype=np.float32) * 100.0
        model.replay_buffer.update_priorities(idxs, high_priorities)

        batch2 = model.replay_buffer.sample(32)

        # We don't expect equality anymore
        assert not np.array_equal(batch1.idxs, batch2.idxs)

    def test_per_weights_bounds(self):
        env = make_env(1, "Pong", framestack=4, repeat_probs=0.0).envs[0]

        model = Rainbow(
            RainbowPolicy,
            env,
            learning_starts=10,
            batch_size=32,
            buffer_size=2000,
        )

        model.learn(500)

        batch = model.replay_buffer.sample(32)

        assert torch.all(batch.weights > 0)
        assert torch.all(batch.weights <= 1.0)

    def test_per_beta_annealing(self):
        env = make_env(1, "Pong", framestack=4, repeat_probs=0.0).envs[0]

        model = Rainbow(
            RainbowPolicy,
            env,
            learning_starts=10,
            batch_size=8,
            buffer_size=1000,
        )

        buffer = model.replay_buffer
        initial_beta = buffer.beta

        model.learn(500)

        assert buffer.beta > initial_beta
        assert buffer.beta <= 1.0


class TestRainbowTraining:

    def test_gradients_flow(self):
        env = make_env(1, "Pong", framestack=4, repeat_probs=0.0).envs[0]

        model = Rainbow(
            RainbowPolicy,
            env,
            learning_starts=10,
            batch_size=8,
            buffer_size=1000,
        )

        model.learn(200)

        params = list(model.q_net.parameters())
        has_grad = any(p.grad is not None for p in params)

        assert has_grad

    def test_parameters_update(self):
        env = make_env(1, "Pong", framestack=4, repeat_probs=0.0).envs[0]

        model = Rainbow(
            RainbowPolicy,
            env,
            learning_starts=10,
            batch_size=8,
            buffer_size=1000,
        )

        params_before = [p.clone().detach() for p in model.q_net.parameters()]

        model.learn(500)

        params_after = list(model.q_net.parameters())

        changed = False
        for p_before, p_after in zip(params_before, params_after):
            if not torch.equal(p_before, p_after):
                changed = True
                break

        assert changed

    def test_loss_is_finite(self):
        env = make_env(1, "Pong", framestack=4, repeat_probs=0.0).envs[0]

        model = Rainbow(
            RainbowPolicy,
            env,
            learning_starts=10,
            batch_size=8,
            buffer_size=1000,
        )

        model.learn(200)

        loss = model.last_loss

        assert not np.isnan(loss)
        assert not np.isinf(loss)

    def test_longer_training_runs(self):
        env = make_env(1, "Pong", framestack=4, repeat_probs=0.0).envs[0]

        model = Rainbow(
            RainbowPolicy,
            env,
            learning_starts=10,
            batch_size=16,
            buffer_size=5000,
        )

        model.learn(2000)


class TestRegressionVsAgent:
    """
    These are to check Agent -> SB3 to ensure completeness. TODO should be removed before submitting to SB3
    """

    def test_action_shape_consistency_vs_agent(self):
        env = make_env(1, "Pong", framestack=4, repeat_probs=0.0)
        obs, _ = env.reset()

        n_actions = env.action_space[0].n

        agent = Agent(
            n_actions=n_actions,
            input_dims=[4, 84, 84],
            device="cpu",
            num_envs=1,
            agent_name="test",
            total_steps=1000,
            testing=True
        )

        model = Rainbow(
            RainbowPolicy,
            env.envs[0],
            learning_starts=10,
            batch_size=8,
            buffer_size=1000,
            device="cpu",
        )

        # Convert observation for policy
        obs_tensor = torch.tensor(obs, dtype=torch.float32)

        agent_action = agent.choose_action(obs)
        model_action = model.policy._predict(obs_tensor)

        assert agent_action.shape == model_action.shape

    def test_short_training_behaviour_vs_agent(self):
        env = make_env(1, "Pong", framestack=4, repeat_probs=0.0)

        obs, _ = env.reset()
        n_actions = env.action_space[0].n

        agent = Agent(
            n_actions=n_actions,
            input_dims=[4, 84, 84],
            device="cpu",
            num_envs=1,
            agent_name="test",
            total_steps=2000,
            testing=True,
            batch_size=8
        )

        model = Rainbow(
            RainbowPolicy,
            env.envs[0],
            learning_starts=10,
            batch_size=8,
            buffer_size=2000,
            device="cpu",
        )

        steps = 200

        agent_rewards = []
        model_rewards = []

        obs_agent, _ = env.reset()
        obs_model, _ = env.reset()

        for _ in range(steps):
            # Agent step
            action_agent = agent.choose_action(obs_agent)
            next_obs, reward, done, trunc, _ = env.step(action_agent)

            agent.store_transition(
                obs_agent[0],
                action_agent[0],
                reward[0],
                next_obs[0],
                done[0],
                trunc[0],
                stream=0
            )
            agent.learn()

            obs_agent = next_obs
            agent_rewards.append(reward[0])

            # Model step
            action_model, _ = model.predict(obs_model, deterministic=False)
            next_obs, reward, done, trunc, _ = env.step(action_model)

            model.learn(1)

            obs_model = next_obs
            model_rewards.append(reward[0])

        # Compare rough behaviour (not equality!)
        agent_mean = np.mean(agent_rewards)
        model_mean = np.mean(model_rewards)

        assert np.isfinite(agent_mean)
        assert np.isfinite(model_mean)
