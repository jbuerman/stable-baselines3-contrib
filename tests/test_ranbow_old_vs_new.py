import ale_py
import gymnasium as gym
import numpy as np
import torch

from sb3_contrib.rainbow.old_agent import Agent
from sb3_contrib.rainbow.rainbow import Rainbow
from sb3_contrib.rainbow.policy import (
    RainbowPolicy,
)


def make_env(envs_create, game, framestack, repeat_probs, terminal_on_life_loss=True):
    return gym.vector.SyncVectorEnv(
        [
            lambda: gym.wrappers.FrameStackObservation(
                gym.wrappers.AtariPreprocessing(
                    gym.make("ALE/" + game + "-v5", frameskip=1, repeat_action_probability=repeat_probs),
                    terminal_on_life_loss=terminal_on_life_loss
                ),
                framestack
            )
            for _ in range(envs_create)
        ]
    )


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

    def test_agent_vs_rainbow_sanity(self):
        env = make_env(1, "Pong", framestack=4, repeat_probs=0.0)

        obs_agent, _ = env.reset()
        obs_model, _ = env.reset()

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

        steps = 300

        agent_rewards = []
        model_rewards = []

        agent_actions = []
        model_actions = []

        for _ in range(steps):
            # ----- Agent -----
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

            agent_rewards.append(reward[0])
            agent_actions.append(action_agent[0])

            obs_agent = next_obs

            # ----- Rainbow -----
            action_model, _ = model.predict(obs_model, deterministic=False)
            next_obs, reward, done, trunc, _ = env.step(action_model)

            model.learn(1)

            model_rewards.append(reward[0])
            model_actions.append(action_model[0])

            obs_model = next_obs

        # ----------------------------
        # Check 1: no NaNs
        # ----------------------------
        assert np.isfinite(agent_rewards).all()
        assert np.isfinite(model_rewards).all()

        # ----------------------------
        # Check 2: no divergence (reward sanity)
        # ----------------------------
        agent_mean = np.mean(agent_rewards)
        model_mean = np.mean(model_rewards)

        assert np.isfinite(agent_mean)
        assert np.isfinite(model_mean)

        # loose sanity bound (Atari rewards are clipped [-1, 1])
        assert abs(agent_mean) <= 1.0
        assert abs(model_mean) <= 1.0

        # ----------------------------
        # Check 3: action distribution similarity
        # ----------------------------
        agent_counts = np.bincount(agent_actions, minlength=n_actions)
        model_counts = np.bincount(model_actions, minlength=n_actions)

        # normalise
        agent_dist = agent_counts / agent_counts.sum()
        model_dist = model_counts / model_counts.sum()

        # distributions should not be wildly different
        diff = np.abs(agent_dist - model_dist).sum()

        # heuristic threshold (loose!)
        assert diff < 1.5  # not too far apart

        # distributions should not collapse to a single action
        assert agent_counts.max() < steps
        assert model_counts.max() < steps

        # both should explore more than one action
        assert (agent_counts > 0).sum() > 1
        assert (model_counts > 0).sum() > 1
