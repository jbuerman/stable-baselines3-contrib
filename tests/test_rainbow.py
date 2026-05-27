import gymnasium as gym
from sb3_contrib import Rainbow

def test_rainbow_runs():
    env = gym.make("CartPole-v1")
    model = Rainbow("MlpPolicy", env, learning_starts=10, buffer_size=1000)
    model.learn(100)
