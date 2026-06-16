from sb3_contrib.rainbow2.rainbow import Rainbow
from sb3_contrib.rainbow2.rainbow_policy import RainbowPolicy
from sb3_contrib.rainbow2.main_agent import make_env


def main():
    env = make_env(1, "Pong", framestack=4, repeat_probs=0.0).envs[0]

    model = Rainbow(
        RainbowPolicy,
        env,
        learning_starts=10,
        batch_size=4,
        buffer_size=50_000,
    )

    print("Model created successfully")

    model.learn(1000)

if __name__ == '__main__':
    main()
