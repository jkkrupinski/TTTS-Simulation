import gymnasium as gym

import environment
from misc.actions import Actions


def main():
    env = gym.make("camera-v3", render_mode="human")

    observations = env.reset()[0]

    for _ in range(5):

        action = env.action_space.sample()
        observations, reward, terminated, truncated, info = env.step(action)

        print(
            Actions(action),
            reward,
        )




if __name__ == "__main__":
    main()
