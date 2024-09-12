import gymnasium as gym
from gymnasium.utils.env_checker import check_env

import environment


def main():
    env = gym.make("camera-v3", render_mode="human")

    print("Check environment begin")
    check_env(env.unwrapped)
    print("Check environment end")
