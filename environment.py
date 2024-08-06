import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env
from PIL import Image
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
import os

# import torch

import agent as cam
import random
import numpy as np

register(
    id="camera-v3",
    entry_point="environment:Environment",
)

WHITE = 255


class Environment(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 500,
    }

    def __init__(self, render_mode=None):

        utils.EzPickle.__init__(self)

        self.final_reward = 40
        self.discovery_reward = 5
        self.termination_penalty = 19
        self.time_penalty = 1

        self.step_limit = 60

        self.step_counter = 0
        self.render_mode = render_mode


        self.placenta_areas = 30 - 1

        # viewport_width = 256
        # viewport_height = 256
        # step_size = 256

        seed = random.randint(0, 100)

        self.camera = cam.Agent(seed, render_mode)

        self.observation_space = spaces.Box(
            low=0,
            high=32,
            shape=((165,)),
            dtype=np.uint8,
        )

        MujocoEnv.__init__(
            self,
            os.path.abspath("scene/main.xml"),
            1,
            observation_space=self.observation_space,
            render_mode=render_mode,
        )


    def _set_action_space(self):
        self.action_space = spaces.Discrete(len(cam.Actions))
        return self.action_space

    def reset_model(self, seed=None, options=None):

        self.camera.reset(seed)
        self.step_counter = 0

        observations = self.camera.get_observation()

        if self.render_mode == "human":
            print("Seed: ", seed)
            self.render()

        return observations

    def step(self, action):
        action_succes, discovered_new_area = self.camera.perform_action(
            cam.Actions(action)
        )
        self.step_counter += 1

        reward = 0

        if discovered_new_area:
            reward += self.discovery_reward

        reward -= self.time_penalty

        truncated = False
        if self.step_counter > self.step_limit:
            truncated = True

        terminated = False
        if not action_succes:
            reward -= self.termination_penalty
            terminated = True

        elif self.camera.seen_areas == self.placenta_areas:
            reward += self.final_reward
            terminated = True

        observations = self.camera.get_observation()
        info = {}

        if self.render_mode == "human":
            print(
                cam.Actions(action),
                reward,
            )
            print()
            self.render()

        return observations, reward, terminated, truncated, info

    def render(self):
        self.camera.render()
        print(self.camera.map)


if __name__ == "__main__":
    env = gym.make("camera-v3", render_mode="human")

    # print("Check environment begin")
    # check_env(env.unwrapped)
    # print("Check environment end")

    observations = env.reset()[0]

    for i in range(10):
        rand_action = env.action_space.sample()
        observations, reward, terminated, _, _ = env.step(rand_action)
