import os
import random
import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv

from agent import Agent, Actions

register(
    id="camera-v3",
    entry_point="environment:Environment",
)


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

        xml_path = "scene/main.xml"
        self.render_mode = render_mode

        self._init_rewards_info()
        self._init_agent()

        self._set_observation_space()

        utils.EzPickle.__init__(self)

        MujocoEnv.__init__(
            self,
            os.path.abspath(xml_path),
            1,
            observation_space=self.observation_space,
            render_mode=self.render_mode,
        )

    def _init_rewards_info(self):
        self.final_reward = 40
        self.discovery_reward = 5
        self.termination_penalty = 19
        self.time_penalty = 1

        self.step_limit = 60
        self.step_counter = 0

        self.placenta_areas = 30 - 1

    def _init_agent(self):
        seed = random.randint(0, 100)
        self.agent = Agent(seed, self.render_mode)

    def _set_observation_space(self):
        self.observation_space = spaces.Box(
            low=0,
            high=32,
            shape=((165,)),
            dtype=np.uint8,
        )

    def _set_action_space(self):
        self.action_space = spaces.Discrete(len(Actions))
        return self.action_space

    def reset_model(self, seed=None, options=None):

        self.agent.reset(seed)
        self.step_counter = 0

        observations = self.agent.get_observation()

        if self.render_mode == "human":
            print("Seed: ", seed)
            self.render()

        return observations

    def step(self, action):
        action_succes, discovered_new_area = self.agent.perform_action(Actions(action))
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

        elif self.agent.seen_areas == self.placenta_areas:
            reward += self.final_reward
            terminated = True

        observations = self.agent.get_observation()
        info = {}

        if self.render_mode == "human":
            print(
                Actions(action),
                reward,
            )
            self.render()

        return observations, reward, terminated, truncated, info

    def render(self):
        print(self.agent.map.T, "\n")
        self.agent.render()


if __name__ == "__main__":
    env = gym.make("camera-v3", render_mode="human")

    # print("Check environment begin")
    # check_env(env.unwrapped)
    # print("Check environment end")

    observations = env.reset()[0]

    for i in range(10):
        rand_action = env.action_space.sample()
        observations, reward, terminated, _, _ = env.step(rand_action)
