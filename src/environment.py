import os
import random
import numpy as np
import curses


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

        return observations

    def step(self, action):
        action_success, discovered_new_area = self.agent.perform_action(Actions(action))
        self.step_counter += 1

        reward = self.calculate_reward(action_success, discovered_new_area)
        terminated, truncated = self.check_termination_conditions(action_success)

        observations = self.agent.get_observation()
        info = {}

        if self.render_mode == "human":
            self.render()

        return observations, reward, terminated, truncated, info

    def calculate_reward(self, action_success, discovered_new_area):
        reward = 0
        if discovered_new_area:
            reward += self.discovery_reward

        reward -= self.time_penalty

        if not action_success:
            reward -= self.termination_penalty
        elif self.agent.map.seen_areas == self.placenta_areas:
            reward += self.final_reward

        return reward

    def check_termination_conditions(self, action_success):
        terminated = False
        truncated = self.step_counter > self.step_limit

        if not action_success or self.agent.map.seen_areas == self.placenta_areas:
            terminated = True

        return terminated, truncated

    def render(self):
        self.agent.render()


def main(stdscr):
    env = gym.make("camera-v3", render_mode="human")

    # print("Check environment begin")
    # check_env(env.unwrapped)
    # print("Check environment end")

    observations = env.reset()[0]

    run = True
    action = 0

    stdscr.addstr("Press the arrow keys to move or 'q' to quit.\n")
    stdscr.refresh()

    while run:

        key = stdscr.getch()  # Wait for user input

        if key == curses.KEY_UP:
            action = 0
        elif key == curses.KEY_DOWN:
            action = 1
        elif key == curses.KEY_LEFT:
            action = 2
        elif key == curses.KEY_RIGHT:
            action = 3
        elif key == ord("q"):
            stdscr.addstr("Exiting...\n")
            run = False
            break

        # rand_action = env.action_space.sample()
        observations, reward, terminated, truncated, info = env.step(action)

        # stdscr.addstr("Action: %s, Reward: %d \n" % (Actions(action), reward))
        # print(
        #     Actions(action),
        #     reward,
        # )


if __name__ == "__main__":
    curses.wrapper(main)
