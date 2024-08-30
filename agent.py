from enum import Enum

from controller import Controller
from map import Map


class Actions(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class Agent:
    def __init__(self, seed, render_mode) -> None:

        self.render_mode = render_mode

        self.controller = Controller(self.render_mode)
        self.map = Map(self.controller.render_dims, self.render_mode)

        self.robot_step = 0.03
        self.robot_height = 0.15

        self.reset(seed)

    def reset(self, seed):
        self._init_start_position(seed)

        self.controller.reset()

        self.map.reset()
        self.map.update(self.get_viewport())

    def _init_start_position(self, seed):
        self.robot_ee_pos = self.controller.get_ee_pos()
        self.robot_ee_pos -= [0, 0, self.robot_height]

    def get_viewport(self):
        return self.controller.get_feto_image()

    def get_observation(self):
        return self.map.get_observation()

    def perform_action(self, action: Actions) -> bool:

        action_success = False
        discovered_new_area = False

        if action == Actions.LEFT:
            if self.map.is_position_outside(self.map.position - [1, 0]):
                return action_success, discovered_new_area

            self.map.position -= [1, 0]
            self.robot_ee_pos += [0, self.robot_step, 0]

        elif action == Actions.RIGHT:
            if self.map.is_position_outside(self.map.position + [1, 0]):
                return action_success, discovered_new_area

            self.map.position += [1, 0]
            self.robot_ee_pos -= [0, self.robot_step, 0]

        elif action == Actions.UP:
            if self.map.is_position_outside(self.map.position - [0, 1]):
                return action_success, discovered_new_area

            self.map.position -= [0, 1]
            self.robot_ee_pos += [self.robot_step, 0, 0]

        elif action == Actions.DOWN:
            if self.map.is_position_outside(self.map.position + [0, 1]):
                return action_success, discovered_new_area

            self.map.position += [0, 1]
            self.robot_ee_pos -= [self.robot_step, 0, 0]

        action_success = True
        self.controller.move_ee(self.robot_ee_pos)
        self.controller.wait_for_ms(1000)

        discovered_new_area = self.map.update(self.get_viewport())

        return action_success, discovered_new_area

    def render(self):
        self.map.render()
