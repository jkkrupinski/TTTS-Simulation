from enum import Enum
import numpy as np


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

        self.robot_step = 0.01
        self.robot_height = 0.22
        self.steps = 2

        fovy = self.controller.feto_fovy
        viewport_length = (
            2 * (0.3 - self.robot_height - 0.01) * np.tan(np.deg2rad(fovy / 2))
        )

        self.map = Map(self.controller.render_dims, self.render_mode, viewport_length)

        self.reset(seed)

    def reset(self, seed):
        self._init_start_position(seed)

        self.controller.reset()

        self.map.reset()
        self.map.update(self.get_viewport(), self.controller.get_ee_pos())

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

        movement_vector = [0, 0, 0]

        if action == Actions.LEFT:
            if self.map.is_position_outside(self.map.position - [1, 0]):
                return action_success, discovered_new_area

            self.map.position -= [1, 0]
            movement_vector = [0, self.robot_step, 0]

        elif action == Actions.RIGHT:
            if self.map.is_position_outside(self.map.position + [1, 0]):
                return action_success, discovered_new_area

            self.map.position += [1, 0]
            movement_vector = [0, -self.robot_step, 0]

        elif action == Actions.UP:
            if self.map.is_position_outside(self.map.position - [0, 1]):
                return action_success, discovered_new_area

            self.map.position -= [0, 1]
            movement_vector = [self.robot_step, 0, 0]

        elif action == Actions.DOWN:
            if self.map.is_position_outside(self.map.position + [0, 1]):
                return action_success, discovered_new_area

            self.map.position += [0, 1]
            movement_vector = [-self.robot_step, 0, 0]

        action_success = True

        position_difference = self.move_robot(movement_vector)
        discovered_new_area = self.update_map(position_difference)
        self.controller.wait_for_ms(1000)

        # print("Moved by: ", np.round(vec, 3))

        return action_success, discovered_new_area

    def update_map(self, position_difference):
        viewport = self.get_viewport()
        return self.map.update(viewport, position_difference)

    def move_robot(self, movement_vector):

        partial_vector = [x / self.steps for x in movement_vector]

        begin_position = self.controller.get_ee_pos()

        for _ in range(self.steps):

            # a = self.controller.get_ee_pos()

            self.robot_ee_pos += partial_vector
            self.controller.move_ee(self.robot_ee_pos)
            self.controller.wait_for_ms(100)

            # b = self.controller.get_ee_pos()

            # if _ == 0:
            #     self.map.fill_image(self.get_viewport(), b - a)

        end_position = self.controller.get_ee_pos()
        position_difference = end_position - begin_position

        return position_difference

    def render(self):
        self.map.render()
