import numpy as np

from controller import Controller
from map import Map
from misc.actions import Actions


class Agent:
    def __init__(self, render_mode) -> None:

        self.render_mode = render_mode

        self.ee_height = 0.14
        self.rcm_height = 0.28

        self.controller = Controller(
            self.render_mode, self.ee_height, self.rcm_height, rcm_mode=True
        )

        self.robot_step = 0.011
        self.num_of_steps = 2

        placenta_height = 0.01

        height_from_placenta = self.ee_height - placenta_height

        # viewport_length = 2 * height_from_placenta * np.tan(np.deg2rad(fovy / 2))
        # print(viewport_length)

        self.map = Map(
            self.controller.render_dims,
            self.render_mode,
            height_from_placenta,
            self.controller.feto_fovy,
        )

        self.map.update(
            self.get_viewport(), self.controller.get_ee_pos(), theta_x=0, theta_y=0
        )

    def reset(self):

        self.controller.reset()
        self.robot_ee_position = self.controller.get_ee_pos()

        self.map.reset()
        self.map.update(
            self.get_viewport(), self.controller.get_ee_pos(), theta_x=0, theta_y=0
        )

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
        self.controller.wait_for_ms(100)

        return action_success, discovered_new_area

    def update_map(self, position_difference):
        viewport = self.get_viewport()
        theta_x, theta_y = self.controller.get_ee_rotation()
        return self.map.update(viewport, position_difference, theta_x, theta_y)

    def move_robot(self, movement_vector):

        partial_vector = [x / self.num_of_steps for x in movement_vector]

        begin_position = self.controller.get_ee_pos()

        for _ in range(self.num_of_steps):

            # a = self.controller.get_ee_pos()

            self.robot_ee_position += partial_vector
            self.controller.move_ee(self.robot_ee_position, partial_vector)
            self.controller.wait_for_ms(100)

            # b = self.controller.get_ee_pos()

            # if _ == 0:
            #     self.map.fill_image(self.get_viewport(), b - a)

        end_position = self.controller.get_ee_pos()
        position_difference = end_position - begin_position

        return position_difference

    def render(self):
        self.map.render()
