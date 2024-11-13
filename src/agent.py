import numpy as np

from controller import Controller
from map import Map
from segmenter import Segmenter

from misc.actions import Actions


class Agent:
    def __init__(self, render_mode, rcm_mode) -> None:

        self.render_mode = render_mode
        self.rcm_mode = rcm_mode

        self._init_controller()
        self._init_segmenter()
        self._init_map()

        if self.rcm_mode:
            self.robot_step = 0.011  # chosen while testing
        else:
            self.robot_step = 0.022  # viewport_length

        self.num_of_steps = 2

    def _init_controller(self):
        self.ee_height = 0.14
        self.rcm_height = 0.28

        self.controller = Controller(
            self.render_mode, self.ee_height, self.rcm_height, self.rcm_mode
        )

        placenta_height = self.controller.placenta_height
        self.height_from_placenta = self.ee_height - placenta_height

        # viewport_length = 2 * height_from_placenta * np.tan(np.deg2rad(fovy / 2))
        # print(viewport_length)

    def _init_segmenter(self):
        model_list = [
            "TTTSNet_model-fold-0.pt",
            "TTTSNet_model-fold-1.pt",
            "TTTSNet_model-fold-2.pt",
            "TTTSNet_model-fold-3.pt",
            "TTTSNet_model-fold-4.pt",
            "TTTSNet_model-fold-5.pt",
        ]
        self.segmenter = Segmenter(model_list)

    def _init_map(self):
        self.map = Map(
            self.controller.render_dims,
            self.render_mode,
            self.height_from_placenta,
            self.controller.feto_fovy,
            self.rcm_mode,
        )

        self.map.update(
            self.get_segmentation(), self.controller.get_ee_pos(), theta_x=0, theta_y=0
        )

    def reset(self):

        self.controller.reset()
        self.robot_ee_position = self.controller.get_ee_pos()

        self.map.reset()
        self.map.update(
            self.get_segmentation(), self.controller.get_ee_pos(), theta_x=0, theta_y=0
        )

    def get_segmentation(self):
        viewport = self.controller.get_feto_image()
        segmentation = self.segmenter(viewport)
        return segmentation

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
        self.controller.wait_for_ms(10)

        return action_success, discovered_new_area

    def update_map(self, position_difference):
        segmentation = self.get_segmentation()
        theta_x, theta_y = self.controller.get_ee_rotation()
        return self.map.update(segmentation, position_difference, theta_x, theta_y)

    def move_robot(self, movement_vector):

        partial_vector = [x / self.num_of_steps for x in movement_vector]

        begin_position = self.controller.get_ee_pos()

        for _ in range(self.num_of_steps):

            self.robot_ee_position += partial_vector
            self.controller.move_ee(self.robot_ee_position, partial_vector)
            self.controller.wait_for_ms(10)

        end_position = self.controller.get_ee_pos()
        position_difference = end_position - begin_position

        return position_difference

    def render(self):
        self.map.render()
