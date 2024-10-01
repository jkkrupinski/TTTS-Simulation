import cv2

import numpy as np
from copy import deepcopy

from misc.map_renderer import mapRenderer
from misc.warp_perspective import warp_perspective


WHITE = 255
WHITE_THREASHOLD = 100
BLACK = 0

UNDISCOVERED = 0
FILLED = 1
EMPTY = 2


class Map:

    def __init__(self, render_dims, render_mode, ee_height, fovy_deg, rcm_mode):

        self.render_dims = render_dims
        self.render_mode = render_mode
        self.rcm_mode = rcm_mode

        self._init_grid()
        self._init_image()
        self._init_position()
        self._init_renderer()

        self.agent_id = 30
        self.seen_areas = 0

        self.ee_height = ee_height

        self.fovy = np.deg2rad(fovy_deg)
        self.theta_x = 0
        self.theta_y = 0

        self.e_x = self.e_y = 0

        self.init_viewport_length = 2 * self.ee_height * np.tan(self.fovy / 2)

        self.cm2px = np.array(
            [
                self.render_dims / self.init_viewport_length,
                self.render_dims / self.init_viewport_length,
            ]
        )

    def reset(self):

        self._init_grid()
        self._init_image()
        self._init_position()

        self.renderer.reset()

        self.seen_areas = 0

    def _init_grid(self):
        self.width = 15
        self.height = 11
        self.grid = np.zeros((self.width, self.height), dtype=np.uint8)

    def _init_image(self):
        self.image = np.zeros(
            (
                self.width * self.render_dims,
                self.height * self.render_dims,
            ),
            dtype=np.uint8,
        )

    def _init_position(self):
        self.position = np.array([self.width // 2, self.height // 2])
        self.last_position = deepcopy(self.position)

        self.image_position = self.position * self.render_dims

    def _init_renderer(self):
        self.renderer = mapRenderer(self.render_dims, self.image.shape)

    def fill_map_image(self, viewport):
        was_filled = False

        for x_cam, y_cam in np.ndindex(viewport.shape):

            if viewport[x_cam, y_cam] >= WHITE_THREASHOLD:
                was_filled = True

                if self.render_mode == "human":

                    x_map = x_cam + self.image_position[0]
                    y_map = y_cam + self.image_position[1]

                    self.image[x_map, y_map] = WHITE
                else:
                    return was_filled

            else:
                if self.render_mode == "human":

                    x_map = x_cam + self.image_position[0]
                    y_map = y_cam + self.image_position[1]

                    self.image[x_map, y_map] = BLACK

        return was_filled

    def update(self, viewport, position_difference, theta_x, theta_y):

        if self.after_reset():
            self.grid[tuple(self.position)] += self.agent_id
            return False

        if self.rcm_mode:
            self.theta_x = -theta_y
            self.theta_y = -theta_x
            self.update_scaler()

        self.update_image_position(position_difference)

        if self.was_here():
            self.move_agent_marker()
            return False

        if self.rcm_mode:
            viewport = warp_perspective(viewport, self.fovy, theta_x, theta_y)

        was_filled = self.fill_map_image(viewport)
        self.update_grid(was_filled)

        return was_filled

    def update_image_position(self, position_difference):
        x = position_difference[0]
        y = position_difference[1]
        global_position_difference = np.array([-y, -x])

        new_e_x = np.tan(self.theta_x) * self.ee_height
        new_e_y = np.tan(self.theta_y) * self.ee_height

        diff_x = new_e_x - self.e_x
        diff_y = new_e_y - self.e_y

        self.e_x = new_e_x
        self.e_y = new_e_y

        self.image_position[0] += (
            (global_position_difference[0] + diff_x) * self.cm2px[0]
        ).astype(int)
        self.image_position[1] += (
            (global_position_difference[1] + diff_y) * self.cm2px[1]
        ).astype(int)

    def update_scaler(self):

        viewport_length_x = self.init_viewport_length / np.cos(self.theta_x)
        viewport_length_y = self.init_viewport_length / np.cos(self.theta_y)

        if viewport_length_x == 0:
            viewport_length_x = self.init_viewport_length
        if viewport_length_y == 0:
            viewport_length_y = self.init_viewport_length

        self.cm2px = np.array(
            [
                self.render_dims / viewport_length_x,
                self.render_dims / viewport_length_y,
            ]
        )

    def move_agent_marker(self):
        self.grid[tuple(self.position)] += self.agent_id
        self.grid[tuple(self.last_position)] -= self.agent_id
        self.last_position = deepcopy(self.position)

    def update_grid(self, was_filled):
        if was_filled:
            self.grid[tuple(self.position)] = FILLED + self.agent_id
            self.seen_areas += 1
        else:
            self.grid[tuple(self.position)] = EMPTY + self.agent_id

        self.grid[tuple(self.last_position)] -= self.agent_id
        self.last_position = deepcopy(self.position)

    def get_observation(self):
        return self.grid.flatten()

    def is_position_outside(self, position) -> bool:
        x = position[0]
        y = position[1]

        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return True
        return False

    def was_here(self):
        return self.grid[tuple(self.position)] == FILLED

    def after_reset(self):
        return tuple(self.position) == tuple(self.last_position)

    def render(self):
        self.renderer.render(self.image, self.grid, self.image_position)
