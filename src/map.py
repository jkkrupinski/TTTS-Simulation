import cv2

import numpy as np
from copy import deepcopy

from misc.map_renderer import mapRenderer


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

    def fill_image(self, viewport):
        filled = False

        for x_cam, y_cam in np.ndindex(viewport.shape):

            if viewport[x_cam, y_cam] >= WHITE_THREASHOLD:
                filled = True

                if self.render_mode == "human":

                    x_map = x_cam + self.image_position[0]
                    y_map = y_cam + self.image_position[1]

                    self.image[x_map, y_map] = WHITE
                else:
                    return filled

            else:
                if self.render_mode == "human":

                    x_map = x_cam + self.image_position[0]
                    y_map = y_cam + self.image_position[1]

                    self.image[x_map, y_map] = BLACK

        return filled

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
            self.update_grid_position()
            return False

        if self.rcm_mode:
            viewport = self.warp_perspective(viewport, theta_x, theta_y)

        filled = self.fill_image(viewport)
        self.process_filled_position(filled)

        return filled

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

    def update_grid_position(self):
        self.grid[tuple(self.position)] += self.agent_id
        self.grid[tuple(self.last_position)] -= self.agent_id
        self.last_position = deepcopy(self.position)

    def warp_perspective(self, image, theta_x, theta_y):

        R_x = np.array(
            [
                [1, 0, 0],
                [0, np.cos(theta_x), -np.sin(theta_x)],
                [0, np.sin(theta_x), np.cos(theta_x)],
            ]
        )

        R_y = np.array(
            [
                [np.cos(theta_y), 0, np.sin(theta_y)],
                [0, 1, 0],
                [-np.sin(theta_y), 0, np.cos(theta_y)],
            ]
        )

        R = R_y @ R_x

        h, w = image.shape[:2]

        focal_length = 256 / (2 * np.tan(self.fovy / 2))
        cx, cy = (
            w / 2,
            h / 2,
        )  # Principal point (center of the image)

        # Camera intrinsic matrix K
        fx = fy = focal_length[0]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

        H = K @ R @ np.linalg.inv(K)
        H = H / H[2, 2]

        # Calculate new bounding box for the warped image
        # Create a set of points to warp
        points = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype="float32")
        warped_points = cv2.perspectiveTransform(points[None, :, :], H)[0]

        # Determine the bounding box of the warped image
        min_x = int(np.min(warped_points[:, 0]))
        max_x = int(np.max(warped_points[:, 0]))
        min_y = int(np.min(warped_points[:, 1]))
        max_y = int(np.max(warped_points[:, 1]))

        # Calculate the size of the new image
        new_width = max_x - min_x
        new_height = max_y - min_y

        # Offset the homography to ensure the image is correctly positioned
        translation = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]])
        H = translation @ H  # Adjust the homography

        # Warp the perspective to get the top-down view
        warped_image = cv2.warpPerspective(image, H, (new_width, new_height))

        return warped_image

    def process_filled_position(self, filled):
        if filled:
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
