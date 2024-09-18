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

    def __init__(self, render_dims, render_mode, viewport_length, fovy):

        self.render_dims = render_dims
        self.render_mode = render_mode

        self._init_grid()
        self._init_image()
        self._init_position()
        self._init_renderer()

        self.agent_id = 30
        self.seen_areas = 0

        self.fovy = fovy

        self.cm2px = render_dims / viewport_length

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

        self.update_image_position(position_difference)

        if self.was_here():
            self.update_grid_position()
            return False

        # viewport = self.warp_perspective(
        #     viewport, theta_x, theta_y, position_difference
        # )
        filled = self.fill_image(viewport)
        self.process_filled_position(filled)

        return filled

    def update_image_position(self, position_difference):
        x = position_difference[0]
        y = position_difference[1]
        global_position_difference = np.array([-y, -x])

        self.image_position += (global_position_difference * self.cm2px).astype(int)

    def update_grid_position(self):
        self.grid[tuple(self.position)] += self.agent_id
        self.grid[tuple(self.last_position)] -= self.agent_id
        self.last_position = deepcopy(self.position)

    def warp_perspective(self, image, theta_x, theta_y, translation_vector):

        cv2.imshow("Pre", image)

        # Rotation matrices for x and y axis
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

        # Combined rotation
        R = np.dot(R_y, R_x)

        # Create the 4x4 homogeneous transformation matrix
        pose_matrix = np.eye(4)
        pose_matrix[:3, :3] = R
        pose_matrix[:3, 3] = translation_vector  # Translation only in x and y, z=0

        # Image dimensions
        h, w = image.shape[:2]

        # Camera intrinsic parameters (example, you should adjust based on your setup)
        self.ee_height = 0.14
        focal_length = self.ee_height / (np.tan(np.deg2rad(self.fovy) / 2))
        print(focal_length, np.deg2rad(self.fovy))
        cx, cy = w / 2, h / 2  # Principal point (center of the image)

        # Extract rotation and translation from pose matrices
        R1 = np.eye(3)
        R2 = R

        # Compute relative rotation and translation in x-y plane (since z=0)
        R_rel = R2 @ R1.T

        # Camera intrinsic matrix K
        fx = fy = focal_length[0]  # * self.cm2px

        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

        # Homography matrix for the x-y plane
        H = np.dot(K, np.dot(R, np.linalg.inv(K)))

        # Normalize H to make H[2,2] = 1 (optional but common)
        H /= H[2, 2]

        # Warp the perspective to get the top-down view
        warped_image = cv2.warpPerspective(image, H, (w, h))

        cv2.imshow("Post", warped_image)

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
