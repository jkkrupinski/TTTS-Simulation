import numpy as np
from copy import deepcopy
import cv2


WHITE = 255
WHITE_THREASHOLD = 100
BLACK = 0

UNDISCOVERED = 0
FILLED = 1
EMPTY = 2


class Map:

    def __init__(self, render_dims, render_mode, viewport_length):

        self.render_dims = render_dims
        self.render_mode = render_mode

        self.reset()

        self.agent_id = 30

        self.cm2px = render_dims / viewport_length

    def reset(self):
        self._init_grid()
        self._init_image()
        self._init_position()

        cv2.destroyAllWindows()

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

    def update(self, viewport, position_difference):

        if self.after_reset():
            self.grid[tuple(self.position)] += self.agent_id
            return False

        self.update_image_position(position_difference)

        if self.was_here():
            self.update_grid_position()
            return False

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

    def process_filled_position(self, filled):
        if filled:
            self.grid[tuple(self.position)] = FILLED + self.agent_id
            self.seen_areas += 1
        else:
            self.grid[tuple(self.position)] = EMPTY + self.agent_id

        self.grid[tuple(self.last_position)] -= self.agent_id
        self.last_position = deepcopy(self.position)

    def get_position_px(self):
        return self.position * self.render_dims

    def get_position_idx(self):
        return self.position

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

        scale = 8

        original_height, original_width = self.image.shape[:2]

        new_width = original_width // scale
        new_height = original_height // scale

        self.render_map(new_width, new_height)
        self.render_grid(new_width, new_height)

    def render_map(self, new_width, new_height):

        image = deepcopy(self.image)
        image = self.add_marker(image)

        resized_image = cv2.resize(
            image, (new_width, new_height), interpolation=cv2.INTER_AREA
        )

        transposed_image = np.transpose(resized_image, (1, 0))

        cv2.imshow("Map", transposed_image)
        cv2.moveWindow("Map", 1200, 600)
        cv2.waitKey(1)

    def render_grid(self, new_width, new_height):
        resized_grid = cv2.resize(
            (self.grid + 10) * 4, (new_width, new_height), interpolation=cv2.INTER_AREA
        )

        transposed_grid = np.transpose(resized_grid, (1, 0))

        cv2.imshow("Grid", transposed_grid)
        cv2.moveWindow("Grid", 1200, 0)
        cv2.waitKey(1)

        # print(self.grid.T, "\n")

    def add_marker(self, image):

        image = cv2.rectangle(
            image,
            (self.image_position[1], self.image_position[0]),
            (
                self.image_position[1] + self.render_dims,
                self.image_position[0] + self.render_dims,
            ),
            color=100,
            thickness=6,
        )

        return image
