import numpy as np
from copy import deepcopy
import cv2


WHITE = 255
BLACK = 0

UNDISCOVERED = 0
FILLED = 1
EMPTY = 2


class Map:

    def __init__(self, render_dims, render_mode):

        self.render_dims = render_dims
        self.render_mode = render_mode

        self.reset()

        self.agent_id = 30

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

    def fill_image(self, viewport):
        filled = False

        for x_cam, y_cam in np.ndindex(viewport.shape):

            if viewport[x_cam, y_cam] == WHITE:
                filled = True

                if self.render_mode == "human":
                    x_map, y_map = self.cam_2_map(x_cam, y_cam)
                    self.image[x_map, y_map] = WHITE
                else:
                    return filled

        return filled

    def update(self, viewport):

        if self.was_here():
            self.update_position()
            return False

        filled = self.fill_image(viewport)
        self.process_filled_position(filled)

        return filled

    def update_position(self):
        self.grid[tuple(self.position)] += self.agent_id
        self.grid[tuple(self.last_position)] -= self.agent_id
        self.last_position = deepcopy(self.position)

    def process_filled_position(self, filled):
        if filled:
            self.grid[tuple(self.position)] = FILLED + self.agent_id
            self.seen_areas += 1
        else:
            self.grid[tuple(self.position)] = EMPTY + self.agent_id

        # After reset positions are the same
        if tuple(self.position) != tuple(self.last_position):
            self.grid[tuple(self.last_position)] -= self.agent_id

        self.last_position = deepcopy(self.position)

    def get_position_px(self):
        return self.position * self.render_dims

    def get_position_idx(self):
        return self.position

    def get_observation(self):
        return self.grid.flatten()

    def cam_2_map(self, x_cam, y_cam):
        return [x_cam, y_cam] + self.get_position_px()

    def is_position_outside(self, position) -> bool:
        x = position[0]
        y = position[1]

        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return True
        return False

    def was_here(self):
        return self.grid[tuple(self.position)] == FILLED

    def render(self):
        original_height, original_width = self.image.shape[:2]

        new_width = original_width // 6
        new_height = original_height // 6

        resized_image = cv2.resize(
            self.image, (new_width, new_height), interpolation=cv2.INTER_AREA
        )

        # image = self.add_marker(resized_image)
        transposed_image = np.transpose(resized_image, (1, 0))  # Swap x and y axes

        cv2.imshow("Map", transposed_image)
        cv2.moveWindow("Map", 1200, 600)
        cv2.waitKey(1)

        print(self.grid.T, "\n")


##############################################################


def fill_map_image2(self):
    observation = self.get_viewport()
    filled = False

    for x_cam in range(observation.shape[0]):
        for y_cam in range(observation.shape[1]):
            x_map, y_map = self.cam2map(x_cam, y_cam)

            if observation[x_cam, y_cam] == WHITE:
                filled = True

                if self.render_mode == "human":
                    self.map_image[x_map, y_map] = WHITE
                else:
                    return filled

            elif observation[x_cam, y_cam] == BLACK:
                if self.render_mode == "human":
                    self.map_image[x_map, y_map] = BLACK

    return filled


def update_map2(self):
    filled = False

    map_indexes = self.get_map_indexes()

    filled = self.fill_map_image2()

    if self.been_there(map_indexes):
        self.map[map_indexes] = FILLED + self.agent_id
        return False

    if filled:
        self.map[map_indexes] = FILLED + self.agent_id
        self.seen_areas += 1
        return True
    else:
        self.map[map_indexes] = EMPTY + self.agent_id
        return False


def add_marker(self, image, marker_size=60, color=100):

    x, y = self.map.get_position_px()

    # Calculate the top-left corner of the marker
    half_marker_size = marker_size // 2
    marker_top_left_x = x - half_marker_size
    marker_top_left_y = y - half_marker_size

    image = cv2.rectangle(
        image,
        (marker_top_left_x, marker_top_left_y),
        (marker_top_left_x + marker_size, marker_top_left_y + marker_size),
        color,
        -1,
    )

    return image