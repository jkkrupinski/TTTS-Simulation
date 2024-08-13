from enum import Enum
import numpy as np
import random
import numpy as np
from matplotlib import pyplot as plt
from controller import Controller
import cv2

WHITE = 255

UNDISCOVERED = 0
FILLED = 1
EMPTY = 2


class Actions(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class Placenta:
    width_idx = 7
    height_idx = 5


class Agent:
    def __init__(self, seed, render_mode) -> None:

        self.controller = Controller()

        self.placenta = Placenta()
        self.render_mode = render_mode
        self.agent_id = 30

        self.step = self.controller.render_dims
        self.viewport_width = 256
        self.viewport_height = 256

        self.robot_step = 0.05

        self.reset(seed)

    def _init_map(self):
        self.map_width_idx = 15
        self.map_height_idx = 11
        self.map = np.zeros((self.map_width_idx, self.map_height_idx), dtype=np.uint8)

    def _init_map_image(self):
        self.map_image = np.zeros(
            (
                self.map_width_idx * self.viewport_width,
                self.map_height_idx * self.viewport_height,
            ),
            dtype=np.uint8,
        )

    def _init_start_position(self, seed):
        random.seed(seed)

        self.map_position = [
            self.placenta.width_idx * self.step,
            self.placenta.height_idx * self.step,
        ]

        # Not to spawn on empty edges, instead somewhere in middle of placenta
        offset = 2

        self.placenta_start_position = [
            random.randint(0, self.placenta.width_idx - offset) * self.step,
            random.randint(0, self.placenta.height_idx - offset) * self.step,
        ]

        self.placenta_start_position = [
            int(self.placenta.width_idx * self.step / 2),
            int(self.placenta.height_idx * self.step / 2),
        ]
        self.placenta_position = self.placenta_start_position

        self.robot_ee_pos = self.controller.get_ee_pos()

    def get_viewport(self):
        viewport = self.controller.get_feto_image()
        return viewport

    def map2placenta(self):
        x = self.map_position[0] - self.placenta_start_position[0]
        y = self.map_position[1] - self.placenta_start_position[1]

        return x, y

    def cam2map(self, x_cam, y_cam):
        x_placenta = x_cam + self.placenta_position[0]
        y_placenta = y_cam + self.placenta_position[1]

        t_x, t_y = self.map2placenta()

        x_map = x_placenta + t_x
        y_map = y_placenta + t_y

        return x_map, y_map

    def fill_map_image(self):
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

        return filled

    def get_observation(self):
        return self.map.flatten()

    def get_map_indexes(self):
        return int(self.map_position[0] / 256), int(self.map_position[1] / 256)

    def update_map(self):
        filled = False

        map_indexes = self.get_map_indexes()

        if self.been_there(map_indexes):
            self.map[map_indexes] = FILLED + self.agent_id
            return False

        filled = self.fill_map_image()

        if filled:
            self.map[map_indexes] = FILLED + self.agent_id
            self.seen_areas += 1
            return True
        else:
            self.map[map_indexes] = EMPTY + self.agent_id
            return False

    def reset(self, seed):
        self._init_map()
        self._init_map_image()
        self._init_start_position(seed)

        self.seen_areas = 0
        self.update_map()

        cv2.destroyAllWindows()

        self.controller.reset()

    def is_x_in_map(self, x) -> bool:
        if x >= 0 and x <= self.map_image.shape[0] - self.viewport_width:
            return True
        return False

    def is_y_in_map(self, y) -> bool:
        if y >= 0 and y <= self.map_image.shape[1] - self.viewport_height:
            return True
        return False

    def been_there(self, map_indexes):
        return self.map[map_indexes] == FILLED

    def perform_action(self, action: Actions) -> bool:
        action_succes = False
        discovered_new_area = False

        previous_map_indexes = self.get_map_indexes()
        # print(np.round(self.robot_ee_pos,2))

        if action == Actions.LEFT:
            if self.is_x_in_map(self.map_position[0] - self.step):
                self.map_position[0] -= self.step
                self.placenta_position[0] -= self.step
                self.robot_ee_pos += [0, self.robot_step, 0]
                self.controller.move_ee(self.robot_ee_pos)
                action_succes = True

        elif action == Actions.RIGHT:
            if self.is_x_in_map(self.map_position[0] + self.step):
                self.map_position[0] += self.step
                self.placenta_position[0] += self.step
                self.robot_ee_pos -= [0, self.robot_step, 0]
                self.controller.move_ee(self.robot_ee_pos)
                action_succes = True

        elif action == Actions.UP:
            if self.is_y_in_map(self.map_position[1] - self.step):
                self.map_position[1] -= self.step
                self.placenta_position[1] -= self.step
                self.robot_ee_pos += [self.robot_step, 0, 0]
                self.controller.move_ee(self.robot_ee_pos)
                action_succes = True

        elif action == Actions.DOWN:
            if self.is_y_in_map(self.map_position[1] + self.step):
                self.map_position[1] += self.step
                self.placenta_position[1] += self.step
                self.robot_ee_pos -= [self.robot_step, 0, 0]
                self.controller.move_ee(self.robot_ee_pos)
                action_succes = True

        self.controller.wait_for_ms(2000)

        if action_succes:
            self.map[previous_map_indexes] -= self.agent_id
            discovered_new_area = self.update_map()

        return action_succes, discovered_new_area

    def render(self):

        if self.map_image.dtype != np.uint8:
            raise ValueError("Image data type must be uint8")

        # Check the value range
        if self.map_image.min() < 0 or self.map_image.max() > 255:
            raise ValueError("Image values must be in the range [0, 255]")

        original_height, original_width = self.map_image.shape[:2]

        # Calculate new dimensions (3x smaller)
        new_width = original_width // 3
        new_height = original_height // 3

        # Resize the image
        resized_image = cv2.resize(
            self.map_image, (new_width, new_height), interpolation=cv2.INTER_AREA
        )
        transposed_image = np.transpose(resized_image, (1, 0))  # Swap x and y axes

        cv2.imshow("Map", transposed_image)
        cv2.moveWindow('Map', 2000, 0)  
        cv2.waitKey(1)

    def draw_mark(self, swapped_map_image):
        mark_size = 50
        mark_color = 128

        marked_map = swapped_map_image.__deepcopy__(None)

        marked_map[
            self.map_position[1]
            + int((self.step - mark_size) / 2) : self.map_position[1]
            + int((self.step + mark_size) / 2),
            self.map_position[0]
            + int((self.step - mark_size) / 2) : self.map_position[0]
            + int((self.step + mark_size) / 2),
        ] = (
            np.ones((mark_size, mark_size)) * mark_color
        )

        self.axes_img.set_data(marked_map)
