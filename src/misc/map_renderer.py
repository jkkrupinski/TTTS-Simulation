import cv2
import numpy as np
from copy import deepcopy


class mapRenderer:

    def __init__(self, render_dims, shape):

        self.render_dims = render_dims
        self.scale = 8

        self.new_width = shape[1] // self.scale
        self.new_height = shape[0] // self.scale

    def render(self, image, grid, image_position):

        self.image = image
        self.grid = grid

        self.render_map(image_position)
        self.render_grid()

    def render_map(self, image_position):

        image = deepcopy(self.image)
        image = self.add_marker(image, image_position)

        resized_image = cv2.resize(
            image, (self.new_width, self.new_height), interpolation=cv2.INTER_AREA
        )

        transposed_image = np.transpose(resized_image, (1, 0))

        cv2.imwrite("Map.png", transposed_image)

        cv2.imshow("Map", transposed_image)
        cv2.moveWindow("Map", 1400, 600)
        cv2.waitKey(1)

    def render_grid(self):
        color_offset = 10
        color_modifier = 4

        resized_grid = cv2.resize(
            (self.grid + color_offset) * color_modifier,
            (self.new_width, self.new_height),
            interpolation=cv2.INTER_AREA,
        )

        transposed_grid = np.transpose(resized_grid, (1, 0))

        cv2.imwrite("Grid.png", transposed_grid)

        cv2.imshow("Grid", transposed_grid)
        cv2.moveWindow("Grid", 1400, 0)
        cv2.waitKey(1)

        # print(self.grid.T, "\n")

    def add_marker(self, image, image_position):

        image = cv2.rectangle(
            image,
            (image_position[1], image_position[0]),
            (
                image_position[1] + self.render_dims,
                image_position[0] + self.render_dims,
            ),
            color=100,
            thickness=6,
        )

        return image

    def reset(self):
        cv2.destroyAllWindows()
