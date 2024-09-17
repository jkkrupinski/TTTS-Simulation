from controller import Controller

import numpy as np


def do_loop(contr):
    # Define the radius and the center of the circle
    radius = 0.05
    center_x = 0
    center_y = 0

    # Number of points to generate
    num_points = 20

    # Initialize lists to hold the x and y coordinates
    x = []
    y = []

    # Loop through the angles to generate the circle's coordinates
    for i in range(num_points):
        theta = 2 * np.pi * i / num_points
        x = center_x + radius * np.cos(theta)
        y = center_y + radius * np.sin(theta)
        pos = [x, y, 0.4]
        contr.move_ee(pos, [0, 0, 0])
        contr.wait_for_ms(1_000)


def main():

    contr = Controller(render_mode="human")

    pos = [0.0, 0.0, 0.6]
    contr.move_ee(pos, [0, 0, 0])
    contr.save_feto_image(with_mask=False)
    contr.wait_for_ms(5_000)

    do_loop()


if __name__ == "__main__":
    main()
