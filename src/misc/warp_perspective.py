import cv2

import numpy as np


def warp_perspective(image, fovy, theta_x, theta_y):

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

    focal_length = 256 / (2 * np.tan(fovy / 2))
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
