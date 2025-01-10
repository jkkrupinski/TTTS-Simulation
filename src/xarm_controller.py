import numpy as np
import time
import cv2 as cv

from xarm.wrapper import XArmAPI


class XARMController:
    def __init__(self, ee_height=0.14, rcm_height=0.28, rcm_mode=True):
        self._init_camera()
        self._initialize_robot()
        self._init_attributes(ee_height, rcm_height, rcm_mode)

        self.yaw_offset = -12

        # self.move_to_start_position()
        self.arm.set_position(x=350, y=0, z=(rcm_height)*1000, roll=180, pitch=0, yaw=self.yaw_offset, speed=50, wait=True)
        self.arm.set_position(x=350, y=0, z=ee_height*1000, roll=180, pitch=0, yaw=self.yaw_offset, speed=50, wait=True)




    def _initialize_robot(self):

        self.arm = XArmAPI("192.168.1.238")
        self.arm.connect()

        self.arm.set_simulation_robot(False)

        if not self.arm.connected:
            raise RuntimeError("Failed to connect to the xArm API.")

        try:
            self.arm.clean_error()
            self.arm.clean_warn()
            self.arm.motion_enable(enable=True)
            self.arm.set_mode(0)  # Position control mode
            self.arm.set_state(0)  # Ready state

        except Exception as e:
            print(f"Failed to initialize the robot: {e}")

    def _init_attributes(self, ee_height, rcm_height, rcm_mode):

        self.ee_height = ee_height
        self.rcm_height = rcm_height
        self.rcm_mode = rcm_mode

        self.render_dims = 256
        self.feto_fovy = 10.0
        self.placenta_height = 0.01

        # self.base_position = np.array([0, 0, 0])  # TO ADJUST
        # self.base_rotation = np.array([0, 0, 0])  # TO ADJUST

        self.theta_x = 0
        self.theta_y = 0

    def _init_camera(self):
        self.camera = cv.VideoCapture(0)

        if not self.camera.isOpened():
            print("Error: Could not access the camera.")

        self.camera.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
        self.camera.set(cv.CAP_PROP_FRAME_WIDTH, 620)

    def reset(self):
        pass
        # self.arm.reset(wait=True)
        # self.move_to_start_position()

    def move_ee(self, ee_position, movement_vector):
        if self.rcm_mode:
            d_theta_x, d_theta_y = self.calculate_tool_rotation(movement_vector)
            self.theta_x += d_theta_x
            self.theta_y += d_theta_y

        ee_target_position = np.array(ee_position)
        ee_target_orientation = [
            self.theta_x,
            self.theta_y,
            0,
        ]

        print(
            "Target pos    ",
            np.round(ee_target_position[0] * 1000, 2),
            np.round(ee_target_position[1] * 1000, 2),
            np.round(ee_target_position[2] * 1000, 2),
        )
        print(
            "Target orient      ",
            (ee_target_orientation[0] - 3.14) * 180 / 3.14,
            ee_target_orientation[1] * 180 / 3.14,
            ee_target_orientation[2] * 180 / 3.14,
        )

        self.get_feto_image()
        code = self.arm.set_position(
            ee_target_position[0] * 1000,
            ee_target_position[1] * 1000,
            ee_target_position[2] * 1000,
            (ee_target_orientation[0] - 3.14) * 180 / 3.14,
            ee_target_orientation[1] * 180 / 3.14,
            self.yaw_offset,
            # -180,
            # 0,
            # 0,
            speed=50,
            wait=True,
            is_radian=False,
        )

        print("done", code)
        if code == 9:
            exit()

    def move_to_start_position(self):
        self.theta_x = 0
        self.theta_y = 0

        # TO ADJUST
        positions = [
            [-0.00893436 * 1000, -0.00017784 * 1000, self.rcm_height],
            [
                -0.00893436 * 1000,
                -0.00017784 * 1000,
                (self.ee_height + self.rcm_height) / 2,
            ],
            [-0.00893436 * 1000, -0.00017784 * 1000, self.ee_height],
        ]

        for pos in positions:
            self.get_feto_image()
            self.move_ee(self.get_ee_pos(), pos)

    def wait_for_ms(self, duration):
        time.sleep(duration / 1000.0)

    def calculate_tool_rotation(self, movement_vector):
        height = self.rcm_height - self.ee_height
        d_theta_x = np.arctan2(movement_vector[0], height)
        d_theta_y = np.arctan2(movement_vector[1], height)
        return d_theta_x, d_theta_y

    def get_ee_pos(self):
        ee_position = self.arm.get_position()[1][0:3]

        ee_position[0] /= 1000
        ee_position[1] /= 1000
        ee_position[2] /= 1000

        return ee_position

    def get_ee_rotation(self):
        ee_rotation = self.arm.get_position(is_radian=True)[1][3:5]

        ee_rotation[0] -= 3.14
        return ee_rotation
        # return self.theta_x, self.theta_y

    def get_feto_image(self):

        ret, frame = self.camera.read()

        if not ret:
            print("Error: Could not capture an image.")

        height, width, _ = frame.shape

        # # Calculate the center crop region for 480x480
        # x_center = width // 2
        # y_center = height // 2
        # x_start = x_center - 240
        # y_start = y_center - 240
        # x_end = x_center + 240
        # y_end = y_center + 240

        # # Crop the frame
        # cropped_frame = frame[y_start:y_end, x_start:x_end]

        # frame = cv.resize(cropped_frame, (self.render_dims, self.render_dims))

        height, width, _ = frame.shape

        fov_ratio = 0.4 #0.145

        new_width = int(width * fov_ratio)
        new_height = int(height * fov_ratio)

        # Calculate the center crop coordinates
        x_center = width // 2
        y_center = height // 2
        x_start = x_center - new_width // 2
        y_start = y_center - new_height // 2
        x_end = x_center + new_width // 2
        y_end = y_center + new_height // 2
        # Crop the frame
        cropped_frame = frame[y_start:y_end, x_start+12:x_end-12]

        frame = cv.resize(cropped_frame, (256, 256)) 
        frame = cv.rotate(frame, 0)


        cv.imshow("Fetoscope", frame)

        frame = cv.rotate(frame, 0)
        frame = cv.rotate(frame, 0)

        frame = cv.rotate(frame, 0)

        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

        return frame

    def __del__(self):
        if self.camera.isOpened():
            self.camera.release()

        cv.destroyAllWindows()

        if self.arm.connected:
            self.arm.disconnect()
