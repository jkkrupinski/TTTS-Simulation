import time

import mujoco as mj
import numpy as np
import cv2 as cv

from random import randint, uniform

import ikpy.chain
from mujoco.glfw import glfw
from misc.mujoco_base import MuJoCoBase


class Controller(MuJoCoBase):
    """
    Class for control of an robotic arm in MuJoCo.
    It can be used on its own, in which case a new model, simulation and viewer will be created.
    It can also be passed these objects when creating an instance, in which case the class can be used
    to perform tasks on an already instantiated simulation.
    """

    def __init__(self, render_mode, ee_height=0.14, rcm_height=0.28, rcm_mode=True):

        xml_path = "scene/main.xml"
        super().__init__(xml_path)

        self.render_mode = render_mode
        self.rcm_mode = rcm_mode

        self.rcm_height = rcm_height
        self.ee_height = ee_height

        self.theta_x = 0
        self.theta_y = 0

        self._init_main_cam()
        self._init_feto_cam()
        self._init_feto_window()

        self._init_kinematic_chain()
        self._init_robot_info()

    def _init_main_cam(self):
        self.cam.azimuth = 90.0
        self.cam.elevation = -24.0
        self.cam.distance = 2.0
        self.cam.lookat = np.array([0.0, 0.0, 0.0])

    def _init_feto_cam(self):
        self.feto_cam = mj.MjvCamera()
        self.render_dims = 256

        self.feto_viewport = mj.MjrRect(0, 0, self.render_dims, self.render_dims)
        self.context_secondary = mj.MjrContext(
            self.model, mj.mjtFontScale.mjFONTSCALE_150
        )

        camera_name = "eye"
        camera_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_CAMERA, camera_name)

        self.feto_fovy = self.model.camera("eye").fovy

        self.feto_cam.type = mj.mjtCamera.mjCAMERA_FIXED
        self.feto_cam.fixedcamid = camera_id

    def _init_feto_window(self):
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        self.second_window = glfw.create_window(
            self.render_dims, self.render_dims, "Fetoscope View", None, None
        )

        glfw.set_window_pos(self.second_window, 20, 200)

        if not self.second_window:
            glfw.terminate()
            raise Exception("Second GLFW window could not be created")

        glfw.make_context_current(self.second_window)
        glfw.show_window(self.second_window)

    def _init_kinematic_chain(self):
        urdf_path = "scene/ufactory_xarm7/xarm7.urdf"
        active_links_mask = [
            False,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            False,
        ]
        self.ee_chain = ikpy.chain.Chain.from_urdf_file(
            urdf_path, active_links_mask=active_links_mask
        )

    def _init_robot_info(self):
        self.init_qpos = [0, -0.247, 0, 0.909, 0, 1.15644, 0]  # values from urdf file
        self.num_of_accuators = len(self.data.ctrl)
        self.current_target_joint_values = np.zeros(self.num_of_accuators)
        self.base_pos = self.model.body("link_base").pos
        self.last_movement_steps = 0

    def reset(self):

        self.data.qpos[:] = self.init_qpos
        self.data.qvel[:] = np.zeros((7,))

        self.offset_placenta_position()

        self.move_to_start_position()

        self.last_movement_steps = 0

    def move_joints(
        self,
        target,
        tolerance=0.012,
        max_steps=1000,
    ):
        """
        Moves the specified joint group to a joint target.

        Args:
            group: String specifying the group to move.
            target: List of target joint values for the group.
            tolerance: Threshold within which the error of each joint must be before the method finishes.
            max_steps: maximum number of steps to actuate before breaking
        """

        steps = 1
        result = ""

        self.reached_target = False
        deltas = np.zeros(self.num_of_accuators)

        # Update self target joint values
        # ee_chain has an additional fixed joint compared to robot defined usng MuJoCo format
        for i in range(self.num_of_accuators):
            self.current_target_joint_values[i] = target[i + 1]

        while not self.reached_target:
            current_joint_values = self.data.qpos

            for i in range(self.num_of_accuators):
                self.data.ctrl[i] = self.current_target_joint_values[i]

                deltas[i] = abs(
                    self.current_target_joint_values[i] - current_joint_values[i]
                )

            if max(deltas) < tolerance:
                result = "success"
                self.reached_target = True
                break

            if steps > max_steps:
                result = "max. steps reached: {}".format(max_steps)
                break

            mj.mj_step(self.model, self.data)
            steps += 1

            if self.render_mode == "human":
                self.render()

        self.last_movement_steps = steps

        return result

    def move_ee(self, ee_position, movement_vector):
        """
        Moves the robot arm so that the gripper center ends up at the requested XYZ-position,
        with a vertical gripper position.

        Args:
            ee_position: List of XYZ-coordinates of the end-effector (ee_link for UR5 setup).
        """

        if self.rcm_mode:
            d_theta_x, d_theta_y = self.calculate_tool_rotation(movement_vector)

            self.theta_x += d_theta_x  # Tilt by d_theta_x radians in x-axis
            self.theta_y += d_theta_y  # Tilt by d_theta_y radians in y-axis

        # move marker where the ee should be
        self.model.body("ee_marker").pos = ee_position

        joint_angles = self.inverse_kinematic(ee_position)
        if joint_angles is not None:
            result = self.move_joints(target=joint_angles)
        else:
            result = "No valid joint angles received, could not move EE to position."
            self.last_movement_steps = 0

        return result

    def move_to_start_position(self):

        self.theta_x = 0
        self.theta_y = 0

        self.move_ee([0, 0, self.rcm_height], np.zeros(3))
        self.move_ee([0, 0, self.ee_height], np.zeros(3))

    def wait_for_ms(self, duration):
        """
        Holds the current position by actuating the joints towards their current target position.

        Args:
            duration: Time in ms to hold the position.
        """

        starting_time = time.time()
        elapsed = 0
        while elapsed < duration:

            if self.render_mode == "human":
                self.render()

            elapsed = (time.time() - starting_time) * 1000

    def tilt_tool_orientation(self, target_orientation, theta_x, theta_y):
        """
        Returns the new orientation matrix after tilting the tool in x and y axes.

        Parameters:
        - target_orientation: The original 3x3 orientation matrix
        - theta_x: Rotation angle around the x-axis in radians
        - theta_y: Rotation angle around the y-axis in radians

        Returns:
        - new_orientation: The new orientation matrix
        """
        # Rotation matrix around x-axis
        R_x = np.array(
            [
                [1, 0, 0],
                [0, np.cos(theta_x), -np.sin(theta_x)],
                [0, np.sin(theta_x), np.cos(theta_x)],
            ]
        )

        # Rotation matrix around y-axis
        R_y = np.array(
            [
                [np.cos(theta_y), 0, np.sin(theta_y)],
                [0, 1, 0],
                [-np.sin(theta_y), 0, np.cos(theta_y)],
            ]
        )

        # Apply rotations to the target orientation
        new_orientation = target_orientation @ R_y @ R_x

        return new_orientation

    def calculate_tool_rotation(self, movement_vector):
        height = self.rcm_height - self.ee_height
        d_theta_x = np.arctan(movement_vector[0] / height)
        d_theta_y = np.arctan(movement_vector[1] / height)

        return d_theta_x, d_theta_y

    def inverse_kinematic(self, ee_position):
        """
        Method for solving simple inverse kinematic problems.
        This was developed for top down grasping, therefore the solution will be one where the gripper is
        vertical. This might need adjustment for other gripper models.

        Args:
            ee_position: List of XYZ-coordinates of the end-effector (ee_link for UR5 setup).

        Returns:
            joint_angles: List of joint angles that will achieve the desired ee position.
        """

        orientation_axis = "all"
        target_orientation = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

        if self.rcm_mode:
            target_orientation = self.tilt_tool_orientation(
                target_orientation, self.theta_y, self.theta_x
            )

        ee_position_base = ee_position - self.base_pos

        joint_angles = self.ee_chain.inverse_kinematics(
            ee_position_base,
            target_orientation=target_orientation,
            orientation_mode=orientation_axis,
        )

        prediction = (
            self.ee_chain.forward_kinematics(joint_angles)[:3, 3] + self.base_pos
        )

        diff = abs(prediction - ee_position)
        error = np.sqrt(diff.dot(diff))

        if error <= 0.3:
            return joint_angles

        print("Failed to find IK solution.")
        return None

    def get_ee_pos(self):
        # ee_chain has an additional fixed joint compared to robot defined usng MuJoCo format
        extended_joints = np.append(0, self.data.qpos)
        extended_joints = np.append(extended_joints, 0)

        ee_pos = (
            self.ee_chain.forward_kinematics(extended_joints)[:3, 3] + self.base_pos
        )

        return ee_pos

    def get_feto_image(self):
        dimensions = 3
        pixels_buffer = np.zeros(
            (self.render_dims * self.render_dims * dimensions, 1), dtype=np.uint8
        )
        mj.mjr_readPixels(
            pixels_buffer, None, self.feto_viewport, self.context_secondary
        )

        reshaped_array = pixels_buffer.reshape(
            self.render_dims, self.render_dims, dimensions
        )
        rotated_array = cv.rotate(reshaped_array, 0)
        gray_array = cv.cvtColor(rotated_array, cv.COLOR_RGB2GRAY)
        return gray_array

    def save_feto_image(self, with_mask, frame_counter=1):
        path = "imgs/img_" + str(frame_counter) + ".png"
        gray_array = self.get_feto_image()
        if with_mask:
            mask = cv.imread("mask/mask.png")
            mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
            gray_array = cv.bitwise_and(gray_array, gray_array, mask=mask)
        cv.imwrite(path, gray_array)

    def offset_placenta_position(self):

        x_offset = uniform(-0.05, 0.05)
        y_offset = uniform(-0.05, 0.05)

        self.model.body("placenta_seg").pos = [x_offset, y_offset, 0.01]

        # theta = uniform(-1.5708, 1.5708)

        # Rz = np.array(
        #     [
        #         [np.cos(theta), -np.sin(theta), 0],
        #         [np.sin(theta), np.cos(theta), 0],
        #         [0, 0, 1],
        #     ]
        # )

        # Rz_flattened = Rz.flatten()

        # self.data.body("placenta_seg").xmat = Rz_flattened

    def render(self):
        self.render_main_window()
        self.render_secondary_window()
        glfw.poll_events()

    def render_main_window(self):
        viewport_width, viewport_height = glfw.get_framebuffer_size(self.window)
        main_viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)
        glfw.make_context_current(self.window)

        mj.mjv_updateScene(
            self.model,
            self.data,
            self.opt,
            None,
            self.cam,
            mj.mjtCatBit.mjCAT_ALL.value,
            self.scene,
        )

        mj.mjr_render(main_viewport, self.scene, self.context)
        glfw.swap_buffers(self.window)

    def render_secondary_window(self):
        glfw.make_context_current(self.second_window)
        self.context_secondary = mj.MjrContext(
            self.model, mj.mjtFontScale.mjFONTSCALE_150
        )

        mj.mjv_updateScene(
            self.model,
            self.data,
            self.opt,
            None,
            self.feto_cam,
            mj.mjtCatBit.mjCAT_ALL.value,
            self.scene,
        )

        mj.mjr_render(self.feto_viewport, self.scene, self.context_secondary)
        glfw.swap_buffers(self.second_window)
