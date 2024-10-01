import time

import mujoco as mj
import numpy as np
import cv2 as cv

from random import randint, uniform

import ikpy.chain
from misc.mujoco_base import MuJoCoBase

from misc.controller_renderer import contollerRenderer


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

        self.placenta_height = self.model.body("placenta_seg").pos[2]

        self._init_renderer()
        self._init_kinematic_chain()
        self._init_robot_info()

    def _init_renderer(self):
        self.render_dims = 256
        self.renderer = contollerRenderer(
            self.window,
            self.cam,
            self.model,
            self.data,
            self.opt,
            self.scene,
            self.context,
            self.render_dims,
        )

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

        self.feto_fovy = self.model.camera("eye").fovy

        self.theta_x = 0
        self.theta_y = 0

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
        starting_time = time.time()
        elapsed = 0
        while elapsed < duration:

            if self.render_mode == "human":
                self.render()

            elapsed = (time.time() - starting_time) * 1000

    def tilt_tool_orientation(self, target_orientation, theta_x, theta_y):

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

        new_orientation = target_orientation @ R_y @ R_x

        return new_orientation

    def calculate_tool_rotation(self, movement_vector):
        height = self.rcm_height - self.ee_height
        d_theta_x = np.arctan(movement_vector[0] / height)
        d_theta_y = np.arctan(movement_vector[1] / height)

        return d_theta_x, d_theta_y

    def inverse_kinematic(self, ee_position):
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

    def get_ee_rotation(self):
        return self.theta_x, self.theta_y

    def get_feto_image(self):
        dimensions = 3
        pixels_buffer = np.zeros(
            (self.render_dims * self.render_dims * dimensions, 1), dtype=np.uint8
        )
        mj.mjr_readPixels(
            pixels_buffer,
            None,
            self.renderer.feto_viewport,
            self.renderer.context_secondary,
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

        x_offset = uniform(-0.02, 0.02)
        y_offset = uniform(-0.02, 0.02)

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
        self.renderer.render()
