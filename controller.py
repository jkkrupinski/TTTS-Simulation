import time

import mujoco as mj
import numpy as np
import cv2 as cv

import ikpy.chain
from mujoco.glfw import glfw
from mujoco_base import MuJoCoBase


class Controller(MuJoCoBase):
    """
    Class for control of an robotic arm in MuJoCo.
    It can be used on its own, in which case a new model, simulation and viewer will be created.
    It can also be passed these objects when creating an instance, in which case the class can be used
    to perform tasks on an already instantiated simulation.
    """

    def __init__(self):

        xml_path = "scene/main.xml"
        super().__init__(xml_path)

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

        self.feto_cam.type = mj.mjtCamera.mjCAMERA_FIXED
        self.feto_cam.fixedcamid = camera_id

    def _init_feto_window(self):
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        self.second_window = glfw.create_window(
            self.render_dims, self.render_dims, "Fetoscope View", None, None
        )

        glfw.set_window_pos(self.second_window, 1400, 200)

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
        ]
        self.ee_chain = ikpy.chain.Chain.from_urdf_file(
            urdf_path, active_links_mask=active_links_mask
        )

    def _init_robot_info(self):
        self.init_qpos = [0, -0.247, 0, 0.909, 0, 1.15644, 0]
        self.num_of_accuators = len(self.data.ctrl)
        self.current_target_joint_values = np.zeros(self.num_of_accuators)
        self.base_pos = self.model.body("link_base").pos
        self.last_movement_steps = 0

    def reset(self):
        self.data.qpos[:] = self.init_qpos
        self.data.qvel[:] = np.zeros((7,))

        self.last_movement_steps = 0

    def move_joints(
        self,
        target,
        tolerance=0.02,
        max_steps=10000,
        render=True,
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

            for j in range(self.num_of_accuators):
                self.data.ctrl[j] = self.current_target_joint_values[j]

            for i in range(self.num_of_accuators):
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

            if render:
                self.render()

        self.last_movement_steps = steps

        return result

    def move_ee(self, ee_position, render=True):
        """
        Moves the robot arm so that the gripper center ends up at the requested XYZ-position,
        with a vertical gripper position.

        Args:
            ee_position: List of XYZ-coordinates of the end-effector (ee_link for UR5 setup).
        """

        # move marker where the ee should be
        self.model.body("sphere").pos = ee_position - np.array([0, 0, 0.15])

        joint_angles = self.inverse_kinematic(ee_position)
        if joint_angles is not None:
            result = self.move_joints(target=joint_angles, render=render)
        else:
            result = "No valid joint angles received, could not move EE to position."
            self.last_movement_steps = 0

        return result

    def wait_for_ms(self, duration, render=True):
        """
        Holds the current position by actuating the joints towards their current target position.

        Args:
            duration: Time in ms to hold the position.
        """

        starting_time = time.time()
        elapsed = 0
        while elapsed < duration:

            if render:
                self.render()

            elapsed = (time.time() - starting_time) * 1000

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

        # orientation_axis = "Z"
        # target_orientation = [0.0, 0.0, -1]

        orientation_axis = "all"
        target_orientation = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

        ee_position_base = ee_position - self.base_pos

        joint_angles = self.ee_chain.inverse_kinematics(
            ee_position_base,
            target_orientation=target_orientation,
            orientation_mode=orientation_axis,
        )

        # position = self.ee_chain.forward_kinematics(joint_angles)[:3, 3]
        # orientation = self.ee_chain.forward_kinematics(joint_angles)[:3, :3]

        # print(
        #     "Requested position: {} vs Reached position: {}".format(
        #         np.round(ee_position_base, 3), np.round(position, 3)
        #     )
        # )
        # print("Requested orientation on the X axis: {} vs Reached orientation on the X axis: {}".format(target_orientation, np.round(orientation,2)))
        # print()

        prediction = (
            self.ee_chain.forward_kinematics(joint_angles)[:3, 3] + self.base_pos
        )
        # print("pred",np.round(prediction,2))

        diff = abs(prediction - ee_position)
        # print("diff",diff)

        error = np.sqrt(diff.dot(diff))
        # print("error: ",error)

        if error <= 0.3:
            return joint_angles

        print("Failed to find IK solution.")
        return None

    def get_ee_pos(self):
        # ee_chain has an additional fixed joint compared to robot defined usng MuJoCo format
        extended_joints = np.append(0, self.data.qpos)
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


if __name__ == "__main__":

    def do_loop():
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
            contr.move_ee(pos)
            contr.wait_for_ms(1_000)

    contr = Controller()

    pos = [0.0, 0.0, 0.6]
    contr.move_ee(pos)
    contr.save_feto_image(with_mask=False)
    contr.wait_for_ms(5_000)

    do_loop()
