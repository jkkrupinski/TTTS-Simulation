import mujoco as mj
import time
import numpy as np
import ikpy.chain
import cv2 as cv
import matplotlib.pyplot as plt
import copy
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

        self.render_dims = 256

        self._init_main_cam()
        self._init_feto_cam()
        self._init_second_window()

        self.num_of_accuators = len(self.data.ctrl)

        self._init_kinematic_chain()

        self.current_target_joint_values = np.zeros(self.num_of_accuators)
        self.image_counter = 0

        self.last_movement_steps = 0

    def _init_second_window(self):
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

    def _init_main_cam(self):
        self.cam.azimuth = 90.0
        self.cam.elevation = -24.0
        self.cam.distance = 2.0
        self.cam.lookat = np.array([0.0, 0.0, 0.0])

    def _init_feto_cam(self):
        self.feto_cam = mj.MjvCamera()
        self.feto_viewport = mj.MjrRect(0, 0, self.render_dims, self.render_dims)
        self.context_secondary = mj.MjrContext(
            self.model, mj.mjtFontScale.mjFONTSCALE_150
        )

        camera_name = "eye"
        camera_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_CAMERA, camera_name)

        self.feto_cam.type = mj.mjtCamera.mjCAMERA_FIXED
        self.feto_cam.fixedcamid = camera_id

    def _init_kinematic_chain(self):
        urdf_path = "scene/ufactory_xarm7/xarm.urdf"
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

    def save_feto_image(self, with_mask, frame_counter=1):

        path = "imgs/img_" + str(frame_counter) + ".png"
        gray_array = self.get_feto_image()

        if with_mask:
            mask = cv.imread("mask/mask.png")
            mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
            gray_array = cv.bitwise_and(gray_array, gray_array, mask=mask)

        cv.imwrite(path, gray_array)

    def get_feto_image(self):
        pixels_buffer = np.zeros(
            (self.render_dims * self.render_dims * 3, 1), dtype=np.uint8
        )
        mj.mjr_readPixels(
            pixels_buffer, None, self.feto_viewport, self.context_secondary
        )

        reshaped_array = pixels_buffer.reshape(self.render_dims, self.render_dims, 3)
        gray_array = cv.cvtColor(reshaped_array, cv.COLOR_RGB2GRAY)
        return gray_array

    def get_ee_pos(self):
        extended_joints = np.append(0, self.data.qpos)
        return (
            self.ee_chain.forward_kinematics(extended_joints)[:3, 3]
            + self.model.body("link_base").pos
        )

    def move_joints(
        self,
        target,
        tolerance=0.05,
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
        for i in range(self.num_of_accuators):
            self.current_target_joint_values[i] = target[i + 1]

        while not self.reached_target:
            current_joint_values = self.data.qpos

            # self.get_image_data(width=200, height=200, show=True)

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

    def move_ee(self, ee_position):
        """
        Moves the robot arm so that the gripper center ends up at the requested XYZ-position,
        with a vertical gripper position.

        Args:
            ee_position: List of XYZ-coordinates of the end-effector (ee_link for UR5 setup).
        """

        self.model.body("sphere").pos = ee_position

        joint_angles = self.ik(ee_position)
        if joint_angles is not None:
            result = self.move_joints(target=joint_angles)
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

    def ik(self, ee_position):
        """
        Method for solving simple inverse kinematic problems.
        This was developed for top down graspig, therefore the solution will be one where the gripper is
        vertical. This might need adjustment for other gripper models.

        Args:
            ee_position: List of XYZ-coordinates of the end-effector (ee_link for UR5 setup).

        Returns:
            joint_angles: List of joint angles that will achieve the desired ee position.
        """

        orientation_axis = "Z"
        target_orientation = [0, 0, -1]

        ee_position_base = ee_position - self.model.body("link_base").pos

        joint_angles = self.ee_chain.inverse_kinematics(
            ee_position_base,
            target_orientation=target_orientation,
            orientation_mode=orientation_axis,
        )

        prediction = (
            self.ee_chain.forward_kinematics(joint_angles)[:3, 3]
            + self.model.body("link_base").pos
        )

        diff = abs(prediction - ee_position)
        error = np.sqrt(diff.dot(diff))

        if error <= 0.4:
            return joint_angles

        print("Failed to find IK solution.")
        return None

    def render(self):
        self.render_main()
        self.render_secondary()
        glfw.poll_events()

    def render_main(self):
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

    def render_secondary(self):
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

    ####################################################################################

    def show_model_info(self):
        """
        Displays relevant model info for the user, namely bodies, joints, actuators, as well as their IDs and ranges.
        Also gives info on which actuators control which joints and which joints are included in the kinematic chain,
        as well as the PID controller info for each actuator.
        """

        print("\nNumber of bodies: {}".format(self.model.nbody))
        for i in range(self.model.nbody):
            print("Body ID: {}, Body Name: {}".format(i, self.model.body_id2name(i)))

        print("\nNumber of joints: {}".format(self.model.njnt))
        for i in range(self.model.njnt):
            print(
                "Joint ID: {}, Joint Name: {}, Limits: {}".format(
                    i, self.model.joint_id2name(i), self.model.jnt_range[i]
                )
            )

        print("\nNumber of Actuators: {}".format(self.num_of_accuators))
        for i in range(self.num_of_accuators):
            print(
                "Actuator ID: {}, Actuator Name: {}, Controlled Joint: {}, Control Range: {}".format(
                    i,
                    self.model.actuator_id2name(i),
                    self.actuators[i][3],
                    self.model.actuator_ctrlrange[i],
                )
            )

        print(
            "\nJoints in kinematic chain: {}".format(
                [i.name for i in self.ee_chain.links]
            )
        )

        print("\nPID Info: \n")
        for i in range(len(self.actuators)):
            print(
                "{}: P: {}, I: {}, D: {}, setpoint: {}, sample_time: {}".format(
                    self.actuators[i][3],
                    self.actuators[i][4].tunings[0],
                    self.actuators[i][4].tunings[1],
                    self.actuators[i][4].tunings[2],
                    self.actuators[i][4].setpoint,
                    self.actuators[i][4].sample_time,
                )
            )

        print("\n Camera Info: \n")
        for i in range(self.model.ncam):
            print(
                "Camera ID: {}, Camera Name: {}, Camera FOV (y, degrees): {}, Position: {}, Orientation: {}".format(
                    i,
                    self.model.camera_id2name(i),
                    self.model.cam_fovy[i],
                    self.model.cam_pos0[i],
                    self.model.cam_mat0[i],
                )
            )

    def display_current_values(self):
        """
        Debug method, simply displays some relevant data at the time of the call.
        """

        print("\n################################################")
        print("CURRENT JOINT POSITIONS (ACTUATED)")
        print("################################################")
        for i in range(len(self.actuated_joint_ids)):
            print(
                "Current angle for joint {}: {}".format(
                    self.actuators[i][3], self.sim.data.qpos[self.actuated_joint_ids][i]
                )
            )

        print("\n################################################")
        print("CURRENT JOINT POSITIONS (ALL)")
        print("################################################")
        for i in range(len(self.model.jnt_qposadr)):
            # for i in range(self.model.njnt):
            name = self.model.joint_id2name(i)
            print(
                "Current angle for joint {}: {}".format(
                    name, self.sim.data.get_joint_qpos(name)
                )
            )
            # print('Current angle for joint {}: {}'.format(self.model.joint_id2name(i), self.sim.data.qpos[i]))

        print("\n################################################")
        print("CURRENT BODY POSITIONS")
        print("################################################")
        for i in range(self.model.nbody):
            print(
                "Current position for body {}: {}".format(
                    self.model.body_id2name(i), self.sim.data.body_xpos[i]
                )
            )

        print("\n################################################")
        print("CURRENT BODY ROTATION MATRIZES")
        print("################################################")
        for i in range(self.model.nbody):
            print(
                "Current rotation for body {}: {}".format(
                    self.model.body_id2name(i), self.sim.data.body_xmat[i]
                )
            )

        print("\n################################################")
        print("CURRENT BODY ROTATION QUATERNIONS (w,x,y,z)")
        print("################################################")
        for i in range(self.model.nbody):
            print(
                "Current rotation for body {}: {}".format(
                    self.model.body_id2name(i), self.sim.data.body_xquat[i]
                )
            )

        print("\n################################################")
        print("CURRENT ACTUATOR CONTROLS")
        print("################################################")
        for i in range(self.num_of_accuators):
            print(
                "Current activation of actuator {}: {}".format(
                    self.actuators[i][1], self.sim.data.ctrl[i]
                )
            )

    def get_image_data(self, show=False, camera="top_down", width=200, height=200):
        """
        Returns the RGB and depth images of the provided camera.

        Args:
            show: If True displays the images for five seconds or until a key is pressed.
            camera: String specifying the name of the camera to use.
        """

        rgb, depth = copy.deepcopy(
            self.sim.render(width=width, height=height, camera_name=camera, depth=True)
        )
        if show:
            cv.imshow("rbg", cv.cvtColor(rgb, cv.COLOR_BGR2RGB))
            # cv.imshow('depth', depth)
            cv.waitKey(1)
            # cv.waitKey(delay=5000)
            # cv.destroyAllWindows()

        return np.array(np.fliplr(np.flipud(rgb))), np.array(
            np.fliplr(np.flipud(depth))
        )

    def create_camera_data(self, width, height, camera):
        """
        Initializes all camera parameters that only need to be calculated once.
        """

        cam_id = self.model.camera_name2id(camera)
        # Get field of view
        fovy = self.model.cam_fovy[cam_id]
        # Calculate focal length
        f = 0.5 * height / np.tan(fovy * np.pi / 360)
        # Construct camera matrix
        self.cam_matrix = np.array(((f, 0, width / 2), (0, f, height / 2), (0, 0, 1)))
        # Rotation of camera in world coordinates
        self.cam_rot_mat = self.model.cam_mat0[cam_id]
        self.cam_rot_mat = np.reshape(self.cam_rot_mat, (3, 3))
        # Position of camera in world coordinates
        self.cam_pos = self.model.cam_pos0[cam_id]
        self.cam_init = True

    def world_2_pixel(self, world_coordinate, width=200, height=200, camera="top_down"):
        """
        Takes a XYZ world position and transforms it into pixel coordinates.
        Mainly implemented for testing the correctness of the camera matrix, focal length etc.

        Args:
            world_coordinate: XYZ world coordinate to be transformed into pixel space.
            width: Width of the image (pixel).
            height: Height of the image (pixel).
            camera: Name of camera used to obtain the image.
        """

        if not self.cam_init:
            self.create_camera_data(width, height, camera)

        # Homogeneous image point
        hom_pixel = (
            self.cam_matrix @ self.cam_rot_mat @ (world_coordinate - self.cam_pos)
        )
        # Real image point
        pixel = hom_pixel[:2] / hom_pixel[2]

        return np.round(pixel[0]).astype(int), np.round(pixel[1]).astype(int)

    def pixel_2_world(
        self, pixel_x, pixel_y, depth, width=200, height=200, camera="top_down"
    ):
        """
        Converts pixel coordinates into world coordinates.

        Args:
            pixel_x: X-coordinate in pixel space.
            pixel_y: Y-coordinate in pixel space.
            depth: Depth value corresponding to the pixel.
            width: Width of the image (pixel).
            height: Height of the image (pixel).
            camera: Name of camera used to obtain the image.
        """

        if not self.cam_init:
            self.create_camera_data(width, height, camera)

        # Create coordinate vector
        pixel_coord = np.array([pixel_x, pixel_y, 1]) * (-depth)
        # Get position relative to cameraa
        pos_c = np.linalg.inv(self.cam_matrix) @ pixel_coord
        # Get world position
        pos_w = np.linalg.inv(self.cam_rot_mat) @ (pos_c + self.cam_pos)

        return pos_w


def loop():
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
        # contr.wait_for_ms(1_000)


if __name__ == "__main__":
    contr = Controller()

    pos = [0.0, 0.0, 0.4]
    contr.move_ee(pos)
    contr.save_feto_image(with_mask=False)
    contr.wait_for_ms(1_000)

    loop()

