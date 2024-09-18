import numpy as np
import mujoco as mj
from mujoco.glfw import glfw


class contollerRenderer:
    def __init__(self, window, cam, model, data, opt, scene, context, render_dims):
        self.window = window
        self.cam = cam
        self.model = model
        self.data = data
        self.opt = opt
        self.render_dims = render_dims
        self.scene = scene
        self.context = context

        self._init_main_cam()
        self._init_main_window()

        self._init_feto_cam()
        self._init_feto_window()

    def _init_main_cam(self):
        self.cam.azimuth = 90.0
        self.cam.elevation = -24.0
        self.cam.distance = 2.0
        self.cam.lookat = np.array([0.0, 0.0, 0.0])

    def _init_main_window(self):
        viewport_width, viewport_height = glfw.get_framebuffer_size(self.window)
        self.main_viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

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

    def render(self):
        self.render_main_window()
        self.render_secondary_window()

        glfw.poll_events()

    def render_main_window(self):

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

        mj.mjr_render(self.main_viewport, self.scene, self.context)
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
