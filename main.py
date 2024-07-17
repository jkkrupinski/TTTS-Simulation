import cv2
import mujoco as mj
import numpy as np
from mujoco.glfw import glfw
from mujoco_base import MuJoCoBase


class Fetoscope(MuJoCoBase):
    def __init__(self, xml_path):
        super().__init__(xml_path)
        self.sim_end = 100.
        self.render_dims = 256
        self.save_imgs = False
        self.feto_cam = mj.MjvCamera() 



    def reset(self):
        # Set camera configuration
        self.cam.azimuth = 90.0
        self.cam.elevation = -24.0
        self.cam.distance = 2.0
        self.cam.lookat = np.array([0.0, 0.0, 0.0])

        # Set controller
        mj.set_mjcb_control(self.controller)


    def controller(self, model, data):
        # Set angles for x and y axis
        self.data.ctrl[0] = 0.5
        self.data.ctrl[1] = -0.2
        # pass


    def renderMainScreen(self, viewport_width, viewport_height):

        viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

        mj.mjv_updateScene(self.model, self.data, self.opt, None, self.cam,
                            mj.mjtCatBit.mjCAT_ALL.value, self.scene)
        mj.mjr_render(viewport, self.scene, self.context)


    def renderSecondaryScreen(self, viewport_width, viewport_height, frame_counter):
        # Define viewport recangle position and size     
        pos_x = viewport_width-self.render_dims
        pos_y = viewport_height-self.render_dims
        width = self.render_dims
        height = self.render_dims

        feto_viewport = mj.MjrRect(pos_x,pos_y,width ,height )
    
        # Define fetoscope camera
        camera_name = 'eye'
        camera_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_CAMERA, camera_name)

        self.feto_cam.type = mj.mjtCamera.mjCAMERA_FIXED 
        self.feto_cam.fixedcamid = camera_id

        mj.mjv_updateScene(self.model, self.data, self.opt, None, self.feto_cam,
                            mj.mjtCatBit.mjCAT_ALL.value, self.scene)
        
        # Placeholder for pixel data
        pixels = np.zeros((height * width * 3, 1), dtype=np.uint8)  

        mj.mjr_render(feto_viewport, self.scene, self.context)

        mj.mjr_readPixels(pixels, None, feto_viewport, self.context)

        if self.save_imgs:
            path = "imgs/img_" + str(frame_counter) + ".png"
            reshaped_array = pixels.reshape(self.render_dims,self.render_dims,3)
            bgr_array = cv2.cvtColor(reshaped_array, cv2.COLOR_RGB2BGR)

            mask = cv2.imread("mask/mask.png")
            mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
            masked_img = cv2.bitwise_and(bgr_array, bgr_array, mask=mask)
            
            cv2.imwrite(path, masked_img)

        mj.mjr_drawPixels(pixels, None, feto_viewport, self.context)


    def simulate(self):

        # Set initial angles [0]-> x, [1]-> y
        # self.data.qpos[0] = 0.1
        # self.data.qpos[1] = 0.1

        frame_counter = 0
        while not glfw.window_should_close(self.window):

            frame_counter+=1
            sim_start = self.data.time

            while (self.data.time - sim_start < 1.0/60.0):
                # Step simulation environment
                mj.mj_step(self.model, self.data)

            if self.data.time >= self.sim_end:
                break

            # Get framebuffer viewport
            viewport_width, viewport_height = glfw.get_framebuffer_size(
                self.window)
           
            # Update scene and render
            self.renderMainScreen(viewport_width, viewport_height)
            self.renderSecondaryScreen(viewport_width, viewport_height, frame_counter)

            # Swap OpenGL buffers (blocking call due to v-sync)
            glfw.swap_buffers(self.window)

            # Process pending GUI events, call GLFW callbacks
            glfw.poll_events()

        glfw.terminate()


def main():
    xml_path = "scene/main.xml"
    sim = Fetoscope(xml_path)
    sim.reset()
    sim.simulate()


if __name__ == "__main__":
    main()


# get fov for fetoscope
# get distance from placenta
# check fetoscope dimensions
  
# augment image from fetoscope
# create contoller for fetoscope
# try to move fetoscope in plane XY

# input segmented placenta
# implement pix2pix or stable diffusion
