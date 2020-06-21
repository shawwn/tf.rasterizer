"""Create a window for rendering."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glfw
import numpy as np
import OpenGL.GL as gl

import time


class Window(object):
    """Create a window for rendering."""

    def __init__(self, width, height, title, upscale=1):
        self.width = width
        self.height = height
        self.upscale = upscale

        if not glfw.init():
            return

        self.win = glfw.create_window(width*upscale, height*upscale, title, None, None)
        if not self.win:
            glfw.terminate()
            return

        zeros = np.zeros([height, width, 3], dtype=np.uint8)

        glfw.make_context_current(self.win)

        self.tid = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tid)
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, width, height, 0,
                        gl.GL_RGB, gl.GL_UNSIGNED_BYTE, zeros)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER,
                           gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER,
                           gl.GL_NEAREST)

    def loop(self, update_fn):
        """Loop."""
        i = 0
        self.last_time = time.time()
        
        while not glfw.window_should_close(self.win):
            image = update_fn()

            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

            if True:
              gl.glEnable(gl.GL_TEXTURE_2D)
              gl.glBindTexture(gl.GL_TEXTURE_2D, self.tid)
              #gl.glTexSubImage2D(
              #    gl.GL_TEXTURE_2D, 0, 0, 0, self.width, self.height, gl.GL_RGB,
              #    gl.GL_UNSIGNED_BYTE, image)
              gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
              gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, self.width, self.height, 0,
                              gl.GL_RGB, gl.GL_UNSIGNED_BYTE, image)

              halfw = -0.5 / self.width
              halfh = -0.5 / self.height

              gl.glBegin(gl.GL_QUADS)

              gl.glTexCoord2f(-halfw+0.0, -halfh+0.0)
              gl.glVertex3f(-1.0, 1.0, 0.0)

              gl.glTexCoord2f(-halfw+1.0, -halfh+0.0)
              gl.glVertex3f(1.0, 1.0, 0.0)

              gl.glTexCoord2f(-halfw+1.0, -halfh+1.0)
              gl.glVertex3f(1.0, -1.0, 0.0)

              gl.glTexCoord2f(-halfw+0.0, -halfh+1.0)
              gl.glVertex3f(-1.0, -1.0, 0.0)

              gl.glEnd()

            glfw.swap_buffers(self.win)
            i += 1
            if i % 1 == 0:
              glfw.poll_events()
            cur_time = time.time()
            dtime = cur_time - self.last_time
            fps = 1.0 / float(dtime + 1e-9)
            self.last_time = cur_time
            print("%.1f" % fps)

        glfw.terminate()
