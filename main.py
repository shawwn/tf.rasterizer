"""Main module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from PIL import Image
import pyassimp
import renderer
import shaders
import tensorflow as tf
import time
import utils
import window


class App(object):
    """Application."""

    def __init__(self, width=800, height=600):
        # Create a window
        self.win = window.Window(width, height, "App")

        # Load mesh
        mesh = pyassimp.load("data/african_head/african_head.obj")
        self.indices = mesh.meshes[0].faces
        vertices = mesh.meshes[0].vertices
        normals = mesh.meshes[0].normals
        uvs = mesh.meshes[0].texturecoords[0, :, 1::-1]

        # Load texture
        image = Image.open("data/african_head/african_head_diffuse.tga")
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        texture = utils.pack_colors(
            np.array(image, dtype=np.float32), 2, False)

        # Shader
        shader = shaders.TexturedLitShader()
        shader.vertices = vertices
        shader.normals = normals
        shader.uvs = uvs
        shader.texture = texture

        # Create renderer
        self.rend = renderer.Renderer(width, height)
        self.rend.init()
        self.clear = self.rend.clear_fn()
        self.draw = self.rend.draw_fn(shader)

        self.last_time = time.time()

        self.win.loop(self)


    def __call__(self):
        self.clear([0.1, 0.1, 0.1])
        self.draw(self.indices)
        image = self.rend.execute()

        # Display frames per second
        cur_time = time.time()
        dtime = cur_time - self.last_time
        fps = 1.0 / float(dtime + 1e-9)
        self.last_time = cur_time
        print("%.1f" % fps)

        return image


if __name__ == "__main__":
    App()
