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

import tflex

import m


class App(object):
    """Application."""

    #def __init__(self, width=256, height=256):
    def __init__(self, width=1024, height=768):
        # Create a window
        self.win = window.Window(width, height, "App")

        # Load mesh
        mesh = pyassimp.load("data/african_head/african_head.obj")
        #mesh = pyassimp.load("data/african_head/box.obj")
        #mesh = pyassimp.load("data/quad.obj")
        self.indices = mesh.meshes[0].faces
        self.vertices = mesh.meshes[0].vertices
        self.normals = mesh.meshes[0].normals
        self.uvs = mesh.meshes[0].texturecoords[0, :, 1::-1]

        # Load texture
        image = Image.open("data/african_head/african_head_diffuse.tga")#.resize([32,32])
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        self.texture = np.array(image, dtype=np.float32)

        # Create renderer
        self.rend = renderer.Renderer(width, height)
        self.texture_in = tf.get_variable(
            "texture_in", shape=self.texture.shape, dtype=tf.float32, use_resource=True)
        #self.rend.session.run(tf.assign(self.texture_in, self.texture))
        self.rend.init()
        tflex.assign_values([self.texture_in], [self.texture], session=self.rend.session)
        self.clear = self.rend.clear_fn()
        self.draw = self.rend.draw_fn(shaders.TexturedLitShader(self.texture_in))
        self.rend.finalize()

        self.start_time = self.last_time = time.time()

        self.win.loop(self)


    def __call__(self):
        cur_time = time.time()
        elapsed = cur_time - self.start_time
        theta = 0.3 * elapsed
        #wvp = utils.rotation(0., theta, 0.0)

        aspect = self.rend.width / self.rend.height
        proj = m.GrProjection.perspective(m.deg_to_rad(60.0), 1000.0, aspect, m.MPlane([0,0,-1], [0,0,-1]))
        #wvp = proj.matrix * m.MMat4x4(utils.rotation(0., theta, 0.0))

        cam = m.GrCamera();
        cam.proj = proj
        cam.look_at( m.MVec3(0.0, 0.0, 1.5), m.MVec3(0,0,0))
        world = m.MMat4x4(utils.rotation(0., theta, 0.0))
        inv_world = world.inverse()
        wvp = cam.view_proj_matrix * world
        norm = cam.normal_matrix

        self.clear([_ for _ in range(self.rend.iterations)], [0.1, 0.1, 0.1])
        self.draw(self.indices,
                  vertices=self.vertices,
                  normals=self.normals,
                  uvs=self.uvs,
                  #texture=self.texture_in.handle,
                  wvp=wvp.data,
                  world=world.data,
                  #inv_world=inv_world.data,
                  norm=norm.data)
        self.image = self.rend.execute()

        self.last_time = cur_time
        return self.image


if __name__ == "__main__":
    App()
