"""Shaders."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import renderer
import utils


class TexturedLitShader(renderer.Shader):
    """Textured shader class."""

    def __init__(self):
        self.vertices = None
        self.normals = None
        self.uvs = None
        self.texture = None
        self.light_dir = np.array([-1, -1, -1], dtype=np.float32)
        self.ambient = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        self.diffuse = np.array([1, 1, 1], dtype=np.int32)
        self.wvp = np.eye(4, dtype=np.float32)
        self.iwvp = np.eye(4, dtype=np.float32)

        self.varying_uv = [None, None, None]
        self.varying_norm = [None, None, None]

    @utils.op_scope
    def vertex(self, indices, vertex_id):
        num_verts = tf.shape(indices)[0]
        vertices = tf.gather(self.vertices, indices)
        normals = tf.gather(self.normals, indices)
        uvs = tf.gather(self.uvs, indices)
        vertices = tf.concat([vertices, tf.ones([num_verts, 1])], 1)
        normals = tf.concat([normals, tf.zeros([num_verts, 1])], 1)
        self.varying_uv[vertex_id] = uvs
        self.varying_norm[vertex_id] = tf.matmul(
            normals, self.iwvp, transpose_b=True)[:, :3]
        return tf.matmul(vertices, self.wvp, transpose_b=True)

    @utils.op_scope
    def fragment(self, bc, i):
        num_frags = tf.shape(bc[0])[0]
        bc = utils.tri_expand_dims(bc, 1)
        norm = utils.tri_dot(utils.tri_gather(self.varying_norm, i), bc)
        norm = tf.nn.l2_normalize(norm, 1)
        l = tf.expand_dims(tf.nn.l2_normalize(self.light_dir, 0), 1)
        d = utils.clamp(tf.matmul(norm, l), 0., 1.)
        uv = utils.tri_dot(utils.tri_gather(self.varying_uv, i), bc)
        tex = utils.unpack_colors(utils.sample(self.texture, uv), 1)
        return (self.ambient + self.diffuse * d) * tex
