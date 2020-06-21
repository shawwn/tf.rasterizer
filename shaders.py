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

    def __init__(self, texture=None):
        self.vertices = tf.placeholder(tf.float32, [None, 3])
        self.normals = tf.placeholder(tf.float32, [None, 3])
        self.uvs = tf.placeholder(tf.float32, [None, 2])
        self.texture = tf.placeholder(tf.float32, [None, None, 3]) if texture is None else texture

        default_light_dir = np.array([-1, -1, -1], dtype=np.float32)
        default_ambient = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        default_diffuse = np.array([1, 1, 1], dtype=np.float32)
        default_wvp = np.eye(4, dtype=np.float32)

        self.light_dir = tf.placeholder_with_default(default_light_dir, [3])
        self.ambient = tf.placeholder_with_default(default_ambient, [3])
        self.diffuse = tf.placeholder_with_default(default_diffuse, [3])
        self.wvp = tf.placeholder_with_default(default_wvp, [4, 4])

        self.packed_texture = utils.pack_colors(self.texture, 2, False)
        self.iwvp = tf.matrix_inverse(self.wvp)

        self.varying_uv = [None, None, None]
        self.varying_tc = [None, None, None]
        self.varying_norm = [None, None, None]

    @utils.op_scope
    def vertex(self, indices, vertex_id):
        num_verts = tf.shape(indices)[0]
        vertices = tf.gather(self.vertices, indices)
        normals = tf.gather(self.normals, indices)
        uvs = tf.gather(self.uvs, indices)
        #import pdb; pdb.set_trace()
        vertices = tf.concat([vertices, tf.ones([num_verts, 1])], 1)
        normals = tf.concat([normals, tf.zeros([num_verts, 1])], 1)
        self.varying_uv[vertex_id] = uvs
        self.varying_norm[vertex_id] = tf.matmul(
            normals, self.iwvp, transpose_b=True)[:, :3]
        result = tf.matmul(vertices, self.wvp, transpose_b=True)
        w = result[:, 3]
        z = result[:, 2] / w
        u_z = uvs[:, 0] / w
        v_z = uvs[:, 1] / w
        i_z = 1.0 / w
        texcoords = tf.stack([u_z, v_z, i_z], 1)
        self.varying_tc[vertex_id] = texcoords
        return result

    @utils.op_scope
    def fragment(self, bc, i):
        num_frags = tf.shape(bc[0])[0]
        #import pdb; pdb.set_trace()
        bc = utils.tri_expand_dims(bc, 1)
        norm = utils.tri_dot(utils.tri_gather(self.varying_norm, i), bc)
        norm = tf.nn.l2_normalize(norm, 1)
        l = tf.expand_dims(tf.nn.l2_normalize(self.light_dir, 0), 1)
        d = utils.clamp(tf.matmul(norm, l), 0., 1.)
        uv = utils.tri_dot(utils.tri_gather(self.varying_uv, i), bc)
        tc = utils.tri_dot(utils.tri_gather(self.varying_tc, i), bc)
        #import pdb; pdb.set_trace()
        tc_z = 1.0 / tc[:, 2]
        tc_u = tc[:, 0] * tc_z
        tc_v = tc[:, 1] * tc_z
        tc_uv = tf.stack([tc_u, tc_v], 1)
        tex = utils.unpack_colors(utils.sample(self.packed_texture, tc_uv), 1)
        result = (self.ambient + self.diffuse * d) * tex
        #result = (self.ambient + self.diffuse * d)
        #import pdb; pdb.set_trace()
        #result = tf.broadcast_to([uv[0], uv[1], 0.0])
        return result
