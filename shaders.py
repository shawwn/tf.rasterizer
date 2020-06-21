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
        default_world = np.eye(4, dtype=np.float32)
        default_inv_world = np.eye(4, dtype=np.float32)
        default_norm = np.eye(3, dtype=np.float32)

        self.light_dir = tf.placeholder_with_default(default_light_dir, [3])
        self.ambient = tf.placeholder_with_default(default_ambient, [3])
        self.diffuse = tf.placeholder_with_default(default_diffuse, [3])
        self.wvp = tf.placeholder_with_default(default_wvp, [4, 4])
        self.world = tf.placeholder_with_default(default_world, [4, 4])
        #self.inv_world = tf.placeholder_with_default(default_inv_world, [4, 4])
        self.inv_world = tf.matrix_inverse(self.world)
        self.norm = tf.placeholder_with_default(default_norm, [3, 3])

        self.packed_texture = utils.pack_colors(self.texture, 2, False)
        #self.iwvp = tf.matrix_inverse(self.wvp)

        self.varying_pos = [None, None, None]
        self.varying_uv = [None, None, None]
        self.varying_norm = [None, None, None]

    @utils.op_scope
    def output(self, w, v, k):
      return tf.stack([v[:, j] / w for j in range(k)], 1)

    @utils.op_scope
    def vertex(self, indices, vertex_id):
        num_verts = tf.shape(indices)[0]
        vertices = tf.gather(self.vertices, indices)
        normals = tf.gather(self.normals, indices)
        uvs = tf.gather(self.uvs, indices)
        #import pdb; pdb.set_trace()
        vertices = tf.concat([vertices, tf.ones([num_verts, 1])], 1)
        normals = tf.concat([normals, tf.zeros([num_verts, 1])], 1)
        #self.varying_uv[vertex_id] = uvs
        #self.varying_norm[vertex_id] = tf.matmul(
        #    normals, self.iwvp, transpose_b=True)[:, :3]
        result = tf.matmul(vertices, self.wvp, transpose_b=True)
        # w = result[:, 3]
        # u_w = uvs[:, 0] / w
        # v_w = uvs[:, 1] / w
        # z_w = result[:, 2] / w
        # i_w = 1.0 / w
        # texcoords = tf.stack([u_w, v_w, z_w, i_w], 1)
        # self.varying_tc[vertex_id] = texcoords
        w = result[:, 3]
        x = result[:, 0] / w
        y = result[:, 1] / w
        z = result[:, 2] / w
        self.varying_pos[vertex_id] = tf.stack([x, y, z, 1.0 / w], 1)
        #self.varying_uv[vertex_id] /= w[vertex_id]
        #self.varying_norm[vertex_id] /= w[vertex_id]
        self.varying_uv[vertex_id] = self.output(w, uvs, 2)
        norm = tf.matmul(normals, self.world, transpose_b=True)[:, :3]
        self.varying_norm[vertex_id] = self.output(w, norm, 3)
        return result

    @utils.op_scope
    def varying(self, w, v, bc, i, k):
        # pos = utils.tri_dot(utils.tri_gather(self.varying_pos, i), bc)
        # w = 1.0 / pos[:, 3]
        # #tc = utils.tri_dot(utils.tri_gather(v, i) / w, bc) * w
        # #return tc
        tc = utils.tri_dot(utils.tri_gather(v, i), bc)
        tc *= tf.stack([w for _ in range(k)], 1)
        return tc
        # assert k in [2, 3]
        # if k == 2:
        #   tc_u = tc[:, 0] * w
        #   tc_v = tc[:, 1] * w
        #   return tf.stack([tc_u, tc_v], 1)
        # elif k == 3:
        #   tc_x = tc[:, 0] * w
        #   tc_y = tc[:, 1] * w
        #   tc_z = tc[:, 2] * w
        #   return tf.stack([tc_x, tc_y, tc_z], 1)
        # # tc_u = tc[:, 0] * w
        # # tc_v = tc[:, 1] * w
        # # tc_z = tc[:, 2] * w
        # # tc_uv = tf.stack([tc_u, tc_v], 1)
        # # tex = utils.unpack_colors(utils.sample(self.packed_texture, tc_uv), 1)
      

    @utils.op_scope
    def fragment(self, bc, i):
        num_frags = tf.shape(bc[0])[0]
        #import pdb; pdb.set_trace()
        bc = utils.tri_expand_dims(bc, 1)
        pos = utils.tri_dot(utils.tri_gather(self.varying_pos, i), bc)
        w = 1.0 / pos[:, 3]
        #norm = utils.tri_dot(utils.tri_gather(self.varying_norm, i), bc)
        norm = self.varying(w, self.varying_norm, bc, i, 3)
        norm = tf.nn.l2_normalize(norm, 1)
        l = tf.expand_dims(tf.nn.l2_normalize(self.light_dir, 0), 1)
        d = utils.clamp(tf.matmul(norm, l), 0., 1.)
        uv = self.varying(w, self.varying_uv, bc, i, 2)
        #uv = utils.tri_dot(utils.tri_gather(self.varying_uv, i), bc)
        #tc = utils.tri_dot(utils.tri_gather(self.varying_tc, i), bc)
        #import pdb; pdb.set_trace()
        # tc_w = 1.0 / tc[:, 3]
        # tc_u = tc[:, 0] * tc_w
        # tc_v = tc[:, 1] * tc_w
        # tc_z = tc[:, 2] * tc_w
        # tc_uv = tf.stack([tc_u, tc_v], 1)
        # tex = utils.unpack_colors(utils.sample(self.packed_texture, tc_uv), 1)
        tex = utils.unpack_colors(utils.sample(self.packed_texture, uv), 1)
        result = (self.ambient + self.diffuse * d) * tex
        #result = (self.ambient + self.diffuse * d)
        #import pdb; pdb.set_trace()
        #result = tf.broadcast_to([uv[0], uv[1], 0.0])
        return result
