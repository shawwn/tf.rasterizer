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

    def __init__(self, texture=None, texture_sum=None):
        self.vertices = tf.placeholder(tf.float32, [None, 3])
        self.normals = tf.placeholder(tf.float32, [None, 3])
        self.uvs = tf.placeholder(tf.float32, [None, 2])
        self.texture = tf.placeholder(tf.float32, [None, None, 3]) if texture is None else texture
        #self.texture_op = self.texture / 255.0
        self.texture_sum = texture_sum if texture_sum is not None else tf.cumsum(tf.cumsum(self.texture / 255.0, 0), 1)

        default_light_dir = np.array([-1, -1, -1], dtype=np.float32)
        default_ambient = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        default_diffuse = np.array([1, 1, 1], dtype=np.float32)
        default_wvp = np.eye(4, dtype=np.float32)
        default_world = np.eye(4, dtype=np.float32)
        default_norm = np.eye(3, dtype=np.float32)

        self.light_dir = tf.placeholder_with_default(default_light_dir, [3])
        self.ambient = tf.placeholder_with_default(default_ambient, [3])
        self.diffuse = tf.placeholder_with_default(default_diffuse, [3])
        self.wvp = tf.placeholder_with_default(default_wvp, [4, 4])
        self.world = tf.placeholder_with_default(default_world, [4, 4])
        #self.inv_world = tf.matrix_inverse(self.world)
        self.norm = tf.placeholder_with_default(default_norm, [3, 3])

        self.packed_texture = utils.pack_colors(self.texture, 2, False)

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
        #uvs -= 0.5
        #uvs /= 16.0
        #uvs *= 3.0
        #uvs += 0.5 / 10.0
        #uvs /= 2.0
        vertices = tf.concat([vertices, tf.ones([num_verts, 1])], 1)
        normals = tf.concat([normals, tf.zeros([num_verts, 1])], 1)
        pos = tf.matmul(vertices, self.wvp, transpose_b=True)
        x = pos[:, 0]
        y = pos[:, 1]
        z = pos[:, 2]
        w = pos[:, 3]
        self.varying_pos[vertex_id] = tf.stack([x/w, y/w, z/w, 1.0/w], 1)
        self.varying_uv[vertex_id] = self.output(w, uvs, 2)
        norm = tf.matmul(normals, self.world, transpose_b=True)[:, :3]
        self.varying_norm[vertex_id] = self.output(w, norm, 3)
        return pos

    @utils.op_scope
    def varying(self, w, v, bc, i, k, bcx, bcy, bcxy):
        tc = utils.tri_dot(utils.tri_gather(v, i), bc)
        tc *= tf.stack([w for _ in range(k)], 1)
        tcx = utils.tri_dot(utils.tri_gather(v, i), bcx)
        tcx *= tf.stack([w for _ in range(k)], 1)
        tcy = utils.tri_dot(utils.tri_gather(v, i), bcy)
        tcy *= tf.stack([w for _ in range(k)], 1)
        tcxy = utils.tri_dot(utils.tri_gather(v, i), bcxy)
        tcxy *= tf.stack([w for _ in range(k)], 1)
        return tc, tcx, tcy, tcxy

    @utils.op_scope
    def fragment(self, bc, i, bcx, bcy, bcxy):
        num_frags = tf.shape(bc[0])[0]
        bc = utils.tri_expand_dims(bc, 1)
        bcx = utils.tri_expand_dims(bcx, 1)
        bcy = utils.tri_expand_dims(bcy, 1)
        bcxy = utils.tri_expand_dims(bcxy, 1)
        pos = utils.tri_dot(utils.tri_gather(self.varying_pos, i), bc)
        w = 1.0 / pos[:, 3]
        norm, norm_x, norm_y, norm_xy = self.varying(w, self.varying_norm, bc, i, 3, bcx, bcy, bcxy)
        norm = tf.nn.l2_normalize(norm, 1)
        l = tf.expand_dims(tf.nn.l2_normalize(self.light_dir, 0), 1)
        d = utils.clamp(tf.matmul(norm, l), 0., 1.)
        uv, uv_x, uv_y, uv_xy = self.varying(w, self.varying_uv, bc, i, 2, bcx, bcy, bcxy)
        tex_0 = utils.sample(self.packed_texture, uv, "nearest")
        return tex_0 # why does area sampling fail with a slice out of bounds error?
        wh = tf.to_float(tf.shape(self.texture_sum[:, :, 0]))
        uv_00 = tf.reduce_min([uv, uv_x, uv_y, uv_xy], axis=0)
        uv_11 = tf.reduce_max([uv, uv_x, uv_y, uv_xy], axis=0)
        uv_01 = tf.stack([uv_00[:, 0], uv_11[:, 1]], 1)
        uv_10 = tf.stack([uv_11[:, 0], uv_00[:, 1]], 1)
        #import pdb; pdb.set_trace()
        R = wh[0]*(uv_11[:, 0] - uv_00[:, 0]) * wh[1]*(uv_11[:, 1] - uv_00[:, 1])
        #uv_h = tf.stack([uv_11[:, 0], uv_00[:, 1]], 1)
        #R = tf.reduce_prod((uv_11 - uv_00) * tf.to_float(wh)) / 4
        #R = 100
        #pw = uv_11[:, 0]

        # AREA filtering
        tex_00 = utils.sample(self.texture_sum, uv_00, "bilinear", unpack=False)
        tex_10 = utils.sample(self.texture_sum, uv_10, "bilinear", unpack=False)
        tex_01 = utils.sample(self.texture_sum, uv_01, "bilinear", unpack=False)
        tex_11 = utils.sample(self.texture_sum, uv_11, "bilinear", unpack=False)
        den = tf.cond(tf.greater(tf.shape(R)[0], 1), lambda: R[i], lambda: wh[0]*wh[1])
        #R = utils.tf_prn(R, i, R[i])
        den = utils.tf_prn(den, i, den)
        return (tex_11 - tex_10 - tex_01 + tex_00) / den
        # #tex_0 = utils.sample(self.texture_sum, uv, "nearest", unpack=False)
        # #tex_x = utils.sample(self.texture_sum, uv_x, "nearest", unpack=False)
        # #tex_y = utils.sample(self.texture_sum, uv_y, "nearest", unpack=False)
        # #tex_xy = utils.sample(self.texture_sum, uv_xy, "nearest", unpack=False)
        # #return (tex_xy - tex_x - tex_y + tex_0)
        # A = tex_0
        # D = tex_x
        # B = tex_xy
        # C = tex_y
        # return (B - C - D + A)
        # #import pdb; pdb.set_trace()
        # #return tf.concat([uv, tf.zeros_like(tex_0[:, 0])[:, tf.newaxis]], 1)
        #return tex_0 / (1024*100)
        tex_x = utils.sample(self.packed_texture, uv_x, "nearest")
        tex_y = utils.sample(self.packed_texture, uv_y, "nearest")
        tex_xy = utils.sample(self.packed_texture, uv_xy, "nearest")
        tex = 0.25 * (tex_0 + tex_x + tex_y + tex_xy)
        return tex
        result = (self.ambient + self.diffuse * d) * tex
        return result
