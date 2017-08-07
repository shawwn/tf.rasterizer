"""Renderer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import utils

FLT_MIN = -1.0e30
FLT_MAX = 1.0e30


@utils.op_scope
def barycentric(verts, p):
    ab = verts[2] - verts[0]
    ac = verts[1] - verts[0]
    pa = verts[0] - p
    u = utils.tri_cross(
        [ab[0], ac[0], pa[:, 0]],
        [ab[1], ac[1], pa[:, 1]])
    v = [u[0] / u[2], u[1] / u[2]]
    bc = [1. - v[0] - v[1], v[1], v[0]]
    valid = tf.logical_and(
        tf.abs(u[2]) >= 1.0,
        tf.reduce_all(tf.stack(bc, axis=1) >= 0, axis=1))
    return bc, valid


class Shader(object):
    """Shader class."""

    def vertex(self, unused_indices, unused_vertex_id):
        raise NotImplementedError("Vertex program not implemented.")

    def fragment(self, unused_bc, unused_i):
        raise NotImplementedError("Fragment program not implemented")


class Renderer(object):
    """Renderer class."""

    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.graph_options.optimizer_options.global_jit_level = (
            tf.OptimizerOptions.ON_1)

        self.session = tf.Session(config=config)

        self.commands = []
        self.args = {}
        self.color = tf.get_variable(
            "color", shape=[height, width], dtype=tf.int32)
        self.depth = tf.get_variable(
            "depth", shape=[height, width], dtype=tf.float32)

    @utils.op_scope
    def clear_fn(self):
        color = tf.placeholder(tf.float32, [3], name="ph_color")
        depth = tf.placeholder(tf.float32, [], name="ph_depth")
        packed_color = utils.pack_colors(color, 0)
        tiled_color = tf.fill([self.height, self.width], packed_color)
        tiled_depth = tf.fill([self.height, self.width], depth)
        assign_color = tf.assign(self.color, tiled_color)
        assign_depth = tf.assign(self.depth, tiled_depth)
        self.commands.append(assign_color)
        self.commands.append(assign_depth)

        def _clear(color_val=[0., 0., 0.], depth_val=FLT_MIN):
            self.args[color] = color_val
            self.args[depth] = depth_val

        return _clear

    @utils.op_scope
    def draw_fn(self, shader):
        indices = tf.placeholder(tf.int32, [None, 3], name="ph_indices")
        num_faces = tf.shape(indices)[0]
        verts = [None, None, None]
        vp = utils.viewport(0., 0., self.width, self.height)

        for i in range(3):
            verts[i] = shader.vertex(indices[:, i], i)
            verts[i] = tf.matmul(verts[i], vp, transpose_b=True,)
            verts[i] = verts[i] / tf.expand_dims(verts[i][:, 3], 1)

        bbmin = [None, None]
        bbmax = [None, None]
        wh = [self.width - 1., self.height - 1.]
        for j in range(2):
            bbmin[j] = tf.fill([num_faces], FLT_MAX)
            bbmax[j] = tf.fill([num_faces], FLT_MIN)

            for i in range(3):
                bbmin[j] = tf.minimum(bbmin[j], verts[i][:, j])
                bbmax[j] = tf.maximum(bbmax[j], verts[i][:, j])

            bbmin[j] = utils.clamp(bbmin[j], 0., wh[j])
            bbmax[j] = utils.clamp(bbmax[j], 0., wh[j])

        bbmin = tf.stack(bbmin, axis=1)
        bbmax = tf.stack(bbmax, axis=1)

        def _fn(i):
            bbmin_i = tf.gather(bbmin, i)
            bbmax_i = tf.gather(bbmax, i)
            verts_i = [tf.gather(verts[0], i),
                       tf.gather(verts[1], i),
                       tf.gather(verts[2], i)]

            x, y = tf.meshgrid(tf.range(bbmin_i[0], bbmax_i[0]),
                               tf.range(bbmin_i[1], bbmax_i[1]))

            num_frags = tf.reduce_prod(tf.shape(x))
            p = tf.stack([tf.reshape(x, [-1]),
                          tf.reshape(y, [-1]),
                          tf.zeros([num_frags], dtype=tf.float32),
                          tf.ones([num_frags], dtype=tf.float32)], axis=1)

            bc, valid = barycentric(verts_i, p)

            p = tf.boolean_mask(p, valid)
            bc = [tf.boolean_mask(bc[k], valid) for k in range(3)]
            z = utils.tri_dot([verts_i[k][2] for k in range(3)], bc)

            inds = tf.to_int32(tf.stack([p[:, 1], p[:, 0]], axis=1))
            cur_z = tf.gather_nd(self.depth, inds)
            visible = tf.less_equal(cur_z, z)

            inds = tf.boolean_mask(inds, visible)
            bc = [tf.boolean_mask(bc[k], visible) for k in range(3)]
            z = tf.boolean_mask(z, visible)

            c = utils.pack_colors(shader.fragment(bc, i), 1)

            updates = [
                tf.scatter_nd_update(self.color, inds, c, use_locking=False),
                tf.scatter_nd_update(self.depth, inds, z, use_locking=False)]
            return updates

        updates = utils.sequential_for(_fn, 0, num_faces)
        self.commands.append(updates)

        def _draw(indices_val, **kwargs):
            self.args[indices] = indices_val
            for k, v in kwargs.items():
                self.args[shader.getattr(k)] = v

        return _draw

    def init(self):
        self.session.run(tf.global_variables_initializer())

    def execute(self):
        with tf.control_dependencies(self.commands):
            color = utils.unpack_colors(self.color, 2, False)
        color_val = self.session.run(color, self.args)
        self.args = {}
        return color_val
