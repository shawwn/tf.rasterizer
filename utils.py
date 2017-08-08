"""Auxilary functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def op_scope(fn, name=None):
    if name is None:
        name = fn.__name__
    def _fn(*args, **kwargs):
        with tf.name_scope(fn.__name__):
            return fn(*args, **kwargs)
    return _fn


def viewport(x, y, width, height):
    hw = width * 0.5
    hh = height * 0.5
    return np.array([
        [hw, 0., 0., hw + x],
        [0., hh, 0., hh + y],
        [0., 0., 1., 0.],
        [0., 0., 0., 1.]], dtype=np.float32)


def rotation(x, y, z):
    sin_x, sin_y, sin_z = np.sin([x, y, z])
    cos_x, cos_y, cos_z = np.cos([x, y, z])
    return [
        [
            cos_x * cos_y,
            cos_x * sin_y * sin_z - sin_x * cos_z,
            cos_x * sin_y * cos_z + sin_x * sin_z,
            0.
        ],
        [
            sin_x * cos_y,
            sin_x * sin_y * sin_z + cos_x * cos_z,
            sin_x * sin_y * cos_z - cos_x * sin_z,
            0.
        ],
        [-sin_y, cos_y * sin_z, cos_y * cos_z, 0.],
        [0., 0., 0., 1.]
    ]


@op_scope
def clamp(v, min=0., max=1.):
    return tf.minimum(tf.maximum(v, min), max)


@op_scope
def sample(tex, uv):
    uv = clamp(uv, 0., 1.) * tf.to_float(tf.shape(tex) - 1)
    return tf.gather_nd(tex, tf.to_int32(uv))


@op_scope
def tri_cross(v1, v2):
    return [v1[1] * v2[2] - v1[2] * v2[1],
            v1[2] * v2[0] - v1[0] * v2[2],
            v1[0] * v2[1] - v1[1] * v2[0]]


@op_scope
def tri_dot(v1, v2):
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]


@op_scope
def tri_expand_dims(v, axis):
    return (tf.expand_dims(v[0], axis),
            tf.expand_dims(v[1], axis),
            tf.expand_dims(v[2], axis))


@op_scope
def tri_gather(v, inds):
    return (tf.gather(v[0], inds),
            tf.gather(v[1], inds),
            tf.gather(v[2], inds))


@op_scope
def pack_colors(color, axis, scale=True):
    if scale:
        color = color * 255.
    color = clamp(color, 0., 255.)
    color = tf.to_int32(color)
    r, g, b = tf.unstack(color, axis=axis)
    color = r + (g * 256) + b * (256 ** 2)
    return color


@op_scope
def unpack_colors(color, axis, normalize=True):
    r = tf.mod(color, 256)
    g = tf.mod(tf.floordiv(color, 256), 256)
    b = tf.mod(tf.floordiv(color, 256 ** 2), 256 ** 2)
    color = tf.stack([r, g, b], axis=axis)
    if normalize:
        color = tf.div(tf.to_float(color), 255.)
    return color


@op_scope
def sequential_for(fn, begin, end):

    def _cond(i):
        return tf.less(i, end)

    def _body(i):
        ops = fn(i)
        with tf.control_dependencies(ops):
            return i + 1

    return tf.while_loop(_cond, _body, [begin])


@op_scope
def affine_to_cartesian(points):
    return points[:, :3] / tf.expand_dims(points[:, 3], 1)
