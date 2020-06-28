"""Auxilary functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import functools


def op_scope(fn, name=None):
    if name is None:
        name = fn.__name__
    @functools.wraps(fn)
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
    #return tf.minimum(tf.maximum(v, min), max)
    return tf.clip_by_value(v, min, max)

@op_scope
def wrap(v, mode):
  assert mode in ["clamp", "wrap", "reflect"]
  if mode == "wrap":
    return tf.floormod(v, 1.0)
  elif mode == "reflect":
    return tf.abs(tf.floormod(v, 2.0) - 1.0)
  elif mode == "clamp":
    return clamp(v)

@op_scope
def iround(u):
  return tf.to_int32(tf.floordiv(tf.to_float(u), 1.0))

@op_scope
def sample(tex, uv, mode="bilinear", wrap_mode="reflect", unpack=True):
  assert mode in ["nearest", "bilinear"]
  wh = tf.shape(tex if unpack else tex[:, :, 0])
  grab = lambda u, v: tf.gather_nd(tex, tf.stack([
    clamp(iround(u), 0, wh[0] - 1),
    clamp(iround(v), 0, wh[1] - 1),
    ], 1))
  get = lambda u, v: (unpack_colors(grab(u, v), 1) if unpack else grab(u, v))
  if mode == "nearest":
    uv = wrap(uv, wrap_mode) * tf.to_float(wh)
    u = uv[:, 0]
    v = uv[:, 1]
    return get(u, v)
  elif mode == "bilinear":
    # https://github.com/pytorch/pytorch/blob/f064c5aa33483061a48994608d890b968ae53fb5/aten/src/THNN/generic/SpatialGridSamplerBilinear.c#L105
    uv = wrap(uv, wrap_mode) * tf.to_float(wh)
    ix = uv[:, 0] - 0.5
    iy = uv[:, 1] - 0.5

    # get NE, NW, SE, SW pixel values from (x, y)
    ix_nw = iround(ix)
    iy_nw = iround(iy)
    ix_ne = ix_nw + 1
    iy_ne = iy_nw
    ix_sw = ix_nw
    iy_sw = iy_nw + 1
    ix_se = ix_nw + 1
    iy_se = iy_nw + 1

    sub = lambda a, b: tf.to_float(a) - tf.to_float(b)

    # get surfaces to each neighbor:
    nw = sub(ix_se , ix)    * sub(iy_se , iy);
    ne = sub(ix    , ix_sw) * sub(iy_sw , iy);
    sw = sub(ix_ne , ix)    * sub(iy    , iy_ne);
    se = sub(ix    , ix_nw) * sub(iy    , iy_nw);

    nw_val = get(ix_nw, iy_nw)
    ne_val = get(ix_ne, iy_ne)
    sw_val = get(ix_sw, iy_sw)
    se_val = get(ix_se, iy_se)

    a = lambda x: x[:, tf.newaxis]
    out = nw_val * a(nw)
    out += ne_val * a(ne)
    out += sw_val * a(sw)
    out += se_val * a(se)
    return out


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
def sequential_for(fn, begin, end, *args):

    def _cond(i, *xs):
        return tf.less(i, end)

    def _body(i, *xs):
        ops, *ys = fn(i, *xs)
        with tf.control_dependencies(ops):
            return [i + 1] + list(ys)

    return tf.while_loop(_cond, _body, [begin] + list(args))


@op_scope
def affine_to_cartesian(points):
    return points[:, :3] / tf.expand_dims(points[:, 3], 1)


@op_scope
def tf_prn(x, *args):
  with tf.control_dependencies([tf.print(*args)]):
    return tf.identity(x)
