"""Renderer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tflex
import tflex_tpu
import utils
import os
import time
from pprint import pprint as pp

from tensorflow.contrib import tpu
from tensorflow.contrib.tpu.python.tpu import tpu_function
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.data.util import nest as data_nest
from tensorflow.python.framework import graph_io

FLT_MIN = -1.0e30
FLT_MAX = 1.0e30

_INITIAL_LOSS = 1e7

@utils.op_scope
def bounds(verts, width, height):
    bbmin = [FLT_MAX, FLT_MAX]
    bbmax = [FLT_MIN, FLT_MIN]
    wh = [width - 1., height - 1.]
    for j in range(2):
        for i in range(3):
            bbmin[j] = tf.minimum(bbmin[j], verts[i][:, j])
            bbmax[j] = tf.maximum(bbmax[j], verts[i][:, j])
        bbmin[j] = utils.clamp(bbmin[j], 0., wh[j])
        bbmax[j] = utils.clamp(bbmax[j], 0., wh[j])
    bbmin = tf.stack(bbmin, axis=1)
    bbmax = tf.stack(bbmax, axis=1)
    return bbmin, bbmax


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

def edge_function(a, b, c):
  return (c[0] - a[0]) * (b[1] - a[1]) - (c[1] - a[1]) * (b[0] - a[0]);

@utils.op_scope
def barycentric2(verts, p):
  v0 = verts[0]
  v1 = verts[1]
  v2 = verts[2]
  area = edge_function(v0, v1, v2)
  w0 = edge_function(v1, v2, p); 
  w1 = edge_function(v2, v0, p); 
  w2 = edge_function(v0, v1, p); 
  w0 /= area
  w1 /= area
  w2 /= area
  c0 = [1, 0, 0]
  c1 = [0, 1, 0]
  c2 = [0, 0, 1]
  #r = w0 * c0[0] + w1 * c1[0] + w2 * c2[0]; 
  #g = w0 * c0[1] + w1 * c1[1] + w2 * c2[1]; 
  #b = w0 * c0[2] + w1 * c1[2] + w2 * c2[2]; 
  #u = w1 - w0
  #v = w2 - w0
  bc = [w0 / area, w1 / area, w2 / area]
  #bc = [r, g, b]
  #valid = w0 >= 0 and w1 >= 0 and w2 >= 0
  valid = tf.logical_and(
      tf.greater_equal(w0, 0.0),
      tf.logical_and(
        tf.greater_equal(w1, 0.0),
        tf.greater_equal(w2, 0.0)))
  #valid = tf.reduce_all(tf.stack([w0, w1, w2], axis=1) >= 0, axis=1)
  return bc, tf_prn(valid, w0, w1, w2, area)


class Shader(object):
    """Shader class."""

    def vertex(self, unused_indices, unused_vertex_id):
        raise NotImplementedError("Vertex program not implemented.")

    def fragment(self, unused_bc, unused_i):
        raise NotImplementedError("Fragment program not implemented")

@utils.op_scope
def tf_prn(x, *args):
  with tf.control_dependencies([tf.print(*args)]):
    return tf.identity(x)

class Renderer(object):
    """Renderer class."""

    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        self.iterations = 32
        self.need_finalize = True

        config = tf.ConfigProto()
        #config.gpu_options.allow_growth = True
        #config.graph_options.optimizer_options.global_jit_level = (tf.OptimizerOptions.ON_1)

        self.session = tflex.Session(config=config)

        self.commands = []
        self.args = {}
        self.blank = np.zeros(shape=[height, width, 3], dtype=np.int32)
        self.result = tf.get_variable(
            "result", shape=[self.iterations], dtype=tf.float32, use_resource=True)
        self.color = tf.get_variable(
            "color", shape=[height, width], dtype=tf.int32, use_resource=True)
        self.depth = tf.get_variable(
            "depth", shape=[height, width], dtype=tf.float32, use_resource=True)

        self.viewport = utils.viewport(0., 0., self.width, self.height)

    @utils.op_scope
    def clear_fn(self):
        color = tf.placeholder(tf.float32, [3], name="ph_color")
        depth = tf.placeholder(tf.float32, [], name="ph_depth")
        result = tf.placeholder(tf.float32, [self.iterations], name="ph_result")
        packed_color = utils.pack_colors(color, 0)
        tiled_color = tf.fill([self.height, self.width], packed_color)
        tiled_depth = tf.fill([self.height, self.width], depth)
        #tiled_result = tf.fill([self.iterations], result)
        tiled_result = result
        assign_color = tf.assign(self.color, tiled_color)
        assign_depth = tf.assign(self.depth, tiled_depth)
        assign_result = tf.assign(self.result, tiled_result)
        self.commands.append(assign_color)
        self.commands.append(assign_depth)
        self.commands.append(assign_result)

        def _clear(result_val, color_val=[0., 0., 0.], depth_val=FLT_MIN):
            assert(len(result_val) == self.iterations)
            self.args[color] = color_val
            self.args[depth] = depth_val
            self.args[result] = result_val

        return _clear

    @utils.op_scope
    def draw_fn(self, shader):
        indices = tf.placeholder(tf.int32, [None, 3], name="ph_indices")
        verts = [None, None, None]

        for i in range(3):
            verts[i] = shader.vertex(indices[:, i], i)
            #import pdb; pdb.set_trace()
            #with tf.control_dependencies([tf.print([i, indices[:, i], verts[i]])]):
            #with tf.control_dependencies([tf.print([i, verts[i][0]])]):
            verts[i] = tf.matmul(verts[i], self.viewport, transpose_b=True)
            verts[i] = utils.affine_to_cartesian(verts[i])
        #with tf.control_dependencies([tf.print([i, verts[0][0]])]):
        #  verts[0] = tf.identity(verts[0])

        bbmin, bbmax = bounds(verts, self.width, self.height)

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
                          tf.zeros([num_frags], dtype=tf.float32)], axis=1)

            bc, valid = barycentric(verts_i, p)

            #bc = tf_prn(bc, bbmin_i, bbmax_i, i, [0, verts_i[0], tf.shape(bc[0])], [1, verts_i[1], tf.shape(bc[1])], [2, verts_i[2], tf.shape(bc[2])], tf.shape(p))

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

        num_faces = tf.shape(indices)[0]
        updates = utils.sequential_for(_fn, 0, num_faces)
        self.commands.append(updates)

        def _draw(indices_val, **kwargs):
            self.args[indices] = indices_val
            for k, v in kwargs.items():
                self.args[getattr(shader, k)] = v

        return _draw

    def init(self):
        self.session.run(tf.global_variables_initializer())

    def finalize(self):
        self.v = tf.constant(self.blank)
        def tpu_step(i, prev_result):
            #import pdb; pdb.set_trace()
            with tf.control_dependencies(self.commands):
                color_op = utils.unpack_colors(self.color, 2, False)
                #color_op = tf.identity(self.blank)
                #color_op = tf.identity(self.v)
                #color_op = tf.no_op()
                #color_op = tf.gather(self.result, i)
                #import pdb; pdb.set_trace()
                #x = x.write(i, [prev_result]*3 + x.gather(indices=[i - 1])[0])
                #z = x.read(i-1)
                #z = tf.constant([1.0, 1.0, 1.0])
                #import pdb; pdb.set_trace()
                #x = x.write(i, z)
                #x = x.write(0, 1.0)
                #import pdb; pdb.set_trace()
                #ta = ta.write(i, matrix[i] * 2)
                #ta = ta.write(i, self.foo.handle.op)
                #ta = ta.write(i, self.foo[:])
                #return color_op + prev_result, ta
                return color_op

        #x = tf.TensorArray(dtype=tf.float32,size=1, dynamic_size=True,clear_after_read=False, element_shape=())
        
        #self.foo = tf.get_variable("foo", shape=[100], dtype=tf.int32, use_resource=True)
        #matrix = tf.placeholder(tf.int32, shape=(100, 1000), name="input_matrix")
        #matrix_rows = tf.shape(matrix)[0]
        #ta = tf.TensorArray(dtype=tf.int32, size=matrix_rows, dynamic_size=True, element_shape=(100,))
        #import pdb; pdb.set_trace()
        #x = x.write(0, [0., 0., 0.])

        @tpu_function.on_device_training_loop
        def tpu_loop():
          return tflex_tpu.repeat(self.iterations, tpu_step, [self.blank], arrays=[])

        if True:
          self.color_op = tpu_step(0, self.blank)
        elif True:
          (self.color_op,) = tpu.rewrite(tpu_loop, inputs=[])
        else:
          (self.color_op,) = tpu.shard(
              tpu_loop,
              inputs=[],
              num_shards=int(os.environ['TPU_CORES']) if 'TPU_CORES' in os.environ else 8,
              outputs_from_all_shards=False,
              #outputs_from_all_shards=True,
          )
            
        self.need_finalize = False

    def execute(self):
        if self.need_finalize:
            self.finalize()
        now = time.time()
        color_val = self.session.run(self.color_op, self.args)
        #pp(color_val)
        print('examples/sec: ', self.iterations / (time.time() - now))
        #color_val = self.blank
        #import pdb; pdb.set_trace()
        self.args = {}
        return color_val
