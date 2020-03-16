# Copyright (c) 2020 smarsu. All Rights Reserved.

import ctypes
import os.path as osp
import numpy as np

def get_smcb_lib_path():
    rlpt = osp.realpath(__file__)
    rldir = osp.split(osp.split(rlpt)[0])[0]
    lbpt = osp.join(rldir, 'third_party', 'lib', 'libsmcb.so')
    return lbpt


libsmcb = ctypes.cdll.LoadLibrary(get_smcb_lib_path())


def c_data(data):
  host_ptr = data.ctypes.data_as(ctypes.c_void_p)
  return host_ptr


def broadcast_shape(shape1, shape2):
  shape1 = list(shape1)
  shape2 = list(shape2)

  length = max(len(shape1), len(shape2))
  shape1 = [1] * (length - len(shape1)) + shape1
  shape2 = [1] * (length - len(shape2)) + shape2

  shape = []
  for dim1, dim2 in zip(shape1, shape2):
    assert dim1 == dim2 or dim1 == 1 or dim2 == 1, 'Can not broadcast ' \
      'shape {} with {}'.format(shape1, shape2)

    shape.append(max(dim1, dim2))

  return shape, shape1, shape2


def infer_shape(shape1, shape2):
  shape2 = list(shape2)

  size1 = np.prod(shape1)
  size2 = np.prod([1 if dim == -1 else dim for dim in shape2])
  infer_dim = size1 // size2
  for idx in range(len(shape2)):
    if shape2[idx] == -1:
      shape2[idx] = infer_dim
      return shape2


def get_reduce_axis(shape, res_shape):
  shape = list(shape)
  res_shape = list(res_shape)

  length = len(res_shape) - len(shape)
  shape = [1] * length + shape

  axis = list(range(length))
  for index, (dim1, dim2) in enumerate(zip(shape, res_shape)):
    if dim1 != dim2:
      assert dim1 == 1
      if index not in axis:
        axis.append(index)
  
  return tuple(axis)


def get_3level_shape(shape, axis):
  if axis < 0:
    axis += len(shape)

  shape = [1] + list(shape) + [1]
  axis += 1

  out_dim = int(np.prod(shape[:axis]))
  dim = int(np.prod(shape[axis:axis+1]))
  inner_dim = int(np.prod(shape[axis+1:]))

  return out_dim, dim, inner_dim


def acc(lst):
  sum = 0
  dst = []
  for v in lst:
    sum += v
    dst.append(sum)
  return dst


def pos_axis(axis, length):
  return axis if axis >= 0 else length + axis


def sigmoid(x):
  return 1 / (1 + np.exp(-x))


def softmax(x, dim=-1):
  x = x - np.max(x, axis=dim, keepdims=True)
  ex = np.exp(x)
  sum_ex = np.sum(ex, axis=dim, keepdims=True)
  return ex / sum_ex


def conv2d(output, input, filter, strides, paddings, dilations):
  strides = np.array(strides, dtype=np.int32)
  paddings = np.array(paddings, dtype=np.int32)
  dilations = np.array(dilations, dtype=np.int32)

  output_shape = np.array(output.shape, dtype=np.int32)
  input_shape = np.array(input.shape, dtype=np.int32)
  filter_shape = np.array(filter.shape, dtype=np.int32)

  libsmcb.conv2d_FP32(c_data(output),
                      c_data(input),
                      c_data(filter),
                      c_data(strides),
                      c_data(paddings),
                      c_data(dilations),
                      c_data(output_shape),
                      c_data(input_shape),
                      c_data(filter_shape))


def conv2d_backward_data(input_grad, output_grad, filter, strides, paddings, dilations, alpha):
  strides = np.array(strides, dtype=np.int32)
  paddings = np.array(paddings, dtype=np.int32)
  dilations = np.array(dilations, dtype=np.int32)

  output_shape = np.array(output_grad.shape, dtype=np.int32)
  input_shape = np.array(input_grad.shape, dtype=np.int32)
  filter_shape = np.array(filter.shape, dtype=np.int32)

  libsmcb.conv2d_backward_data_FP32(c_data(input_grad),
                                    c_data(output_grad),
                                    c_data(filter),
                                    c_data(strides),
                                    c_data(paddings),
                                    c_data(dilations),
                                    c_data(output_shape),
                                    c_data(input_shape),
                                    c_data(filter_shape),
                                    ctypes.c_float(alpha))


def conv2d_backward_filter(filter_grad, input, output_grad, strides, paddings, dilations, alpha):
  strides = np.array(strides, dtype=np.int32)
  paddings = np.array(paddings, dtype=np.int32)
  dilations = np.array(dilations, dtype=np.int32)

  output_shape = np.array(output_grad.shape, dtype=np.int32)
  input_shape = np.array(input.shape, dtype=np.int32)
  filter_shape = np.array(filter_grad.shape, dtype=np.int32)

  libsmcb.conv2d_backward_filter_FP32(c_data(filter_grad),
                                      c_data(input),
                                      c_data(output_grad),
                                      c_data(strides),
                                      c_data(paddings),
                                      c_data(dilations),
                                      c_data(output_shape),
                                      c_data(input_shape),
                                      c_data(filter_shape),
                                      ctypes.c_float(alpha))


def max_pool2d(output, value, ksize, strides):
  ksize = np.array(ksize, dtype=np.int32)
  strides = np.array(strides, dtype=np.int32)

  output_shape = np.array(output.shape, dtype=np.int32)
  value_shape = np.array(value.shape, dtype=np.int32)

  libsmcb.max_pool2d_FP32(c_data(output),
                          c_data(value),
                          c_data(ksize),
                          c_data(strides),
                          c_data(output_shape),
                          c_data(value_shape))


def max_pool2d_backward(value_grad, output_grad, value, ksize, strides, alpha):
  ksize = np.array(ksize, dtype=np.int32)
  strides = np.array(strides, dtype=np.int32)

  output_shape = np.array(output_grad.shape, dtype=np.int32)
  value_shape = np.array(value_grad.shape, dtype=np.int32)

  libsmcb.max_pool2d_backward_FP32(c_data(value_grad),
                                   c_data(output_grad),
                                   c_data(value),
                                   c_data(ksize),
                                   c_data(strides),
                                   c_data(output_shape),
                                   c_data(value_shape),
                                   ctypes.c_float(alpha)) 


def avg_pool2d(output, value, ksize, strides, paddings):
  ksize = np.array(ksize, dtype=np.int32)
  strides = np.array(strides, dtype=np.int32)
  paddings = np.array(paddings, dtype=np.int32)

  output_shape = np.array(output.shape, dtype=np.int32)
  value_shape = np.array(value.shape, dtype=np.int32)

  libsmcb.avg_pool2d_FP32(c_data(output),
                          c_data(value),
                          c_data(ksize),
                          c_data(strides),
                          c_data(paddings),
                          c_data(output_shape),
                          c_data(value_shape))


def avg_pool2d_backward(value_grad, output_grad, ksize, strides, paddings, alpha):
  ksize = np.array(ksize, dtype=np.int32)
  strides = np.array(strides, dtype=np.int32)
  paddings = np.array(paddings, dtype=np.int32)

  output_shape = np.array(output_grad.shape, dtype=np.int32)
  value_shape = np.array(value_grad.shape, dtype=np.int32)

  libsmcb.avg_pool2d_backward_FP32(c_data(value_grad),
                                   c_data(output_grad),
                                   c_data(ksize),
                                   c_data(strides),
                                   c_data(paddings),
                                   c_data(output_shape),
                                   c_data(value_shape),
                                   ctypes.c_float(alpha))


def softmax_cross_entropy_with_logits_backward_logits(logits_grad,
                                                      output_grad,
                                                      label,
                                                      softmax_logits,
                                                      axis,
                                                      alpha):
  out_dim, dim, inner_dim = get_3level_shape(logits_grad.shape, axis) 

  libsmcb.softmax_cross_entropy_with_logits_backward_logits_FP32(c_data(logits_grad),
                                                                 c_data(output_grad),
                                                                 c_data(label),
                                                                 c_data(softmax_logits),
                                                                 ctypes.c_int(out_dim),
                                                                 ctypes.c_int(dim),
                                                                 ctypes.c_int(inner_dim),
                                                                 ctypes.c_float(alpha))
