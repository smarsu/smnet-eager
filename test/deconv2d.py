# Copyright (c) 2020 smarsu. All Rights Reserved.

import numpy as np
import smnet as sm
import tensorflow as tf
np.random.seed(196)

from _base import TestBase


def sm_func(input, filter, output_shape, strides, paddings, bias):
  sm.reset()
  y = sm.deconv2d(input, filter, output_shape, strides, paddings, bias)
  return y, ()


def gt_func(input, filter, output_shape, strides, paddings, bias):
  input = tf.transpose(input, [0, 2, 3, 1])
  filter = tf.transpose(filter, [2, 3, 1, 0])
  output_shape = [output_shape[dim] for dim in [0, 2, 3, 1]]

  strides = [1] + strides + [1]

  y = tf.nn.conv2d_transpose(input, filter, output_shape, strides, paddings)
  y = tf.transpose(y, [0, 3, 1, 2])
  y += bias
  return y, tuple()


def to_inputs(shape_input, shape_filter, shape_output, strides, paddings, **params):
  """
  Args:
    shape_input: [n, ci, hi, wi]
    shape_filter: [ci, co, hf, wf]
  """
  loc = params['loc']
  scale = params['scale']

  input = np.random.normal(loc=loc, scale=scale, size=shape_input)
  filter = np.random.normal(loc=loc, scale=scale, size=shape_filter)
  bias = np.random.normal(loc=loc, scale=scale, size=[1, shape_filter[1], 1, 1])

  return (sm.Variable(input, dtype=np.float32), sm.Variable(filter, dtype=np.float32), shape_output, strides, paddings, sm.Variable(bias, dtype=np.float32)), \
         (tf.Variable(input, dtype=tf.float32), tf.Variable(filter, dtype=tf.float32), shape_output, strides, paddings, tf.Variable(bias, dtype=tf.float32)),


def get_output_dim(input_dim, pad, filter_dim, dilation, stride, padding):
  if padding == 'VALID':
    return filter_dim + stride * (input_dim - 1)
  elif padding == 'SAME':
    return input_dim * stride
  else:
    raise ValueError(padding)


if __name__ == '__main__':
  testbase = TestBase('Conv2D', sm_func, gt_func, to_inputs, lr=0.01, momentum=0.9, weight_decay=0., epoch=5, loc=0, scale=0.0001)

  # test0
  ni, ci, hi, wi = 2, 3, 44, 55
  hf, wf = 6, 7
  co = 8
  hs, ws = 4, 5
  paddings = 'VALID'
  ho = get_output_dim(hi, 0, hf, 0, hs, paddings)
  wo = get_output_dim(wi, 0, wf, 0, ws, paddings)

  shape_input = [ni, ci, hi, wi]
  shape_filter = [ci, co, hf, wf]
  strides = [hs, ws]
  shape_output = [ni, 
                  co, 
                  ho, 
                  wo]
  sm_device = 'cpu'
  base_device = 'cpu'

  testbase.test_case(shape_input=shape_input, shape_filter=shape_filter, shape_output=shape_output, strides=strides, paddings=paddings, sm_device=sm_device, base_device=base_device)

  # test1
  ni, ci, hi, wi = 2, 3, 44, 55
  hf, wf = 6, 7
  co = 8
  hs, ws = 4, 5
  paddings = 'SAME'
  ho = get_output_dim(hi, 0, hf, 0, hs, paddings)
  wo = get_output_dim(wi, 0, wf, 0, ws, paddings)

  shape_input = [ni, ci, hi, wi]
  shape_filter = [ci, co, hf, wf]
  strides = [hs, ws]
  shape_output = [ni, 
                  co, 
                  ho, 
                  wo]
  sm_device = 'cpu'
  base_device = 'cpu'

  testbase.test_case(shape_input=shape_input, shape_filter=shape_filter, shape_output=shape_output, strides=strides, paddings=paddings, sm_device=sm_device, base_device=base_device)

  # test2
  ni, ci, hi, wi = 2, 512, 112, 112
  hf, wf = 3, 3
  co = 256
  hs, ws = 2, 2
  paddings = 'SAME'

  ho = get_output_dim(hi, 0, hf, 0, hs, paddings)
  wo = get_output_dim(wi, 0, wf, 0, ws, paddings)
  shape_input = [ni, ci, hi, wi]
  shape_filter = [ci, co, hf, wf]
  strides = [hs, ws]
  shape_output = [ni, 
                  co, 
                  ho, 
                  wo]
  sm_device = 'cpu'
  base_device = 'cpu'

  testbase.test_case(shape_input=shape_input, shape_filter=shape_filter, shape_output=shape_output, strides=strides, paddings=paddings, sm_device=sm_device, base_device=base_device)

  # test3
  ni, ci, hi, wi = 2, 512, 112, 112
  hf, wf = 3, 3
  co = 256
  hs, ws = 2, 2
  paddings = 'VALID'

  ho = get_output_dim(hi, 0, hf, 0, hs, paddings)
  wo = get_output_dim(wi, 0, wf, 0, ws, paddings)
  shape_input = [ni, ci, hi, wi]
  shape_filter = [ci, co, hf, wf]
  strides = [hs, ws]
  shape_output = [ni, 
                  co, 
                  ho, 
                  wo]
  sm_device = 'cpu'
  base_device = 'cpu'

  testbase.test_case(shape_input=shape_input, shape_filter=shape_filter, shape_output=shape_output, strides=strides, paddings=paddings, sm_device=sm_device, base_device=base_device)
