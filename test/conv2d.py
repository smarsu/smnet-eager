# Copyright (c) 2020 smarsu. All Rights Reserved.

import numpy as np
import smnet as sm
import tensorflow as tf
np.random.seed(196)

from _base import TestBase


def sm_func(input, filter, bias, strides, paddings, dilations):
  sm.reset()
  y = sm.conv2d(input, filter, strides, paddings, dilations, bias)
  return y, ()


def gt_func(input, filter, bias, strides, paddings, dilations):
  input = tf.transpose(input, [0, 2, 3, 1])
  filter = tf.transpose(filter, [2, 3, 1, 0])

  strides = [1] + strides + [1]
  dilations = [1] + dilations + [1]

  y = tf.nn.conv2d(input, filter, strides, paddings, dilations=dilations)
  y = tf.transpose(y, [0, 3, 1, 2])
  y += bias
  return y, tuple()


def to_inputs(shape_input, shape_filter, strides, paddings, dilations, **params):
  loc = params['loc']
  scale = params['scale']

  input = np.random.normal(loc=loc, scale=scale, size=shape_input)
  filter = np.random.normal(loc=loc, scale=scale, size=shape_filter)
  bias = np.random.normal(loc=loc, scale=scale, size=[1, shape_filter[0], 1, 1])

  return (sm.Variable(input, dtype=np.float32), sm.Variable(filter, dtype=np.float32), sm.Variable(bias, dtype=np.float32), strides, paddings, dilations), \
         (tf.Variable(input, dtype=tf.float32), tf.Variable(filter, dtype=tf.float32), tf.Variable(bias, dtype=tf.float32), strides, paddings, dilations),


if __name__ == '__main__':
  testbase = TestBase('Conv2D', sm_func, gt_func, to_inputs, lr=0.01, momentum=0.9, weight_decay=0., epoch=5)

  # test0
  shape_input = [2, 2, 4, 4]
  shape_filter = [2, 2, 2, 2]
  strides = [2, 2]
  paddings = 'VALID'
  dilations = [1, 1]
  sm_device = 'cpu'
  base_device = 'cpu'

  testbase.test_case(shape_input=shape_input, shape_filter=shape_filter, strides=strides, paddings=paddings, dilations=dilations, sm_device=sm_device, base_device=base_device)

  # test1
  shape_input = [2, 3, 8, 32]
  shape_filter = [5, 3, 4, 16]
  strides = [2, 2]
  paddings = 'VALID'
  dilations = [1, 1]
  sm_device = 'cpu'
  base_device = 'cpu'
  
  testbase.test_case(shape_input=shape_input, shape_filter=shape_filter, strides=strides, paddings=paddings, dilations=dilations, sm_device=sm_device, base_device=base_device)

  # test2
  shape_input = [2, 3, 9, 33]
  shape_filter = [5, 3, 5, 17]
  strides = [2, 2]
  paddings = 'SAME'
  dilations = [1, 1]
  sm_device = 'cpu'
  base_device = 'cpu'

  testbase.test_case(shape_input=shape_input, shape_filter=shape_filter, strides=strides, paddings=paddings, dilations=dilations, sm_device=sm_device, base_device=base_device)

  # test3 mtcnn
  shape_input = [1, 3, 224, 224]
  shape_filter = [10, 3, 3, 3]
  strides = [1, 1]
  paddings = 'SAME'
  dilations = [1, 1]
  sm_device = 'cpu'
  base_device = 'cpu'

  testbase.test_case(shape_input=shape_input, shape_filter=shape_filter, strides=strides, paddings=paddings, dilations=dilations, sm_device=sm_device, base_device=base_device)

  # test4 mtcnn
  shape_input = [1, 32, 224, 224]
  shape_filter = [16, 32, 1, 1]
  strides = [1, 1]
  paddings = 'SAME'
  dilations = [1, 1]
  sm_device = 'cpu'
  base_device = 'cpu'

  testbase.test_case(shape_input=shape_input, shape_filter=shape_filter, strides=strides, paddings=paddings, dilations=dilations, sm_device=sm_device, base_device=base_device)

