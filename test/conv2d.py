# Copyright (c) 2020 smarsu. All Rights Reserved.

import numpy as np
import smnet as sm
import tensorflow as tf

from _base import TestBase


def sm_func(input, filter, strides, paddings, dilations):
  input = sm.Variable(input, dtype=np.float32, name='a')
  filter = sm.Variable(filter, dtype=np.float32, name='b')
  y = sm.conv2d(input, filter, strides, paddings, dilations)
  return y, (input.data, filter.data, strides, paddings, dilations)


def gt_func(input, filter, strides, paddings, dilations):
  input = input.transpose(0, 2, 3, 1)
  filter = filter.transpose(2, 3, 1, 0)

  strides = [1] + strides + [1]
  dilations = [1] + dilations + [1]

  input = tf.Variable(input, dtype=tf.float32)
  filter = tf.Variable(filter, dtype=tf.float32)
  y = tf.nn.conv2d(input, filter, strides, paddings, dilations=dilations)
  y = tf.transpose(y, [0, 3, 1, 2])
  return y, tuple()


def to_inputs(shape_input, shape_filter, strides, paddings, dilations, **params):
  loc = params['loc']
  scale = params['scale']

  input = np.random.normal(loc=loc, scale=scale, size=shape_input)
  filter = np.random.normal(loc=loc, scale=scale, size=shape_filter)

  return input, filter, strides, paddings, dilations


if __name__ == '__main__':
  testbase = TestBase('Conv2D', sm_func, gt_func, to_inputs, lr=0.001, momentum=0., weight_decay=4e-5, epoch=3)

  # test0
  shape_input = [1, 1, 1, 1]
  shape_filter = [1, 1, 1, 1]
  strides = [1, 1]
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
  paddings = 'VALID'
  dilations = [1, 1]
  sm_device = 'cpu'
  base_device = 'cpu'

  testbase.test_case(shape_input=shape_input, shape_filter=shape_filter, strides=strides, paddings=paddings, dilations=dilations, sm_device=sm_device, base_device=base_device)

  # test4 mtcnn
  shape_input = [1, 32, 224, 224]
  shape_filter = [16, 32, 1, 1]
  strides = [1, 1]
  paddings = 'VALID'
  dilations = [1, 1]
  sm_device = 'cpu'
  base_device = 'cpu'

  testbase.test_case(shape_input=shape_input, shape_filter=shape_filter, strides=strides, paddings=paddings, dilations=dilations, sm_device=sm_device, base_device=base_device)

