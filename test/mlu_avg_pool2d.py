# Copyright (c) 2020 smarsu. All Rights Reserved.

import numpy as np
import smnet as sm
import tensorflow as tf

from _base import TestBase


def sm_func(value, ksize, strides, padding):
  sm.reset()
  y = sm.avg_pool2d(value, ksize, strides, padding)
  return y, ()


def gt_func(value, ksize, strides, padding):
  ksize = [1] + ksize + [1]
  strides = [1] + strides + [1]

  y = tf.nn.avg_pool(value, ksize, strides, padding)
  return y, tuple()


def to_inputs(shape_value, ksize, strides, padding, **params):
  loc = params['loc']
  scale = params['scale']

  value = np.random.normal(loc=loc, scale=scale, size=shape_value)

  return (sm.Variable(value, dtype=np.float32), ksize, strides, padding), \
         (tf.Variable(value, dtype=tf.float32), ksize, strides, padding)


if __name__ == '__main__':
  testbase = TestBase('Avgpool2D', sm_func, gt_func, to_inputs, momentum=0., weight_decay=0., epoch=50, bp=False)

  # test0
  shape_value = [1, 1, 1, 1]
  ksize = [1, 1]
  strides = [1, 1]
  padding = 'VALID'
  sm_device = 'cpu'
  base_device = 'cpu'

  testbase.test_case(shape_value=shape_value, ksize=ksize, strides=strides, padding=padding, sm_device=sm_device, base_device=base_device)

  # test1
  shape_value = [2, 32, 64, 16]
  ksize = [3, 3]
  strides = [1, 1]
  padding = 'VALID'
  sm_device = 'cpu'
  base_device = 'cpu'

  testbase.test_case(shape_value=shape_value, ksize=ksize, strides=strides, padding=padding, sm_device=sm_device, base_device=base_device)

  # test2
  shape_value = [2, 8, 32, 3]
  ksize = [3, 3]
  strides = [2, 2]
  padding = 'VALID'
  sm_device = 'cpu'
  base_device = 'cpu'

  testbase.test_case(shape_value=shape_value, ksize=ksize, strides=strides, padding=padding, sm_device=sm_device, base_device=base_device)

  # test3
  shape_value = [2, 8, 32, 3]
  ksize = [3, 3]
  strides = [2, 2]
  padding = 'SAME'
  sm_device = 'cpu'
  base_device = 'cpu'

  testbase.test_case(shape_value=shape_value, ksize=ksize, strides=strides, padding=padding, sm_device=sm_device, base_device=base_device)
