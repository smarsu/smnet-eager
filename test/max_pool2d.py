# Copyright (c) 2020 smarsu. All Rights Reserved.

import numpy as np
import smnet as sm
import tensorflow as tf

from _base import TestBase


def sm_func(value, ksize, strides, padding):
  y = sm.max_pool2d(value, ksize, strides, padding)
  return y, ()


def gt_func(value, ksize, strides, padding):
  value = tf.transpose(value, [0, 2, 3, 1])

  ksize = [1] + ksize + [1]
  strides = [1] + strides + [1]

  y = tf.nn.max_pool(value, ksize, strides, padding)
  y = tf.transpose(y, [0, 3, 1, 2])
  return y, tuple()


def to_inputs(shape_value, ksize, strides, padding, **params):
  loc = params['loc']
  scale = params['scale']

  value = np.random.normal(loc=loc, scale=scale, size=shape_value)

  return (sm.Variable(value, dtype=np.float32), ksize, strides, padding), \
         (tf.Variable(value, dtype=tf.float32), ksize, strides, padding),


if __name__ == '__main__':
  testbase = TestBase('Maxpool2D', sm_func, gt_func, to_inputs, lr=1., momentum=0.9, weight_decay=0., epoch=50)

  # test0
  shape_value = [1, 1, 1, 1]
  ksize = [1, 1]
  strides = [1, 1]
  padding = 'VALID'
  sm_device = 'cpu'
  base_device = 'cpu'

  testbase.test_case(shape_value=shape_value, ksize=ksize, strides=strides, padding=padding, sm_device=sm_device, base_device=base_device)

  # test1
  shape_value = [2, 16, 32, 64]
  ksize = [3, 3]
  strides = [1, 1]
  padding = 'VALID'
  sm_device = 'cpu'
  base_device = 'cpu'

  testbase.test_case(shape_value=shape_value, ksize=ksize, strides=strides, padding=padding, sm_device=sm_device, base_device=base_device)

  # test2
  shape_value = [2, 3, 8, 32]
  ksize = [3, 3]
  strides = [2, 2]
  padding = 'VALID'
  sm_device = 'cpu'
  base_device = 'cpu'

  testbase.test_case(shape_value=shape_value, ksize=ksize, strides=strides, padding=padding, sm_device=sm_device, base_device=base_device)

  # test3
  shape_value = [2, 3, 8, 32]
  ksize = [3, 3]
  strides = [2, 2]
  padding = 'SAME'
  sm_device = 'cpu'
  base_device = 'cpu'

  testbase.test_case(shape_value=shape_value, ksize=ksize, strides=strides, padding=padding, sm_device=sm_device, base_device=base_device)

  # test4 mtcnn
  shape_value = [1, 10, 222, 222]
  ksize = [3, 3]
  strides = [2, 2]
  padding = 'SAME'
  sm_device = 'cpu'
  base_device = 'cpu'

  testbase.test_case(shape_value=shape_value, ksize=ksize, strides=strides, padding=padding, sm_device=sm_device, base_device=base_device)

