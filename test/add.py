# Copyright (c) 2020 smarsu. All Rights Reserved.

import numpy as np
import smnet as sm
import tensorflow as tf

from _base import TestBase


def sm_func(a, b):
  a = sm.Variable(a, dtype=np.float32, name='a')
  b = sm.Variable(b, dtype=np.float32, name='b')
  y = a + b
  return y, (a.data, b.data)


def gt_func(a, b):
  a = tf.Variable(a, dtype=tf.float32)
  b = tf.Variable(b, dtype=tf.float32)
  y = a + b
  return y, tuple()


def to_inputs(shape_a, shape_b, **params):
  loc = params['loc']
  scale = params['scale']

  a = np.random.normal(loc=loc, scale=scale, size=shape_a)
  b = np.random.normal(loc=loc, scale=scale, size=shape_b)

  return a, b


if __name__ == '__main__':
  testbase = TestBase('Add', sm_func, gt_func, to_inputs, momentum=0.)

  # test0
  shape_a = (32, 10)
  shape_b = (10, )
  sm_device = 'cpu'
  base_device = 'cpu'
  
  testbase.test_case(shape_a=shape_a, shape_b=shape_b, sm_device=sm_device, base_device=base_device)

  # test1
  shape_a = (32, 256)
  shape_b = (256, )
  sm_device = 'cpu'
  base_device = 'cpu'
  
  testbase.test_case(shape_a=shape_a, shape_b=shape_b, sm_device=sm_device, base_device=base_device)

  # test2
  shape_a = (1, )
  shape_b = (1, )
  sm_device = 'cpu'
  base_device = 'cpu'

  testbase.test_case(shape_a=shape_a, shape_b=shape_b, sm_device=sm_device, base_device=base_device)

  # test3 mtcnn
  shape_a = (1, 10, 222, 222)
  shape_b = (1, 10, 1, 1)
  sm_device = 'cpu'
  base_device = 'cpu'

  testbase.test_case(shape_a=shape_a, shape_b=shape_b, sm_device=sm_device, base_device=base_device)

