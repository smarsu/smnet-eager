# Copyright (c) 2020 smarsu. All Rights Reserved.

import numpy as np
import smnet as sm
import tensorflow as tf

from _base import TestBase


def sm_func(a):
  y = sm.relu(a)
  return y, ()


def gt_func(a):
  y = tf.nn.relu(a)
  return y, tuple()


def to_inputs(shape_a, **params):
  loc = params['loc']
  scale = params['scale']

  a = np.random.normal(loc=loc, scale=scale, size=shape_a)

  return (sm.Variable(a, dtype=np.float32), ), \
         (tf.Variable(a, dtype=tf.float32), )


if __name__ == '__main__':
  testbase = TestBase('Relu', sm_func, gt_func, to_inputs, momentum=0.9)

  # test0
  shape_a = (32, 10)
  sm_device = 'cpu'
  base_device = 'cpu'

  testbase.test_case(shape_a=shape_a, sm_device=sm_device, base_device=base_device)

  # test1
  shape_a = (32, 256)
  sm_device = 'cpu'
  base_device = 'cpu'
  
  testbase.test_case(shape_a=shape_a, sm_device=sm_device, base_device=base_device)

  # test2 mtcnn
  shape_a = (1, 10, 222, 222)
  sm_device = 'cpu'
  base_device = 'cpu'
  
  testbase.test_case(shape_a=shape_a, sm_device=sm_device, base_device=base_device)

