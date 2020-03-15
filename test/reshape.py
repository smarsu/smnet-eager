# Copyright (c) 2020 smarsu. All Rights Reserved.

import numpy as np
import smnet as sm
import tensorflow as tf

from _base import TestBase


def sm_func(x, shape):
  sm.reset()
  y = sm.reshape(x, shape)
  return y, ()


def gt_func(x, shape):
  y = tf.reshape(x, shape)
  return y, tuple()


def to_inputs(shape_a, shape_b, **params):
  loc = params['loc']
  scale = params['scale']

  a = np.random.normal(loc=loc, scale=scale, size=shape_a)

  return (sm.Variable(a, dtype=np.float32), shape_b), \
         (tf.Variable(a, dtype=tf.float32), shape_b)


if __name__ == '__main__':
  testbase = TestBase('Add', sm_func, gt_func, to_inputs, momentum=0.9, weight_decay=0.)

  # test0
  shape_a = [1, 28, 19, 19]
  shape_b = [1, 4, 7, 19, 19]
  sm_device = 'cpu'
  base_device = 'cpu'
  
  testbase.test_case(shape_a=shape_a, shape_b=shape_b, sm_device=sm_device, base_device=base_device)

  # test1
  shape_a = [8, 28, 19, 19]
  shape_b = [8, 4, 7, 19, 19]
  sm_device = 'cpu'
  base_device = 'cpu'
  
  testbase.test_case(shape_a=shape_a, shape_b=shape_b, sm_device=sm_device, base_device=base_device)

