# Copyright (c) 2020 smarsu. All Rights Reserved.

import numpy as np
import smnet as sm
import tensorflow as tf

from _base import TestBase


def sm_func(a, b):
  sm.reset()
  y = a - b
  return y, ()


def gt_func(a, b):
  y = a - b
  return y, tuple()


def to_inputs(shape_a, shape_b, **params):
  loc = params['loc']
  scale = params['scale']

  a = np.random.normal(loc=loc, scale=scale, size=shape_a)
  b = np.random.normal(loc=loc, scale=scale, size=shape_b)

  return (sm.Variable(a, dtype=np.float32), sm.Variable(b, dtype=np.float32)), \
         (tf.Variable(a, dtype=tf.float32), tf.Variable(b, dtype=tf.float32))


if __name__ == '__main__':
  testbase = TestBase('Sub', sm_func, gt_func, to_inputs, momentum=0, weight_decay=0)

  # test0
  shape_a = (32, 10, 112, 112)
  shape_b = (32, 10, 112, 112)
  sm_device = 'cpu'
  base_device = 'cpu'

  testbase.test_case(shape_a=shape_a, shape_b=shape_b, sm_device=sm_device, base_device=base_device)
