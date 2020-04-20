# Copyright (c) 2020 smarsu. All Rights Reserved.

import numpy as np
import smnet as sm
import tensorflow as tf

from _base import TestBase


def sm_func(a, num_or_size_splits, axis):
  y = sm.split(a, num_or_size_splits, axis)
  return y[-1], ()


def gt_func(a, num_or_size_splits, axis):
  y = tf.split(a, num_or_size_splits, axis)
  return y[-1], tuple()


def to_inputs(shape_a, num_or_size_splits, axis, **params):
  loc = params['loc']
  scale = params['scale']

  a = np.random.normal(loc=loc, scale=scale, size=shape_a)

  return (sm.Variable(a, dtype=np.float32), num_or_size_splits, axis), \
         (tf.Variable(a, dtype=tf.float32), num_or_size_splits, axis)


if __name__ == '__main__':
  testbase = TestBase('Split', sm_func, gt_func, to_inputs, weight_decay=0., bp=False)

  # test0
  shape_a = (1, 5)
  num_or_size_splits = 5
  axis = -1
  sm_device = 'cpu'
  base_device = 'cpu'

  testbase.test_case(shape_a=shape_a, num_or_size_splits=num_or_size_splits, axis=axis, sm_device=sm_device, base_device=base_device)

  # test1
  shape_a = (32, 16, 16, 16)
  num_or_size_splits = (2, 4, 10)
  axis = 1
  sm_device = 'cpu'
  base_device = 'cpu'

  testbase.test_case(shape_a=shape_a, num_or_size_splits=num_or_size_splits, axis=axis, sm_device=sm_device, base_device=base_device)

