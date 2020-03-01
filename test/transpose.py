# Copyright (c) 2020 smarsu. All Rights Reserved.

import numpy as np
import smnet as sm
import tensorflow as tf

from _base import TestBase


def sm_func(a, perm):
  a = sm.Variable(a, dtype=np.float32, name='a')
  y = sm.transpose(a, perm)
  return y, (a.data, perm)


def gt_func(a, perm):
  a = tf.Variable(a, dtype=tf.float32)
  y = tf.transpose(a, perm)
  return y, tuple()


def to_inputs(shape_a, perm, **params):
  loc = params['loc']
  scale = params['scale']

  a = np.random.normal(loc=loc, scale=scale, size=shape_a)

  return a, perm


if __name__ == '__main__':
  testbase = TestBase('Transpose', sm_func, gt_func, to_inputs, momentum=0.)

  # test0
  shape_a = (1, 2, 69, 58)
  perm = (0, 2, 3, 1)
  sm_device = 'cpu'
  base_device = 'cpu'

  testbase.test_case(shape_a=shape_a, perm=perm, sm_device=sm_device, base_device=base_device)
