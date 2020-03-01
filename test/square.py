# Copyright (c) 2020 smarsu. All Rights Reserved.

import numpy as np
import smnet as sm
import tensorflow as tf

from _base import TestBase


def sm_func(a):
  a = sm.Variable(a, dtype=np.float32, name='a')
  y = sm.square(a)
  return y, (a.data, )


def gt_func(a):
  a = tf.Variable(a, dtype=tf.float32)
  y = a * a
  return y, tuple()


def to_inputs(shape_a, **params):
  loc = params['loc']
  scale = params['scale']

  a = np.random.normal(loc=loc, scale=scale, size=shape_a)

  return (a, )


if __name__ == '__main__':
  testbase = TestBase('Square', sm_func, gt_func, to_inputs, lr=0.1)

  # test0
  shape_a = (32, 10)
  sm_device = 'cpu'
  base_device = 'cpu'
  
  testbase.test_case(shape_a=shape_a, sm_device=sm_device, base_device=base_device)
