# Copyright (c) 2020 smarsu. All Rights Reserved.

import numpy as np
import smnet as sm
import tensorflow as tf

from _base import TestBase


def sm_func(a, b):
  a = sm.Variable(a, dtype=np.float32, name='a')
  b = sm.Variable(b, dtype=np.float32, name='b')
  y = sm.matmul(a, b)
  return y, (a.data, b.data)


def gt_func(a, b):
  a = tf.Variable(a, dtype=tf.float32)
  b = tf.Variable(b, dtype=tf.float32)
  y = tf.matmul(a, b)
  return y, tuple()


def to_inputs(shape_a, shape_b, **params):
  loc = params['loc']
  scale = params['scale']

  a = np.random.normal(loc=loc, scale=scale, size=shape_a)
  b = np.random.normal(loc=loc, scale=scale, size=shape_b)

  return a, b


if __name__ == '__main__':
  testbase = TestBase('Matmul', sm_func, gt_func, to_inputs, epoch=10)

  # test0
  m = 32
  n = 256
  k = 784
  shape_a = (m, k)
  shape_b = (k, n)
  sm_device = 'cpu'
  base_device = 'cpu'

  testbase.test_case(shape_a=shape_a, shape_b=shape_b, sm_device=sm_device, base_device=base_device)

  # test1
  m = 32
  n = 10
  k = 256
  shape_a = (m, k)
  shape_b = (k, n)
  sm_device = 'cpu'
  base_device = 'cpu'
  
  testbase.test_case(shape_a=shape_a, shape_b=shape_b, sm_device=sm_device, base_device=base_device)
