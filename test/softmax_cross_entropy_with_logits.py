# Copyright (c) 2020 smarsu. All Rights Reserved.

import numpy as np
import smnet as sm
import tensorflow as tf

from _base import TestBase


def sm_func(a, b, axis):
  y = sm.softmax_cross_entropy_with_logits(labels=a, logits=b, axis=axis)
  return y, ()


def gt_func(a, b, axis):
  # y = tf.nn.softmax_cross_entropy_with_logits_v2(labels=a, logits=b, dim=axis)
  b = b - tf.reduce_max(b, axis=axis, keep_dims=True)
  b = tf.exp(b)
  b = tf.log(b / tf.reduce_sum(b, axis=axis, keep_dims=True))
  y = -a * b
  return y, tuple()


def to_inputs(shape_a, shape_b, axis, **params):
  loc = params['loc']
  scale = params['scale']

  # def one_hot(label, num=10):
  #   zeros = np.zeros([num, num])
  #   for i in range(num):
  #     zeros[i][i] = 1
  #   return zeros[label]

  # a = one_hot(np.random.randint(0, shape_a[-1], size=shape_a[:-1]), shape_a[-1])
  a = np.random.normal(loc=loc, scale=scale, size=shape_a)
  b = np.random.normal(loc=loc, scale=scale, size=shape_b)

  return (sm.Variable(a, dtype=np.float32), sm.Variable(b, dtype=np.float32), axis), \
         (tf.Variable(a, dtype=tf.float32), tf.Variable(b, dtype=tf.float32), axis), \


if __name__ == '__main__':
  testbase = TestBase('Sub', sm_func, gt_func, to_inputs, lr=1, momentum=0., weight_decay=0., epoch=1)

  # test0
  shape_a = (1, 12, 13, 10)
  shape_b = (1, 12, 13, 10)
  axis = 1
  sm_device = 'cpu'
  base_device = 'cpu'

  testbase.test_case(shape_a=shape_a, shape_b=shape_b, axis=axis, sm_device=sm_device, base_device=base_device)

  # test1
  shape_a = (1, 12, 13, 10)
  shape_b = (1, 12, 13, 10)
  axis = 3
  sm_device = 'cpu'
  base_device = 'cpu'

  testbase.test_case(shape_a=shape_a, shape_b=shape_b, axis=axis, sm_device=sm_device, base_device=base_device)

  # test2
  shape_a = (1, 12, 13, 10)
  shape_b = (1, 12, 13, 10)
  axis = -1
  sm_device = 'cpu'
  base_device = 'cpu'

  testbase.test_case(shape_a=shape_a, shape_b=shape_b, axis=axis, sm_device=sm_device, base_device=base_device)

  # test3
  shape_a = (32, 10)
  shape_b = (32, 10)
  axis = -1
  sm_device = 'cpu'
  base_device = 'cpu'

  testbase.test_case(shape_a=shape_a, shape_b=shape_b, axis=axis, sm_device=sm_device, base_device=base_device)

