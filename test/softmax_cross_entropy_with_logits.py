# Copyright (c) 2020 smarsu. All Rights Reserved.

import numpy as np
import smnet as sm
import tensorflow as tf

from _base import TestBase


def sm_func(a, b):
  a = sm.Tensor(a, dtype=np.float32, name='a')
  b = sm.Variable(b, dtype=np.float32, name='b')
  a1 = sm.transpose(a, (0, 2, 3, 1))
  b1 = sm.transpose(b, (0, 2, 3, 1))
  y = sm.softmax_cross_entropy_with_logits(labels=a1, logits=b1)
  y = sm.transpose(y, (0, 3, 1, 2))
  return y, (a.data, b.data)


def gt_func(a, b):
  a = tf.constant(a, dtype=tf.float32)
  b = tf.Variable(b, dtype=tf.float32)
  # y = tf.nn.softmax_cross_entropy_with_logits_v2(labels=a, logits=b)
  a = tf.transpose(a, (0, 2, 3, 1))
  b = tf.transpose(b, (0, 2, 3, 1))
  y = -a * tf.log(tf.nn.softmax(b))
  y = tf.transpose(y, (0, 3, 1, 2))
  return y, tuple()


def to_inputs(shape_a, shape_b, **params):
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

  return a, b


if __name__ == '__main__':
  testbase = TestBase('Sub', sm_func, gt_func, to_inputs, lr=1, momentum=0., weight_decay=0., epoch=1)

  # test0
  shape_a = (1, 12, 13, 10)
  shape_b = (1, 12, 13, 10)
  sm_device = 'cpu'
  base_device = 'cpu'

  testbase.test_case(shape_a=shape_a, shape_b=shape_b, sm_device=sm_device, base_device=base_device)
