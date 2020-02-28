# Copyright (c) 2020 smarsu. All Rights Reserved.

import numpy as np
import smnet as sm
import tensorflow as tf

from _base import TestBase


def sm_func(inputs,
            mean, 
            variance, 
            offset, 
            scale,
            axis,
            momentum,
            epsilon,
            training):
  x = sm.Variable(inputs, dtype=np.float32, name='x')
  mean = sm.Variable(mean, dtype=np.float32, name='mean')
  variance = sm.Variable(variance, dtype=np.float32, name='variable')
  offset = sm.Variable(offset, dtype=np.float32, name='offset')
  scale = sm.Variable(scale, dtype=np.float32, name='scale')
  y = sm.batch_normalization(x,
                             mean, 
                             variance, 
                             offset,
                             scale,
                             axis,
                             momentum,
                             epsilon,
                             training)
  return y, (x.data, mean.data, variance.data, offset.data, scale.data, axis, momentum, epsilon, training)


def gt_func(inputs,
            mean, 
            variance, 
            offset, 
            scale,
            axis,
            momentum,
            epsilon,
            training):
  x = tf.Variable(inputs, dtype=tf.float32)
  y = tf.layers.batch_normalization(
    x,
    axis=axis,
    momentum=momentum,
    epsilon=epsilon,
    training=training)
  return y, tuple()


def to_inputs(shape_x, axis, training, **params):
  loc = params['loc']
  scale = params['scale']

  inputs = np.random.normal(loc=loc, scale=scale, size=shape_x)

  shape = [1] * len(shape_x)
  shape[axis] = shape_x[axis]
  mean = np.zeros(shape, dtype=np.float32)
  variance = np.ones(shape, dtype=np.float32)
  offset = np.zeros(shape, dtype=np.float32)
  scale = np.ones(shape, dtype=np.float32)
  momentum = 0.99
  epsilon = 1e-3

  return (inputs,
          mean, 
          variance, 
          offset, 
          scale,
          axis,
          momentum,
          epsilon,
          training)


if __name__ == '__main__':
  testbase = TestBase('BatchNorm', sm_func, gt_func, to_inputs, epoch=5)

  # test0
  shape_x = (1, 64, 112, 112)
  axis = 1
  training = True

  sm_device = 'cpu'
  base_device = 'cpu'

  testbase.test_case(shape_x=shape_x, axis=axis, training=training, sm_device=sm_device, base_device=base_device)

  # test0
  shape_x = (1, 64, 112, 112)
  axis = -1
  training = True

  sm_device = 'cpu'
  base_device = 'cpu'

  testbase.test_case(shape_x=shape_x, axis=axis, training=training, sm_device=sm_device, base_device=base_device)

