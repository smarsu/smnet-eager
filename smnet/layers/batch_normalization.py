# Copyright (c) 2020 smarsu. All Rights Reserved.

import glog
import numpy as np

from ..layer import Layer
from . import _math_utils as math_utils

from ..third_party import nvarray


class BatchNormalization(Layer):
  def __init__(self, 
               inputs,
               mean, 
               variance, 
               offset, 
               scale,
               axis=-1,
               momentum=0.99,
               epsilon=0.001,
               training=False,
               name=None):
    super(BatchNormalization, self).__init__(name=name)
    self._setup(inputs,
                mean, 
                variance, 
                offset, 
                scale,
                axis,
                momentum,
                epsilon,
                training)


  def _setup(self, input, mean, variance, offset, scale, axis, momentum, epsilon, training):
    self.x = self._to_tensor(input)
    self.offset = self._to_tensor(offset)
    self.scale = self._to_tensor(scale)
    self.res = self._res_tensor([self.x, self.offset, self.scale])

    self.mean = self._to_variable(mean)
    self.variance = self._to_variable(variance)
    self.axis = axis
    self.momentum = momentum
    self.epsilon = epsilon
    self.training = training

  
  def forward(self):
    if self.training:
      axis = list(range(len(self.x.shape)))
      axis.remove(math_utils.pos_axis(self.axis, len(self.x.shape)))
      axis = tuple(axis)
      self.brd_axis = axis
      mean = np.mean(self.x.data, axis=axis, keepdims=True)
      variance = np.var(self.x.data, axis=axis, keepdims=True)

      self.mean.feed(self.momentum * self.mean.data + (1 - self.momentum) * mean)
      self.variance.feed(self.momentum * self.variance.data + (1 - self.momentum) * variance)
    else:
      mean = self.mean
      variance = self.variance

    self.norm = (self.x.data - mean) / np.sqrt(variance + self.epsilon)
    self.res.feed(self.norm * self.scale.data + self.offset.data)
  

  def backward(self):
    if self.scale.need_grad:
      self.scale.feed_grad(np.sum(self.res.grad * self.norm, axis=self.brd_axis, keepdims=True))

    if self.offset.need_grad:
      self.offset.feed_grad(np.sum(self.res.grad, axis=self.brd_axis, keepdims=True))

    if self.x.need_grad:
      self.x.feed_grad(self.res.grad * self.scale.data)


def batch_normalization(inputs,
                        mean, 
                        variance, 
                        offset, 
                        scale,
                        axis=-1,
                        momentum=0.99,
                        epsilon=0.001,
                        training=False,
                        name=None,
                        device='cpu'):
  if nvarray.with_cuda is True:
    device = 'gpu'

  layer = BatchNormalization(inputs, mean, variance, offset, scale, axis, momentum, epsilon, training, name)
  layer.forward()

  glog.info('Run {} BatchNorm Layer ... <{}> -> <{}>'.format(
    device, layer.inputs.shape, layer.res.shape))
  return layer.res
