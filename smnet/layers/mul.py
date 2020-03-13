# Copyright (c) 2020 smarsu. All Rights Reserved.

import glog
import numpy as np

from . import _math_utils as math_utils
from .elementwise import GpuBinaryElementwise
from ..layer import Layer

class Mul(Layer):
  def __init__(self, a, b, name=None):
    super(Mul, self).__init__(name=name)
    self._setup(a, b)

  
  def _setup(self, a, b):
    self.a = self._to_tensor(a)
    self.b = self._to_tensor(b)
    self.res = self._res_tensor([self.a, self.b])

  
  def forward(self):
    self.res.feed(self.a.data * self.b.data)

  
  def backward(self):    
    if self.a.need_grad:
      axis = math_utils.get_reduce_axis(self.a.shape, self.res.shape)
      grad = np.sum(self.res.grad * self.b.data, axis=axis, keepdims=True)
      grad = grad.reshape(self.a.shape)
      self.a.feed_grad(grad)

    if self.b.need_grad:
      axis = math_utils.get_reduce_axis(self.b.shape, self.res.shape)
      grad = np.sum(self.res.grad * self.a.data, axis=axis, keepdims=True)
      grad = grad.reshape(self.b.shape)
      self.b.feed_grad(grad)


def mul(a, b, name=None, device='gpu'):
  if device == 'gpu':
    layer = GpuBinaryElementwise(a, b, 'Mul', name)
  else:
    layer = Mul(a, b, name)

  layer.forward()

  glog.info('Run {} Mul Layer ... <{}, {}> -> <{}>'.format(
    device, layer.a.shape, layer.b.shape, layer.res.shape))
  return layer.res

