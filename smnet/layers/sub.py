# Copyright (c) 2020 smarsu. All Rights Reserved.

import glog
import numpy as np

from . import _math_utils as math_utils
from .elementwise import GpuBinaryElementwise
from ..layer import Layer

class Sub(Layer):
  def __init__(self, a, b, name=None):
    super(Sub, self).__init__(name=name)
    self._setup(a, b)

  
  def _setup(self, a, b):
    self.a = self._to_tensor(a)
    self.b = self._to_tensor(b)
    self.res = self._res_tensor([self.a, self.b])

  
  def forward(self):
    self.res.feed(self.a.data - self.b.data)

  
  def backward(self): 
    if self.a.need_grad:   
      axis = math_utils.get_reduce_axis(self.a.shape, self.res.shape)
      grad = np.sum(self.res.grad, axis=axis, keepdims=True)
      grad = grad.reshape(self.a.shape)
      self.a.feed_grad(grad)

    if self.b.need_grad:
      axis = math_utils.get_reduce_axis(self.b.shape, self.res.shape)
      grad = np.sum(self.res.grad, axis=axis, keepdims=True)
      grad = grad.reshape(self.b.shape)
      self.b.feed_grad(-grad)


def sub(a, b, name=None, device='gpu'):
  if device == 'gpu':
    layer = GpuBinaryElementwise(a, b, 'Sub', name)
  else:
    layer = Sub(a, b, name)

  layer.forward()

  glog.info('Run {} Sub Layer ... <{}, {}> -> <{}>'.format(
    device, a.shape, b.shape, layer.res.shape))
  return layer.res
