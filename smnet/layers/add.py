# Copyright (c) 2020 smarsu. All Rights Reserved.

import numpy as np

from . import _math_utils as math_utils
from ..layer import Layer

class Add(Layer):
  def __init__(self, a, b, name=None):
    super(Add, self).__init__(name=name)
    self._setup(a, b)

  
  def _setup(self, a, b):
    self.a = self._to_tensor(a)
    self.b = self._to_tensor(b)
    self.res = self._res_tensor([self.a, self.b])

  
  def forward(self):
    self.res.feed(self.a.data + self.b.data)

  
  def backward(self):    
    axis = math_utils.get_reduce_axis(self.a.shape, self.res.shape)
    grad = np.sum(self.res.grad, axis=axis, keepdims=True)
    grad = grad.reshape(self.a.shape)
    self.a.feed_grad(grad)

    axis = math_utils.get_reduce_axis(self.b.shape, self.res.shape)
    grad = np.sum(self.res.grad, axis=axis, keepdims=True)
    grad = grad.reshape(self.b.shape)
    self.b.feed_grad(grad)


def add(a, b, name=None):
  layer = Add(a, b, name)
  layer.forward()
  return layer.res
