# Copyright (c) 2020 smarsu. All Rights Reserved.

import numpy as np

from ..layer import Layer


class Square(Layer):
  def __init__(self, a, name):
    super(Square, self).__init__(name=name)
    self._setup(a)

  
  def _setup(self, a):
    self.a = self._to_tensor(a)
    self.res = self._res_tensor([self.a])

  
  def forward(self):
    self.res.feed(self.a.data * self.a.data)


  def backward(self):
    self.a.feed_grad(self.res.grad * 2 * self.a.data)


def square(a, name=None):
  layer = Square(a, name)
  layer.forward()
  return layer.res
