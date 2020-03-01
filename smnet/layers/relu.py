# Copyright (c) 2020 smarsu. All Rights Reserved.

import numpy as np

from ..layer import Layer


class Relu(Layer):
  def __init__(self, a, name):
    super(Relu, self).__init__(name=name)
    self._setup(a)

  
  def _setup(self, a):
    self.a = self._to_tensor(a)
    self.res = self._res_tensor([self.a])

  
  def forward(self):
    self.res.feed(np.maximum(self.a.data, 0))


  def backward(self):
    keep = self.a.data > 0
    self.a.feed_grad(self.res.grad * keep)


def relu(a, name=None):
  layer = Relu(a, name)
  layer.forward()
  return layer.res
