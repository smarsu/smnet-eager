# Copyright (c) 2020 smarsu. All Rights Reserved.

import glog
import numpy as np

from ..layer import Layer


class Matmul(Layer):
  def __init__(self, a, b, name=None):
    super(Matmul, self).__init__(name=name)
    self._setup(a, b)

  
  def _setup(self, a, b):
    self.a = self._to_tensor(a)
    self.b = self._to_tensor(b)
    self.res = self._res_tensor([self.a, self.b])

  
  def forward(self):
    self.res.feed(np.matmul(self.a.data, self.b.data))

  
  def backward(self):
    if self.a.need_grad:
      self.a.feed_grad(np.matmul(self.res.grad, self.b.data.T))
    if self.b.need_grad:
      self.b.feed_grad(np.matmul(self.a.data.T, self.res.grad))


def matmul(a, b, name=None, device='cpu'):
  layer = Matmul(a, b, name)
  layer.forward()

  glog.info('Run {} Matmul Layer ... <{}, {}> -> <{}>'.format(
    device, a.shape, b.shape, layer.res.shape))
  return layer.res
