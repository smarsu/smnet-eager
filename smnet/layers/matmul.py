# Copyright (c) 2020 smarsu. All Rights Reserved.

import glog
import numpy as np

from ..layer import Layer

from ..third_party import nvarray as nv


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
  if nv.with_cuda is True:
    device = 'gpu'
  
  layer = Matmul(a, b, name)
  layer.forward()

  glog.info('Run {} Matmul Layer ... <{}, {}> -> <{}>'.format(
    device, layer.a.shape, layer.b.shape, layer.res.shape))
  return layer.res
