# Copyright (c) 2020 smarsu. All Rights Reserved.

import glog
import numpy as np

from ..layer import Layer

from ..third_party import nvarray as nv


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
    if self.a.need_grad:
      self.a.feed_grad(self.res.grad * 2 * self.a.data)


def square(a, name=None, device='cpu'):
  if nv.with_cuda is True:
    device = 'gpu'

  glog.info('Run {} Square Layer ... <{}> -> <{}>'.format(
    device, np.array(a).shape, np.array(a).shape))

  if device == 'gpu':
    return a * a

  layer = Square(a, name)
  layer.forward()
  return layer.res
