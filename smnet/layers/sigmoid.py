# Copyright (c) 2020 smarsu. All Rights Reserved.

import glog
import numpy as np

from . import _math_utils as math_utils
from ..layer import Layer


class Sigmoid(Layer):
  def __init__(self, a, name):
    super(Sigmoid, self).__init__(name=name)
    self._setup(a)

  
  def _setup(self, a):
    self.a = self._to_tensor(a)
    self.res = self._res_tensor([self.a])

  
  def forward(self):
    self.res.feed(math_utils.sigmoid(self.a.data))


  def backward(self):
    if self.a.need_grad:
      self.a.feed_grad(self.res.grad * self.res.data * (1 - self.res.data))


def sigmoid(a, name=None, device='cpu'):
  layer = Sigmoid(a, name)
  layer.forward()


  glog.info('Run {} Sigmoid Layer ... <{}> -> <{}>'.format(
    device, layer.a.shape, layer.res.shape))
  return layer.res
