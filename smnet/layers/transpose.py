# Copyright (c) 2020 smarsu. All Rights Reserved.

import numpy as np

from ..layer import Layer


def inv_perm(perm):
  mp = {dim : indice for indice, dim in enumerate(perm)}
  return [mp[i] for i in range(len(perm))]


class Transpose(Layer):
  def __init__(self, x, perm, name=None):
    super(Transpose, self).__init__(name=name)
    self._setup(x, perm)


  def _setup(self, x, perm):
    self.x = self._to_tensor(x)
    self.res = self._res_tensor([self.x])

    self.perm = perm
    self.inv_perm = inv_perm(self.perm)


  def forward(self):
    self.res.feed(np.transpose(self.x.data, self.perm))


  def backward(self):
    self.x.feed_grad(np.transpose(self.res.grad, self.inv_perm))


def transpose(x, perm, name=None):
  layer = Transpose(x, perm, name)
  layer.forward()
  return layer.res
