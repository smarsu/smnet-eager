# Copyright (c) 2020 smarsu. All Rights Reserved.

import numpy as np

from . import _math_utils as math_utils
from ..layer import Layer


class Split(Layer):
  def __init__(self,
               value,
               num_or_size_splits,
               axis=0,
               name=None):
    super(Split, self).__init__(name=name)
    self._setup(value, num_or_size_splits, axis)


  def _setup(self, value, num_or_size_splits, axis):
    num_res_tensor = (num_or_size_splits 
                      if isinstance(num_or_size_splits, int) 
                      else len(num_or_size_splits))

    self.value = self._to_tensor(value)
    self.res = [self._res_tensor([self.value]) 
                for _ in range(num_res_tensor)]

    self.num_or_size_splits = num_or_size_splits
    self.axis = axis


  def forward(self):
    value_shape = self.value.shape
    dim = value_shape[self.axis]

    if isinstance(self.num_or_size_splits, int):
      self.splt = np.arange(1, self.num_or_size_splits + 1) * (dim // self.num_or_size_splits)
    else:
      self.splt = math_utils.acc(self.num_or_size_splits)

    result = np.split(self.value.data, self.splt, axis=self.axis)[:-1]
    for tensor, value in zip(self.res, result):
      tensor.feed(value)


  def backward(self):
    grads = []
    for i in range(len(self.res)):
      if self.res[i]._grad_seted:
        grads.append(self.res[i].grad)
      else:
        grads.append(np.zeros(self.res[i].shape, self.res[i].dtype))
    self.value.feed_grad(np.concatenate(grads, axis=self.axis))


def split(value, num_or_size_splits, axis=0, name=None):
  layer = Split(value, num_or_size_splits, axis, name)
  layer.forward()
  return layer.res
