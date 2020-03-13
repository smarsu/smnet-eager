# Copyright (c) 2020 smarsu. All Right Reserved.

import glog
import numpy as np

from ..layer import Layer


class Concat(Layer):
  def __init__(self, values, axis, name=None):
    super(Concat, self).__init__(name=name)
    self._setup(values, axis)

  
  def _setup(self, values, axis):
    self.values = [self._to_tensor(value) for value in values]
    self.res = self._res_tensor(self.values)

    self.axis = axis

  
  def forward(self):
    self.res.feed(np.concatenate([tensor.data for tensor in self.values], self.axis))

  
  def backward(self):
    size_splits = [tensor.shape[self.axis] for tensor in self.values]
    splt = np.split(self.res.grad, size_splits, self.axis)[:-1]

    for tensor, value in zip(self.values, splt):
      if tensor.need_grad:
        tensor.feed_grad(value)


def concat(values, axis=-1, name=None):
  layer = Concat(values, axis, name)
  layer.forward()

  glog.info('Run {} Concat Layer ... <{}> -> <{}>'.format(
    device, [v.shape for v in values], layer.res.shape))
  return layer.res
