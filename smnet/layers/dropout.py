# Copyright (c) 2020 smarsu. All Rights Reserved.

import glog
import numpy as np

from ..layer import Layer

from ..third_party import nvarray as nv


class Dropout(Layer):
  def __init__(self, x, keep_prob, name=None):
    super(Dropout, self).__init__(name=name)
    self.__setup(x, keep_prob)

  
  def __setup(self, x, keep_prob):
    self.x = self._to_tensor(x)
    self.res = self._res_tensor([self.x])

    self.keep_prob = keep_prob

  
  def forward(self):
    size = self.x.size
    one_size = int(round(size * self.keep_prob))
    zero_size = size - one_size

    ones = np.ones([one_size], dtype=self.x.dtype) * 1 / self.keep_prob
    zeros = np.zeros([zero_size], dtype=self.x.dtype)

    self.mask = np.concatenate([ones, zeros], 0)
    np.random.shuffle(self.mask)

    self.mask = self.mask.reshape(self.x.shape)

    self.res.feed(self.x.data * self.mask)

  
  def backward(self):
    if self.x.need_grad:
      self.x.feed_grad(self.res.grad * self.mask)


def dropout(x, keep_prob, name=None, device='cpu'):
  if nv.with_cuda is True:
    device = 'gpu'

  layer = Dropout(x, keep_prob, name)
  layer.forward()

  glog.info('Run {} Dropout Layer ... <{}> -> <{}>'.format(
    device, layer.x.shape, layer.res.shape))
  return layer.res
