# Copyright (c) 2020 smarsu. All Rights Reserved.

import glog
import numpy as np

from . import _math_utils as math_utils
from ..layer import Layer
from ..kernels.gpu import *


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


  def prepare(self):
    value_shape = self.value.shape
    dim = value_shape[self.axis]

    if isinstance(self.num_or_size_splits, int):
      self.splt = np.arange(1, self.num_or_size_splits + 1) * (dim // self.num_or_size_splits)
      self.splt = self.splt.tolist()
    else:
      self.splt = math_utils.acc(self.num_or_size_splits)


  def forward(self):
    self.prepare()

    result = np.split(self.value.data, self.splt, axis=self.axis)[:-1]
    for tensor, value in zip(self.res, result):
      tensor.feed(value)


  def backward(self):
    if self.value.need_grad:
      grads = []
      for i in range(len(self.res)):
        if self.res[i]._grad_seted:
          grads.append(self.res[i].grad)
        else:
          grads.append(np.zeros(self.res[i].shape, self.res[i].dtype))
      self.value.feed_grad(np.concatenate(grads, axis=self.axis))


class GpuSplit(Split):
  def __init__(self,
               value,
               num_or_size_splits,
               axis=0,
               name=None):
    super(GpuSplit, self).__init__(value, num_or_size_splits, axis, name)


  def __del__(self):
    del self.split_kernel

  
  def forward(self):
    self.prepare()

    splt = [0] + self.splt
    for idx in range(len(self.res)):
      shape = list(self.value.shape)
      shape[self.axis] = splt[idx + 1] - splt[idx]
      self.res[idx].reshape(shape)

    self.split_kernel = SplitKernel(self.value, self.res, self.axis, self.splt)
    self.split_kernel.forward()


  def backward(self):
    self.split_kernel.backward()


def split(value, num_or_size_splits, axis=0, name=None, device='gpu'):
  if device == 'gpu':
    layer = GpuSplit(value, num_or_size_splits, axis, name)
  else:
    layer = Split(value, num_or_size_splits, axis, name)

  layer.forward()

  glog.info('Run {} Split Layer ... <{}> -> <{}>'.format(
    device, value.shape, [tensor.shape for tensor in layer.res]))
  return layer.res
