# Copyright (c) 2020 smarsu. All Rights Reserved.

import glog
import ctypes
import numpy as np

from . import _math_utils as math_utils
from ..layer import Layer

from ..third_party import nvarray
if nvarray.with_cuda is True:
  from .elementwise import GpuBinaryElementwise

from ..third_party import cnarray as cn


class Add(Layer):
  def __init__(self, a, b, name=None):
    super(Add, self).__init__(name=name)
    self._setup(a, b)

  
  def _setup(self, a, b):
    self.a = self._to_tensor(a)
    self.b = self._to_tensor(b)
    self.res = self._res_tensor([self.a, self.b])

  
  def forward(self):
    self.res.feed(self.a.data + self.b.data)

  
  def backward(self):  
    if self.a.need_grad:  
      axis = math_utils.get_reduce_axis(self.a.shape, self.res.shape)
      grad = np.sum(self.res.grad, axis=axis, keepdims=True)
      grad = grad.reshape(self.a.shape)
      self.a.feed_grad(grad)

    if self.b.need_grad:
      axis = math_utils.get_reduce_axis(self.b.shape, self.res.shape)
      grad = np.sum(self.res.grad, axis=axis, keepdims=True)
      grad = grad.reshape(self.b.shape)
      self.b.feed_grad(grad)


class MluAdd(Add):
  def __init__(self, a, b, name):
    super(MluAdd, self).__init__(a, b, name)

  
  def forward(self):
    self.res.reshape(self.a.shape)
    cn.libsmcn.Add(self.a.mlu, self.b.mlu, self.res.mlu, ctypes.c_size_t(self.res.size))


def add(a, b, name=None, device='cpu'):
  if nvarray.with_cuda is True:
    device = 'gpu'
  elif cn.with_mlu is True:
    device = 'mlu'

  if device == 'gpu':
    layer = GpuBinaryElementwise(a, b, 'Add', name)
  elif device == 'mlu':
    layer = MluAdd(a, b, name)
  else:
    layer = Add(a, b, name)

  layer.forward()
  glog.info('Run {} Add Layer ... <{}, {}> -> <{}>'.format(
    device, layer.a.shape, layer.b.shape, layer.res.shape))
  return layer.res
