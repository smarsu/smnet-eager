# Copyright (c) 2020 smarsu. All Rights Reserved.

import glog
import ctypes
import numpy as np

from ..layer import Layer
from . import _math_utils as  math_utils
from ..third_party import nvarray as nv


class Reshape(Layer):
  def __init__(self, x, shape, name=None):
    super(Reshape, self).__init__(name=name)
    self._setup(x, shape)

  
  def _setup(self, x, shape):
    self.x = self._to_tensor(x)
    self.res = self._res_tensor([self.x])

    self.shape = math_utils.infer_shape(self.x.shape, shape)


  def forward(self):
    self.res.feed(self.x.data.reshape(self.shape))

  
  def backward(self):
    if self.x.need_grad:
      self.x.feed_grad(self.res.grad)


class GpuReshape(Reshape):
  def __init__(self, x, shape, name=None):
    super(GpuReshape, self).__init__(x, shape, name)


  def forward(self):
    self.res.reshape(self.shape)
    nv.libsmnv.Assign(self.x.size, self.x.gpu, ctypes.c_float(0), self.res.gpu)


  def backward(self):
    if self.x.need_grad:
      nv.libsmnv.Assign(self.x.size, 
                        self.res.gpu_grad, 
                        ctypes.c_float(1) if self.x._grad_seted else ctypes.c_float(0), 
                        self.x.gpu_grad)
      self.x._grad_seted = True


def reshape(x, shape, name=None, device='gpu'):
  if device == 'gpu':
    layer = GpuReshape(x, shape, name)
  else:
    layer = Reshape(x, shape, name)

  layer.forward()
  glog.info('Run {} Reshape layer ... <{}> -> <{}>'.format(
    device, layer.x.shape, layer.res.shape))
  return layer.res
