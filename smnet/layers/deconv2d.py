# Copyright (c) 2020 smarsu. All Rights Reserved.

import glog
import math
import numpy as np

from . import _math_utils as math_utils
from ..blob import Tensor
from ..layer import Layer

from ..third_party import nvarray as nv
if nv.with_cuda is True:
  from ..kernels.gpu import *


def get_output_dim(input_dim, pad, filter_dim, dilation, stride):
  return 1 + (input_dim + pad - (((filter_dim - 1) * dilation) + 1)) // stride


class Deconv2D(Layer):
  def __init__(self,
               value,
               filter,
               output_shape,
               strides,
               padding='SAME',
               bias=None,
               name=None):
    super(Deconv2D, self).__init__(name=name)
    self._setup(value, filter, output_shape, strides, padding, bias)

  
  def _setup(self, value, filter, output_shape, strides, padding, bias):
    self.value = self._to_tensor(value)
    self.filter = self._to_tensor(filter)
    self.bias = bias
    input_tensors = [self.value, self.filter]
    if self.bias is not None:
      self.bias = self._to_tensor(bias)
      input_tensors.append(self.bias)
    self.res = self._res_tensor(input_tensors)

    self.output_shape = output_shape
    self.strides = strides
    self.padding = padding


  def prepare(self):
    # ni, ci, hi, wi = self.value.shape
    ci, co, hf, wf = self.filter.shape
    ni, co, ho, wo = self.output_shape

    hs, ws = self.strides
    hd, wd = 1, 1

    if self.padding == 'VALID':
      pad_t, pad_b, pad_l, pad_r = 0, 0, 0, 0
    elif self.padding == 'SAME':
      hi, wi = math.ceil(ho / hs), math.ceil(wo / ws)
      hp = hs * (hi - 1) + (hf - 1) * hd + 1 - ho
      wp = ws * (wi - 1) + (wf - 1) * wd + 1 - wo
      pad_t, pad_b, pad_l, pad_r = hp // 2, hp - hp // 2, wp // 2, wp - wp // 2
    else:
      (pad_t, pad_b), (pad_l, pad_r) = self.padding
    
    hi = get_output_dim(ho, pad_t + pad_b, hf, hd, hs)
    wi = get_output_dim(wo, pad_l + pad_r, wf, wd, ws)

    assert hi == self.value.shape[2] and wi == self.value.shape[3], [hi, self.value.shape[2], wi, self.value.shape[3]]

    return pad_t, pad_b, pad_l, pad_r


  def forward(self):
    pass


  def backward(self):
    pass


class GpuDeconv2D(Deconv2D):
  def __init__(self, 
               value,
               filter,
               output_shape,
               strides,
               padding='SAME',
               bias=None,
               name=None):
    super(GpuDeconv2D, self).__init__(value, filter, output_shape, strides, padding, bias, name)


  def forward(self):
    pad_t, pad_b, pad_l, pad_r = self.prepare()
    ni, ci, hi, wi = self.value.shape
    n, co, ho, wo = self.output_shape

    self.res.reshape(self.output_shape)

    self.crop_kernel = None
    if sum([pad_t, pad_b, pad_l, pad_r]) > 0:
      self.deconved = Tensor(dtype=self.value.dtype, need_grad=True)
      shape = [ni, co, ho + pad_t + pad_b, wo + pad_l + pad_r]
      self.deconved.reshape(shape)
      self.deconv2d_kernel = Deconv2DKernel(self.value, 
                                            self.filter, 
                                            self.deconved, 
                                            (0, 0), 
                                            self.strides, 
                                            (1, 1), 
                                            self.bias)

      self.crop_kernel = CropConstNCHWKernel(self.deconved, 
                                             self.res, 
                                             pad_t,
                                             pad_b,
                                             pad_l,
                                             pad_r)
    else:
      self.deconv2d_kernel = Deconv2DKernel(self.value, 
                                            self.filter, 
                                            self.res, 
                                            (0, 0), 
                                            self.strides, 
                                            (1, 1), 
                                            self.bias)

    self.deconv2d_kernel.forward()
    if self.crop_kernel is not None:
      self.crop_kernel.forward()


  def backward(self):
    if self.crop_kernel is not None:
      self.crop_kernel.backward()
    self.deconv2d_kernel.backward()


def deconv2d(value, 
             filter, 
             output_shape, 
             strides, 
             padding, 
             bias=None, 
             name=None):
  if nv.with_cuda is True:
    device = 'gpu'
  
  if device == 'gpu':
    layer = GpuDeconv2D(value, filter, output_shape, strides, padding, bias, name)
  elif device == 'cpu':
    layer = Deconv2D(value, filter, output_shape, strides, padding, bias, name)
  else:
    raise ValueError('Unsupport device {} for Deconv2D'.format(device))

  layer.forward()
  glog.info('Run {} Deconv2D Layer ... <{}> -> <{}>'.format(
    device, layer.value.shape, layer.res.shape))
  return layer.res
