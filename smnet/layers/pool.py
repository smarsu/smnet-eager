# Copyright (c) 2020 smarsu. All Rights Reserved.

import glog
import math
import numpy as np

from .import _math_utils as math_utils
from ..blob import Tensor
from ..layer import Layer

from ..third_party import nvarray as nv
if nv.with_cuda is True:
  from ..kernels.gpu import *

from ..third_party import cnarray as cn
if cn.with_mlu is True:
  from ..kernels.mlu import MluAddPadKernel, MluPool2DKernel


def get_output_dim(input_dim, pad, filter_dim, dilation, stride):
  return 1 + (input_dim + pad - (((filter_dim - 1) * dilation) + 1)) // stride


class Pool2D(Layer):
  def __init__(self,     
               value,
               ksize,
               strides,
               padding,
               mode,
               name=None):
    super(Pool2D, self).__init__(name=name)
    self._setup(value, ksize, strides, padding, mode)

  
  def _setup(self, value, ksize, strides, padding, mode):
    self.value = self._to_tensor(value)
    self.res = self._res_tensor([self.value])

    self.ksize = ksize
    self.strides = strides
    self.padding = padding
    self.mode = mode


  def prepare(self):
    n, ci, hi, wi = self.value.shape
    hf, wf = self.ksize
    hs, ws = self.strides

    if self.padding == 'VALID':
      pad_t, pad_b, pad_l, pad_r = 0, 0, 0, 0
    elif self.padding == 'SAME':
      ho, wo = math.ceil(hi / hs), math.ceil(wi / ws)
      hp = hs * (ho - 1) + (hf - 1) + 1 - hi
      wp = ws * (wo - 1) + (wf - 1) + 1 - wi
      pad_t, pad_b, pad_l, pad_r = hp // 2, hp - hp // 2, wp // 2, wp - wp // 2
    else:
      (pad_t, pad_b), (pad_l, pad_r) = self.padding

    ho = get_output_dim(hi, pad_t + pad_b, hf, 1, hs)
    wo = get_output_dim(wi, pad_l + pad_r, wf, 1, ws)

    if self.mode == 'MAX':
      constant_values = -np.inf
    elif self.mode == 'AVG':
      constant_values = 0

    return constant_values, pad_t, pad_b, pad_l, pad_r, n, ci, ho, wo

  
  def forward(self):
    constant_values, pad_t, pad_b, pad_l, pad_r, n, ci, ho, wo = self.prepare()

    self.pad_value = np.pad(
      self.value.data, 
      ((0, 0), 
       (0, 0), 
       (pad_t, pad_b), 
       (pad_l, pad_r)), 
      'constant', 
      constant_values=constant_values)

    self.res.feed(np.empty(shape=(n, ci, ho, wo), dtype=self.res.dtype))

    self.pad_shape = [pad_t, pad_b, pad_l, pad_r]

    if self.mode == 'MAX':
      math_utils.max_pool2d(self.res.data, 
                            self.pad_value, 
                            self.ksize, 
                            self.strides)
    elif self.mode == 'AVG':
      math_utils.avg_pool2d(self.res.data, 
                            self.pad_value, 
                            self.ksize, 
                            self.strides, 
                            self.pad_shape)

  
  def backward(self):
    if self.value.need_grad:
      _, _, pad_input_h, pad_input_w = self.pad_value.shape

      value_grad = np.zeros(shape=self.pad_value.shape, dtype=self.value.dtype)
      if self.mode == 'MAX':
        math_utils.max_pool2d_backward(value_grad, 
                                      self.res.grad, 
                                      self.pad_value,
                                      self.ksize, 
                                      self.strides, 
                                      0)
      elif self.mode == 'AVG':
        math_utils.avg_pool2d_backward(value_grad, 
                                      self.res.grad, 
                                      self.ksize, 
                                      self.strides, 
                                      self.pad_shape, 
                                      0)

      pad_t, pad_b, pad_l, pad_r = self.pad_shape
      self.value.feed_grad(value_grad[:, :, pad_t:pad_input_h-pad_b, pad_l:pad_input_w-pad_r])


class GpuPool2D(Pool2D):
  def __init__(self, 
               value,
               ksize,
               strides,
               padding,
               mode,
               name=None):
    super(GpuPool2D, self).__init__(value, ksize, strides, padding, mode, name=name)


  def __del__(self):
    del self.pool2d_kernel
    if self.pad is not None:
      del self.pad
    if self.addpad_kernel is not None:
      del self.addpad_kernel


  def forward(self):
    n, ci, hi, wi = self.value.shape
    constant_values, pad_t, pad_b, pad_l, pad_r, n, ci, ho, wo = self.prepare()

    self.res.reshape([n, ci ,ho, wo])

    mode = 0 if self.mode == 'MAX' else 1
    self.pad = None
    self.addpad_kernel = None

    addpad_h, addpad_w = (pad_t + pad_b) % 2, (pad_l + pad_r) % 2
    if addpad_h != 0 or addpad_w != 0:
      shape = [n, ci, hi + addpad_h, wi + addpad_w]
      self.pad = Tensor(dtype=self.value.dtype, need_grad=True)
      self.pad.reshape(shape)
      self.addpad_kernel = PadConstNCHWKernel(self.value, self.pad, addpad_h, addpad_w, constant_values)

      self.pool2d_kernel = Pool2DKernel(self.pad, 
                                        self.res, 
                                        mode, 
                                        self.ksize, 
                                        ((pad_t + pad_b) // 2, (pad_l + pad_r) // 2), 
                                        self.strides)
    else:
      self.pool2d_kernel = Pool2DKernel(self.value, 
                                        self.res, 
                                        mode, 
                                        self.ksize, 
                                        ((pad_t + pad_b) // 2, (pad_l + pad_r) // 2), 
                                        self.strides)

    if self.addpad_kernel is not None:
      self.addpad_kernel.forward()
    self.pool2d_kernel.forward()

  
  def backward(self):
    self.pool2d_kernel.backward()
    if self.addpad_kernel is not None:
      self.addpad_kernel.backward()


class MluPool2D(Pool2D):
  def __init__(self, 
               value,
               ksize,
               strides,
               padding,
               mode,
               name=None):
    super(MluPool2D, self).__init__(value, ksize, strides, padding, mode, name=name)


  def prepare(self):
    n, hi, wi, ci = self.value.shape
    hf, wf = self.ksize
    hs, ws = self.strides

    if self.padding == 'VALID':
      pad_t, pad_b, pad_l, pad_r = 0, 0, 0, 0
    elif self.padding == 'SAME':
      ho, wo = math.ceil(hi / hs), math.ceil(wi / ws)
      hp = hs * (ho - 1) + (hf - 1) + 1 - hi
      wp = ws * (wo - 1) + (wf - 1) + 1 - wi
      pad_t, pad_b, pad_l, pad_r = hp // 2, hp - hp // 2, wp // 2, wp - wp // 2
    else:
      (pad_t, pad_b), (pad_l, pad_r) = self.padding

    ho = get_output_dim(hi, pad_t + pad_b, hf, 1, hs)
    wo = get_output_dim(wi, pad_l + pad_r, wf, 1, ws)

    if self.mode == 'MAX':
      constant_values = -np.inf
    elif self.mode == 'AVG':
      constant_values = 0

    return constant_values, pad_t, pad_b, pad_l, pad_r, n, ci, ho, wo


  def forward(self):
    n, hi, wi, ci = self.value.shape
    constant_values, pad_t, pad_b, pad_l, pad_r, n, ci, ho, wo = self.prepare()

    self.res.reshape([n, ho, wo, ci])

    mode = 0 if self.mode == 'MAX' else 1
    self.pad = None
    self.addpad_kernel = None

    if sum([pad_t, pad_b, pad_l, pad_r]) != 0:
      shape = [n, hi + pad_t + pad_b, wi + pad_l + pad_r, ci]
      self.pad = Tensor(dtype=self.value.dtype, need_grad=True)
      self.pad.reshape(shape)
      self.addpad_kernel = MluAddPadKernel(self.value, self.pad, pad_t, pad_b, pad_l, pad_r, constant_values)

      self.pool2d_kernel = MluPool2DKernel(self.pad, 
                                           self.res, 
                                           mode, 
                                           self.ksize,
                                           (0, 0),
                                           self.strides)
    else:
      self.pool2d_kernel = MluPool2DKernel(self.value,
                                           self.res,
                                           mode,
                                           self.ksize,
                                           (0, 0),
                                           self.strides)

    if self.addpad_kernel is not None:
      self.addpad_kernel.forward()
    self.pool2d_kernel.forward()


  def backward(self):
    pass


def max_pool2d(value,
               ksize,
               strides,
               padding,
               name=None,
               device='cpu'):
  if nv.with_cuda is True:
    device = 'gpu'
  elif cn.with_mlu is True:
    device = 'mlu'

  if device == 'gpu':
    layer = GpuPool2D(value, ksize, strides, padding, 'MAX', name)
  elif device == 'mlu':
    layer = MluPool2D(value, ksize, strides, padding, 'MAX', name)
  else:
    layer = Pool2D(value, ksize, strides, padding, 'MAX', name)

  layer.forward()

  glog.info('Run {} MaxPool2D Layer ... <{}> -> <{}>'.format(
    device, layer.value.shape, layer.res.shape))
  return layer.res


def avg_pool2d(value,
               ksize,
               strides,
               padding,
               name=None,
               device='cpu'):
  if nv.with_cuda is True:
    device = 'gpu'
  elif cn.with_mlu is True:
    device = 'mlu'

  if device == 'gpu':
    layer = GpuPool2D(value, ksize, strides, padding, 'AVG', name)
  elif device == 'mlu':
    layer = MluPool2D(value, ksize, strides, padding, 'AVG', name)
  else:
    layer = Pool2D(value, ksize, strides, padding, 'AVG', name)

  layer.forward()

  glog.info('Run {} AvgPool2D Layer ... <{}> -> <{}>'.format(
    device, layer.value.shape, layer.res.shape))
  return layer.res
