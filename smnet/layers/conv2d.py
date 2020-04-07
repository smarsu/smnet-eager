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

from ..third_party import cnarray as cn
if cn.with_mlu is True:
  from ..kernels.mlu import MluAddPadKernel, MluConv2DKernel


def get_output_dim(input_dim, pad, filter_dim, dilation, stride):
  return 1 + (input_dim + pad - (((filter_dim - 1) * dilation) + 1)) // stride


class Conv2D(Layer):
  def __init__(self, 
               input, 
               filter, 
               strides, 
               padding, 
               dilations=[1, 1], 
               bias=None,
               name=None):
    super(Conv2D, self).__init__(name=name)
    self._setup(input, filter, strides, padding, dilations, bias)


  def _setup(self, input, filter, strides, padding, dilations, bias):
    self.input = self._to_tensor(input)
    self.filter = self._to_tensor(filter)
    self.bias = bias
    input_tensors = [self.input, self.filter]
    if self.bias is not None:
      self.bias = self._to_tensor(bias)
      input_tensors.append(self.bias)
    self.res = self._res_tensor(input_tensors)

    self.strides = strides
    self.padding = padding
    self.dilations = dilations


  def prepare(self):
    n, ci, hi, wi = self.input.shape
    co, ci, hf, wf = self.filter.shape

    hs, ws = self.strides
    hd, wd = self.dilations

    if self.padding == 'VALID':
      pad_t, pad_b, pad_l, pad_r = 0, 0, 0, 0
    elif self.padding == 'SAME':
      ho, wo = math.ceil(hi / hs), math.ceil(wi / ws)
      hp = hs * (ho - 1) + (hf - 1) * hd + 1 - hi
      wp = ws * (wo - 1) + (wf - 1) * wd + 1 - wi
      pad_t, pad_b, pad_l, pad_r = hp // 2, hp - hp // 2, wp // 2, wp - wp // 2
    else:
      (pad_t, pad_b), (pad_l, pad_r) = self.padding

    ho = get_output_dim(hi, pad_t + pad_b, hf, hd, hs)
    wo = get_output_dim(wi, pad_l + pad_r, wf, wd, ws)

    assert np.sum(np.array([n, co, ho, wo]) > 0) == 4, [n, co, ho, wo]

    return pad_t, pad_b, pad_l, pad_r, n, co, ho, wo

  
  def forward(self):
    pad_t, pad_b, pad_l, pad_r, n, co, ho, wo = self.prepare()

    self.pad_input = np.pad(
      self.input.data, 
      ((0, 0), 
       (0, 0), 
       (pad_t, pad_b), 
       (pad_l, pad_r)), 
      'constant', 
      constant_values=0)
    self.pad_shape = [pad_t, pad_b, pad_l, pad_r]

    self.res.feed(np.empty(shape=(n, co, ho, wo), dtype=self.res.dtype))
  
    math_utils.conv2d(self.res.data, 
                      self.pad_input, 
                      self.filter.data, 
                      self.strides, 
                      self.pad_shape,
                      self.dilations)

    if self.bias is not None:
      self.res.feed(self.res.data + self.bias.data)

  
  def backward(self):
    _, _, pad_input_h, pad_input_w = self.pad_input.shape

    if self.input.need_grad:
      input_grad = np.empty(shape=self.pad_input.shape, dtype=np.float32)
      math_utils.conv2d_backward_data(input_grad, 
                                      self.res.grad, 
                                      self.filter.data, 
                                      self.strides, 
                                      self.pad_shape, 
                                      self.dilations, 
                                      0.)

      pad_t, pad_b, pad_l, pad_r = self.pad_shape
      self.input.feed_grad(input_grad[:, :, pad_t:pad_input_h-pad_b, pad_l:pad_input_w-pad_r])

    if self.filter.need_grad:
      math_utils.conv2d_backward_filter(self.filter.grad, 
                                        self.pad_input, 
                                        self.res.grad, 
                                        self.strides, 
                                        self.pad_shape, 
                                        self.dilations,
                                        1 if self.filter._grad_seted else 0)

    if self.bias is not None and self.bias.need_grad:
      self.bias.feed_grad(self.res.grad)


class GpuConv2D(Conv2D):
  def __init__(self, 
               input, 
               filter, 
               strides, 
               padding, 
               dilations=[1, 1], 
               bias=None,
               name=None):
    super(GpuConv2D, self).__init__(input, filter, strides, padding, dilations, bias, name=name)


  def __del__(self):
    del self.conv2d_kernel
    if self.pad is not None:
      del self.pad


  def forward(self):
    n, ci, hi, wi = self.input.shape
    pad_t, pad_b, pad_l, pad_r, n, co, ho, wo = self.prepare()

    self.res.reshape([n, co, ho, wo])

    self.addpad_kernel = None
    self.pad = None

    addpad_h, addpad_w = (pad_t + pad_b) % 2, (pad_l + pad_r) % 2
    if addpad_h != 0 or addpad_w != 0:
      shape = [n, ci, hi + addpad_h, wi + addpad_w]
      self.pad = Tensor(dtype=self.input.dtype, need_grad=True)
      self.pad.reshape(shape)
      self.addpad_kernel = PadConstNCHWKernel(self.input, self.pad, addpad_h, addpad_w, 0)

      self.conv2d_kernel = Conv2DKernel(self.pad, 
                                        self.filter, 
                                        self.res, 
                                        ((pad_t + pad_b) // 2, (pad_l + pad_r) // 2), 
                                        self.strides, 
                                        self.dilations,
                                        self.bias)
    else:
      self.conv2d_kernel = Conv2DKernel(self.input, 
                                        self.filter, 
                                        self.res, 
                                        ((pad_t + pad_b) // 2, (pad_l + pad_r) // 2), 
                                        self.strides, 
                                        self.dilations,
                                        self.bias)

    if self.addpad_kernel is not None:
      self.addpad_kernel.forward()
    self.conv2d_kernel.forward()
  

  def backward(self):
    self.conv2d_kernel.backward()

    if self.addpad_kernel is not None:
      self.addpad_kernel.backward()


class MluConv2D(Conv2D):
  def __init__(self,
               input,
               filter,
               strides,
               padding,
               dilations=[1, 1],
               bias=None,
               name=None):
    super(MluConv2D, self).__init__(input, filter, strides, padding, dilations, bias, name=name)


  def prepare(self):
    n, hi, wi, ci = self.input.shape
    co, hf, wf, ci = self.filter.shape

    hs, ws = self.strides
    hd, wd = self.dilations

    if self.padding == 'VALID':
      pad_t, pad_b, pad_l, pad_r = 0, 0, 0, 0
    elif self.padding == 'SAME':
      ho, wo = math.ceil(hi / hs), math.ceil(wi / ws)
      hp = hs * (ho - 1) + (hf - 1) * hd + 1 - hi
      wp = ws * (wo - 1) + (wf - 1) * wd + 1 - wi
      pad_t, pad_b, pad_l, pad_r = hp // 2, hp - hp // 2, wp // 2, wp - wp // 2
    else:
      (pad_t, pad_b), (pad_l, pad_r) = self.padding

    ho = get_output_dim(hi, pad_t + pad_b, hf, hd, hs)
    wo = get_output_dim(wi, pad_l + pad_r, wf, wd, ws)

    assert np.sum(np.array([n, co, ho, wo]) > 0) == 4, [n, co, ho, wo]

    return pad_t, pad_b, pad_l, pad_r, n, co, ho, wo


  def forward(self):
    n, hi, wi, ci = self.input.shape
    pad_t, pad_b, pad_l, pad_r, n, co, ho, wo = self.prepare()

    self.res.reshape([n, ho, wo, co])

    self.addpad_kernel = None
    self.pad = None

    if sum([pad_t, pad_b, pad_l, pad_r]) != 0:
      shape = [n, hi + pad_t + pad_b, wi + pad_l + pad_r, ci]
      self.pad = Tensor(dtype=self.input.dtype, need_grad=True)
      self.pad.reshape(shape)
      self.addpad_kernel = MluAddPadKernel(self.input, self.pad, pad_t, pad_b, pad_l, pad_r, 0)

      self.conv2d_kernel = MluConv2DKernel(self.pad, 
                                           self.filter, 
                                           self.res, 
                                           (0, 0), 
                                           self.strides, 
                                           self.dilations,
                                           self.bias)
    else:
      self.conv2d_kernel = MluConv2DKernel(self.input, 
                                           self.filter, 
                                           self.res, 
                                           ((pad_t + pad_b) // 2, (pad_l + pad_r) // 2), 
                                           self.strides, 
                                           self.dilations,
                                           self.bias)

    if self.addpad_kernel is not None:
      self.addpad_kernel.forward()
    self.conv2d_kernel.forward()

  
  def backward(self):
    pass


def conv2d(input, 
           filter, 
           strides, 
           padding, 
           dilations=[1, 1], 
           bias=None,
           name=None,
           device='cpu'):
  if nv.with_cuda is True:
    device = 'gpu'
  elif cn.with_mlu is True:
    device = 'mlu'

  if device == 'gpu':
    layer = GpuConv2D(input, filter, strides, padding, dilations, bias, name)
  elif device == 'mlu':
    layer = MluConv2D(input, filter, strides, padding, dilations, bias, name)
  elif device == 'cpu':
    layer = Conv2D(input, filter, strides, padding, dilations, bias, name)
  else:
    raise ValueError('Unsupport device {} for Conv2D'.format(device))

  layer.forward()
  glog.info('Run {} Conv2D Layer ... <{}> -> <{}>'.format(
    device, layer.input.shape, layer.res.shape))
  return layer.res
