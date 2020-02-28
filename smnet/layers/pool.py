# Copyright (c) 2020 smarsu. All Rights Reserved.

import math
import numpy as np

from .import _math_utils as math_utils
from ..layer import Layer


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

  
  def forward(self):
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


def max_pool2d(value,
               ksize,
               strides,
               padding,
               name=None):
  layer = Pool2D(value, ksize, strides, padding, 'MAX', name)
  layer.forward()
  return layer.res


def avg_pool2d(value,
               ksize,
               strides,
               padding,
               name=None):
  layer = Pool2D(value, ksize, strides, padding, 'AVG', name)
  layer.forward()
  return layer.res
