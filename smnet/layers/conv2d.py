# Copyright (c) 2020 smarsu. All Rights Reserved.

import math
import numpy as np

from . import _math_utils as math_utils
from ..layer import Layer


def get_output_dim(input_dim, pad, filter_dim, dilation, stride):
  return 1 + (input_dim + pad - (((filter_dim - 1) * dilation) + 1)) // stride


class Conv2D(Layer):
  def __init__(self, 
               input, 
               filter, 
               strides, 
               padding, 
               dilations=[1, 1], 
               name=None):
    super(Conv2D, self).__init__(name=name)
    self._setup(input, filter, strides, padding, dilations)
  
  
  def _setup(self, input, filter, strides, padding, dilations):
    self.input = self._to_tensor(input)
    self.filter = self._to_tensor(filter)
    self.res = self._res_tensor([self.input, self.filter])

    self.strides = strides
    self.padding = padding
    self.dilations = dilations

  
  def forward(self):
    n, ci, hi, wi = self.input.shape
    co, ci, hf, wf = self.filter.shape

    hd, wd = self.dilations
    hs, ws = self.strides

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

    self.pad_input = np.pad(
      self.input.data, 
      ((0, 0), 
       (0, 0), 
       (pad_t, pad_b), 
       (pad_l, pad_r)), 
      'constant', 
      constant_values=0)

    self.res.feed(np.empty(shape=(n, co, ho, wo), dtype=self.res.dtype))
  
    self.pad_shape = [pad_t, pad_b, pad_l, pad_r]

    math_utils.conv2d(self.res.data, 
                      self.pad_input, 
                      self.filter.data, 
                      self.strides, 
                      self.pad_shape,
                      self.dilations)

  
  def backward(self):
    _, _, pad_input_h, pad_input_w = self.pad_input.shape

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

    math_utils.conv2d_backward_filter(self.filter.grad, 
                                      self.pad_input, 
                                      self.res.grad, 
                                      self.strides, 
                                      self.pad_shape, 
                                      self.dilations,
                                      1 if self.filter._grad_seted else 0)


def conv2d(input, 
           filter, 
           strides, 
           padding, 
           dilations=[1, 1], 
           name=None):
  layer = Conv2D(input, filter, strides, padding, dilations, name)
  layer.forward()
  return layer.res
