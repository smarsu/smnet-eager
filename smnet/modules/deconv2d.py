# Copyright (c) 2020 smarsu. All Rights Reserved.

import numpy as np

from ..blob import Variable
from ..layers import deconv2d


def get_output_dim(input_dim, pad, filter_dim, dilation, stride, padding):
  if padding == 'VALID':
    return filter_dim + stride * (input_dim - 1)
  elif padding == 'SAME':
    return input_dim * stride
  else:
    raise ValueError(padding)


class Deconv2D(object):
  def __init__(self,
               in_channels,
               out_channels,
               kernel_size,
               stride=1,
               padding='SAME',
               bias=True,
               name='Deconv2D'):
    self.strides = [stride, stride]
    self.padding = padding
    self.bias = bias
    self.name = name

    self.filter_shape = [in_channels, out_channels, kernel_size, kernel_size]
    limit = np.sqrt(6 / (in_channels * kernel_size * kernel_size + out_channels * kernel_size * kernel_size))

    self.weight_variable = Variable(
      np.random.uniform(-limit, limit, self.filter_shape),
      name='/'.join([self.name, 'weight']))
    if self.bias:
      self.bias_variable = Variable(
        np.zeros(shape=(1, out_channels, 1, 1)),
        name='/'.join([self.name, 'bias']))
    else:
      self.bias_variable = None

  
  def __call__(self, x):
    n, _, hi, wi = x.shape
    ho = get_output_dim(hi, 0, self.filter_shape[2], 1, self.strides[0], self.padding)
    wo = get_output_dim(wi, 0, self.filter_shape[3], 1, self.strides[1], self.padding)
    co = self.filter_shape[1]

    y = deconv2d(x, self.weight_variable, [n, co, ho, wo], self.strides, self.padding, self.bias_variable, self.name)
    return y
