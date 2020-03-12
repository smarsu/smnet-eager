# Copyright (c) 2020 smarsu. All Rights Reserved.

import numpy as np

from ..blob import Variable
from ..layers import conv2d


class Conv2D(object):
  def __init__(self, 
               in_channels, 
               out_channels, 
               kernel_size, 
               stride=1, 
               padding='SAME', 
               dilation=1, 
               bias=True,
               name='Conv2D'):
    self.strides = [stride, stride]
    self.padding = padding
    self.dilations = [dilation, dilation]
    self.bias = bias
    self.name = name

    filter_shape = [out_channels, in_channels, kernel_size, kernel_size]
    limit = np.sqrt(6 / (in_channels * kernel_size * kernel_size + out_channels * kernel_size * kernel_size))
    # God bless glorot_uniform
    self.weight_varaible = Variable(
      np.random.uniform(-limit, limit, filter_shape), name='/'.join([self.name, 'weight']))
    if self.bias:
      self.bias_variable = Variable(
        np.zeros(shape=(1, out_channels, 1, 1)), name='/'.join([self.name, 'bias']))
    else:
      self.bias_variable = None

  
  def __call__(self, x):
    y = conv2d(x, self.weight_varaible, self.strides, self.padding, self.dilations, self.bias_variable, name=self.name)
    # if self.bias:
    #   y += self.bias_variable
    return y
