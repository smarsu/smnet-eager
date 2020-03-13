# Copyright (c) 2020 smarsu. All Rights Reserved.

import numpy as np

from . import _math_utils as math_utils
from ..blob import Tensor
from ..layer import Layer
from ..kernels import *


class GpuBinaryElementwise(Layer):
  def __init__(self, a, b, mode, name=None):
    super(GpuBinaryElementwise, self).__init__(name=name)
    self._setup(a, b, mode)


  def __del__(self):
    if self.broadcast_a is not None:
      del self.broadcast_a
    if self.broadcast_b is not None:
      del self.broadcast_b
    if self.broadcast_a_kernel is not None:
      del self.broadcast_a_kernel
    if self.broadcast_b_kernel is not None:
      del self.broadcast_b_kernel

    del self.elementwise_kernel


  def _setup(self, a, b, mode):
    self.a = self._to_tensor(a)
    self.b = self._to_tensor(b)
    self.res = self._res_tensor([self.a, self.b])

    self.mode = mode


  def forward(self):
    self.res_shape, self.broadcast_a_shape, self.broadcast_b_shape = \
      math_utils.broadcast_shape(self.a.shape, self.b.shape)

    self.res.reshape(self.res_shape)

    self.broadcast_a = None
    self.broadcast_b = None
    self.broadcast_a_kernel = None
    self.broadcast_b_kernel = None

    if self.a.size != int(np.prod(self.res_shape)):
      self.broadcast_a = Tensor(dtype=self.a.dtype, need_grad=True)
      self.broadcast_a.reshape(self.res_shape)
      self.broadcast_a_kernel = BroadcastKernel(self.a, 
                                                self.broadcast_a, 
                                                self.broadcast_a_shape, 
                                                self.res_shape)

      self.broadcast_a_kernel.forward()

    if self.b.size != int(np.prod(self.res_shape)):
      self.broadcast_b = Tensor(dtype=self.b.dtype, need_grad=True)
      self.broadcast_b.reshape(self.res_shape)
      self.broadcast_b_kernel = BroadcastKernel(self.b,
                                                self.broadcast_b,
                                                self.broadcast_b_shape,
                                                self.res_shape)
    
      self.broadcast_b_kernel.forward()

    self.elem_a = self.broadcast_a if self.broadcast_a is not None else self.a
    self.elem_b = self.broadcast_b if self.broadcast_b is not None else self.b

    self.elementwise_kernel = BinaryElementwiseKernel(self.elem_a, self.elem_b, self.res, self.mode)
    self.elementwise_kernel.forward()

  
  def backward(self):
    self.elementwise_kernel.backward()
    if self.broadcast_a_kernel is not None:
      self.broadcast_a_kernel.backward()
    if self.broadcast_b_kernel is not None:
      self.broadcast_b_kernel.backward()
