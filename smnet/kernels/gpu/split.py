# Copyright (c) 2020 smarsu. All Rights Reserved.

import ctypes
import numpy as np
from ...third_party import nvarray as nv
from ...layers import _math_utils as math_utils


class SplitKernel(object):
  def __init__(self, x, res, axis, splt):
    self.x = x
    self.res = res
    self.axis = axis
    self.splt = splt

    self.out_dim, self.dim, self.inner_dim = math_utils.get_3level_shape(self.x.shape, axis)
    self.split_dims = nv.array(np.array([0] + self.splt), dtype=np.int32)
    self.y = nv.list_array([tensor.gpu for tensor in self.res])

    zeros = np.zeros(shape=self.x.shape, dtype=self.x.dtype)
    self.zeros = nv.array(zeros, dtype=self.x.dtype, name='static/zeros')

    self.grad_concat = None


  def __del__(self):
    del self.split_dims
    nv.libsmnv.CudaFree(self.y)
    del self.zeros

    if self.grad_concat is not None:
      nv.libsmnv.CudaFree(self.grad_concat)

  
  def forward(self):
    nv.libsmnv.Split(self.x.size,
                     self.x.gpu,
                     ctypes.c_float(0),
                     self.y,
                     self.out_dim,
                     self.dim,
                     self.inner_dim,
                     len(self.splt),
                     self.split_dims.gpu)


  def backward(self):
    grads = []
    for i in range(len(self.res)):
      if self.res[i]._grad_seted:
        grads.append(self.res[i].gpu_grad)
      else:
        grads.append(self.zeros.gpu)
    self.grad_concat = nv.list_array(grads)
    
    nv.libsmnv.Concat(self.x.size,
                      self.grad_concat,
                      ctypes.c_float(1) if self.x._grad_seted else ctypes.c_float(0),
                      self.x.gpu_grad,
                      self.out_dim,
                      self.dim,
                      self.inner_dim,
                      len(self.splt),
                      self.split_dims.gpu)
