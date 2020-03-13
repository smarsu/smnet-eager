# Copyright (c) 2020 smarsu. All Rights Reserved.

import ctypes
import numpy as np
from ...third_party import nvarray as nv

nv.libsmnv.CudnnActivationCreate.restype = ctypes.c_void_p


class ActivationKernel(object):
  def __init__(self, x, y, mode, coef=0):
    self.x = x
    self.y = y

    assert self.x.shape == self.y.shape, [self.x.shape, self.y.shape]

    shape = list(self.x.shape)
    if len(shape) < 4:
      shape = [1] * (4 - len(shape)) + shape

    self.params = ctypes.c_void_p(nv.libsmnv.CudnnActivationCreate(len(shape),
                                                                   nv.c_data(np.array(shape, dtype=np.int32)),
                                                                   mode,
                                                                   ctypes.c_double(coef)))

  
  def __del__(self):
    nv.libsmnv.DestroyActivationParams(self.params)


  def forward(self):
    nv.libsmnv.CudnnActivationForward(nv.cudnn_handle,
                                      self.params,
                                      ctypes.c_float(1),
                                      self.x.gpu,
                                      ctypes.c_float(0),
                                      self.y.gpu)

  
  def backward(self):
    if self.x.need_grad:
      nv.libsmnv.CudnnActivationBackward(nv.cudnn_handle,
                                         self.params,
                                         ctypes.c_float(1),
                                         self.y.gpu,
                                         self.y.gpu_grad,
                                         self.x.gpu,
                                         ctypes.c_float(1) if self.x._grad_seted else ctypes.c_float(0),
                                         self.x.gpu_grad)
      self.x._grad_seted = True
