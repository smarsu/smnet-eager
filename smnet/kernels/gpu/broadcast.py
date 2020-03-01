# Copyright (c) 2020 smarsu. All Rights Reserved.

import ctypes
import numpy as np

from ...third_party import nvarray as nv

nv.libsmnv.CudnnReduceCreate.restype = ctypes.c_void_p


class BroadcastKernel(object):
  def __init__(self, x, y, shape_x, shape_y):
    self.x = x
    self.y = y
    self.shape_x = shape_x
    self.shape_y = shape_y

    assert len(self.shape_x) == len(self.shape_y), '{} broadcast to {}'.format(self.shape_x, self.shape_y)
    assert self.x.size == int(np.prod(self.shape_x))
    assert self.y.size == int(np.prod(self.shape_y))

    self.shape_x_gpu = nv.array(self.shape_x, dtype=np.int32)
    self.shape_y_gpu = nv.array(self.shape_y, dtype=np.int32)

    ndims = len(self.shape_x)
    dis = 0
    if ndims < 4:
      dis = 4 - ndims
      ndims = 4

    # Reduce y to x
    self.params = ctypes.c_void_p(nv.libsmnv.CudnnReduceCreate(nv.cudnn_handle,
                                                               ndims,
                                                               nv.c_data(np.array([1] * dis + self.shape_y, dtype=np.int32)),
                                                               nv.c_data(np.array([1] * dis + self.shape_x, dtype=np.int32)),
                                                               0,  # sum
                                                               ctypes.c_bool(False)))

  
  def __del__(self):
    del self.shape_x_gpu
    del self.shape_y_gpu

    del self.params


  def forward(self):
    nv.libsmnv.Broadcast(self.y.size,
                         self.x.gpu,
                         ctypes.c_float(0),
                         self.y.gpu,
                         len(self.shape_x),
                         self.shape_x_gpu.gpu,
                         self.shape_y_gpu.gpu)


  def backward(self):
    nv.libsmnv.CudnnReduceForward(nv.cudnn_handle,
                                  self.params,
                                  ctypes.c_void_p(0),
                                  ctypes.c_float(1),
                                  self.y.gpu_grad,
                                  ctypes.c_float(1) if self.x._grad_seted else ctypes.c_float(0),
                                  self.x.gpu_grad)
