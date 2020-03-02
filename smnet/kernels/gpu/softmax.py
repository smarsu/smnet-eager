# Copyright (c) 2020 smarsu. All Rights Reserved.

import ctypes
from ...third_party import nvarray as nv
from ...layers import _math_utils as math_utils

nv.libsmnv.CudnnSoftmaxCreate.restype = ctypes.c_void_p


class SoftmaxKernel(object):
  def __init__(self, x, y, axis, algo):
    self.x = x
    self.y = y
    self.algo = algo

    assert self.x.shape == self.y.shape, '{} ... {}'.format(self.x.shape, self.y.shape)

    out_dim, dim, inner_dim = math_utils.get_3level_shape(self.x.shape, axis)

    self.params = ctypes.c_void_p(nv.libsmnv.CudnnSoftmaxCreate(out_dim,
                                                                dim,
                                                                inner_dim,
                                                                algo))


  def __del__(self):
    nv.libsmnv.DestroySoftmaxParams(self.params)


  def forward(self, alpha=1):
    nv.libsmnv.CudnnSoftmaxForward(nv.cudnn_handle,
                                   self.params,
                                   ctypes.c_float(alpha),
                                   self.x.gpu,
                                   ctypes.c_float(0),
                                   self.y.gpu)


  def backward(self, alpha=1):
    nv.libsmnv.CudnnSoftmaxBackward(nv.cudnn_handle,
                                    self.params,
                                    ctypes.c_float(alpha),
                                    self.y.gpu,
                                    self.y.gpu_grad,
                                    ctypes.c_float(1) if self.x._grad_seted else ctypes.c_float(0),
                                    self.x.gpu_grad)
