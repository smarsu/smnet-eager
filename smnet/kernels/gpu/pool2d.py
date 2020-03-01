# Copyright (c) 2020 smarsu. All Rights Reserved.

import ctypes
from ...third_party import nvarray as nv

nv.libsmnv.CudnnPool2DCreate.restype = ctypes.c_void_p


class Pool2DKernel(object):
  def __init__(self, x, y, mode, windows, pads, strides):
    self.x = x
    self.y = y

    ni, ci, hi, wi = self.x.shape
    _, _, ho, wo = self.y.shape

    hw, ww = windows
    hp, wp = pads
    hs, ws = strides

    self.params = ctypes.c_void_p(nv.libsmnv.CudnnPool2DCreate(ni,
                                                               ci,
                                                               hi,
                                                               wi,
                                                               ho,
                                                               wo,
                                                               mode,
                                                               hw,
                                                               ww,
                                                               hp,
                                                               wp,
                                                               hs,
                                                               ws))


  def __del__(self):
    nv.libsmnv.DestroyPool2DParams(self.params)


  def forward(self):
    nv.libsmnv.CudnnPool2DForward(nv.cudnn_handle,
                                  self.params,
                                  ctypes.c_float(1),
                                  self.x.gpu,
                                  ctypes.c_float(0),
                                  self.y.gpu)

  
  def backward(self):
    nv.libsmnv.CudnnPool2DBackward(nv.cudnn_handle,
                                   self.params,
                                   ctypes.c_float(1),
                                   self.y.gpu,
                                   self.y.gpu_grad,
                                   self.x.gpu,
                                   ctypes.c_float(1) if self.x._grad_seted else ctypes.c_float(0),
                                   self.x.gpu_grad)
