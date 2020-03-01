# Copyright (c) 2020 smarsu. All Rights Reserved.

import ctypes
from ...third_party import nvarray as nv

nv.libsmnv.CudnnConv2DCreate.restype = ctypes.c_void_p


class Conv2DKernel(object):
  def __init__(self, x, w, y, pads, strides, dilations):
    self.x = x
    self.w = w
    self.y = y

    ni, ci, hi, wi = self.x.shape
    co, ci, hf, wf = self.w.shape
    ni, co, ho, wo = self.y.shape

    hp, wp = pads
    hs, ws = strides
    hd, wd = dilations

    self.params = ctypes.c_void_p(nv.libsmnv.CudnnConv2DCreate(nv.cudnn_handle,
                                                               ni, 
                                                               ci, 
                                                               hi, 
                                                               wi,
                                                               co, 
                                                               hf, 
                                                               wf,
                                                               ho,
                                                               wo,
                                                               hp,
                                                               wp,
                                                               hs,
                                                               ws,
                                                               hd,
                                                               wd))

  
  def __del__(self):
    nv.libsmnv.DestroyConv2DParams(self.params)


  def forward(self):
    nv.libsmnv.CudnnConv2DForward(nv.cudnn_handle,
                                  self.params,
                                  ctypes.c_float(1),
                                  self.x.gpu,
                                  self.w.gpu,
                                  ctypes.c_float(0),
                                  self.y.gpu)

  
  def backward(self):
    self.backward_data()
    self.backward_filter()


  def backward_data(self):
    nv.libsmnv.CudnnConv2DBackwardData(nv.cudnn_handle,
                                       self.params,
                                       ctypes.c_float(1),
                                       self.w.gpu,
                                       self.y.gpu_grad,
                                       ctypes.c_float(1) if self.x._grad_seted else ctypes.c_float(0),
                                       self.x.gpu_grad)

  
  def backward_filter(self):
    nv.libsmnv.CudnnConv2DBackwardFilter(nv.cudnn_handle,
                                         self.params,
                                         ctypes.c_float(1),
                                         self.x.gpu,
                                         self.y.gpu_grad,
                                         ctypes.c_float(1) if self.w._grad_seted else ctypes.c_float(0),
                                         self.w.gpu_grad)
