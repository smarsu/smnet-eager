# Copyright (c) 2020 smarsu. All Rights Reserved.

import ctypes
from ...third_party import nvarray as nv

nv.libsmnv.CudnnConv2DCreate.restype = ctypes.c_void_p


class Conv2DKernel(object):
  def __init__(self, x, w, y, pads, strides, dilations, bias=None):
    self.x = x
    self.w = w
    self.y = y
    self.bias = bias

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
    if self.bias is not None:
      nv.libsmnv.CudnnConv2DForwardBias(nv.cudnn_handle,
                                        self.params,
                                        ctypes.c_float(1),
                                        self.bias.gpu,
                                        ctypes.c_float(1),
                                        self.y.gpu)

  
  def backward(self):
    self.backward_data()
    self.backward_filter()
    self.backward_bias()


  def backward_data(self):
    if self.x.need_grad:
      nv.libsmnv.CudnnConv2DBackwardData(nv.cudnn_handle,
                                         self.params,
                                         ctypes.c_float(1),
                                         self.w.gpu,
                                         self.y.gpu_grad,
                                         ctypes.c_float(1) if self.x._grad_seted else ctypes.c_float(0),
                                         self.x.gpu_grad)
      self.x._grad_seted = True

  
  def backward_filter(self):
    if self.w.need_grad:
      nv.libsmnv.CudnnConv2DBackwardFilter(nv.cudnn_handle,
                                           self.params,
                                           ctypes.c_float(1),
                                           self.x.gpu,
                                           self.y.gpu_grad,
                                           ctypes.c_float(1) if self.w._grad_seted else ctypes.c_float(0),
                                           self.w.gpu_grad)
      self.w._grad_seted = True


  def backward_bias(self):
    if self.bias is not None and self.bias.need_grad:
      nv.libsmnv.CudnnConv2DBackwardBias(nv.cudnn_handle,
                                        self.params,
                                        ctypes.c_float(1),
                                        self.y.gpu_grad,
                                        ctypes.c_float(1) if self.bias._grad_seted else ctypes.c_float(0),
                                        self.bias.gpu_grad)
      self.bias._grad_seted = True
