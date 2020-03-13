# Copyright (c) 2020 smarsu. All Rights Reserved.

import ctypes
from ...third_party import nvarray as nv


class PadConstNCHWKernel(object):
  def __init__(self, x, y, pad_b, pad_r, pad_value):
    self.x = x
    self.y = y
    self.pad_b = pad_b
    self.pad_r = pad_r
    self.pad_value = pad_value


  def forward(self):
    _, _, h, w = self.y.shape

    nv.libsmnv.PadConstNCHW(self.y.size, 
                            self.x.gpu, 
                            self.y.gpu, 
                            h,
                            w,
                            self.pad_b,
                            self.pad_r,
                            ctypes.c_float(self.pad_value))

  
  def backward(self):
    _, _, h, w = self.x.shape

    if self.x.need_grad:
      nv.libsmnv.PadConstNCHWGradient(self.x.size,
                                      self.y.gpu_grad,
                                      ctypes.c_float(1) if self.x._grad_seted else ctypes.c_float(0),
                                      self.x.gpu_grad,
                                      h,
                                      w,
                                      self.pad_b,
                                      self.pad_r)
      self.x._grad_seted = True
