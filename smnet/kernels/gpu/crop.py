# Copyright (c) 2020 smarsu. All Rights Reserved.

import ctypes
from ...third_party import nvarray as nv


class CropConstNCHWKernel(object):
  def __init__(self, x, y, crop_t, crop_b, crop_l, crop_r):
    self.x = x
    self.y = y

    self.crop_t = crop_t
    self.crop_b = crop_b
    self.crop_l = crop_l
    self.crop_r = crop_r


  def forward(self):
    _, _, h, w = self.y.shape

    nv.libsmnv.CropConstNCHW(self.y.size,
                             self.x.gpu,
                             ctypes.c_float(0),
                             self.y.gpu,
                             h,
                             w,
                             self.crop_t,
                             self.crop_b,
                             self.crop_l,
                             self.crop_r)


  def backward(self):
    _, _, h, w = self.x.shape

    if self.x.need_grad:
      nv.libsmnv.CropConstNCHWGradient(self.x.size,
                                       self.y.gpu_grad,
                                       ctypes.c_float(1) if self.x._grad_seted else ctypes.c_float(0),
                                       self.x.gpu_grad,
                                       h,
                                       w,
                                       self.crop_t,
                                       self.crop_b,
                                       self.crop_l,
                                       self.crop_r)
      self.x._grad_seted = True
