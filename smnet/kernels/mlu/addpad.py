# Copyright (c) 2020 smarsu. All Rights Reserved.

import ctypes
from ...third_party import cnarray as cn


class MluAddPadKernel(object):
  def __init__(self, x, y, pad_t, pad_b, pad_l, pad_r, pad_value):
    self.x = x
    self.y = y
    self.pad_t = pad_t
    self.pad_b = pad_b
    self.pad_l = pad_l
    self.pad_r = pad_r
    self.pad_value = pad_value

    assert len(self.x.shape) == 4, len(self.x.shape)


  def forward(self):
    n, h, w, c = self.x.shape 

    cn.libsmcn.AddPad(self.x.mlu, 
                      self.y.mlu, 
                      ctypes.c_size_t(n), 
                      ctypes.c_size_t(h), 
                      ctypes.c_size_t(w), 
                      ctypes.c_size_t(c), 
                      self.pad_t, 
                      self.pad_b, 
                      self.pad_l, 
                      self.pad_r, 
                      ctypes.c_float(self.pad_value))
