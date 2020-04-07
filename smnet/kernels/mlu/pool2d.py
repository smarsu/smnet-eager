# Copyright (c) 2020 smarsu. All Rights Reserved.

import ctypes
from ...third_party import cnarray as cn


class MluPool2DKernel(object):
  def __init__(self, x, y, mode, windows, pads, strides):
    self.x = x
    self.y = y

    self.ni, self.hi, self.wi, self.ci = self.x.shape
    self.no, self.ho, self.wo, self.co = self.y.shape

    self.mode = mode

    self.hw, self.ww = windows
    self.hp, self.wp = pads
    self.hs, self.ws = strides

    assert self.hp == self.wp == 0, [self.hp, self.wp]

  
  def forward(self):
    if self.mode == 0:  # Max
      cn.libsmcn.MaxPool2D(self.x.mlu,
                           self.y.mlu,
                           self.ni,
                           self.hi,
                           self.wi,
                           self.ci,
                           self.no,
                           self.ho,
                           self.wo,
                           self.co,
                           self.hw,
                           self.ww,
                           self.hs,
                           self.ws)
    elif self.mode == 1:  # Avg
      cn.libsmcn.AvgPool2D(self.x.mlu,
                           self.y.mlu,
                           self.ni,
                           self.hi,
                           self.wi,
                           self.ci,
                           self.no,
                           self.ho,
                           self.wo,
                           self.co,
                           self.hw,
                           self.ww,
                           self.hs,
                           self.ws)
