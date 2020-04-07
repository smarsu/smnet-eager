# Copyright (c) 2020 smarsu. All Rights Reserved.

import ctypes
from ...third_party import cnarray as cn


class MluConv2DKernel(object):
  def __init__(self, x, w, y, pads, strides, dilations, bias=None, with_relu=False):
    self.x = x
    self.w = w
    self.y = y
    self.bias = bias

    self.ni, self.hi, self.wi, self.ci = self.x.shape
    self.co, self.hf, self.wf, self.ci = self.w.shape
    self.ni, self.ho, self.wo, self.co = self.y.shape

    self.hp, self.wp = pads
    self.hs, self.ws = strides
    self.hd, self.wd = dilations

    assert self.hp == self.wp == 0, [self.hp, self.wp]
    assert self.hf * self.wf <= 31, 'hs * ws <= 32, {} * {} -> {}'.format(self.hf, self.wf, self.hf * self.wf)

    self.with_relu = with_relu

  
  def forward(self):
    cn.libsmcn.Conv2D(self.x.mlu,
                      self.w.mlu,
                      self.bias.mlu if self.bias is not None else None,
                      self.y.mlu,
                      self.ni,
                      self.co,
                      self.ho,
                      self.wo,
                      self.ni,
                      self.ci,
                      self.hi,
                      self.wi,
                      self.co,
                      self.ci,
                      self.hf,
                      self.wf,
                      self.hs,
                      self.ws,
                      0,
                      0,
                      0,
                      0,
                      ctypes.c_bool(True if self.bias is not None else False),
                      ctypes.c_bool(self.with_relu))
