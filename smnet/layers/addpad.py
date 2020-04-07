# Copyright (c) 2020 smarsu. All Rights Reserved.

import glog
import numpy as np

from ..layer import Layer

from ..third_party import nvarray as nv

from ..third_party import cnarray as cn
if cn.with_mlu:
  from ..kernels import MluAddPadKernel


class AddPad(Layer):
  def __init__(self, x, pad_t, pad_b, pad_l, pad_r, pad_value, name=None):
    super(AddPad, self).__init__(name=name)
    self._setup(x, pad_t, pad_b, pad_l, pad_r, pad_value)
  
  
  def _setup(self, x, pad_t, pad_b, pad_l, pad_r, pad_value):
    self.x = self._to_tensor(x)
    self.res = self._res_tensor([self.x])

    self.pad_t = pad_t
    self.pad_b = pad_b
    self.pad_l = pad_l
    self.pad_r = pad_r
    self.pad_value = pad_value

  
  def forward(self):
    assert len(self.x.shape) == 4, len(self.x.shape)
    self.res.feed(np.pad(self.x.data, ((0, 0), (self.pad_t, self.pad_b), (self.pad_l, self.pad_r), (0, 0)), 'constant', constant_values=self.pad_value))

  
class MluAddPad(AddPad):
  def __init__(self, x, pad_t, pad_b, pad_l, pad_r, pad_value, name=None):
    super(MluAddPad, self).__init__(x, pad_t, pad_b, pad_l, pad_r, pad_value, name)
  

  def forward(self):
    assert len(self.x.shape) == 4, len(self.x.shape)

    n, h, w, c = self.x.shape
    self.res.reshape([n, h + self.pad_t + self.pad_b, w + self.pad_l + self.pad_r, c])
    self.addpad_kernel = MluAddPadKernel(self.x, self.res, self.pad_t, self.pad_b, self.pad_l, self.pad_r, self.pad_value)
    self.addpad_kernel.forward()


def addpad(x, pad_t, pad_b, pad_l, pad_r, pad_value, name=None, device='cpu'):
  if nv.with_cuda is True:
    device = 'gpu'
  elif cn.with_mlu is True:
    device = 'mlu'

  if device == 'gpu':
    raise NotImplementedError
  elif device == 'mlu':
    layer = MluAddPad(x, pad_t, pad_b, pad_l, pad_r, pad_value, name)
  else:
    layer = AddPad(x, pad_t, pad_b, pad_l, pad_r, pad_value, name)

  layer.forward()
  glog.info('Run {} AddPad layer ... <{}> -> <{}>'.format(
    device, layer.x.shape, layer.res.shape))
  return layer.res
