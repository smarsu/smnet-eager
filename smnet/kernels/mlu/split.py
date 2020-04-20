# Copyright (c) 2020 smarsu. All Rights Reserved.

import ctypes
import numpy as np
from ...third_party import cnarray as cn
from ...layers import _math_utils as math_utils


class MluSplitKernel(object):
  def __init__(self, x, res, axis, splt):
    self.x = x
    self.res = res
    self.axis = axis
    self.splt = splt

    self.out_dim, self.dim, self.inner_dim = math_utils.get_3level_shape(self.x.shape, axis)
    self.split_dims = cn.array(np.array([0] + self.splt), dtype=np.int32)
    self.y = cn.list_array([tensor.mlu for tensor in self.res])


  def forward(self):
    cn.libsmcn.Split(self.x.mlu,
                     self.y,
                     self.out_dim,
                     self.dim,
                     self.inner_dim,
                     self.split_dims.gpu,
                     len(self.splt))
