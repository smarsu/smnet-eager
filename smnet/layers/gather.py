# Copyright (c) 2020 smarsu. All Rights Reserved.

"""Compare with tf, sm gather do not support batch_dims"""

import glog
import numpy as np

from . import _math_utils as math_utils
from ..layer import Layer

from ..third_party import nvarray as nv
if nv.with_cuda is True:
  from ..kernels import GatherKernel


class Gather(Layer):
  def __init__(self,     
               params,
               indices,
               axis=0,
               name=None):
    super(Gather, self).__init__(name=name)
    self._setup(params, indices, axis, batch_dims)


  def _setup(self, params, indices, axis, batch_dims):
    self.params = self._to_tensor(params)
    self.indices = self._to_tensor(indices)
    self.res = self._res_tensor([self.params, self.indices])

    self.axis = axis
    self.batch_dims = batch_dims


  def forward(self):
    self.res.feed(np.take(self.params.data, self.indices.data, self.axis))

  
  def backward(self):
    raise NotImplementedError


class GpuGather(Gather):
  def __init__(self,     
               params,
               indices,
               axis=0,
               name=None):
    super(GpuGather, self).__init__(name=name)


  def __del__(self):
    del self.gather_kernel

  
  def forward(self):
    self.gather_kernel = GatherKernel()
