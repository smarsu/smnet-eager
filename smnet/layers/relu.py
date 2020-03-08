# Copyright (c) 2020 smarsu. All Rights Reserved.

import numpy as np

from ..layer import Layer
from ..kernels import ActivationKernel


class Relu(Layer):
  def __init__(self, a, name):
    super(Relu, self).__init__(name=name)
    self._setup(a)

  
  def _setup(self, a):
    self.a = self._to_tensor(a)
    self.res = self._res_tensor([self.a])

  
  def forward(self):
    self.res.feed(np.maximum(self.a.data, 0))


  def backward(self):
    keep = self.a.data > 0
    self.a.feed_grad(self.res.grad * keep)


class GpuRelu(Relu):
  def __init__(self, a, name):
    super(GpuRelu, self).__init__(a, name)


  def __del__(self):
    del self.act_kernel


  def forward(self):
    self.res.reshape(self.a.shape)

    self.act_kernel = ActivationKernel(self.a, self.res, 1, np.inf)
    self.act_kernel.forward()

  
  def backward(self):
    self.act_kernel.backward()
    self.a._grad_seted = True


def relu(a, name=None, device='gpu'):
  if device == 'gpu':
    layer = GpuRelu(a, name)
  else:
    layer = Relu(a, name)

  layer.forward()
  return layer.res
