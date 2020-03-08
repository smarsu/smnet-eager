# Copyright (c) 2020 smarsu. All Rights Reserved.

import ctypes
import numpy as np

from .blob import Tensor, Variable
from .net import Net
from .third_party import nvarray as nv


class Optimizer(object):
  def __init__(self):
    pass


  def _get_blobs(self, blobs):
    if isinstance(blobs, (Tensor, Variable)):
      return [blobs]
    else:
      return blobs


  def _get_backlayers_variables(self, blobs):
    net = Net()
    for blob in blobs:
      net.merge_net(blob.net)
      
    self._backlayers, self._variables = net.get_backlayers_variables(
      blobs)


  def save(self, path, blobs):
    self._blobs = self._get_blobs(blobs)
    self._get_backlayers_variables(self._blobs)

    import os
    os.makedirs(os.path.split(path)[0], exist_ok=True)

    variable_dict = {variable.name: variable.data 
                     for variable in self._variables}
    np.savez(path, **variable_dict)


class SGD(Optimizer):
  def __init__(self, lr=0.1, momentum=0.9, weight_decay=5e-4):
    self._lr = lr
    self._momentum = momentum
    self._weight_decay = weight_decay

    self._variable_momentum = {}

  
  def minimum(self, blobs):
    self._blobs = self._get_blobs(blobs)
    self._get_backlayers_variables(self._blobs)

    for blob in self._blobs:
      blob.set_grad(np.full(blob.shape, self._lr, dtype=blob.dtype))
    
    for layer in self._backlayers:
      layer.backward()

    # if self._weight_decay != 0:
    #   for varaible in self._variables:
    #     varaible.set_grad(varaible.grad + self._lr * self._weight_decay * varaible.data)

    if self._momentum != 0:
      for varaible in self._variables:
        if varaible._grad_device == 'cpu':
          mmt = varaible.momentum * self._momentum + varaible.grad
          varaible.set_grad(mmt)
          varaible.set_momentum(mmt)
        elif varaible._grad_device == 'gpu':
          nv.libsmnv.Assign(varaible.size, varaible.gpu_grad, ctypes.c_float(self._momentum), varaible.gpu_momentum)
          nv.libsmnv.Assign(varaible.size, varaible.gpu_momentum, ctypes.c_float(0), varaible.gpu_grad)

    for varaible in self._variables:
      varaible.add_grad()
