# Copyright (c) 2020 smarsu. All Rights Reserved.

import numpy as np

from .blob import Tensor, Variable
from .net import Net


class Optimizer(object):
  def __init__(self):
    pass


class SGD(Optimizer):
  def __init__(self, lr=0.1, momentum=0.9, weight_decay=5e-4):
    self._lr = lr
    self._momentum = momentum
    self._weight_decay = weight_decay

    self._variable_momentum = {}

  
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

    # if self._momentum != 0:
    #   for varaible in self._variables:
    #     mmt = self._variable_momentum.get(varaible.name, 0)
    #     mmt = mmt * self._momentum + varaible.grad
    #     varaible.set_grad(mmt)
    #     self._variable_momentum[varaible.name] = mmt

    for varaible in self._variables:
      varaible.add_grad()
