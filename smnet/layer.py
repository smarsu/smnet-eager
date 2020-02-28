# Copyright (c) 2020 smarsu. All Rights Reserved.

import numpy as np

from .blob import Tensor, Variable


class Layer(object):
  def __init__(self, name=None):
    self._name = name


  def _to_tensor(self, data, dtype=np.float32):
    if isinstance(data, (Tensor, Variable)):
      return data
    else:
      return Tensor(data, dtype=dtype)


  def _to_variable(self, data, dtype=np.float32):
    if isinstance(data, (Tensor, Variable)):
      return data
    else:
      return Variable(data, dtype=dtype)

  
  def _res_tensor(self, blobs):
    res = Tensor(name=self._name)

    # for blob in blobs:
    #   if not blob.net.empty():
    #     res._net = blob.net
    #     break
    for blob in blobs:
      res.net.merge_net(blob.net)

    res.net.add_layer(self)
    res.net.add_flow(blobs, self, res)
    return res
