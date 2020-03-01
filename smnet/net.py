# Copyright (c) 2020 smarsu. All Rights Reserved.

from . import manager

class Net(object):
  def __init__(self, blob=None):
    self._layers = []
    self._backlayers = []

    self._tensor2layer = {}
    self._tensor2tensors = {}

    self.blobs = set()
    self.blobs_name = set()
    if blob is not None:
      self.blobs.add(blob)
      self.blobs_name.add(blob.name)

  
  # def __del__(self):
  #   self.clear()

  
  def merge_net(self, other):
    for layer in other._layers:
      if layer not in self._layers:
        self._layers.append(layer)

    for tensor in other._tensor2layer:
      if tensor in self._tensor2layer:
        assert other._tensor2layer[tensor] == self._tensor2layer[tensor]
      else:
        self._tensor2layer[tensor] = other._tensor2layer[tensor]

    for tensor in other._tensor2tensors:
      if tensor in self._tensor2tensors:
        assert other._tensor2tensors[tensor] == self._tensor2tensors[tensor]
      else:
        self._tensor2tensors[tensor] = other._tensor2tensors[tensor]

    for blob in other.blobs:
      if blob not in self.blobs:
        self.blobs.add(blob)
        assert blob.name not in self.blobs_name, '{} ... {}\n{} ... {}'.format(blob.name, self.blobs_name, id(blob), [id(b) for b in self.blobs])
        self.blobs_name.add(blob.name)
  

  def add_layer(self, layer):
    # if len(self._layers) <= 0 or layer != self._layers[-1]:
    #   self._layers.append(layer)
    if layer not in self._layers:
      self._layers.append(layer)


  def add_flow(self, intensors, layer, outtensor):
    from .blob import Tensor, Variable
    assert isinstance(outtensor, (Tensor, Variable))
    for tensor in intensors:
      assert isinstance(tensor, (Tensor, Variable))

    self._tensor2layer[outtensor] = layer
    self._tensor2tensors[outtensor] = list(intensors)


  def empty(self):
    return len(self._layers) == 0

  
  def clear(self):
    self._layers = []
    self._backlayers = []

    self._tensor2layer = {}
    self._tensor2tensors = {}    

    # Variable._id = 0
    manager.tensor_id = 0


  def get_backlayers_variables(self, blobs):
    from .blob import Tensor, Variable

    for blob in blobs:
      assert isinstance(blob, (Tensor, Variable))

    layers = set()
    blobs = list(blobs)
    blob_passed = set()
    variable_passed = set()
    while blobs:
      blob = blobs.pop()
      if blob in blob_passed:
        continue

      blob_passed.add(blob)
      if isinstance(blob, Variable):
        variable_passed.add(blob)

      if blob in self._tensor2layer:
        layers.add(self._tensor2layer[blob])
      if blob in self._tensor2tensors:
        blobs += self._tensor2tensors[blob]
    
    shrink_backlayers = []
    for layer in self._layers[::-1]:
      if layer in layers:
        shrink_backlayers.append(layer)

    return shrink_backlayers, variable_passed
