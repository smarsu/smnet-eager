# Copyright (c) 2020 smarsu. All Rights Reserved.

import numpy as np

from . import _math_utils as math_utils
from ..layer import Layer


class SoftmaxCrossEntropyWithLogits(Layer):
  def __init__(self, labels, logits, axis=-1, name=None):
    super(SoftmaxCrossEntropyWithLogits, self).__init__(name=name)
    self._setup(labels, logits, axis)

  
  def _setup(self, labels, logits, axis):
    self.labels = self._to_tensor(labels)
    self.logits = self._to_tensor(logits)
    self.res = self._res_tensor([self.labels, self.logits])

    self.axis = axis
    self._clip_value = 1e-6

  
  def forward(self):
    self.softmax_logits = math_utils.softmax(self.logits.data, self.axis)
    self.p = -np.log(np.maximum(self.softmax_logits, self._clip_value))
    # self.p = -np.log(self.softmax_logits)
    self.res.feed(self.labels.data * self.p)

  
  def backward(self):
    grad = self.res.grad

    self.labels.feed_grad(grad * self.p)

    # 1. Prepare data
    labels = self.labels.data
    softmax_logits = self.softmax_logits

    lgt_grad = np.zeros(shape=self.logits.shape, dtype=self.logits.dtype)

    softmax_logits = np.where(
      softmax_logits>self._clip_value, softmax_logits, 0)
    # 2. Compute and add grad
    for i in range(labels.shape[-1]):
        softmax_logits_i = np.copy(softmax_logits)
        softmax_logits_i[..., i] -= 1
        lgt_grad += grad[..., i:i+1] * labels[..., i:i+1] * softmax_logits_i

    self.logits.feed_grad(lgt_grad)

    # softmax_logits = np.where(
    #     self.softmax_logits>self._clip_value, self.softmax_logits, 0)
    # math_utils.softmax_cross_entropy_with_logits_backward_logits(self.logits.grad,
    #                                                              grad,
    #                                                              self.labels.data,
    #                                                              softmax_logits,
    #                                                              self.axis,
    #                                                              1 if self.logits._grad_seted else 0)


def softmax_cross_entropy_with_logits(labels, logits, axis=-1, name=None):
  layer = SoftmaxCrossEntropyWithLogits(labels, logits, axis, name)
  layer.forward()
  return layer.res
