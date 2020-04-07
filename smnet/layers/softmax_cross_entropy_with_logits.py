# Copyright (c) 2020 smarsu. All Rights Reserved.

import glog
import numpy as np

from . import _math_utils as math_utils
from ..blob import Tensor
from ..layer import Layer

from ..third_party import nvarray as nv
if nv.with_cuda is True:
  from ..kernels import SoftmaxKernel, BinaryElementwiseKernel


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

    if self.labels.need_grad:
      self.labels.feed_grad(grad * self.p)

    if self.logits.need_grad:
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


class GpuSoftmaxCrossEntropyWithLogits(SoftmaxCrossEntropyWithLogits):
  def __init__(self, labels, logits, axis=-1, name=None):
    super(GpuSoftmaxCrossEntropyWithLogits, self).__init__(labels, logits, axis, name)


  def __del__(self):
    del self.log_sft
    del self.softmax_kernel
    del self.mul_kernel


  def forward(self):
    self.res.reshape(self.logits.shape)

    self.log_sft = Tensor(dtype=self.logits.dtype, need_grad=True)
    self.log_sft.reshape(self.logits.shape)

    self.softmax_kernel = SoftmaxKernel(self.logits, self.log_sft, self.axis, 1)  # 1 for CUDNN_SOFTMAX_LOG
    self.softmax_kernel.forward(1)

    self.mul_kernel = BinaryElementwiseKernel(self.labels, self.log_sft, self.res, 'Mul')
    self.mul_kernel.forward()


  def backward(self):
    self.mul_kernel.backward()
    self.softmax_kernel.backward(1)


def softmax_cross_entropy_with_logits(labels, logits, axis=-1, name=None, device='cpu'):
  if nv.with_cuda is True:
    device = 'cpu'

  if device == 'gpu':
    layer = GpuSoftmaxCrossEntropyWithLogits(labels, logits, axis, name)
  else:
    layer = SoftmaxCrossEntropyWithLogits(labels, logits, axis, name)

  layer.forward()

  glog.info('Run {} SoftmaxCrossEntropy Layer ... <{}> -> <{}>'.format(
    device, layer.logits.shape, layer.res.shape))

  if device == 'gpu':
    return -1 * layer.res
  else:
    return layer.res
