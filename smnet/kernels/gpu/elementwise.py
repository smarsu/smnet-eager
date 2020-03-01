# Copyright (c) 2020 smarsu. All Rights Reserved.

import ctypes
from ...third_party import nvarray as nv


class BinaryElementwiseKernel(object):
  def __init__(self, x, y, z, mode):
    self.x = x
    self.y = y
    self.z = z
    self.mode = mode


  def forward(self):
    if self.mode == 'Add':
      nv.libsmnv.Add(self.x.size, 
                     self.x.gpu, 
                     self.y.gpu, 
                     ctypes.c_float(0), 
                     self.z.gpu, 
                     self.x.size, 
                     self.y.size)
    elif self.mode == 'Sub':
      nv.libsmnv.Sub(self.x.size, 
                     self.x.gpu, 
                     self.y.gpu, 
                     ctypes.c_float(0), 
                     self.z.gpu, 
                     self.x.size, 
                     self.y.size)
    elif self.mode == 'Mul':
      nv.libsmnv.Mul(self.x.size, 
                     self.x.gpu, 
                     self.y.gpu, 
                     ctypes.c_float(0), 
                     self.z.gpu, 
                     self.x.size, 
                     self.y.size)
    elif self.mode == 'Div':
      nv.libsmnv.Div(self.x.size, 
                     self.x.gpu, 
                     self.y.gpu, 
                     ctypes.c_float(0), 
                     self.z.gpu, 
                     self.x.size, 
                     self.y.size)
    else:
      assert 0, self.mode

  
  def backward(self):
    self.backward_x()
    self.backward_y()

  
  def backward_x(self):
    if self.mode == 'Add' or self.mode == 'Sub':
      nv.libsmnv.Assign(self.x.size, 
                        self.z.gpu_grad, 
                        ctypes.c_float(1) if self.x._grad_seted else ctypes.c_float(0),
                        self.x.gpu_grad)
    elif self.mode == 'Mul':
      nv.libsmnv.Mul(self.y.size, 
                     self.y.gpu, 
                     self.z.gpu_grad, 
                     ctypes.c_float(1) if self.x._grad_seted else ctypes.c_float(0),
                     self.x.gpu_grad, 
                     self.y.size, 
                     self.z.size)
    elif self.mode == 'Div':
      nv.libsmnv.Div(self.x.size, 
                     self.z.gpu_grad, 
                     self.y.gpu, 
                     ctypes.c_float(1) if self.x._grad_seted else ctypes.c_float(0),
                     self.x.gpu_grad, 
                     self.x.size, 
                     self.y.size)
    else:
      assert 0, self.mode

  
  def backward_y(self):
    if self.mode == 'Add':
      nv.libsmnv.Assign(self.x.size, 
                        self.z.gpu_grad, 
                        ctypes.c_float(1) if self.y._grad_seted else ctypes.c_float(0),
                        self.y.gpu_grad)
    elif self.mode == 'Sub':
      nv.libsmnv.Neg(self.x.size, 
                     self.z.gpu_grad, 
                     ctypes.c_float(1) if self.y._grad_seted else ctypes.c_float(0),
                     self.y.gpu_grad)
    elif self.mode == 'Mul':
      nv.libsmnv.Mul(self.y.size, 
                     self.x.gpu, 
                     self.z.gpu_grad, 
                     ctypes.c_float(1) if self.y._grad_seted else ctypes.c_float(0),
                     self.y.gpu_grad, 
                     self.y.size, 
                     self.z.size)
    elif self.mode == 'Div':
      nv.libsmnv.DivRGradient(self.x.size,
                              self.x.gpu,
                              self.y.gpu,
                              self.z.gpu_grad,
                              ctypes.c_float(1) if self.y._grad_seted else ctypes.c_float(0),
                              self.y.gpu_grad)
    else:
      assert 0, self.mode
