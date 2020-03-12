# Copyright (c) 2020 smarsu. All Rights Reserved.

import os.path as osp
import ctypes
import numpy as np


def get_smnv_lib_path():
  rlpt = osp.realpath(__file__)
  rldir = osp.split(osp.split(rlpt)[0])[0]
  lbpt = osp.join(rldir, 'third_party', 'lib', 'libsmnv.so')
  return lbpt


ctypes.c_float_p = ctypes.POINTER(ctypes.c_float)
ctypes.c_int_p = ctypes.POINTER(ctypes.c_int)

libsmnv = ctypes.cdll.LoadLibrary(get_smnv_lib_path())
libsmnv.CudnnHandle.restype = ctypes.c_void_p
libsmnv.CudaMalloc.restype = ctypes.c_void_p
libsmnv.CudaArray.restype = ctypes.c_void_p

cudnn_handle = ctypes.c_void_p(libsmnv.CudnnHandle())


def c_data(data):
  host_ptr = data.ctypes.data_as(ctypes.c_void_p)
  return host_ptr


class NvArray(object):
  def __init__(self, data=None, shape=None, nbytes=None, dtype=np.float32):
    """
    """
    self._data = data
    self._shape = shape
    self._nbytes = nbytes
    self._dtype = dtype

    if self._data is not None:
      self._data = np.ascontiguousarray(self._data, dtype=self._dtype)
      self._shape = self._data.shape
      self._nbytes = self._data.nbytes

    self._dev_ptr = ctypes.c_void_p(libsmnv.CudaMalloc(ctypes.c_size_t(self._nbytes)))

    if self._data is not None:
      self.feed(self._data)

    self._capacity = self._nbytes

  
  def __del__(self):
    # Here remove host ptr can solve oom?
    libsmnv.CudaFree(self._dev_ptr)


  def __sub__(self, other):
    size = int(np.prod(self._shape))
    libsmnv.Sub(size, self._dev_ptr, other.gpu, ctypes.c_float(0), self._dev_ptr, size, size)
    return self

  
  def reshape(self, shape):
    self._shape = shape
    self._nbytes = 4 * int(np.prod(self._shape))

    if self._nbytes > self._capacity:
      libsmnv.CudaFree(self._dev_ptr)
      self._dev_ptr = ctypes.c_void_p(libsmnv.CudaMalloc(ctypes.c_size_t(self._nbytes)))

      self._capacity = self._nbytes



  def feed(self, data):
    self._data = np.ascontiguousarray(data, dtype=self._dtype)
    libsmnv.CudaMemcpyHostToDevice(self._dev_ptr, c_data(self._data), self._nbytes)

  
  @property
  def gpu(self):
    return self._dev_ptr


  @property
  def numpy(self):
    data = np.empty(shape=self._shape, dtype=self._dtype)
    libsmnv.CudaMemcpyDeviceToHost(c_data(data), self._dev_ptr, self._nbytes)
    return data


gpu_memory = {}


def list_array(nvarrs):
  ptrs = (ctypes.c_void_p * len(nvarrs))(*[arr for arr in nvarrs])
  dev_ptrs = ctypes.c_void_p(libsmnv.CudaArray(ptrs, len(nvarrs)))
  return dev_ptrs


def array(data=None, shape=None, nbytes=None, dtype=np.float32, name=None):
  if name is None:
    return NvArray(data, shape, nbytes, dtype)
  
  if name in gpu_memory:
    ar = gpu_memory[name]
    shape = data.shape if data is not None else shape
    ar.reshape(shape)
    if data is not None:
      ar.feed(data)
    return ar
  else:
    gpu_memory[name] = NvArray(data, shape, nbytes, dtype)
    return gpu_memory[name]
