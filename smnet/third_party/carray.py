# Copyright (c) 2020 smarsu. All Rights Reserved.

import os.path as osp
import ctypes
import numpy as np

def get_smcb_lib_path():
    rlpt = osp.realpath(__file__)
    rldir = osp.split(osp.split(rlpt)[0])[0]
    lbpt = osp.join(rldir, 'third_party', 'lib', 'libsmcb.so')
    return lbpt


ctypes.c_float_p = ctypes.POINTER(ctypes.c_float)
ctypes.c_int_p = ctypes.POINTER(ctypes.c_int)

libsmcb = ctypes.cdll.LoadLibrary(get_smcb_lib_path())
libsmcb.CpuMalloc.restype = ctypes.c_void_p

def c_data(data):
  host_ptr = data.ctypes.data_as(ctypes.c_void_p)
  return host_ptr


class CArray(object):
  def __init__(self, data=None, shape=None, nbytes=None, dtype=np.float32):
    """
    """
    self._data = data
    self._shape = shape
    self._nbytes = nbytes
    self._dtype = dtype
