# Copyright (c) 2020 smarsu. All Rights Reserved.

import numpy as np

from .third_party import nvarray as nv
from .net import Net
from . import manager


class VariableManager(object):
  def __init__(self):
    self._variables = {}
    self._names = set()


  def _get_unique_name(self, name):
    cnt = 1
    while name in self._names:
      name = name + '_' + str(cnt)
      cnt += 1
    return name


  def add_variable(self, variable):
    name = self._get_unique_name(variable.name)
    self._names.add(name)
    self._variables[name] = variable


  def save(self, path):
    import os
    os.makedirs(os.path.split(path)[0], exist_ok=True)

    variable_dict = {name: variable.data 
                     for name, variable in self._variables.items()}
    np.savez(path, **variable_dict)


  def restore(self, path):
    """Load the variables' data from the .npz file.
    
    Args:
        path: The .npz file stores the variables.
    Raise:
        NameError: path not endwith .npz
    """
    variable_dict = np.load(path)

    for name, data in variable_dict.items():
      if name in self._variables:
        self._variables[name].copy_from(data)
      else:
        print("[smnet]: Not found variable {} int {}".format(name, path))


variable_manager = VariableManager()


def save(path):
  variable_manager.save(path)


def restore(path):
  variable_manager.restore(path)


class Blob(object):
  def __init__(self, data=None, dtype=np.float32, name=None, type='Blob'):
    self._grad_seted = False
    self._dtype = dtype
    self._data_device = self._grad_device = 'cpu'

    self._data = None

    self._gpu = None
    self._gpu_grad = None

    if data is not None:
      self.copy_from(data)

    self._grad = None

    self._type = type
    self._name = name

    self._net = Net(self)


  def __del__(self):
    pass
    # if self._gpu is not None:
    #   del self._gpu
    # if self._gpu_grad is not None:
    #   del self._gpu_grad
  

  def copy_from(self, data):
    data = np.array(data, dtype=self._dtype, ndmin=1)
    self.feed(data)


  def share_from(self, data):
    data = np.array(data, dtype=self._dtype, copy=False, ndmin=1)
    self.feed(data)

  
  def feed(self, data):
    self._data = data
    self._shape = data.shape
    self._size = data.size

  
  def feed_grad(self, grad):
    if not self._grad_seted:
      self._grad = grad
    else:
      self._grad += grad
    self._grad_seted = True


  def set_grad(self, grad):
    """Set initialized grad for loss Tensor."""
    self._grad = grad
    self._grad_seted = True


  def reshape(self, shape):
    self._shape = shape
    self._size = int(np.prod(shape))


  def to_gpu(self, type='data'):
    if type == 'data':
      if self._data_device == 'gpu':
        return

      if self._data is not None:
        self._gpu = nv.array(data=self._data, dtype=self._dtype, name=self._name)
      else:
        self._gpu = nv.array(shape=self._shape, nbytes=4 * self._size, dtype=self._dtype, name=self._name)

      self._data_device = 'gpu'

    if type == 'grad':
      if self._grad_device == 'gpu':
        return

      if self._grad_seted and self._grad is not None:
        self._gpu_grad = nv.array(data=self._grad, dtype=self._dtype, name=self._name + '_grad')
      else:
        self._gpu_grad = nv.array(shape=self._shape, nbytes=4 * self._size, dtype=self._dtype, name=self._name + '_grad')

      self._grad_device = 'gpu'

  
  def to_cpu(self, type='data'):
    if type == 'data':
      if self._data_device == 'cpu':
        return

      self._data = self._gpu.numpy

      self._data_device = 'cpu'

    if type == 'grad':
      if self._grad_device == 'cpu':
        return

      if self._grad_seted:
        self._grad = self._gpu_grad.numpy
      else:
        self._grad = np.empty(shape=self._shape, dtype=self._dtype)

      self._grad_device = 'cpu'


  @property
  def data(self):
    # if self._gpu is not None:
    #   self._data = self._gpu.numpy

    # return self._data
    self.to_cpu('data')

    return self._data


  @property
  def gpu(self):
    # if self._gpu is None:
    #   if self._data is None:
    #     self._gpu = nv.array(shape=self._shape, nbytes=4 * self._size, dtype=self._dtype)
    #   else:
    #     self._gpu = nv.array(data=self._data, dtype=self._dtype)
    
    # return self._gpu.gpu
    self.to_gpu('data')

    return self._gpu.gpu


  @property
  def grad(self):
    # if self._gpu_grad is not None and self._grad_seted:
    #     self._grad = self._gpu_grad.numpy
    # elif self._grad is None:
    #   self._grad = np.empty_like(self._data)

    # return self._grad
    self.to_cpu('grad')

    if self._grad is None:
      self._grad = np.empty_like(self._data)

    return self._grad

  
  @property
  def gpu_grad(self):
    # if self._gpu_grad is None:
    #   if not self._grad_seted:
    #     self._gpu_grad = nv.array(shape=self.shape, nbytes=4 * self._size, dtype=self._dtype)
    #   else:
    #     self._gpu_grad = nv.array(data=self._grad, dtype=self._dtype)
    
    # return self._gpu_grad.gpu
    self.to_gpu('grad')

    return self._gpu_grad.gpu

  
  @property
  def shape(self):
    return self._shape

  
  @property
  def size(self):
    return self._size

  
  @property
  def dtype(self):
    return self._dtype

  
  @property
  def name(self):
    return self._name

  
  @property
  def net(self):
    return self._net


  def __add__(self, other):
    from .layers import add
    return add(self, other)


  def __sub__(self, other):
    from .layers import sub
    return sub(self, other)


  def __mul__(self, other):
    from .layers import mul
    return mul(self, other)


  def __truediv__(self, other):
    from .layers import div
    return div(self, other)


  def __radd__(self, other):
    from .layers import add
    return add(other, self)


  def __rsub__(self, other):
    from .layers import sub
    return sub(other, self)

  
  def __rmul__(self, other):
    from .layers import mul
    return mul(other, self)


  def __rtruediv__(self, other):
    from .layers import div
    return div(other, self)


  def __repr__(self):
    return '<sm.{} {} shape={} dtype={}>'.format(self._type, self._name, self._shape, self._dtype)


class Variable(Blob):
  _id = 0

  def __init__(self, data, dtype=np.float32, name=None):
    super(Variable, self).__init__(data=data, 
                                   dtype=dtype, 
                                   name=self._get_name(name), 
                                   type='Variable')
    variable_manager.add_variable(self)

  
  def _get_name(self, name):
    if not name:
      name = ''
    Variable._id += 1
    return name + str(Variable._id)


  def restore(self, data):
    self.copy_from(data)


  def add_grad(self):
    if self._data_device == self._grad_device == 'gpu':
      self._gpu = self._gpu - self._gpu_grad
    elif self._data_device == self._grad_device == 'cpu':
      self._data -= self._grad
    else:
      assert 0, '{} ... {}'.format(self._data_device, self._grad_device)

    self._grad_seted = False


class Tensor(Blob):
  def __init__(self, data=None, dtype=np.float32, name=None):
    super(Tensor, self).__init__(data=data,
                                 dtype=dtype, 
                                 name=self._get_name(name),
                                 type='Tensor')

  
  def _get_name(self, name):
    if not name:
      name = ''
    manager.tensor_id += 1
    return name + str(manager.tensor_id) + '_Tensor'
