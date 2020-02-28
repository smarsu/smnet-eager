# Copyright (c) 2020 smarsu. All Rights Reserved.

import numpy as np

from .net import Net


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
      self._variables[name].copy_from(data)


variable_manager = VariableManager()


def save(path):
  variable_manager.save(path)


def restore(path):
  variable_manager.restore(path)


class Blob(object):
  def __init__(self, data=None, dtype=np.float32, name=None, type='Blob'):
    self._net = Net()

    self._grad_seted = False
    self._dtype = dtype

    if data is not None:
      self.copy_from(data)

    self._grad = None

    self._type = type
    self._name = name
  

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


  @property
  def data(self):
    return self._data


  @property
  def grad(self):
    if self._grad is None:
      self._grad = np.empty_like(self._data)
    return self._grad

  
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
    super(Variable, self).__init__(data=data, dtype=dtype, 
                                   name=self._get_name(name), type='Variable')
    variable_manager.add_variable(self)

  
  def _get_name(self, name):
    if not name:
      name = ''
    Variable._id += 1
    return name + str(Variable._id)
  

  def restore(self, data):
    self.copy_from(data)


  def add_grad(self):
    self._data -= self._grad
    self._grad_seted = False


class Tensor(Blob):
  _id = 0

  def __init__(self, data=None, dtype=np.float32, name=None):
    super(Tensor, self).__init__(data=data, dtype=dtype, 
                                 name=self._get_name(name), type='Tensor')

  
  def _get_name(self, name):
    # if not name:
    #   name = ''
    # Tensor._id += 1
    # return name + str(Tensor._id)
    return name
