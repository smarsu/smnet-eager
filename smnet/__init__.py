# Copyright (c) 2020 smarsu. All Rights Reserved.

from .layers import *
from .modules import Conv2D
from .blob import Tensor, Variable, save, restore
from .optimizer import *

from .third_party import nvarray as nv
from . import manager


def reset():
  manager.tensor_id = 0
