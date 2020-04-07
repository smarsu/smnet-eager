# Copyright (c) 2020 smarsu. All Rights Reserved.

from .layers import *
from .modules import Conv2D
from .blob import Tensor, Variable, save, restore
from .optimizer import *

from .third_party import nvarray as nv
from .third_party import cnarray as cn
from . import manager

import os
import glog
if 'GLOG_minloglevel' in os.environ:
  glog.setLevel((int(os.environ['GLOG_minloglevel']) + 2) * 10)


def reset():
  manager.tensor_id = 0
