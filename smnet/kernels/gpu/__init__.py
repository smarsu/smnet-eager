# Copyright (c) 2020 smarsu. All Rights Reserved.

from .broadcast import BroadcastKernel
from .conv2d import Conv2DKernel
from .elementwise import BinaryElementwiseKernel
from .pad import PadConstNCHWKernel
from .pool2d import Pool2DKernel

# #define CUDA_KERNEL_LOOP(i, n) \
#   for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
#        i < (n); \
#        i += blockDim.x * gridDim.x)
