// Copyright (c) 2020 smarsu. All Rights Reserved.

#include "core.h"

namespace kernel {

__device__ int IndexConvert(int idx, int ndims, const int *x_shape, const int *y_shape) {
  int sum = 0;
  int prod = 1;
  for (int i = ndims - 1; i >= 0; --i) {
    int dim = idx % x_shape[i];
    idx /= x_shape[i];

    dim = min(dim, y_shape[i] - 1);

    sum += prod * dim;
    prod *= y_shape[i];
  }

  return sum;
}

template <typename T>
__global__ void Broadcast(int nthreads,
                          const T *x,
                          T beta,
                          T *y,
                          int ndims,
                          const int *x_shape,
                          const int *y_shape) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < nthreads) {
    int x_idx = kernel::IndexConvert(idx, ndims, y_shape, x_shape);
    if (beta == 0) {
      y[idx] = x[x_idx];
    }
    else {
      y[idx] = x[x_idx] + beta * y[idx];
    }
  }
}

}  // namespace kernel

extern "C" {
  
bool Broadcast(int y_size,
               const float *x,
               float beta,
               float *y,
               int ndims,
               const int *x_shape,
               const int *y_shape) {
  kernel::Broadcast<float>
    <<<CUDA_GET_BOLCKS(y_size),
       CUDA_NUM_THREADS>>>(y_size,
                           x,
                           beta,
                           y,
                           ndims,
                           x_shape,
                           y_shape);

  CALL_CUDA(cudaPeekAtLastError());
  return true;
}

}  // extern "C"
