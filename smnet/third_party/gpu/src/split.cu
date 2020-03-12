// Copyright (c) 2020 smarsu. All Rights Reserved.

#include "core.h"

namespace kernel {

template <typename T>
__global__ void Split(int nthreads,
                      const T *x,
                      T beta,
                      T **y,
                      int n,
                      int t,
                      int c,
                      int nsplit,
                      const int *split_dims) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < nthreads) {
    int _c = idx % c;
    int up = idx / c;
    int _t = up % t;
    int _n = up / t;

    int ny;
    int yT;
    for (int i = 0; i < nsplit; ++i) {
      if (_t < split_dims[i + 1]) {
        ny = i;
        _t -= split_dims[i];
        yT = split_dims[i + 1] - split_dims[i];
        break;
      }
    }

    int yidx = To3DIndex(_n, _t, _c, 0, yT, c);
    if (beta == 0) {
      y[ny][yidx] = x[idx];
    }
    else {
      y[ny][yidx] = x[idx] + beta * y[ny][yidx];
    }
  }
}

}  // namespace kernel

extern "C" {

bool Split(int x_size,
           const float *x,
           float beta,
           float **y,
           int n,
           int t,
           int c,
           int nsplit,
           const int *split_dims) {
  kernel::Split<float>
    <<<CUDA_GET_BLOCKS(x_size),
       CUDA_NUM_THREADS>>>(x_size,
                           x,
                           beta,
                           y,
                           n,
                           t,
                           c,
                           nsplit,
                           split_dims);
  
  CALL_CUDA(cudaPeekAtLastError());
  return true;
}

}  // extern "C"

