// Copyright (c) 2020 smarsu. All Rights Reserved.

#include "core.h"

namespace kernel {
 
template <typename T>
__global__ void Concat(int nthreads,
                       const T **x,
                       T beta,
                       T *y,
                       int n,
                       int t,
                       int c,
                       int nconcat,
                       const int *concat_dims) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < nthreads) {
    int _c = idx % c;
    int up = idx / c;
    int _t = up % t;
    int _n = up / t;

    int nx;
    int xT;
    for (int i = 0; i < nconcat; ++i) {
      if (_t < concat_dims[i + 1]) {
        nx = i;
        _t -= concat_dims[i];
        xT = concat_dims[i + 1] - concat_dims[i];
        break;
      }
    }

    int xidx = To3DIndex(_n, _t, _c, 0, xT, c);
    if (beta == 0) {
      y[idx] = x[nx][xidx];
    }
    else {
      y[idx] = x[nx][xidx] + beta * y[idx];
    }
  }
}

}  // namespace kernel

extern "C" {

bool Concat(int y_size,
            const float **x,
            float beta,
            float *y,
            int n,
            int t,
            int c,
            int nconcat,
            const int *concat_dims) {
  kernel::Concat<float>
    <<<CUDA_GET_BLOCKS(y_size),
       CUDA_NUM_THREADS>>>(y_size,
                           x,
                           beta,
                           y,
                           n,
                           t,
                           c,
                           nconcat,
                           concat_dims);

  CALL_CUDA(cudaPeekAtLastError());
  return true;
}

};  // extern "C"
