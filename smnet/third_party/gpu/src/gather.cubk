// Copyright (c) 2020 smarsu. All Rights Reserved.

#include "core.h"

namespace kernel {

template <typename T, typename IT>
__global__ void Gather(int nthread,
                       const T *x,
                       const IT *index,
                       float beta,
                       T *y,
                       int n,
                       int t,
                       int c) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < nthreads) {
    int _n = idx / (t * c);
    int _t = idx % (t * c) / c;
    int _c = idx % c;

    IT indice = index[To2DIndex(_n, _t, 0, t)];
    if (beta == 0) {
      y[idx] = x[To2DIndex(_n, indice, 0, c)];
    }
    else {
      y[idx] = x[To2DIndex(_n, indice, 0, c)] + beta * y[idx];
    }
  }
}

template <typename T, typename IT>
__global__ void GatherGradient(int nthread,
                               const T *dy,
                               const IT *index,
                               float beta,
                               T *dx,
                               int n,
                               int t,
                               int c) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < nthreads) {
    int _n = idx / (t * c);
    int _t = idx % (t * c) / c;
    int _c = idx % c;

    IT indice = index[To2DIndex(_n, _t, 0, t)];
    if (beta == 0) {
      atomicAdd(dx[To2DIndex(_n, indice, 0, c)], dy[idx]);
    }
    else {
      atomicAdd(dx[To2DIndex(_n, indice, 0, c)], dx[To2DIndex(_n, indice, 0, c)] * beta);
      atomicAdd(dx[To2DIndex(_n, indice, 0, c)], dy[idx]);
    }
  }
}

}  // namespace kernel

extern "C" {

bool GatherF32I32(int y_size,
                  const float *x,
                  const int *index,
                  float beta,
                  float *y,
                  int n,
                  int t,
                  int c) {
  kernel::Gather<float, int32_t>
    <<<CUDA_GET_BLOCKS(y_size),
       CUDA_NUM_THREADS>>>(y_size,
                           x,
                           index,
                           beta,
                           y,
                           n,
                           t,
                           c);
  CALL_CUDA(cudaPeekAtLastError());
  return true;
}

bool GatherGradientF32I32(int nthread,
                          const float *dy,
                          const int *index,
                          float beta,
                          float *dx,
                          int n,
                          int t,
                          int c) {
  kernel::GatherGradient<float, int32_t>
    <<<CUDA_GET_BLOCKS(y_size),
       CUDA_NUM_THREADS>>>(y_size,
                           dy,
                           index,
                           beta,
                           dx,
                           n,
                           t,
                           c);
  CALL_CUDA(cudaPeekAtLastError());
  return true;
}

}  // extern "C"
