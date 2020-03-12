// Copyright (c) 2020 smarsu. All Rights Reserved.

#include "core.h"

namespace kernel {

template <typename T>
__global__ void PadConstNCHW(int nthreads, 
                             const T *x, 
                             T *y, 
                             int height,  // padded_height
                             int width,  // padded_width
                             int pad_b,
                             int pad_r,
                             T pad_value) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < nthreads) {
    int n = idx / (height * width);
    int c = idx % (height * width);
    int h = c / width;
    int w = c % width;

    if (h >= height - pad_b || w >= width - pad_r) {
      y[idx] = pad_value;
    }
    else {
      y[idx] = x[To3DIndex(n, h, w, 0, height - pad_b, width - pad_r)];
    }
  }
}

template <typename T>
__global__ void PadConstNCHWGradient(int nthreads, 
                                     const T *dy, 
                                     T beta,
                                     T *dx,
                                     int height,  // height before pad
                                     int width,  // width before pad
                                     int pad_b,
                                     int pad_r) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < nthreads) {
    int n = idx / (height * width);
    int c = idx % (height * width);
    int h = c / width;
    int w = c % width;

    if (beta == 0) {
      dx[idx] = dy[To3DIndex(n, h, w, 0, height + pad_b, width + pad_r)];
    }
    else {
      dx[idx] = dy[To3DIndex(n, h, w, 0, height + pad_b, width + pad_r)] + beta * dx[idx];
    }
  }
}

}  // namespace kernel

extern "C" {

bool PadConstNCHW(int y_size, 
                  const float *x, 
                  float *y, 
                  int height,  // padded_height
                  int width,  // padded_width
                  int pad_b,
                  int pad_r,
                  float pad_value) {
  kernel::PadConstNCHW<float>
    <<<CUDA_GET_BLOCKS(y_size), 
       CUDA_NUM_THREADS>>>(y_size, 
                           x, 
                           y, 
                           height,  // padded_height
                           width,  // padded_width
                           pad_b,
                           pad_r,
                           pad_value);
  CALL_CUDA(cudaPeekAtLastError());
  return true;
}

bool PadConstNCHWGradient(int x_size, 
                          const float *dy, 
                          float beta,
                          float *dx,
                          int height,  // height before pad
                          int width,  // width before pad
                          int pad_b,
                          int pad_r) {
  kernel::PadConstNCHWGradient
    <<<CUDA_GET_BLOCKS(x_size), 
       CUDA_NUM_THREADS>>>(x_size,
                           dy,
                           beta,
                           dx,
                           height,  // padded_height
                           width,  // padded_width
                           pad_b,
                           pad_r);
  CALL_CUDA(cudaPeekAtLastError());
  return true;
}

}  // extern "C"
