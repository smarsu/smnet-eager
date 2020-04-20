// Copyright (c) 2020 smarsu. All Rights Reserved.

#include "core.h"

namespace kernel {

/** @brief
 * @param[in] height: The height of y.
 * @param[in] width: The width of y.
 **/
template <typename T>
__global__ void CropConstNCHW(int nthreads,
                              const T *x,
                              T beta,
                              T *y,
                              int height,
                              int width,
                              int crop_t,
                              int crop_b,
                              int crop_l,
                              int crop_r) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < nthreads) {
    int n = idx / (height * width);
    int c = idx % (height * width);  // This is fake c.
    int h = c / width;
    int w = c % width;

    T this_x = x[To3DIndex(n, h + crop_t, w + crop_l, 0, height + crop_t + crop_b, width + crop_l + crop_r)];
    if (beta == 0) {
      y[idx] = this_x;
    }
    else {
      y[idx] = this_x + beta * y[idx];
    }
  }
}

/** @brief
 * @param[in] height: The height of x.
 * @param[in] width: The width of x.
 **/
template <typename T>
__global__ void CropConstNCHWGradient(int nthreads,
                                      const T *dy,
                                      T beta,
                                      T *dx,
                                      int height,
                                      int width,
                                      int crop_t,
                                      int crop_b,
                                      int crop_l,
                                      int crop_r) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < nthreads) {
    int n = idx / (height * width);
    int c = idx % (height * width);  // This is fake c.
    int h = c / width;
    int w = c % width;

    T this_dy = h >= crop_t && h < height - crop_b && w >= crop_l && w < width - crop_r ?
                dy[To3DIndex(n, h - crop_t, w - crop_l, 0, height - crop_t - crop_b, width - crop_l - crop_r)] :
                0;
    if (beta == 0) {
      dx[idx] = this_dy;
    }
    else {
      dx[idx] = this_dy + beta * dx[idx];
    }
  }
}

}  // namespace kernel

extern "C" {

bool CropConstNCHW(int y_size,
                   const float *x,
                   float beta,
                   float *y,
                   int height,
                   int width,
                   int crop_t,
                   int crop_b,
                   int crop_l,
                   int crop_r) {
  kernel::CropConstNCHW<float>
    <<<CUDA_GET_BLOCKS(y_size),
       CUDA_NUM_THREADS>>>(y_size,
                           x,
                           beta,
                           y,
                           height,
                           width,
                           crop_t,
                           crop_b,
                           crop_l,
                           crop_r);
  CALL_CUDA(cudaPeekAtLastError());
  return true;
}

bool CropConstNCHWGradient(int x_size,
                           const float *dy,
                           float beta,
                           float *dx,
                           int height,
                           int width,
                           int crop_t,
                           int crop_b,
                           int crop_l,
                           int crop_r) {
  kernel::CropConstNCHWGradient<float>
    <<<CUDA_GET_BLOCKS(x_size),
       CUDA_NUM_THREADS>>>(x_size,
                           dy,
                           beta,
                           dx,
                           height,
                           width,
                           crop_t,
                           crop_b,
                           crop_l,
                           crop_r);
  CALL_CUDA(cudaPeekAtLastError());
  return true;
}

}  // extern "C"
