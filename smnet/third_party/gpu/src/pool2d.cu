// Copyright (c) 2020 smarsu. All Rights Reserved.

#include "core.h"

namespace kernel {

/* Copy from 
 * https://github.com/pytorch/pytorch/blob/master/caffe2/operators/pool_op.cu
 */
template <typename T>
__global__ void AveragePool2DForwardNCHWCUDAKernel(
    const int X_H,
    const int X_W,
    const int Y_H,
    const int Y_W,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_t,
    const int pad_l,
    const bool count_include_pad,
    const T* X,
    T* Y) {
  const int X_HxW = X_H * X_W;
  const int Y_HxW = Y_H * Y_W;
  const int nc = blockIdx.x / Y_H;
  const int yh = blockIdx.x % Y_H;
  const T* X_ptr = X + nc * X_HxW;
  T* Y_ptr = Y + nc * Y_HxW;
  const int xh = yh * stride_h - pad_t;
  const int t = max(xh, 0);
  const int b = min(xh + kernel_h, X_H);
  for (int yw = threadIdx.x; yw < Y_W; yw += blockDim.x) {
    const int xw = yw * stride_w - pad_l;
    const int l = max(xw, 0);
    const int r = min(xw + kernel_w, X_W);
    const T scale = T(1) /
        static_cast<T>(count_include_pad ? kernel_h * kernel_w
                                         : (b - t) * (r - l));
    T sum = 0;
    for (int i = t; i < b; ++i) {
      for (int j = l; j < r; ++j) {
#if __CUDA_ARCH__ >= 350
        sum += __ldg(X_ptr + i * X_W + j);
#else
        sum += X_ptr[i * X_W + j];
#endif
      }
    }
    Y_ptr[yh * Y_W + yw] = sum * scale;
  }
}

template <typename T, bool kCountIncludePad>
__global__ void AveragePool2DBackwardNCHWCUDAKernel(
    const int X_H,
    const int X_W,
    const int Y_H,
    const int Y_W,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_t,
    const int pad_l,
    const T* dY,
    T* dX) {
  const int X_HxW = X_H * X_W;
  const int Y_HxW = Y_H * Y_W;
  const int nc = blockIdx.x / X_H;
  const int hh = blockIdx.x % X_H;
  const T* dY_ptr = dY + nc * Y_HxW;
  T* dX_ptr = dX + nc * X_HxW;
  const int h = hh + pad_t;
  const int t = h < kernel_h ? 0 : (h - kernel_h) / stride_h + 1;
  const int b = min(h / stride_h + 1, Y_H);
  for (int ww = threadIdx.x; ww < X_W; ww += blockDim.x) {
    const int w = ww + pad_l;
    const int l = w < kernel_w ? 0 : (w - kernel_w) / stride_w + 1;
    const int r = min(w / stride_w + 1, Y_W);
    T sum = 0;
    for (int i = t; i < b; ++i) {
      for (int j = l; j < r; ++j) {
        if (kCountIncludePad) {
#if __CUDA_ARCH__ >= 350
          sum += __ldg(dY_ptr + i * Y_W + j);
#else
          sum += dY_ptr[i * Y_W + j];
#endif
        } else {
          const int xh = i * stride_h - pad_t;
          const int xw = j * stride_w - pad_l;
          const int xt = max(xh, 0);
          const int xb = min(xh + kernel_h, X_H);
          const int xl = max(xw, 0);
          const int xr = min(xw + kernel_w, X_W);
#if __CUDA_ARCH__ >= 350
          sum += __ldg(dY_ptr + i * Y_W + j) /
              static_cast<T>((xb - xt) * (xr - xl));
#else
          sum += dY_ptr[i * Y_W + j] / static_cast<T>((xb - xt) * (xr - xl));
#endif
        }
      }
    }
    if (kCountIncludePad) {
      dX_ptr[hh * X_W + ww] = sum / static_cast<T>(kernel_h * kernel_w);
    } else {
      dX_ptr[hh * X_W + ww] = sum;
    }
  }
}

}  // namespace kernel

struct Pool2DParams {
  cudnnTensorDescriptor_t x_desc;
  cudnnTensorDescriptor_t y_desc;
  cudnnPoolingDescriptor_t pool_desc;
  
  float fwd_alpha{1};
  float fwd_beta{0};

  float bwd_alpha{1};
  float bwd_beta{0};
};

extern "C" {

void DestroyPool2DParams(Pool2DParams *params) {
  CALL_CUDNN(cudnnDestroyTensorDescriptor(params->x_desc));
  CALL_CUDNN(cudnnDestroyTensorDescriptor(params->y_desc));
  CALL_CUDNN(cudnnDestroyPoolingDescriptor(params->pool_desc));
  
  delete params;
}

Pool2DParams *CudnnPool2DCreate(int ni,
                                int ci,
                                int hi,
                                int wi,
                                int ho,
                                int wo,
                                int mode,
                                int hw,
                                int ww,
                                int hp,
                                int wp,
                                int hs,
                                int ws) {
  Pool2DParams *params = new Pool2DParams;

  CALL_CUDNN(cudnnCreateTensorDescriptor(&params->x_desc));
  CALL_CUDNN(cudnnSetTensor4dDescriptor(params->x_desc,
                                        CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT,
                                        ni,
                                        ci,
                                        hi,
                                        wi));

  CALL_CUDNN(cudnnCreateTensorDescriptor(&params->y_desc));
  CALL_CUDNN(cudnnSetTensor4dDescriptor(params->y_desc,
                                        CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT,
                                        ni,
                                        ci,
                                        ho,
                                        wo));
  
  cudnnPoolingMode_t pool_mode;
  switch (mode) {
    case 0:
      pool_mode = CUDNN_POOLING_MAX_DETERMINISTIC;
      break;

    case 1:
      pool_mode = CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
      break;

    default:
      LOG(FATAL) << "Unexpected pool mode";
  }
  
  CALL_CUDNN(cudnnCreatePoolingDescriptor(&params->pool_desc));
  CALL_CUDNN(cudnnSetPooling2dDescriptor(params->pool_desc,
                                         pool_mode,
                                         CUDNN_NOT_PROPAGATE_NAN,
                                         hw,
                                         ww,
                                         hp,
                                         wp,
                                         hs,
                                         ws));

    return params;
}

void CudnnPool2DForward(cudnnHandle_t cudnn_handle,
                        Pool2DParams *params,
                        float alpha,
                        const void *x,
                        float beta,
                        void *y) {
  params->fwd_alpha = alpha;
  params->fwd_beta = beta;

  CALL_CUDNN(cudnnPoolingForward(cudnn_handle,
                                 params->pool_desc,
                                 &params->fwd_alpha,
                                 params->x_desc,
                                 x,
                                 &params->fwd_beta,
                                 params->y_desc,
                                 y));
}

void CudnnPool2DBackward(cudnnHandle_t cudnn_handle,
                         Pool2DParams *params,
                         float alpha,
                         const void *y,
                         const void *dy,
                         const void *x,
                         float beta,
                         void *dx) {
  params->bwd_alpha = alpha;
  params->bwd_beta = beta;

  CALL_CUDNN(cudnnPoolingBackward(cudnn_handle,
                                  params->pool_desc,
                                  &params->bwd_alpha,
                                  params->y_desc,
                                  y,
                                  params->y_desc,
                                  dy,
                                  params->x_desc,
                                  x,
                                  &params->bwd_beta,
                                  params->x_desc,
                                  dx));
}

}  // extern "C"
