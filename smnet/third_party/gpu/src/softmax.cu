// Copyright (c) 2020 smarsu. All Rights Reserved.

#include "core.h"

struct SoftmaxParams {
  cudnnTensorDescriptor_t x_desc;
  cudnnTensorDescriptor_t y_desc;

  cudnnSoftmaxAlgorithm_t algo;

  float fwd_alpha{1};
  float fwd_beta{0};

  float bwd_alpha{1};
  float bwd_beta{0};
};

extern "C" {

void DestroySoftmaxParams(SoftmaxParams *params) {
  CALL_CUDNN(cudnnDestroyTensorDescriptor(params->x_desc));
  CALL_CUDNN(cudnnDestroyTensorDescriptor(params->y_desc));

  delete params;
}

cudnnSoftmaxAlgorithm_t GetSoftmaxMode(int algo) {
  static cudnnSoftmaxAlgorithm_t algos[2] = {
    CUDNN_SOFTMAX_ACCURATE,
    CUDNN_SOFTMAX_LOG
  };

  return algos[algo];
}

SoftmaxParams *CudnnSoftmaxCreate(int n,
                                  int c,
                                  int h,
                                  int algo) {
  SoftmaxParams *params = new SoftmaxParams;

  params->algo = GetSoftmaxMode(algo);

  CALL_CUDNN(cudnnCreateTensorDescriptor(&params->x_desc));
  CALL_CUDNN(cudnnSetTensor4dDescriptor(params->x_desc,
                                        CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT,
                                        n,
                                        c,
                                        h,
                                        1));

  CALL_CUDNN(cudnnCreateTensorDescriptor(&params->y_desc));
  CALL_CUDNN(cudnnSetTensor4dDescriptor(params->y_desc,
                                        CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT,
                                        n,
                                        c,
                                        h,
                                        1));

  return params;
}

void CudnnSoftmaxForward(cudnnHandle_t cudnn_handle,
                         SoftmaxParams *params,
                         float alpha,
                         const void *x,
                         float beta,
                         void *y) {
  params->fwd_alpha = alpha;
  params->fwd_beta = beta;

  CALL_CUDNN(cudnnSoftmaxForward(cudnn_handle,
                                 params->algo,
                                 CUDNN_SOFTMAX_MODE_CHANNEL,
                                 &params->fwd_alpha,
                                 params->x_desc,
                                 x,
                                 &params->fwd_beta,
                                 params->y_desc,
                                 y));
}

void CudnnSoftmaxBackward(cudnnHandle_t cudnn_handle,
                          SoftmaxParams *params,
                          float alpha,
                          const void *y,
                          const void *dy,
                          float beta,
                          void *dx) {
  params->bwd_alpha = alpha;
  params->bwd_beta = beta;

  CALL_CUDNN(cudnnSoftmaxBackward(cudnn_handle,
                                  params->algo,
                                  CUDNN_SOFTMAX_MODE_CHANNEL,
                                  &params->bwd_alpha,
                                  params->y_desc,
                                  y,
                                  params->y_desc,
                                  dy,
                                  &params->bwd_beta,
                                  params->x_desc,
                                  dx));
}

}  // extern "C" 
