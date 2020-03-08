// Copyright (c) 2020 smarsu. All Rights Reserved.

#include <vector>

#include "core.h"

struct ActivationParams {
  cudnnTensorDescriptor_t x_desc;
  cudnnTensorDescriptor_t y_desc;
  cudnnActivationDescriptor_t act_desc;

  float fwd_alpha{1};
  float fwd_beta{0};
  float bwd_alpha{1};
  float bwd_beta{0};

  std::vector<int> strides;
};

extern "C" {

void DestroyActivationParams(ActivationParams *params) {
  CALL_CUDNN(cudnnDestroyTensorDescriptor(params->x_desc));
  CALL_CUDNN(cudnnDestroyTensorDescriptor(params->y_desc));
  CALL_CUDNN(cudnnDestroyActivationDescriptor(params->act_desc));

  delete params;
}

cudnnActivationMode_t GetActivationMode(int mode) {
  CHECK(mode >= 0 && mode < 5);

  static cudnnActivationMode_t activation_modes[5] = {
    CUDNN_ACTIVATION_SIGMOID,
    CUDNN_ACTIVATION_RELU,
    CUDNN_ACTIVATION_TANH,
    CUDNN_ACTIVATION_CLIPPED_RELU,
    CUDNN_ACTIVATION_ELU
  };

  return activation_modes[mode];
}

ActivationParams *CudnnActivationCreate(int ndims,
                                        int *shape,
                                        int mode,
                                        double coef) {
  ActivationParams *params = new ActivationParams;

  CHECK(ndims >= 4 && ndims <= CUDNN_DIM_MAX);

  params->strides = Shape2Strides(shape, ndims);

  CALL_CUDNN(cudnnCreateTensorDescriptor(&params->x_desc));
  CALL_CUDNN(cudnnSetTensorNdDescriptor(params->x_desc,
                                        CUDNN_DATA_FLOAT,
                                        ndims,
                                        shape,
                                        params->strides.data()/* x_strides */));

  CALL_CUDNN(cudnnCreateTensorDescriptor(&params->y_desc));
  CALL_CUDNN(cudnnSetTensorNdDescriptor(params->y_desc,
                                        CUDNN_DATA_FLOAT,
                                        ndims,
                                        shape,
                                        params->strides.data()/* x_strides */));

  CALL_CUDNN(cudnnCreateActivationDescriptor(&params->act_desc));
  CALL_CUDNN(cudnnSetActivationDescriptor(params->act_desc,
                                          GetActivationMode(mode),
                                          CUDNN_NOT_PROPAGATE_NAN,
                                          coef));

  return params;
}

void CudnnActivationForward(cudnnHandle_t cudnn_handle,
                            ActivationParams *params,
                            float alpha,
                            const void *x,
                            float beta,
                            void *y) {
  params->fwd_alpha = alpha;
  params->fwd_beta = beta;

  CALL_CUDNN(cudnnActivationForward(cudnn_handle,
                                    params->act_desc,
                                    &params->fwd_alpha,
                                    params->x_desc,
                                    x,
                                    &params->fwd_beta,
                                    params->y_desc,
                                    y));
}

void CudnnActivationBackward(cudnnHandle_t cudnn_handle,
                             ActivationParams *params,
                             float alpha,
                             const void *y,
                             const void *dy,
                             const void *x,
                             float beta,
                             void *dx) {
  params->bwd_alpha = alpha;
  params->bwd_beta = beta;

  CALL_CUDNN(cudnnActivationBackward(cudnn_handle,
                                     params->act_desc,
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
