// Copyright (c) 2020 smarsu. All Rights Reserved.

#include "core.h"

extern "C" {

struct Deconv2DParams {
  cudnnTensorDescriptor_t x_desc;
  cudnnFilterDescriptor_t w_desc;
  cudnnTensorDescriptor_t y_desc;
  cudnnTensorDescriptor_t bias_desc;
  cudnnConvolutionDescriptor_t conv_desc;

  cudnnConvolutionBwdDataAlgo_t fwd_algo;
  cudnnConvolutionFwdAlgo_t bwd_data_algo;
  cudnnConvolutionBwdFilterAlgo_t bwd_filter_algo;

  size_t size = 0;

  float fwd_alpha = 1;
  float fwd_beta = 0;

  float fwd_bias_alpha = 1;
  float fwd_bias_beta = 0;

  float bwd_bias_alpha = 1;
  float bwd_bias_beta = 0;

  float bwd_data_alpha = 1;
  float bwd_data_beta = 0;

  float bwd_filter_alpha = 1;
  float bwd_filter_beta = 0;
};

void DestroyDeconv2DParams(Deconv2DParams *params) {
  CALL_CUDNN(cudnnDestroyTensorDescriptor(params->x_desc));
  CALL_CUDNN(cudnnDestroyFilterDescriptor(params->w_desc));
  CALL_CUDNN(cudnnDestroyTensorDescriptor(params->y_desc));
  CALL_CUDNN(cudnnDestroyTensorDescriptor(params->bias_desc));
  CALL_CUDNN(cudnnDestroyConvolutionDescriptor(params->conv_desc));

  delete params;
}

Deconv2DParams *CudnnDeconv2DCreate(cudnnHandle_t cudnn_handle,
                                    int ni, 
                                    int ci, 
                                    int hi, 
                                    int wi,
                                    int co, 
                                    int hf, 
                                    int wf,
                                    int ho,
                                    int wo,
                                    int hp,
                                    int wp,
                                    int hs,
                                    int ws,
                                    int hd,
                                    int wd) {
  Deconv2DParams *params = new Deconv2DParams;

  CALL_CUDNN(cudnnCreateTensorDescriptor(&params->x_desc));
  CALL_CUDNN(cudnnSetTensor4dDescriptor(params->x_desc,
                                        CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT,
                                        ni,
                                        ci,
                                        hi,
                                        wi));

  CALL_CUDNN(cudnnCreateFilterDescriptor(&params->w_desc));
  CALL_CUDNN(cudnnSetFilter4dDescriptor(params->w_desc,
                                        CUDNN_DATA_FLOAT,
                                        CUDNN_TENSOR_NCHW,
                                        ci,
                                        co,
                                        hf,
                                        wf));

  CALL_CUDNN(cudnnCreateTensorDescriptor(&params->y_desc));
  CALL_CUDNN(cudnnSetTensor4dDescriptor(params->y_desc,
                                        CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT,
                                        ni,
                                        co,
                                        ho,
                                        wo));

  CALL_CUDNN(cudnnCreateTensorDescriptor(&params->bias_desc));
  CALL_CUDNN(cudnnSetTensor4dDescriptor(params->bias_desc,
                                        CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT,
                                        1,
                                        co,
                                        1,
                                        1));

  CALL_CUDNN(cudnnCreateConvolutionDescriptor(&params->conv_desc));
  CALL_CUDNN(cudnnSetConvolution2dDescriptor(params->conv_desc,
                                             hp,
                                             wp,
                                             hs,
                                             ws,
                                             hd,
                                             wd,
                                             CUDNN_CROSS_CORRELATION,
                                             CUDNN_DATA_FLOAT));

  size_t size = 0;
  CALL_CUDNN(cudnnGetConvolutionBackwardDataAlgorithm(cudnn_handle,
                                                      params->w_desc,
                                                      params->x_desc,
                                                      params->conv_desc,
                                                      params->y_desc,
                                                      // CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE,
                                                      CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
                                                      0,
                                                      &params->fwd_algo));
  // params->bwd_data_algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
  CALL_CUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn_handle,
                                                          params->w_desc,
                                                          params->x_desc,
                                                          params->conv_desc,
                                                          params->y_desc,
                                                          params->fwd_algo,
                                                          &size));
  params->size = std::max(params->size, size);

  CALL_CUDNN(cudnnGetConvolutionForwardAlgorithm(cudnn_handle,
                                                 params->y_desc,
                                                 params->w_desc,
                                                 params->conv_desc,
                                                 params->x_desc,
                                                //  CUDNN_CONVOLUTION_FWD_NO_WORKSPACE,
                                                 CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                                 0,
                                                 &params->bwd_data_algo));
  // params->fwd_algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;

  CALL_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle,
                                                     params->y_desc,
                                                     params->w_desc,
                                                     params->conv_desc,
                                                     params->x_desc,
                                                     params->bwd_data_algo,
                                                     &size));
  params->size = std::max(params->size, size);

  CALL_CUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(cudnn_handle,
                                                        params->y_desc,
                                                        params->x_desc,
                                                        params->conv_desc,
                                                        params->w_desc,
                                                        // CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE,
                                                        CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
                                                        0,
                                                        &params->bwd_filter_algo));
  // params->bwd_filter_algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
  CALL_CUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn_handle,
                                                            params->y_desc,
                                                            params->x_desc,
                                                            params->conv_desc,
                                                            params->w_desc,
                                                            params->bwd_filter_algo,
                                                            &size));
  params->size = std::max(params->size, size);

  CudaBuffer::Buffer()->Resize(params->size);
  LOG(INFO) << "Get Deconv2D Params ... " << params->size;

  return params;
}

void CudnnDeconv2DForward(cudnnHandle_t cudnn_handle,
                          Deconv2DParams *params,
                          float alpha,
                          const void *x,
                          const void *w,
                          float beta,
                          void *y) {
  params->fwd_alpha = alpha;
  params->fwd_beta = beta;

  CALL_CUDNN(cudnnConvolutionBackwardData(cudnn_handle,
                                          &params->fwd_alpha,
                                          params->w_desc,
                                          w,
                                          params->x_desc,
                                          x,
                                          params->conv_desc,
                                          params->fwd_algo,
                                          CudaBuffer::Buffer()->data(),
                                          params->size,
                                          &params->fwd_beta,
                                          params->y_desc,
                                          y));
}

void CudnnDeconv2DForwardBias(cudnnHandle_t cudnn_handle,
                              Deconv2DParams *params,
                              float alpha,
                              const float *bias,
                              float beta,
                              float *y) {
  params->fwd_bias_alpha = alpha;
  params->fwd_bias_beta = beta;

  CALL_CUDNN(cudnnAddTensor(cudnn_handle,
                            &params->fwd_bias_alpha,
                            params->bias_desc,
                            bias,
                            &params->fwd_bias_beta,
                            params->y_desc,
                            y));
}

void CudnnDeconv2DBackwardBias(cudnnHandle_t cudnn_handle,
                               Deconv2DParams *params,
                               float alpha,
                               const void *dy,
                               float beta,
                               void *db) {
  params->bwd_bias_alpha = alpha;
  params->bwd_bias_beta = beta;

  CALL_CUDNN(cudnnConvolutionBackwardBias(cudnn_handle,
                                          &params->bwd_bias_alpha,
                                          params->y_desc,
                                          dy,
                                          &params->bwd_bias_beta,
                                          params->bias_desc,
                                          db));
}

void CudnnDeconv2DBackwardData(cudnnHandle_t cudnn_handle,
                               Deconv2DParams *params,
                               float alpha,
                               const void *w,
                               const void *dy,
                               float beta,
                               void *dx) {
  params->bwd_data_alpha = alpha;
  params->bwd_data_beta = beta;
  
  CALL_CUDNN(cudnnConvolutionForward(cudnn_handle,
                                     &params->bwd_data_alpha,
                                     params->y_desc,
                                     dy,
                                     params->w_desc,
                                     w,
                                     params->conv_desc,
                                     params->bwd_data_algo,
                                     CudaBuffer::Buffer()->data(),
                                     params->size,
                                     &params->bwd_data_beta,
                                     params->x_desc,
                                     dx));
}

void CudnnDeconv2DBackwardFilter(cudnnHandle_t cudnn_handle,
                                 Deconv2DParams *params,
                                 float alpha,
                                 const void *x,
                                 const void *dy,
                                 float beta,
                                 void *dw) {
  params->bwd_filter_alpha = alpha;
  params->bwd_filter_beta = beta;

  CALL_CUDNN(cudnnConvolutionBackwardFilter(cudnn_handle,
                                            &params->bwd_filter_alpha,
                                            params->y_desc,
                                            dy,
                                            params->x_desc,
                                            x,
                                            params->conv_desc,
                                            params->bwd_filter_algo,
                                            CudaBuffer::Buffer()->data(),
                                            params->size,
                                            &params->bwd_filter_beta,
                                            params->w_desc,
                                            dw));
}

}  // extern "C"
