// // Copyright (c) 2020 smarsu. All Rights Reserved.

// #include "core.h"

// extern "C" {

// struct BatchNormParams {

// };

// void DestroyBatchNormParams(BatchNormParams *params) {
//   delete params;
// }

// BatchNormParams *CudnnBatchNormCreate() {
//   BatchNormParams *params = new BatchNormParams;


// }

// void CudnnBatchNormForwardTraining(cudnnHandle_t cudnn_handle,
//                                    BatchNormParams *params,
//                                    float alpha,
//                                    float beta,
//                                    const void *x,
//                                    void *y,
//                                    const void *scale,
//                                    const void *bias,
//                                    double exponentialAverageFactor,
//                                    void *running_mean,
//                                    void *running_variance,
//                                    double epsilon,
//                                    void *save_mean,
//                                    void *save_inv_variance) {
//  cudnnStatus_t cudnnBatchNormalizationForwardTraining(
//       cudnnHandle_t                    handle,
//       cudnnBatchNormMode_t             mode,
//       const void                      *alpha,
//       const void                      *beta,
//       const cudnnTensorDescriptor_t    xDesc,
//       const void                      *x,
//       const cudnnTensorDescriptor_t    yDesc,
//       void                            *y,
//       const cudnnTensorDescriptor_t    bnScaleBiasMeanVarDesc,
//       const void                      *bnScale,
//       const void                      *bnBias,
//       double                           exponentialAverageFactor,
//       void                            *resultRunningMean,
//       void                            *resultRunningVariance,
//       double                           epsilon,
//       void                            *resultSaveMean,
//       void                            *resultSaveInvVariance)
// }

// void CudnnBatchNormForwardInference(cudnnHandle_t cudnn_handle,
//                                     BatchNormParams *params,
//                                     float alpha,
//                                     float beta,
//                                     const void *x,
//                                     void *y,
//                                     const void *scale,
//                                     const void *bias,
//                                     const void *mean,
//                                     const void *variance,
//                                     double epsilon) {
//  cudnnStatus_t cudnnBatchNormalizationForwardInference(
//       cudnnHandle_t                    handle,
//       cudnnBatchNormMode_t             mode,
//       const void                      *alpha,
//       const void                      *beta,
//       const cudnnTensorDescriptor_t    xDesc,
//       const void                      *x,
//       const cudnnTensorDescriptor_t    yDesc,
//       void                            *y,
//       const cudnnTensorDescriptor_t    bnScaleBiasMeanVarDesc,
//       const void                      *bnScale,
//       const void                      *bnBias,
//       const void                      *estimatedMean,
//       const void                      *estimatedVariance,
//       double                           epsilon)
// }

// void CudnnBatchNormBackward(cudnnHandle_t cudnn_handle,
//                             BatchNormParams *params,
//                             float alpha,
//                             float beta,) {
// cudnnStatus_t cudnnBatchNormalizationBackward(
//       cudnnHandle_t                    handle,
//       cudnnBatchNormMode_t             mode,
//       const void                      *alphaDataDiff,
//       const void                      *betaDataDiff,
//       const void                      *alphaParamDiff,
//       const void                      *betaParamDiff,
//       const cudnnTensorDescriptor_t    xDesc,
//       const void                      *x,
//       const cudnnTensorDescriptor_t    dyDesc,
//       const void                      *dy,
//       const cudnnTensorDescriptor_t    dxDesc,
//       void                            *dx,
//       const cudnnTensorDescriptor_t    bnScaleBiasDiffDesc,
//       const void                      *bnScale,
//       void                            *resultBnScaleDiff,
//       void                            *resultBnBiasDiff,
//       double                           epsilon,
//       const void                      *savedMean,
//       const void                      *savedInvVariance)
// }

// }  // extern "C"
