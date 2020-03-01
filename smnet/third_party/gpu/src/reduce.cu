// Copyright (c) 2020 smarsu. All Rights Reserved.

#include <vector>

#include "core.h"

struct ReduceParams {
  cudnnTensorDescriptor_t x_desc;
  cudnnTensorDescriptor_t y_desc;
  cudnnReduceTensorDescriptor_t reduce_desc;

  size_t indice_size{0};
  size_t wksp_size{0};
  float alpha{1};
  float beta{0};

  bool need_indices{false};
  
  std::vector<int> x_strides;
  std::vector<int> y_strides;
};

extern "C" {

void DestroyReduceParams(ReduceParams *params) {
  CALL_CUDNN(cudnnDestroyTensorDescriptor(params->x_desc));
  CALL_CUDNN(cudnnDestroyTensorDescriptor(params->y_desc));
  CALL_CUDNN(cudnnDestroyReduceTensorDescriptor(params->reduce_desc));

  delete params;
}

cudnnReduceTensorOp_t GetReduceOp(int reduce_op) {
  static cudnnReduceTensorOp_t ops[9] = {
    CUDNN_REDUCE_TENSOR_ADD,
    CUDNN_REDUCE_TENSOR_MUL,
    CUDNN_REDUCE_TENSOR_MIN,
    CUDNN_REDUCE_TENSOR_MAX,
    CUDNN_REDUCE_TENSOR_AMAX,
    CUDNN_REDUCE_TENSOR_AVG,
    CUDNN_REDUCE_TENSOR_NORM1,
    CUDNN_REDUCE_TENSOR_NORM2,
    CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS
  };

  CHECK_LT(reduce_op, 9);
  CHECK_GE(reduce_op, 0);

  return ops[reduce_op];
}

ReduceParams *CudnnReduceCreate(cudnnHandle_t cudnn_handle,
                                int ndims,
                                int *x_dims,
                                int *y_dims,
                                int reduce_op,
                                bool need_indices) {
  ReduceParams *params = new ReduceParams;
  params->need_indices = need_indices;

  params->x_strides = Shape2Strides(x_dims, ndims);
  params->y_strides = Shape2Strides(y_dims, ndims);

  CALL_CUDNN(cudnnCreateTensorDescriptor(&params->x_desc));
  CALL_CUDNN(cudnnSetTensorNdDescriptor(params->x_desc,
                                        CUDNN_DATA_FLOAT,
                                        ndims,
                                        x_dims,
                                        params->x_strides.data()/* x_strides */));

  CALL_CUDNN(cudnnCreateTensorDescriptor(&params->y_desc));
  CALL_CUDNN(cudnnSetTensorNdDescriptor(params->y_desc,
                                        CUDNN_DATA_FLOAT,
                                        ndims,
                                        y_dims,
                                        params->y_strides.data()/* x_strides */));

  CALL_CUDNN(cudnnCreateReduceTensorDescriptor(&params->reduce_desc));
  CALL_CUDNN(cudnnSetReduceTensorDescriptor(params->reduce_desc,
                                            GetReduceOp(reduce_op),
                                            CUDNN_DATA_FLOAT,
                                            CUDNN_NOT_PROPAGATE_NAN,
                                            need_indices ? CUDNN_REDUCE_TENSOR_FLATTENED_INDICES : CUDNN_REDUCE_TENSOR_NO_INDICES,
                                            CUDNN_32BIT_INDICES));

  CALL_CUDNN(cudnnGetReductionIndicesSize(cudnn_handle,
                                          params->reduce_desc,
                                          params->x_desc,
                                          params->y_desc,
                                          &params->indice_size));
  
  CALL_CUDNN(cudnnGetReductionWorkspaceSize(cudnn_handle,
                                            params->reduce_desc,
                                            params->x_desc,
                                            params->y_desc,
                                            &params->wksp_size));

  size_t size = std::max(params->indice_size, params->wksp_size);
  CudaBuffer::Buffer()->Resize(size);

  LOG(INFO) << "Get Reduce Params ... " << size
            << " From " << ToString(x_dims, ndims) << " To " << ToString(y_dims, ndims);

  return params;
}

void CudnnReduceForward(cudnnHandle_t cudnn_handle,
                        ReduceParams *params,
                        void *indices,
                        float alpha,
                        const void *x,
                        float beta,
                        void *y) {
  params->alpha = alpha;
  params->beta = beta;

  CALL_CUDNN(cudnnReduceTensor(cudnn_handle,
                               params->reduce_desc,
                               indices,
                              //  params->need_indices ? indices : CudaBuffer::Buffer()->data(),
                               params->indice_size,
                               CudaBuffer::Buffer()->data(),
                               params->wksp_size,
                               &params->alpha,
                               params->x_desc,
                               x,
                               &params->beta,
                               params->y_desc,
                               y));

  CALL_CUDA(cudaDeviceSynchronize());
}

}  // extern "C"
