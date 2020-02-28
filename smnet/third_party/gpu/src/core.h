// Copyright (c) 2020 smarsu. All Rights Reserved.

#pragma once
#include "cuda_runtime_api.h"
#include "cudnn.h"

#include "glog/logging.h"

#define CALL_CUDA(x)                                         \
  CHECK_EQ(static_cast<cudaError_t>(x), CUDA_STATUS_SUCCESS) \
    << cudaGetErrorString(x)

#define CALL_CUBLAS(x) \
  CHECK_EQ(static_cast<cublasStatus_t>(x), CUBLAS_STATUS_SUCCESS)

#define CALL_CUDNN(x)                                           \
  CHECK_EQ(static_cast<cudnnStatus_t>(x), CUDNN_STATUS_SUCCESS) \
    << cudnnGetErrorString(x)

cudnnHandle_t *CudnnHandle() {
  static cudnnHandle_t *handle = nullptr;
  if (!handle) {
    CALL_CUDNN(cudnnCreate(&handle)) << " Create cudnn handle failed."; 
  }
  return handle;
}

void *CudaMalloc(size_t size) {
  void *ptr = nullptr;
  CALL_CUDA(cudaMalloc(&ptr, size)) << " cuda malloc " << size << " failed.";
  return ptr;
}

void CudaFree(void *ptr) {
  if (ptr) {
    CALL_CUDA(cudaFree(ptr));
  }
}

struct CudaBuffer {
  static CudaBuffer *Buffer(size_t size = 0) {
    static CudaBuffer *buffer = new CudaBuffer;
    return buffer;
  }

  CudaBuffer(size_t size = 0) {
    Resize(size);
  }

  void Resize(size_t size) {
    if (size > capacity_) {
      CudaFree(data_);
      data_ = CudaMalloc(size);
      capacity_ = size;
    }
  }

  ~CudaBuffer() {
    CudaFree(data_);
  }

  void *data_{0};
  size_t capacity_{0};
};
