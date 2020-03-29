// Copyright (c) 2020 smarsu. All Rights Reserved.

#pragma once
#include <vector>
#include <string>

#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "cudnn.h"

#include "glog/logging.h"

#define CALL_CUDA(x)                                         \
  CHECK_EQ(static_cast<cudaError_t>(x), cudaSuccess) \
    << cudaGetErrorString(x)

#define CALL_CUBLAS(x) \
  CHECK_EQ(static_cast<cublasStatus_t>(x), CUBLAS_STATUS_SUCCESS)

#define CALL_CUDNN(x)                                           \
  CHECK_EQ(static_cast<cudnnStatus_t>(x), CUDNN_STATUS_SUCCESS) \
    << cudnnGetErrorString(x)

#define CUDA_NUM_THREADS (1024)

#define CUDA_GET_BLOCKS(n) (((n) + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS)

#define To3DIndex(x, y, z, X, Y, Z) \
  (((x) * (Y) + (y)) * (Z) + (z))

#define To2DIndex(x, y, X, Y) \
  ((x) * (Y) + (y))

std::string ToString(int *shape, int ndims);

std::vector<int> Shape2Strides(int *shape, int ndims);

extern "C" {

cudnnHandle_t CudnnHandle();

void *CudaMalloc(size_t size);

void **CudaArray(const void **ptr, size_t num);

void CudaFree(void *ptr);

void CudaMemcpyHostToDevice(void *dev, const void *host, size_t size);

void CudaMemcpyDeviceToHost(void *host, const void *dev, size_t size);

struct CudaBuffer {
  static CudaBuffer *Buffer(size_t size = 0) {
    static CudaBuffer *buffer = new CudaBuffer;
    return buffer;
  }

  CudaBuffer(size_t size = 0) {
    Resize(size);
  }

  void *data() { return data_; }

  void Resize(size_t size) {
    if (size > capacity_) {
      // CudaFree(data_);
      data_ = CudaMalloc(size);
      capacity_ = size;
    }
  }

  ~CudaBuffer() {
    CudaFree(data_);
  }

  void *data_{nullptr};
  size_t capacity_{0};
};

}  // extern "C"
