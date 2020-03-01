// Copyright (c) 2020 smarsu. All Rights Reserved.

#include <vector>
#include <string>
#include <sstream>

#include "core.h"

std::string ToString(int *shape, int ndims) {
  if (ndims == 0) return "";

  std::stringstream ss;
  ss << "[";
  for (int i = 0; i < ndims - 1; ++i) {
    ss << shape[i] << ", ";
  }
  ss << shape[ndims - 1] << "]";
  return ss.str();
}

std::vector<int> Shape2Strides(int *shape, int ndims) {
  std::vector<int> strides(ndims);
  int prod = 1;
  for (int i = ndims - 1; i >= 0; --i) {
    strides[i] = prod;
    prod *= shape[i];
  }

  return strides;
}

cudnnHandle_t CudnnHandle() {
  static cudnnHandle_t handle = NULL;
  if (!handle) {
    CALL_CUDNN(cudnnCreate(&handle)) << " Create cudnn handle failed."; 
  }
  return handle;
}

void *CudaMalloc(size_t size) {
  void *ptr = NULL;
  CALL_CUDA(cudaMalloc(&ptr, size)) << " cuda malloc " << size << " failed.";
  LOG(INFO) << "Malloc " << ptr << " ... " << size;
  return ptr;
}

void CudaFree(void *ptr) {
  if (ptr) {
    CALL_CUDA(cudaFree(ptr)) << " cuda free " << ptr << " failed.";
  }
  LOG(INFO) << "Free " << ptr;
}

void CudaMemcpyHostToDevice(void *dev, const void *host, size_t size) {
  CALL_CUDA(cudaMemcpy(dev, host, size, cudaMemcpyHostToDevice));
}

void CudaMemcpyDeviceToHost(void *host, const void *dev, size_t size) {
  CALL_CUDA(cudaDeviceSynchronize());
  CALL_CUDA(cudaMemcpy(host, dev, size, cudaMemcpyDeviceToHost));
}
