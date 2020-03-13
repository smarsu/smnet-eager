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
  static cudnnHandle_t handle = nullptr;
  if (!handle) {
    CALL_CUDNN(cudnnCreate(&handle)) << " Create cudnn handle failed."; 
  }
  return handle;
}

void *CudaMalloc(size_t size) {
  void *ptr = nullptr;
  CALL_CUDA(cudaMalloc(&ptr, size)) << " cuda malloc " << size << " failed.";
  LOG(INFO) << "Malloc " << ptr << " ... " << size;
  return ptr;
}

void **CudaArray(const void **ptr, size_t num) {
  size_t size = sizeof(void *) * num;

  void *ret = CudaMalloc(size);
  CudaMemcpyHostToDevice(ret, reinterpret_cast<const void *>(ptr), size);
  return reinterpret_cast<void **>(ret);
}

void CudaFree(void *ptr) {
  if (ptr) {
    CALL_CUDA(cudaFree(ptr)) << " cuda free " << ptr << " failed.";
  }
  LOG(INFO) << "Free " << ptr;
}

void CudaMemcpyHostToDevice(void *dev, const void *host, size_t size) {
  CALL_CUDA(cudaMemcpyAsync(dev, host, size, cudaMemcpyHostToDevice));
  LOG(INFO) << "Memcpy host " << host << " to device " << dev << " " << size << " bytes.";
}

void CudaMemcpyDeviceToHost(void *host, const void *dev, size_t size) {
  CALL_CUDA(cudaMemcpyAsync(host, dev, size, cudaMemcpyDeviceToHost));
  CALL_CUDA(cudaDeviceSynchronize());
  LOG(WARNING) << "Memcpy device " << dev << " to host " << host << " " << size << " bytes.";
}
