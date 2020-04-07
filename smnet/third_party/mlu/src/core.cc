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

void SetDevice(int id) {
  CALL_CNRT(cnrtInit(0));
  cnrtDev_t dev;
  CALL_CNRT(cnrtGetDeviceHandle(&dev, id));
  CALL_CNRT(cnrtSetCurrentDevice(dev));
}

cnrtQueue_t MluStream() {
  static cnrtQueue_t queue = nullptr;
  if (!queue) {
    CALL_CNRT(cnrtCreateQueue(&queue)) << " Create mlu stream failed."; 
  }
  return queue;
}

void *MluMalloc(size_t size) {
  void *ptr = nullptr;
  CALL_CNRT(cnrtMalloc(&ptr, size)) << " mlu malloc " << size << " failed.";
  LOG(INFO) << "Mlu Malloc " << ptr << " ... " << size;
  return ptr;
}

void **MluArray(const void **ptr, size_t num) {
  size_t size = sizeof(void *) * num;

  void *ret = MluMalloc(size);
  MluMemcpyHostToDevice(ret, reinterpret_cast<const void *>(ptr), size);
  return reinterpret_cast<void **>(ret);
}

void MluFree(void *ptr) {
  if (ptr) {
    CALL_CNRT(cnrtFree(ptr)) << " mlu free " << ptr << " failed.";
  }
  LOG(INFO) << "Mlu Free " << ptr;
}

void MluMemcpyHostToDevice(void *dev, const void *host, size_t size) {
  CALL_CNRT(cnrtMemcpyAsync(dev, const_cast<void *>(host), size, MluStream(), CNRT_MEM_TRANS_DIR_HOST2DEV));
  LOG(INFO) << "Mlu Memcpy host " << host << " to device " << dev << " " << size << " bytes.";
}

void MluMemcpyDeviceToHost(void *host, const void *dev, size_t size) {
  CALL_CNRT(cnrtMemcpyAsync(host, const_cast<void *>(dev), size, MluStream(), CNRT_MEM_TRANS_DIR_DEV2HOST));
  CALL_CNRT(cnrtSyncQueue(MluStream()));
  LOG(WARNING) << "Mlu Memcpy device " << dev << " to host " << host << " " << size << " bytes.";
}
