// Copyright (c) 2020 smarsu. All Rights Reserved.

#pragma once
#include <vector>
#include <string>

#include "cnrt.h"

#include "glog/logging.h"

#define CALL_CNRT(x)                                         \
  CHECK_EQ(static_cast<cnrtRet_t>(x), CNRT_RET_SUCCESS) \
    << cnrtGetErrorStr(x)

#define To3DIndex(x, y, z, X, Y, Z) \
  (((x) * (Y) + (y)) * (Z) + (z))

#define To2DIndex(x, y, X, Y) \
  ((x) * (Y) + (y))

std::string ToString(int *shape, int ndims);

std::vector<int> Shape2Strides(int *shape, int ndims);

extern "C" {

void SetDevice(int id);

cnrtQueue_t MluStream();

void *MluMalloc(size_t size);

void **MluArray(const void **ptr, size_t num);

void MluFree(void *ptr);

void MluMemcpyHostToDevice(void *dev, const void *host, size_t size);

void MluMemcpyDeviceToHost(void *host, const void *dev, size_t size);

struct MluBuffer {
  static MluBuffer *Buffer(size_t size = 0) {
    static MluBuffer *buffer = new MluBuffer;
    return buffer;
  }

  MluBuffer(size_t size = 0) {
    Resize(size);
  }

  void *data() { return data_; }

  void Resize(size_t size) {
    if (size > capacity_) {
      // CudaFree(data_);
      data_ = MluMalloc(size);
      capacity_ = size;
    }
  }

  ~MluBuffer() {
    MluFree(data_);
  }

  void *data_{nullptr};
  size_t capacity_{0};
};

}  // extern "C"
