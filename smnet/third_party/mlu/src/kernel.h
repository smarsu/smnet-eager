// Copyright (c) 2020 smarsu. All Rights Reserved.

#pragma once
#include "cnrt.h"
#include "core.h"
#include "kernel/macro.h"

extern "C" {
  void add_entry();
  void addpad_entry();
  void conv2d_entry();
  void pool_entry();
}  // extern "C"

class KernelParam {
 public:
  static cnrtDim3_t Dim3() {
    cnrtDim3_t dim3;
    dim3.x = 16;
    dim3.y = 1;
    dim3.z = 1;
    return dim3;
  }

  static cnrtFunctionType_t FuncType() {
    return CNRT_FUNC_TYPE_UNION4;
  }
};

class KernelInitParam {
 public:
#define INITPARAM(func) \
  static cnrtKernelInitParam_t func##InitParam() { \
    static cnrtKernelInitParam_t init_param = nullptr; \
    if (!init_param) { \
      CALL_CNRT(cnrtCreateKernelInitParam(&init_param)); \
      CALL_CNRT(cnrtInitKernelMemory(reinterpret_cast<void *>(&func##_entry), init_param)); \
    } \
    return init_param; \
  }

  INITPARAM(add);
  INITPARAM(addpad);
  INITPARAM(conv2d);
  INITPARAM(pool);

#undef INITPARAM
};

class KernelParamsBuffer {
 public:
  template <typename... Args>
  KernelParamsBuffer(Args... rest) {
    CALL_CNRT(cnrtGetKernelParamsBuffer(&params_));
    AddParam(rest...);
  }

  ~KernelParamsBuffer() {
    if (params_) {
      CALL_CNRT(cnrtDestroyKernelParamsBuffer(params_));
    }
  }

 public:
  cnrtKernelParamsBuffer_t params() { return params_; }

 private:
  void AddParam() {}

  template <typename T>
  void AddParam(T param) {
    CALL_CNRT(cnrtKernelParamsBufferAddParam(params_, &param, sizeof(T)));
  }

  template <typename T, typename... Args>
  void AddParam(T param, Args... rest) {
    AddParam(param);
    AddParam(rest...);
  }

 private:
  cnrtKernelParamsBuffer_t params_{nullptr};
};
