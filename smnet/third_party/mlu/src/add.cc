// Copyright (c) 2020 smarsu. All Rights Reserved.

#include "kernel.h"
#include "core.h"

#include "cnrt.h"

extern "C" {

void Add(float *A, float *B, float *C, size_t size) {
  KernelParamsBuffer params(C, A, B, static_cast<int>(size));
  CALL_CNRT(cnrtInvokeKernel_V3(reinterpret_cast<void *>(add_entry),
                                KernelInitParam::addInitParam(),
                                KernelParam::Dim3(),
                                params.params(),
                                KernelParam::FuncType(),
                                MluStream(),
                                nullptr));
}

}  // extern "C"
