// Copyright (c) 2020 smarsu. All Rights Reserved.

#include "kernel.h"
#include "core.h"

#include "cnrt.h"

extern "C" {

void Split(float *src,
           float **dst,
           int n,
           int t,
           int c,
           int *size_splits,
           int num_output) {
  KernelParamsBuffer params(src, dst, n, t, c, size_splits, num_output);
  CALL_CNRT(cnrtInvokeKernel_V3(reinterpret_cast<void *>(&split_entry),
                                KernelInitParam::splitInitParam(),
                                KernelParam::Dim3(),
                                params.params(),
                                KernelParam::FuncType(),
                                MluStream(),
                                nullptr));
}

}  // extern "C"
