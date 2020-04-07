// Copyright (c) 2020 smarsu. All Rights Reserved.

#include "kernel.h"
#include "core.h"

#include "cnrt.h"

extern "C" {

#define REGISTER_POOL2D(MODE) \
void MODE##Pool2D(float *x, \
                  float *y, \
                  int ni, \
                  int hi, \
                  int wi, \
                  int ci, \
                  int no, \
                  int ho, \
                  int wo, \
                  int co, \
                  int hf, \
                  int wf, \
                  int hs, \
                  int ws) { \
  KernelParamsBuffer params(x, \
                            y, \
                            ni, \
                            hi, \
                            wi, \
                            ci, \
                            no, \
                            ho, \
                            wo, \
                            co, \
                            hf, \
                            wf, \
                            hs, \
                            ws, \
                            PoolMode::k##MODE, \
                            DataType::kFloat32); \
  CALL_CNRT(cnrtInvokeKernel_V3(reinterpret_cast<void *>(pool_entry), \
                                KernelInitParam::poolInitParam(), \
                                KernelParam::Dim3(), \
                                params.params(), \
                                KernelParam::FuncType(), \
                                MluStream(), \
                                nullptr)); \
}

REGISTER_POOL2D(Max);
REGISTER_POOL2D(Avg);

}  // extern "C"
