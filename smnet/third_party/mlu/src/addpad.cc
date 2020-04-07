// Copyright (c) 2020 smarsu. All Rights Reserved.

#include "kernel.h"
#include "core.h"

#include "cnrt.h"

extern "C" {

void AddPad(float *A, float *C, size_t n, size_t h, size_t w, size_t c, int pad_t, int pad_b, int pad_l, int pad_r, float pad_value) {
  KernelParamsBuffer params(A, 
                            C, 
                            static_cast<int>(n),
                            static_cast<int>(h),
                            static_cast<int>(w),
                            static_cast<int>(c),
                            pad_t,
                            pad_b,
                            pad_l,
                            pad_r,
                            kConstant,
                            kNHWC,
                            pad_value);
  CALL_CNRT(cnrtInvokeKernel_V3(reinterpret_cast<void *>(addpad_entry),
                                KernelInitParam::addpadInitParam(),
                                KernelParam::Dim3(),
                                params.params(),
                                KernelParam::FuncType(),
                                MluStream(),
                                nullptr));
}

}
