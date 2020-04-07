// Copyright (c) 2020 smarsu. All Rights Reserved.

#include "kernel.h"
#include "core.h"

#include "cnrt.h"

extern "C" {

void Conv2D(float *x,
            float *w,
            float *bias,
            float *y,
            int no,
            int co,
            int ho,
            int wo,
            int ni,
            int ci,
            int hi,
            int wi,
            int kf,
            int cf,
            int hf,
            int wf,
            int hs,
            int ws,
            int pt,
            int pb,
            int pl,
            int pr,
            bool with_bias,
            bool with_relu) {
  KernelParamsBuffer params(x,
                            w,
                            bias,
                            y,
                            no,
                            co,
                            ho,
                            wo,
                            ni,
                            ci,
                            hi,
                            wi,
                            kf,
                            cf,
                            hf,
                            wf,
                            hs,
                            ws,
                            pt,
                            pb,
                            pl,
                            pr,
                            with_bias,
                            with_relu,
                            DataType::kFloat32,
                            DataType::kInt16);
  CALL_CNRT(cnrtInvokeKernel_V3(reinterpret_cast<void *>(conv2d_entry),
                                KernelInitParam::conv2dInitParam(),
                                KernelParam::Dim3(),
                                params.params(),
                                KernelParam::FuncType(),
                                MluStream(),
                                nullptr));
}

}  // extern "C"
