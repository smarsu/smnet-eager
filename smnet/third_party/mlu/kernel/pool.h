// Copyright (c) 2020 smarsu. All Rights Reserved.

#pragma once
#include "kernel.h"

template <typename T>
__mlu_func__ void  pool(T *src,
                        T *dst,
                        int ni,
                        int hi,
                        int wi,
                        int ci,
                        int no,
                        int ho,
                        int wo,
                        int co,
                        int hf,
                        int wf,
                        int hs,
                        int ws,
                        int nram_buf_size,
                        PoolMode mode,
                        T *nram_buf) {
  int block_size = MIN(
    PAD_DN(nram_buf_size / (sizeof(T) * (hf * wf + 1)), 64),
    PAD_UP(ci, 64));

  T *src_inchip = nram_buf;  // hf * wf * block_size
  T *dst_inchip = src_inchip + hf * wf * block_size;  // block_size

  int start, end;
  SegTask(ho, 1, &start, &end);

  for (int _no = 0; _no < no; ++_no) {
    for (int _ho = start; _ho < end; ++_ho) {
      for (int _wo = 0; _wo < wo; ++_wo) {
        for (int _co = 0; _co < co; _co += block_size) {
          int _ni = _no;
          int _hi = _ho * hs;
          int _wi = _wo * ws;
          int _ci = _co;

          int _rco = MIN(block_size, co - _co);

          for (int i = 0; i < hf; ++i) {
            __memcpy(src_inchip + i * wf * block_size,
                     src + To4DIndex(_ni, _hi + i, _wi, _ci, 0, hi, wi, ci),
                     sizeof(T) * _rco,
                     GDRAM2NRAM,
                     sizeof(T) * block_size,
                     sizeof(T) * co,
                     wf - 1);
          }

          if (mode == PoolMode::kMax) {
            __bang_maxpool(dst_inchip, src_inchip, block_size, hf, wf, hf, wf, 1, 1);
          }
          else if (mode == PoolMode::kAvg) {
            __bang_avgpool(dst_inchip, src_inchip, block_size, hf, wf, hf, wf, 1, 1);
          }
          
          __memcpy(dst + To4DIndex(_no, _ho, _wo, _co, 0, ho, wo, co),
                   dst_inchip,
                   sizeof(T) * _rco,
                   NRAM2GDRAM);
        }
      }
    }
  }
}
