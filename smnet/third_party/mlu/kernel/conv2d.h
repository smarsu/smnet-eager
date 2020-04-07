// Copyright (c) 2020 smarsu. All Rights Reserved.

#include "kernel.h"

#define NRAM_BUF_REMAIN (1024 + 128 + 128 + 128)
#define NRAM_BUF_SIZE (512 * 1024 - NRAM_BUF_REMAIN)
#define SRAM_BUF_SIZE (2048 * 1024)

#if __BANG_ARCH__ == 270
#define WRAM_BUF_SIZE (1024 * 1024)
#elif __BANG_ARCH__ == 220
#define WRAM_BUF_SIZE (512 * 1024)
#endif 

template <typename T, typename IT>
__mlu_func__ void  conv2d(T *x,
                          T *w,
                          T *bias,
                          T *y,
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
                          bool with_relu,
                          int nram_buf_size,
                          int sram_buf_size,
                          int wram_buf_size,
                          T *nram_buf,
                          T *sram_buf,
                          IT *wram_buf) {
  T *dst_inchip = nram_buf;  // 64

  T *bias_inchip = dst_inchip + 64;  // 64

  T *weight_inchip = dst_inchip + 64;  // hf * wf * 64 * 64

  T *src_inchip = dst_inchip + 64;  // hf * wf * 64
  T *dst_once = src_inchip + hf * wf * 64;  // 64

  int start, end, s;
  SegTask(ho, 1, &start, &end);
  s = end - start;

  int sx = 0;

  for (int _ni = 0; _ni < ni; ++_ni) {
    for (int _ho = start; _ho < end; ++_ho) {
      for (int _wo = 0; _wo < wo; ++_wo) {
        for (int _co = 0; _co < co; _co += 64) {
          int _cof = MIN(64, co - _co);
          __bang_write_zero(dst_inchip, 64);

          for (int _ci = 0; _ci < ci; _ci += 64) {
            int _hi = _ho * hs;
            int _wi = _wo * ws;
            int _cf = MIN(64, ci - _ci);

            if (_cf < 64 || _cof < 64) {
              __bang_write_zero(weight_inchip, 64 * hf * wf * 64);
            }

            // ATTENTION(smarsu): sizeof(T) >= sizeof(IT)
            for (int i = 0; i < _cof * hf; ++i) {
              __memcpy(weight_inchip + i * wf * 64, 
                       w + To4DIndex(_co, i, 0, _ci, 0, hf, wf, ci),
                       sizeof(T) * _cf,
                       GDRAM2NRAM,
                       sizeof(T) * 64,
                       sizeof(T) * ci,
                       wf - 1);
            }

            float w_scale;
            int w_pos;
            quant((IT *)weight_inchip, weight_inchip, 64 * hf * wf * 64, &w_scale, &w_pos, weight_inchip + 64 * hf * wf * 64);

            __memcpy(wram_buf, weight_inchip, sizeof(IT) * 64 * hf * wf * 64, NRAM2WRAM);

            if (_cf < 64) {
              __bang_write_zero(src_inchip, hf * wf * 64);
            }

            // TODO(smarsu): Add stride for dilation.
            for (int i = 0; i < hf; ++i) {
              __memcpy(src_inchip + i * wf * 64,
                       x + To4DIndex(_ni, _hi + i, _wi, _ci, 0, hi, wi, ci),
                       sizeof(T) * _cf,
                       GDRAM2NRAM,
                       sizeof(T) * 64,
                       sizeof(T) * ci,
                       wf - 1);
            }

            float src_scale;
            int src_pos;
            quant((IT *)src_inchip, src_inchip, hf * wf * 64, &src_scale, &src_pos, src_inchip + hf * wf * 64);

            __bang_conv(dst_once, 
                        (IT *)src_inchip, 
                        (IT *)wram_buf,
                        64, 
                        hf, 
                        wf, 
                        hf, 
                        wf, 
                        hs, 
                        ws, 
                        64, 
                        src_pos + w_pos);
            __bang_mul_const(dst_once, dst_once, 1 / (src_scale * w_scale), 64);
            __bang_add(dst_inchip, dst_inchip, dst_once, 64);
          }

          if (with_bias) {
            __memcpy(bias_inchip, bias + _co, sizeof(T) * _cof, GDRAM2NRAM);
            __bang_add(dst_inchip, dst_inchip, bias_inchip, 64);
          }

          if (with_relu) {
            __bang_active_relu(dst_inchip, dst_inchip, 64);
          }
          __memcpy(y + To4DIndex(_ni, _ho, _wo, _co, 0, ho, wo, co),
                   dst_inchip,
                   sizeof(T) * _cof,
                   NRAM2GDRAM);
        }
      }
    }
  }
}
