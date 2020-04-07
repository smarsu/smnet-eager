// Copyright (c) 2020 smarsu. All Rights Reserved.

#pragma once
#include "kernel.h"

/* V1: solve the problem of bankwidth. */
/* R: Release */
/* 0402: Add Support for NHWC */

#define NRAM_REMAIN_SIZE (1024 + 128)
#define NRAM_BUF_SIZE (512 * 1024 - NRAM_REMAIN_SIZE)
#define SRAM_BUF_SIZE (2048 * 1024)

template <typename T>
__mlu_func__ void PadUpDn(T *dst_ddr,
                          int h,
                          int w,
                          int c,
                          int pad_t,
                          int pad_b,
                          int pad_l,
                          int pad_r,
                          PadMode mode,
                          Layout layout,
                          T pad_value,
                          int block_size,
                          T *nram_buf) {
  int dst_h = h + pad_t + pad_b;
  int dst_w = w + pad_l + pad_r;

  int start, end;

  if (layout == kNCHW) {
    // up
    SegTask(pad_t * dst_w, int(128 / sizeof(T)), &start, &end);
    for (int i = start; i < end; i += block_size) {
      int s = MIN(block_size, end - i);
      __mlvm_memcpy_stride_nram_to_gdram(dst_ddr + i,
                                               nram_buf,
                                               sizeof(T) * s,
                                               sizeof(T) * dst_h * dst_w,
                                               0,
                                               c - 1);
    }

    // dn
    SegTask(pad_b * dst_w, int(128 / sizeof(T)), &start, &end);
    for (int i = start; i < end; i += block_size) {
      int s = MIN(block_size, end - i);
      __mlvm_memcpy_stride_nram_to_gdram(dst_ddr + i + (pad_t + h) * dst_w,
                                               nram_buf,
                                               sizeof(T) * s,
                                               sizeof(T) * dst_h * dst_w,
                                               0,
                                               c - 1);
    }
  }
  else if (layout == kNHWC) {
    SegTask(pad_t * dst_w * c, int(128 / sizeof(T)), &start, &end);
    for (int i = start; i < end; i += block_size) {
      int s = MIN(block_size, end - i);
      __mlvm_memcpy_nram_to_gdram(dst_ddr + i, nram_buf, sizeof(T) * s);
    }

    SegTask(pad_b * dst_w * c, int(128 / sizeof(T)), &start, &end);
    for (int i = start; i < end; i += block_size) {
      int s = MIN(block_size, end - i);
      __mlvm_memcpy_nram_to_gdram(dst_ddr + i + (pad_t + h) * dst_w * c, nram_buf, sizeof(T) * s);
    }
  }
}

template <typename T>
__mlu_func__ void addpad(T *src_ddr,
                         T *dst_ddr,
                         int h,
                         int w,
                         int c,
                         int pad_t,
                         int pad_b,
                         int pad_l,
                         int pad_r,
                         PadMode mode,
                         Layout layout,
                         T pad_value,
                         T *nram_buf,
                         T *sram_buf) {
  int dst_h = h + pad_t + pad_b;
  int dst_w = w + pad_l + pad_r;

  int block_h = (NRAM_BUF_SIZE / sizeof(T) / dst_w / c) - 1;
  int block_size = PAD_UP(block_h * dst_w * c, 128 / sizeof(T));

  __nramset(nram_buf, block_size, pad_value);

  PadUpDn(dst_ddr,
          h,
          w,
          c,
          pad_t,
          pad_b,
          pad_l,
          pad_r,
          mode,
          layout,
          pad_value,
          block_size,
          nram_buf);

  if (layout == kNCHW) {
    int start, end;
    SegTask(h, 1, &start, &end);

    for (int i = 0; i < c; ++i) {
      T *src_slice = src_ddr + i * h * w;
      T *dst_slice = dst_ddr + i * dst_h * dst_w;

      for (int j = start; j < end; j += block_h) {
        int size = MIN(block_h, end - j);
        // GDRAM2SRAM & SRAM2GDRAM make U1 to slow.
        // GDRAM2NRAM with stride is similar to GDRAM2NRAM without stride.
        __memcpy(nram_buf + pad_l, src_slice + j * w, sizeof(T) * w, GDRAM2NRAM, sizeof(T) * dst_w, sizeof(T) * w, size - 1);

        if (coreId != 0x80) __bang_lock(0, 0);
        __memcpy(dst_slice + (pad_t + j) * dst_w, nram_buf, sizeof(T) * size * dst_w, NRAM2GDRAM);
        if (coreId != 0x80) __bang_unlock(0, 0);
      }
    } 
  }
  else if (layout == kNHWC) {
    int start, end;
    SegTask(h, 1, &start, &end);

    for (int j = start; j < end; j += block_h) {
      int size = MIN(block_h, end - j);

      __memcpy(nram_buf + pad_l * c, src_ddr + j * w * c, sizeof(T) * w * c, GDRAM2NRAM, sizeof(T) * dst_w * c, sizeof(T) * w * c, size - 1);

      if (coreId != 0x80) __bang_lock(0, 0);
      __memcpy(dst_ddr + (pad_t + j) * dst_w * c, nram_buf, sizeof(T) * size * dst_w * c, NRAM2GDRAM);
      if (coreId != 0x80) __bang_unlock(0, 0);
    }
  }
}

template <typename T>
__mlu_func__ void addpad_wrap(T *src_ddr,
                              T *dst_ddr,
                              int n,
                              int h,
                              int w,
                              int c,
                              int pad_t,
                              int pad_b,
                              int pad_l,
                              int pad_r,
                              PadMode mode,
                              Layout layout,
                              T pad_value,
                              T *nram_buf,
                              T *sram_buf) {
  int dst_h = h + pad_t + pad_b;
  int dst_w = w + pad_l + pad_r;

  for (int i = 0; i < n; ++i) {
    T *src_slice = src_ddr + i * h * w * c;
    T *dst_slice = dst_ddr + i * dst_h * dst_w * c;

    addpad(src_slice,
           dst_slice,
           h,
           w,
           c,
           pad_t,
           pad_b,
           pad_l,
           pad_r,
           mode,
           layout,
           pad_value,
           nram_buf,
           sram_buf);

    __sync_all();
  }
}

// __mlu_entry__ void addpad_entry(void *src,
//                                 void *dst,
//                                 int n,
//                                 int h,
//                                 int w,
//                                 int c,
//                                 int pad_t,
//                                 int pad_b,
//                                 int pad_l,
//                                 int pad_r,
//                                 PadMode mode,
//                                 Layout layout,
//                                 DataType dtype,
//                                 float pad_value) {
//   __nram__ int8_t nram_buf[NRAM_BUF_SIZE];
//   __mlu_shared__ int8_t sram_buf[SRAM_BUF_SIZE];

// #define ADDPAD(T) \
//     addpad_wrap((T *)src, \
//                 (T *)dst, \
//                 n, \
//                 h, \
//                 w, \
//                 c, \
//                 pad_t, \
//                 pad_b, \
//                 pad_l, \
//                 pad_r, \
//                 mode, \
//                 layout, \
//                 (T)pad_value, \
//                 (T *)nram_buf, \
//                 (T *)sram_buf);

//   if (dtype == DataType::kUint8) {
//     ADDPAD(uint8_t);
//   }
//   if (dtype == DataType::kFloat32) {
//     ADDPAD(float);
//   }

// #undef ADDPAD
// }
