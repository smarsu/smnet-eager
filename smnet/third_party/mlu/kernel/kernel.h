// Copyright (c) 2020 smarsu. All Rights Reseved.

#pragma once
#include "macro.h"

#define PAD_UP(x, y) (((x) + (y) - 1) / (y) * (y))
#define PAD_DN(x, y) ((x) / (y) * (y))
#define DIV_UP(x, y) (((x) + (y) - 1) / (y))

#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define MAX(x, y) ((x) > (y) ? (x) : (y))

#define To4DIndex(a, b, c, d, A, B, C, D) \
  ((((a) * (B) + (b)) * (C) + (c)) * (D) + (d))

template <typename T>
__mlu_func__ T *Rowoff(T *src, uint32_t m, uint32_t n) {
  T *ptr = NULL;
  uint32_t SHIFT = 16;

  uint32_t n_ds = n >> SHIFT;
  uint32_t n_rs = n % (1 << SHIFT);
  uint32_t tmp_ds = m * n_ds * sizeof(T);
  uint32_t tmp_rs = m * n_rs * sizeof(T);

  __asm__ volatile("sll.gpr.u48 %[ptr], %[tmp_ds], %[SHIFT];\n\t"
                   "add.gpr.ptr %[ptr], %[ptr], %[src];\n\t"
                   "add.gpr.ptr %[ptr], %[ptr], %[tmp_rs];\n\t"
                   :[ptr]"+&r"(ptr)
                   :[src]"r"(src), [tmp_ds]"r"(tmp_ds), [tmp_rs]"r"(tmp_rs), [SHIFT]"i"(SHIFT)
                   );

  return ptr;
}

__mlu_func__ void SegTask(int n, int up, int *start, int *end) {
  int div = DIV_UP(n, taskDim * up);
  div *= up;

  *start = taskId * div;
  *end = MIN(n, (taskId + 1) * div);
}

__mlu_func__ void F2I(int16_t *dst, float *src, int size, int pos) {
  __bang_float2int16_rd(dst, src, size, pos);
}

template <typename T, typename IT>
__mlu_func__ void quant(IT *dst, T *src, int size, float *scale, int *pos, T *nram_buf) {
  __bang_max(nram_buf, src, size);
  T max_v = fabsf(nram_buf[0]);
  __bang_min(nram_buf, src, size);
  max_v = MAX(max_v, fabsf(nram_buf[0]));

  constexpr int scale_m = (2 << (sizeof(IT) * 8 - 1)) - 1;

  *scale = scale_m / max_v;
  *pos = (int)(log2f(*scale));
  *scale = *scale / (2 << *pos);
  *pos = -*pos;

  __bang_mul_const(src, src, *scale, size);
  F2I(dst, src, size, *pos);
}

template <typename T>
__mlu_func__ void __mlvm_memcpy_gdram_to_gdram(T *dst, 
                                               T *src, 
                                               int size, 
                                               int dst_stride, 
                                               int src_stride, 
                                               int count, 
                                               int nram_buf_size,
                                               T *nram_buf) {
  ++count;
  if (size * count > nram_buf_size) {
    for (int i = 0; i < count; ++i) {
      __mlvm_memcpy_gdram_to_gdram(dst + i * dst_stride, 
                                   src + i * src_stride, 
                                   size);
    }
  }
  else {
    __memcpy(nram_buf, src, size, GDRAM2NRAM, size, src_stride, count - 1);
    __memcpy(dst, nram_buf, size, NRAM2GDRAM, dst_stride, size, count - 1);
  }
}
