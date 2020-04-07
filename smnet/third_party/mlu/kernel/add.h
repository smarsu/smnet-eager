// Copyright (c) 2020 smarsu. All Rights Reseved.

#pragma once
#include "kernel.h"

template <typename T>
__mlu_func__ void add(T *C, T *A, T *B, int size, int nram_buf_size, T *nram_buf) {
  constexpr int LINE = 128 / sizeof(T);

  int start, end;
  SegTask(size, LINE, &start, &end);

  int block_size = PAD_DN(nram_buf_size / (sizeof(T) * 2), LINE); 

  T *A_inchip = nram_buf;  // block_size
  T *B_inchip = A_inchip + block_size;  // block_size

  for (int i = start; i < end; i += block_size) {
    int s = MIN(end - i, block_size);
    int s_up = PAD_UP(s, LINE);

    __memcpy(A_inchip, A + i, sizeof(T) * s, GDRAM2NRAM);
    __memcpy(B_inchip, B + i, sizeof(T) * s, GDRAM2NRAM);
    __bang_add(A_inchip, A_inchip, B_inchip, s_up);
    __memcpy(C + i, A_inchip, sizeof(T) * s, NRAM2GDRAM);
  }
}
