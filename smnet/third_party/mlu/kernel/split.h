// Copyright (c) 2020 smarsu. All Rights Reserved.

#pragma once
#include "kernel.h"

template <typename T>
__mlu_func__ void split(T *src,
                        T **dst,
                        int n,
                        int t,
                        int c,
                        int *size_splits,
                        int num_output,
                        int nram_buf_size,
                        T *nram_buf) {
  int start, end, s;
  SegTask(n, 1, &start, &end);
  s = end - start;

  if (s > 0) {
    for (int i = 0; i < num_output; ++i) {
      int size = size_splits[i + 1] - size_splits[i];
      __mlvm_memcpy_gdram_to_gdram(dst[i] + start * size * c, 
                                   src + size_splits[i] * c + start * t * c,
                                   sizeof(T) * size * c,
                                   sizeof(T) * size * c,
                                   sizeof(T) * t * c,
                                   s - 1,
                                   nram_buf_size,
                                   nram_buf);
    }
  }
}
