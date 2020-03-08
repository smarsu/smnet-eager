// Copyright (c) 2020 smarsu. All Rights Reserved.

#include <string>

#include "core.h"

extern "C" {

void *CpuMalloc(size_t size) {
  void *ptr = nullptr;
  ptr = malloc(size);
  return ptr;
}

void CpuFree(void *ptr) {
  if (ptr) {
    free(ptr);
  }
}

}
