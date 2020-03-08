// Copyright (c) 2020 smarsu. All Rights Reserved.

#pragma once
#include <string>

extern "C" {

void *CpuMalloc(size_t size);

void CpuFree(void *ptr);

}
