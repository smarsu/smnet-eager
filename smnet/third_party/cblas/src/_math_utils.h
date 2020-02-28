// Copyright (c) 2020 smarsu. All Rights Reserved.

#pragma once

#define INFINITE 0x7F800000

#define PtoI2D(a, b, A, B) \
  ((a) * (B) + (b))

#define PtoI3D(a, b, c, A, B, C) \
  ((a) * (B) * (C) + (b) * (C) + (c))

#define PtoI4D(a, b, c, d, A, B, C, D) \
  ((a) * (B) * (C) * (D) + (b) * (C) * (D) + (c) * (D) + (d))
