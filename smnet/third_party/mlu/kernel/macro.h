// Copyright (c) 2020 smarsu. All Rights Reseved.

enum class DataType : int {
  kInvalid,
  kFloat32,
  kFloat16,
  kUint8,
  kInt8,
  kInt16,
  kInt32,
  kUint32,
};

enum class PoolMode : int {
  kInvalid,
  kMax,
  kAvg,
};

enum Layout {
  kNCHW,
  kNHWC,
};

enum PadMode {
  kConstant,
  kEdge,
};
