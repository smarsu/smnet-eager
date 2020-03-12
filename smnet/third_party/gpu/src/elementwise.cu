// Copyright (c) 2020 smarsu. All Rights Reserved.

#include "core.h"

namespace kernel {

#define ASSIGN(x) (x)
#define NEG(x) (-(x))
#define RECIP(x) (1 / (x))

#define REGISTER_UNARY_OP(name, func) \
template <typename T> \
__global__ void UnaryElementwise##name##Op(int nthreads, \
                                           const T *x, \
                                           T beta, \
                                           T *y) { \
  int idx = blockIdx.x * blockDim.x + threadIdx.x; \
 \
  if (idx < nthreads) { \
    if (beta == 0) { \
      y[idx] = func(x[idx]); \
    } \
    else { \
      y[idx] = func(x[idx]) + beta * y[idx]; \
    } \
  } \
}

REGISTER_UNARY_OP(Assign, ASSIGN);
REGISTER_UNARY_OP(Neg, NEG);
REGISTER_UNARY_OP(Recip, RECIP);

#define ADD(x, y) (x) + (y)
#define SUB(x, y) (x) - (y)
#define MUL(x, y) (x) * (y)
#define DIV(x, y) (x) / (y)

#define REGISTER_BINARY_OP(name, func) \
template <typename T> \
__global__ void BinaryElementwise##name##Op(int nthreads, \
                                            const T *x, \
                                            const T *y, \
                                            T beta, \
                                            T *z) { \
  int idx = blockIdx.x * blockDim.x + threadIdx.x; \
 \
  if (idx < nthreads) { \
    if (beta == 0) { \
      z[idx] = func(x[idx], y[idx]); \
    } \
    else { \
      z[idx] = func(x[idx], y[idx]) + beta * z[idx]; \
    } \
  } \
} \
 \
template <typename T> \
__global__ void BinaryElementwiseCycleR##name##Op(int nthreads, \
                                                  const T *x, \
                                                  const T *y, \
                                                  T beta, \
                                                  T *z, \
                                                  int size_r) { \
  int idx = blockIdx.x * blockDim.x + threadIdx.x; \
 \
  if (idx < nthreads) { \
    if (beta == 0) { \
      z[idx] = func(x[idx], y[idx % size_r]); \
    } \
    else { \
      z[idx] = func(x[idx], y[idx % size_r]) + beta * z[idx]; \
    } \
  } \
} \
 \
template <typename T> \
__global__ void BinaryElementwiseCycleL##name##Op(int nthreads, \
                                                  const T *x, \
                                                  const T *y, \
                                                  T beta, \
                                                  T *z, \
                                                  int size_l) { \
  int idx = blockIdx.x * blockDim.x + threadIdx.x; \
 \
  if (idx < nthreads) { \
    if (beta == 0) { \
      z[idx] = func(x[idx % size_l], y[idx]); \
    } \
    else { \
      z[idx] = func(x[idx % size_l], y[idx]) + beta * z[idx]; \
    } \
  } \
}

REGISTER_BINARY_OP(Add, ADD);
REGISTER_BINARY_OP(Sub, SUB);
REGISTER_BINARY_OP(Mul, MUL);
REGISTER_BINARY_OP(Div, DIV);

template <typename T>
__global__ void DivRGradient(int nthreads,
                             const T *x,
                             const T *y,
                             const T *dz,
                             T beta,
                             T *dy) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < nthreads) {
    if (beta == 0) {
      dy[idx] = dz[idx] * -(x[idx] / (y[idx] * y[idx]));
    }
    else {
      dy[idx] = dz[idx] * -(x[idx] / (y[idx] * y[idx])) + beta * dy[idx];
    }
  }
}

}  // kernel

extern "C" {

#define REGISTER_C_UNARY_OP(name) \
bool name(int N, const float *x, float beta, float *y) { \
    kernel::UnaryElementwise##name##Op<float> \
      <<<CUDA_GET_BLOCKS(N), \
         CUDA_NUM_THREADS>>>(N, \
                             x, \
                             beta, \
                             y); \
  \
  CALL_CUDA(cudaPeekAtLastError()); \
  return true; \
}

REGISTER_C_UNARY_OP(Assign);
REGISTER_C_UNARY_OP(Neg);
REGISTER_C_UNARY_OP(Recip);

#define REGISTER_C_BINARY_OP(name) \
bool name(int N, const float *x, const float *y, float beta, float *z, int size_x, int size_y) { \
  if (size_x > size_y) { \
    kernel::BinaryElementwiseCycleR##name##Op<float> \
      <<<CUDA_GET_BLOCKS(N), \
         CUDA_NUM_THREADS>>>(N, \
                             x, \
                             y, \
                             beta, \
                             z, \
                             size_y); \
  } \
  else if (size_y > size_x) { \
    kernel::BinaryElementwiseCycleL##name##Op<float> \
      <<<CUDA_GET_BLOCKS(N), \
         CUDA_NUM_THREADS>>>(N, \
                             x, \
                             y, \
                             beta, \
                             z, \
                             size_x); \
  } \
  else { \
    kernel::BinaryElementwise##name##Op<float> \
      <<<CUDA_GET_BLOCKS(N), \
         CUDA_NUM_THREADS>>>(N, \
                             x, \
                             y, \
                             beta, \
                             z); \
  } \
  CALL_CUDA(cudaPeekAtLastError()); \
 \
  LOG(INFO) << "Run "#name" ... beta:" << beta << " N: " << N << " size x: " << size_x << " size y: " << size_y; \
 \
  return true; \
}

REGISTER_C_BINARY_OP(Add);
REGISTER_C_BINARY_OP(Sub);
REGISTER_C_BINARY_OP(Mul);
REGISTER_C_BINARY_OP(Div);

bool DivRGradient(int N,
                  const float *x,
                  const float *y,
                  const float *dz,
                  float beta,
                  float *dy) {
  kernel::DivRGradient<float>
  <<<CUDA_GET_BLOCKS(N),
     CUDA_NUM_THREADS>>>(N,
                         x,
                         y,
                         dz,
                         beta,
                         dy);

  CALL_CUDA(cudaPeekAtLastError());                        
  return true;
}

}  // extern "C"
