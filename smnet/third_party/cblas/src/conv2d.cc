// Copyright (c) 2020 smarsu. All Rights Reserved.

#include <cstdlib>
#include <cstring>

#include "_math_utils.h"

static void *work_space = nullptr;
static int work_size = 0;

template <typename T>
void conv2d(T *output,
            T *input,
            T *filter,
            int *strides,
            int *paddings,
            int *dilations,
            int *output_shape,
            int *input_shape,
            int *filter_shape) {
  int n = output_shape[0];
  int co = output_shape[1];
  int ho = output_shape[2];
  int wo = output_shape[3];
  
  int ci = input_shape[1];
  int hi = input_shape[2];
  int wi = input_shape[3];

  int hf = filter_shape[2];
  int wf = filter_shape[3];

  int hs = strides[0];
  int ws = strides[1];

  int hd = dilations[0];
  int wd = dilations[1];

  for (int _n = 0; _n < n; ++_n) {
    for (int _co = 0; _co < co; ++_co) {
      for (int _ho = 0; _ho < ho; ++_ho) {
        for (int _wo = 0; _wo < wo; ++_wo) {
          T sum = 0;
          for (int _ci = 0; _ci < ci; ++_ci) {
            for (int _hf = 0; _hf < hf; ++_hf) {
              for (int _wf = 0; _wf < wf; ++_wf) {
                sum += input[PtoI4D(_n, _ci, _ho * hs + _hf * hd, _wo * ws + _wf * wd, n, ci, hi, wi)] \
                  * filter[PtoI4D(_co, _ci, _hf, _wf, co, ci, hf, wf)];
              }
            }
          }
          output[PtoI4D(_n, _co, _ho, _wo, n, co, ho, wo)] = sum;
        }
      }
    }
  }
}

template <typename T>
void conv2d_backward_data(T *input_grad,
                          T *output_grad,
                          T *filter,
                          int *strides,
                          int *paddings,
                          int *dilations,
                          int *output_shape,
                          int *input_shape,
                          int *filter_shape,
                          T alpha) {
  int n = output_shape[0];
  int co = output_shape[1];
  int ho = output_shape[2];
  int wo = output_shape[3];
  
  int ci = input_shape[1];
  int hi = input_shape[2];
  int wi = input_shape[3];

  int hf = filter_shape[2];
  int wf = filter_shape[3];

  int hs = strides[0];
  int ws = strides[1];

  int hd = dilations[0];
  int wd = dilations[1];

  int hoxwo = ho * wo;
  int cixhfxwf = ci * hf * wf;

  int size = sizeof(T) * n * ho * wo * ci * hf * wf;
  if (size > work_size) {
    work_size = size;
    free(work_space);
    work_space = malloc(size);
  }

  T *work_space_T = reinterpret_cast<T *>(work_space);

  for (int _n = 0; _n < n; ++_n) {
    for (int _hoxwo = 0; _hoxwo < hoxwo; ++_hoxwo) {
      for (int _cixhfxwf = 0; _cixhfxwf < cixhfxwf; ++_cixhfxwf) {
        T sum = 0;
        for (int _co = 0; _co < co; ++_co) {
          sum += output_grad[PtoI3D(_n, _co, _hoxwo, n, co, hoxwo)] \
            * filter[PtoI2D(_co, _cixhfxwf, co, cixhfxwf)];
        }
        work_space_T[PtoI3D(_n, _hoxwo, _cixhfxwf, n, hoxwo, cixhfxwf)] = sum;
      }
    }
  }

  if (alpha == 0) {
    memset(input_grad, 0, sizeof(T) * n * ci * hi * wi);
  }

  for (int _n = 0; _n < n; ++_n) {
    for (int _ci = 0; _ci < ci; ++_ci) {
      for (int _ho = 0; _ho < ho; ++_ho) {
        for (int _wo = 0; _wo < wo; ++_wo) {
          for (int _hf = 0; _hf < hf; ++_hf) {
            for (int _wf = 0; _wf < wf; ++_wf) {
              int _hoxwo = _ho * wo + _wo;
              int _cixhfxwf = _ci * hf * wf + _hf * wf + _wf;
              input_grad[PtoI4D(_n, _ci, _ho * hs + _hf * hd, _wo * ws + _wf * wd, n, ci, hi, wi)] += \
                work_space_T[PtoI3D(_n, _hoxwo, _cixhfxwf, n, hoxwo, cixhfxwf)];
            }
          }
        }
      }
    }
  }
}

template <typename T>
void conv2d_backward_filter(T *filter_grad,
                            T *input,
                            T *output_grad,
                            int *strides,
                            int *paddings,
                            int *dilations,
                            int *output_shape,
                            int *input_shape,
                            int *filter_shape,
                            T alpha) {
  int n = output_shape[0];
  int co = output_shape[1];
  int ho = output_shape[2];
  int wo = output_shape[3];
  
  int ci = input_shape[1];
  int hi = input_shape[2];
  int wi = input_shape[3];

  int hf = filter_shape[2];
  int wf = filter_shape[3];

  int hs = strides[0];
  int ws = strides[1];

  int hd = dilations[0];
  int wd = dilations[1];

  if (alpha == 0) {
    memset(filter_grad, 0, sizeof(T) * co * ci * hf * wf);
  }

  if (alpha == 0) {
    for (int _co = 0; _co < co; ++_co) {
      for (int _ci = 0; _ci < ci; ++_ci) {
        for (int _hf = 0; _hf < hf; ++_hf) {
          for (int _wf = 0; _wf < wf; ++_wf) {
            T sum = 0;
            for (int _n = 0; _n < n; ++_n) {
              for (int _ho = 0; _ho < ho; ++_ho) {
                for (int _wo = 0; _wo < wo; ++_wo) {
                  sum += input[PtoI4D(_n, _ci, _ho * hs + _hf * hd, _wo * ws + _wf * wd, n, ci, hi, wi)] \
                    * output_grad[PtoI4D(_n, _co, _ho, _wo, n, co, ho, wo)];
                }
              }
            }

            filter_grad[PtoI4D(_co, _ci, _hf, _wf, co, ci, hf, wf)] = sum;
          }
        }
      }
    }
  }
  else {
    for (int _co = 0; _co < co; ++_co) {
      for (int _ci = 0; _ci < ci; ++_ci) {
        for (int _hf = 0; _hf < hf; ++_hf) {
          for (int _wf = 0; _wf < wf; ++_wf) {
            T sum = 0;
            for (int _n = 0; _n < n; ++_n) {
              for (int _ho = 0; _ho < ho; ++_ho) {
                for (int _wo = 0; _wo < wo; ++_wo) {
                  sum += input[PtoI4D(_n, _ci, _ho * hs + _hf * hd, _wo * ws + _wf * wd, n, ci, hi, wi)] \
                    * output_grad[PtoI4D(_n, _co, _ho, _wo, n, co, ho, wo)];
                }
              }
            }

            int index = PtoI4D(_co, _ci, _hf, _wf, co, ci, hf, wf);
            filter_grad[index] = alpha * filter_grad[index] + sum;
          }
        }
      }
    }
  }
}

extern "C" {

void conv2d_FP32(float *output,
                 float *input,
                 float *filter,
                 int *strides,
                 int *paddings,
                 int *dilations,
                 int *output_shape,
                 int *input_shape,
                 int *filter_shape) {
  conv2d<float>(output,
                input,
                filter,
                strides,
                paddings,
                dilations,
                output_shape,
                input_shape,
                filter_shape);
}

void conv2d_backward_data_FP32(float *input_grad,
                               float *output_grad,
                               float *filter,
                               int *strides,
                               int *paddings,
                               int *dilations,
                               int *output_shape,
                               int *input_shape,
                               int *filter_shape,
                               float alpha) {
  conv2d_backward_data<float>(input_grad,
                              output_grad,
                              filter,
                              strides,
                              paddings,
                              dilations,
                              output_shape,
                              input_shape,
                              filter_shape,
                              alpha);
}

void conv2d_backward_filter_FP32(float *filter_grad,
                                 float *input,
                                 float *output_grad,
                                 int *strides,
                                 int *paddings,
                                 int *dilations,
                                 int *output_shape,
                                 int *input_shape,
                                 int *filter_shape,
                                 float alpha) {
  conv2d_backward_filter<float>(filter_grad,
                                input,
                                output_grad,
                                strides,
                                paddings,
                                dilations,
                                output_shape,
                                input_shape,
                                filter_shape,
                                alpha);
}

}  // extern "C"
