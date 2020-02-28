// Copyright (c) 2020 smarsu. All Rights Reserved.

#include <algorithm>
#include <cstring>

#include "_math_utils.h"

template <typename T>
void max_pool2d(T *output,
                T *value,
                int *ksize,
                int *strides,
                int *output_shape,
                int *value_shape) {
  int n = value_shape[0];
  int c = value_shape[1];
  int hi = value_shape[2];
  int wi = value_shape[3];

  int ho = output_shape[2];
  int wo = output_shape[3];

  int hf = ksize[0];
  int wf = ksize[1];

  int hs = strides[0];
  int ws = strides[1];

  for (int _n = 0; _n < n; ++_n) {
    for (int _c = 0; _c < c; ++_c) {
      for (int _ho = 0; _ho < ho; ++_ho) {
        for (int _wo = 0; _wo < wo; ++_wo) {
          T max_value = -INFINITE;
          for (int _hf = 0; _hf < hf; ++_hf) {
            for (int _wf = 0; _wf < wf; ++_wf) {
              max_value = std::max(max_value, 
                value[PtoI4D(_n, _c, _ho * hs + _hf, _wo * ws + _wf, n, c, hi, wi)]);
            }
          }
          output[PtoI4D(_n, _c, _ho, _wo, n, c, ho, wo)] = max_value;
        }
      }
    }
  }
}

template <typename T>
void max_pool2d_backward(T *value_grad,
                         T *output_grad,
                         T *value,
                         int *ksize,
                         int *strides,
                         int *output_shape,
                         int *value_shape,
                         T alpha) {
  int n = value_shape[0];
  int c = value_shape[1];
  int hi = value_shape[2];
  int wi = value_shape[3];

  int ho = output_shape[2];
  int wo = output_shape[3];

  int hf = ksize[0];
  int wf = ksize[1];

  int hs = strides[0];
  int ws = strides[1];

  if (alpha == 0) {
    memset(value_grad, 0, sizeof(T) * n * c * hi * wi);
  }
  else if (alpha != 1) {
    for (int i = 0; i < n * c * hi * wi; ++i) {
      value_grad[i] *= alpha;
    }
  }

  for (int _n = 0; _n < n; ++_n) {
    for (int _c = 0; _c < c; ++_c) {
      for (int _ho = 0; _ho < ho; ++_ho) {
        for (int _wo = 0; _wo < wo; ++_wo) {
          int max_index = 0;
          T max_value = -INFINITE;
          for (int _hf = 0; _hf < hf; ++_hf) {
            for (int _wf = 0; _wf < wf; ++_wf) {
              int index = PtoI4D(_n, _c, _ho * hs + _hf, _wo * ws + _wf, n, c, hi, wi);
              if (value[index] > max_value) {
                max_value = value[index];
                max_index = index;
              }
            }
          }
          value_grad[max_index] += output_grad[PtoI4D(_n, _c, _ho, _wo, n, c, ho, wo)];
        }
      }
    }
  }
}

template <typename T>
void avg_pool2d(T *output,
                T *value,
                int *ksize,
                int *strides,
                int *paddings,
                int *output_shape,
                int *value_shape) {
  int n = value_shape[0];
  int c = value_shape[1];
  int hi = value_shape[2];
  int wi = value_shape[3];

  int ho = output_shape[2];
  int wo = output_shape[3];

  int hf = ksize[0];
  int wf = ksize[1];

  int hs = strides[0];
  int ws = strides[1];

  int pad_t = paddings[0];
  int pad_b = paddings[1];
  int pad_l = paddings[2];
  int pad_r = paddings[3];

  for (int _n = 0; _n < n; ++_n) {
    for (int _c = 0; _c < c; ++_c) {
      for (int _ho = 0; _ho < ho; ++_ho) {
        for (int _wo = 0; _wo < wo; ++_wo) {
          T sum = 0;
          int value_cnt = 0;
          for (int _hf = 0; _hf < hf; ++_hf) {
            int _hi = _ho * hs + _hf;
            if (_hi < pad_t || _hi >= hi - pad_b) {
              continue;
            }
            for (int _wf = 0; _wf < wf; ++_wf) {
              int _wi = _wo * ws + _wf;
              if (_wi < pad_l || _wi >= wi - pad_r) {
                continue;
              }

              sum += value[PtoI4D(_n, _c, _hi, _wi, n, c, hi, wi)];
              ++value_cnt;
            }
          }
          output[PtoI4D(_n, _c, _ho, _wo, n, c, ho, wo)] = sum / value_cnt;
        }
      }
    }
  }
}

template <typename T>
void avg_pool2d_backward(T *value_grad,
                         T *output_grad,
                         int *ksize,
                         int *strides,
                         int *paddings,
                         int *output_shape,
                         int *value_shape,
                         T alpha) {
  int n = value_shape[0];
  int c = value_shape[1];
  int hi = value_shape[2];
  int wi = value_shape[3];

  int ho = output_shape[2];
  int wo = output_shape[3];

  int hf = ksize[0];
  int wf = ksize[1];

  int hs = strides[0];
  int ws = strides[1];

  int pad_t = paddings[0];
  int pad_b = paddings[1];
  int pad_l = paddings[2];
  int pad_r = paddings[3];

  if (alpha == 0) {
    memset(value_grad, 0, sizeof(T) * n * c * hi * wi);
  }  
  else if (alpha != 1) {
    for (int i = 0; i < n * c * hi * wi; ++i) {
      value_grad[i] *= alpha;
    }
  }

  for (int _n = 0; _n < n; ++_n) {
    for (int _c = 0; _c < c; ++_c) {
      for (int _ho = 0; _ho < ho; ++_ho) {
        for (int _wo = 0; _wo < wo; ++_wo) {
          int value_cnt = 0;
          for (int _hf = 0; _hf < hf; ++_hf) {
            for (int _wf = 0; _wf < wf; ++_wf) {
              int _hi = _ho * hs + _hf;
              int _wi = _wo * ws + _wf;
              if (_hi < pad_t || _hi >= hi - pad_b) {
                continue;
              }
              else if (_wi < pad_l || _wi >= wi - pad_r) {
                continue;
              }

              ++value_cnt;
            }
          }

          T grad = output_grad[PtoI4D(_n, _c, _ho, _wo, n, c, ho, wo)] / value_cnt;
          for (int _hf = 0; _hf < hf; ++_hf) {
            for (int _wf = 0; _wf < wf; ++_wf) {
              int _hi = _ho * hs + _hf;
              int _wi = _wo * ws + _wf;
              if (_hi < pad_t || _hi >= hi - pad_b) {
                continue;
              }
              else if (_wi < pad_l || _wi >= wi - pad_r) {
                continue;
              }

              value_grad[PtoI4D(_n, _c, _hi, _wi, n, c, hi, wi)] += grad;
            }
          }
        }
      }
    }
  }
}

extern "C" {

void max_pool2d_FP32(float *output,
                     float *value,
                     int *ksize,
                     int *strides,
                     int *output_shape,
                     int *value_shape) {
  max_pool2d<float>(output,
                    value,
                    ksize,
                    strides,
                    output_shape,
                    value_shape);
}

void max_pool2d_backward_FP32(float *value_grad,
                              float *output_grad,
                              float *value,
                              int *ksize,
                              int *strides,
                              int *output_shape,
                              int *value_shape,
                              float alpha) {
  max_pool2d_backward<float>(value_grad,
                             output_grad,
                             value,
                             ksize,
                             strides,
                             output_shape,
                             value_shape,
                             alpha);
}

void avg_pool2d_FP32(float *output,
                     float *value,
                     int *ksize,
                     int *strides,
                     int *paddings,
                     int *output_shape,
                     int *value_shape) {
  avg_pool2d<float>(output,
                    value,
                    ksize,
                    strides,
                    paddings,
                    output_shape,
                    value_shape);
}

void avg_pool2d_backward_FP32(float *value_grad,
                              float *output_grad,
                              int *ksize,
                              int *strides,
                              int *paddings,
                              int *output_shape,
                              int *value_shape,
                              float alpha) {
  avg_pool2d_backward<float>(value_grad,
                             output_grad,
                             ksize,
                             strides,
                             paddings,
                             output_shape,
                             value_shape,
                             alpha);
}

}  // extern "C"
