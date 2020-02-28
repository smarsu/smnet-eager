// Copyright (c) 2020 smarsu. All Rights Reserved.

#include <cstring>

#include "_math_utils.h"

template <typename T>
void softmax_cross_entropy_with_logits_backward_logits(T *logits_grad,
                                                       T *output_grad,
                                                       T *label,
                                                       T *softmax_logits,
                                                       int out_dim,
                                                       int dim,
                                                       int inner_dim,
                                                       T alpha) {
  if (alpha == 0) {
    memset(logits_grad, 0, sizeof(T) * out_dim * dim * inner_dim);
  }
  else if (alpha != 1) {
    for (int i = 0; i < out_dim * dim * inner_dim; ++i) {
      logits_grad[i] *= alpha;
    }
  }

  for (int _out_dim = 0; _out_dim < out_dim; ++_out_dim) {
    for (int _dim = 0; _dim < dim; ++_dim) {
      for (int _inner_dim = 0; _inner_dim < inner_dim; ++_inner_dim) {
        for (int _dim2 = 0; _dim2 < dim; ++_dim2) {
          int index = PtoI3D(_out_dim, _dim, _inner_dim, out_dim, dim, inner_dim);
          int index2 = PtoI3D(_out_dim, _dim2, _inner_dim, out_dim, dim, inner_dim);
          if (_dim == _dim2) {
            logits_grad[index2] += output_grad[index] * label[index] * (softmax_logits[index2] - 1);
          }
          else {
            logits_grad[index2] += output_grad[index] * label[index] * softmax_logits[index2];
          }
        }
      }
    }
  }
}

extern "C" {

void softmax_cross_entropy_with_logits_backward_logits_FP32(float *logits_grad,
                                                            float *output_grad,
                                                            float *label,
                                                            float *softmax_logits,
                                                            int out_dim,
                                                            int dim,
                                                            int inner_dim,
                                                            float alpha) {
  softmax_cross_entropy_with_logits_backward_logits<float>(logits_grad,
                                                           output_grad,
                                                           label,
                                                           softmax_logits,
                                                           out_dim,
                                                           dim,
                                                           inner_dim,
                                                           alpha); 
}

}  // extern "C"
