#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>

#include "quantization/w8a8/per_token_group_quant_8bit.h"

void per_token_group_quant_int8(const torch::Tensor& input,
                                torch::Tensor& output_q,
                                torch::Tensor& output_s, int64_t group_size,
                                double eps, double int8_min, double int8_max) {
  per_token_group_quant_8bit(input, output_q, output_s, group_size, eps,
                             int8_min, int8_max);
}