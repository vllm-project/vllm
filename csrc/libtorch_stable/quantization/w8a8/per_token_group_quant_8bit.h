#pragma once

#include <torch/csrc/stable/tensor.h>

// 8-bit per-token-group quantization helper used by both FP8 and INT8
void per_token_group_quant_8bit(torch::stable::Tensor input,
                                torch::stable::Tensor output_q,
                                torch::stable::Tensor output_s,
                                int64_t group_size, double eps, double min_8bit,
                                double max_8bit, bool scale_ue8m0 = false);
