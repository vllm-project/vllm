#pragma once

#include <torch/csrc/stable/tensor.h>

// Internal shared function for 8-bit per-token-group quantization.
// Defined in fp8/per_token_group_quant.cu, used by both FP8 and INT8 kernels.
void per_token_group_quant_8bit(const torch::stable::Tensor& input,
                                torch::stable::Tensor& output_q,
                                torch::stable::Tensor& output_s,
                                int64_t group_size, double eps, double min_8bit,
                                double max_8bit, bool scale_ue8m0 = false);
