#pragma once
#include <torch/all.h>

// 8-bit per-token-group quantization helper used by both FP8 and INT8
void per_token_group_quant_8bit(const torch::Tensor& input,
                                torch::Tensor& output_q,
                                torch::Tensor& output_s, int64_t group_size,
                                double eps, double min_8bit, double max_8bit,
                                bool scale_ue8m0 = false);

// Fused SiLU+mul + per-token-group FP8 quantization with UE8M0-packed
// scales for DeepGEMM. Input: [mn, 2*N], output: [mn, N] in FP8.
void silu_mul_per_token_group_quant_fp8_packed(const torch::Tensor& input,
                                               torch::Tensor& output_q,
                                               torch::Tensor& output_s_packed,
                                               int64_t group_size, double eps,
                                               double min_8bit,
                                               double max_8bit);