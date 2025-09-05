#pragma once
#include <torch/all.h>

void per_token_group_quant_8bit(const torch::Tensor& input,
                                torch::Tensor& output_q,
                                torch::Tensor& output_s, int64_t group_size,
                                double eps, double min_8bit, double max_8bit,
                                bool scale_ue8m0 = false);