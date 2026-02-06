#pragma once

#include <torch/extension.h>
#include <optional>

// Declaration only - implementation is in .cu file
void silu_and_mul_per_block_quant(
    torch::Tensor& out,
    torch::Tensor const& input,
    torch::Tensor& scales,
    int64_t group_size,
    std::optional<torch::Tensor> scale_ub,
    bool is_scale_transposed
);