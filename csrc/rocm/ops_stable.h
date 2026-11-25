// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Declarations of the ROCm ops built on the LibTorch stable ABI (linked into
// _C_stable_libtorch). They cannot live in rocm/ops.h while it includes
// <torch/all.h>, which errors out when TORCH_TARGET_VERSION is defined.
#pragma once

#include <torch/csrc/stable/tensor.h>

#include <cstdint>
#include <optional>

// Skinny GEMMs, registered by skinny_gemms_torch_bindings.cpp.

torch::stable::Tensor LLMM1(torch::stable::Tensor& in_a,
                            torch::stable::Tensor& in_b,
                            const int64_t rows_per_block);

torch::stable::Tensor wvSplitK(
    const torch::stable::Tensor& in_a, const torch::stable::Tensor& in_b,
    const std::optional<torch::stable::Tensor>& in_bias, const int64_t CuCount);

torch::stable::Tensor wvSplitK_int4_g(
    const torch::stable::Tensor& in_a, const torch::stable::Tensor& in_b,
    const torch::stable::Tensor& in_scale,
    const std::optional<torch::stable::Tensor>& in_zero_points,
    const std::optional<torch::stable::Tensor>& in_bias, const int64_t CuCount,
    const int64_t group_size);

torch::stable::Tensor wvSplitKrc(
    const torch::stable::Tensor& in_a, const torch::stable::Tensor& in_b,
    const std::optional<torch::stable::Tensor>& in_bias, const int64_t CuCount);

torch::stable::Tensor swmmacGEMM(
    const torch::stable::Tensor& in_a, const torch::stable::Tensor& in_b,
    const std::optional<torch::stable::Tensor>& in_bias,
    const int64_t logical_M, const int64_t CuCount);

void wvSplitKQ(const torch::stable::Tensor& in_a,
               const torch::stable::Tensor& in_b,
               const std::optional<torch::stable::Tensor>& in_bias,
               torch::stable::Tensor& out_c,
               const torch::stable::Tensor& scale_a,
               const torch::stable::Tensor& scale_b, const int64_t CuCount);
