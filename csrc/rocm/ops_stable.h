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
#include <string>

// Paged attention, registered by paged_attention_torch_bindings.cpp.

void paged_attention(
    torch::stable::Tensor& out, torch::stable::Tensor& exp_sums,
    torch::stable::Tensor& max_logits, torch::stable::Tensor& tmp_out,
    torch::stable::Tensor& query, torch::stable::Tensor& key_cache,
    torch::stable::Tensor& value_cache, int64_t num_kv_heads, double scale,
    torch::stable::Tensor& block_tables, torch::stable::Tensor& seq_lens,
    const std::optional<torch::stable::Tensor>& query_start_loc,
    int64_t block_size, int64_t max_seq_len,
    const std::optional<torch::stable::Tensor>& alibi_slopes,
    const std::string& kv_cache_dtype, torch::stable::Tensor& k_scale,
    torch::stable::Tensor& v_scale,
    const std::optional<torch::stable::Tensor>& fp8_out_scale,
    const std::string& mfma_type, bool use_interleaved_v_cache);
