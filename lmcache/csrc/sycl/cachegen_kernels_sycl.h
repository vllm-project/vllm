// SPDX-License-Identifier: Apache-2.0
//
// Declarations of SYCL/XPU implementations of CacheGen + RoPE kernels.
// Implementations live in csrc/sycl/*.cpp and are exposed to Python via
// lmcache.xpu_ops.
//
#pragma once

#include <torch/all.h>

// CacheGen ---------------------------------------------------------------
//
// Calculate per-(layer, channel) CDF from an int8/uint8 input tensor.
// Input:  [nlayers, ntokens, nchannels] (uint8)
// Output: [nlayers, nchannels, max_bins + 1] (int16)
at::Tensor calculate_cdf_xpu(const at::Tensor& input, int64_t max_bins);

// Arithmetic encoder (forward): produces a byte buffer + per-channel
// length tensor.
void encode_fast_new_xpu(const at::Tensor& cdf, const at::Tensor& input_sym,
                         at::Tensor& output_buffer, at::Tensor& output_lengths);

// Arithmetic decoder (per-channel buffer): inverse of encode_fast_new.
void decode_fast_new_xpu(const at::Tensor& cdf, const at::Tensor& bytestreams,
                         const at::Tensor& lengths, at::Tensor& output);

// Arithmetic decoder (1-D bytestream with prefix-sum offsets):
// inverse of encode_fast_new packed via prefix sums.
void decode_fast_prefsum_xpu(const at::Tensor& cdf,
                             const at::Tensor& bytestreams,
                             const at::Tensor& lengths_prefsum,
                             at::Tensor& output);

// Position encoding -------------------------------------------------------
//
// Fused undo-then-apply rotary embedding on key tensor (in-place).
void rotary_embedding_k_fused_xpu(const torch::Tensor& old_positions,
                                  const torch::Tensor& new_positions,
                                  torch::Tensor& key, int64_t head_size,
                                  const torch::Tensor& cos_sin_cache,
                                  bool is_neox);
