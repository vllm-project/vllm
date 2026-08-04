// SPDX-License-Identifier: Apache-2.0

#include <torch/all.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Exception.h>
#include <vector>

void rotary_embedding_k_fused(const torch::Tensor& old_positions,
                              const torch::Tensor& new_positions,
                              torch::Tensor& key, int64_t head_size,
                              const torch::Tensor& cos_sin_cache, bool is_neox);

// rotary_embedding_k_fused with an explicit per-head stride: contiguous keys
// pass head_size; fused K/V pass 2*head_size to rotate only the K half.
void rotary_embedding_k_fused_strided(const torch::Tensor& old_positions,
                                      const torch::Tensor& new_positions,
                                      torch::Tensor& key, int64_t head_size,
                                      int64_t head_stride,
                                      const torch::Tensor& cos_sin_cache,
                                      bool is_neox);

// Fused multi-chunk ramp re-RoPE: one launch rotates up to
// MAX_FUSED_TRANSFER_CHUNKS same-geometry tmp slots.
void rotary_embedding_k_fused_ramp_multi_ptr(
    const std::vector<uintptr_t>& key_ptrs, at::ScalarType key_dtype,
    int64_t num_tokens, const std::vector<int64_t>& old_sts,
    const std::vector<int64_t>& new_sts, int64_t slots, int64_t head_size,
    int64_t head_stride, int64_t num_kv_heads, uintptr_t cos_sin_cache_ptr,
    int rot_dim, bool is_neox);