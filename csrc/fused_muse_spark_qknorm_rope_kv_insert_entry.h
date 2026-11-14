/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 *
 * Declaration of the fused MUSE-Spark qknorm/rope/kv-insert entry, one
 * instantiation per head_dim. The API unit dispatches on head_dim and only
 * needs this header; the kernels stay in
 * fused_muse_spark_qknorm_rope_kv_insert_kernel.cuh, which the per-head_dim
 * units instantiate through VLLM_MUSE_SPARK_ENTRY below.
 *
 * The parameter list is spelled once here and reused by the declaration, the
 * `extern template` lines and the explicit instantiations, so the three cannot
 * drift apart into different overloads.
 */

#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include "torch_utils.h"

#define VLLM_MUSE_SPARK_ENTRY_SIG                                             \
  torch::stable::Tensor &qkv, torch::stable::Tensor const &positions,         \
      int64_t num_heads, int64_t num_kv_heads, int64_t head_dim, double eps,  \
      std::optional<torch::stable::Tensor> cos_sin_cache, int64_t rotary_dim, \
      std::optional<torch::stable::Tensor> slot_mapping,                      \
      std::optional<torch::stable::Tensor> kv_cache, int64_t block_size,      \
      const std::string &kv_cache_dtype,                                      \
      std::optional<torch::stable::Tensor> k_scale,                           \
      std::optional<torch::stable::Tensor> v_scale,                           \
      std::optional<torch::stable::Tensor> q_fp8_out,                         \
      std::optional<torch::stable::Tensor> q_scale

// One instantiation per supported head_dim; EXT is `extern` to name an
// instantiation that another unit provides, and empty in the unit that
// provides it.
#define VLLM_MUSE_SPARK_ENTRY(EXT, HD)                                \
  EXT template void fused_muse_spark_qknorm_rope_kv_insert_typed<HD>( \
      VLLM_MUSE_SPARK_ENTRY_SIG)

template <int HEAD_DIM>
void fused_muse_spark_qknorm_rope_kv_insert_typed(VLLM_MUSE_SPARK_ENTRY_SIG);

VLLM_MUSE_SPARK_ENTRY(extern, 64);
VLLM_MUSE_SPARK_ENTRY(extern, 128);
VLLM_MUSE_SPARK_ENTRY(extern, 256);
