/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 *
 * Declaration of the int4 skinny GEMM and its fused MoE variant, one
 * instantiation per N_in (1..5; block_size_m for the MoE op). The API unit
 * only dispatches on N_in and needs this header; the kernels stay in
 * skinny_gemms_int4.cuh, which the per-N_in units instantiate through
 * VLLM_SKINNY_INT4_INSTANTIATE below.
 *
 * The parameter list is spelled once here and reused by the declaration, the
 * `extern template` lines and the instantiations, so they cannot drift apart
 * into different overloads.
 */

#pragma once

#include <torch/all.h>

#include <cstdint>
#include <optional>

#define VLLM_SKINNY_INT4_SIG                                                  \
  const at::Tensor &in_a, const at::Tensor &in_b, const at::Tensor &in_scale, \
      const std::optional<at::Tensor>&in_zero_points,                         \
      const std::optional<at::Tensor>&in_bias, const int64_t CuCount,         \
      const int64_t group_size

// The fused MoE variant; its N is block_size_m.
#define VLLM_MOE_SKINNY_INT4_SIG                                             \
  const at::Tensor &a, const at::Tensor &w, const at::Tensor &scales,        \
      at::Tensor &c, const at::Tensor &expert_ids,                           \
      const std::optional<at::Tensor>&sorted_token_ids, const int64_t top_k, \
      const int64_t CuCount, const int64_t group_size

// One instantiation per supported N_in; EXT is `extern` to name an
// instantiation that another unit provides, and empty in the unit that
// provides it.
#define VLLM_SKINNY_INT4_DECLARE(EXT, N)                                    \
  EXT template torch::Tensor wvSplitK_int4_g_impl<N>(VLLM_SKINNY_INT4_SIG); \
  EXT template void fused_moe_wvSplitK_int4_gemm_impl<N>(                   \
      VLLM_MOE_SKINNY_INT4_SIG)

#define VLLM_SKINNY_INT4_EXTERN(N) VLLM_SKINNY_INT4_DECLARE(extern, N);
#define VLLM_SKINNY_INT4_INSTANTIATE(N) VLLM_SKINNY_INT4_DECLARE(, N);

template <int N>
torch::Tensor wvSplitK_int4_g_impl(VLLM_SKINNY_INT4_SIG);

template <int N>
void fused_moe_wvSplitK_int4_gemm_impl(VLLM_MOE_SKINNY_INT4_SIG);

VLLM_SKINNY_INT4_EXTERN(1)
VLLM_SKINNY_INT4_EXTERN(2)
VLLM_SKINNY_INT4_EXTERN(3)
VLLM_SKINNY_INT4_EXTERN(4)
VLLM_SKINNY_INT4_EXTERN(5)
