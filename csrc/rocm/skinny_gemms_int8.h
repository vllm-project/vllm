/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 *
 * Declaration of the int8 skinny GEMM, one instantiation per N_in (1..5). The
 * API unit only dispatches on N_in and needs this header; the kernels stay in
 * skinny_gemms_int8.cuh, which the per-N_in units instantiate through
 * VLLM_SKINNY_INT8_INSTANTIATE below.
 *
 * The parameter list is spelled once here and reused by the declaration, the
 * `extern template` lines and the instantiations, so they cannot drift apart
 * into different overloads.
 */

#pragma once

#include <torch/types.h>

#include <cstdint>
#include <optional>

#define VLLM_SKINNY_INT8_SIG                                                  \
  const at::Tensor &in_a, const at::Tensor &in_b, const at::Tensor &in_scale, \
      const std::optional<at::Tensor>&in_bias, const int64_t CuCount

// One instantiation per supported N_in; EXT is `extern` to name an
// instantiation that another unit provides, and empty in the unit that
// provides it.
#define VLLM_SKINNY_INT8_DECLARE(EXT, N) \
  EXT template torch::Tensor wvSplitK_int8_impl<N>(VLLM_SKINNY_INT8_SIG)

#define VLLM_SKINNY_INT8_EXTERN(N) VLLM_SKINNY_INT8_DECLARE(extern, N);
#define VLLM_SKINNY_INT8_INSTANTIATE(N) VLLM_SKINNY_INT8_DECLARE(, N);

template <int N>
torch::Tensor wvSplitK_int8_impl(VLLM_SKINNY_INT8_SIG);

VLLM_SKINNY_INT8_EXTERN(1)
VLLM_SKINNY_INT8_EXTERN(2)
VLLM_SKINNY_INT8_EXTERN(3)
VLLM_SKINNY_INT8_EXTERN(4)
VLLM_SKINNY_INT8_EXTERN(5)
