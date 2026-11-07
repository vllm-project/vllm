// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Entry point of the int4 skinny GEMM. The kernels are instantiated one unit
// per N_in (see skinny_gemms_int4_n<n>.cu); this unit only dispatches, so it
// includes the declaration header rather than the kernel header.
#include "skinny_gemms_int4.h"
// ops.h carries the public declaration of wvSplitK_int4_g(), so the defining
// translation unit also sees it (matches attention_dispatch.cu); a mismatch
// with VLLM_SKINNY_INT4_SIG would otherwise only show up as a link error.
#include "ops.h"

#include <stdexcept>
#include <string>

torch::Tensor wvSplitK_int4_g(VLLM_SKINNY_INT4_SIG) {
  switch (in_b.size(0)) {
    case 1:
      return wvSplitK_int4_g_impl<1>(in_a, in_b, in_scale, in_zero_points,
                                     in_bias, CuCount, group_size);
    case 2:
      return wvSplitK_int4_g_impl<2>(in_a, in_b, in_scale, in_zero_points,
                                     in_bias, CuCount, group_size);
    case 3:
      return wvSplitK_int4_g_impl<3>(in_a, in_b, in_scale, in_zero_points,
                                     in_bias, CuCount, group_size);
    case 4:
      return wvSplitK_int4_g_impl<4>(in_a, in_b, in_scale, in_zero_points,
                                     in_bias, CuCount, group_size);
    case 5:
      return wvSplitK_int4_g_impl<5>(in_a, in_b, in_scale, in_zero_points,
                                     in_bias, CuCount, group_size);
    default:
      throw std::runtime_error("Unsupported N value: " +
                               std::to_string(in_b.size(0)));
  }
}

void fused_moe_wvSplitK_int4_gemm(
    const at::Tensor& a, const at::Tensor& w, const at::Tensor& scales,
    at::Tensor& c, const at::Tensor& expert_ids,
    const std::optional<at::Tensor>& sorted_token_ids,
    const int64_t block_size_m, const int64_t top_k, const int64_t CuCount,
    const int64_t group_size) {
  switch (block_size_m) {
    case 1:
      return fused_moe_wvSplitK_int4_gemm_impl<1>(a, w, scales, c, expert_ids,
                                                  sorted_token_ids, top_k,
                                                  CuCount, group_size);
    case 2:
      return fused_moe_wvSplitK_int4_gemm_impl<2>(a, w, scales, c, expert_ids,
                                                  sorted_token_ids, top_k,
                                                  CuCount, group_size);
    case 3:
      return fused_moe_wvSplitK_int4_gemm_impl<3>(a, w, scales, c, expert_ids,
                                                  sorted_token_ids, top_k,
                                                  CuCount, group_size);
    case 4:
      return fused_moe_wvSplitK_int4_gemm_impl<4>(a, w, scales, c, expert_ids,
                                                  sorted_token_ids, top_k,
                                                  CuCount, group_size);
    case 5:
      return fused_moe_wvSplitK_int4_gemm_impl<5>(a, w, scales, c, expert_ids,
                                                  sorted_token_ids, top_k,
                                                  CuCount, group_size);
    default:
      TORCH_CHECK(false, "block_size_m must be in [1, 5], got ", block_size_m);
  }
}
