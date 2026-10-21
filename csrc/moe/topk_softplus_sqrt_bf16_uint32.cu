// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Instantiations of the fused softplus-sqrt topk launcher for
// gating_output=bfloat16, topk_indices=uint32.

#include "topk_softplus_sqrt_kernels.cuh"

namespace vllm {
namespace moe {

VLLM_TOPK_SOFTPLUS_SQRT_INSTANTIATE(uint32_t, __nv_bfloat16, int64_t)
VLLM_TOPK_SOFTPLUS_SQRT_INSTANTIATE(uint32_t, __nv_bfloat16, int)
VLLM_TOPK_SOFTPLUS_SQRT_INSTANTIATE(uint32_t, __nv_bfloat16, uint32_t)

}  // namespace moe
}  // namespace vllm
