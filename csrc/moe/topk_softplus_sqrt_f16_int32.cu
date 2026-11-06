// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Fused softplus-sqrt topk launcher: gating_output=half, topk_indices=int32.

#include "topk_softplus_sqrt_kernels.cuh"

namespace vllm::moe {

VLLM_TOPK_SOFTPLUS_SQRT_INSTANTIATE(int, __half, int64_t)
VLLM_TOPK_SOFTPLUS_SQRT_INSTANTIATE(int, __half, int)

}  // namespace vllm::moe
