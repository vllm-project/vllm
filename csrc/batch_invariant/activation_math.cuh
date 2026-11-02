// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once

namespace vllm::batch_invariant {

__device__ __forceinline__ float silu(const float& val) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  float half = 0.5f * val;
  float t = __tanhf(half);
  return half * (1.0f + t);
#else
  return val / (1.0f + expf(-val));
#endif
}

}  // namespace vllm::batch_invariant
