// Shared device-side activation function templates used by both
// activation_kernels.cu and quantization/activation_kernels.cu.
//
// Previously, silu_kernel (and its scalar helpers) were defined independently
// in each translation unit.  Moving them here eliminates that duplication.
#pragma once

namespace vllm {

template <typename T>
__device__ __forceinline__ T silu_kernel(const T& x) {
  // x * sigmoid(x)
  return (T)(((float)x) / (1.0f + expf((float)-x)));
}

}  // namespace vllm
