// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Short-T instantiations reuse the same attributed Marlin template core.
#include "kernel.h"
#include "mamba_input_conv_short_epilogue.cuh"
#include "mamba_input_conv_short_launch.h"
#define VLLM_MAMBA_INPUT_CONV_SHORT
#include "mamba_input_conv_template.h"
#undef VLLM_MAMBA_INPUT_CONV_SHORT

namespace vllm_mamba_input_conv {
namespace {
template <int Tokens>
cudaError_t launch(const int4* input, const int4* weight, int4* projected,
                   int4* scratch, const int4* scales, const float* global_scale,
                   int* locks, ConvParams conv, cudaStream_t stream) {
  constexpr auto kernel = marlin::MarlinMambaInputConvShort<
      vllm::kBFloat16.id(), vllm::kFE2M1f.id(), vllm::kBFloat16.id(),
      vllm::kFE4M3fn.id(), 128, 1, 8, 4, true, 3, 1, false, Tokens>;
  constexpr int shared = 16896;
  // Same fixed shared requirement as the legacy full-K S3 kernel.
  auto status = cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared);
  if (status != cudaSuccess) return status;
  marlin::MarlinMambaInputConvShort<vllm::kBFloat16.id(), vllm::kFE2M1f.id(),
                                    vllm::kBFloat16.id(), vllm::kFE4M3fn.id(),
                                    128, 1, 8, 4, true, 3, 1, false, Tokens>
      <<<145, 128, shared, stream>>>(input, weight, projected, scratch, nullptr,
                                     scales, global_scale, nullptr, Tokens,
                                     18560, 4096, 4096, locks, false, false,
                                     true, shared, conv);
  return cudaGetLastError();
}
}  // namespace
cudaError_t launch_short(int tokens, const int4* input, const int4* weight,
                         int4* projected, int4* scratch, const int4* scales,
                         const float* global_scale, int* locks, ConvParams conv,
                         cudaStream_t stream) {
#define VLLM_LAUNCH_SHORT(T)                                                  \
  case T:                                                                     \
    return launch<T>(input, weight, projected, scratch, scales, global_scale, \
                     locks, conv, stream);
  switch (tokens) {
    VLLM_LAUNCH_SHORT(1);
    VLLM_LAUNCH_SHORT(2);
    VLLM_LAUNCH_SHORT(3);
    VLLM_LAUNCH_SHORT(4);
    VLLM_LAUNCH_SHORT(5);
    VLLM_LAUNCH_SHORT(6);
    VLLM_LAUNCH_SHORT(7);
    default:
      return cudaErrorInvalidValue;
  }
#undef VLLM_LAUNCH_SHORT
}
}  // namespace vllm_mamba_input_conv
