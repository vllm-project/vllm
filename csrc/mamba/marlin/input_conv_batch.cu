// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Batched M<=8 Marlin input projection + BF16 convolution. The device template
// preserves its original Neural Magic / Elias Frantar / IST-DASLab notices.

#include "../../quantization/marlin/kernel.h"
 #include "input_conv_epilogue.cuh"

#include "input_conv_kernel.cuh"
#undef VLLM_MAMBA_INPUT_CONV_BATCH
namespace vllm_mamba_input_conv {
namespace {
template <int Tokens>
cudaError_t launch_batch8(const int4* input, const int4* weight,
                          int4* projected, int4* scratch, const int4* scales,
                          const float* global_scale, int* locks,
                          ConvParams conv, int rows, cudaStream_t stream) {
  constexpr auto kernel = marlin::MarlinMambaInputConv<
      vllm::kBFloat16.id(), vllm::kFE2M1f.id(), vllm::kBFloat16.id(),
      vllm::kFE4M3fn.id(), 128, 1, 8, 4, true, 3, 1, false, Tokens, ConvFamily::Batched>;
  constexpr int shared = 16896;
  auto status = cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared);
  if (status != cudaSuccess) return status;
  marlin::MarlinMambaInputConv<vllm::kBFloat16.id(), vllm::kFE2M1f.id(),
                                    vllm::kBFloat16.id(), vllm::kFE4M3fn.id(),
                                    128, 1, 8, 4, true, 3, 1, false, Tokens, ConvFamily::Batched>
      <<<145, 128, shared, stream>>>(input, weight, projected, scratch, nullptr,
                                     scales, global_scale, nullptr, rows, 18560,
                                     4096, 4096, locks, false, false, true,
                                     shared, conv);
  return cudaGetLastError();
}
}  // namespace
}  // namespace vllm_mamba_input_conv
namespace vllm_mamba_input_conv {
cudaError_t launch_batch_small(int tokens, int rows, const int4* input,
                              const int4* weight, int4* projected, int4* scratch,
                              const int4* scales, const float* global_scale,
                              int* locks, ConvParams conv, cudaStream_t stream) {
  switch (tokens) {
    case 1: return launch_batch8<1>(input, weight, projected, scratch, scales, global_scale, locks, conv, rows, stream);
    case 2: return launch_batch8<2>(input, weight, projected, scratch, scales, global_scale, locks, conv, rows, stream);
    case 3: return launch_batch8<3>(input, weight, projected, scratch, scales, global_scale, locks, conv, rows, stream);
    case 4: return launch_batch8<4>(input, weight, projected, scratch, scales, global_scale, locks, conv, rows, stream);
  }
  return cudaErrorInvalidValue;
}
}  // namespace vllm_mamba_input_conv
