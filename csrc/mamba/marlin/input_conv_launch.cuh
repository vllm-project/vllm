// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once
#include "../../quantization/marlin/kernel.h"
#include "input_conv_epilogue.cuh"
#include "input_conv_kernel.cuh"
#include <utility>

namespace vllm_mamba_input_conv::detail {
template <int Tokens, ConvFamily Family, bool Small>
cudaError_t launch(int rows, const int4* input, const int4* weight,
                   int4* projected, const int4* scales,
                   const float* global_scale, ConvParams conv,
                   cudaStream_t stream) {
  constexpr auto kernel = marlin::MarlinMambaInputConv<Tokens, Family, Small>;
  constexpr int shared = Small ? 16896 : 19968;
  auto status = cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared);
  if (status != cudaSuccess) return status;
  kernel<<<145, 128, shared, stream>>>(input, weight, projected, scales,
                                       global_scale, rows, conv);
  return cudaGetLastError();
}

template <ConvFamily Family, bool Small, int... Tokens>
cudaError_t launch_tokens(int tokens, int rows, const int4* input,
                          const int4* weight, int4* projected,
                          const int4* scales, const float* global_scale,
                          ConvParams conv, cudaStream_t stream,
                          std::integer_sequence<int, Tokens...>) {
  auto status = cudaErrorInvalidValue;
  ((tokens == Tokens &&
    (status = launch<Tokens, Family, Small>(rows, input, weight, projected,
                                            scales, global_scale, conv, stream),
     true)) ||
   ...);
  return status;
}
}  // namespace vllm_mamba_input_conv::detail
