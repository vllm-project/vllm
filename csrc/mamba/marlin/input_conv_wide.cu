// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Wider uniform batches retain one complete-M, full-K Marlin weight stream.
// Original Marlin/Neural Magic/IST-DASLab attribution remains in the template.
#include "../../quantization/marlin/kernel.h"

 #include "input_conv_epilogue.cuh"
#include "input_conv_kernel.cuh"
#undef VLLM_MAMBA_INPUT_CONV_WIDE

namespace vllm_mamba_input_conv {
namespace {
template <int TileRows, int Tokens>
cudaError_t launch(const int4* input, const int4* weight, int4* projected,
                   int4* scratch, const int4* scales, const float* global_scale,
                   int* locks, ConvParams conv, int rows, cudaStream_t stream) {
  static_assert(TileRows == 16 || TileRows == 32);
  static_assert((TileRows == 16 && Tokens >= 3 && Tokens <= 8) ||
                (TileRows == 32 && Tokens >= 5 && Tokens <= 8));
  constexpr auto kernel =
      marlin::MarlinMambaInputConv<vllm::kBFloat16.id(), vllm::kFE2M1f.id(),
                                       vllm::kBFloat16.id(),
                                       vllm::kFE4M3fn.id(), 128, TileRows / 16,
                                       8, 4, false, 3, 1, false, Tokens, ConvFamily::Wide>;
  constexpr int shared = 12288 + 1536 + 3 * TileRows * 64 * 2;
  static_assert(shared == (TileRows == 16 ? 19968 : 26112));
  auto status = cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared);
  if (status != cudaSuccess) return status;
  marlin::MarlinMambaInputConv<
      vllm::kBFloat16.id(), vllm::kFE2M1f.id(), vllm::kBFloat16.id(),
      vllm::kFE4M3fn.id(), 128, TileRows / 16, 8, 4, false, 3, 1, false, Tokens, ConvFamily::Wide>
      <<<145, 128, shared, stream>>>(input, weight, projected, scratch, nullptr,
                                     scales, global_scale, nullptr, rows, 18560,
                                     4096, 4096, locks, false, false, true,
                                     shared, conv);
  return cudaGetLastError();
}
}  // namespace

cudaError_t launch_batch_wide(int tokens, int rows, const int4* input,
                              const int4* weight, int4* projected,
                              int4* scratch, const int4* scales,
                              const float* global_scale, int* locks,
                              ConvParams conv, cudaStream_t stream) {
#define VLLM_LAUNCH_WIDE(M, T)                                     \
  case T:                                                          \
    return launch<M, T>(input, weight, projected, scratch, scales, \
                        global_scale, locks, conv, rows, stream);
  if (rows > 8 && rows <= 16) {
    switch (tokens) {
      VLLM_LAUNCH_WIDE(16, 3);
      VLLM_LAUNCH_WIDE(16, 4);
      VLLM_LAUNCH_WIDE(16, 5);
      VLLM_LAUNCH_WIDE(16, 6);
      VLLM_LAUNCH_WIDE(16, 7);
      VLLM_LAUNCH_WIDE(16, 8);
    }
  } else if (rows > 16 && rows <= 32) {
    switch (tokens) {
      VLLM_LAUNCH_WIDE(32, 5);
      VLLM_LAUNCH_WIDE(32, 6);
      VLLM_LAUNCH_WIDE(32, 7);
      VLLM_LAUNCH_WIDE(32, 8);
    }
  }
#undef VLLM_LAUNCH_WIDE
  return cudaErrorInvalidValue;
}
}  // namespace vllm_mamba_input_conv
