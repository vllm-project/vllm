// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include "input_conv_launch.cuh"

namespace vllm_mamba_input_conv {
cudaError_t launch_batch_wide(int tokens, int rows, const int4* input,
                              const int4* weight, int4* projected,
                              const int4* scales, const float* global_scale,
                              ConvParams conv, cudaStream_t stream) {
  return detail::launch_tokens<ConvFamily::Wide, false>(
      tokens, rows, input, weight, projected, scales, global_scale, conv,
      stream, std::integer_sequence<int, 3, 4, 5, 6, 7, 8>{});
}
}  // namespace vllm_mamba_input_conv
