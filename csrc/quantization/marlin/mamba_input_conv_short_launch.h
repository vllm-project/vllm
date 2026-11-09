// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_runtime.h>
#include "mamba_input_conv_params.h"
namespace vllm_mamba_input_conv {
// Ordinary private C++ launch function; CUDA kernels keep the explicit18
// arguments.
cudaError_t launch_short(int tokens, const int4* input, const int4* weight,
                         int4* projected, int4* scratch, const int4* scales,
                         const float* global_scale, int* locks, ConvParams conv,
                         cudaStream_t stream);
}  // namespace vllm_mamba_input_conv
