// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once

#include <cuda_runtime_api.h>

#include "mamba_input_conv_params.h"

namespace vllm_mamba_input_conv {
cudaError_t launch_batch_wide(int tokens, int rows, const int4* input,
                              const int4* weight, int4* projected,
                              int4* scratch, const int4* scales,
                              const float* global_scale, int* locks,
                              ConvParams conv, cudaStream_t stream);
}  // namespace vllm_mamba_input_conv
