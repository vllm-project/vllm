#pragma once

// Forward declarations for NVFP4 SiLU+Mul quantization kernel launcher.
// Void-pointer ABI to avoid CUDA dependencies in torch_ops.cu includes.

#include <cstdint>

typedef struct CUstream_st* cudaStream_t;

namespace nvfp4 {

void launch_silu_mul_nvfp4_quant(void* output, void* output_scale,
                                 void const* input,
                                 void const* input_global_scale, void* mask,
                                 int32_t m_topk, int32_t k, int32_t n_experts,
                                 cudaStream_t stream,
                                 int grid_size_override = -1,
                                 int block_size_override = -1);

}  // namespace nvfp4
