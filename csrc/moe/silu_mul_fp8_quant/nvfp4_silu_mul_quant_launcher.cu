// Launcher implementation for NVFP4 SiLU+Mul quantization kernel.
// Compiled separately for fast kernel iteration.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "nvfp4_silu_mul_quant.cuh"
#include "nvfp4_silu_mul_quant_launcher.h"

namespace nvfp4 {

void launch_silu_mul_nvfp4_quant(void* output, void* output_scale,
                                 void const* input,
                                 void const* input_global_scale, void* mask,
                                 int32_t m_topk, int32_t k, int32_t n_experts,
                                 cudaStream_t stream, int grid_size_override,
                                 int block_size_override) {
  // Tuned grid/block lookup table from upstream
  struct TunedPoint {
    int m_topk;
    int grid;
    int block;
  };
  static constexpr TunedPoint kTuned[] = {
      {96, 256, 64},    {192, 384, 64},     {384, 512, 128},
      {768, 768, 128},  {1024, 1024, 128},  {3072, 512, 128},
      {8192, 768, 128}, {16384, 1216, 128}, {32768, 1216, 128},
  };
  static constexpr int kNumTuned = sizeof(kTuned) / sizeof(kTuned[0]);

  int tuned_grid = kTuned[kNumTuned - 1].grid;
  int tuned_block = kTuned[kNumTuned - 1].block;
  for (int i = 0; i < kNumTuned; ++i) {
    if (m_topk <= kTuned[i].m_topk) {
      if (i == 0) {
        tuned_grid = kTuned[0].grid;
        tuned_block = kTuned[0].block;
      } else {
        float t = static_cast<float>(m_topk - kTuned[i - 1].m_topk) /
                  (kTuned[i].m_topk - kTuned[i - 1].m_topk);
        tuned_grid =
            kTuned[i - 1].grid +
            static_cast<int>(t * (kTuned[i].grid - kTuned[i - 1].grid));
        tuned_block = kTuned[i].block;
      }
      break;
    }
  }
  tuned_grid = (tuned_grid + n_experts - 1) / n_experts * n_experts;

  dim3 grid(tuned_grid);
  dim3 block(tuned_block);

  if (grid_size_override > 0) {
    grid.x = grid_size_override;
  }
  if (block_size_override > 0) {
    block.x = block_size_override;
  }

  cvt_fp16_to_fp4_expert<__nv_bfloat16, false><<<grid, block, 0, stream>>>(
      m_topk, k, reinterpret_cast<__nv_bfloat16 const*>(input),
      reinterpret_cast<float const*>(input_global_scale),
      reinterpret_cast<uint32_t*>(output),
      reinterpret_cast<uint32_t*>(output_scale),
      reinterpret_cast<int32_t*>(mask), n_experts);
}

}  // namespace nvfp4
