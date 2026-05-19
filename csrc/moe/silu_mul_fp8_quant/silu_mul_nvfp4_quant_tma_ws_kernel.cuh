#pragma once

// BF16 → SiLU+Mul → NVFP4 (e2m1) warp-specialized TMA kernel.
// Producer warp (warpId 0) loads BF16 gate+up via TMA tensor descriptors
// with SWIZZLE_128B into smem to reduce bank conflicts from 8-way to 4-way.
// Consumer warps call cvt_silu_mul_fp16_to_fp4 and write FP4 + swizzled FP8
// scales.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "nvfp4_utils.cuh"
#include "silu_mul_fp8_quant_tma_ws_kernel.cuh"

namespace vllm {
namespace tma_v5 {

template <int N_COMPUTE = 7, int NUM_STAGES = 4, int BATCH_SIZE = 2,
          bool USE_TANH_SILU = false>
__global__ void __launch_bounds__((N_COMPUTE + 1) * 32)
    silu_mul_nvfp4_quant_tma_ws_kernel_bf16(
        const __nv_bfloat16* __restrict__ input, uint32_t* __restrict__ output,
        uint32_t* __restrict__ output_sf,
        const float* __restrict__ global_scale_ptr, int32_t n_tokens, int64_t H,
        const __grid_constant__ CUtensorMap tensorMap) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  static constexpr int ELTS_PER_THREAD = 16;
  static constexpr int WARP_ELTS = 32 * ELTS_PER_THREAD;  // 512
  static constexpr int NC_SLICE_BYTES = N_COMPUTE * WARP_ELTS * 2;
  static constexpr int ROW_BYTES = 2 * NC_SLICE_BYTES;
  static constexpr int STAGE_BYTES = BATCH_SIZE * ROW_BYTES;
  static constexpr int MBAR_REGION = ((2 * NUM_STAGES * 8) + 1023) & ~1023;

  int const warpId = threadIdx.x / 32;
  int const laneId = threadIdx.x % 32;
  bool const isProducer = (warpId == 0);

  int const totalN = n_tokens;
  int const halfDim = static_cast<int>(H);
  int const numSFBlocks = halfDim / 16;
  float const globalScaleVal = *global_scale_ptr;

  int const numGroups = halfDim / WARP_ELTS;
  int const validGroups =
      min(N_COMPUTE, numGroups - (int)blockIdx.x * N_COMPUTE);
  int const validSliceBytes = validGroups * WARP_ELTS * 2;

  extern __shared__ char smem_raw[];
  uint64_t* full_mbar = reinterpret_cast<uint64_t*>(smem_raw);
  uint64_t* empty_mbar = &full_mbar[NUM_STAGES];

  if (threadIdx.x == 0) {
  #pragma unroll
    for (int s = 0; s < NUM_STAGES; s++) {
      mbarrier_init(&full_mbar[s], 1);
      mbarrier_init(&empty_mbar[s], N_COMPUTE);
    }
  #pragma unroll
    for (int s = 0; s < NUM_STAGES; s++) {
      for (int i = 0; i < N_COMPUTE; i++) {
        mbarrier_arrive(&empty_mbar[s]);
      }
    }
    fence_async_shared();
    fence_barrier_init();
  }
  __syncthreads();

  int const gridY = static_cast<int>(gridDim.y);
  int const batchStride = gridY * BATCH_SIZE;
  int const firstBatch = static_cast<int>(blockIdx.y) * BATCH_SIZE;

  if (firstBatch >= totalN) {
    if (threadIdx.x == 0) {
  #pragma unroll
      for (int s = 0; s < NUM_STAGES; s++) {
        mbarrier_inval(&full_mbar[s]);
        mbarrier_inval(&empty_mbar[s]);
      }
    }
    return;
  }

  int const gateDim1 = static_cast<int>(blockIdx.x) * N_COMPUTE * 8;
  int const upDim1 = halfDim / 64 + gateDim1;

  auto stage_ptr = [&](int stage) -> char* {
    return smem_raw + MBAR_REGION + stage * STAGE_BYTES;
  };

  int totalBatches = 0;
  for (int b = firstBatch; b < totalN; b += batchStride) totalBatches++;

  if (isProducer) {
    // ===== PRODUCER WARP =====
    if (laneId != 0) return;

    int fillStage = 0;
    int phase_empty[NUM_STAGES] = {};
    int nextBatchStart = firstBatch;

    for (int batch = 0; batch < totalBatches; batch++) {
      mbarrier_wait(&empty_mbar[fillStage], phase_empty[fillStage]);
      phase_empty[fillStage] ^= 1;

      int actual_load = min(BATCH_SIZE, max(0, totalN - nextBatchStart));
      if (actual_load > 0) {
        uint32_t load_bytes =
            static_cast<uint32_t>(actual_load * 2 * validSliceBytes);
        mbarrier_arrive_expect_tx(&full_mbar[fillStage], load_bytes);
        char* dst = stage_ptr(fillStage);
        for (int t = 0; t < actual_load; t++) {
          for (int w = 0; w < validGroups; w++) {
            tma_load_3d(dst + t * ROW_BYTES + w * 1024, tensorMap, 0,
                        gateDim1 + w * 8, nextBatchStart + t,
                        &full_mbar[fillStage]);
            tma_load_3d(dst + t * ROW_BYTES + NC_SLICE_BYTES + w * 1024,
                        tensorMap, 0, upDim1 + w * 8, nextBatchStart + t,
                        &full_mbar[fillStage]);
          }
        }
      } else {
        mbarrier_arrive_expect_tx(&full_mbar[fillStage], 0);
      }

      nextBatchStart += batchStride;
      fillStage = (fillStage + 1) % NUM_STAGES;
    }

  #pragma unroll
    for (int s = 0; s < NUM_STAGES; s++) {
      mbarrier_inval(&full_mbar[s]);
      mbarrier_inval(&empty_mbar[s]);
    }
  } else {
    // ===== CONSUMER WARPS =====
    int const consumerWarpId = warpId - 1;

    int const warpElemStart =
        blockIdx.x * N_COMPUTE * WARP_ELTS + consumerWarpId * WARP_ELTS;
    bool const valid = warpElemStart < halfDim;

    int const sfBlockGlobal =
        (blockIdx.x * N_COMPUTE + consumerWarpId) * 32 + laneId;
    bool const validSF = sfBlockGlobal < numSFBlocks;

    int const elemBase = warpElemStart + laneId * ELTS_PER_THREAD;
    int const warpSliceOff = consumerWarpId * WARP_ELTS * 2;
    int const outByteOff = elemBase / 2;

    // Swizzle address computation for SWIZZLE_128B
    int const sw_row = laneId / 4;
    int const sw_col = (laneId % 4) * 32;
    int const sw_xor = (sw_row & 7) << 4;
    int const sw_lo = sw_col ^ sw_xor;
    int const sw_hi = (sw_col + 16) ^ sw_xor;

    using PVec16 = nvfp4::PackedVec<__nv_bfloat16, 16>;

    int consumeStage = 0;
    int phase_full[NUM_STAGES] = {};

    for (int batch = 0; batch < totalBatches; batch++) {
      int batchStart = firstBatch + batch * batchStride;

      mbarrier_wait(&full_mbar[consumeStage], phase_full[consumeStage]);
      phase_full[consumeStage] ^= 1;

      if (valid) {
        char* sp = stage_ptr(consumeStage);
        int actual_bs = min(BATCH_SIZE, totalN - batchStart);

        for (int t = 0; t < actual_bs; t++) {
          int token = batchStart + t;

          PVec16 gate_vec, up_vec;
          char* gate_base = sp + t * ROW_BYTES + warpSliceOff + sw_row * 128;
          char* up_base =
              sp + t * ROW_BYTES + NC_SLICE_BYTES + warpSliceOff + sw_row * 128;
          reinterpret_cast<nvfp4::PackedU32x4&>(gate_vec.elts[0]) =
              *reinterpret_cast<nvfp4::PackedU32x4 const*>(gate_base + sw_lo);
          reinterpret_cast<nvfp4::PackedU32x4&>(gate_vec.elts[4]) =
              *reinterpret_cast<nvfp4::PackedU32x4 const*>(gate_base + sw_hi);
          reinterpret_cast<nvfp4::PackedU32x4&>(up_vec.elts[0]) =
              *reinterpret_cast<nvfp4::PackedU32x4 const*>(up_base + sw_lo);
          reinterpret_cast<nvfp4::PackedU32x4&>(up_vec.elts[4]) =
              *reinterpret_cast<nvfp4::PackedU32x4 const*>(up_base + sw_hi);

          uint8_t* sf_out = validSF
                                ? nvfp4::get_sf_out_offset(token, sfBlockGlobal,
                                                           halfDim, output_sf)
                                : nullptr;

          uint64_t fp4 =
              nvfp4::cvt_silu_mul_fp16_to_fp4<__nv_bfloat16, 16, 16, false>(
                  gate_vec, up_vec, globalScaleVal, sf_out);

          *reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(output) +
                                       (int64_t)token * (halfDim / 2) +
                                       outByteOff) = fp4;
        }
      }

      if (laneId == 0) mbarrier_arrive(&empty_mbar[consumeStage]);
      consumeStage = (consumeStage + 1) % NUM_STAGES;
    }
  }
#endif
}

}  // namespace tma_v5
}  // namespace vllm
