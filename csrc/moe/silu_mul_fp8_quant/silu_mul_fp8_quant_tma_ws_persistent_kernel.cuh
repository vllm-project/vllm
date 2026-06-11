#pragma once

// Persistent variant of FP8 -> SiLU+Mul -> FP8 TMA warp-specialized kernel.
// Uses 3D TMA descriptors with SWIZZLE_128B: producer issues tma_load_3d
// per token (gate + up), with swizzle reducing smem bank conflicts.
// CTAs loop until all work is consumed via atomic counter.

#include <cuda.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cstdint>

#include "silu_mul_fp8_quant_tma_ws_kernel.cuh"

namespace vllm {
namespace tma_v5 {

template <int N_COMPUTE = 7, int NUM_STAGES = 2, int BATCH_SIZE = 2,
          bool USE_TANH_SILU = false>
__global__ void __launch_bounds__((N_COMPUTE + 1) * 32)
    silu_mul_fp8_quant_tma_ws_persistent_kernel(
        const __nv_fp8_e4m3* __restrict__ input,
        const float* __restrict__ input_scales,
        __nv_fp8_e4m3* __restrict__ output, float* __restrict__ output_scales,
        int32_t n_tokens, int64_t H, int64_t scale_stride,
        int32_t* __restrict__ work_counter,
        const __grid_constant__ CUtensorMap tensorMap) {
  static constexpr int ELTS_PER_THREAD = 4;
  static constexpr int NC_BYTES = N_COMPUTE * 128;
  static constexpr int TOKEN_BYTES = 2 * NC_BYTES;
  static constexpr int STAGE_BYTES = BATCH_SIZE * TOKEN_BYTES;
  static constexpr int MBAR_REGION =
      ((2 * NUM_STAGES * 8 + sizeof(int32_t)) + 127) & ~127;

  int const warpId = threadIdx.x / 32;
  int const laneId = threadIdx.x % 32;
  bool const isProducer = (warpId == 0);

  int const totalN = n_tokens;
  int const halfDim = static_cast<int>(H);
  int const G = halfDim / SCALE_BLOCK_SIZE;

  int const totalWorkItems = (totalN + BATCH_SIZE - 1) / BATCH_SIZE;

  extern __shared__ char smem_raw[];
  uint64_t* full_mbar = reinterpret_cast<uint64_t*>(smem_raw);
  uint64_t* empty_mbar = &full_mbar[NUM_STAGES];
  int32_t* batch_token_start =
      reinterpret_cast<int32_t*>(&empty_mbar[NUM_STAGES]);

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

  int const gateDim1 = static_cast<int>(blockIdx.x) * N_COMPUTE;
  int const upDim1 = G + gateDim1;
  int const validGroups =
      min(N_COMPUTE, G - static_cast<int>(blockIdx.x) * N_COMPUTE);
  int const validSliceBytes = validGroups * 128;

  auto stage_ptr = [&](int stage) -> char* {
    return smem_raw + MBAR_REGION + stage * STAGE_BYTES;
  };

  constexpr int SCALE_STRIDE = BATCH_SIZE * N_COMPUTE * 2;
  float* smem_scales = reinterpret_cast<float*>(smem_raw + MBAR_REGION +
                                                NUM_STAGES * STAGE_BYTES);

  int32_t* my_counter = &work_counter[blockIdx.x];

  if (isProducer) {
    // ===== PRODUCER WARP =====
    if (laneId != 0) return;

    int fillStage = 0;
    int phase_empty[NUM_STAGES] = {};

    while (true) {
      mbarrier_wait(&empty_mbar[fillStage], phase_empty[fillStage]);
      phase_empty[fillStage] ^= 1;

      int workItem = atomicAdd(my_counter, 1);
      if (workItem >= totalWorkItems) {
        batch_token_start[fillStage] = -1;
        mbarrier_arrive_expect_tx(&full_mbar[fillStage], 0);
        fillStage = (fillStage + 1) % NUM_STAGES;
        break;
      }

      int batchStart = workItem * BATCH_SIZE;
      int actual_load = min(BATCH_SIZE, totalN - batchStart);
      batch_token_start[fillStage] = batchStart;

      int sOff = fillStage * SCALE_STRIDE;
      for (int t = 0; t < actual_load; t++) {
        int tok = batchStart + t;
        for (int w = 0; w < N_COMPUTE; w++) {
          int sb = static_cast<int>(blockIdx.x) * N_COMPUTE + w;
          if (sb < G) {
            cp_async_f32(&smem_scales[sOff + t * N_COMPUTE * 2 + w * 2],
                         &input_scales[tok + scale_stride * sb]);
            cp_async_f32(&smem_scales[sOff + t * N_COMPUTE * 2 + w * 2 + 1],
                         &input_scales[tok + scale_stride * (sb + G)]);
          }
        }
      }
      cp_async_commit();

      char* dst = stage_ptr(fillStage);
      for (int t = 0; t < actual_load; t++) {
        tma_load_3d(dst + t * TOKEN_BYTES, tensorMap, 0, gateDim1,
                    batchStart + t, &full_mbar[fillStage]);
        tma_load_3d(dst + t * TOKEN_BYTES + NC_BYTES, tensorMap, 0, upDim1,
                    batchStart + t, &full_mbar[fillStage]);
      }

      cp_async_wait_all();
      __threadfence_block();
      uint32_t load_bytes =
          static_cast<uint32_t>(actual_load * 2 * validSliceBytes);
      mbarrier_arrive_expect_tx(&full_mbar[fillStage], load_bytes);

      fillStage = (fillStage + 1) % NUM_STAGES;
    }

    for (int remaining = 0; remaining < NUM_STAGES - 1; remaining++) {
      mbarrier_wait(&empty_mbar[fillStage], phase_empty[fillStage]);
      phase_empty[fillStage] ^= 1;
      batch_token_start[fillStage] = -1;
      mbarrier_arrive_expect_tx(&full_mbar[fillStage], 0);
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
    int const scaleBlock = blockIdx.x * N_COMPUTE + consumerWarpId;
    bool const valid = scaleBlock < G;
    int const elemBase =
        valid ? (scaleBlock * SCALE_BLOCK_SIZE + laneId * ELTS_PER_THREAD) : 0;
    int const col = laneId * 4;

    int consumeStage = 0;
    int phase_full[NUM_STAGES] = {};

    while (true) {
      mbarrier_wait(&full_mbar[consumeStage], phase_full[consumeStage]);
      phase_full[consumeStage] ^= 1;

      int batchStart = batch_token_start[consumeStage];
      if (batchStart < 0) break;

      if (valid) {
        char* sp = stage_ptr(consumeStage);
        int actual_bs = min(BATCH_SIZE, totalN - batchStart);

        int t = 0;

        for (; t + 3 < actual_bs; t += 4) {
          int tok[4] = {batchStart + t, batchStart + t + 1, batchStart + t + 2,
                        batchStart + t + 3};

          uint32_t px1[4], px2[4];
#pragma unroll
          for (int k = 0; k < 4; k++) {
            int off = (t + k) * TOKEN_BYTES + consumerWarpId * 128;
            px1[k] = *reinterpret_cast<uint32_t const*>(sp + off + col);
            px2[k] =
                *reinterpret_cast<uint32_t const*>(sp + off + NC_BYTES + col);
          }

          float sc1[4], sc2[4];
          int sOff4 = consumeStage * SCALE_STRIDE;
#pragma unroll
          for (int k = 0; k < 4; k++) {
            sc1[k] = smem_scales[sOff4 + (t + k) * N_COMPUTE * 2 +
                                 consumerWarpId * 2];
            sc2[k] = smem_scales[sOff4 + (t + k) * N_COMPUTE * 2 +
                                 consumerWarpId * 2 + 1];
          }

          float r[4][4];
          float m[4] = {0.0f, 0.0f, 0.0f, 0.0f};

#pragma unroll
          for (int k = 0; k < 4; k++) {
            __nv_fp8_e4m3 x1v[4], x2v[4];
            memcpy(x1v, &px1[k], 4);
            memcpy(x2v, &px2[k], 4);
#pragma unroll
            for (int i = 0; i < 4; i++) {
              float f1 = sc1[k] * fp8_to_float(x1v[i]);
              float f2 = sc2[k] * fp8_to_float(x2v[i]);
              if constexpr (USE_TANH_SILU) {
                r[k][i] = silu_tanh_f(f2) * f1;
              } else {
                r[k][i] = silu_f(f2) * f1;
              }
              m[k] = fmaxf(m[k], fabsf(r[k][i]));
            }
          }

          warp_reduce_max_4(m[0], m[1], m[2], m[3]);

          float s[4];
          if (laneId == 0) {
#pragma unroll
            for (int k = 0; k < 4; k++) {
              s[k] = fmaxf(m[k] / E4M3_MAX, FLT_MIN);
              output_scales[tok[k] + scale_stride * scaleBlock] = s[k];
            }
          }
#pragma unroll
          for (int k = 0; k < 4; k++) s[k] = __shfl_sync(0xffffffff, s[k], 0);

#pragma unroll
          for (int k = 0; k < 4; k++) {
            float inv = 1.0f / s[k];
            __nv_fp8_e4m3 ov[4];
#pragma unroll
            for (int i = 0; i < 4; i++) ov[i] = float_to_fp8(r[k][i] * inv);
            uint32_t po;
            memcpy(&po, ov, 4);
            *reinterpret_cast<uint32_t*>(
                &output[(int64_t)tok[k] * halfDim + elemBase]) = po;
          }
        }

        for (; t + 1 < actual_bs; t += 2) {
          int tokenA = batchStart + t;
          int tokenB = batchStart + t + 1;

          int offA = t * TOKEN_BYTES + consumerWarpId * 128;
          int offB = (t + 1) * TOKEN_BYTES + consumerWarpId * 128;

          uint32_t px1A = *reinterpret_cast<uint32_t const*>(sp + offA + col);
          uint32_t px2A =
              *reinterpret_cast<uint32_t const*>(sp + offA + NC_BYTES + col);
          uint32_t px1B = *reinterpret_cast<uint32_t const*>(sp + offB + col);
          uint32_t px2B =
              *reinterpret_cast<uint32_t const*>(sp + offB + NC_BYTES + col);

          __nv_fp8_e4m3 x1A[4], x2A[4], x1B[4], x2B[4];
          memcpy(x1A, &px1A, 4);
          memcpy(x2A, &px2A, 4);
          memcpy(x1B, &px1B, 4);
          memcpy(x2B, &px2B, 4);

          int sOff2 = consumeStage * SCALE_STRIDE;
          float sc1A =
              smem_scales[sOff2 + t * N_COMPUTE * 2 + consumerWarpId * 2];
          float sc2A =
              smem_scales[sOff2 + t * N_COMPUTE * 2 + consumerWarpId * 2 + 1];
          float sc1B =
              smem_scales[sOff2 + (t + 1) * N_COMPUTE * 2 + consumerWarpId * 2];
          float sc2B = smem_scales[sOff2 + (t + 1) * N_COMPUTE * 2 +
                                   consumerWarpId * 2 + 1];

          float rA[4], rB[4];
          float mA = 0.0f, mB = 0.0f;

#pragma unroll
          for (int i = 0; i < 4; i++) {
            float f1A = sc1A * fp8_to_float(x1A[i]);
            float f2A = sc2A * fp8_to_float(x2A[i]);
            float f1B = sc1B * fp8_to_float(x1B[i]);
            float f2B = sc2B * fp8_to_float(x2B[i]);
            if constexpr (USE_TANH_SILU) {
              rA[i] = silu_tanh_f(f2A) * f1A;
              rB[i] = silu_tanh_f(f2B) * f1B;
            } else {
              rA[i] = silu_f(f2A) * f1A;
              rB[i] = silu_f(f2B) * f1B;
            }
            mA = fmaxf(mA, fabsf(rA[i]));
            mB = fmaxf(mB, fabsf(rB[i]));
          }

          warp_reduce_max_2(mA, mB);

          float sA, sB;
          if (laneId == 0) {
            sA = fmaxf(mA / E4M3_MAX, FLT_MIN);
            sB = fmaxf(mB / E4M3_MAX, FLT_MIN);
            output_scales[tokenA + scale_stride * scaleBlock] = sA;
            output_scales[tokenB + scale_stride * scaleBlock] = sB;
          }
          sA = __shfl_sync(0xffffffff, sA, 0);
          sB = __shfl_sync(0xffffffff, sB, 0);

          float invA = 1.0f / sA, invB = 1.0f / sB;
          __nv_fp8_e4m3 oA[4], oB[4];
#pragma unroll
          for (int i = 0; i < 4; i++) {
            oA[i] = float_to_fp8(rA[i] * invA);
            oB[i] = float_to_fp8(rB[i] * invB);
          }
          uint32_t poA, poB;
          memcpy(&poA, oA, 4);
          memcpy(&poB, oB, 4);
          *reinterpret_cast<uint32_t*>(
              &output[(int64_t)tokenA * halfDim + elemBase]) = poA;
          *reinterpret_cast<uint32_t*>(
              &output[(int64_t)tokenB * halfDim + elemBase]) = poB;
        }

        if (t < actual_bs) {
          int token = batchStart + t;

          int off = t * TOKEN_BYTES + consumerWarpId * 128;

          uint32_t packed_x1 =
              *reinterpret_cast<uint32_t const*>(sp + off + col);
          uint32_t packed_x2 =
              *reinterpret_cast<uint32_t const*>(sp + off + NC_BYTES + col);

          __nv_fp8_e4m3 x1_vals[4], x2_vals[4];
          memcpy(x1_vals, &packed_x1, 4);
          memcpy(x2_vals, &packed_x2, 4);

          int sOff1 = consumeStage * SCALE_STRIDE;
          float sc1 =
              smem_scales[sOff1 + t * N_COMPUTE * 2 + consumerWarpId * 2];
          float sc2 =
              smem_scales[sOff1 + t * N_COMPUTE * 2 + consumerWarpId * 2 + 1];

          float results[4];
          float localMax = 0.0f;

#pragma unroll
          for (int i = 0; i < 4; i++) {
            float f1 = sc1 * fp8_to_float(x1_vals[i]);
            float f2 = sc2 * fp8_to_float(x2_vals[i]);
            if constexpr (USE_TANH_SILU) {
              results[i] = silu_tanh_f(f2) * f1;
            } else {
              results[i] = silu_f(f2) * f1;
            }
            localMax = fmaxf(localMax, fabsf(results[i]));
          }

          float aMax = warp_reduce_max(localMax);

          float scaleOut;
          if (laneId == 0) {
            scaleOut = fmaxf(aMax / E4M3_MAX, FLT_MIN);
            output_scales[token + scale_stride * scaleBlock] = scaleOut;
          }
          scaleOut = __shfl_sync(0xffffffff, scaleOut, 0);

          float invScale = 1.0f / scaleOut;
          __nv_fp8_e4m3 out_vals[4];
#pragma unroll
          for (int i = 0; i < 4; i++) {
            out_vals[i] = float_to_fp8(results[i] * invScale);
          }
          uint32_t packed_out;
          memcpy(&packed_out, out_vals, 4);

          int64_t const outOffset = (int64_t)token * halfDim + elemBase;
          *reinterpret_cast<uint32_t*>(&output[outOffset]) = packed_out;
        }
      }

      if (laneId == 0) mbarrier_arrive(&empty_mbar[consumeStage]);
      consumeStage = (consumeStage + 1) % NUM_STAGES;
    }
  }
}

}  // namespace tma_v5
}  // namespace vllm
