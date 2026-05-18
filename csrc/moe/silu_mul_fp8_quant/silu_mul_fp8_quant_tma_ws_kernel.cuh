#pragma once

// V5 kernel: CTA-wide partial-row TMA pipeline.
// Thread 0 loads NC_SLICE bytes (gate) + NC_SLICE bytes (up) per token via
// cp.async.bulk into CTA-shared smem. All warps read their slice after
// mbarrier_wait. __syncthreads guards stage reuse.
// Scales hardcoded to 1.0f for perf baseline (TODO: add async scale loading).

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cstdint>
#include <type_traits>

namespace vllm {
namespace tma_v5 {

static constexpr int SCALE_BLOCK_SIZE = 128;
static constexpr float E4M3_MAX = 448.0f;

__device__ __forceinline__ float fp8_to_float(__nv_fp8_e4m3 v) {
  return __half2float(__nv_cvt_fp8_to_halfraw(v.__x, __NV_E4M3));
}

__device__ __forceinline__ __nv_fp8_e4m3 float_to_fp8(float v) {
  __nv_fp8_e4m3 r;
  r.__x = __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
  return r;
}

__device__ __forceinline__ float silu_f(float x) {
  return x / (1.0f + __expf(-x));
}

__device__ __forceinline__ float silu_tanh_f(float x) {
  return x * (0.5f + 0.5f * __tanhf(x * 0.5f));
}

__device__ __forceinline__ float warp_reduce_max(float val) {
  val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 16));
  val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 8));
  val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 4));
  val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 2));
  val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 1));
  return val;
}

__device__ __forceinline__ void warp_reduce_max_2(float& a, float& b) {
#pragma unroll
  for (int mask = 16; mask >= 1; mask >>= 1) {
    float tmpA = __shfl_xor_sync(0xffffffff, a, mask);
    float tmpB = __shfl_xor_sync(0xffffffff, b, mask);
    a = fmaxf(a, tmpA);
    b = fmaxf(b, tmpB);
  }
}

__device__ __forceinline__ void warp_reduce_max_4(float& a, float& b, float& c,
                                                  float& d) {
#pragma unroll
  for (int mask = 16; mask >= 1; mask >>= 1) {
    float tA = __shfl_xor_sync(0xffffffff, a, mask);
    float tB = __shfl_xor_sync(0xffffffff, b, mask);
    float tC = __shfl_xor_sync(0xffffffff, c, mask);
    float tD = __shfl_xor_sync(0xffffffff, d, mask);
    a = fmaxf(a, tA);
    b = fmaxf(b, tB);
    c = fmaxf(c, tC);
    d = fmaxf(d, tD);
  }
}

// --- mbarrier PTX helpers (sm_90+) ---

__device__ __forceinline__ void mbarrier_init(uint64_t* mbar, uint32_t count) {
  uint32_t smem = static_cast<uint32_t>(
      __cvta_generic_to_shared(reinterpret_cast<void*>(mbar)));
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(smem),
               "r"(count));
}

__device__ __forceinline__ void fence_async_shared() {
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void fence_barrier_init() {
  asm volatile("fence.mbarrier_init.release.cluster;\n" ::);
}

__device__ __forceinline__ void mbarrier_inval(uint64_t* mbar) {
  uint32_t smem = static_cast<uint32_t>(
      __cvta_generic_to_shared(reinterpret_cast<void*>(mbar)));
  asm volatile("mbarrier.inval.shared::cta.b64 [%0];\n" ::"r"(smem));
}

__device__ __forceinline__ void mbarrier_arrive(uint64_t* mbar) {
  uint32_t smem = static_cast<uint32_t>(
      __cvta_generic_to_shared(reinterpret_cast<void*>(mbar)));
  asm volatile(
      "{\n"
      ".reg .b64 state;\n"
      "mbarrier.arrive.shared::cta.b64 state, [%0];\n"
      "}\n" ::"r"(smem));
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t* mbar,
                                                          uint32_t tx_bytes) {
  uint32_t smem = static_cast<uint32_t>(
      __cvta_generic_to_shared(reinterpret_cast<void*>(mbar)));
  asm volatile(
      "{\n"
      ".reg .b64 state;\n"
      "mbarrier.arrive.expect_tx.shared::cta.b64 state, [%0], %1;\n"
      "}\n" ::"r"(smem),
      "r"(tx_bytes));
}

__device__ __forceinline__ void mbarrier_wait(uint64_t* mbar,
                                              uint32_t phase_parity) {
  uint32_t smem = static_cast<uint32_t>(
      __cvta_generic_to_shared(reinterpret_cast<void*>(mbar)));
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "LAB_WAIT:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1, %2;\n"
      "@p bra DONE;\n"
      "bra LAB_WAIT;\n"
      "DONE:\n"
      "}\n" ::"r"(smem),
      "r"(phase_parity), "r"(0x989680));
}

__device__ __forceinline__ void bulk_copy_g2s(void* smem_dst,
                                              const void* gmem_src,
                                              uint32_t bytes, uint64_t* mbar) {
  uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
  uint32_t mbar_smem = static_cast<uint32_t>(
      __cvta_generic_to_shared(reinterpret_cast<void*>(mbar)));
  asm volatile(
      "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
      " [%0], [%1], %2, [%3];\n" ::"r"(dst),
      "l"(gmem_src), "r"(bytes), "r"(mbar_smem)
      : "memory");
}

__device__ __forceinline__ void cp_async_f32(float* smem_dst,
                                             const float* gmem_src) {
  uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(dst),
               "l"(gmem_src)
               : "memory");
}

__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::: "memory");
}

__device__ __forceinline__ void cp_async_wait_all() {
  asm volatile("cp.async.wait_group 0;\n" ::: "memory");
}

__device__ __forceinline__ void tma_load_3d(void* smem_dst,
                                            const CUtensorMap& desc, int32_t d0,
                                            int32_t d1, int32_t d2,
                                            uint64_t* mbar) {
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
  uint32_t mbar_smem = static_cast<uint32_t>(
      __cvta_generic_to_shared(reinterpret_cast<void*>(mbar)));
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cta.global.tile"
      ".mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3, %4}], [%5];\n" ::"r"(smem),
      "l"(&desc), "r"(d0), "r"(d1), "r"(d2), "r"(mbar_smem)
      : "memory");
}

// Warp-specialized FP8 kernel: 1 producer warp + N_COMPUTE consumer warps.
// Each consumer warp handles 2 scale blocks with interleaved reduces for ILP.
// Producer continuously fills TMA pipeline via full/empty mbarrier pairs.
// No __syncthreads in main loop. Scales hardcoded to 1.0f for perf baseline.
template <int N_COMPUTE = 7, int NUM_STAGES = 8, int BATCH_SIZE = 2,
          bool USE_TANH_SILU = false>
__global__ void __launch_bounds__((N_COMPUTE + 1) * 32)
    silu_mul_fp8_quant_tma_ws_kernel(const __nv_fp8_e4m3* __restrict__ input,
                                     const float* __restrict__ input_scales,
                                     __nv_fp8_e4m3* __restrict__ output,
                                     float* __restrict__ output_scales,
                                     int32_t n_tokens, int64_t H,
                                     int64_t scale_stride) {
  static constexpr int ELTS_PER_THREAD = 4;
  static constexpr int NC_SLICE = N_COMPUTE * SCALE_BLOCK_SIZE;
  static constexpr int ROW_BYTES = 2 * NC_SLICE;
  static constexpr int STAGE_BYTES = BATCH_SIZE * ROW_BYTES;
  static constexpr int MBAR_REGION = ((2 * NUM_STAGES * 8) + 127) & ~127;

  int const warpId = threadIdx.x / 32;
  int const laneId = threadIdx.x % 32;
  bool const isProducer = (warpId == 0);

  int const totalN = n_tokens;
  int const halfDim = static_cast<int>(H);
  int const G = halfDim / SCALE_BLOCK_SIZE;
  int const twoH = 2 * halfDim;

  extern __shared__ char smem_raw[];
  uint64_t* full_mbar = reinterpret_cast<uint64_t*>(smem_raw);
  uint64_t* empty_mbar = &full_mbar[NUM_STAGES];

  // --- Init mbarriers (thread 0 only) ---
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

  int const gateGmemOff = static_cast<int>(blockIdx.x) * NC_SLICE;
  int const upGmemOff = halfDim + gateGmemOff;

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
        uint32_t load_bytes = static_cast<uint32_t>(actual_load * ROW_BYTES);
        mbarrier_arrive_expect_tx(&full_mbar[fillStage], load_bytes);
        char* dst = stage_ptr(fillStage);
        for (int t = 0; t < actual_load; t++) {
          int64_t row_base = (int64_t)(nextBatchStart + t) * twoH;
          bulk_copy_g2s(dst + t * ROW_BYTES, &input[row_base + gateGmemOff],
                        static_cast<uint32_t>(NC_SLICE), &full_mbar[fillStage]);
          bulk_copy_g2s(dst + t * ROW_BYTES + NC_SLICE,
                        &input[row_base + upGmemOff],
                        static_cast<uint32_t>(NC_SLICE), &full_mbar[fillStage]);
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
    int const scaleBlock = blockIdx.x * N_COMPUTE + consumerWarpId;
    bool const valid = scaleBlock < G;
    int const elemBase =
        valid ? (scaleBlock * SCALE_BLOCK_SIZE + laneId * ELTS_PER_THREAD) : 0;
    int const warpSliceOff = consumerWarpId * SCALE_BLOCK_SIZE;

    int consumeStage = 0;
    int phase_full[NUM_STAGES] = {};

    for (int batch = 0; batch < totalBatches; batch++) {
      int batchStart = firstBatch + batch * batchStride;

      mbarrier_wait(&full_mbar[consumeStage], phase_full[consumeStage]);
      phase_full[consumeStage] ^= 1;

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
            px1[k] = *reinterpret_cast<uint32_t const*>(
                sp + (t + k) * ROW_BYTES + warpSliceOff + laneId * 4);
            px2[k] = *reinterpret_cast<uint32_t const*>(
                sp + (t + k) * ROW_BYTES + NC_SLICE + warpSliceOff +
                laneId * 4);
          }

          float sc1[4], sc2[4];
          if (laneId == 0) {
#pragma unroll
            for (int k = 0; k < 4; k++) {
              sc1[k] = input_scales[tok[k] + scale_stride * scaleBlock];
              sc2[k] = input_scales[tok[k] + scale_stride * (scaleBlock + G)];
            }
          }
#pragma unroll
          for (int k = 0; k < 4; k++) {
            sc1[k] = __shfl_sync(0xffffffff, sc1[k], 0);
            sc2[k] = __shfl_sync(0xffffffff, sc2[k], 0);
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

          uint32_t px1A = *reinterpret_cast<uint32_t const*>(
              sp + t * ROW_BYTES + warpSliceOff + laneId * 4);
          uint32_t px2A = *reinterpret_cast<uint32_t const*>(
              sp + t * ROW_BYTES + NC_SLICE + warpSliceOff + laneId * 4);
          uint32_t px1B = *reinterpret_cast<uint32_t const*>(
              sp + (t + 1) * ROW_BYTES + warpSliceOff + laneId * 4);
          uint32_t px2B = *reinterpret_cast<uint32_t const*>(
              sp + (t + 1) * ROW_BYTES + NC_SLICE + warpSliceOff + laneId * 4);

          __nv_fp8_e4m3 x1A[4], x2A[4], x1B[4], x2B[4];
          memcpy(x1A, &px1A, 4);
          memcpy(x2A, &px2A, 4);
          memcpy(x1B, &px1B, 4);
          memcpy(x2B, &px2B, 4);

          float sc1A, sc2A, sc1B, sc2B;
          if (laneId == 0) {
            sc1A = input_scales[tokenA + scale_stride * scaleBlock];
            sc2A = input_scales[tokenA + scale_stride * (scaleBlock + G)];
            sc1B = input_scales[tokenB + scale_stride * scaleBlock];
            sc2B = input_scales[tokenB + scale_stride * (scaleBlock + G)];
          }
          sc1A = __shfl_sync(0xffffffff, sc1A, 0);
          sc2A = __shfl_sync(0xffffffff, sc2A, 0);
          sc1B = __shfl_sync(0xffffffff, sc1B, 0);
          sc2B = __shfl_sync(0xffffffff, sc2B, 0);

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

          uint32_t packed_x1 = *reinterpret_cast<uint32_t const*>(
              sp + t * ROW_BYTES + warpSliceOff + laneId * 4);
          uint32_t packed_x2 = *reinterpret_cast<uint32_t const*>(
              sp + t * ROW_BYTES + NC_SLICE + warpSliceOff + laneId * 4);

          __nv_fp8_e4m3 x1_vals[4], x2_vals[4];
          memcpy(x1_vals, &packed_x1, 4);
          memcpy(x2_vals, &packed_x2, 4);

          float sc1, sc2;
          if (laneId == 0) {
            sc1 = input_scales[token + scale_stride * scaleBlock];
            sc2 = input_scales[token + scale_stride * (scaleBlock + G)];
          }
          sc1 = __shfl_sync(0xffffffff, sc1, 0);
          sc2 = __shfl_sync(0xffffffff, sc2, 0);

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

// BF16 kernel: cooperative LDG (no TMA, no scales) — unchanged from original v5
template <int N_COMPUTE = 4, int UNUSED_STAGES = 4, int UNUSED_BS = 8,
          bool USE_TANH_SILU = false>
__global__ void __launch_bounds__(N_COMPUTE * 32)
    silu_mul_fp8_quant_tma_ws_kernel_bf16(
        const __nv_bfloat16* __restrict__ input,
        __nv_fp8_e4m3* __restrict__ output, float* __restrict__ output_scales,
        int32_t n_tokens, int64_t H) {
  static constexpr int ELTS_PER_THREAD = 4;

  int const totalN = n_tokens;
  int const halfDim = static_cast<int>(H);
  int const G = halfDim / SCALE_BLOCK_SIZE;

  int const warpId = threadIdx.x / 32;
  int const laneId = threadIdx.x % 32;
  int const scaleBlock = blockIdx.x * N_COMPUTE + warpId;

  if (scaleBlock >= G) return;

  int const elemBase = scaleBlock * SCALE_BLOCK_SIZE + laneId * ELTS_PER_THREAD;
  int const gridY = static_cast<int>(gridDim.y);

  for (int token = static_cast<int>(blockIdx.y); token < totalN;
       token += gridY) {
    float results[ELTS_PER_THREAD];
    float localMax = 0.0f;

    int64_t const x1Offset = (int64_t)token * (2 * halfDim) + elemBase;

    __nv_bfloat162 const* x1_bf2 =
        reinterpret_cast<__nv_bfloat162 const*>(&input[x1Offset]);
    __nv_bfloat162 const* x2_bf2 =
        reinterpret_cast<__nv_bfloat162 const*>(&input[x1Offset + halfDim]);

#pragma unroll
    for (int k = 0; k < 2; k++) {
      float2 f1 = __bfloat1622float2(x1_bf2[k]);
      float2 f2 = __bfloat1622float2(x2_bf2[k]);
      if constexpr (USE_TANH_SILU) {
        results[2 * k] = silu_tanh_f(f2.x) * f1.x;
        results[2 * k + 1] = silu_tanh_f(f2.y) * f1.y;
      } else {
        results[2 * k] = silu_f(f2.x) * f1.x;
        results[2 * k + 1] = silu_f(f2.y) * f1.y;
      }
      localMax = fmaxf(localMax,
                       fmaxf(fabsf(results[2 * k]), fabsf(results[2 * k + 1])));
    }

    float aMax = warp_reduce_max(localMax);

    float scaleOut;
    if (laneId == 0) {
      scaleOut = fmaxf(aMax / E4M3_MAX, FLT_MIN);
      output_scales[token * G + scaleBlock] = scaleOut;
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

}  // namespace tma_v5
}  // namespace vllm
