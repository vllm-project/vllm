// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Skinny GEMM: activation(bf16) x weight(bf16)^T -> bf16, for decode-time
// M <= 32 with a large reduction dim. Replaces cuBLAS splitK (GEMM +
// splitKreduce) with a single block-per-output-column kernel; these shapes
// are weight-bandwidth-bound, so one coalesced pass over the weight at
// fp32 accumulation is optimal. Adapted from fp32_router_gemm.cu.
//
// First user: the DeepSeek-V32/GLM-5.2 MTP eh_proj (K=2*hidden=12288,
// N=hidden/TP), whose cuBLAS splitK pick costs ~34.6us vs the ~19us
// bandwidth floor per replicated read (and ~4us once column-parallel).

#include <cuda_bf16.h>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Load helpers (8 x bf16 = one uint4 load, converted to fp32)
// ---------------------------------------------------------------------------

namespace skinny {

constexpr int VPT = 8;  // bf16 values per thread per load

__device__ __forceinline__ void load_bf16x8(__nv_bfloat16 const* ptr,
                                            float* dst) {
  uint4 v = *reinterpret_cast<uint4 const*>(ptr);
  __nv_bfloat16 const* p = reinterpret_cast<__nv_bfloat16 const*>(&v);
#pragma unroll
  for (int i = 0; i < VPT; i++) dst[i] = __bfloat162float(p[i]);
}

// Streaming variant for the weight: each row is read exactly once across the
// whole grid, so bypass L2 residency (evict-first). Measured -1.4us at M=1.
__device__ __forceinline__ void load_bf16x8_cs(__nv_bfloat16 const* ptr,
                                               float* dst) {
  uint4 v = __ldcs(reinterpret_cast<uint4 const*>(ptr));
  __nv_bfloat16 const* p = reinterpret_cast<__nv_bfloat16 const*>(&v);
#pragma unroll
  for (int i = 0; i < VPT; i++) dst[i] = __bfloat162float(p[i]);
}

// ---------------------------------------------------------------------------
// Kernel: each block computes kNPB output columns for all kNumTokens rows.
// grid = kN / kNPB, block = kBlockSize threads. K is reduced VPT elements
// per thread per iteration; fp32 accumulation, warp butterfly + smem
// finalize, bf16 store.
// ---------------------------------------------------------------------------

template <int kBlockSize, int kNumTokens, int kNPB, int kPF, int kN, int kK>
__global__ __launch_bounds__(kBlockSize, 1) void bf16_skinny_gemm_kernel(
    __nv_bfloat16* out, __nv_bfloat16 const* mat_a,
    __nv_bfloat16 const* mat_b, int64_t out_stride) {
  constexpr int k_elems_per_iter = VPT * kBlockSize;
  constexpr int k_iterations = kK / k_elems_per_iter;
  static_assert(kK % k_elems_per_iter == 0);
  constexpr int kWarpSize = 32;
  constexpr int kNumWarps = kBlockSize / kWarpSize;

  int const n_base = blockIdx.x * kNPB;
  int const tid = threadIdx.x;
  int const warpId = tid / kWarpSize;
  int const laneId = tid % kWarpSize;

  float acc[kNumTokens][kNPB] = {};
  __shared__ float sm_reduction[kNumTokens][kNPB][kNumWarps];

  // Register prefetch (kPF > 0): W does not depend on the predecessor, so
  // the first kPF iterations' weight chunks are loaded raw BEFORE the
  // dependency sync; with a PDL-releasing producer (fused_eh_norm fires
  // gdc_launch_dependents early) these DRAM round trips overlap the norm.
  // Pair-measured on B300 (norm+gemm in one graph, full 6144x12288):
  // M=1 pf2 28.43us vs pf0 29.00us; deeper prefetch or M >= 2 regresses
  // (register pressure), hence the per-M selection in the launcher.
  uint4 w_pre[kPF > 0 ? kPF : 1][kNPB];
#pragma unroll
  for (int pf = 0; pf < kPF; pf++) {
#pragma unroll
    for (int n = 0; n < kNPB; n++) {
      w_pre[pf][n] = *reinterpret_cast<uint4 const*>(
          mat_b + (size_t)(n_base + n) * kK + pf * k_elems_per_iter +
          tid * VPT);
    }
  }

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  cudaGridDependencySynchronize();
  // Trigger right after our own wait (see fp32_router_gemm.cu): gridsync-ing
  // consumers are unaffected; independent consumers may overlap our body.
  cudaTriggerProgrammaticLaunchCompletion();
#endif

#pragma unroll
  for (int ki = 0; ki < k_iterations; ki++) {
    int const k_base = ki * k_elems_per_iter + tid * VPT;

    float b_float[kNPB][VPT];
    if (ki < kPF) {
#pragma unroll
      for (int n = 0; n < kNPB; n++) {
        __nv_bfloat16 const* p =
            reinterpret_cast<__nv_bfloat16 const*>(&w_pre[ki][n]);
#pragma unroll
        for (int v = 0; v < VPT; v++) b_float[n][v] = __bfloat162float(p[v]);
      }
    } else {
#pragma unroll
      for (int n = 0; n < kNPB; n++) {
        load_bf16x8_cs(mat_b + (size_t)(n_base + n) * kK + k_base, b_float[n]);
      }
    }

#pragma unroll
    for (int m = 0; m < kNumTokens; m++) {
      float a_float[VPT];
      load_bf16x8(mat_a + (size_t)m * kK + k_base, a_float);
#pragma unroll
      for (int n = 0; n < kNPB; n++) {
#pragma unroll
        for (int k = 0; k < VPT; k++) {
          acc[m][n] += a_float[k] * b_float[n][k];
        }
      }
    }
  }

  // Warp-level butterfly reduction
#pragma unroll
  for (int m = 0; m < kNumTokens; m++) {
#pragma unroll
    for (int n = 0; n < kNPB; n++) {
      float sum = acc[m][n];
      sum += __shfl_xor_sync(0xffffffff, sum, 16);
      sum += __shfl_xor_sync(0xffffffff, sum, 8);
      sum += __shfl_xor_sync(0xffffffff, sum, 4);
      sum += __shfl_xor_sync(0xffffffff, sum, 2);
      sum += __shfl_xor_sync(0xffffffff, sum, 1);
      if (laneId == 0) sm_reduction[m][n][warpId] = sum;
    }
  }

  __syncthreads();

  // Parallel finalize: one thread per (m, n) output.
  for (int idx = tid; idx < kNumTokens * kNPB; idx += kBlockSize) {
    int const m = idx / kNPB;
    int const n = idx % kNPB;
    float final_sum = 0.0f;
#pragma unroll
    for (int w = 0; w < kNumWarps; w++) final_sum += sm_reduction[m][n][w];
    out[(size_t)m * out_stride + n_base + n] = __float2bfloat16(final_sum);
  }
}

}  // namespace skinny

// ---------------------------------------------------------------------------
// Launcher
// ---------------------------------------------------------------------------

template <int kBlockSize, int kNPB, int kNumTokens, int kN, int kK>
void invokeBf16SkinnyGemm(__nv_bfloat16* output, __nv_bfloat16 const* mat_a,
                          __nv_bfloat16 const* mat_b, int64_t out_stride,
                          cudaStream_t stream) {
  static_assert(kN % kNPB == 0);
  // Weight prefetch depth: only M=1 measured a win (see kernel comment).
  constexpr int kPF = (kNumTokens == 1) ? 2 : 0;
  cudaLaunchConfig_t config;
  config.gridDim = kN / kNPB;
  config.blockDim = kBlockSize;
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = 1;
  config.numAttrs = 1;
  config.attrs = attrs;
  cudaLaunchKernelEx(
      &config,
      skinny::bf16_skinny_gemm_kernel<kBlockSize, kNumTokens, kNPB, kPF,
                                      kN, kK>,
      output, mat_a, mat_b, out_stride);
}

// ---------------------------------------------------------------------------
// Explicit instantiations. M = 1..32; (N, K) pairs:
//   (768, 12288)  eh_proj shard, TP8
//   (1536, 12288) eh_proj shard, TP4
//   (6144, 12288) eh_proj unsharded
// kNPB (B300 sweep, M=1): 6144 -> 2 (3072 blocks, 23.2us vs 25.3 at kNPB=8;
// narrow blocks minimize wave quantization); shards 768/1536 keep 4.
// ---------------------------------------------------------------------------

#define INSTANTIATE(M, NPB, N, K)                                      \
  template void invokeBf16SkinnyGemm<128, NPB, M, N, K>(               \
      __nv_bfloat16*, __nv_bfloat16 const*, __nv_bfloat16 const*,      \
      int64_t, cudaStream_t);

#define INSTANTIATE_ALL_M(NPB, N, K)                                   \
  INSTANTIATE(1, NPB, N, K)                                            \
  INSTANTIATE(2, NPB, N, K)                                            \
  INSTANTIATE(3, NPB, N, K)                                            \
  INSTANTIATE(4, NPB, N, K)                                            \
  INSTANTIATE(5, NPB, N, K)                                            \
  INSTANTIATE(6, NPB, N, K)                                            \
  INSTANTIATE(7, NPB, N, K)                                            \
  INSTANTIATE(8, NPB, N, K)                                            \
  INSTANTIATE(9, NPB, N, K)                                            \
  INSTANTIATE(10, NPB, N, K)                                           \
  INSTANTIATE(11, NPB, N, K)                                           \
  INSTANTIATE(12, NPB, N, K)                                           \
  INSTANTIATE(13, NPB, N, K)                                           \
  INSTANTIATE(14, NPB, N, K)                                           \
  INSTANTIATE(15, NPB, N, K)                                           \
  INSTANTIATE(16, NPB, N, K)                                           \
  INSTANTIATE(17, NPB, N, K)                                           \
  INSTANTIATE(18, NPB, N, K)                                           \
  INSTANTIATE(19, NPB, N, K)                                           \
  INSTANTIATE(20, NPB, N, K)                                           \
  INSTANTIATE(21, NPB, N, K)                                           \
  INSTANTIATE(22, NPB, N, K)                                           \
  INSTANTIATE(23, NPB, N, K)                                           \
  INSTANTIATE(24, NPB, N, K)                                           \
  INSTANTIATE(25, NPB, N, K)                                           \
  INSTANTIATE(26, NPB, N, K)                                           \
  INSTANTIATE(27, NPB, N, K)                                           \
  INSTANTIATE(28, NPB, N, K)                                           \
  INSTANTIATE(29, NPB, N, K)                                           \
  INSTANTIATE(30, NPB, N, K)                                           \
  INSTANTIATE(31, NPB, N, K)                                           \
  INSTANTIATE(32, NPB, N, K)

INSTANTIATE_ALL_M(4, 768, 12288)
INSTANTIATE_ALL_M(4, 1536, 12288)
INSTANTIATE_ALL_M(2, 6144, 12288)
// LL-mode (M<=8 wiring guard) backbone shapes, B300 sweep vs cuBLAS:
//   q_b_proj  (2048, 2048): 1.67x/1.29x/1.15x at M=4/6/8 (NPB=4 within
//     0.1us of per-M best)
//   shared-expert gate_up (512, 6144): 1.95x/1.58x/1.40x at M=4/6/8
// cuBLAS keeps qkv_a (2624,6144) and o_proj (6144,2048) — already at
// 3.4-3.8 TB/s there; the GEMV loses on activation re-reads.
INSTANTIATE_ALL_M(4, 2048, 2048)
INSTANTIATE_ALL_M(4, 512, 6144)
// fused_qkv_a (2624, 6144), 32MB: skinny wins ONLY at M<=2 (B300: M=1
// 6.99us vs cuBLAS 9.21 = 1.32x, M=2 1.24x; M>=4 cuBLAS holds at 3.5TB/s
// and every alternative loses — cublasLt top-8 3.6TB/s wall, DeepGEMM
// 0.71x, wmma+cp.async custom 0.30x pending a TMA rewrite).
INSTANTIATE_ALL_M(4, 2624, 6144)
// DSv3.2 (TP8) siblings of the GLM shapes above, same dual-chip matrix:
//   fused_qkv_a (2112, 7168), 30MB: wins M<=2 (M=1 1.30-1.34x)
//   MTP eh_proj (7168, 14336), 205MB: wins M<=2 (M=1 1.12-1.16x)
INSTANTIATE_ALL_M(4, 2112, 7168)
INSTANTIATE_ALL_M(2, 7168, 14336)

#undef INSTANTIATE_ALL_M
#undef INSTANTIATE
