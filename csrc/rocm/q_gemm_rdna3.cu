// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// W4A16 GPTQ kernel for RDNA3 (gfx1100 / RX 7900 XTX class), templated on the
// activation dtype (half or __hip_bfloat16). Adapted from exllamav2's 4-bit
// kernel (csrc/quantization/gptq/q_gemm.cu) with the following changes:
//
//   1. Deterministic split-K epilogue. K is split across gridDim.z blocks;
//      each block stores its FP32 block partial to a scratch tensor, and the
//      last z block of each (x, y) tile reduces the z-slices in fixed
//      ascending order with a single final cast to T — the reduction runs
//      in the GEMM kernel itself, gated by a per-call ticket counter (no
//      cross-launch state; see the epilogue). When gridDim.z == 1 each
//      output element has exactly one writer and the kernel stores the
//      rounded accumulator directly — no scratch, no reduce, no atomics.
//      (Legacy design, pre-#54706: a packed CAS-loop atomic add on a 64-bit
//      word emulating v_global_atomic_pk_add_{f16,bf16}, which gfx11 lacks.
//      It narrowed every split partial to bf16/fp16 BEFORE accumulation, so
//      the execution-dependent CAS completion order changed the rounded
//      result for identical inputs. The CAS epilogue is kept below the
//      partials branch for A/B comparison only; it requires a pre-zeroed
//      output tensor.)
//
//   2. The bf16 path uses a dedicated bit-trick that avoids the fp16-only
//      "upper nibble * 16" trick, which would overflow the 7-bit bf16
//      mantissa. See qdq_4_rdna3.cuh for details. The fp16 path uses the
//      exact factored dequant (also in qdq_4_rdna3.cuh): magic values in
//      fp16, the (y, z) bias correction in fp32.
//
//   3. Wave32 geometry sized for high CU saturation: THREADS_X=256
//      (8 waves per block) and BLOCK_KN_SIZE=256, with each thread
//      computing 4 N output columns. gridDim.z = K / BLOCK_KN_SIZE
//      splits K; the epilogue stores FP32 partials for the fixed-order
//      reduce (see item 1). fp16 uses
//      v_dot2_f32_f16 (__builtin_amdgcn_fdot2) for the inner dot;
//      bf16 widens to fp32 (no v_pk_fma_bf16 on gfx11) and accumulates
//      with v_fma_f32. M_COUNT ∈ {1,2,4,8} is selected at launch
//      based on size_m.
//
//   4. The dispatch with M >= 12 forwards to the WMMA kernel in
//      q_gemm_rdna3_wmma.cu (separate translation unit) where
//      v_wmma_f32_16x16x16_bf16_w32 wins. Below that the scalar path wins
//      and is the more accurate of the two.

#include <cstdint>
#include <cstdio>

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>

#include "qdq_4_rdna3.cuh"

#if defined(__HIPCC__) && defined(__gfx1100__)
  #define __HIP__RDNA3__
#endif

namespace vllm {
namespace gptq_rdna3 {

// BLOCK_KN_SIZE = 256 (was 128 in exllama). Each block covers 256 K
// elements and THREADS_X*4 = 1024 N columns. For Qwen-class K=4096 this
// halves gridDim.z (32 → 16) and therefore halves the atomic count per
// output position vs the exllama default. THREADS_X=256 = 8 waves on RDNA3
// wave32; with ~32 wave slots per CU we still fit 4 blocks per CU at peak.
//
// We tried BLOCK_KN_SIZE=512 (microbench on Qwen3.6-27B): bf16 improved
// 5-10% at large M (atomic CAS halved), but fp16 decode regressed up to
// +40% on qkv-square (32 → 45 μs at M=1). Cause: 16 waves/block × 16
// total blocks for [M=1, K=N=4096] only saturates ~8 of the 96 CUs,
// breaking memory-latency hiding for the fp16 path which is already
// memory-bound. Reverted to 256; bf16 keeps most of its gains from the
// fp32 dequant rewrite alone.
#define BLOCK_KN_SIZE 256
#define THREADS_X 256

// Device code below is RDNA3-only; non-RDNA3 device passes fall through to
// the empty __global__ stub at the #else below for symbol parity.
#if defined(__HIP__RDNA3__) || !defined(__HIP_DEVICE_COMPILE__)

// ---------------------------------------------------------------------------
// Per-dtype helpers. We avoid heavy template metaprogramming and just provide
// overloaded inline functions; the kernel below selects via `if constexpr`.
// ---------------------------------------------------------------------------

// Type-generic zero — both half and bf16_t in HIP/ROCm have a converting
// constructor from float, but going through __float2half_rn / __float2bfloat16
// is the unambiguously correct path on every ROCm version.
template <typename T>
__forceinline__ __device__ T tzero();

template <>
__forceinline__ __device__ half tzero<half>() {
  return __float2half_rn(0.0f);
}

template <>
__forceinline__ __device__ bf16_t tzero<bf16_t>() {
  return __float2bfloat16(0.0f);
}

// ---------------------------------------------------------------------------
// Packed atomic-add via CAS-loop on a 64-bit word (4 fp16/bf16 lanes per CAS).
// RDNA3 (gfx11) does NOT have native v_global_atomic_pk_add_f16 / _bf16 (those
// landed on gfx940 / gfx1250 respectively), so this lowers to
// global_atomic_cmpswap_b64 plus retry. This is the LEGACY (pre-#54706)
// epilogue: the caller must pre-zero the output, and every split block
// atomically adds its low-precision partial into it. Because the addition
// happens in bf16/fp16 AFTER narrowing, the result depends on CAS completion
// order — the nondeterminism PR #54706 fixes. Kept for A/B comparison with
// the deterministic path; the shipped dispatch never selects it.
//
// 64-bit alignment: the kernel writes at `out + n` where n = offset_n + t*4
// (always multiple of 4), and partition_weight_shape[1] is required to be a
// multiple of 8 by can_implement(), so every (m, n) write target is 8-byte
// aligned. Required by global_atomic_cmpswap_b64.
// ---------------------------------------------------------------------------

__forceinline__ __device__ void atomic_add_pk4_f16(half* addr, half2 v01,
                                                   half2 v23) {
  unsigned long long* addr_u = reinterpret_cast<unsigned long long*>(addr);
  unsigned long long old = *addr_u;
  while (true) {
    union {
      unsigned long long u;
      half2 h2[2];
    } cur, sum;
    cur.u = old;
    sum.h2[0] = __hadd2(cur.h2[0], v01);
    sum.h2[1] = __hadd2(cur.h2[1], v23);
    unsigned long long prev = atomicCAS(addr_u, old, sum.u);
    if (prev == old) break;
    old = prev;
  }
}

__forceinline__ __device__ void atomic_add_pk4_bf16(bf16_t* addr, bf162_t v01,
                                                    bf162_t v23) {
  unsigned long long* addr_u = reinterpret_cast<unsigned long long*>(addr);
  unsigned long long old = *addr_u;
  while (true) {
    union {
      unsigned long long u;
      bf162_t b2[2];
    } cur, sum;
    cur.u = old;
    sum.b2[0] = __hadd2(cur.b2[0], v01);
    sum.b2[1] = __hadd2(cur.b2[1], v23);
    unsigned long long prev = atomicCAS(addr_u, old, sum.u);
    if (prev == old) break;
    old = prev;
  }
}

// Load one row's worth of 4 packed zeros (column n..n+3) from a [groups, N/8]
// uint32 tensor. n is a multiple of 4 by construction (n = offset_n + t*4 with
// offset_n = blockIdx.x * 512), so the 4 nibbles always live within one or two
// uint32 words; in practice within one because n & 7 is 0 or 4.
__forceinline__ __device__ void load4_zeros(const uint32_t* qzeros_row, int n,
                                            int (&zeros)[4]) {
  int qcol = n / 8;
  int shift = (n & 0x07) * 4;
  uint32_t d = qzeros_row[qcol] >> shift;
  zeros[0] = (int)(d & 0xF);
  zeros[1] = (int)((d >> 4) & 0xF);
  zeros[2] = (int)((d >> 8) & 0xF);
  zeros[3] = (int)((d >> 12) & 0xF);
}

template <typename T>
__forceinline__ __device__ void load4_scales(const T* scales_row, int n,
                                             T (&scales)[4]) {
  scales[0] = scales_row[n + 0];
  scales[1] = scales_row[n + 1];
  scales[2] = scales_row[n + 2];
  scales[3] = scales_row[n + 3];
}

// ---------------------------------------------------------------------------
// Main kernel.
// ---------------------------------------------------------------------------

template <typename T, int M_COUNT>
__global__ void gemm_q4_kernel_rdna3(
    const T* __restrict__ a, const uint32_t* __restrict__ b_q_weight,
    const uint32_t* __restrict__ b_qzeros, const T* __restrict__ b_scales,
    T* __restrict__ c, const int size_m, const int size_n, const int size_k,
    const int groups, const int zero_offset,
    // Deterministic split-K epilogue: when
    // non-null, each split block stores its
    // FP32 partial to
    // partials[(z*size_m + m)*size_n + n]
    // instead of CAS-atomically accumulating
    // a low-precision partial into c. The
    // last z block of each (x, y) tile then
    // performs the fixed ascending-z
    // reduction in-kernel (see the epilogue
    // below). `tickets` (same length as the
    // number of tiles) is a per-call
    // zero-filled arrival counter array —
    // one ticket per tile — so concurrent
    // invocations never share state.
    float* __restrict__ partials, int* __restrict__ tickets) {
  const int t = threadIdx.x;
  const int offset_n = blockIdx.x * BLOCK_KN_SIZE * 4;
  const int offset_m = blockIdx.y * M_COUNT;
  const int offset_k = blockIdx.z * BLOCK_KN_SIZE;
  const int end_k = min(offset_k + BLOCK_KN_SIZE, size_k);
  const int n = offset_n + t * 4;

  // LDS layout: [M_COUNT][BLOCK_KN_SIZE + LDS_PAD]. The PAD=8 elements per M
  // row break the natural 256-element/512-byte alignment that would otherwise
  // collide on the same LDS bank when a thread reads block_a[0..M_COUNT-1][k]
  // (same k, different m). Row stride becomes 264 elements * 2B = 528B = 132
  // 4-byte banks, so m-stride hits banks (m*132)%32 = (m*4)%32 — distinct for
  // all M_COUNT ≤ 8. Cost: 16B LDS per block, irrelevant.
  constexpr int LDS_PAD = 8;
  __shared__ T block_a[M_COUNT][BLOCK_KN_SIZE + LDS_PAD];

  // Stage A: each thread loads 1 K element per M row into LDS.
  // THREADS_X == BLOCK_KN_SIZE so this is a 1:1 map.
  // For M_COUNT > 1 with size_m not a multiple of M_COUNT, slots past size_m
  // are zero-padded so the dot product contribution is 0 (we then skip the
  // atomic write for those rows below).
  //
  // M=1 fast path: skip LDS staging + __syncthreads entirely. All 256 threads
  // read the SAME 8-element A window per inner step (a_off is uniform across
  // the block), so the cache-line broadcast through L1 makes global reads as
  // cheap as LDS reads. Measured: ~1% on 4B b=1, ~6% on 27B b=1 in=128.
  static_assert(BLOCK_KN_SIZE == THREADS_X,
                "BLOCK_KN_SIZE must equal THREADS_X (1 K element per thread)");
  // The M=1 fast path (skip LDS) only has a global-read code path for bf16
  // (the v_dot2_f32_bf16 branch).  The fp16 inner loop still indexes
  // block_a[m][a_off] unconditionally, so for fp16 we MUST stage A through
  // LDS even at M=1 to avoid reading uninitialized shared memory.
  constexpr bool USE_LDS_A = (M_COUNT > 1) || std::is_same<T, half>::value;
  if constexpr (USE_LDS_A) {
    if (offset_k + t < end_k) {
  #pragma unroll
      for (int m = 0; m < M_COUNT; ++m) {
        T av;
        if (offset_m + m < size_m) {
          const T* a_row = a + (offset_m + m) * size_k;
          av = a_row[offset_k + t];
        } else {
          av = tzero<T>();  // zero-pad invalid M rows
        }
        block_a[m][t] = av;
      }
    }

    // Threads beyond the right edge of N have nothing to do. Note: we must NOT
    // return before __syncthreads() if any thread in the block participates in
    // the LDS load above — but here all THREADS_X (=256) threads always do,
    // regardless of whether their `n` is in bounds.
    __syncthreads();
  }
  if (n >= size_n) return;

  // Group bookkeeping. We require size_k % groups == 0 (groupsize divides K).
  const int groupsize = size_k / groups;
  int group = offset_k / groupsize;
  int nextgroup = (group + 1) * groupsize;

  // qweight stride: weights are [K/8, N] uint32 with K packed at dim 0.
  int qk = offset_k / 8;
  const uint32_t* b_ptr = b_q_weight + qk * size_n + n;

  // Per-column dequant constants, fp32 for both dtypes. fp16 needs two (y, z)
  // pairs because the upper-nibble trick carries a factor of 16; bf16 needs one
  // because its dequant produces fp32 directly — see prep_zero_scale_bf16_f32 /
  // the FMA bypass for the missing v_pk_fma_bf16 on gfx11. Neither narrows the
  // bias to the activation dtype: see the exact factored form in
  // qdq_4_rdna3.cuh for why that matters.
  float z_b_f[4], y_b_f[4];
  float yf_h[4][2], zf_h[4][2];

  // bf16-only group refresh (fp16 preps from prefetched raw words below).
  auto refresh_group = [&](int g) {
    if constexpr (!std::is_same<T, half>::value) {
      const uint32_t* qz_row = b_qzeros + g * (size_n / 8);
      const T* sc_row = b_scales + g * size_n;
      int zeros[4];
      T scales[4];
      load4_zeros(qz_row, n, zeros);
      load4_scales<T>(sc_row, n, scales);
  #pragma unroll
      for (int i = 0; i < 4; ++i) {
        prep_zero_scale_bf16_f32((uint32_t)(zeros[i] + zero_offset), scales[i],
                                 z_b_f[i], y_b_f[i]);
      }
    }
  };

  // fp16: the raw zero/scale words of the NEXT group are loaded during the
  // round before the boundary, so a group change no longer stalls on a full
  // memory round trip before the weights of the round are even issued.
  uint32_t rz = 0;
  T rs[4];
  auto load_raw_group = [&](int g) {
    rz = b_qzeros[g * (size_n / 8) + n / 8] >> ((n & 7) * 4);
  #pragma unroll
    for (int i = 0; i < 4; ++i) rs[i] = b_scales[g * size_n + n + i];
  };
  auto prep_raw_fp16 = [&]() {
    if constexpr (std::is_same<T, half>::value) {
  #pragma unroll
      for (int i = 0; i < 4; ++i) {
        prep_zero_scale_fp16_f32(((rz >> (4 * i)) & 0xF) + zero_offset, rs[i],
                                 yf_h[i], zf_h[i]);
      }
    }
  };

  if constexpr (std::is_same<T, half>::value) {
    load_raw_group(group);
    prep_raw_fp16();
  } else {
    refresh_group(group);
  }

  float block_c[M_COUNT][4];
  #pragma unroll
  for (int m = 0; m < M_COUNT; ++m) {
  #pragma unroll
    for (int j = 0; j < 4; ++j) block_c[m][j] = 0.0f;
  }

  // Note on group-transition granularity: we check `k == nextgroup` at the
  // start of each outer iteration (which advances K by 32), so a group
  // boundary must never fall strictly inside a 32-K round: groupsize must
  // be a multiple of 32 (all real GPTQ group sizes 32/64/128/256, or
  // groups == 1). groupsize in {16, 8, ...} or non-multiples like 48 would
  // skip the boundary check and silently use stale constants; the host-side
  // TORCH_CHECK below enforces it. Same assumption as exllama.
  //
  // Software pipelining: we issue all 4 vectorized weight loads up front
  // before any dequant/FMA depends on them. This gives the AMDGPU backend
  // freedom to schedule the global_loads early and overlap their latency
  // with dequant + v_pk_fma_f16 of earlier iterations. Cost: 4×int4 = 16
  // VGPRs in flight per thread, plenty of headroom on RDNA3.
  int k = offset_k;
  while (k < end_k) {
    if (k == nextgroup) {
      group++;
      nextgroup += groupsize;
      if constexpr (std::is_same<T, half>::value) {
        prep_raw_fp16();  // from the words prefetched in the previous round
      } else {
        refresh_group(group);
      }
    }

    // Prefetch all four j-iterations' weight words. The compiler emits 4
    // global_load_b128 instructions back-to-back; the dependent dequant +
    // FMA work below hides their latency.
    int4 b_w[4];
  #pragma unroll
    for (int j = 0; j < 4; ++j) {
      b_w[j] = *(const int4*)(b_ptr + j * size_n);
    }
    b_ptr += 4 * size_n;
    if constexpr (std::is_same<T, half>::value) {
      if (k + 32 == nextgroup && k + 32 < end_k) load_raw_group(group + 1);
    }

    // fp16 exact factored form: raw dot products and activation sums, both
    // split by nibble slot, accumulated across the whole 32-K round. Neither
    // depends on the output column, and the (y, z) constants are group
    // constants, so both corrections are applied once after the j loop
    // instead of four times inside it — see qdq_4_rdna3.cuh.
    float sum_lo[M_COUNT] = {}, sum_hi[M_COUNT] = {};
    float acc_lo[M_COUNT][4] = {}, acc_hi[M_COUNT][4] = {};

  #pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int a_off = (k - offset_k) + 8 * j;

      if constexpr (std::is_same<T, half>::value) {
        // Magic values with no bias folded in; the (y, z) correction is
        // applied in fp32 below — see the exact factored form in
        // qdq_4_rdna3.cuh.
        half2 qm[4][4];
        magic_4bit_8_fp16((uint32_t)b_w[j].x, qm[0]);
        magic_4bit_8_fp16((uint32_t)b_w[j].y, qm[1]);
        magic_4bit_8_fp16((uint32_t)b_w[j].z, qm[2]);
        magic_4bit_8_fp16((uint32_t)b_w[j].w, qm[3]);

        const half2 ones = ones_half2_fp16();

  #pragma unroll
        for (int m = 0; m < M_COUNT; ++m) {
          const half2* a2 = reinterpret_cast<const half2*>(&block_a[m][a_off]);
          // sum_a split by nibble slot (the high pairs carry a factor of
          // 16). It does not depend on the output column, so one v_dot2 per
          // pair is shared by the four columns this thread owns.
          sum_lo[m] = __builtin_amdgcn_fdot2(a2[0], ones, sum_lo[m], false);
          sum_hi[m] = __builtin_amdgcn_fdot2(a2[1], ones, sum_hi[m], false);
          sum_lo[m] = __builtin_amdgcn_fdot2(a2[2], ones, sum_lo[m], false);
          sum_hi[m] = __builtin_amdgcn_fdot2(a2[3], ones, sum_hi[m], false);
  #pragma unroll
          for (int c = 0; c < 4; ++c) {
            acc_lo[m][c] =
                __builtin_amdgcn_fdot2(qm[c][0], a2[0], acc_lo[m][c], false);
            acc_hi[m][c] =
                __builtin_amdgcn_fdot2(qm[c][1], a2[1], acc_hi[m][c], false);
            acc_lo[m][c] =
                __builtin_amdgcn_fdot2(qm[c][2], a2[2], acc_lo[m][c], false);
            acc_hi[m][c] =
                __builtin_amdgcn_fdot2(qm[c][3], a2[3], acc_hi[m][c], false);
          }
        }
      } else if constexpr (M_COUNT == 1) {
        // bf16 decode (M=1), v_dot2_f32_bf16 path. Mirrors the data-flow of
        // Hybrid PR #40977's wvSplitK_int4 kernel exactly so clang's
        // InstCombine cannot fold the bf16→fp32 widening (LLVM #76000):
        //   * activations and magic-value weights share a fp32-aliased
        //     union (bytes written as uint32, read as bf16x2_t for the
        //     dot — pointer-cast opacity defeats the fold)
        //   * sum_a computed via a *second* v_dot2 with bf162(1,1) as the
        //     second operand, avoiding any explicit bf16→fp32 widen of A
        //   * bias correction y_b_f * partial + z_b_f * sum_a, identical
        //     to the previous fp32-FMA-chain path
        //
        // Net: 20 v_dot2_f32_bf16 + 8 fp32 FMA per int32 weight vs the
        // previous 40 fp32 FMA. v_dot2 runs at full rate on gfx1100, so
        // the substitution is ~2× cheaper for the inner accumulator.
        typedef short __attribute__((ext_vector_type(2))) bf16x2_t;
        constexpr uint32_t BF16_MAGIC = 0x43004300u;  // bf162(128, 128)
        constexpr uint32_t BF16_ONES = 0x3F803F80u;   // bf162(1.0, 1.0)
        union pack4 {
          float f[4];
          uint32_t u[4];
        };

        uint32_t w[4];
        __builtin_memcpy(w, &b_w[j], sizeof(int4));

        // Load 8 bf16 activations as 4 uint32s (= 4 bf16x2 pairs) into a
        // fp32-aliased union. Storing as uint32 keeps the IR-level type
        // opaque so the inner v_dot2 cannot be folded to fp32 widening.
        //
        // A is read direct from global (no LDS staging — see USE_LDS_A above).
        pack4 a_pack;
        {
          const uint32_t* a_words =
              reinterpret_cast<const uint32_t*>(a + offset_k + a_off);
          a_pack.u[0] = a_words[0];
          a_pack.u[1] = a_words[1];
          a_pack.u[2] = a_words[2];
          a_pack.u[3] = a_words[3];
        }

        // sum_a = Σ a[i]. Computed via 4× v_dot2_f32_bf16 with bf162(1,1) as
        // the second operand — every bf16 pair contributes 1·a_lo + 1·a_hi.
        // No fp32 widening of activations: the bytes go straight from LDS
        // through v_dot2 into the fp32 accumulator.
        float sum_a = 0.0f;
  #pragma unroll
        for (int b = 0; b < 4; ++b) {
          sum_a = __builtin_amdgcn_fdot2_f32_bf16(
              *((bf16x2_t*)(&a_pack.f[b])), *((const bf16x2_t*)&BF16_ONES),
              sum_a, /*clamp=*/false);
        }

        // unroll 1 keeps q_pack alive only one col at a time (8 fp32 VGPRs
        // recycled across cols), avoiding straight-line expansion that
        // would inflate live-range to 32 VGPRs.
  #pragma unroll 1
        for (int col = 0; col < 4; ++col) {
          // Build dequant magic values bf16(128 + nibble) directly into a
          // fp32-aliased union via uint32 stores. No fp32 in the data flow
          // until v_dot2 consumes the bytes.
          pack4 q_pack;
          const uint32_t qa = w[col];
          q_pack.u[0] = ((qa >> 0) & 0x000F000Fu) | BF16_MAGIC;
          q_pack.u[1] = ((qa >> 4) & 0x000F000Fu) | BF16_MAGIC;
          q_pack.u[2] = ((qa >> 8) & 0x000F000Fu) | BF16_MAGIC;
          q_pack.u[3] = ((qa >> 12) & 0x000F000Fu) | BF16_MAGIC;

          // partial = Σ (128 + nibble[i]) · a[i], via 4× v_dot2_f32_bf16.
          float partial = 0.0f;
  #pragma unroll
          for (int b = 0; b < 4; ++b) {
            partial = __builtin_amdgcn_fdot2_f32_bf16(
                *((bf16x2_t*)(&a_pack.f[b])), *((bf16x2_t*)(&q_pack.f[b])),
                partial, /*clamp=*/false);
          }

          // block_c += y_b_f * partial + z_b_f * sum_a
          //   y_b_f = scale, z_b_f = -(128+zero)*scale
          // partial holds (128 + nibble) · a; subtracting (128+zero)·sum_a
          // and scaling yields scale · (nibble - zero) · a as required.
          block_c[0][col] =
              __fmaf_rn(y_b_f[col], partial,
                        __fmaf_rn(z_b_f[col], sum_a, block_c[0][col]));
        }
      } else {
        // bf16 M_COUNT > 1 path with v_dot2_f32_bf16. Same opacity trick as
        // the M=1 branch: activations + magic-value weights stored in
        // fp32-aliased unions, dot via __builtin_amdgcn_fdot2_f32_bf16 with
        // pointer-cast to bf16x2_t. sum_a[m] computed via second v_dot2
        // with BF16_ONES; bias correction (y_b_f * partial + z_b_f * sum_a)
        // applied after the dot. Magic values built once per col and reused
        // across all M rows — amortizes dequant cost across M_COUNT.
        typedef short __attribute__((ext_vector_type(2))) bf16x2_t;
        constexpr uint32_t BF16_MAGIC = 0x43004300u;  // bf162(128, 128)
        constexpr uint32_t BF16_ONES = 0x3F803F80u;   // bf162(1.0, 1.0)
        union pack4 {
          float f[4];
          uint32_t u[4];
        };

        uint32_t w[4];
        __builtin_memcpy(w, &b_w[j], sizeof(int4));

        // Load M_COUNT × 8 bf16 activations as 4 uint32s each into pack4
        // unions. Stored as uint32 to keep IR-level types opaque (defeats
        // InstCombine fold). At M_COUNT=8 this is 32 fp32 VGPRs — within RDNA3
        // budget.
        pack4 a_pack[M_COUNT];
  #pragma unroll
        for (int m = 0; m < M_COUNT; ++m) {
          const uint32_t* a_words =
              reinterpret_cast<const uint32_t*>(&block_a[m][a_off]);
          a_pack[m].u[0] = a_words[0];
          a_pack[m].u[1] = a_words[1];
          a_pack[m].u[2] = a_words[2];
          a_pack[m].u[3] = a_words[3];
        }

        // sum_a[m] = Σ a[m][i] via 4× v_dot2 with bf162(1,1) — no fp32 widen.
        float sum_a[M_COUNT];
  #pragma unroll
        for (int m = 0; m < M_COUNT; ++m) {
          float s = 0.0f;
  #pragma unroll
          for (int b = 0; b < 4; ++b) {
            s = __builtin_amdgcn_fdot2_f32_bf16(*((bf16x2_t*)(&a_pack[m].f[b])),
                                                *((const bf16x2_t*)&BF16_ONES),
                                                s, /*clamp=*/false);
          }
          sum_a[m] = s;
        }

        // Per col: build magic-value pack, dot against all M activations.
        // unroll 1 keeps q_pack live one col at a time (8 fp32 VGPRs recycled)
        // — same register-pressure trick as the previous fp32 path.
  #pragma unroll 1
        for (int col = 0; col < 4; ++col) {
          pack4 q_pack;
          const uint32_t qa = w[col];
          q_pack.u[0] = ((qa >> 0) & 0x000F000Fu) | BF16_MAGIC;
          q_pack.u[1] = ((qa >> 4) & 0x000F000Fu) | BF16_MAGIC;
          q_pack.u[2] = ((qa >> 8) & 0x000F000Fu) | BF16_MAGIC;
          q_pack.u[3] = ((qa >> 12) & 0x000F000Fu) | BF16_MAGIC;

  #pragma unroll
          for (int m = 0; m < M_COUNT; ++m) {
            float partial = 0.0f;
  #pragma unroll
            for (int b = 0; b < 4; ++b) {
              partial = __builtin_amdgcn_fdot2_f32_bf16(
                  *((bf16x2_t*)(&a_pack[m].f[b])), *((bf16x2_t*)(&q_pack.f[b])),
                  partial, /*clamp=*/false);
            }
            // block_c += y_b_f * partial + z_b_f * sum_a (same correction as
            // M=1)
            block_c[m][col] =
                __fmaf_rn(y_b_f[col], partial,
                          __fmaf_rn(z_b_f[col], sum_a[m], block_c[m][col]));
          }
        }
      }
    }

    if constexpr (std::is_same<T, half>::value) {
      // y and z are group constants and this round's dots and activation
      // sums are complete: one pair of FMAs per (row, column) per round
      // instead of per j-iteration. Per-round z applications sum exactly to
      // the per-group correction because z does not change inside a group.
  #pragma unroll
      for (int m = 0; m < M_COUNT; ++m) {
  #pragma unroll
        for (int c = 0; c < 4; ++c) {
          block_c[m][c] += yf_h[c][0] * acc_lo[m][c] +
                           yf_h[c][1] * acc_hi[m][c] + zf_h[c][0] * sum_lo[m] +
                           zf_h[c][1] * sum_hi[m];
        }
      }
    }
    k += 32;  // 4 weight words * 8 nibbles = 32 K elements
  }

  // ---- Epilogue: three store modes, selected by launch shape ----
  //
  // partials != nullptr: deterministic split-K. Plain FP32 stores to
  //   partials[(blockIdx.z * size_m + m) * size_n + n]
  // (one writer per in-range slot; threads with n >= size_n returned before
  // the epilogue and rows past size_m are skipped — exactly the slots the
  // in-kernel fixed-order reduce below never reads). n is a multiple of 4
  // and size_n % 8 == 0, so the 4-lane store never crosses the right edge.
  // After the stores, the block publishes its arrival on the tile's ticket
  // (a per-call zero-filled counter); the LAST z block of the tile then
  // sums the z slices in fixed ascending order with a single final cast —
  // the reduction order is a pure function of the launch shape, so the
  // result stays bit-reproducible for identical inputs. Tickets are
  // per-invocation state (see launch_gemm_q4_deterministic); nothing is
  // shared across kernel launches, so concurrent invocations on other
  // streams cannot interfere.
  //
  // partials == nullptr && gridDim.z == 1: single split block per output
  //   element — direct store of the rounded accumulator. No scratch, no
  //   reduce, no atomics; c may be left uninitialized (torch::empty).
  //
  // partials == nullptr && gridDim.z > 1: LEGACY pre-#54706 CAS epilogue
  //   (A/B control only; never selected by launch_gemm_q4_deterministic).
  //   Adds the narrowed 4-lane partial into c with one 64-bit CAS per 4
  //   columns; REQUIRES a pre-zeroed output because the CAS adds into
  //   whatever is already there.
  #pragma unroll
  for (int m = 0; m < M_COUNT; ++m) {
    if (offset_m + m >= size_m) continue;  // skip padding rows past size_m
    if (partials != nullptr) {
      float* p =
          partials + ((long)blockIdx.z * size_m + (offset_m + m)) * size_n + n;
      p[0] = block_c[m][0];
      p[1] = block_c[m][1];
      p[2] = block_c[m][2];
      p[3] = block_c[m][3];
      continue;
    }
    T* out = c + (offset_m + m) * size_n + n;
    if (gridDim.z == 1) {
      // Single writer per element: round once and store the packed 4 lanes
      // (8-byte aligned, see the note above) in one go.
      if constexpr (std::is_same<T, half>::value) {
        half2 packed[2] = {__halves2half2(__float2half_rn(block_c[m][0]),
                                          __float2half_rn(block_c[m][1])),
                           __halves2half2(__float2half_rn(block_c[m][2]),
                                          __float2half_rn(block_c[m][3]))};
        __builtin_memcpy(out, packed, sizeof(packed));
      } else {
        bf162_t packed[2];
        packed[0].x = __float2bfloat16(block_c[m][0]);
        packed[0].y = __float2bfloat16(block_c[m][1]);
        packed[1].x = __float2bfloat16(block_c[m][2]);
        packed[1].y = __float2bfloat16(block_c[m][3]);
        __builtin_memcpy(out, packed, sizeof(packed));
      }
      continue;
    }
    // Legacy CAS-atomic epilogue: see the store-mode comment above.
    if constexpr (std::is_same<T, half>::value) {
      half2 r01 = __halves2half2(__float2half_rn(block_c[m][0]),
                                 __float2half_rn(block_c[m][1]));
      half2 r23 = __halves2half2(__float2half_rn(block_c[m][2]),
                                 __float2half_rn(block_c[m][3]));
      atomic_add_pk4_f16(out, r01, r23);
    } else {
      bf162_t r01;
      r01.x = __float2bfloat16(block_c[m][0]);
      r01.y = __float2bfloat16(block_c[m][1]);
      bf162_t r23;
      r23.x = __float2bfloat16(block_c[m][2]);
      r23.y = __float2bfloat16(block_c[m][3]);
      atomic_add_pk4_bf16(out, r01, r23);
    }
  }

  // Deterministic in-kernel split-K reduction (fp16 only — on gfx1100 the
  // bf16 scalar kernel measured neutral at M=1 and up to 10% slower at
  // M=4-8 with the reduction fused, so bf16 keeps the separate reducer;
  // see launch_gemm_q4_deterministic). Every (x, y) tile has gridDim.z
  // producer blocks; the last one to finish sums the tile's z slices in
  // fixed ascending order and writes the output with a single final cast.
  // Memory ordering: each thread's partials stores precede its
  // __threadfence() (release, device scope); the fence precedes the block
  // barrier, so thread 0's ticket increment happens after every store of
  // the block is device-visible. The last block's second __threadfence()
  // pairs with the writers' fences (acquire side), making the other blocks'
  // partials visible before they are read. The ticket is re-armed after the
  // reduction so the same per-call array serves the next row tile
  // (launches on a stream are ordered, so the re-arm is complete before any
  // later launch reads the ticket).
  if constexpr (std::is_same<T, half>::value) {
    if (partials != nullptr) {
      __shared__ int s_last;
      __threadfence();
      __syncthreads();
      if (t == 0) {
        s_last = atomicAdd(tickets + blockIdx.y * gridDim.x + blockIdx.x, 1) ==
                 (int)gridDim.z - 1;
      }
      __syncthreads();
      if (!s_last) return;
      __threadfence();

      typedef float f4v __attribute__((ext_vector_type(4)));
      const long zs = ((long)size_m * size_n) / 4;  // z-slice stride, f4 units
      const int zc = (int)gridDim.z;
  #pragma unroll
      for (int m = 0; m < M_COUNT; ++m) {
        if (offset_m + m >= size_m) continue;  // skip padding rows past size_m
        const f4v* p =
            (const f4v*)(partials + (long)(offset_m + m) * size_n + n);
        f4v acc = {};
        int z = 0;
        // Ascending-z accumulation with batched loads (one L2 round trip per
        // eight slices); the addition order is fixed, so the result is
        // bit-reproducible.
        for (; z + 8 <= zc; z += 8) {
          f4v v[8];
  #pragma unroll
          for (int u = 0; u < 8; ++u)
            v[u] = __builtin_nontemporal_load(p + (long)(z + u) * zs);
  #pragma unroll
          for (int u = 0; u < 8; ++u) acc += v[u];
        }
        for (; z < zc; ++z) acc += __builtin_nontemporal_load(p + (long)z * zs);

        T* out = c + (long)(offset_m + m) * size_n + n;
        half2 packed[2] = {
            __halves2half2(__float2half_rn(acc[0]), __float2half_rn(acc[1])),
            __halves2half2(__float2half_rn(acc[2]), __float2half_rn(acc[3]))};
        __builtin_memcpy(out, packed, sizeof(packed));
      }
      if (t == 0) *(tickets + blockIdx.y * gridDim.x + blockIdx.x) = 0;
    }
  }
}

#else  // non-RDNA3 device pass: empty __global__ for symbol parity.

template <typename T, int M_COUNT>
__global__ void gemm_q4_kernel_rdna3(const T*, const uint32_t*, const uint32_t*,
                                     const T*, T*, const int, const int,
                                     const int, const int, const int, float*,
                                     int*) {}

#endif  // __HIP__RDNA3__ || !__HIP_DEVICE_COMPILE__

// ---------------------------------------------------------------------------
// Launcher.
// ---------------------------------------------------------------------------

template <typename T, int M_COUNT>
void launch_gemm_q4_for_mcount(const T* a, const uint32_t* b_q_weight,
                               const uint32_t* b_qzeros, const T* b_scales,
                               T* c, int size_m, int size_n, int size_k,
                               int groups, int zero_offset, float* partials,
                               int* tickets, cudaStream_t stream) {
  dim3 block(THREADS_X);
  dim3 grid((size_n + BLOCK_KN_SIZE * 4 - 1) / (BLOCK_KN_SIZE * 4),
            (size_m + M_COUNT - 1) / M_COUNT,
            (size_k + BLOCK_KN_SIZE - 1) / BLOCK_KN_SIZE);

  gemm_q4_kernel_rdna3<T, M_COUNT><<<grid, block, 0, stream>>>(
      a, b_q_weight, b_qzeros, b_scales, c, size_m, size_n, size_k, groups,
      zero_offset, partials, tickets);
}

// The M_COUNT template the launcher below picks for a given row count.
static inline int mcount_for_rows(int rows) {
  if (rows == 1) return 1;
  if (rows <= 3) return 2;
  if (rows <= 7) return 4;
  return 8;
}

// Dispatch to the largest M_COUNT template that doesn't waste more than
// half a tile. Caps at 8: above that, the WMMA-prefill kernel (M >= 12) is
// the right tool, not bigger M_COUNT in the scalar dot-product path.
//
// Tile-waste table:
//   M=1   -> M_COUNT=1   (no waste)
//   M=2,3 -> M_COUNT=2   (M=3 wastes 1/2 of last tile)
//   M=4-7 -> M_COUNT=4   (worst case M=5: wastes 3/4 of last tile)
//   M=8-15-> M_COUNT=8   (worst case M=9: wastes 7/8 of last tile)
// "Wasted" rows are zero-padded in LDS and skip the atomic write, so they
// only burn instructions on the last block, never affect correctness.
template <typename T>
void launch_gemm_q4(const T* a, const uint32_t* b_q_weight,
                    const uint32_t* b_qzeros, const T* b_scales, T* c,
                    int size_m, int size_n, int size_k, int groups,
                    bool use_v2_format, float* partials, int* tickets,
                    cudaStream_t stream) {
  const int zero_offset = use_v2_format ? 0 : 1;
  switch (mcount_for_rows(size_m)) {
    case 1:
      launch_gemm_q4_for_mcount<T, 1>(a, b_q_weight, b_qzeros, b_scales, c,
                                      size_m, size_n, size_k, groups,
                                      zero_offset, partials, tickets, stream);
      break;
    case 2:
      launch_gemm_q4_for_mcount<T, 2>(a, b_q_weight, b_qzeros, b_scales, c,
                                      size_m, size_n, size_k, groups,
                                      zero_offset, partials, tickets, stream);
      break;
    case 4:
      launch_gemm_q4_for_mcount<T, 4>(a, b_q_weight, b_qzeros, b_scales, c,
                                      size_m, size_n, size_k, groups,
                                      zero_offset, partials, tickets, stream);
      break;
    default:
      // M_COUNT=8 covers the whole scalar domain (M <= 11 in practice: the
      // public dispatch routes M >= 12 to WMMA); this branch is only reached
      // when the caller falls through, and still produces correct output.
      launch_gemm_q4_for_mcount<T, 8>(a, b_q_weight, b_qzeros, b_scales, c,
                                      size_m, size_n, size_k, groups,
                                      zero_offset, partials, tickets, stream);
      break;
  }
}

// Deterministic split-K reduction for the bf16 launches: one thread per
// output element sums the grid.z FP32 partial slices in fixed ascending-z
// order and rounds to the output dtype exactly once. The order is a pure
// function of the launch shape, so the result is bit-reproducible.
template <typename T>
__global__ void reduce_partials_rdna3(const float* __restrict__ partials,
                                      T* __restrict__ c, const int z_count,
                                      const int size_m, const int size_n) {
  const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (long)size_m * size_n) return;
  const int m = (int)(idx / size_n);
  const int n = (int)(idx % size_n);
  float acc = 0.0f;
  for (int z = 0; z < z_count; ++z)
    acc += partials[((long)z * size_m + m) * size_n + n];
  if constexpr (std::is_same<T, half>::value) {
    c[idx] = __float2half_rn(acc);
  } else {
    c[idx] = __float2bfloat16(acc);
  }
}

// Deterministic scalar GEMM: split-K blocks write FP32 partials to scratch
// (no atomics, no intermediate low-precision rounding), and the last z
// block of each (x, y) tile performs the fixed ascending-z reduction
// in-kernel with a single final cast (see the kernel epilogue). The scalar
// kernel writes every in-range (z, m, n) partial exactly once, so a plain
// at::empty scratch suffices; row-tile bound:
//   scratch_bytes = z_count * TILE_M * size_n * 4
// is independent of the caller's M (the scalar domain is M < 64, so a single
// tile covers it). The PyTorch caching allocator (including its CUDA-graph
// capture pool) owns scratch reuse and lifetime.
//
// The arrival tickets are per-call at::zeros state: every launch of this
// call counts on tickets that start at zero (the kernel re-arms them for
// the next row tile), and concurrent invocations get distinct allocations
// from the stream-aware caching allocator, so no synchronization state is
// ever shared between invocations. When z_count == 1 the kernel's
// direct-store epilogue (gridDim.z == 1) is already deterministic, so the
// scratch and ticket allocations are skipped entirely.
template <typename T>
void launch_gemm_q4_deterministic(const T* a, const uint32_t* b_q_weight,
                                  const uint32_t* b_qzeros, const T* b_scales,
                                  T* c, int size_m, int size_n, int size_k,
                                  int groups, bool use_v2_format,
                                  cudaStream_t stream) {
  constexpr int TILE_M = 64;  // single tile covers the scalar domain
  const int z_count = (size_k + BLOCK_KN_SIZE - 1) / BLOCK_KN_SIZE;
  if (z_count == 1) {
    launch_gemm_q4(a, b_q_weight, b_qzeros, b_scales, c, size_m, size_n, size_k,
                   groups, use_v2_format,
                   /*partials=*/nullptr, /*tickets=*/nullptr, stream);
    return;
  }
  at::TensorOptions dev_opts = at::TensorOptions().device(
      at::Device(at::kCUDA, c10::cuda::current_device()));
  at::Tensor partials = at::empty({z_count, std::min(TILE_M, size_m), size_n},
                                  dev_opts.dtype(at::kFloat));
  float* partials_ptr = partials.data_ptr<float>();

  int* tickets_ptr = nullptr;
  if constexpr (std::is_same<T, half>::value) {
    // fp16: the reduction runs in the kernel (see the epilogue). One
    // ticket per (x, y) tile, sized for the widest row-tile launch.
    const int gx = (size_n + BLOCK_KN_SIZE * 4 - 1) / (BLOCK_KN_SIZE * 4);
    int max_gy = 1;
    for (int row0 = 0; row0 < size_m; row0 += TILE_M) {
      const int rows = std::min(TILE_M, size_m - row0);
      const int mc = mcount_for_rows(rows);
      max_gy = std::max(max_gy, (rows + mc - 1) / mc);
    }
    at::Tensor tickets =
        at::zeros({(long)gx * max_gy}, dev_opts.dtype(at::kInt));
    for (int row0 = 0; row0 < size_m; row0 += TILE_M) {
      const int rows = std::min(TILE_M, size_m - row0);
      launch_gemm_q4(a + (long)row0 * size_k, b_q_weight, b_qzeros, b_scales,
                     c + (long)row0 * size_n, rows, size_n, size_k, groups,
                     use_v2_format, partials_ptr, tickets.data_ptr<int>(),
                     stream);
    }
  } else {
    // bf16: separate fixed-order reduce (measured faster than the fused
    // form on this path).
    for (int row0 = 0; row0 < size_m; row0 += TILE_M) {
      const int rows = std::min(TILE_M, size_m - row0);
      launch_gemm_q4(a + (long)row0 * size_k, b_q_weight, b_qzeros, b_scales,
                     c + (long)row0 * size_n, rows, size_n, size_k, groups,
                     use_v2_format, partials_ptr, tickets_ptr, stream);
      const long total = (long)rows * size_n;
      const int threads = 256;
      const int blocks = (int)((total + threads - 1) / threads);
      reduce_partials_rdna3<T><<<blocks, threads, 0, stream>>>(
          partials_ptr, c + (long)row0 * size_n, z_count, rows, size_n);
    }
  }
}

}  // namespace gptq_rdna3
}  // namespace vllm

// ---------------------------------------------------------------------------
// Public entry point.
// ---------------------------------------------------------------------------
//
// Inputs:
//   a         [M, K]            half or bfloat16
//   b_q_weight[K/8, N]          uint32 (already shuffled via gptq_shuffle)
//   b_qzeros  [groups, N/8]     uint32 (packed 4-bit zeros)
//   b_scales  [groups, N]       half or bfloat16
//   use_v2_format                bool   (true = GPTQv2, no +1 zero offset)
//
// Output:
//   c         [M, N]            same dtype as a

torch::Tensor gptq_gemm_rdna3_wmma(torch::Tensor a, torch::Tensor b_q_weight,
                                   torch::Tensor b_qzeros,
                                   torch::Tensor b_scales, bool use_v2_format);

// WMMA dispatch threshold, from the measured scalar/WMMA crossover on
// gfx1100 (ratio = scalar_us / wmma_us, >1 means WMMA faster; median of
// 300 CUDA-event timings per arm, identical inputs, both ops called
// directly):
//
//   K / N (dtype)     M=1   M=4   M=8  M=11  M=12  M=16  M=24  M=32
//   1536/5120 fp16   0.69  0.97  1.36  1.60  1.59  1.60  1.22  1.62
//   4096/4096 fp16   0.46  0.69  1.01  1.30  1.30  1.32  1.05  1.40
//   4352/5120 fp16   0.42  0.63  0.93  1.37  1.37  1.37  1.19  1.74
//   5120/2048 fp16   0.44  0.62  0.87  1.00  1.00  1.01  1.15  1.34
//   5120/8704 fp16   0.30  0.53  0.84  1.39  1.42  1.43  1.24  1.58
//   1536/5120 bf16   1.07  1.23  1.50  1.71  1.69  1.00  1.00  1.00
//   4096/4096 bf16   0.70  0.82  1.08  1.33  1.34  1.00  1.00  1.00
//   4352/5120 bf16   0.62  0.74  0.98  1.40  1.40  1.00  1.00  1.00
//   5120/2048 bf16   0.64  0.73  0.89  1.00  1.00  1.00  1.01  1.00
//   5120/8704 bf16   0.45  0.65  0.92  1.46  1.49  1.00  1.00  1.00
//
// At M >= 12 WMMA wins or ties on every shape for both dtypes (exact 1.00
// cells and the one sub-1.0 sample, bf16 4096/4096 at M=24 = 0.996, are
// within measurement noise); below
// that the scalar path wins on most shapes (and is the more accurate of
// the two since the exact fp16 factored dequant, so it keeps the band).
// fp16 used to stay scalar until M=64 because the old baked-dequant
// scalar kernel was faster there — that speed came with ~100x more
// dequant error, and with the exact dequant the crossover moved down to
// the same 12 as bf16. A shape-aware (N/K) dispatch would do better
// still; that is tuning work beyond this fix.
constexpr int64_t WMMA_MIN_M = 12;

torch::Tensor gptq_gemm_rdna3(torch::Tensor a, torch::Tensor b_q_weight,
                              torch::Tensor b_qzeros, torch::Tensor b_scales,
                              bool use_v2_format) {
  if (a.dim() == 2 && b_q_weight.dim() == 2 && a.size(1) % 16 == 0 &&
      b_q_weight.size(1) % 16 == 0 && a.size(0) >= WMMA_MIN_M) {
    return gptq_gemm_rdna3_wmma(a, b_q_weight, b_qzeros, b_scales,
                                use_v2_format);
  }

  TORCH_CHECK(a.is_cuda(), "a must be a CUDA/HIP tensor");
  TORCH_CHECK(b_q_weight.is_cuda(), "b_q_weight must be a CUDA/HIP tensor");
  TORCH_CHECK(b_qzeros.is_cuda(), "b_qzeros must be a CUDA/HIP tensor");
  TORCH_CHECK(b_scales.is_cuda(), "b_scales must be a CUDA/HIP tensor");
  TORCH_CHECK(a.dim() == 2, "a must be 2D [M, K]");
  TORCH_CHECK(b_q_weight.dim() == 2, "b_q_weight must be 2D [K/8, N]");
  TORCH_CHECK(
      a.scalar_type() == torch::kHalf || a.scalar_type() == torch::kBFloat16,
      "a must be half or bfloat16");
  TORCH_CHECK(a.scalar_type() == b_scales.scalar_type(),
              "b_scales dtype must match a");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
  auto stream = at::cuda::getCurrentCUDAStream();

  int size_m = (int)a.size(0);
  int size_k = (int)a.size(1);
  int size_n = (int)b_q_weight.size(1);
  int groups = (int)b_qzeros.size(0);

  TORCH_CHECK(b_q_weight.size(0) * 8 == size_k,
              "b_q_weight first dim must be K/8");
  TORCH_CHECK(size_k % (groups * 32) == 0,
              "group size (K/groups = ", size_k / groups,
              ") must be a multiple of 32: the kernel checks group "
              "transitions at 32-K granularity");
  TORCH_CHECK(b_scales.size(0) == groups,
              "b_scales must have same group count as qzeros");
  TORCH_CHECK(b_scales.size(1) == size_n, "b_scales last dim must be N");
  TORCH_CHECK(size_n % 8 == 0,
              "N must be a multiple of 8 (packed qzeros layout: 8 4-bit zero "
              "points per uint32 along N; also keeps the 4-column epilogue "
              "store in bounds)");

  auto opts = torch::TensorOptions().dtype(a.dtype()).device(a.device());
  // The deterministic epilogue writes every output element exactly once
  // (reduce pass below — or the direct store when z_count == 1), so c needs
  // no zero-initialization. This is a one-way door: the legacy CAS epilogue
  // (see the kernel's store-mode comment) ADDS into c and would require
  // torch::zeros here if it were ever re-selected.
  at::Tensor c = torch::empty({size_m, size_n}, opts);

  // Deterministic split-K: FP32 partials + fixed-order reduction (see
  // launch_gemm_q4_deterministic). The CAS-atomic low-precision epilogue
  // this replaces was order-dependent and produced different results for
  // identical inputs on gfx11 once more than a few split blocks contended.
  if (a.scalar_type() == torch::kHalf) {
    vllm::gptq_rdna3::launch_gemm_q4_deterministic<half>(
        (const half*)a.data_ptr(), (const uint32_t*)b_q_weight.data_ptr(),
        (const uint32_t*)b_qzeros.data_ptr(), (const half*)b_scales.data_ptr(),
        (half*)c.data_ptr(), size_m, size_n, size_k, groups, use_v2_format,
        stream);
  } else {
    vllm::gptq_rdna3::launch_gemm_q4_deterministic<vllm::gptq_rdna3::bf16_t>(
        (const vllm::gptq_rdna3::bf16_t*)a.data_ptr(),
        (const uint32_t*)b_q_weight.data_ptr(),
        (const uint32_t*)b_qzeros.data_ptr(),
        (const vllm::gptq_rdna3::bf16_t*)b_scales.data_ptr(),
        (vllm::gptq_rdna3::bf16_t*)c.data_ptr(), size_m, size_n, size_k, groups,
        use_v2_format, stream);
  }

  return c;
}
