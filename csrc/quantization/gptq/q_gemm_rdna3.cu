// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// W4A16 GPTQ kernel for RDNA3 (gfx1100 / RX 7900 XTX class), templated on the
// activation dtype (half or __hip_bfloat16). Adapted from exllamav2's 4-bit
// kernel (csrc/quantization/gptq/q_gemm.cu) with the following changes:
//
//   1. Direct write to the T-typed output via packed CAS-loop on a 64-bit
//      word (atomic_add_pk4_{f16,bf16}). gfx11 has no native
//      v_global_atomic_pk_add_{f16,bf16}, so the kernel emulates one with
//      global_atomic_cmpswap_b64. This avoids the M*N*4-byte FP32 scratch
//      buffer + memset + cast-pass that an fp32-accumulator design would
//      need; the caller passes a zero-initialised T-typed output tensor
//      and every block atomically adds its partial sum into it.
//
//   2. The bf16 path uses a dedicated bit-trick that avoids the fp16-only
//      "upper nibble * 16" trick, which would overflow the 7-bit bf16
//      mantissa. See qdq_4_rdna3.cuh for details.
//
//   3. Wave32 geometry sized for high CU saturation: THREADS_X=256
//      (8 waves per block) and BLOCK_KN_SIZE=256, with each thread
//      computing 4 N output columns. gridDim.z = K / BLOCK_KN_SIZE
//      splits K and the output is atomically accumulated. fp16 uses
//      v_dot2_f32_f16 (__builtin_amdgcn_fdot2) for the inner dot;
//      bf16 widens to fp32 (no v_pk_fma_bf16 on gfx11) and accumulates
//      with v_fma_f32. M_COUNT ∈ {1,2,4,8} is selected at launch
//      based on size_m.
//
//   4. The bf16 dispatch with M >= 16 forwards to the WMMA kernel in
//      q_gemm_rdna3_wmma.cu (separate translation unit) where
//      v_wmma_f32_16x16x16_bf16_w32 wins. The fp16 path always stays
//      scalar (the bit-trick dequant beats WMMA below M=64).

#include <cstdint>
#include <cstdio>

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#if defined(USE_ROCM)
  #include <hip/hip_runtime.h>
  #include <hip/hip_bf16.h>
  #include <hip/hip_fp16.h>
#else
  #include <cuda_runtime.h>
  #include <cuda_bf16.h>
  #include <cuda_fp16.h>
#endif

#include "qdq_4_rdna3.cuh"

#if defined(USE_ROCM)

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

__forceinline__ __device__ float dot22_8_f(half2 (&dq)[4], const half* a_ptr) {
  // RDNA3 has v_dot2_f32_f16 (`__builtin_amdgcn_fdot2`) which computes
  // fp32 += a.x*b.x + a.y*b.y in a single instruction with the accumulator
  // staying in fp32 throughout. hipcc 7.2 does NOT peephole the obvious
  // `__hfma2 + cast + add` pattern into v_dot2 (verified by ISA
  // disassembly: 0 v_dot2_f32_f16 vs 256 v_cvt_f32_f16 + 218 v_add_f32 in
  // the M_COUNT=8 kernel before this change), so we issue the builtin
  // explicitly. Saves the trailing 2× v_cvt_f32_f16 + v_add_f32 (3 ops)
  // per dot22_8_f call vs the half2-accumulator form. With 128 calls per
  // K=32 step that's ~384 ops/K-step less issue pressure on the VALU.
  //
  // Numerical bonus: accumulator stays fp32 throughout the dot. The old
  // form accumulated 8 muladds in fp16 (10-bit mantissa) before casting,
  // which could lose ~3 bits of precision on borderline magnitudes.
  float result = 0.0f;
  const half2* a2_ptr = (const half2*)a_ptr;
#pragma unroll
  for (int i = 0; i < 4; i++) {
    result = __builtin_amdgcn_fdot2(dq[i], *a2_ptr++, result, /*clamp=*/false);
  }
  return result;
}

__forceinline__ __device__ float dot22_8_f(bf162_t (&dq)[4],
                                           const bf16_t* a_ptr) {
  // RDNA3 (gfx1100) lacks a packed bf16 FMA: there is no v_pk_fma_bf16 in
  // the gfx11 ISA (it only landed on CDNA3+ / gfx94x and later). hipcc
  // therefore lowers __hfma2(bf162_t, bf162_t, bf162_t) to a serialised
  // fallback (single-element FMAs or fp32 round-trips), which empirically
  // runs ~2× the cycle count of v_pk_fma_f16 on the same VALU. The bf16
  // decode path was paying that tax in full, scaling linearly with M (the
  // fp16 path scales sub-linearly because its v_pk_fma_f16 is full rate
  // and the kernel becomes memory-bound).
  //
  // Fix: widen bf16 → fp32 explicitly (a left-shift by 16, free in VGPRs)
  // and accumulate with v_fma_f32, which IS full rate on RDNA3. Same FMA
  // count, but each FMA is fast. Bonus: the accumulator is now fp32
  // throughout instead of bf16, which is also numerically more accurate
  // (no compounding bf16-rounding inside the dot loop).
  float result = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; i++) {
    uint32_t aw, dw;
    __builtin_memcpy(&aw, a_ptr + 2 * i, sizeof(uint32_t));
    __builtin_memcpy(&dw, &dq[i], sizeof(uint32_t));
    // bf16 in low 16 bits  → fp32 by left-shifting into the upper half.
    // bf16 in high 16 bits → already aligned with fp32's upper half.
    float a_x = __uint_as_float((aw & 0xFFFFu) << 16);
    float a_y = __uint_as_float(aw & 0xFFFF0000u);
    float d_x = __uint_as_float((dw & 0xFFFFu) << 16);
    float d_y = __uint_as_float(dw & 0xFFFF0000u);
    result = __fmaf_rn(d_x, a_x, result);
    result = __fmaf_rn(d_y, a_y, result);
  }
  return result;
}

// fp32-input dot product: paired with dequant_4bit_8_bf16_f32 which already
// produces fp32 dq[8]. Saves the bf16→fp32 widening that the bf162_t
// overload above does for dq (still need to widen A from bf16). Wins more
// at high N: the bf162_t version's per-call widening cost scales with the
// number of dequants × M_COUNT × 4 dot calls; the fp32 version pays only
// for A widening (M_COUNT × 4 × 4 widens, half as many).
__forceinline__ __device__ float dot22_8_f(float (&dq)[8],
                                           const bf16_t* a_ptr) {
  float result = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; i++) {
    uint32_t aw;
    __builtin_memcpy(&aw, a_ptr + 2 * i, sizeof(uint32_t));
    float a_x = __uint_as_float((aw & 0xFFFFu) << 16);
    float a_y = __uint_as_float(aw & 0xFFFF0000u);
    result = __fmaf_rn(dq[2 * i + 0], a_x, result);
    result = __fmaf_rn(dq[2 * i + 1], a_y, result);
  }
  return result;
}

// ---------------------------------------------------------------------------
// Packed atomic-add via CAS-loop on a 64-bit word (4 fp16/bf16 lanes per CAS).
// RDNA3 (gfx11) does NOT have native v_global_atomic_pk_add_f16 / _bf16 (those
// landed on gfx940 / gfx1250 respectively), so this lowers to
// global_atomic_cmpswap_b64 plus retry. We use this in the kernel epilogue to
// write 4 output columns per row in a single atomic operation — half the
// atomic instruction count and half the contention vs two 32-bit CAS calls.
//
// Writing directly to fp16/bf16 (instead of through an FP32 scratch buffer +
// cast pass) saves M*N*4 bytes of allocation, the memset, and the epilogue
// cast pass that an fp32-accumulator design would need.
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
    const int groups, const int zero_offset, const int* __restrict__ b_q_perm) {
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

  // Stage A: each thread loads 1 K element per M row into LDS (with optional
  // act-order permutation). THREADS_X == BLOCK_KN_SIZE so this is a 1:1 map.
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
          if (b_q_perm)
            av = a_row[b_q_perm[offset_k + t]];
          else
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

  // Per-column dequant constants. We hold one set of (z, y) pairs per column.
  // fp16 uses the exllama (z1z16, y1y16) double-pair to enable the upper-
  // nibble-*16 trick. bf16 uses fp32 scalars (z, y) because the dequant
  // produces fp32 directly — see prep_zero_scale_bf16_f32 / the FMA
  // bypass for the missing v_pk_fma_bf16 on gfx11.
  half2 z1z16_h[4][2], y1y16_h[4][2];
  float z_b_f[4], y_b_f[4];

  auto refresh_group = [&](int g) {
    const uint32_t* qz_row = b_qzeros + g * (size_n / 8);
    const T* sc_row = b_scales + g * size_n;
    int zeros[4];
    T scales[4];
    load4_zeros(qz_row, n, zeros);
    load4_scales<T>(sc_row, n, scales);
    if constexpr (std::is_same<T, half>::value) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        prep_zero_scale_fp16((uint32_t)(zeros[i] + zero_offset), scales[i],
                             z1z16_h[i], y1y16_h[i]);
      }
    } else {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        prep_zero_scale_bf16_f32((uint32_t)(zeros[i] + zero_offset), scales[i],
                                 z_b_f[i], y_b_f[i]);
      }
    }
  };

  refresh_group(group);

  float block_c[M_COUNT][4];
#pragma unroll
  for (int m = 0; m < M_COUNT; ++m) {
#pragma unroll
    for (int j = 0; j < 4; ++j) block_c[m][j] = 0.0f;
  }

  // Note on group-transition granularity: we check `k == nextgroup` at the
  // start of each outer iteration (which advances K by 32). This is correct
  // when group_size >= 32 OR group_size divides 32 evenly (groupsize is one
  // of {1,2,4,8,16,32,64,128,...}). For group_size in {16, 8, 4, ...} the
  // inner loop would cross a group boundary between j-iterations; we require
  // group_size >= 32 here, mirroring exllama's assumption.
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
      refresh_group(group);
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

#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int a_off = (k - offset_k) + 8 * j;

      if constexpr (std::is_same<T, half>::value) {
        half2 dq[4][4];
        dequant_4bit_8_fp16((uint32_t)b_w[j].x, dq[0], z1z16_h[0], y1y16_h[0]);
        dequant_4bit_8_fp16((uint32_t)b_w[j].y, dq[1], z1z16_h[1], y1y16_h[1]);
        dequant_4bit_8_fp16((uint32_t)b_w[j].z, dq[2], z1z16_h[2], y1y16_h[2]);
        dequant_4bit_8_fp16((uint32_t)b_w[j].w, dq[3], z1z16_h[3], y1y16_h[3]);

#pragma unroll
        for (int m = 0; m < M_COUNT; ++m) {
          const half* a_ptr = reinterpret_cast<const half*>(&block_a[m][a_off]);
          block_c[m][0] += dot22_8_f(dq[0], a_ptr);
          block_c[m][1] += dot22_8_f(dq[1], a_ptr);
          block_c[m][2] += dot22_8_f(dq[2], a_ptr);
          block_c[m][3] += dot22_8_f(dq[3], a_ptr);
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
    k += 32;  // 4 weight words * 8 nibbles = 32 K elements
  }

  // Pack the 4 FP32 partial sums into 2 packed pairs and atomically add all
  // four lanes in a single 64-bit CAS write directly to the T-typed output
  // (caller pre-zeros it). On gfx11 the packed atomic is a CAS-loop, but with
  // a single b64 op we halve the atomic instruction count vs two b32 CAS
  // calls, AND save the FP32 buffer + memset + cast pass entirely.
#pragma unroll
  for (int m = 0; m < M_COUNT; ++m) {
    if (offset_m + m >= size_m) continue;  // skip padding rows past size_m
    T* out = c + (offset_m + m) * size_n + n;
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
}

// ---------------------------------------------------------------------------
// Launcher.
// ---------------------------------------------------------------------------

template <typename T, int M_COUNT>
void launch_gemm_q4_for_mcount(const T* a, const uint32_t* b_q_weight,
                               const uint32_t* b_qzeros, const T* b_scales,
                               const int* b_q_perm, T* c, int size_m,
                               int size_n, int size_k, int groups,
                               int zero_offset, cudaStream_t stream) {
  dim3 block(THREADS_X);
  dim3 grid((size_n + BLOCK_KN_SIZE * 4 - 1) / (BLOCK_KN_SIZE * 4),
            (size_m + M_COUNT - 1) / M_COUNT,
            (size_k + BLOCK_KN_SIZE - 1) / BLOCK_KN_SIZE);

  gemm_q4_kernel_rdna3<T, M_COUNT><<<grid, block, 0, stream>>>(
      a, b_q_weight, b_qzeros, b_scales, c, size_m, size_n, size_k, groups,
      zero_offset, b_q_perm);
}

// Dispatch to the largest M_COUNT template that doesn't waste more than
// half a tile. Caps at 8: above that, the WMMA-prefill kernel (M >= 16) is
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
                    const uint32_t* b_qzeros, const T* b_scales,
                    const int* b_q_perm, T* c, int size_m, int size_n,
                    int size_k, int groups, bool use_v2_format,
                    cudaStream_t stream) {
  const int zero_offset = use_v2_format ? 0 : 1;

  if (size_m == 1) {
    launch_gemm_q4_for_mcount<T, 1>(a, b_q_weight, b_qzeros, b_scales, b_q_perm,
                                    c, size_m, size_n, size_k, groups,
                                    zero_offset, stream);
  } else if (size_m <= 3) {
    launch_gemm_q4_for_mcount<T, 2>(a, b_q_weight, b_qzeros, b_scales, b_q_perm,
                                    c, size_m, size_n, size_k, groups,
                                    zero_offset, stream);
  } else if (size_m <= 7) {
    launch_gemm_q4_for_mcount<T, 4>(a, b_q_weight, b_qzeros, b_scales, b_q_perm,
                                    c, size_m, size_n, size_k, groups,
                                    zero_offset, stream);
  } else {
    // M_COUNT=8 covers M up to 15 here; M >= 16 should ideally take the
    // WMMA path, but if it falls through we still produce correct output —
    // just leaving 3-5× of throughput on the table for prefill workloads.
    launch_gemm_q4_for_mcount<T, 8>(a, b_q_weight, b_qzeros, b_scales, b_q_perm,
                                    c, size_m, size_n, size_k, groups,
                                    zero_offset, stream);
  }
}

}  // namespace gptq_rdna3
}  // namespace vllm

#endif  // USE_ROCM

// ---------------------------------------------------------------------------
// Public entry point.
// ---------------------------------------------------------------------------
//
// Inputs:
//   a         [M, K]            half or bfloat16
//   b_q_weight[K/8, N]          uint32 (already shuffled via gptq_shuffle)
//   b_qzeros  [groups, N/8]     uint32 (packed 4-bit zeros)
//   b_scales  [groups, N]       half or bfloat16
//   b_g_idx   [K] or empty      int32 (act-order permutation; empty=identity)
//   use_v2_format                bool   (true = GPTQv2, no +1 zero offset)
//
// Output:
//   c         [M, N]            same dtype as a

#if defined(USE_ROCM)
// Forward declaration of the WMMA prefill entry. Defined in
// q_gemm_rdna3_wmma.cu (separate TU so the WMMA builtins are kept localised;
// cross-TU call resolved at link time, no codegen interaction).
torch::Tensor gptq_gemm_rdna3_wmma(torch::Tensor a, torch::Tensor b_q_weight,
                                   torch::Tensor b_qzeros,
                                   torch::Tensor b_scales,
                                   torch::Tensor b_g_idx, bool use_v2_format);
#endif

torch::Tensor gptq_gemm_rdna3(torch::Tensor a, torch::Tensor b_q_weight,
                              torch::Tensor b_qzeros, torch::Tensor b_scales,
                              torch::Tensor b_g_idx, bool use_v2_format) {
#if defined(USE_ROCM)
  if (a.dim() == 2 && b_q_weight.dim() == 2 && a.size(1) % 16 == 0 &&
      b_q_weight.size(1) % 16 == 0 &&
      ((a.scalar_type() == torch::kBFloat16 && a.size(0) >= 16) ||
       (a.scalar_type() == torch::kHalf && a.size(0) >= 64))) {
    return gptq_gemm_rdna3_wmma(a, b_q_weight, b_qzeros, b_scales, b_g_idx,
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
  TORCH_CHECK(b_scales.size(0) == groups,
              "b_scales must have same group count as qzeros");
  TORCH_CHECK(b_scales.size(1) == size_n, "b_scales last dim must be N");
  TORCH_CHECK(size_n % 8 == 0, "N must be a multiple of 8 (64-bit atomic CAS)");

  auto opts = torch::TensorOptions().dtype(a.dtype()).device(a.device());
  at::Tensor c = torch::zeros({size_m, size_n}, opts);

  const int* g_idx_ptr = nullptr;
  if (!b_g_idx.device().is_meta() && b_g_idx.numel() > 0) {
    TORCH_CHECK(b_g_idx.scalar_type() == torch::kInt32,
                "b_g_idx must be int32");
    g_idx_ptr = (const int*)b_g_idx.data_ptr();
  }

  if (a.scalar_type() == torch::kHalf) {
    vllm::gptq_rdna3::launch_gemm_q4<half>(
        (const half*)a.data_ptr(), (const uint32_t*)b_q_weight.data_ptr(),
        (const uint32_t*)b_qzeros.data_ptr(), (const half*)b_scales.data_ptr(),
        g_idx_ptr, (half*)c.data_ptr(), size_m, size_n, size_k, groups,
        use_v2_format, stream);
  } else {
    vllm::gptq_rdna3::launch_gemm_q4<vllm::gptq_rdna3::bf16_t>(
        (const vllm::gptq_rdna3::bf16_t*)a.data_ptr(),
        (const uint32_t*)b_q_weight.data_ptr(),
        (const uint32_t*)b_qzeros.data_ptr(),
        (const vllm::gptq_rdna3::bf16_t*)b_scales.data_ptr(), g_idx_ptr,
        (vllm::gptq_rdna3::bf16_t*)c.data_ptr(), size_m, size_n, size_k, groups,
        use_v2_format, stream);
  }

  return c;
#else
  TORCH_CHECK(false, "gptq_gemm_rdna3 is only available on ROCm (gfx11)");
  return a;
#endif
}
