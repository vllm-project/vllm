// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Small device helpers shared by the persistent kernel, its gate sample and
// the exact rescan.
#pragma once

#include <cuda_fp16.h>
#include <cutlass/arch/barrier.h>

#include <cstdint>

#include "constants.h"

namespace fused_mqa_topk {

// Warp roles of both pipelines (persistent kernel and dense rescan): warps 0-7
// are the math warps (two warpgroups of 4 query rows each), warps 8-11 the
// producer / UMMA / drain warps.
constexpr uint32_t kMathThreads = 256, kMathWgThreads = 128;
// Register split: ptxas gives each of the 384 threads 168 registers
// (65536 / 384, rounded down to a multiple of 8); setmaxnreg moves them from
// the producer / UMMA / drain warpgroup to the two math warpgroups.
constexpr uint32_t kMathRegs = 224, kSpecRegs = 168 * 3 - 2 * kMathRegs;

// Named barrier ids (cutlass::arch::NamedBarrier offsets them past the ids it
// reserves). kBarMath / kBarSelect: the 256 math threads; kBarDrain: the math
// threads and the two drain warps; kBarDenseMath: the dense rescan's math
// threads.
constexpr uint32_t kBarDenseMath = 0, kBarMath = 2, kBarDrain = 4,
                   kBarSelect = 5;
constexpr uint32_t kDrainSyncThreads = kMathThreads + 2 * 32;

// Barrier of the 256 math threads (warps 0-7); the compiler barriers keep
// shared-memory accesses on their side of it.
__device__ __forceinline__ void math_wg_sync() {
  asm volatile("" ::: "memory");
  cutlass::arch::NamedBarrier(kMathThreads, kBarMath).sync();
  asm volatile("" ::: "memory");
}

// Shared-memory histograms pad one word per 32 bins (bank conflicts):
// bin b lives at padded_bin(b), a histogram of n bins takes padded_bins(n).
__host__ __device__ constexpr uint32_t padded_bins(uint32_t n) {
  return n + n / 32;
}
__device__ __forceinline__ uint32_t padded_bin(uint32_t b) {
  return b + b / 32;
}

// Inclusive prefix sum of v over the warp's lanes.
__device__ __forceinline__ uint32_t warp_inclusive_scan(uint32_t v,
                                                        uint32_t lane) {
#pragma unroll
  for (uint32_t d = 1; d < 32; d *= 2) {
    const uint32_t u = __shfl_up_sync(0xffffffffu, v, d);
    if (lane >= d) v += u;
  }
  return v;
}

__device__ __forceinline__ void mbar_inval(const void* b) {
  asm volatile("mbarrier.inval.shared::cta.b64 [%0];" ::"r"(
                   static_cast<uint32_t>(__cvta_generic_to_shared(b)))
               : "memory");
}

// Order-preserving uint32 code of a float (higher code = larger value) and
// its inverse.
__device__ __forceinline__ uint32_t ordered_code(float x) {
  const uint32_t u = __float_as_uint(x);
  return (u & 0x80000000u) ? ~u : (u ^ 0x80000000u);
}
__device__ __forceinline__ float ordered_float(uint32_t code) {
  return __uint_as_float((code & 0x80000000u) ? (code ^ 0x80000000u) : ~code);
}

// Histogram bin of a score: its fp16 value (rounded down) as an
// order-preserving 16-bit code, >> Shift. half_bin_floor is the smallest value
// in bin `b`.
template <uint32_t Shift>
__device__ __forceinline__ uint32_t half_bin(float x) {
  const uint32_t u = __half_as_ushort(__float2half_rd(x));
  return ((u & 0x8000u) ? ((~u) & 65535u) : (u ^ 0x8000u)) >> Shift;
}
template <uint32_t Shift>
__device__ __forceinline__ float half_bin_floor(uint32_t b) {
  const uint32_t o = b << Shift,
                 u = (o & 0x8000u) ? (o ^ 0x8000u) : ((~o) & 65535u);
  return __half2float(__ushort_as_half(u));
}

// Weighted head reduction of one row's 32 head scores s_h in two FFMA2 chains
// seeded with sum_0 / sum_1: sum_h w_h * |s_h| (kWithLinear = false: the
// persistent kernel adds the linear part on the tensor core) or
// sum_h w_h * (s_h + |s_h|) (kWithLinear = true); relu(s) = (s + |s|) / 2.
template <bool kWithLinear>
__device__ __forceinline__ float reduce_heads(const float* s, const float* w,
                                              float2 sum_0 = {},
                                              float2 sum_1 = {}) {
  auto term = [&](uint32_t at, float2 sum) {
    const auto abs_s = make_float2(fabsf(s[at]), fabsf(s[at + 1]));
    const auto x =
        kWithLinear ? __fadd2_rn(make_float2(s[at], s[at + 1]), abs_s) : abs_s;
    return __ffma2_rn(x, make_float2(w[at], w[at + 1]), sum);
  };
#pragma unroll
  for (uint32_t j = 0; j < kNumHeads; j += 4) {
    sum_0 = term(j, sum_0);
    sum_1 = term(j + 2, sum_1);
  }
  const auto sum = __fadd2_rn(sum_0, sum_1);
  return sum.x + sum.y;
}

}  // namespace fused_mqa_topk
