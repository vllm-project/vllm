// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include "common.cuh"

// SM120 MMA instruction wrappers.
//
// Standard (no scale):
//   FP8:  mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32
//   BF16: mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
//
// Block-scaled (UE8M0 scale applied in hardware, zero overhead):
//   FP8:  mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32
//         .row.col.f32.e4m3.e4m3.f32.ue8m0

struct MmaFp8Result {
  float d0, d1, d2, d3;
};
struct MmaBf16Result {
  float d0, d1, d2, d3;
};

__device__ __forceinline__ float multiply_ue8m0(uint32_t a, uint32_t b) {
  const int biased_exp = static_cast<int>(a) + static_cast<int>(b) - 127;
  if (__builtin_expect(a != 0xff && b != 0xff && biased_exp > 0 && biased_exp < 255, 1)) {
    return __uint_as_float(static_cast<uint32_t>(biased_exp) << 23);
  }
  if (a == 0xff || b == 0xff) return __uint_as_float(0x7fc00000);
  if (biased_exp >= 255) return __uint_as_float(0x7f800000);
  if (biased_exp < -22) return 0.f;
  return __uint_as_float(1u << (biased_exp + 22));
}

__device__ __forceinline__ MmaFp8Result mma_fp8_m16n8k32(uint32_t a0, uint32_t a1, uint32_t a2,
                                                         uint32_t a3, uint32_t b0, uint32_t b1,
                                                         float c0, float c1, float c2, float c3) {
  MmaFp8Result r;
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(r.d0), "=f"(r.d1), "=f"(r.d2), "=f"(r.d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c0), "f"(c1), "f"(c2), "f"(c3));
  return r;
}

__device__ __forceinline__ MmaFp8Result mma_fp8_block_scaled_m16n8k32(
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1, float c0,
    float c1, float c2, float c3, uint8_t scale_a, uint8_t scale_b) {
#if SPARSE_MLA_USE_SM89_PRIMS
  MmaFp8Result mma = mma_fp8_m16n8k32(a0, a1, a2, a3, b0, b1, 0.f, 0.f, 0.f, 0.f);
  const int lane = threadIdx.x & 31;
  const int quad_base = lane & ~3;
  const int col_pair = lane & 3;

  // The SM120 instruction gathers distributed scale-vector operands. Rebuild
  // those per-accumulator scales for Ada's unscaled FP8 MMA: d0/d1 belong to
  // row gid, d2/d3 to row gid+8, while adjacent columns use distinct B scales.
  const uint32_t sa0 = __shfl_sync(0xffffffff, static_cast<uint32_t>(scale_a), quad_base);
  const uint32_t sa1 = __shfl_sync(0xffffffff, static_cast<uint32_t>(scale_a), quad_base + 1);
  const uint32_t sb0 = __shfl_sync(0xffffffff, static_cast<uint32_t>(scale_b), col_pair * 8);
  const uint32_t sb1 = __shfl_sync(0xffffffff, static_cast<uint32_t>(scale_b), col_pair * 8 + 4);

  return MmaFp8Result{
      fmaf(mma.d0, multiply_ue8m0(sa0, sb0), c0), fmaf(mma.d1, multiply_ue8m0(sa0, sb1), c1),
      fmaf(mma.d2, multiply_ue8m0(sa1, sb0), c2), fmaf(mma.d3, multiply_ue8m0(sa1, sb1), c3)};
#else
  MmaFp8Result r;
  asm volatile(
      "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32"
      ".row.col.f32.e4m3.e4m3.f32.ue8m0 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, "
      "{%14}, {%15, %16}, {%17}, {%18, %19};\n"
      : "=f"(r.d0), "=f"(r.d1), "=f"(r.d2), "=f"(r.d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c0), "f"(c1), "f"(c2), "f"(c3),
        "r"(static_cast<uint32_t>(scale_a)), "n"(static_cast<uint16_t>(0)),
        "n"(static_cast<uint16_t>(0)), "r"(static_cast<uint32_t>(scale_b)),
        "n"(static_cast<uint16_t>(0)), "n"(static_cast<uint16_t>(0)));
  return r;
#endif
}

__device__ __forceinline__ MmaBf16Result mma_bf16_m16n8k16(uint32_t a0, uint32_t a1, uint32_t a2,
                                                           uint32_t a3, uint32_t b0, uint32_t b1,
                                                           float c0, float c1, float c2, float c3) {
  MmaBf16Result r;
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(r.d0), "=f"(r.d1), "=f"(r.d2), "=f"(r.d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c0), "f"(c1), "f"(c2), "f"(c3));
  return r;
}
