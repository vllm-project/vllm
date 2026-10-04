// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Byte layout of one Octave slot (one token, one KV head). The PyTorch
// reference in vllm/v1/attention/ops/rocm_octave.py is the source of truth.
//
//   [0, OFF_KN)       K RoPE block (dims 0-63), rotated, 4-bit codes
//   [OFF_KN, OFF_V)   K NoPE blocks (dims 64-255), rotated, 3-bit codes
//   [OFF_V, OFF_SC)   V, rotated, VBITS-bit codes
//   [OFF_SC, ...)     fp16 scales: K block 0..3, V
//
// Compact K (KC): K rotated over all 256 dims like V, 3-bit codes for every
// dim (OFF_KN = 0), one K scale; slots pad to 4 bytes instead of 8.
//
// 4-bit codes: dims 8i..8i+7 in one dword, low nibbles = dims 0-3, high
// nibbles = dims 4-7. 3-bit codes: a 2-bit plane (16 dims per dword, dims
// 4k..4k+3 at bit 2k of bytes 0-3) followed by a 1-bit plane (32 dims per
// dword, dims 4k..4k+3 at bit k of bytes 0-3). A code c of a block with
// scale s stands for LUT[c] * s / LUT_ONE, LUT being the int8 Lloyd-Max
// codebook of its width.

#pragma once

#include <cstdint>

namespace octave {

constexpr int D = 256;       // head size
constexpr int R = 64;        // RoPE dims (K block 0)
constexpr int N = D - R;     // NoPE dims (K blocks 1-3)
constexpr int NKB = D / 64;  // K Hadamard blocks
constexpr float LUT_ONE = 64.0f;

template <int VBITS, bool KC = false>
struct Format {
  static constexpr int KR = KC ? 0 : R;  // K dims with 4-bit codes
  static constexpr int KN = D - KR;      // K dims with 3-bit codes
  static constexpr int OFF_KN = KR * 4 / 8;
  static constexpr int OFF_V = OFF_KN + KN * 3 / 8;
  static constexpr int OFF_SC = OFF_V + D * VBITS / 8;
  static constexpr int NUM_KSC = KC ? 1 : NKB;
  static constexpr int NUM_SC = NUM_KSC + 1;
  static constexpr int ALIGN = KC ? 4 : 8;
  static constexpr int SLOT = (OFF_SC + 2 * NUM_SC + ALIGN - 1) / ALIGN * ALIGN;
};

// Format id passed by the host: V bits, plus 16 for compact K.
constexpr int kCompactFlag = 16;

// int8 Lloyd-Max codebooks as perm tables: byte c of the table is LUT[c].
//   3-bit: -127 -79 -45 -14 14 45 79 127
//   4-bit: -126 -95 -74 -58 -43 -30 -18 -6 6 18 30 43 58 74 95 126
constexpr uint32_t LUT3_LO = 0xF2D3B181u, LUT3_HI = 0x7F4F2D0Eu;
constexpr uint32_t LUT4_0 = 0xC6B6A182u, LUT4_1 = 0xFAEEE2D5u;
constexpr uint32_t LUT4_2 = 0x2B1E1206u, LUT4_3 = 0x7E5F4A3Au;

// Four 3-bit codes (one per byte) -> their four int8 codebook values.
__device__ __forceinline__ uint32_t lut3(uint32_t codes) {
  return __builtin_amdgcn_perm(LUT3_HI, LUT3_LO, codes);
}
// Four 4-bit codes (one per byte) -> their four int8 codebook values.
__device__ __forceinline__ uint32_t lut4(uint32_t codes) {
  const uint32_t lo =
      __builtin_amdgcn_perm(LUT4_1, LUT4_0, codes & 0x07070707u);
  const uint32_t hi =
      __builtin_amdgcn_perm(LUT4_3, LUT4_2, codes & 0x07070707u);
  const uint32_t m = ((codes >> 3) & 0x01010101u) * 0xFFu;
  return (hi & m) | (lo & ~m);
}
template <int BITS>
__device__ __forceinline__ uint32_t lut(uint32_t codes) {
  if constexpr (BITS == 4)
    return lut4(codes);
  else
    return lut3(codes);
}

}  // namespace octave
