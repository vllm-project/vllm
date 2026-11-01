// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Octave-W3 MoE on CDNA3 (gfx942) with MFMA 16x16x16 f16 on wave64.
//
// As on RDNA3 (moe_octw3_wmma_rdna3.cu): octw3_rotate_cdna applies the block
// Hadamard (and SiLU-and-mul for the down projection) once per row, then the
// GEMM is gather-A x dequant-B with the 3-bit codes expanded to fp16
// (s * level). gemm_kernel stages both operands in LDS (prefill);
// gemm_dec_kernel expands the codes straight into MFMA fragments (decode).
//
// Weight layouts: w [E, K/32, 3, N] int32 + scales [E, K/32, N] fp16 from
// octw3_repack (moe_octw3_mfma_gemm_cdna*), or the decode layout from
// octw3_dec_pack, which carries the scales (moe_octw3_pk_gemm_cdna, the
// model path).

#include <cstdint>

#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#if defined(__HIPCC__) && (defined(__gfx942__) || defined(__gfx950__))
  #define __HIP__CDNA3__
#endif

namespace octw3_cdna {

constexpr uint32_t LV_LO = 0x4e600cd8u;
constexpr uint32_t LV_HI_POS = 0x403d3a33u;
constexpr uint32_t LV_HI_NEG = 0xc0bdbab3u;

using v4f16 = _Float16 __attribute__((ext_vector_type(4)));
using v8f16 = _Float16 __attribute__((ext_vector_type(8)));
using v4f32 = float __attribute__((ext_vector_type(4)));

__forceinline__ __device__ void octw3_codes(uint32_t x0, uint32_t x1,
                                            uint32_t x2, uint32_t (&y)[8]) {
  constexpr uint32_t M3 = 0x07070707u;
  y[0] = x0 & M3;
  y[1] = (x0 >> 3) & M3;
  y[2] = x1 & M3;
  y[3] = (x1 >> 3) & M3;
  y[4] = x2 & M3;
  y[5] = (x2 >> 3) & M3;
  y[6] = ((x0 >> 6) & 0x03030303u) | ((x1 >> 4) & 0x04040404u);
  y[7] = ((x2 >> 6) & 0x03030303u) | ((x1 >> 5) & 0x04040404u);
}

__forceinline__ __device__ void octw3_lookup(uint32_t y, uint32_t& w01,
                                             uint32_t& w23) {
  const uint32_t lo = __builtin_amdgcn_perm(LV_LO, LV_LO, y);
  const uint32_t hi = __builtin_amdgcn_perm(LV_HI_NEG, LV_HI_POS, y);
  w01 = __builtin_amdgcn_perm(hi, lo, 0x05010400u);
  w23 = __builtin_amdgcn_perm(hi, lo, 0x07030602u);
}

// y[part * QPER + q] with compile-time register indices only.
template <int QPER>
__forceinline__ __device__ void pick_codes(const uint32_t (&y)[8], int part,
                                           uint32_t (&o)[QPER]) {
#pragma unroll
  for (int q = 0; q < QPER; ++q) o[q] = y[q];
#pragma unroll
  for (int p = 1; p < 8 / QPER; ++p)
#pragma unroll
    for (int q = 0; q < QPER; ++q)
      if (part == p) o[q] = y[p * QPER + q];
}

__forceinline__ __device__ half silu_mul(half g, half u) {
  const float gf = __half2float(g);
  return __float2half_rn(
      __half2float(__float2half_rn(gf / (1.0f + __expf(-gf)))) *
      __half2float(u));
}

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

// y[r] = Hb x[r] / sqrt(had), x[r] = silu(gate) * up when silu. Each thread
// owns 8 consecutive columns: Hadamard bits 0-2 in registers, the rest across
// the 32-lane halves of the wave (had <= 256).
__global__ void __launch_bounds__(256)
    rotate_kernel(const half* __restrict__ x, half* __restrict__ y, int rows,
                  int k, int x_stride, bool silu, int had) {
  const int64_t idx = (int64_t)blockIdx.x * 256 + threadIdx.x;
  const int chunks = k / 8;
  if (idx >= (int64_t)rows * chunks) return;
  const int r = idx / chunks;
  const int c = (idx % chunks) * 8;
  const half* xr = x + (int64_t)r * x_stride + c;

  float v[8];
  const uint4 u = *reinterpret_cast<const uint4*>(xr);
  const half* h = reinterpret_cast<const half*>(&u);
  if (silu) {
    const uint4 ug = *reinterpret_cast<const uint4*>(xr + k);
    const half* hu = reinterpret_cast<const half*>(&ug);
#pragma unroll
    for (int i = 0; i < 8; ++i) v[i] = __half2float(silu_mul(h[i], hu[i]));
  } else {
#pragma unroll
    for (int i = 0; i < 8; ++i) v[i] = __half2float(h[i]);
  }

#pragma unroll
  for (int b = 1; b < 8; b <<= 1) {
#pragma unroll
    for (int i = 0; i < 8; ++i)
      if (b < had && !(i & b)) {
        const float p = v[i], q = v[i | b];
        v[i] = p + q;
        v[i | b] = p - q;
      }
  }
  const int lane = threadIdx.x & 31;
  for (int b = 1; b * 8 < had; b <<= 1) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float p = __shfl_xor(v[i], b, 32);
      v[i] = (lane & b) ? p - v[i] : v[i] + p;
    }
  }

  const float rnorm = had > 0 ? rsqrtf((float)had) : 1.0f;
  uint4 o;
  half* ho = reinterpret_cast<half*>(&o);
#pragma unroll
  for (int i = 0; i < 8; ++i) ho[i] = __float2half_rn(v[i] * rnorm);
  *reinterpret_cast<uint4*>(y + (int64_t)r * k + c) = o;
}

// C[out_row, n] (+)= sum_k A[a_row, k] * W_e[k, n] for one BM-token block of
// moe_align_block_size output (all rows share expert e) and BN columns, as in
// the RDNA3 kernel. WM x WN waves of 64 lanes, each owning a (BM/WM) x
// (BN/WN) patch of 16x16 MFMA tiles; KG weight groups (32 k each) per LDS
// stage, NBUF LDS buffers (with NBUF == 1 the register prefetch is the only
// double buffer, at the cost of a second barrier per stage). MFMA 16x16x16
// f16 fragments: lane l holds A[l % 16][k], B[k][l % 16] for its 4 k of the
// step and C[4 * (l / 16) + 0..3][l % 16].
// C is staged through LDS in slices of 16-row tiles sized to fit in the A/B
// buffers, so the epilogue does not raise the LDS footprint.
#if defined(__HIP__CDNA3__) || !defined(__HIP_DEVICE_COMPILE__)

template <int BM, int BN, int WM, int WN, int KG, int NBUF, bool PK = false>
__global__ void __launch_bounds__(WM * WN * 64)
    gemm_kernel(const half* __restrict__ a, half* __restrict__ c,
                const uint32_t* __restrict__ w, const half* __restrict__ scales,
                const float* __restrict__ topk_weights,
                const int32_t* __restrict__ sorted_token_ids,
                const int32_t* __restrict__ expert_ids,
                const int32_t* __restrict__ num_tokens_post_padded,
                int num_valid, int size_n, int size_k, int a_div,
                int output_topk) {
  constexpr int NT = WM * WN * 64;
  constexpr int TM = BM / WM / 16;
  constexpr int TN = BN / WN / 16;
  constexpr int KS = 32 * KG;  // k per stage
  constexpr int KP = KS + 8;   // padded LDS row of one stage
  constexpr int CP = BN + 4;
  constexpr int P = NT / BN;  // threads per weight column
  constexpr int QPER = 8 / P;
  static_assert(NT % BN == 0 && (P == 1 || P == 2 || P == 4));

  const int t = threadIdx.x;
  const int wave = t >> 6;
  const int lane = t & 63;
  const int lane_lo = lane & 15;
  const int lane_hi = lane >> 4;
  const int wm = wave % WM;
  const int wn = wave / WM;

  const int m_tile = blockIdx.y;
  const int m_base = m_tile * BM;
  const int n_base = blockIdx.x * BN;
  if (m_base >= num_tokens_post_padded[0]) return;
  const int e = expert_ids[m_tile];
  if (e < 0) return;

  const int stages = size_k / KS;
  // PK: w is the decode layout (octw3_dec_pack), scales are inside it.
  const uint32_t* w_e = w + (int64_t)e * (size_k / 32) * (PK ? 4 : 3) * size_n;
  const half* s_e = PK ? nullptr : scales + (int64_t)e * (size_k / 32) * size_n;

  constexpr int AB_BYTES = NBUF * (BM + BN) * KP * 2;
  // Wave rows staged per epilogue slice: as many as fit in the A/B buffers.
  constexpr int WS = [] {
    int ws = WM;
    while (ws > 1 && ws * 16 * CP * 4 > AB_BYTES) ws /= 2;
    return ws;
  }();
  constexpr int C_BYTES = WS * 16 * CP * 4;
  constexpr int SMEM = AB_BYTES > C_BYTES ? AB_BYTES : C_BYTES;
  static_assert(SMEM + BM * 4 <= 65536, "LDS budget");
  __shared__ __attribute__((aligned(16))) char smem[SMEM];
  half(*sa)[BM * KP] = reinterpret_cast<half(*)[BM * KP]>(smem);
  half(*sb)[BN * KP] =
      reinterpret_cast<half(*)[BN * KP]>(smem + NBUF * BM * KP * 2);
  __shared__ int32_t s_tok[BM];

  for (int i = t; i < BM; i += NT) s_tok[i] = sorted_token_ids[m_base + i];
  __syncthreads();

  constexpr int A_ROW_CHUNKS = KS / 8;
  constexpr int A_CHUNKS = BM * A_ROW_CHUNKS;
  constexpr int A_PER_T = (A_CHUNKS + NT - 1) / NT;
  const half* a_src[A_PER_T];
  int a_dst[A_PER_T];
#pragma unroll
  for (int i = 0; i < A_PER_T; ++i) {
    const int idx = t + i * NT;
    const int row = idx / A_ROW_CHUNKS;
    const int col = (idx % A_ROW_CHUNKS) * 8;
    a_src[i] = nullptr;
    a_dst[i] = -1;
    if (idx < A_CHUNKS) {
      a_dst[i] = row * KP + col;
      const int tok = s_tok[row];
      if (tok < num_valid)
        a_src[i] = a + (int64_t)(tok / a_div) * size_k + col;
    }
  }

  const int bcol = t % BN;
  const int bpart = t / BN;
  const int n_glob = n_base + bcol;
  const bool bvalid = n_glob < size_n;

  uint4 ra[A_PER_T];
  uint32_t rw[KG][3];
  half rs[KG];

  auto gload = [&](int st) {
#pragma unroll
    for (int i = 0; i < A_PER_T; ++i) {
      ra[i] = make_uint4(0, 0, 0, 0);
      if (a_src[i])
        ra[i] = *reinterpret_cast<const uint4*>(a_src[i] + st * KS);
    }
#pragma unroll
    for (int kg = 0; kg < KG; ++kg) {
      const int g = st * KG + kg;
      if (bvalid && PK) {
        const uint4 q = reinterpret_cast<const uint4*>(w_e)
            [((int64_t)(n_glob >> 4) * (size_k / 128) + (g >> 2)) * 64 +
             (g & 3) * 16 + (n_glob & 15)];
        rw[kg][0] = q.x;
        rw[kg][1] = q.y;
        rw[kg][2] = q.z;
        rs[kg] = __builtin_bit_cast(half, (unsigned short)(q.w & 0xffffu));
      } else if (bvalid) {
        const uint32_t* wp = w_e + (int64_t)g * 3 * size_n + n_glob;
        rw[kg][0] = wp[0];
        rw[kg][1] = wp[size_n];
        rw[kg][2] = wp[2 * size_n];
        rs[kg] = s_e[(int64_t)g * size_n + n_glob];
      } else {
        rw[kg][0] = rw[kg][1] = rw[kg][2] = 0;
        rs[kg] = __float2half_rn(0.0f);
      }
    }
  };
  auto lstore = [&](int buf) {
#pragma unroll
    for (int i = 0; i < A_PER_T; ++i)
      if (a_dst[i] >= 0) *reinterpret_cast<uint4*>(&sa[buf][a_dst[i]]) = ra[i];
#pragma unroll
    for (int kg = 0; kg < KG; ++kg) {
      uint32_t y[8];
      octw3_codes(rw[kg][0], rw[kg][1], rw[kg][2], y);
      uint32_t ys[QPER];
      pick_codes<QPER>(y, bpart, ys);
      const half2 s2 = __half2half2(rs[kg]);
      uint32_t wq[QPER * 2];
#pragma unroll
      for (int q = 0; q < QPER; ++q) {
        uint32_t w01, w23;
        octw3_lookup(ys[q], w01, w23);
        wq[2 * q] = __builtin_bit_cast(
            uint32_t, __hmul2(__builtin_bit_cast(half2, w01), s2));
        wq[2 * q + 1] = __builtin_bit_cast(
            uint32_t, __hmul2(__builtin_bit_cast(half2, w23), s2));
      }
      uint4* dst = reinterpret_cast<uint4*>(
          &sb[buf][bcol * KP + kg * 32 + bpart * QPER * 4]);
#pragma unroll
      for (int i = 0; i < QPER / 2; ++i)
        dst[i] =
            make_uint4(wq[4 * i], wq[4 * i + 1], wq[4 * i + 2], wq[4 * i + 3]);
    }
  };

  v4f32 acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j)
#pragma unroll
      for (int r = 0; r < 4; ++r) acc[i][j][r] = 0.0f;

  gload(0);
  lstore(0);
  __syncthreads();
  for (int st = 0; st < stages; ++st) {
    const int buf = NBUF == 2 ? st & 1 : 0;
    if (st + 1 < stages) gload(st + 1);
    // Within each 32-k group lane_hi owns k = 8 * lane_hi + 0..7 for both
    // MFMA k-steps (the same permutation on A and B), so one 128-bit LDS
    // read feeds two MFMAs.
#pragma unroll
    for (int kg = 0; kg < KG; ++kg) {
      v8f16 af[TM], bf[TN];
#pragma unroll
      for (int i = 0; i < TM; ++i)
        af[i] = *reinterpret_cast<const v8f16*>(
            &sa[buf][((wm * TM + i) * 16 + lane_lo) * KP + kg * 32 +
                     lane_hi * 8]);
#pragma unroll
      for (int j = 0; j < TN; ++j)
        bf[j] = *reinterpret_cast<const v8f16*>(
            &sb[buf][((wn * TN + j) * 16 + lane_lo) * KP + kg * 32 +
                     lane_hi * 8]);
#pragma unroll
      for (int h = 0; h < 2; ++h)
#pragma unroll
        for (int i = 0; i < TM; ++i)
#pragma unroll
          for (int j = 0; j < TN; ++j) {
            const v4f16 a4 = h ? af[i].hi : af[i].lo;
            const v4f16 b4 = h ? bf[j].hi : bf[j].lo;
            acc[i][j] = __builtin_amdgcn_mfma_f32_16x16x16f16(a4, b4,
                                                              acc[i][j], 0, 0, 0);
          }
    }
    if (st + 1 < stages) {
      if constexpr (NBUF == 1) __syncthreads();
      lstore(NBUF == 2 ? buf ^ 1 : 0);
    }
    __syncthreads();
  }

  float* sc = reinterpret_cast<float*>(smem);
  constexpr int SLICE = WS * 16 * BN / 4;
#pragma unroll
  for (int sl = 0; sl < TM * (WM / WS); ++sl) {
    const int i = sl / (WM / WS);
    const int wg = sl % (WM / WS);
    if (sl > 0) __syncthreads();
    if (wm / WS == wg) {
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        const int col = (wn * TN + j) * 16 + lane_lo;
#pragma unroll
        for (int r = 0; r < 4; ++r)
          sc[((wm % WS) * 16 + lane_hi * 4 + r) * CP + col] = acc[i][j][r];
      }
    }
    __syncthreads();
    for (int idx = t; idx < SLICE; idx += NT) {
      const int lr = idx / (BN / 4);
      const int col = (idx % (BN / 4)) * 4;
      const int row = ((wg * WS + (lr >> 4)) * TM + i) * 16 + (lr & 15);
      const int tok = s_tok[row];
      if (tok >= num_valid || n_base + col >= size_n) continue;
      const float* v = &sc[lr * CP + col];
      float tw = 1.0f;
      if (topk_weights != nullptr) tw = topk_weights[tok];
      const half2 v01 = __floats2half2_rn(v[0] * tw, v[1] * tw);
      const half2 v23 = __floats2half2_rn(v[2] * tw, v[3] * tw);
      if (output_topk > 0) {
        atomic_add_pk4_f16(
            c + (int64_t)(tok / output_topk) * size_n + n_base + col, v01,
            v23);
      } else {
        uint2 u;
        u.x = __builtin_bit_cast(uint32_t, v01);
        u.y = __builtin_bit_cast(uint32_t, v23);
        *reinterpret_cast<uint2*>(c + (int64_t)tok * size_n + n_base + col) =
            u;
      }
    }
  }
}

// Decode layout wd [E, N/16, K/128, 4, 16, 4] int32: for 16 columns and 4
// consecutive 32-k groups, lane 16 * g + c holds the 3 code words of column c
// in group g plus its fp16 scale (low half of the 4th word), so a wave reads
// 1 KB with one 16-byte load per lane. Lane l works on group l / 16 of every
// 4-group step with all 32 of its k: the MFMA k slots of the step are a
// permutation of the 128 k, the same for A and B.
template <int CT, int W, bool AL = false>
__global__ void __launch_bounds__(W * 64)
    gemm_dec_kernel(const half* __restrict__ a, half* __restrict__ c,
                     const uint4* __restrict__ wd,
                     const float* __restrict__ topk_weights,
                     const int32_t* __restrict__ sorted_token_ids,
                     const int32_t* __restrict__ expert_ids,
                     const int32_t* __restrict__ num_tokens_post_padded,
                     int num_valid, int size_n, int size_k, int a_div,
                     int output_topk) {
  const int lane = threadIdx.x & 63;
  const int wave = threadIdx.x >> 6;
  const int lane_lo = lane & 15;
  const int lane_hi = lane >> 4;
  const int m_tile = blockIdx.y;
  if (m_tile * 16 >= num_tokens_post_padded[0]) return;
  const int e = expert_ids[m_tile];
  if (e < 0) return;
  const int t0 = (blockIdx.x * W + wave) * CT;  // first 16-column tile
  const int tiles = size_n / 16;
  const bool active = t0 < tiles;
  if (!AL && !active) return;

  const int tok_a = sorted_token_ids[m_tile * 16 + lane_lo];
  const half* a_row = tok_a < num_valid
                          ? a + (int64_t)(tok_a / a_div) * size_k + lane_hi * 32
                          : nullptr;
  const int steps = size_k / 128;
  const uint4* w_e = wd + (int64_t)e * tiles * steps * 64 + lane;

  v4f32 acc[CT];
#pragma unroll
  for (int j = 0; j < CT; ++j)
#pragma unroll
    for (int r = 0; r < 4; ++r) acc[j][r] = 0.0f;

  // AL: the block's 16 x 128 A slice of each step goes through LDS once
  // (one 16-byte load per thread) instead of once per wave.
  constexpr int AKP = 128 + 8;
  __shared__ __attribute__((aligned(16))) half sa[AL ? 2 : 1][AL ? 16 * AKP : 8];
  const int at = threadIdx.x;
  const half* a_ld = nullptr;
  if (AL && at < 256) {
    const int tok = sorted_token_ids[m_tile * 16 + (at >> 4)];
    if (tok < num_valid)
      a_ld = a + (int64_t)(tok / a_div) * size_k + (at & 15) * 8;
  }
  uint4 a_reg = make_uint4(0, 0, 0, 0);
  auto a_store = [&](int buf) {
    if (AL && at < 256)
      *reinterpret_cast<uint4*>(&sa[buf][(at >> 4) * AKP + (at & 15) * 8]) =
          a_reg;
  };
  if constexpr (AL) {
    if (a_ld) a_reg = *reinterpret_cast<const uint4*>(a_ld);
    a_store(0);
    __syncthreads();
  }

  for (int st = 0; st < steps; ++st) {
    v8f16 ra[4];
    if constexpr (AL) {
      if (a_ld && st + 1 < steps)
        a_reg = *reinterpret_cast<const uint4*>(a_ld + (st + 1) * 128);
#pragma unroll
      for (int i = 0; i < 4; ++i)
        ra[i] = *reinterpret_cast<const v8f16*>(
            &sa[st & 1][lane_lo * AKP + lane_hi * 32 + i * 8]);
    } else {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        if (a_row) {
          ra[i] = *reinterpret_cast<const v8f16*>(a_row + st * 128 + i * 8);
        } else {
#pragma unroll
          for (int x = 0; x < 8; ++x) ra[i][x] = 0;
        }
      }
    }
    uint4 q[CT];
#pragma unroll
    for (int j = 0; j < CT; ++j) {
      q[j] = make_uint4(0, 0, 0, 0);
      if (active && t0 + j < tiles)
        q[j] = w_e[((int64_t)(t0 + j) * steps + st) * 64];
    }
#pragma unroll
    for (int j = 0; j < CT; ++j) {
      uint32_t y[8];
      octw3_codes(q[j].x, q[j].y, q[j].z, y);
      const half2 s2 = __half2half2(
          __builtin_bit_cast(half, (unsigned short)(q[j].w & 0xffffu)));
#pragma unroll
      for (int k4 = 0; k4 < 8; ++k4) {
        uint32_t w01, w23;
        octw3_lookup(y[k4], w01, w23);
        v4f16 b4;
        const half2 h01 = __hmul2(__builtin_bit_cast(half2, w01), s2);
        const half2 h23 = __hmul2(__builtin_bit_cast(half2, w23), s2);
        b4 = __builtin_bit_cast(
            v4f16, make_uint2(__builtin_bit_cast(uint32_t, h01),
                              __builtin_bit_cast(uint32_t, h23)));
        const v4f16 a4 = (k4 & 1) ? ra[k4 >> 1].hi : ra[k4 >> 1].lo;
        acc[j] = __builtin_amdgcn_mfma_f32_16x16x16f16(a4, b4, acc[j], 0, 0, 0);
      }
    }
    if constexpr (AL) {
      if (st + 1 < steps) {
        a_store((st + 1) & 1);
        __syncthreads();
      }
    }
  }
  if (!active) return;

  const int orow = lane >> 2;
  const int ocol = (lane & 3) * 4;
  const int tok = __shfl(tok_a, orow);
  float tw = 1.0f;
  if (topk_weights != nullptr && tok < num_valid) tw = topk_weights[tok];
#pragma unroll
  for (int j = 0; j < CT; ++j) {
    float v[4];
#pragma unroll
    for (int qq = 0; qq < 4; ++qq) {
      const int src = (orow >> 2) * 16 + ocol + qq;
      float t[4];
#pragma unroll
      for (int r = 0; r < 4; ++r) t[r] = __shfl(acc[j][r], src);
      const int sel = orow & 3;
      v[qq] = sel == 0 ? t[0] : sel == 1 ? t[1] : sel == 2 ? t[2] : t[3];
    }
    const int n = (t0 + j) * 16 + ocol;
    if (tok >= num_valid || t0 + j >= tiles) continue;
    const half2 v01 = __floats2half2_rn(v[0] * tw, v[1] * tw);
    const half2 v23 = __floats2half2_rn(v[2] * tw, v[3] * tw);
    if (output_topk > 0) {
      atomic_add_pk4_f16(c + (int64_t)(tok / output_topk) * size_n + n, v01,
                         v23);
    } else {
      uint2 u;
      u.x = __builtin_bit_cast(uint32_t, v01);
      u.y = __builtin_bit_cast(uint32_t, v23);
      *reinterpret_cast<uint2*>(c + (int64_t)tok * size_n + n) = u;
    }
  }
}

#else  // non-CDNA3: empty stub for symbol parity

template <int BM, int BN, int WM, int WN, int KG, int NBUF, bool PK = false>
__global__ void gemm_kernel(const half*, half*, const uint32_t*, const half*,
                            const float*, const int32_t*, const int32_t*,
                            const int32_t*, int, int, int, int, int) {}

template <int CT, int W, bool AL = false>
__global__ void gemm_dec_kernel(const half*, half*, const uint4*,
                                const float*, const int32_t*, const int32_t*,
                                const int32_t*, int, int, int, int, int) {}

#endif  // __HIP__CDNA3__ || !__HIP_DEVICE_COMPILE__

}  // namespace octw3_cdna

void octw3_rotate_cdna(torch::Tensor x, torch::Tensor y, bool silu_mul,
                       int64_t had) {
  TORCH_CHECK(x.scalar_type() == torch::kHalf &&
              y.scalar_type() == torch::kHalf);
  TORCH_CHECK(x.dim() == 2 && y.dim() == 2 && y.is_contiguous() &&
              x.stride(1) == 1 && x.size(0) == y.size(0));
  const int rows = y.size(0);
  const int k = y.size(1);
  TORCH_CHECK(x.size(1) == (silu_mul ? 2 * k : k), "x has the wrong width");
  TORCH_CHECK(k % 8 == 0 && x.stride(0) % 8 == 0);
  TORCH_CHECK(had == 0 || (had >= 8 && had <= 256 && (had & (had - 1)) == 0 &&
                           (k / 8) % (had / 8) == 0));
  if (rows == 0) return;
  const int64_t n = (int64_t)rows * (k / 8);
  hipLaunchKernelGGL(octw3_cdna::rotate_kernel, dim3((n + 255) / 256),
                     dim3(256), 0, at::hip::getCurrentHIPStream(),
                     (const half*)x.data_ptr(), (half*)y.data_ptr(), rows, k,
                     (int)x.stride(0), silu_mul, (int)had);
}

// Tile configs: {BM, BN, WM, WN, KG, NBUF}. BM must equal
// moe_align_block_size.
#define OCTW3_MFMA_CONFIGS(X)  \
  X(0, 32, 128, 2, 2, 1, 2)    \
  X(1, 64, 128, 2, 2, 1, 2)    \
  X(2, 64, 256, 2, 4, 1, 2)    \
  X(3, 128, 128, 4, 2, 1, 2)   \
  X(4, 64, 128, 2, 2, 1, 1)    \
  X(5, 64, 256, 2, 4, 1, 1)    \
  X(6, 128, 128, 4, 2, 1, 1)   \
  X(7, 128, 128, 2, 4, 1, 1)   \
  X(8, 128, 256, 4, 2, 1, 1)   \
  X(9, 64, 128, 2, 2, 2, 1)    \
  X(10, 64, 256, 2, 4, 2, 1)   \
  X(11, 32, 128, 2, 2, 1, 1)

int64_t moe_octw3_mfma_cfg_block_m(int64_t cfg) {
#define OCTW3_BM(ID, BM, BN, WM, WN, KG, NBUF) \
  if (cfg == ID) return BM;
  OCTW3_MFMA_CONFIGS(OCTW3_BM)
#undef OCTW3_BM
  TORCH_CHECK(false, "unknown octw3 mfma config ", cfg);
}

void moe_octw3_mfma_gemm_cdna_cfg(torch::Tensor a, torch::Tensor c,
                                  torch::Tensor w, torch::Tensor scales,
                                  torch::Tensor topk_weights,
                                  torch::Tensor sorted_token_ids,
                                  torch::Tensor expert_ids,
                                  torch::Tensor num_tokens_post_padded,
                                  int64_t a_div, int64_t output_topk,
                                  int64_t cfg) {
  TORCH_CHECK(a.scalar_type() == torch::kHalf &&
              c.scalar_type() == torch::kHalf &&
              scales.scalar_type() == torch::kHalf);
  TORCH_CHECK(a.dim() == 2 && a.is_contiguous() && c.dim() == 2 &&
              c.is_contiguous());
  TORCH_CHECK(w.dim() == 4 && w.size(2) == 3, "w must be [E, K/32, 3, N]");
  const int size_k = (int)w.size(1) * 32;
  const int size_n = (int)w.size(3);
  TORCH_CHECK(a.size(1) == size_k && c.size(1) == size_n && size_n % 4 == 0);
  const int num_valid = (int)(a.size(0) * a_div);
  const int64_t block_m = moe_octw3_mfma_cfg_block_m(cfg);
  const int num_m_tiles = (int)(sorted_token_ids.size(0) / block_m);
  const float* tw =
      topk_weights.numel() > 0 ? topk_weights.data_ptr<float>() : nullptr;
  auto stream = at::hip::getCurrentHIPStream();

#define OCTW3_MFMA_LAUNCH(ID, BM, BN, WM, WN, KG, NBUF)                           \
  if (cfg == ID) {                                                            \
    TORCH_CHECK(size_k % (32 * KG) == 0, "K must be a multiple of ", 32 * KG); \
    hipLaunchKernelGGL((octw3_cdna::gemm_kernel<BM, BN, WM, WN, KG, NBUF>),       \
                       dim3((size_n + BN - 1) / BN, num_m_tiles),             \
                       dim3(WM * WN * 64), 0, stream,                         \
                       (const half*)a.data_ptr(), (half*)c.data_ptr(),        \
                       (const uint32_t*)w.data_ptr<int32_t>(),                \
                       (const half*)scales.data_ptr(), tw,                    \
                       sorted_token_ids.data_ptr<int32_t>(),                  \
                       expert_ids.data_ptr<int32_t>(),                        \
                       num_tokens_post_padded.data_ptr<int32_t>(), num_valid, \
                       size_n, size_k, (int)a_div, (int)output_topk);         \
    return;                                                                   \
  }
  OCTW3_MFMA_CONFIGS(OCTW3_MFMA_LAUNCH)
#undef OCTW3_MFMA_LAUNCH
}

void moe_octw3_mfma_gemm_cdna(torch::Tensor a, torch::Tensor c,
                              torch::Tensor w, torch::Tensor scales,
                              torch::Tensor topk_weights,
                              torch::Tensor sorted_token_ids,
                              torch::Tensor expert_ids,
                              torch::Tensor num_tokens_post_padded,
                              int64_t a_div, int64_t output_topk,
                              int64_t block_size_m) {
  int64_t cfg;
  switch (block_size_m) {
    case 32:
      cfg = 11;
      break;
    case 64:
      cfg = 5;
      break;
    case 128:
      cfg = 6;
      break;
    default:
      TORCH_CHECK(false, "block_size_m must be 32, 64 or 128, got ",
                  block_size_m);
  }
  moe_octw3_mfma_gemm_cdna_cfg(a, c, w, scales, topk_weights, sorted_token_ids,
                               expert_ids, num_tokens_post_padded, a_div,
                               output_topk, cfg);
}

// The model path: w is octw3_dec_pack output viewed as [E, K/32, 4, N].
// block_size_m 16 runs the decode kernel, 32/64/128 the LDS-tiled GEMM.
void moe_octw3_pk_gemm_cdna(torch::Tensor a, torch::Tensor c, torch::Tensor w,
                            torch::Tensor topk_weights,
                            torch::Tensor sorted_token_ids,
                            torch::Tensor expert_ids,
                            torch::Tensor num_tokens_post_padded,
                            int64_t a_div, int64_t output_topk,
                            int64_t block_size_m) {
  TORCH_CHECK(a.scalar_type() == torch::kHalf && c.scalar_type() == torch::kHalf);
  TORCH_CHECK(a.dim() == 2 && a.is_contiguous() && c.dim() == 2 &&
              c.is_contiguous());
  TORCH_CHECK(w.dim() == 4 && w.size(2) == 4 && w.is_contiguous() &&
                  w.scalar_type() == torch::kInt32,
              "w must be the packed layout viewed as [E, K/32, 4, N]");
  const int size_k = (int)w.size(1) * 32;
  const int size_n = (int)w.size(3);
  TORCH_CHECK(a.size(1) == size_k && c.size(1) == size_n &&
              size_k % 128 == 0 && size_n % 16 == 0);
  const int num_valid = (int)(a.size(0) * a_div);
  const int num_m_tiles = (int)(sorted_token_ids.size(0) / block_size_m);
  const float* tw =
      topk_weights.numel() > 0 ? topk_weights.data_ptr<float>() : nullptr;
  auto stream = at::hip::getCurrentHIPStream();
  const uint32_t* wp = (const uint32_t*)w.data_ptr<int32_t>();
  if (block_size_m == 16) {
    constexpr int CT = 2, W = 4;
    hipLaunchKernelGGL((octw3_cdna::gemm_dec_kernel<CT, W, true>),
                       dim3((size_n / 16 + CT * W - 1) / (CT * W), num_m_tiles),
                       dim3(W * 64), 0, stream, (const half*)a.data_ptr(),
                       (half*)c.data_ptr(), (const uint4*)wp, tw,
                       sorted_token_ids.data_ptr<int32_t>(),
                       expert_ids.data_ptr<int32_t>(),
                       num_tokens_post_padded.data_ptr<int32_t>(), num_valid,
                       size_n, size_k, (int)a_div, (int)output_topk);
    return;
  }
#define OCTW3_PK_LAUNCH(BMV, BM, BN, WM, WN)                                   \
  if (block_size_m == BMV) {                                                  \
    hipLaunchKernelGGL((octw3_cdna::gemm_kernel<BM, BN, WM, WN, 1, 1, true>), \
                       dim3((size_n + BN - 1) / BN, num_m_tiles),             \
                       dim3(WM * WN * 64), 0, stream,                         \
                       (const half*)a.data_ptr(), (half*)c.data_ptr(), wp,    \
                       (const half*)nullptr, tw,                              \
                       sorted_token_ids.data_ptr<int32_t>(),                  \
                       expert_ids.data_ptr<int32_t>(),                        \
                       num_tokens_post_padded.data_ptr<int32_t>(), num_valid, \
                       size_n, size_k, (int)a_div, (int)output_topk);         \
    return;                                                                   \
  }
  OCTW3_PK_LAUNCH(32, 32, 128, 2, 2)
  OCTW3_PK_LAUNCH(64, 64, 256, 2, 4)
  OCTW3_PK_LAUNCH(128, 128, 128, 4, 2)
#undef OCTW3_PK_LAUNCH
  TORCH_CHECK(false, "block_size_m must be 16, 32, 64 or 128, got ",
              block_size_m);
}
