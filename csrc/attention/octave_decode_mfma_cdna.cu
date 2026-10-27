// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Octave decode on MFMA for CDNA3 (gfx942). Same scheme as the WMMA decode
// (octave_decode_wmma_rdna3.cu): independent waves, one (split, 16-query row
// tile) per wave, transposed orientation, codebook bytes fed to fp16 as
// x + 1152 and debiased afterwards. With 16x16x16 MFMA on wave64 (lane l
// holds column l % 16 and rows 4 (l / 16) .. + 3 of the result):
//
//   QK  S^T[token][q] = K x Q^T. A lane owns one token (l % 16) and, in the
//       four k-steps of each 64-dim block, the 16 contiguous dims
//       16 (l / 16) .. + 15 of that block, so its codes are one contiguous
//       run of the slot. One accumulator per K scale.
//   PV  O^T[dim][q] = V^T x P^T. Output row r of fragment c is dim 16 r + c,
//       so a lane reads the 16 contiguous dims 16 (l % 16) .. + 15 of tokens
//       4 (l / 16) .. + 3. P^T is the S^T accumulator, one query per lane.

#include <algorithm>
#include <cstdint>
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>

#if defined(USE_ROCM)
  #include <hip/hip_runtime.h>
  #include <hip/hip_bf16.h>
  #include <hip/hip_fp16.h>

  #include "octave_format.cuh"

  #if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx942__)
    #define OCTAVE_MFMA_STUB
  #endif

namespace octave_mfma {

typedef _Float16 h4v __attribute__((ext_vector_type(4)));
typedef _Float16 h8v __attribute__((ext_vector_type(8)));
typedef _Float16 h2v __attribute__((ext_vector_type(2)));
typedef float f4v __attribute__((ext_vector_type(4)));
typedef int i4v __attribute__((ext_vector_type(4)));
typedef const int __attribute__((address_space(4))) cint;

using octave::D;
using octave::Format;
using octave::LUT_ONE;
[[maybe_unused]] constexpr int QROW = D + 8;
[[maybe_unused]] constexpr float PSCALE = 256.0f;
[[maybe_unused]] constexpr float BIAS = 1152.0f;  // byte x fed as x + 1152
[[maybe_unused]] constexpr uint32_t X = 0x80808080u;
// Bytes of a 16-token tile (slot <= 256).
[[maybe_unused]] constexpr int TILE_MAX = 4096;

// Bytes b of a dword -> fp16 1024 + b, exact.
__device__ __forceinline__ h4v bytes_to_h4(uint32_t w) {
  const uint32_t lo = __builtin_amdgcn_perm(0x64646464u, w, 0x07010700u);
  const uint32_t hi = __builtin_amdgcn_perm(0x64646464u, w, 0x07030702u);
  return __builtin_bit_cast(h4v, (uint2){lo, hi});
}

template <bool BIASED>
__device__ __forceinline__ uint32_t lut4_fast(uint32_t codes) {
  constexpr uint32_t B = BIASED ? X : 0u;
  const uint32_t lo = __builtin_amdgcn_perm(
      octave::LUT4_1 ^ B, octave::LUT4_0 ^ B, codes & 0x07070707u);
  const uint32_t hi = __builtin_amdgcn_perm(
      octave::LUT4_3 ^ B, octave::LUT4_2 ^ B, codes & 0x07070707u);
  const uint32_t t = codes & 0x08080808u;
  const uint32_t m = (t << 5) - (t >> 3);
  return (hi & m) | (lo & ~m);
}

// 4 dwords of 4 tokens each (same 4 dims) -> 4 dwords of 4 dims each.
__device__ __forceinline__ void tr4(uint32_t a0, uint32_t a1, uint32_t a2,
                                    uint32_t a3, uint32_t o[4]) {
  const uint32_t lo01 = __builtin_amdgcn_perm(a1, a0, 0x05010400u);
  const uint32_t hi01 = __builtin_amdgcn_perm(a1, a0, 0x07030602u);
  const uint32_t lo23 = __builtin_amdgcn_perm(a3, a2, 0x05010400u);
  const uint32_t hi23 = __builtin_amdgcn_perm(a3, a2, 0x07030602u);
  o[0] = __builtin_amdgcn_perm(lo23, lo01, 0x05040100u);
  o[1] = __builtin_amdgcn_perm(lo23, lo01, 0x07060302u);
  o[2] = __builtin_amdgcn_perm(hi23, hi01, 0x05040100u);
  o[3] = __builtin_amdgcn_perm(hi23, hi01, 0x07060302u);
}

template <typename T>
__device__ __forceinline__ float to_f(T x) {
  return (float)x;
}
template <>
__device__ __forceinline__ float to_f<half>(half x) {
  return __half2float(x);
}

__device__ __forceinline__ float sign_of(const int* bits, int i) {
  return ((bits[i >> 5] >> (i & 31)) & 1) ? -1.0f : 1.0f;
}
__device__ __forceinline__ float lo16(uint32_t x) {
  return __half2float(__ushort_as_half((unsigned short)(x & 0xFFFF)));
}
__device__ __forceinline__ float hi16(uint32_t x) {
  return __half2float(__ushort_as_half((unsigned short)(x >> 16)));
}
// Over the four lanes of a query (l, l ^ 16, l ^ 32, l ^ 48).
__device__ __forceinline__ float qmax4(float v) {
  v = fmaxf(v, __shfl_xor(v, 16));
  return fmaxf(v, __shfl_xor(v, 32));
}
__device__ __forceinline__ float qsum4(float v) {
  v += __shfl_xor(v, 16);
  return v + __shfl_xor(v, 32);
}

// QG: query tokens per block (4 for multi-token verification, else 1).
template <int VB, bool KC, int NW, int QG, typename QT>
__global__ __launch_bounds__(64 * NW) void decode_mfma(
    const QT* __restrict__ Q, const uint8_t* __restrict__ cache,
    const int* __restrict__ block_table, const int* __restrict__ q_to_req,
    const int* __restrict__ q_to_klen, float* __restrict__ mid_o,
    const int* __restrict__ k_signs, float sm_scale, int num_q, int num_q_heads,
    int num_kv_heads, int block_size, int max_blocks, int num_reqs,
    int num_phys_blocks, int num_splits, int min_tps, int num_row_tiles,
    int64_t sq0, int64_t sq1, int64_t scb, int64_t sch, int64_t sct,
    int64_t smo, int64_t smh, int64_t sms) {
  #ifndef OCTAVE_MFMA_STUB
  using F = Format<VB, KC>;
  constexpr int NG = KC ? 1 : 4;  // QK accumulators (one per K scale)
  const int grp = blockIdx.x;
  const int kvh = blockIdx.y;
  const int tid = threadIdx.x;
  const int w = __builtin_amdgcn_readfirstlane(tid >> 6), lane = tid & 63;
  const int col = lane & 15, gq = lane >> 4;
  const int rt = num_row_tiles == 2 ? (w & 1) : 0;
  const int splits_per_block = NW / num_row_tiles;
  const int si =
      blockIdx.z * splits_per_block + (num_row_tiles == 2 ? w >> 1 : w);
  const int hpk = num_q_heads / num_kv_heads;
  const int max_kv = max_blocks * block_size;
  const float sm_log2 = sm_scale * 1.4426950408889634f / LUT_ONE;

  // The staged queries, then (once they are in registers) one 4 KB tile
  // buffer per wave.
  constexpr int SMEM =
      32 * QROW * 2 > NW * TILE_MAX ? 32 * QROW * 2 : NW * TILE_MAX;
  __shared__ __attribute__((aligned(16))) uint8_t smem[SMEM];
  auto sQ = reinterpret_cast<_Float16 (*)[QROW]>(smem);

  // Stage the group's queries rotated like K: each 32-lane half owns one
  // query vector, 8 dims per lane.
  {
    const int l32 = lane & 31;
    for (int v = 2 * w + (lane >> 5); v < 32; v += 2 * NW) {
      const int r = v / hpk, g = v % hpk;
      const int qi = grp * QG + r;
      float x[8];
      const bool live = v < QG * hpk && qi < num_q;
    #pragma unroll
      for (int i = 0; i < 8; ++i)
        x[i] = live ? to_f<QT>(Q[(int64_t)qi * sq0 +
                                 (int64_t)(kvh * hpk + g) * sq1 + l32 * 8 + i])
                    : 0.0f;
    #pragma unroll
      for (int i = 0; i < 8; ++i) x[i] *= sign_of(k_signs, l32 * 8 + i);
    #pragma unroll
      for (int hh = 1; hh < 8; hh <<= 1) {
    #pragma unroll
        for (int i = 0; i < 8; ++i)
          if ((i & hh) == 0) {
            const float a = x[i], b = x[i + hh];
            x[i] = a + b;
            x[i + hh] = a - b;
          }
      }
    #pragma unroll
      for (int m = 1; m <= (KC ? 16 : 4); m <<= 1) {
        const bool hi = l32 & m;
    #pragma unroll
        for (int i = 0; i < 8; ++i) {
          const float o = __shfl_xor(x[i], m);
          x[i] = hi ? o - x[i] : x[i] + o;
        }
      }
    #pragma unroll
      for (int i = 0; i < 8; ++i)
        sQ[v][l32 * 8 + i] = (_Float16)(x[i] * (KC ? 0.0625f : 0.125f));
    }
  }
  __syncthreads();

  int rreq[QG], rlen[QG];
    #pragma unroll
  for (int r = 0; r < QG; ++r) {
    const int qi = grp * QG + r;
    int rq = -1, ln = 0;
    if (qi < num_q) {
      rq = q_to_req[qi];
      ln = q_to_klen[qi];
      if (rq < 0 || rq >= num_reqs) {
        rq = 0;
        ln = 0;
      }
      ln = min(max(ln, 0), max_kv);
    }
    rreq[r] = __builtin_amdgcn_readfirstlane(rq);
    rlen[r] = __builtin_amdgcn_readfirstlane(ln);
  }

  // This lane's query (column) as int8 B fragments for the i8 MFMA: step
  // 2 b + h holds dims 64 b + 16 gq + 8 h .. + 7 (the dot product does not
  // care how dims map to k, only that K uses the same map). Two int8 halves,
  // q ~ qsc (hi + lo / 256), keep ~16 bits of the query.
  const int qv = rt * 16 + col;
  long Qh[8], Ql[8];
  float qsc;
  {
    float x[64];
    float qmax = 1e-20f;
    #pragma unroll
    for (int i = 0; i < 64; ++i) {
      x[i] = (float)sQ[qv][64 * (i >> 4) + 16 * gq + (i & 15)];
      qmax = fmaxf(qmax, fabsf(x[i]));
    }
    qmax = qmax4(qmax);
    const float inv = 127.0f / qmax;
    qsc = qmax * (1.0f / 127.0f);
    #pragma unroll
    for (int st = 0; st < 8; ++st) {
      uint64_t wh = 0, wl = 0;
    #pragma unroll
      for (int e = 0; e < 8; ++e) {
        const float y = x[8 * st + e] * inv;
        const float h = rintf(y);
        const int lo = max(-127, min(127, (int)rintf((y - h) * 256.0f)));
        wh |= (uint64_t)(uint8_t)(int8_t)(int)h << (8 * e);
        wl |= (uint64_t)(uint8_t)(int8_t)lo << (8 * e);
      }
      Qh[st] = (long)wh;
      Ql[st] = (long)wl;
    }
  }
  const float sm_q = sm_log2 * qsc * (1.0f / 256.0f);
  __syncthreads();
  uint8_t* lt = smem + w * TILE_MAX;

  // rreq/rlen are only ever indexed by constants (unrolled selections): a
  // dynamic index would put them in scratch memory.
  auto pick = [&](const int* a, int i) {
    int x = a[0];
    #pragma unroll
    for (int r = 1; r < QG; ++r) x = i == r ? a[r] : x;
    return x;
  };
  int seg0 = 0;
  while (seg0 < QG && pick(rreq, seg0) >= 0) {
    const int req = pick(rreq, seg0);
    int seg1 = seg0 + 1;
    while (seg1 < QG && pick(rreq, seg1) == req) ++seg1;
    int Lmax = 0;
    #pragma unroll
    for (int r = 0; r < QG; ++r)
      if (r >= seg0 && r < seg1) Lmax = max(Lmax, rlen[r]);
    const int qv_lo = seg0 * hpk, qv_hi = seg1 * hpk;
    int tps = (Lmax + num_splits - 1) / num_splits;
    if (tps < min_tps) tps = min_tps;
    tps = (tps + 15) & ~15;

    const bool tile_live = qv_hi > rt * 16 && qv_lo < rt * 16 + 16;
    const bool qlive = qv >= qv_lo && qv < qv_hi;
    if (blockIdx.z * splits_per_block * tps >= Lmax) {
      if (tile_live && si < num_splits && gq == 0 && qlive) {
        float* op = mid_o + (int64_t)(grp * QG + qv / hpk) * smo +
                    (int64_t)(kvh * hpk + qv % hpk) * smh + (int64_t)si * sms;
        op[D] = -INFINITY;
        op[D + 1] = 0.0f;
      }
      seg0 = seg1;
      continue;
    }

    if (tile_live && si < num_splits) {
      int qlen = 0;
    #pragma unroll
      for (int r = 0; r < QG; ++r)
        if (qlive && qv / hpk == r) qlen = rlen[r];

      const int start = si * tps;
      const int end = min(start + tps, Lmax);
      float m = -INFINITY, l = 0.0f, cp = 0.0f;
      f4v O[16];
    #pragma unroll
      for (int c = 0; c < 16; ++c) O[c] = (f4v){0, 0, 0, 0};

      const uint8_t* cache_h = cache + (int64_t)kvh * sch;
      auto tile_ptr = [&](int pos) {
        const int lb = pos / block_size;
        int pb = ((cint*)block_table)[req * max_blocks + lb];
        if (pb < 0 || pb >= num_phys_blocks) pb = 0;
        return cache_h + (int64_t)pb * scb +
               (int64_t)(pos - lb * block_size) * sct;
      };

      // Raw K of token col: per 64-dim block, the codes of dims
      // 16 gq .. + 15. 4-bit: 8 bytes. 3-bit: 2-bit plane word gq and the
      // 16 bits of the 1-bit plane. Scales of tokens 4 gq .. + 3.
      constexpr int NB3 = KC ? 4 : 3;  // 3-bit blocks
      uint2 kr = {0, 0};
      uint2 kn[NB3];
      uint32_t sa[4], sb[4], sv[4];
      // A 16-token tile is one contiguous run of a block: each lane fetches
      // four 16-byte chunks of the next tile into registers, then stores
      // them to its wave's LDS buffer, where the fragments are read.
      constexpr int tb = 16 * F::SLOT;
      uint4 raw[TILE_MAX / 1024], raw2[TILE_MAX / 1024];
      auto fetch = [&](uint4* dst, const uint8_t* tile) {
    #pragma unroll
        for (int j = 0; j < TILE_MAX / 1024; ++j) {
          const int o = 16 * (lane + 64 * j);
          dst[j] = o < tb ? *(const uint4*)(tile + o) : make_uint4(0, 0, 0, 0);
        }
      };
      auto stash = [&]() {
    #pragma unroll
        for (int j = 0; j < TILE_MAX / 1024; ++j) {
          const int o = 16 * (lane + 64 * j);
          if (o < tb) *(uint4*)(lt + o) = raw[j];
        }
      };
      auto load_k = [&](const uint8_t* tile) {
        const uint8_t* pk = tile + (int64_t)col * F::SLOT;
        if constexpr (!KC) kr = *(const uint2*)(pk + 8 * gq);
    #pragma unroll
        for (int b = 0; b < NB3; ++b) {
          kn[b].x = *(const uint32_t*)(pk + F::OFF_KN + 16 * b + 4 * gq);
          kn[b].y = *(const uint32_t*)(pk + F::OFF_KN + F::KN / 4 + 8 * b +
                                       4 * (gq >> 1)) >>
                    (4 * (gq & 1));
        }
    #pragma unroll
        for (int e = 0; e < 4; ++e) {
          const uint8_t* ps =
              tile + (int64_t)(4 * gq + e) * F::SLOT + F::OFF_SC;
          if constexpr (KC) {
            sa[e] = *(const uint32_t*)ps;
            sv[e] = sa[e] >> 16;
          } else {
            const uint2 u = *(const uint2*)ps;
            sa[e] = u.x;
            sb[e] = u.y;
            sv[e] = *(const uint16_t*)(ps + 8);
          }
        }
      };
      // Raw V of tokens 4 gq .. + 3, dims 16 col .. + 15.
      uint2 vr[4];
      auto load_v = [&](const uint8_t* tile) {
    #pragma unroll
        for (int e = 0; e < 4; ++e) {
          const uint8_t* p = tile + (int64_t)(4 * gq + e) * F::SLOT + F::OFF_V;
          if constexpr (VB == 4) {
            vr[e] = *(const uint2*)(p + 8 * col);
          } else {
            vr[e].x = *(const uint32_t*)(p + 4 * col);
            vr[e].y = *(const uint32_t*)(p + D / 4 + 4 * (col >> 1)) >>
                      (4 * (col & 1));
          }
        }
      };
      // Codebook bytes of dims 4 s .. + 3 of a 16-dim run: int8 for K, biased
      // (x + 128, the bias folded into the tables) for V.
      auto codes4 = [&](uint2 raw, int s) -> uint32_t {
        const uint32_t wd = s < 2 ? raw.x : raw.y;
        return (wd >> (4 * (s & 1))) & 0x0F0F0F0Fu;
      };
      auto codes3 = [&](uint2 raw, int s) -> uint32_t {
        return ((raw.x >> (2 * s)) & 0x03030303u) |
               (((raw.y >> s) & 0x01010101u) << 2);
      };
      auto k_bytes = [&](int b, int s) -> uint32_t {
        return (!KC && b == 0) ? lut4_fast<false>(codes4(kr, s))
                               : octave::lut3(codes3(kn[KC ? b : b - 1], s));
      };
      auto v_bytes = [&](uint2 raw, int s) -> uint32_t {
        if constexpr (VB == 4)
          return lut4_fast<true>(codes4(raw, s));
        else
          return __builtin_amdgcn_perm(octave::LUT3_HI ^ X, octave::LUT3_LO ^ X,
                                       codes3(raw, s));
      };

      if (start < end) fetch(raw, tile_ptr(start));
      if (start + 16 < end) fetch(raw2, tile_ptr(start + 16));

      for (int base = start; base < end; base += 16) {
        stash();
        load_k(lt);
    #pragma unroll
        for (int j = 0; j < TILE_MAX / 1024; ++j) raw[j] = raw2[j];
        if (base + 32 < end) fetch(raw2, tile_ptr(base + 32));

        i4v Sh[NG], Sl[NG];
    #pragma unroll
        for (int g = 0; g < NG; ++g) Sh[g] = Sl[g] = (i4v){0, 0, 0, 0};
    #pragma unroll
        for (int b = 0; b < 4; ++b) {
          const int g = KC ? 0 : b;
    #pragma unroll
          for (int h = 0; h < 2; ++h) {
            const long a = (long)(((uint64_t)k_bytes(b, 2 * h + 1) << 32) |
                                  k_bytes(b, 2 * h));
            Sh[g] = __builtin_amdgcn_mfma_i32_16x16x32_i8(a, Qh[2 * b + h],
                                                          Sh[g], 0, 0, 0);
            Sl[g] = __builtin_amdgcn_mfma_i32_16x16x32_i8(a, Ql[2 * b + h],
                                                          Sl[g], 0, 0, 0);
          }
        }
        auto dot = [&](int g, int e) {
          return (float)(Sh[g][e] * 256 + Sl[g][e]);
        };

        load_v(lt);

        float sc[4];
        float mloc = -INFINITY;
    #pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int t = base + 4 * gq + e;
          const bool ok = t < end && t < qlen;
          float v;
          if constexpr (KC) {
            v = lo16(sa[e]) * dot(0, e);
          } else {
            v = lo16(sa[e]) * dot(0, e) + hi16(sa[e]) * dot(1, e) +
                lo16(sb[e]) * dot(2, e) + hi16(sb[e]) * dot(3, e);
          }
          sc[e] = ok ? v * sm_q : -INFINITY;
          mloc = fmaxf(mloc, sc[e]);
        }
        const float mn = fmaxf(m, qmax4(mloc));
        float alpha = 1.0f;
        if (mn > m)
          alpha = (m == -INFINITY) ? 0.0f : __builtin_amdgcn_exp2f(m - mn);
        m = mn;
        h4v pb;
        float psum = 0.0f, lsum = 0.0f;
    #pragma unroll
        for (int e = 0; e < 4; ++e) {
          const float p =
              sc[e] == -INFINITY ? 0.0f : __builtin_amdgcn_exp2f(sc[e] - mn);
          lsum += p;
          const _Float16 hp = (_Float16)(p * lo16(sv[e]) * PSCALE);
          psum += (float)hp;
          pb[e] = hp;
        }
        l = l * alpha + lsum;
        cp = cp * alpha + psum;
        if (__ballot(alpha != 1.0f)) {
    #pragma unroll
          for (int c = 0; c < 16; ++c) O[c] *= alpha;
        }

          // A of fragment 4 s + d: dim 16 col + 4 s + d of tokens 4 gq .. + 3.
          // One perm per token pair transposes the biased bytes and widens
          // them to fp16 [b, 0x00] (selector 0x0C), i.e. the exact denormal
          // b * 2^-24.
    #pragma unroll
        for (int s = 0; s < 4; ++s) {
          uint32_t a[4];
    #pragma unroll
          for (int e = 0; e < 4; ++e) a[e] = v_bytes(vr[e], s);
    #pragma unroll
          for (int d = 0; d < 4; ++d) {
            constexpr uint32_t kSel = 0x0C000C00u;
            const uint32_t sel = kSel | ((4u + d) << 16) | (uint32_t)d;
            const h4v A = __builtin_bit_cast(
                h4v, (uint2){__builtin_amdgcn_perm(a[1], a[0], sel),
                             __builtin_amdgcn_perm(a[3], a[2], sel)});
            O[4 * s + d] = __builtin_amdgcn_mfma_f32_16x16x16f16(
                A, pb, O[4 * s + d], 0, 0, 0);
          }
        }
      }

      l = qsum4(l);
      cp = qsum4(cp);
      if (qlive) {
        float* op = mid_o + (int64_t)(grp * QG + qv / hpk) * smo +
                    (int64_t)(kvh * hpk + qv % hpk) * smh + (int64_t)si * sms;
        const bool any = m != -INFINITY;
        // O holds sum hp (x + 128) 2^-24.
        const float corr = 128.0f * 0x1p-24f * cp;
        constexpr float kOut = 0x1p24f / (PSCALE * LUT_ONE);
    #pragma unroll
        for (int e = 0; e < 4; ++e) {
          float* dp = op + 16 * (4 * gq + e);
    #pragma unroll
          for (int c = 0; c < 16; c += 4) {
            float4 v;
            v.x = any ? (O[c][e] - corr) * kOut : 0.0f;
            v.y = any ? (O[c + 1][e] - corr) * kOut : 0.0f;
            v.z = any ? (O[c + 2][e] - corr) * kOut : 0.0f;
            v.w = any ? (O[c + 3][e] - corr) * kOut : 0.0f;
            *(float4*)(dp + c) = v;
          }
        }
        if (gq == 0) {
          op[D] = m;
          op[D + 1] = l;
        }
      }
    }
    seg0 = seg1;
  }
  #endif  // OCTAVE_MFMA_STUB
}

// Split combine for gfx942: grid (num_q, num_q_heads), 16 waves. Thread t
// weighs split t; wave w sums splits w, w + 16, ... four at a time over
// dims 4 lane .. + 3; wave 0 adds the 16 partials and undoes the 256-dim V
// rotation in registers (two in-lane stages, six shuffle stages).
template <typename OutT>
__global__ __launch_bounds__(1024) void reduce_mfma(
    const float* __restrict__ mid_o, OutT* __restrict__ out,
    const int* __restrict__ v_signs, int num_splits, int64_t smo, int64_t smh,
    int64_t sms, int64_t soo, int64_t soh) {
  #ifndef OCTAVE_MFMA_STUB
  __shared__ float sw[1024], swm[16], swl[16];
  __shared__ __attribute__((aligned(16))) float so[16][D];
  const int qi = blockIdx.x, hi = blockIdx.y;
  const int tid = threadIdx.x, w = tid >> 6, lane = tid & 63;
  const float* base = mid_o + qi * smo + hi * smh;

  const float ms = tid < num_splits ? base[tid * sms + D] : -INFINITY;
  const float ls = tid < num_splits ? base[tid * sms + D + 1] : 0.0f;
  float m = ms;
    #pragma unroll
  for (int k = 1; k < 64; k <<= 1) m = fmaxf(m, __shfl_xor(m, k));
  if (lane == 0) swm[w] = m;
  __syncthreads();
  float mg = swm[0];
    #pragma unroll
  for (int i = 1; i < 16; ++i) mg = fmaxf(mg, swm[i]);
  const float a = ms == -INFINITY ? 0.0f : __builtin_amdgcn_exp2f(ms - mg);
  sw[tid] = a;
  float l = ls * a;
    #pragma unroll
  for (int k = 1; k < 64; k <<= 1) l += __shfl_xor(l, k);
  if (lane == 0) swl[w] = l;
  __syncthreads();

  float4 o = {0, 0, 0, 0};
  for (int s0 = w; s0 < num_splits; s0 += 64) {
    float4 x[4];
    float as[4];
    #pragma unroll
    for (int u = 0; u < 4; ++u) {
      const int s = s0 + 16 * u;
      as[u] = s < num_splits ? sw[s] : 0.0f;
      x[u] = s < num_splits ? *(const float4*)(base + s * sms + 4 * lane)
                            : make_float4(0, 0, 0, 0);
    }
    #pragma unroll
    for (int u = 0; u < 4; ++u) {
      // Splits without tokens weigh zero; their O was never written.
      o.x += as[u] == 0.0f ? 0.0f : x[u].x * as[u];
      o.y += as[u] == 0.0f ? 0.0f : x[u].y * as[u];
      o.z += as[u] == 0.0f ? 0.0f : x[u].z * as[u];
      o.w += as[u] == 0.0f ? 0.0f : x[u].w * as[u];
    }
  }
  *(float4*)&so[w][4 * lane] = o;
  __syncthreads();
  if (w != 0) return;
  float x[4] = {0, 0, 0, 0};
    #pragma unroll
  for (int i = 0; i < 16; ++i) {
    const float4 v = *(const float4*)&so[i][4 * lane];
    x[0] += v.x;
    x[1] += v.y;
    x[2] += v.z;
    x[3] += v.w;
  }
  float lg = 0.0f;
    #pragma unroll
  for (int i = 0; i < 16; ++i) lg += swl[i];
  {
    const float a0 = x[0] + x[1], a1 = x[0] - x[1];
    const float a2 = x[2] + x[3], a3 = x[2] - x[3];
    x[0] = a0 + a2;
    x[1] = a1 + a3;
    x[2] = a0 - a2;
    x[3] = a1 - a3;
  }
    #pragma unroll
  for (int mk = 1; mk < 64; mk <<= 1) {
    const bool hi_lane = lane & mk;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float p = __shfl_xor(x[i], mk);
      x[i] = hi_lane ? p - x[i] : x[i] + p;
    }
  }
  const float inv_l = lg > 0.f ? 0.0625f / lg : 0.f;
  OutT* op = out + qi * soo + hi * soh + 4 * lane;
    #pragma unroll
  for (int i = 0; i < 4; ++i)
    op[i] = (OutT)(x[i] * sign_of(v_signs, 4 * lane + i) * inv_l);
  #endif  // OCTAVE_MFMA_STUB
}

}  // namespace octave_mfma

// Split combine on gfx942; false when the shape is not covered.
bool octave_reduce_mfma(torch::Tensor out, torch::Tensor mid_o,
                        torch::Tensor v_signs, int num_splits,
                        cudaStream_t stream) {
  using namespace octave_mfma;
  // 1024 threads per (query, head) pay off while those blocks alone cannot
  // fill the GPU; big batches keep the 256-thread kernel.
  static const int max_blocks =
      2 * at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  if (num_splits > 1024 || out.size(0) * out.size(1) > max_blocks ||
      out.size(2) != D || out.stride(2) != 1 || mid_o.stride(3) != 1)
    return false;
  dim3 grid(out.size(0), out.size(1));
  if (out.dtype() == at::kBFloat16)
    reduce_mfma<__hip_bfloat16><<<grid, 1024, 0, stream>>>(
        mid_o.data_ptr<float>(), (__hip_bfloat16*)out.data_ptr(),
        v_signs.data_ptr<int>(), num_splits, mid_o.stride(0), mid_o.stride(1),
        mid_o.stride(2), out.stride(0), out.stride(1));
  else
    reduce_mfma<half><<<grid, 1024, 0, stream>>>(
        mid_o.data_ptr<float>(), (half*)out.data_ptr(), v_signs.data_ptr<int>(),
        num_splits, mid_o.stride(0), mid_o.stride(1), mid_o.stride(2),
        out.stride(0), out.stride(1));
  return true;
}

constexpr int kMfmaSplits = 256;
constexpr int kMfmaMinTps = 32;
constexpr int kMfmaNw = 4;

// Launches the MFMA decode when the shape is covered; returns the number of
// splits it wrote, or 0 when the caller must use its own kernel. Dense only.
int octave_decode_mfma(torch::Tensor query, torch::Tensor cache,
                       torch::Tensor block_table, torch::Tensor q_to_req,
                       torch::Tensor q_to_klen, torch::Tensor mid_o,
                       torch::Tensor k_signs, double sm_scale,
                       int64_t num_kv_splits, int64_t fmt,
                       int64_t query_group) {
  using namespace octave_mfma;
  const int num_q = query.size(0);
  const int num_q_heads = query.size(1);
  const int num_kv_heads = cache.size(1);
  const int hpk = num_q_heads / num_kv_heads;
  const int QG = query_group > 1 ? 4 : 1;
  const int slot = fmt == 4   ? octave::Format<4, false>::SLOT
                   : fmt == 3 ? octave::Format<3, false>::SLOT
                              : octave::Format<3, true>::SLOT;
  const bool eligible =
      query.size(2) == D && query.stride(2) == 1 && hpk * QG <= 32 &&
      cache.size(2) % 16 == 0 &&
      cache.stride(2) % (fmt & octave::kCompactFlag ? 4 : 8) == 0 &&
      cache.stride(2) == slot && cache.stride(1) % 16 == 0 &&
      cache.stride(0) % 16 == 0 &&
      reinterpret_cast<uintptr_t>(cache.data_ptr()) % 16 == 0 &&
      num_kv_splits >= 1;
  if (!eligible) return 0;
  const int nrt = std::min(num_q, QG) * hpk > 16 ? 2 : 1;
  // Enough splits for ~two resident waves per SIMD, no more: every extra
  // split is another partial for the reduction to read.
  static const int target_waves =
      8 * at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int groups = std::max(1, (num_q + QG - 1) / QG) * num_kv_heads * nrt;
  const int ns =
      std::min<int>({kMfmaSplits, (int)num_kv_splits,
                     std::max(1, (target_waves + groups - 1) / groups)});
  if (num_q == 0) return ns;
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  const int spb = kMfmaNw / nrt;
  dim3 grid((num_q + QG - 1) / QG, num_kv_heads, (ns + spb - 1) / spb);
  #define SQM_G(B, KC, G, T)                                                   \
    decode_mfma<B, KC, kMfmaNw, G, T><<<grid, 64 * kMfmaNw, 0, stream>>>(      \
        (const T*)query.data_ptr(), (const uint8_t*)cache.data_ptr(),          \
        block_table.data_ptr<int>(), q_to_req.data_ptr<int>(),                 \
        q_to_klen.data_ptr<int>(), mid_o.data_ptr<float>(),                    \
        k_signs.data_ptr<int>(), (float)sm_scale, num_q, num_q_heads,          \
        num_kv_heads, cache.size(2), block_table.size(1), block_table.size(0), \
        cache.size(0), ns, kMfmaMinTps, nrt, query.stride(0), query.stride(1), \
        cache.stride(0), cache.stride(1), cache.stride(2), mid_o.stride(0),    \
        mid_o.stride(1), mid_o.stride(2))
  #define SQM(B, KC, T)     \
    do {                    \
      if (QG == 4)          \
        SQM_G(B, KC, 4, T); \
      else                  \
        SQM_G(B, KC, 1, T); \
    } while (0)
  const bool bf = query.dtype() == at::kBFloat16;
  if (fmt == 4) {
    if (bf)
      SQM(4, false, __hip_bfloat16);
    else
      SQM(4, false, half);
  } else if (fmt == 3) {
    if (bf)
      SQM(3, false, __hip_bfloat16);
    else
      SQM(3, false, half);
  } else {
    if (bf)
      SQM(3, true, __hip_bfloat16);
    else
      SQM(3, true, half);
  }
  #undef SQM
  #undef SQM_G
  return ns;
}

#endif  // USE_ROCM
