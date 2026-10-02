// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Octave prefill for RDNA3 (gfx11), head size 256, fp16. Reads the packed
// cache directly; no intermediate buffers. Everything is in the rotated space
// (the caller rotates Q/K/V of the chunk and rotates the output back).
//
// Phase 1 (prefix_attn): attention of the chunk's queries over the cached
// prefix, split over the prefix length; writes (O, m, l) partials.
//   One block = 128 query vectors (token, head) of one KV head, so the GQA
//   heads share every K/V tile. 16 compute waves in pairs: wave g of a pair
//   holds Q (int8) and O for head dims [128 g, 128 g + 128). 8 loader waves
//   unpack 16-token tiles into double-buffered LDS, one tile ahead.
//   QK  S^T[token][q] = K x Q^T with int8 WMMA. K rows hold the int8 codebook
//       value of every dim. Each wave's 128 dims are two 64-dim blocks with
//       their own per-token scale; the wave scales both and the pair
//       exchanges the float partial through LDS.
//   PV  O^T[dim][q] = V^T x P^T with V as its exact fp16 codebook values; the
//       V scale is folded into P.
// Phase 2 (chunk_attn): causal attention over the chunk's own K/V (fp16),
// merged with the phase-1 partials; writes the output.

#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <algorithm>

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#if defined(USE_ROCM)
  #include <hip/hip_runtime.h>
  #include <hip/hip_fp16.h>

  #include "octave_format.cuh"

  #if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1100__) && \
      !defined(__gfx1101__) && !defined(__gfx1102__) && !defined(__gfx1103__)
    #define OCTAVE_PREFILL_STUB
  #endif

namespace octave_pf {

using octave::D;
using octave::Format;
using octave::LUT_ONE;

typedef _Float16 h16v __attribute__((ext_vector_type(16)));
typedef _Float16 h8v __attribute__((ext_vector_type(8)));
typedef _Float16 h2v __attribute__((ext_vector_type(2)));
typedef float f8v __attribute__((ext_vector_type(8)));
typedef uint32_t u8v __attribute__((ext_vector_type(8)));
typedef uint32_t u4v __attribute__((ext_vector_type(4)));
typedef uint32_t u2v __attribute__((ext_vector_type(2)));
typedef int v4i_t __attribute__((ext_vector_type(4)));
typedef int v8i_t __attribute__((ext_vector_type(8)));

constexpr int KT = 16;  // tokens per tile

// Splits of the cached prefix used by a sequence: one per kMinSplitTokens, at
// most num_splits. Both phases must agree.
constexpr int kMinSplitTokens = 2048;
__device__ __forceinline__ int used_splits(int ctx, int num_splits) {
  if (ctx <= 0) return 0;
  return min(num_splits, (ctx + kMinSplitTokens - 1) / kMinSplitTokens);
}
__device__ __forceinline__ int split_len(int ctx, int used) {
  return ((ctx + used - 1) / used + KT - 1) / KT * KT;
}

// ===========================================================================
// Phase 1: prefix
// ===========================================================================
constexpr int NW = 8;         // compute wave pairs (16 query vectors each)
constexpr int NL = 8;         // loader waves
constexpr int KROW = D + 16;  // bytes per K row in LDS
constexpr float PSCALE = 4096.0f;
// The LDS read of PV step c waits for the WMMA of step c - DEPV, which keeps
// the scheduler from hoisting every read (and spilling).
constexpr int DEPV = 4;
constexpr int NTHREADS = (2 * NW + NL) * 32;
constexpr int VPAD = 16;  // halves after every 8 rows of V^T
constexpr int HD = D / 2;
constexpr int QSTEPS = HD / 16;
constexpr int OFRAGS = HD / 16;

__device__ __forceinline__ int vrow(int row) {
  return row * KT + (row >> 3) * VPAD;
}
__device__ __forceinline__ uint32_t xhalf(uint32_t v) {
  return __builtin_amdgcn_permlanex16(v, v, 0x76543210u, 0xFEDCBA98u, false,
                                      false);
}
__device__ __forceinline__ float xhalf(float v) {
  return __uint_as_float(xhalf(__float_as_uint(v)));
}
// 4 dwords, one per token (same 4 dims) -> 4 dwords, one per dim (4 tokens).
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
// LDS-only barrier: __syncthreads() would also wait for the loaders' global
// prefetch.
__device__ __forceinline__ void tile_barrier() {
  asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
}

// Codes of dims 4k..4k+3 of a 64-dim block, one byte each. 4-bit: raw is
// the block's 32 bytes. 3-bit: raw starts at the block's 16 bytes of the
// 2-bit plane; hi0/hi1 are its two dwords of the 1-bit plane.
template <int BITS>
__device__ __forceinline__ uint32_t block_codes(const uint32_t* raw,
                                                uint32_t hi0, uint32_t hi1,
                                                int k) {
  if constexpr (BITS == 4) {
    return (raw[k >> 1] >> (4 * (k & 1))) & 0x0F0F0F0Fu;
  } else {
    const uint32_t hi = k < 8 ? hi0 : hi1;
    return ((raw[k >> 2] >> (2 * (k & 3))) & 0x03030303u) |
           (((hi >> (k & 7)) & 0x01010101u) << 2);
  }
}

template <int VB, bool KC>
__global__ __launch_bounds__(NTHREADS) void prefix_attn(
    const __half* __restrict__ q, const uint8_t* __restrict__ cache,
    const int* __restrict__ block_table, const int* __restrict__ cu_seqlens_q,
    const int* __restrict__ seq_lens, float* __restrict__ ws, float sm_scale,
    int num_q_heads, int num_kv_heads, int block_size, int max_blocks,
    int num_splits, int total_q_tokens, int64_t sq_tok, int64_t sq_head,
    int64_t scb, int64_t sch, int64_t sct) {
  #ifndef OCTAVE_PREFILL_STUB
  using F = Format<VB, KC>;
  const int seq = blockIdx.x, kvh = blockIdx.y;
  const int rowtile = blockIdx.z / num_splits, split = blockIdx.z % num_splits;
  const int tid = threadIdx.x;
  const int w = __builtin_amdgcn_readfirstlane(tid >> 5), lane = tid & 31;
  const int j = lane & 15, h = lane >> 4;
  const int pr = w >> 1, g = w & 1;
  const int hpk = num_q_heads / num_kv_heads;
  const bool loader = w >= 2 * NW;

  const int q0 = cu_seqlens_q[seq];
  const int qlen = cu_seqlens_q[seq + 1] - q0;
  int ctx = seq_lens[seq] - qlen;
  ctx = max(0, min(ctx, max_blocks * block_size));
  const int nqv = qlen * hpk;
  if (rowtile * NW * 16 >= nqv) return;

  const int used = used_splits(ctx, num_splits);
  if (split >= used) return;
  const int k_len = split_len(ctx, used);
  const int k_begin = min(ctx, split * k_len);
  const int k_stop = min(ctx, k_begin + k_len);
  const int ntiles = k_stop > k_begin ? (k_stop - k_begin + KT - 1) / KT : 0;

  __shared__ __attribute__((aligned(16))) int8_t sK[2][KT][KROW];
  __shared__
      __attribute__((aligned(16))) _Float16 sV[2][D * KT + (D / 8) * VPAD];
  __shared__ __attribute__((aligned(16))) u4v sS[2][2 * NW][2][32];
  // K scales travel with K (block 0 = RoPE, 1-3 = NoPE), V scales with V.
  __shared__ float sKs[2][4][KT], sVs[2][KT];

  // ---- loader waves
  if (loader) {
    // 256 threads. K: thread lt unpacks dims 16 kp .. 16 kp + 15 of block kq
    // of token tk. V: dims 4 vh .. 4 vh + 3 of the 8-dim group vd of token
    // quad vq.
    const int lt = tid - 2 * NW * 32;
    const int tk = lt >> 4, kq = (lt >> 2) & 3, kp = lt & 3;
    const int vu = lt >> 1, vh = lt & 1, vq = vu & 3, vd = (vu >> 2) * 8;
    // Block sizes are multiples of KT, so a tile sits in one cache block.
    // Tiles past the end read the last one; tokens past k_stop read the last
    // token: their K is masked to -inf and their V meets p = 0.
    const int tlast = k_stop - 1;
    const uint8_t* const head = cache + (int64_t)kvh * sch;
    const int* const bt = block_table + seq * max_blocks;
    // Walk the tiles in order: block index and offset inside it, advanced by
    // KT per tile and frozen on the last tile.
    struct Pos {
      int start, blk, off;
    };
    auto first = [&]() {
      return Pos{k_begin, k_begin / block_size, k_begin % block_size};
    };
    auto next = [&](Pos p) {
      if (p.start + KT > tlast) return p;
      p.start += KT;
      p.off += KT;
      if (p.off == block_size) {
        p.off = 0;
        ++p.blk;
      }
      return p;
    };
    auto pbase = [&](Pos p, int pb) {
      return head + (int64_t)(pb < 0 ? 0 : pb) * scb + (int64_t)p.off * sct;
    };
    auto tok = [&](int start, int r) {
      return (int64_t)min(r, tlast - start) * sct;
    };
    // 4-bit: dwords 2 kp, 2 kp + 1 of the RoPE codes. 3-bit: dword kp of the
    // block's 2-bit plane and dword kp / 2 of its 1-bit plane.
    const bool k3 = KC || kq > 0;
    const int b = KC ? kq : kq - 1;  // 3-bit block index
    const int koff0 = k3 ? F::OFF_KN + 16 * b + 4 * kp : 8 * kp;
    const int koff1 =
        k3 ? F::OFF_KN + F::KN / 4 + 8 * b + 4 * (kp >> 1) : 8 * kp + 4;
    constexpr int VW = VB == 4 ? 1 : 2;  // dwords of 8 V dims
    struct Stg {
      uint32_t k0, k1;
      uint32_t v[4][VW];
      uint32_t s0, s1;
    };
    auto fetch = [&](int TK, const uint8_t* bK, int TV, const uint8_t* bV,
                     Stg& S) {
      const uint8_t* pk = bK + tok(TK, tk);
      S.k0 = *(const uint32_t*)(pk + koff0);
      S.k1 = *(const uint32_t*)(pk + koff1);
    #pragma unroll
      for (int i = 0; i < 4; ++i) {
        const uint8_t* pv = bV + tok(TV, 4 * vq + i) + F::OFF_V;
        if constexpr (VB == 4) {
          S.v[i][0] = *(const uint32_t*)(pv + vd / 2);
        } else {
          S.v[i][0] = *(const uint32_t*)(pv + 4 * (vd >> 4));
          S.v[i][1] = *(const uint32_t*)(pv + D / 4 + 4 * (vd >> 5));
        }
      }
      // Threads 0-15: the K scales of token lt; 16-31: the V scale of token
      // lt - KT. Block K: two 8-byte loads inside the slot, V scale in the
      // low half of s0. Compact K: one 4-byte load {k, v}.
      if (lt < 2 * KT) {
        if constexpr (KC) {
          S.s0 = *(const uint32_t*)((lt < KT ? bK + tok(TK, lt)
                                             : bV + tok(TV, lt - KT)) +
                                    F::OFF_SC);
        } else {
          const uint8_t* ps = lt < KT ? bK + tok(TK, lt) + F::OFF_SC
                                      : bV + tok(TV, lt - KT) + F::OFF_SC + 8;
          const u2v x = *(const u2v*)ps;
          S.s0 = x[0];
          S.s1 = x[1];
        }
      }
    };
    auto store_k = [&](const Stg& S, int bf) {
      const uint32_t r4[2] = {S.k0, S.k1};
      const uint32_t r3[1] = {S.k0};
      const uint32_t hi = S.k1 >> (4 * (kp & 1));
      u4v o;
    #pragma unroll
      for (int x = 0; x < 4; ++x)
        o[x] = k3 ? octave::lut3(block_codes<3>(r3, hi, hi, x))
                  : octave::lut4(block_codes<4>(r4, 0, 0, x));
      *(u4v*)&sK[bf][tk][64 * kq + 16 * kp] = o;
      if (lt < KT) {
        const h2v a = __builtin_bit_cast(h2v, S.s0);
        if constexpr (KC) {
          sKs[bf][0][lt] = (float)a[0];
        } else {
          const h2v c = __builtin_bit_cast(h2v, S.s1);
          sKs[bf][0][lt] = (float)a[0];
          sKs[bf][1][lt] = (float)a[1];
          sKs[bf][2][lt] = (float)c[0];
          sKs[bf][3][lt] = (float)c[1];
        }
      }
    };
    auto store_v = [&](const Stg& S, int bf) {
      uint32_t c[4];  // [token] code bytes of dims vd + 4 vh ..
    #pragma unroll
      for (int i = 0; i < 4; ++i) {
        if constexpr (VB == 4) {
          c[i] = (S.v[i][0] >> (4 * vh)) & 0x0F0F0F0Fu;
        } else {
          const int sl = 2 * ((vd & 15) >> 2) + 2 * vh;
          const int sh = ((vd & 31) >> 2) + vh;
          c[i] = ((S.v[i][0] >> sl) & 0x03030303u) |
                 (((S.v[i][1] >> sh) & 0x01010101u) << 2);
        }
        c[i] = octave::lut<VB>(c[i]) ^ 0x80808080u;
      }
      uint32_t t[4];
      tr4(c[0], c[1], c[2], c[3], t);
      // Codebook bytes fed as fp16 1024 + (x + 128), less 1152: exact x.
      const h2v kofs = {(_Float16)1152.0f, (_Float16)1152.0f};
    #pragma unroll
      for (int d = 0; d < 4; ++d) {
        const int row = vd + 4 * vh + d, sw = ((row >> 3) & 1) * 8;
        const h2v p01 = __builtin_bit_cast(
            h2v, __builtin_amdgcn_perm(0x64646464u, t[d], 0x07010700u));
        const h2v p23 = __builtin_bit_cast(
            h2v, __builtin_amdgcn_perm(0x64646464u, t[d], 0x07030702u));
        *(u2v*)&sV[bf][vrow(row) + ((4 * vq) ^ sw)] =
            (u2v){__builtin_bit_cast(uint32_t, p01 - kofs),
                  __builtin_bit_cast(uint32_t, p23 - kofs)};
      }
      if (lt >= KT && lt < 2 * KT)
        sVs[bf][lt - KT] = (float)__builtin_bit_cast(
                               _Float16, (uint16_t)(KC ? S.s0 >> 16 : S.s0)) *
                           (PSCALE / LUT_ONE);
    };
    // Pipeline: K of tile i + 1 and V of tile i are written in iteration i
    // from registers; the block number of a tile is loaded two tiles ahead.
    Stg S;
    if (ntiles > 0) {
      const Pos p0 = first(), p1 = next(p0), p2 = next(p1);
      const uint8_t* b0 = pbase(p0, bt[p0.blk]);
      const int q1 = bt[p1.blk];
      fetch(p0.start, b0, p0.start, b0, S);
      const int q2 = bt[p2.blk];
      Pos p3 = next(p2);
      int pn = bt[p3.blk];  // block of tile i + 3
      const uint8_t* bv = pbase(p1, q1);
      const uint8_t* bk = pbase(p2, q2);
      int sv = p1.start, sk = p2.start;
      store_k(S, 0);
      fetch(p1.start, bv, p0.start, b0, S);
      tile_barrier();
      for (int i = 0; i < ntiles; ++i) {
        store_k(S, (i + 1) & 1);
        store_v(S, i & 1);
        fetch(sk, bk, sv, bv, S);
        bv = bk;
        sv = sk;
        bk = pbase(p3, pn);
        sk = p3.start;
        p3 = next(p3);
        pn = bt[p3.blk];
        tile_barrier();
      }
    } else {
      tile_barrier();
    }
    tile_barrier();
    return;
  }

  // ---- compute waves: this lane's query row, int8 over this wave's dims
  const int qr = pr * 16 + j;
  const int qv = rowtile * NW * 16 + qr;
  const bool qlive = qv < nqv;
  u4v Q[QSTEPS];
  float qsc;
  {
    const int qtok = qlive ? qv / hpk : 0;
    const int qhead = kvh * hpk + (qlive ? qv % hpk : 0);
    const __half* qrow =
        q + (int64_t)(q0 + qtok) * sq_tok + (int64_t)qhead * sq_head;
    float qmax = 1e-8f;
    for (int i = 0; i < D / 8; ++i) {
      const h8v x = qlive ? *(const h8v*)(qrow + 8 * i) : (h8v){};
    #pragma unroll
      for (int k = 0; k < 8; ++k) qmax = fmaxf(qmax, fabsf((float)x[k]));
    }
    const float qinv = 127.0f / qmax;
    qsc = qmax * (1.0f / 127.0f);
    #pragma unroll
    for (int f = 0; f < QSTEPS; ++f) {
      uint32_t wd[4];
    #pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const h8v x =
            qlive ? *(const h8v*)(qrow + HD * g + 16 * f + 8 * hh) : (h8v){};
    #pragma unroll
        for (int d4 = 0; d4 < 2; ++d4) {
          uint32_t word = 0;
    #pragma unroll
          for (int b = 0; b < 4; ++b) {
            int v = (int)lrintf((float)x[4 * d4 + b] * qinv);
            v = max(-127, min(127, v));
            word |= ((uint32_t)(uint8_t)(int8_t)v) << (8 * b);
          }
          wd[2 * hh + d4] = word;
        }
      }
      Q[f] = (u4v){wd[0], wd[1], wd[2], wd[3]};
    }
  }

  f8v O[OFRAGS];
    #pragma unroll
  for (int c = 0; c < OFRAGS; ++c) O[c] = (f8v){0, 0, 0, 0, 0, 0, 0, 0};
  // A operands are only read in lanes with (j % 2) == h; the other lane of
  // each pair reads its partner's row, so the pair costs one LDS fetch.
  const int rr = ((j & 1) == h) ? j : (j ^ 1);
  const int sw = ((rr >> 3) & 1) * 8;
  const float c1 = sm_scale * qsc * (1.4426950408889634f / LUT_ONE);
  float m = -INFINITY, l = 0.f;
  f8v Sp = {0, 0, 0, 0, 0, 0, 0, 0};  // this wave's partial S, tile i-1

  tile_barrier();
  for (int i = 0; i <= ntiles; ++i) {
    f8v Sn = {0, 0, 0, 0, 0, 0, 0, 0};
    if (i < ntiles) {
      // QK over this wave's two 64-dim blocks, S^T = K x Q^T, 4 k-steps
      // each; block s is scaled into Sn before block s + 1 starts, so one
      // int accumulator is live at a time.
      uint32_t koff = (uint32_t)(rr * KROW + HD * g);
      if constexpr (KC) {  // one K scale: one accumulator for both blocks
        v8i_t Si = {0, 0, 0, 0, 0, 0, 0, 0};
    #pragma unroll
        for (int f = 0; f < 8; ++f) {
          const u4v a = *(const u4v*)(&sK[i & 1][0][0] + koff + 16 * f);
          Si = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
              true, __builtin_bit_cast(v4i_t, a), true,
              __builtin_bit_cast(v4i_t, Q[f]), Si, false);
        }
        const float* ks = &sKs[i & 1][0][0];
    #pragma unroll
        for (int e = 0; e < 8; ++e) Sn[e] = ks[2 * e + h] * (float)Si[e];
      }
    #pragma unroll
      for (int sg = 0; sg < (KC ? 0 : 2); ++sg) {
        v8i_t Si = {0, 0, 0, 0, 0, 0, 0, 0};
    #pragma unroll
        for (int f = 4 * sg; f < 4 * sg + 4; ++f) {
          const u4v a = *(const u4v*)(&sK[i & 1][0][0] + koff + 16 * f);
          Si = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
              true, __builtin_bit_cast(v4i_t, a), true,
              __builtin_bit_cast(v4i_t, Q[f]), Si, false);
        }
        const float* ks = &sKs[i & 1][2 * g + sg][0];
    #pragma unroll
        for (int e = 0; e < 8; ++e) Sn[e] += ks[2 * e + h] * (float)Si[e];
        if (sg == 0) asm volatile("" : "+v"(koff) : "v"(Sn[0]));
      }
      const u8v su = __builtin_bit_cast(u8v, Sn);
      sS[i & 1][w][0][lane] = (u4v){su[0], su[1], su[2], su[3]};
      sS[i & 1][w][1][lane] = (u4v){su[4], su[5], su[6], su[7]};
    }
    if (i > 0) {
      const int bf = (i - 1) & 1, base = k_begin + (i - 1) * KT;
      const u4v o0 = sS[bf][w ^ 1][0][lane], o1 = sS[bf][w ^ 1][1][lane];
      const f8v S =
          Sp + __builtin_bit_cast(f8v, (u8v){o0[0], o0[1], o0[2], o0[3], o1[0],
                                             o1[1], o1[2], o1[3]});
      // Scores in log2 units are S * c1 (c1 > 0). Only the last tile has
      // keys past k_stop; exp2(-inf) = 0, and m = -inf only before the first
      // tile, so nothing else needs a guard.
      float s[8];
    #pragma unroll
      for (int e = 0; e < 8; ++e) s[e] = S[e];
      if (base + KT > k_stop) {
    #pragma unroll
        for (int e = 0; e < 8; ++e)
          if (base + 2 * e + h >= k_stop) s[e] = -INFINITY;
      }
      float mloc = fmaxf(fmaxf(fmaxf(s[0], s[1]), fmaxf(s[2], s[3])),
                         fmaxf(fmaxf(s[4], s[5]), fmaxf(s[6], s[7])));
      const float mn = fmaxf(m, fmaxf(mloc, xhalf(mloc)) * c1);
      const float alpha = __builtin_amdgcn_exp2f(m - mn);
      m = mn;
      uint32_t pd[4];
      float lsum = 0.f;
    #pragma unroll
      for (int e = 0; e < 8; e += 2) {
        const float p0 = __builtin_amdgcn_exp2f(fmaf(s[e], c1, -mn));
        const float p1 = __builtin_amdgcn_exp2f(fmaf(s[e + 1], c1, -mn));
        lsum += p0 + p1;
        const _Float16 h0 = (_Float16)(p0 * sVs[bf][2 * e + h]);
        const _Float16 h1 = (_Float16)(p1 * sVs[bf][2 * e + 2 + h]);
        pd[e >> 1] = __builtin_bit_cast(uint32_t, (h2v){h0, h1});
      }
      l = l * alpha + lsum;
      if (__builtin_amdgcn_ballot_w32(alpha != 1.0f)) {
    #pragma unroll
        for (int c = 0; c < OFRAGS; ++c) O[c] *= alpha;
      }
      u8v pf;
        // B operand in natural token order: lane half 0 holds the even tokens,
        // half 1 the odd ones; interleave them per pair.
    #pragma unroll
      for (int k = 0; k < 4; ++k) {
        const uint32_t o = xhalf(pd[k]);
        const uint32_t ev = h ? o : pd[k], od = h ? pd[k] : o;
        pf[2 * k] = __builtin_amdgcn_perm(od, ev, 0x05040100u);
        pf[2 * k + 1] = __builtin_amdgcn_perm(od, ev, 0x07060302u);
      }
      const h16v pb16 = __builtin_bit_cast(h16v, pf);
    #pragma unroll
      for (int c = 0; c < OFRAGS; ++c) {
        uint32_t voff = (uint32_t)vrow(HD * g + 16 * c + rr);
        if (c >= DEPV) asm volatile("" : "+v"(voff) : "v"(O[c - DEPV][0]));
        const _Float16* vr = &sV[bf][0] + voff;
        const h8v lo = *(const h8v*)(vr + sw);
        const h8v hi = *(const h8v*)(vr + (8 ^ sw));
        const h16v a = __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7,
                                               8, 9, 10, 11, 12, 13, 14, 15);
        O[c] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, pb16, O[c]);
      }
    }
    Sp = Sn;
    tile_barrier();
  }

  // ---- partials for the phase-2 merge (natural-log max, unnormalized O)
  l += xhalf(l);
  if (!qlive) return;
  const int qtok = qv / hpk, qhead = kvh * hpk + qv % hpk;
  float* wp = ws + (int64_t)split * total_q_tokens * num_q_heads * (D + 2) +
              ((int64_t)(q0 + qtok) * num_q_heads + qhead) * (D + 2) + HD * g;
    #pragma unroll
  for (int c = 0; c < OFRAGS; ++c)
    #pragma unroll
    for (int e = 0; e < 8; ++e)
      wp[16 * c + 2 * e + h] = O[c][e] * (1.0f / PSCALE);
  if (g == 0 && h == 0) {
    wp[D] = m * 0.6931471805599453f;
    wp[D + 1] = l;
  }
  #endif  // OCTAVE_PREFILL_STUB
}

// ===========================================================================
// Phase 2: the chunk's own K/V (fp16, causal), merged with the prefix
// partials. 8 waves, 16 query rows each, one query head per block.
// ===========================================================================
constexpr int CW = 8;        // waves
constexpr int CM = 16 * CW;  // query rows per block
constexpr int FRAGS = D / 16;
constexpr int X = 8;  // fp16 per 16-byte chunk

__device__ __forceinline__ float row16_max(float v) {
  v = fmaxf(v, __shfl_xor(v, 1));
  v = fmaxf(v, __shfl_xor(v, 2));
  v = fmaxf(v, __shfl_xor(v, 4));
  return fmaxf(v, __shfl_xor(v, 8));
}
__device__ __forceinline__ float row16_sum(float v) {
  v += __shfl_xor(v, 1);
  v += __shfl_xor(v, 2);
  v += __shfl_xor(v, 4);
  return v + __shfl_xor(v, 8);
}

__global__ void __launch_bounds__(D)
    chunk_attn(__half* __restrict__ out, const float* __restrict__ ws,
               const __half* __restrict__ q, const __half* __restrict__ k,
               const __half* __restrict__ v,
               const int* __restrict__ cu_seqlens_q,
               const int* __restrict__ seq_lens, int num_q_heads,
               int num_kv_heads, int max_ctx, float sm_scale, int num_splits,
               int total_q_tokens, int64_t sq_tok, int64_t sq_head,
               int64_t sk_tok, int64_t sk_head, int64_t sv_tok, int64_t sv_head,
               int64_t so_tok, int64_t so_head) {
  #ifndef OCTAVE_PREFILL_STUB
  const int seq = blockIdx.x, head = blockIdx.y;
  const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
  const int lo = lane & 15, hi = lane >> 4;
  const int q0 = cu_seqlens_q[seq];
  const int qlen = cu_seqlens_q[seq + 1] - q0;
  const int ctx = max(0, min(seq_lens[seq] - qlen, max_ctx));
  const int tile0 = blockIdx.z * CM;
  if (tile0 >= qlen) return;
  const int wq0 = tile0 + wave * 16;
  const int nq_wave = max(0, min(16, qlen - wq0));
  const int kvh = head / (num_q_heads / num_kv_heads);

  __shared__ __attribute__((aligned(16))) _Float16 sK[FRAGS * 2 * KT * X];
  __shared__ __attribute__((aligned(16))) _Float16 sV[D * KT];
  __shared__ __attribute__((aligned(16))) _Float16 sP[CW][16 * KT];

  float m_st[8], l_st[8];
  f8v acc[FRAGS];
  // Merge the prefix splits (one weight per split; exp(0) = 1 with one).
  const int64_t split_stride = (int64_t)total_q_tokens * num_q_heads * (D + 2);
  const int used = used_splits(ctx, num_splits);
    #pragma unroll
  for (int i = 0; i < 8; ++i) {
    m_st[i] = -INFINITY;
    l_st[i] = 0.f;
  }
    #pragma unroll
  for (int f = 0; f < FRAGS; ++f) acc[f] = (f8v){0, 0, 0, 0, 0, 0, 0, 0};
    #pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int qp = wq0 + 2 * i + hi;
    if (qp >= qlen) continue;
    const float* w0 = ws + ((int64_t)(q0 + qp) * num_q_heads + head) * (D + 2);
    float mx = -INFINITY;
    for (int s = 0; s < used; ++s) mx = fmaxf(mx, w0[s * split_stride + D]);
    float lsum = 0.f;
    for (int s = 0; s < used; ++s) {
      const float* wp = w0 + s * split_stride;
      if (wp[D] == -INFINITY) continue;
      const float wgt = __expf(wp[D] - mx);
      lsum += wp[D + 1] * wgt;
    #pragma unroll
      for (int f = 0; f < FRAGS; ++f) acc[f][i] += wp[16 * f + lo] * wgt;
    }
    m_st[i] = mx;
    l_st[i] = lsum;
  }

  // Q rows are re-read from global (L1/L2) in the loop instead of being held
  // in registers; out-of-range rows are masked anyway.
  const __half* qrow = q + (int64_t)(q0 + min(wq0 + lo, qlen - 1)) * sq_tok +
                       (int64_t)head * sq_head;
  const int k_end = min(qlen, tile0 + min(CM, qlen - tile0));
  _Float16* sPw = &sP[wave][0];
  for (int n0 = 0; n0 < k_end; n0 += KT) {
    {  // K tile: [16-dim chunk][token][8] fp16
      const int tk = tid / (FRAGS), dc = (tid % FRAGS) * 2;
      const bool ok = n0 + tk < qlen;
      const __half* row =
          k + (int64_t)(q0 + n0 + tk) * sk_tok + (int64_t)kvh * sk_head;
    #pragma unroll
      for (int d = 0; d < 2; ++d) {
        int4 x = ok ? *(const int4*)(row + (dc + d) * X) : int4{0, 0, 0, 0};
        *(int4*)&sK[(dc + d) * (KT * X) + tk * X] = x;
      }
    }
    {  // V tile transposed: [dim][token]
    #pragma unroll
      for (int p = 0; p < 2; ++p) {
        const int tk = tid / (D / KT),
                  dbase = ((tid % (D / KT)) + p * (D / KT)) * 8;
        const bool ok = n0 + tk < qlen;
        int4 x = ok ? *(const int4*)(v + (int64_t)(q0 + n0 + tk) * sv_tok +
                                     (int64_t)kvh * sv_head + dbase)
                    : int4{0, 0, 0, 0};
        _Float16 t8[8];
        __builtin_memcpy(t8, &x, 16);
    #pragma unroll
        for (int e = 0; e < 8; ++e) sV[(dbase + e) * KT + tk] = t8[e];
      }
    }
    __syncthreads();
    if (nq_wave > 0 && n0 <= wq0 + nq_wave - 1) {
      f8v s = {0, 0, 0, 0, 0, 0, 0, 0};
    #pragma unroll 2
      for (int f = 0; f < FRAGS; ++f) {
        h16v b, a;
        const int4 klo = *(const int4*)&sK[(2 * f) * (KT * X) + lo * X];
        const int4 khi = *(const int4*)&sK[(2 * f + 1) * (KT * X) + lo * X];
        __builtin_memcpy(&b, &klo, 16);
        __builtin_memcpy(((char*)&b) + 16, &khi, 16);
        __builtin_memcpy(&a, qrow + 16 * f, 32);
        s = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, s);
      }
      const int kabs = n0 + lo;
      float p[8];
    #pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int r = 2 * i + hi;
        const bool keep = r < nq_wave && kabs < qlen && kabs <= wq0 + r;
        const float sc = keep ? s[i] * sm_scale : -INFINITY;
        const float mi = row16_max(sc);
        const float mn = fmaxf(m_st[i], mi);
        const float al = m_st[i] == -INFINITY ? 0.f : __expf(m_st[i] - mn);
        p[i] = mn == -INFINITY ? 0.f : __expf(sc - mn);
        l_st[i] = l_st[i] * al + row16_sum(p[i]);
        m_st[i] = mn;
    #pragma unroll
        for (int f = 0; f < FRAGS; ++f) acc[f][i] *= al;
        sPw[r * KT + lo] = (_Float16)p[i];
      }
      h16v pa;
      const int4 plo = *(const int4*)&sPw[lo * KT];
      const int4 phi = *(const int4*)&sPw[lo * KT + 8];
      __builtin_memcpy(&pa, &plo, 16);
      __builtin_memcpy(((char*)&pa) + 16, &phi, 16);
    #pragma unroll
      for (int f = 0; f < FRAGS; ++f) {
        h16v vb;
        const int4 vlo = *(const int4*)&sV[(16 * f + lo) * KT];
        const int4 vhi = *(const int4*)&sV[(16 * f + lo) * KT + 8];
        __builtin_memcpy(&vb, &vlo, 16);
        __builtin_memcpy(((char*)&vb) + 16, &vhi, 16);
        acc[f] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(pa, vb, acc[f]);
      }
    }
    __syncthreads();
  }

    #pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int qp = wq0 + 2 * i + hi;
    if (qp >= qlen) continue;
    const float inv = 1.0f / (l_st[i] + 1e-10f);
    __half* orow = out + (int64_t)(q0 + qp) * so_tok + (int64_t)head * so_head;
    #pragma unroll
    for (int f = 0; f < FRAGS; ++f)
      orow[16 * f + lo] = __float2half(acc[f][i] * inv);
  }
  #endif  // OCTAVE_PREFILL_STUB
}

// Partials of phase 1, kept alive for the process: a graph that captured a
// pointer must keep finding it valid. Grows geometrically, so it settles.
float* workspace(int64_t need, const at::TensorOptions& opts) {
  static std::mutex mtx;
  static std::unordered_map<int, std::vector<at::Tensor>> kept;
  std::lock_guard<std::mutex> lock(mtx);
  std::vector<at::Tensor>& bufs = kept[opts.device().index()];
  if (bufs.empty() || bufs.back().numel() < need) {
    int64_t n = 8 << 20;
    while (n < need) n *= 2;
    bufs.push_back(at::empty({n}, opts));
  }
  return (float*)bufs.back().data_ptr();
}

}  // namespace octave_pf

int octave_slot_bytes(int64_t fmt);

void octave_prefill(torch::Tensor out, torch::Tensor q, torch::Tensor k,
                    torch::Tensor v, torch::Tensor cache,
                    torch::Tensor block_table, torch::Tensor cu_seqlens_q,
                    torch::Tensor seq_lens, int64_t max_query_len,
                    double sm_scale, int64_t fmt) {
  using namespace octave_pf;
  TORCH_CHECK(q.dtype() == at::kHalf && k.dtype() == at::kHalf &&
                  v.dtype() == at::kHalf && out.dtype() == at::kHalf,
              "octave_prefill: fp16 only");
  TORCH_CHECK(q.size(2) == D && q.stride(2) == 1 && k.stride(2) == 1 &&
              v.stride(2) == 1 && out.stride(2) == 1);
  TORCH_CHECK(cache.dtype() == at::kByte && cache.dim() == 4 &&
              cache.stride(3) == 1);
  TORCH_CHECK(cache.size(3) == octave_slot_bytes(fmt),
              "octave_prefill: slot size mismatch");
  const int align = fmt & octave::kCompactFlag ? 4 : 8;
  TORCH_CHECK(cache.stride(2) % align == 0 && cache.stride(1) % align == 0 &&
                  cache.stride(0) % align == 0,
              "octave_prefill: misaligned slots");
  static const bool arch_ok = [] {
    const auto* prop = at::cuda::getCurrentDeviceProperties();
    return std::string(prop->gcnArchName).rfind("gfx11", 0) == 0;
  }();
  TORCH_CHECK(arch_ok, "octave_prefill: RDNA3 (gfx11) only");
  const int num_seqs = seq_lens.size(0);
  if (num_seqs == 0 || q.size(0) == 0) return;
  const at::cuda::OptionalCUDAGuard guard(device_of(q));
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  const int hq = q.size(1), hkv = cache.size(1);
  TORCH_CHECK(hq % hkv == 0);
  const int hpk = hq / hkv;
  const int block_size = cache.size(2);
  const int max_blocks = block_table.size(1);
  TORCH_CHECK(block_table.stride(1) == 1 && block_table.stride(0) == max_blocks,
              "octave_prefill: block_table rows must be contiguous");
  const int total_q = q.size(0);

  const int rowtiles = (int)((max_query_len * hpk + 127) / 128);
  const int base_blocks = std::max(1, num_seqs * hkv * rowtiles);
  // Blocks whose split has no keys exit at once, so ask for plenty.
  const int num_splits =
      std::min(16, std::max(1, (1024 + base_blocks - 1) / base_blocks));
  float* ws = workspace((int64_t)num_splits * total_q * hq * (D + 2),
                        q.options().dtype(at::kFloat));

  dim3 grid1(num_seqs, hkv, rowtiles * num_splits);
  #define PREFIX(B, KC)                                                     \
    prefix_attn<B, KC><<<grid1, NTHREADS, 0, stream>>>(                     \
        (const __half*)q.data_ptr(), cache.data_ptr<uint8_t>(),             \
        block_table.data_ptr<int>(), cu_seqlens_q.data_ptr<int>(),          \
        seq_lens.data_ptr<int>(), ws, (float)sm_scale, hq, hkv, block_size, \
        max_blocks, num_splits, total_q, q.stride(0), q.stride(1),          \
        cache.stride(0), cache.stride(1), cache.stride(2))
  if (fmt == 4)
    PREFIX(4, false);
  else if (fmt == 3)
    PREFIX(3, false);
  else
    PREFIX(3, true);
  #undef PREFIX
  dim3 grid2(num_seqs, hq, (max_query_len + CM - 1) / CM);
  chunk_attn<<<grid2, D, 0, stream>>>(
      (__half*)out.data_ptr(), ws, (const __half*)q.data_ptr(),
      (const __half*)k.data_ptr(), (const __half*)v.data_ptr(),
      cu_seqlens_q.data_ptr<int>(), seq_lens.data_ptr<int>(), hq, hkv,
      max_blocks * block_size, (float)sm_scale, num_splits, total_q,
      q.stride(0), q.stride(1), k.stride(0), k.stride(1), v.stride(0),
      v.stride(1), out.stride(0), out.stride(1));
}

#else
void octave_prefill(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                    int64_t, double, int64_t) {
  TORCH_CHECK(false, "octave requires ROCm");
}
#endif
