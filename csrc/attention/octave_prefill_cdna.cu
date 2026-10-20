// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Octave prefill for CDNA3 (gfx942), head size 256, fp16. Same contract as
// octave_prefill_rdna3.cu: reads the packed cache directly, everything in the
// rotated space, phase 1 writes (O, m, l) partials over the cached prefix and
// phase 2 runs causal attention over the chunk and merges.
//
// Both phases compute transposed tiles so that every MFMA result is already in
// the layout the next MFMA consumes (16x16 MFMA, wave64: lane l holds column
// l % 16 and rows 4 (l / 16) .. 4 (l / 16) + 3):
//   S^T[token][q] = K x Q^T      (phase 1: int8 16x16x32, phase 2: fp16)
//   O^T[dim][q]  += V^T x P^T    (fp16 16x16x16)
// so P^T comes straight out of the S^T accumulator, one query per lane.
//
// Phase 1: one block = 128 query vectors (token, head) of one KV head, 8 waves
// of 16 query vectors x 256 dims. 256 threads unpack the next 16-token tile
// into LDS while the current one is consumed.

#include <cstdint>
#include <algorithm>

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#if defined(USE_ROCM)
  #include <hip/hip_runtime.h>
  #include <hip/hip_fp16.h>

  #include "octave_format.cuh"

  #if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx942__)
    #define OCTAVE_CDNA_STUB
  #endif

namespace octave_cdna {

using octave::D;
using octave::Format;
using octave::LUT_ONE;

typedef _Float16 h4v __attribute__((ext_vector_type(4)));
typedef _Float16 h2v __attribute__((ext_vector_type(2)));
typedef float f4v __attribute__((ext_vector_type(4)));
typedef int i4v __attribute__((ext_vector_type(4)));
typedef uint32_t u4v __attribute__((ext_vector_type(4)));
typedef uint32_t u2v __attribute__((ext_vector_type(2)));

constexpr int KT = 16;      // tokens per tile
constexpr int FR = D / 16;  // 16-dim output fragments
// Both phases run 8 waves (128 query vectors of one KV head) per block: every
// block reads the whole K/V it attends to, so more query vectors per block
// means less of that work per query.
constexpr int NW1 = 8;
constexpr int NT1 = NW1 * 64;
constexpr int QB1 = NW1 * 16;
constexpr int KT1 = 32;  // phase-1 tile: two 16-token sub-tiles
[[maybe_unused]] constexpr int VST1 = KT1 + 4;
[[maybe_unused]] constexpr int KROW = D + 16;  // bytes per int8 K row
[[maybe_unused]] constexpr int VST = KT + 4;   // halves per V^T row
[[maybe_unused]] constexpr int HROW = D + 8;   // halves per fp16 K row
// Phase 2 LDS rows, picked for conflict-free banks: a K row is 2 dwords
// mod 32, and V^T rows of 40 halves with the 4-token group XORed by
// (row >> 3) & 7 spread both the transposed stores and the MFMA reads.
constexpr int KROW2 = D + 4;
constexpr int VST2 = KT1 + 8;
[[maybe_unused]] constexpr float PSCALE = 4096.0f;

constexpr int kMinSplitTokens = 2048;
__device__ __forceinline__ int used_splits(int ctx, int num_splits) {
  if (ctx <= 0) return 0;
  return min(num_splits, (ctx + kMinSplitTokens - 1) / kMinSplitTokens);
}
__device__ __forceinline__ int split_len(int ctx, int used) {
  return ((ctx + used - 1) / used + 31) / 32 * 32;
}

// Reductions over the four lanes of a query (l, l ^ 16, l ^ 32, l ^ 48).
__device__ __forceinline__ float qmax4(float v) {
  v = fmaxf(v, __shfl_xor(v, 16));
  return fmaxf(v, __shfl_xor(v, 32));
}
__device__ __forceinline__ float qsum4(float v) {
  v += __shfl_xor(v, 16);
  return v + __shfl_xor(v, 32);
}

// Codes of dims 4k..4k+3 of a 64-dim block (see octave_prefill_rdna3.cu).
template <int BITS>
__device__ __forceinline__ uint32_t block_codes(const uint32_t* raw,
                                                uint32_t hi, int k) {
  if constexpr (BITS == 4) {
    return (raw[k >> 1] >> (4 * (k & 1))) & 0x0F0F0F0Fu;
  } else {
    return ((raw[0] >> (2 * (k & 3))) & 0x03030303u) |
           (((hi >> (k & 7)) & 0x01010101u) << 2);
  }
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

// ===========================================================================
// Phase 1: prefix
// ===========================================================================
template <int VB, bool KC>
__global__ __launch_bounds__(NT1) void prefix_attn(
    const __half* __restrict__ q, const uint8_t* __restrict__ cache,
    const int* __restrict__ block_table, const int* __restrict__ cu_seqlens_q,
    const int* __restrict__ seq_lens, float* __restrict__ ws, float sm_scale,
    int num_q_heads, int num_kv_heads, int block_size, int max_blocks,
    int num_splits, int total_q_tokens, int64_t sq_tok, int64_t sq_head,
    int64_t scb, int64_t sch, int64_t sct) {
  #ifndef OCTAVE_CDNA_STUB
  using F = Format<VB, KC>;
  const int seq = blockIdx.x, kvh = blockIdx.y;
  const int rowtile = blockIdx.z / num_splits, split = blockIdx.z % num_splits;
  const int tid = threadIdx.x;
  const int w = tid >> 6, lane = tid & 63;
  const int col = lane & 15, gq = lane >> 4;
  const int hpk = num_q_heads / num_kv_heads;

  const int q0 = cu_seqlens_q[seq];
  const int qlen = cu_seqlens_q[seq + 1] - q0;
  int ctx = seq_lens[seq] - qlen;
  ctx = max(0, min(ctx, max_blocks * block_size));
  const int nqv = qlen * hpk;
  if (rowtile * QB1 >= nqv) return;

  const int used = used_splits(ctx, num_splits);
  if (split >= used) return;
  const int k_len = split_len(ctx, used);
  const int k_begin = min(ctx, split * k_len);
  const int k_stop = min(ctx, k_begin + k_len);
  const int ntiles = k_stop > k_begin ? (k_stop - k_begin + KT1 - 1) / KT1 : 0;

  __shared__ __attribute__((aligned(16))) int8_t sK[2][KT1][KROW];
  __shared__ __attribute__((aligned(16))) _Float16 sV[2][D * VST1];
  __shared__ float sKs[2][4][KT1], sVs[2][KT1];

  // ---- loader mapping. Thread half sub (256 threads each) loads the 16-token
  // sub-tile sub of a tile; within it, K: dims 16 kp .. + 15 of block kq of
  // token tk; V: dims vd + 4 vh .. + 3 of token quad vq.
  const int sub = tid >> 8, lt = tid & 255;
  const int tk = lt >> 4, kq = (lt >> 2) & 3, kp = lt & 3;
  const int vu = lt >> 1, vh = lt & 1, vq = vu & 3, vd = (vu >> 2) * 8;
  const bool k3 = KC || kq > 0;
  const int b3 = KC ? kq : kq - 1;
  const int koff0 = k3 ? F::OFF_KN + 16 * b3 + 4 * kp : 8 * kp;
  const int koff1 =
      k3 ? F::OFF_KN + F::KN / 4 + 8 * b3 + 4 * (kp >> 1) : 8 * kp + 4;
  constexpr int VW = VB == 4 ? 1 : 2;
  const int tlast = k_stop - 1;
  const uint8_t* const head = cache + (int64_t)kvh * sch;
  const int* const bt = block_table + seq * max_blocks;

  struct Stg {
    uint32_t k0, k1;
    uint32_t v[4][VW];
    uint32_t s0, s1;
  };
  // Sub-tiles start 16-aligned and sit in one cache block (block sizes are
  // multiples of 16), so all 16 slots exist: tokens past k_stop are read as
  // they are and masked (K to -inf, P to 0 by selection, never by product).
  // Slot offsets inside a sub-tile are fixed per thread.
  auto sub_start = [&](int i) {
    return __builtin_amdgcn_readfirstlane(
        min(k_begin + i * KT1 + 16 * sub, tlast & ~15));
  };
  const uint32_t ok0 = tk * (uint32_t)sct + koff0;
  const uint32_t ok1 = tk * (uint32_t)sct + koff1;
  uint32_t ov[4];
    #pragma unroll
  for (int r = 0; r < 4; ++r) {
    ov[r] = (4 * vq + r) * (uint32_t)sct + F::OFF_V +
            (VB == 4 ? vd / 2 : 4 * (vd >> 4));
  }
  const uint32_t os =
      (lt & 15) * (uint32_t)sct + F::OFF_SC + (!KC && lt >= 16 ? 8 : 0);
  auto fetch = [&](int i, Stg& S) {
    const int start = sub_start(i);
    const int pb = __builtin_amdgcn_readfirstlane(bt[start / block_size]);
    const uint8_t* base = head + (int64_t)(pb < 0 ? 0 : pb) * scb +
                          (int64_t)(start % block_size) * sct;
    S.k0 = *(const uint32_t*)(base + ok0);
    S.k1 = *(const uint32_t*)(base + ok1);
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
      S.v[r][0] = *(const uint32_t*)(base + ov[r]);
      if constexpr (VB != 4)
        S.v[r][1] = *(const uint32_t*)(base + ov[r] - 4 * (vd >> 4) + D / 4 +
                                       4 * (vd >> 5));
    }
    if (lt < 32) {
      if constexpr (KC) {
        S.s0 = *(const uint32_t*)(base + os);
      } else {
        const u2v x = *(const u2v*)(base + os);
        S.s0 = x[0];
        S.s1 = x[1];
      }
    }
  };
  auto store = [&](const Stg& S, int bf) {
    const int t0 = 16 * sub;
    {
      const uint32_t r4[2] = {S.k0, S.k1};
      const uint32_t r3[1] = {S.k0};
      const uint32_t hi = S.k1 >> (4 * (kp & 1));
      u4v o;
    #pragma unroll
      for (int x = 0; x < 4; ++x)
        o[x] = k3 ? octave::lut3(block_codes<3>(r3, hi, x))
                  : octave::lut4(block_codes<4>(r4, 0, x));
      *(u4v*)&sK[bf][t0 + tk][64 * kq + 16 * kp] = o;
    }
    uint32_t c[4];
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
      if constexpr (VB == 4) {
        c[r] = (S.v[r][0] >> (4 * vh)) & 0x0F0F0F0Fu;
      } else {
        const int sl = 2 * ((vd & 15) >> 2) + 2 * vh;
        const int sh = ((vd & 31) >> 2) + vh;
        c[r] = ((S.v[r][0] >> sl) & 0x03030303u) |
               (((S.v[r][1] >> sh) & 0x01010101u) << 2);
      }
      c[r] = octave::lut<VB>(c[r]) ^ 0x80808080u;
    }
    uint32_t t[4];
    tr4(c[0], c[1], c[2], c[3], t);
    // Codebook bytes fed as fp16 1024 + (x + 128), less 1152: exact x.
    const h2v kofs = {(_Float16)1152.0f, (_Float16)1152.0f};
    #pragma unroll
    for (int d = 0; d < 4; ++d) {
      const int row = vd + 4 * vh + d;
      const h2v p01 = __builtin_bit_cast(
          h2v, __builtin_amdgcn_perm(0x64646464u, t[d], 0x07010700u));
      const h2v p23 = __builtin_bit_cast(
          h2v, __builtin_amdgcn_perm(0x64646464u, t[d], 0x07030702u));
      *(u2v*)&sV[bf][row * VST1 + t0 + 4 * vq] =
          (u2v){__builtin_bit_cast(uint32_t, p01 - kofs),
                __builtin_bit_cast(uint32_t, p23 - kofs)};
    }
    if (lt < 16) {
      const h2v a = __builtin_bit_cast(h2v, S.s0);
      sKs[bf][0][t0 + lt] = (float)a[0];
      if constexpr (!KC) {
        const h2v cc = __builtin_bit_cast(h2v, S.s1);
        sKs[bf][1][t0 + lt] = (float)a[1];
        sKs[bf][2][t0 + lt] = (float)cc[0];
        sKs[bf][3][t0 + lt] = (float)cc[1];
      }
    } else if (lt < 32) {
      sVs[bf][t0 + lt - 16] =
          (float)__builtin_bit_cast(_Float16,
                                    (uint16_t)(KC ? S.s0 >> 16 : S.s0)) *
          (PSCALE / LUT_ONE);
    }
  };

  // ---- this lane's query vector, int8 over dims 32 s + 8 gq .. + 7
  const int qv = rowtile * QB1 + w * 16 + col;
  const bool qlive = qv < nqv;
  long Q[8];
  float qsc;
  {
    const int qtok = qlive ? qv / hpk : 0;
    const int qhead = kvh * hpk + (qlive ? qv % hpk : 0);
    const __half* qrow =
        q + (int64_t)(q0 + qtok) * sq_tok + (int64_t)qhead * sq_head;
    _Float16 x[8][8];
    float qmax = 1e-8f;
    #pragma unroll
    for (int s = 0; s < 8; ++s) {
      const u4v r = qlive ? *(const u4v*)(qrow + 32 * s + 8 * gq) : (u4v){};
      __builtin_memcpy(x[s], &r, 16);
    #pragma unroll
      for (int e = 0; e < 8; ++e) qmax = fmaxf(qmax, fabsf((float)x[s][e]));
    }
    qmax = qmax4(qmax);
    const float qinv = 127.0f / qmax;
    qsc = qmax * (1.0f / 127.0f);
    #pragma unroll
    for (int s = 0; s < 8; ++s) {
      uint64_t word = 0;
    #pragma unroll
      for (int e = 0; e < 8; ++e) {
        int v = (int)lrintf((float)x[s][e] * qinv);
        v = max(-127, min(127, v));
        word |= ((uint64_t)(uint8_t)(int8_t)v) << (8 * e);
      }
      Q[s] = (long)word;
    }
  }

  f4v O[FR];
    #pragma unroll
  for (int c = 0; c < FR; ++c) O[c] = (f4v){0, 0, 0, 0};
  const float c1 = sm_scale * qsc * (1.4426950408889634f / LUT_ONE);
  float m = -INFINITY, l = 0.f;

  // Pipeline: iteration i computes tile i from LDS while the loads of tile
  // i + 1 are in flight, then stores them into the other buffer.
  Stg S;
  if (ntiles > 0) {
    fetch(0, S);
    store(S, 0);
  }
  __syncthreads();
  for (int i = 0; i < ntiles; ++i) {
    const int bf = i & 1;
    if (i + 1 < ntiles) fetch(i + 1, S);

    // S^T = K x Q^T over both sub-tiles first (independent MFMA chains),
    // then one online-softmax step over the 32 tokens, then PV.
    f4v Sf[2];
    #pragma unroll
    for (int j = 0; j < 2; ++j) {
      Sf[j] = (f4v){0, 0, 0, 0};
      const int8_t* krow = &sK[bf][16 * j + col][8 * gq];
      const int tq = 16 * j + 4 * gq;
      if constexpr (KC) {
        i4v Si = {0, 0, 0, 0};
    #pragma unroll
        for (int s = 0; s < 8; ++s)
          Si = __builtin_amdgcn_mfma_i32_16x16x32_i8(
              *(const long*)(krow + 32 * s), Q[s], Si, 0, 0, 0);
    #pragma unroll
        for (int e = 0; e < 4; ++e)
          Sf[j][e] = sKs[bf][0][tq + e] * (float)Si[e];
      } else {
    #pragma unroll
        for (int b = 0; b < 4; ++b) {
          i4v Si = {0, 0, 0, 0};
    #pragma unroll
          for (int s = 2 * b; s < 2 * b + 2; ++s)
            Si = __builtin_amdgcn_mfma_i32_16x16x32_i8(
                *(const long*)(krow + 32 * s), Q[s], Si, 0, 0, 0);
    #pragma unroll
          for (int e = 0; e < 4; ++e)
            Sf[j][e] += sKs[bf][b][tq + e] * (float)Si[e];
        }
      }
    }

    // Online softmax in log2 units; this lane holds tokens 16 j + 4 gq + e.
    float sc[2][4];
    float mloc = -INFINITY;
    #pragma unroll
    for (int j = 0; j < 2; ++j)
    #pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int t = k_begin + i * KT1 + 16 * j + 4 * gq + e;
        sc[j][e] = t < k_stop ? Sf[j][e] * c1 : -INFINITY;
        mloc = fmaxf(mloc, sc[j][e]);
      }
    const float mn = fmaxf(m, qmax4(mloc));
    const float alpha = __builtin_amdgcn_exp2f(m - mn);
    m = mn;
    h4v pb[2];
    float lsum = 0.f;
    #pragma unroll
    for (int j = 0; j < 2; ++j)
    #pragma unroll
      for (int e = 0; e < 4; ++e) {
        const bool live = sc[j][e] != -INFINITY;
        const float p = live ? __builtin_amdgcn_exp2f(sc[j][e] - mn) : 0.f;
        lsum += p;
        pb[j][e] =
            live ? (_Float16)(p * sVs[bf][16 * j + 4 * gq + e]) : (_Float16)0.f;
      }
    l = l * alpha + lsum;
    if (__ballot(alpha != 1.0f)) {
    #pragma unroll
      for (int c = 0; c < FR; ++c) O[c] *= alpha;
    }

    // O^T += V^T x P^T: A = V^T[dim 16 c + col][tokens 16 j + 4 gq .. + 3].
    #pragma unroll
    for (int j = 0; j < 2; ++j) {
      const _Float16* vrow = &sV[bf][col * VST1 + 16 * j + 4 * gq];
    #pragma unroll
      for (int c = 0; c < FR; ++c)
        O[c] = __builtin_amdgcn_mfma_f32_16x16x16f16(
            *(const h4v*)(vrow + 16 * c * VST1), pb[j], O[c], 0, 0, 0);
    }

    if (i + 1 < ntiles) store(S, bf ^ 1);
    __syncthreads();
  }

  // ---- partials for the phase-2 merge (natural-log max, unnormalized O)
  l = qsum4(l);
  if (!qlive) return;
  const int qtok = qv / hpk, qhead = kvh * hpk + qv % hpk;
  float* wp = ws + (int64_t)split * total_q_tokens * num_q_heads * (D + 2) +
              ((int64_t)(q0 + qtok) * num_q_heads + qhead) * (D + 2);
    #pragma unroll
  for (int c = 0; c < FR; ++c)
    #pragma unroll
    for (int e = 0; e < 4; ++e)
      wp[16 * c + 4 * gq + e] = O[c][e] * (1.0f / PSCALE);
  if (gq == 0) {
    wp[D] = m * 0.6931471805599453f;
    wp[D + 1] = l;
  }
  #endif  // OCTAVE_CDNA_STUB
}

// ===========================================================================
// Phase 2: the chunk's own K/V (fp16, causal), merged with the prefix
// partials. Like phase 1, one block = 128 query vectors (token, head) of one
// KV head, so the GQA heads share every K/V tile; 8 waves of 16 vectors.
// ===========================================================================
__global__ void __launch_bounds__(NT1)
    chunk_attn(__half* __restrict__ out, const float* __restrict__ ws,
               const __half* __restrict__ q, const __half* __restrict__ k,
               const __half* __restrict__ v,
               const int* __restrict__ cu_seqlens_q,
               const int* __restrict__ seq_lens, int num_q_heads,
               int num_kv_heads, int max_ctx, float sm_scale, int num_splits,
               int total_q_tokens, int64_t sq_tok, int64_t sq_head,
               int64_t sk_tok, int64_t sk_head, int64_t sv_tok, int64_t sv_head,
               int64_t so_tok, int64_t so_head) {
  #ifndef OCTAVE_CDNA_STUB
  constexpr float LOG2E = 1.4426950408889634f;
  const int seq = blockIdx.x, kvh = blockIdx.y, rowtile = blockIdx.z;
  const int tid = threadIdx.x, w = tid >> 6, lane = tid & 63;
  const int col = lane & 15, gq = lane >> 4;
  const int hpk = num_q_heads / num_kv_heads;
  const int q0 = cu_seqlens_q[seq];
  const int qlen = cu_seqlens_q[seq + 1] - q0;
  const int nqv = qlen * hpk;
  if (rowtile * QB1 >= nqv) return;
  const int ctx = max(0, min(seq_lens[seq] - qlen, max_ctx));
  const int qv = rowtile * QB1 + w * 16 + col;
  const bool qlive = qv < nqv;
  const int qtok = qlive ? qv / hpk : 0;
  const int head = kvh * hpk + (qlive ? qv % hpk : 0);
  // Keys this block and this wave need (causal: up to their last token).
  const int k_end = min(qlen, (min(nqv, (rowtile + 1) * QB1) - 1) / hpk + 1);
  const int wave_first = rowtile * QB1 + w * 16;
  const bool wave_live = wave_first < nqv;
  const int wave_last_tok = min(nqv - 1, wave_first + 15) / hpk;

  // One 32-token tile in LDS (two would not fit in 64 KB with fp16 K and V);
  // the next tile waits in registers.
  __shared__ __attribute__((aligned(16))) _Float16 sK[KT1 * KROW2];
  __shared__ __attribute__((aligned(16))) _Float16 sV[D * VST2];

  // Merge the prefix splits (log2 units from here on). O^T fragment c holds
  // dims 16 c + 4 gq + e of this lane's vector; l stays partial per lane.
  f4v acc[FR];
    #pragma unroll
  for (int c = 0; c < FR; ++c) acc[c] = (f4v){0, 0, 0, 0};
  float m = -INFINITY, l = 0.f;
  const int64_t split_stride = (int64_t)total_q_tokens * num_q_heads * (D + 2);
  const int used = used_splits(ctx, num_splits);
  if (qlive && used > 0) {
    const float* w0 =
        ws + ((int64_t)(q0 + qtok) * num_q_heads + head) * (D + 2);
    for (int s = 0; s < used; ++s)
      m = fmaxf(m, w0[s * split_stride + D] * LOG2E);
    for (int s = 0; s < used; ++s) {
      const float* wp = w0 + s * split_stride;
      if (wp[D] == -INFINITY) continue;
      const float wgt = __builtin_amdgcn_exp2f(wp[D] * LOG2E - m);
      l += wp[D + 1] * wgt;
    #pragma unroll
      for (int c = 0; c < FR; ++c)
    #pragma unroll
        for (int e = 0; e < 4; ++e) acc[c][e] += wp[16 * c + 4 * gq + e] * wgt;
    }
    if (gq != 0) l = 0.f;
  }

  // Q^T operand: dims 16 s + 4 gq .. + 3 of this lane's vector.
  h4v Qh[FR];
  {
    const __half* qrow =
        q + (int64_t)(q0 + qtok) * sq_tok + (int64_t)head * sq_head;
    #pragma unroll
    for (int s = 0; s < FR; ++s) Qh[s] = *(const h4v*)(qrow + 16 * s + 4 * gq);
  }
  const float c2 = sm_scale * LOG2E;

  // Loaders. K: thread t moves dims 8 (t % 32) .. + 7 of tokens t / 32
  // and t / 32 + 16 of the 32-token tile. V: dims 4 (t % 64) .. + 3 of
  // tokens 4 (t / 64) .. + 3, transposed in registers into V^T rows. Rows
  // past the chunk are zeros.
  const int lt_tok = tid >> 5, lt_d = (tid & 31) * 8;
  const int vt_d = (tid & 63) * 4, vt_g = tid >> 6;
  const __half* kbase = k + (int64_t)kvh * sk_head + lt_d;
  const __half* vbase = v + (int64_t)kvh * sv_head + vt_d;
  u4v Kr[2];
  u2v Vr[4];
  auto fetch = [&](int n0) {
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int t = n0 + lt_tok + 16 * h;
      const bool ok = t < qlen;
      const int64_t r = q0 + (ok ? t : 0);
      Kr[h] = ok ? *(const u4v*)(kbase + r * sk_tok) : (u4v){0, 0, 0, 0};
    }
    #pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int t = n0 + 4 * vt_g + e;
      const bool ok = t < qlen;
      const int64_t r = q0 + (ok ? t : 0);
      Vr[e] = ok ? *(const u2v*)(vbase + r * sv_tok) : (u2v){0, 0};
    }
  };
  auto store = [&]() {
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
      _Float16* kp = &sK[(lt_tok + 16 * h) * KROW2 + lt_d];
      *(u2v*)kp = (u2v){Kr[h][0], Kr[h][1]};
      *(u2v*)(kp + 4) = (u2v){Kr[h][2], Kr[h][3]};
    }
    #pragma unroll
    for (int d = 0; d < 4; ++d) {
      const uint32_t sel = (d & 1) ? 0x07060302u : 0x05040100u;
      const uint32_t lo =
          __builtin_amdgcn_perm(Vr[1][d >> 1], Vr[0][d >> 1], sel);
      const uint32_t hi =
          __builtin_amdgcn_perm(Vr[3][d >> 1], Vr[2][d >> 1], sel);
      const int row = vt_d + d;
      *(u2v*)&sV[row * VST2 + 4 * (vt_g ^ ((row >> 3) & 7))] = (u2v){lo, hi};
    }
  };

  const int ntiles = (k_end + KT1 - 1) / KT1;
  fetch(0);
  for (int i = 0; i < ntiles; ++i) {
    const int n0 = i * KT1;
    store();
    __syncthreads();
    if (i + 1 < ntiles) fetch(n0 + KT1);
    if (wave_live && n0 <= wave_last_tok) {
      // S^T for both 16-token halves, one softmax step, then PV.
      f4v sacc[2];
    #pragma unroll
      for (int j = 0; j < 2; ++j) {
        sacc[j] = (f4v){0, 0, 0, 0};
        const _Float16* krow = &sK[(16 * j + col) * KROW2 + 4 * gq];
    #pragma unroll
        for (int s = 0; s < FR; ++s)
          sacc[j] = __builtin_amdgcn_mfma_f32_16x16x16f16(
              *(const h4v*)(krow + 16 * s), Qh[s], sacc[j], 0, 0, 0);
      }
      float sc[2][4];
      float mloc = -INFINITY;
    #pragma unroll
      for (int j = 0; j < 2; ++j)
    #pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int kabs = n0 + 16 * j + 4 * gq + e;
          sc[j][e] = qlive && kabs <= qtok ? sacc[j][e] * c2 : -INFINITY;
          mloc = fmaxf(mloc, sc[j][e]);
        }
      const float mn = fmaxf(m, qmax4(mloc));
      const float al = mn == -INFINITY ? 1.f : __builtin_amdgcn_exp2f(m - mn);
      h4v pb[2];
      float lsum = 0.f;
    #pragma unroll
      for (int j = 0; j < 2; ++j)
    #pragma unroll
        for (int e = 0; e < 4; ++e) {
          const float p = sc[j][e] == -INFINITY
                              ? 0.f
                              : __builtin_amdgcn_exp2f(sc[j][e] - mn);
          lsum += p;
          pb[j][e] = (_Float16)p;
        }
      m = mn;
      l = l * al + lsum;
      if (__ballot(al != 1.0f)) {
    #pragma unroll
        for (int c = 0; c < FR; ++c) acc[c] *= al;
      }
    #pragma unroll
      for (int j = 0; j < 2; ++j) {
          // Row 16 c + col; its group 4 j + gq sits at (4 j + gq) ^ (row >> 3).
    #pragma unroll
        for (int c = 0; c < FR; ++c) {
          const int row = 16 * c + col;
          const _Float16* vp =
              &sV[row * VST2 + 4 * ((4 * j + gq) ^ ((row >> 3) & 7))];
          acc[c] = __builtin_amdgcn_mfma_f32_16x16x16f16(*(const h4v*)vp, pb[j],
                                                         acc[c], 0, 0, 0);
        }
      }
    }
    __syncthreads();
  }

  l = qsum4(l);
  if (!qlive) return;
  const float inv = 1.0f / (l + 1e-10f);
  __half* orow = out + (int64_t)(q0 + qtok) * so_tok + (int64_t)head * so_head;
    #pragma unroll
  for (int c = 0; c < FR; ++c)
    #pragma unroll
    for (int e = 0; e < 4; ++e)
      orow[16 * c + 4 * gq + e] = __float2half(acc[c][e] * inv);
  #endif  // OCTAVE_CDNA_STUB
}

}  // namespace octave_cdna

// Called by octave_prefill (octave_prefill_rdna3.cu) on gfx942; same
// arguments, already validated there.
void octave_prefill_cdna(torch::Tensor out, torch::Tensor q, torch::Tensor k,
                         torch::Tensor v, torch::Tensor cache,
                         torch::Tensor block_table, torch::Tensor cu_seqlens_q,
                         torch::Tensor seq_lens, int64_t max_query_len,
                         double sm_scale, int64_t fmt, float* ws,
                         int num_splits) {
  using namespace octave_cdna;
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  const int num_seqs = seq_lens.size(0);
  const int hq = q.size(1), hkv = cache.size(1);
  const int hpk = hq / hkv;
  const int block_size = cache.size(2);
  const int max_blocks = block_table.size(1);
  const int total_q = q.size(0);
  const int rowtiles = (int)((max_query_len * hpk + QB1 - 1) / QB1);

  dim3 grid1(num_seqs, hkv, rowtiles * num_splits);
  #define PREFIX(B, KC)                                                     \
    prefix_attn<B, KC><<<grid1, NT1, 0, stream>>>(                          \
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
  dim3 grid2(num_seqs, hkv, rowtiles);
  chunk_attn<<<grid2, NT1, 0, stream>>>(
      (__half*)out.data_ptr(), ws, (const __half*)q.data_ptr(),
      (const __half*)k.data_ptr(), (const __half*)v.data_ptr(),
      cu_seqlens_q.data_ptr<int>(), seq_lens.data_ptr<int>(), hq, hkv,
      max_blocks * block_size, (float)sm_scale, num_splits, total_q,
      q.stride(0), q.stride(1), k.stride(0), k.stride(1), v.stride(0),
      v.stride(1), out.stride(0), out.stride(1));
}

#endif  // USE_ROCM
