// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// SplitQ decode on WMMA for RDNA3 (gfx11). Same structure as the int8
// per-token-head kernel in pth_decode_int8_rdna3_wmma.cu (independent waves,
// one (split, 16-query row tile) per wave, transposed orientation, lane pairs
// sharing A fragments through a DPP swap); see the notes there. What changes is
// the cache side:
//
//   QK  S^T[token][q] = K x Q^T over 16 k-steps of 16 dims. Steps 0-3 are the
//       RoPE dims (int8, fed as x + 1152), steps 4-15 the rotated NoPE codes
//       (fed as c + 1024). Each 64-dim segment has its own per-token scale,
//       so the steps accumulate into four tiles that are scaled and debiased
//       in the softmax.
//   PV  O^T[dim][q] = V^T x P^T with the rotated V codes fed as c + 1024; the
//       per-token V scale is folded into P, the bias removed with sum(P).
//
// The query is rotated (NoPE part) while it is staged into LDS; the output
// stays rotated and splitq_reduce rotates it back.

#include <cstdint>
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>

#if defined(USE_ROCM)
  #include <hip/hip_runtime.h>
  #include <hip/hip_bf16.h>
  #include <hip/hip_fp16.h>

  #if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1100__) && \
      !defined(__gfx1101__) && !defined(__gfx1102__) && !defined(__gfx1103__)
    #define SPLITQ_WMMA_STUB
  #endif

namespace splitq_wmma {

typedef _Float16 h16v __attribute__((ext_vector_type(16)));
typedef _Float16 h8v __attribute__((ext_vector_type(8)));
typedef _Float16 h2v __attribute__((ext_vector_type(2)));
typedef float f8v __attribute__((ext_vector_type(8)));
typedef uint32_t u8v __attribute__((ext_vector_type(8)));
typedef const int __attribute__((address_space(4))) cint;

constexpr int D = 256;
constexpr int R = 64;
constexpr int N = D - R;
constexpr int QG = 4;          // query tokens per block
constexpr int QROW = D + 8;    // padded LDS row (bank spread)
constexpr float PSCALE = 256.0f;
constexpr float RBIAS = 1152.0f;  // int8 byte x fed as x + 1152
constexpr uint32_t X = 0x80808080u;

template <int BITS>
struct Layout {
  static constexpr int OFF_KN = R;
  static constexpr int OFF_V = OFF_KN + N * BITS / 8;
  static constexpr int OFF_SC = OFF_V + D * BITS / 8;
  static constexpr float CBIAS = 1024.0f + ((1 << BITS) - 1) * 0.5f;
};

__device__ __forceinline__ uint32_t swap1(uint32_t v) {
  return (uint32_t)__builtin_amdgcn_update_dpp(0, (int)v, 0xB1, 0xF, 0xF,
                                               false);
}
__device__ __forceinline__ uint32_t xhalf(uint32_t v) {
  return __builtin_amdgcn_permlanex16(v, v, 0x76543210u, 0xFEDCBA98u, false,
                                      false);
}
__device__ __forceinline__ float xhalf(float v) {
  return __uint_as_float(xhalf(__float_as_uint(v)));
}

// Bytes b -> fp16 1024 + b, exact.
__device__ __forceinline__ h16v bytes_to_h16(uint32_t w0, uint32_t w1,
                                             uint32_t w2, uint32_t w3) {
  const uint32_t ws[4] = {w0, w1, w2, w3};
  u8v r;
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    r[2 * i] = __builtin_amdgcn_perm(0x64646464u, ws[i], 0x07010700u);
    r[2 * i + 1] = __builtin_amdgcn_perm(0x64646464u, ws[i], 0x07030702u);
  }
  return __builtin_bit_cast(h16v, r);
}

__device__ __forceinline__ h16v ld16(const _Float16* p) {
  const h8v lo = *(const h8v*)p;
  const h8v hi = *(const h8v*)(p + 8);
  return __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                                 12, 13, 14, 15);
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

// Codes of the 16 dims of chunk c of a packed region of `ndim` dims, as four
// dwords of bytes: dims 0-3, 4-7, 8-11, 12-15 of the chunk.
template <int BITS>
__device__ __forceinline__ void chunk_codes(const uint8_t* region, int ndim,
                                            int c, uint32_t o[4]) {
  if constexpr (BITS == 4) {
    const uint2 w = *(const uint2*)(region + 8 * c);
    o[0] = w.x & 0x0F0F0F0Fu;
    o[1] = (w.x >> 4) & 0x0F0F0F0Fu;
    o[2] = w.y & 0x0F0F0F0Fu;
    o[3] = (w.y >> 4) & 0x0F0F0F0Fu;
  } else {
    const uint32_t lo = *(const uint32_t*)(region + 4 * c);
    const uint32_t hi =
        *(const uint32_t*)(region + ndim / 4 + 4 * (c >> 1)) >> (4 * (c & 1));
  #pragma unroll
    for (int i = 0; i < 4; ++i)
      o[i] = ((lo >> (2 * i)) & 0x03030303u) |
             (((hi >> i) & 0x01010101u) << 2);
  }
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

template <int BITS, int NSB, typename QT>
__global__ __launch_bounds__(64 * NSB) void decode_wmma(
    const QT* __restrict__ Q, const uint8_t* __restrict__ cache,
    const int* __restrict__ block_table, const int* __restrict__ q_to_req,
    const int* __restrict__ q_to_klen, float* __restrict__ mid_o,
    const int* __restrict__ nope_signs, float sm_scale, int num_q,
    int num_q_heads, int num_kv_heads, int block_size, int max_blocks,
    int num_reqs, int num_phys_blocks, int num_splits, int min_tps,
    int num_row_tiles, int64_t sq0, int64_t sq1, int64_t scb, int64_t sch, int64_t sct,
    int64_t smo, int64_t smh, int64_t sms) {
  #ifndef SPLITQ_WMMA_STUB
  using L = Layout<BITS>;
  const int grp = blockIdx.x;
  const int kvh = blockIdx.y;
  const int tid = threadIdx.x;
  const int w = __builtin_amdgcn_readfirstlane(tid >> 5), lane = tid & 31;
  // Two row tiles when the group can hold more than 16 query vectors;
  // otherwise every wave takes its own split.
  const int rt = num_row_tiles == 2 ? (w & 1) : 0;
  const int splits_per_block = 2 * NSB / num_row_tiles;
  const int si = blockIdx.z * splits_per_block + (num_row_tiles == 2 ? w >> 1 : w);
  const int j = lane & 15, h = lane >> 4;
  const bool useful = (j & 1) == h;
  const int rr = useful ? j : (j ^ 1);
  const int hpk = num_q_heads / num_kv_heads;
  const int max_kv = max_blocks * block_size;
  const float sm_log2 = sm_scale * 1.4426950408889634f;

  __shared__ __attribute__((aligned(16))) _Float16 sQ[32][QROW];

  // Stage the group's queries, NoPE part rotated: lane owns 8 dims.
  for (int v = w; v < 32; v += 2 * NSB) {
    const int r = v / hpk, g = v % hpk;
    const int qi = grp * QG + r;
    float x[8];
    const bool live = v < QG * hpk && qi < num_q;
    #pragma unroll
    for (int i = 0; i < 8; ++i)
      x[i] = live ? to_f<QT>(Q[(int64_t)qi * sq0 +
                              (int64_t)(kvh * hpk + g) * sq1 + lane * 8 + i])
                  : 0.0f;
    if (lane >= R / 8) {
    #pragma unroll
      for (int i = 0; i < 8; ++i) x[i] *= sign_of(nope_signs, lane * 8 + i - R);
    }
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
    for (int m = 1; m <= 4; m <<= 1) {
      const bool hi = lane & m;
    #pragma unroll
      for (int i = 0; i < 8; ++i) {
        const float o = __shfl_xor(x[i], m);
        x[i] = hi ? o - x[i] : x[i] + o;
      }
    }
    // RoPE lanes took part in the butterflies on their own 8-lane group;
    // reload their raw values.
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      float raw = live ? to_f<QT>(Q[(int64_t)qi * sq0 +
                                    (int64_t)(kvh * hpk + g) * sq1 +
                                    lane * 8 + i])
                       : 0.0f;
      sQ[v][lane * 8 + i] = (_Float16)(lane >= R / 8 ? x[i] * 0.125f : raw);
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

  // Per-segment sums of this lane's query (debiasing): half h sums dims
  // [128 h, 128 h + 128) as two 64-dim segments; the halves swap.
  const int qv = rt * 16 + j;
  float bq[4];
  {
    const h2v one = {(_Float16)1.0f, (_Float16)1.0f};
    float s0 = 0.f, s1 = 0.f;
    #pragma unroll
    for (int s = 0; s < 8; ++s) {
      const h8v a = *(const h8v*)&sQ[qv][128 * h + 8 * s];
      const h8v b = *(const h8v*)&sQ[qv][128 * h + 64 + 8 * s];
    #pragma unroll
      for (int i = 0; i < 8; i += 2) {
        s0 = __builtin_amdgcn_fdot2((h2v){a[i], a[i + 1]}, one, s0, false);
        s1 = __builtin_amdgcn_fdot2((h2v){b[i], b[i + 1]}, one, s1, false);
      }
    }
    const float o0 = xhalf(s0), o1 = xhalf(s1);
    bq[0] = (h ? o0 : s0) * RBIAS;
    bq[1] = (h ? o1 : s1) * L::CBIAS;
    bq[2] = (h ? s0 : o0) * L::CBIAS;
    bq[3] = (h ? s1 : o1) * L::CBIAS;
  }

  int seg0 = 0;
  while (seg0 < QG && rreq[seg0] >= 0) {
    int seg1 = seg0 + 1;
    while (seg1 < QG && rreq[seg1] == rreq[seg0]) ++seg1;
    const int req = rreq[seg0];
    int Lmax = 0;
    for (int r = seg0; r < seg1; ++r) Lmax = max(Lmax, rlen[r]);
    const int qv_lo = seg0 * hpk, qv_hi = seg1 * hpk;
    int tps = (Lmax + num_splits - 1) / num_splits;
    if (tps < min_tps) tps = min_tps;
    tps = (tps + 15) & ~15;

    const bool tile_live = qv_hi > rt * 16 && qv_lo < rt * 16 + 16;
    const bool qlive = qv >= qv_lo && qv < qv_hi;
    if (blockIdx.z * splits_per_block * tps >= Lmax) {
      if (tile_live && si < num_splits && h == 0 && qlive) {
        int qx = qv;
        asm volatile("" : "+v"(qx));
        float* op = mid_o + (int64_t)(grp * QG + qx / hpk) * smo +
                    (int64_t)(kvh * hpk + qx % hpk) * smh + (int64_t)si * sms;
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
      f8v O[16];
    #pragma unroll
      for (int c = 0; c < 16; ++c) O[c] = (f8v){0, 0, 0, 0, 0, 0, 0, 0};

      const uint8_t* cache_h = cache + (int64_t)kvh * sch;
      auto pickw = [&](uint32_t v, int t) {
        const uint32_t ev = __builtin_amdgcn_readlane(v, t);
        const uint32_t od = __builtin_amdgcn_readlane(v, t + 1);
        return h ? od : ev;
      };
      auto tile_ptr = [&](int pos) {
        const int lb = pos / block_size;
        int pb = ((cint*)block_table)[req * max_blocks + lb];
        if (pb < 0 || pb >= num_phys_blocks) pb = 0;
        return cache_h + (int64_t)pb * scb +
               (int64_t)(pos - lb * block_size) * sct;
      };

      // Raw K loads, one tile ahead. Each lane of a pair owns 8 of the 16
      // k-steps of token rr: slots 0-1 are RoPE steps (16 bytes), slots 2-7
      // NoPE chunks (8 bytes). The useful lane's slots are steps
      // {0,1,4..9}, its partner's {2,3,10..15}; the partner's reach the
      // useful lane through the DPP swap.
      uint4 kr_raw[2];
      uint2 kn_raw[6];
      // Scales of token j (lane j), raw fp16 pairs: {kr, k0}, {k1, k2}, {v}.
      uint32_t sc_a = 0, sc_b = 0, sc_v = 0;
      auto load_k = [&](const uint8_t* tile, float dep) {
        uint32_t off = (uint32_t)(rr * (int)sct);
        asm volatile("" : "+v"(off) : "v"(dep));
        const uint8_t* pk = tile + off;
        const int rope0 = useful ? 0 : 2;
        const int chunk0 = useful ? 0 : 6;
    #pragma unroll
        for (int k = 0; k < 2; ++k)
          kr_raw[k] = *(const uint4*)__builtin_assume_aligned(
              pk + 16 * (rope0 + k), 16);
    #pragma unroll
        for (int k = 0; k < 6; ++k) {
          const int c = chunk0 + k;
          if constexpr (BITS == 4) {
            kn_raw[k] = *(const uint2*)(pk + L::OFF_KN + 8 * c);
          } else {
            kn_raw[k].x = *(const uint32_t*)(pk + L::OFF_KN + 4 * c);
            kn_raw[k].y =
                *(const uint32_t*)(pk + L::OFF_KN + N / 4 + 4 * (c >> 1)) >>
                (4 * (c & 1));
          }
        }
        const uint8_t* ps = tile + (int64_t)j * sct + L::OFF_SC;
        const uint2 u = *(const uint2*)ps;
        sc_a = u.x;
        sc_b = u.y;
        sc_v = *(const uint16_t*)(ps + 8);
      };
      // Raw V loads: lane rr's 16 dims of the even (useful) or odd tokens.
      uint2 vraw[8];
      auto load_v = [&](const uint8_t* tile, float dep) {
        uint32_t off = (uint32_t)((useful ? 0 : 1) * (int)sct);
        asm volatile("" : "+v"(off) : "v"(dep));
        const uint8_t* pv = tile + off + L::OFF_V;
    #pragma unroll
        for (int i = 0; i < 8; ++i) {
          const uint8_t* p = pv + (int64_t)(2 * i) * sct;
          if constexpr (BITS == 4) {
            vraw[i] = *(const uint2*)(p + 8 * rr);
          } else {
            vraw[i].x = *(const uint32_t*)(p + 4 * rr);
            vraw[i].y = *(const uint32_t*)(p + D / 4 + 4 * (rr >> 1));
          }
        }
      };
      // Byte codes of a 16-dim NoPE chunk from its raw words (4-bit: the two
      // packed words; 3-bit: 2-bit plane word, 1-bit plane word pre-shifted).
      auto codes4 = [&](uint2 raw, uint32_t o[4]) {
        if constexpr (BITS == 4) {
          o[0] = raw.x & 0x0F0F0F0Fu;
          o[1] = (raw.x >> 4) & 0x0F0F0F0Fu;
          o[2] = raw.y & 0x0F0F0F0Fu;
          o[3] = (raw.y >> 4) & 0x0F0F0F0Fu;
        } else {
    #pragma unroll
          for (int i = 0; i < 4; ++i)
            o[i] = ((raw.x >> (2 * i)) & 0x03030303u) |
                   (((raw.y >> i) & 0x01010101u) << 2);
        }
      };

      if (start < end) {
        const uint8_t* t0 = tile_ptr(start);
        load_k(t0, 0.f);
        load_v(t0, 0.f);
      }

      for (int base = start; base < end; base += 16) {
        auto lo16 = [](uint32_t x) {
          return __half2float(__ushort_as_half((unsigned short)(x & 0xFFFF)));
        };
        auto hi16 = [](uint32_t x) {
          return __half2float(__ushort_as_half((unsigned short)(x >> 16)));
        };
        const uint32_t sa_c = sc_a, sb_c = sc_b, sv_c = sc_v;

        // QK. Iteration k feeds the useful lane's slot k and, through the
        // swap, the partner's slot k.
        f8v S[4];
    #pragma unroll
        for (int g = 0; g < 4; ++g) S[g] = (f8v){0, 0, 0, 0, 0, 0, 0, 0};
        int qoff = qv * QROW;
        asm volatile("" : "+v"(qoff));
        constexpr int kStepOwn[8] = {0, 1, 4, 5, 6, 7, 8, 9};
        constexpr int kStepPar[8] = {2, 3, 10, 11, 12, 13, 14, 15};
    #pragma unroll
        for (int k = 0; k < 8; ++k) {
          uint32_t wd[4];
          if (k < 2) {
            wd[0] = kr_raw[k].x ^ X;
            wd[1] = kr_raw[k].y ^ X;
            wd[2] = kr_raw[k].z ^ X;
            wd[3] = kr_raw[k].w ^ X;
          } else {
            codes4(kn_raw[k - 2], wd);
          }
          const int so = kStepOwn[k], sp = kStepPar[k];
          const int go = so < 4 ? 0 : (so - 4) / 4 + 1;
          const int gp = sp < 4 ? 0 : (sp - 4) / 4 + 1;
          // One A and one B fragment live at a time. Each read waits for the
          // previous WMMA: without the dependence the compiler hoists all Q
          // fragments and spills.
          asm volatile("" : "+v"(qoff) : "v"(S[gp][0]));
          {
            const h16v b = ld16(&sQ[0][0] + qoff + 16 * so);
            const h16v a = bytes_to_h16(wd[0], wd[1], wd[2], wd[3]);
            S[go] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, S[go]);
          }
          __builtin_amdgcn_sched_barrier(0);
          asm volatile("" : "+v"(qoff) : "v"(S[go][0]));
          {
            const h16v b = ld16(&sQ[0][0] + qoff + 16 * sp);
            const h16v a = bytes_to_h16(swap1(wd[0]), swap1(wd[1]),
                                        swap1(wd[2]), swap1(wd[3]));
            S[gp] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, S[gp]);
          }
          __builtin_amdgcn_sched_barrier(0);
        }

        // Next tile's K and scales now: softmax and PV hide them.
        const bool next = base + 16 < end;
        const uint8_t* tn = next ? tile_ptr(base + 16) : nullptr;
        if (next) load_k(tn, S[0][0]);

        float sc[8];
        float mloc = -INFINITY;
    #pragma unroll
        for (int e = 0; e < 8; ++e) {
          const int t = base + 2 * e + h;
          const bool ok = t < end && t < qlen;
          const uint32_t a = pickw(sa_c, 2 * e), b = pickw(sb_c, 2 * e);
          const float v = lo16(a) * (S[0][e] - bq[0]) +
                          hi16(a) * (S[1][e] - bq[1]) +
                          lo16(b) * (S[2][e] - bq[2]) +
                          hi16(b) * (S[3][e] - bq[3]);
          sc[e] = ok ? v * sm_log2 : -INFINITY;
          mloc = fmaxf(mloc, sc[e]);
        }
        const float mn = fmaxf(m, fmaxf(mloc, xhalf(mloc)));
        float alpha = 1.0f;
        if (mn > m)
          alpha = (m == -INFINITY) ? 0.0f : __builtin_amdgcn_exp2f(m - mn);
        m = mn;
        uint32_t pd[4];
        float psum = 0.0f, lsum = 0.0f;
    #pragma unroll
        for (int e = 0; e < 8; e += 2) {
          const float v0 = lo16(pickw(sv_c, 2 * e));
          const float v1 = lo16(pickw(sv_c, 2 * e + 2));
          const float p0 =
              sc[e] == -INFINITY ? 0.0f : __builtin_amdgcn_exp2f(sc[e] - mn);
          const float p1 = sc[e + 1] == -INFINITY
                               ? 0.0f
                               : __builtin_amdgcn_exp2f(sc[e + 1] - mn);
          lsum += p0 + p1;
          const _Float16 h0 = (_Float16)(p0 * v0 * PSCALE);
          const _Float16 h1 = (_Float16)(p1 * v1 * PSCALE);
          psum += (float)h0 + (float)h1;
          pd[e >> 1] = __builtin_bit_cast(uint32_t, (h2v){h0, h1});
        }
        l = l * alpha + lsum;
        cp = cp * alpha + psum;
        if (__builtin_amdgcn_ballot_w32(alpha != 1.0f)) {
    #pragma unroll
          for (int c = 0; c < 16; ++c) O[c] *= alpha;
        }
        u8v pf;
    #pragma unroll
        for (int i = 0; i < 4; ++i) {
          const uint32_t o = xhalf(pd[i]);
          pf[i] = h ? o : pd[i];
          pf[4 + i] = h ? pd[i] : o;
        }
        const h16v pb16 = __builtin_bit_cast(h16v, pf);

        // Codes of dims 4q..4q+3 of this lane's chunk, token i, as bytes.
        auto vq = [&](int i, int q) -> uint32_t {
          if constexpr (BITS == 4) {
            const uint32_t w = q < 2 ? vraw[i].x : vraw[i].y;
            return (w >> (4 * (q & 1))) & 0x0F0F0F0Fu;
          } else {
            const uint32_t hi = vraw[i].y >> (4 * (rr & 1));
            return ((vraw[i].x >> (2 * q)) & 0x03030303u) |
                   (((hi >> q) & 0x01010101u) << 2);
          }
        };
    #pragma unroll
        for (int q = 0; q < 4; ++q) {
          uint32_t t0[4], t1[4];
          tr4(vq(0, q), vq(1, q), vq(2, q), vq(3, q), t0);
          tr4(vq(4, q), vq(5, q), vq(6, q), vq(7, q), t1);
    #pragma unroll
          for (int b = 0; b < 4; ++b) {
            const int c = 4 * q + b;
            uint32_t o0 = t0[b], o1 = t1[b];
            if (c >= 2)
              asm volatile("" : "+v"(o0), "+v"(o1) : "v"(O[c - 2][0]));
            const h16v a = bytes_to_h16(o0, o1, swap1(o0), swap1(o1));
            O[c] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, pb16, O[c]);
          }
        }
        // Next tile's V, issued after the last PV WMMA.
        if (next) load_v(tn, O[15][0]);
      }

      l += xhalf(l);
      cp += xhalf(cp);
      if (qlive) {
        int qx = qv;
        asm volatile("" : "+v"(qx));
        float* op = mid_o + (int64_t)(grp * QG + qx / hpk) * smo +
                    (int64_t)(kvh * hpk + qx % hpk) * smh + (int64_t)si * sms;
        const bool any = m != -INFINITY;
        const float corr = L::CBIAS * cp;
    #pragma unroll
        for (int e = 0; e < 8; ++e) {
          float* dp = op + 16 * (2 * e + h);
    #pragma unroll
          for (int c = 0; c < 16; c += 4) {
            float4 v;
            v.x = any ? (O[c][e] - corr) * (1.0f / PSCALE) : 0.0f;
            v.y = any ? (O[c + 1][e] - corr) * (1.0f / PSCALE) : 0.0f;
            v.z = any ? (O[c + 2][e] - corr) * (1.0f / PSCALE) : 0.0f;
            v.w = any ? (O[c + 3][e] - corr) * (1.0f / PSCALE) : 0.0f;
            *(float4*)(dp + c) = v;
          }
        }
        if (h == 0) {
          op[D] = m;
          op[D + 1] = l;
        }
      }
    }
    seg0 = seg1;
  }
  #endif  // SPLITQ_WMMA_STUB
}

}  // namespace splitq_wmma

// Splits per row group and minimum tokens per split, as in the int8 kernel.
constexpr int kSqWmmaSplits = 192;
constexpr int kSqWmmaMinTps = 64;
constexpr int kSqWmmaNsb = 2;

// Launches the WMMA decode when the shape is covered; returns the number of
// splits it wrote (for the reduce), or 0 when the caller must use its own
// kernel.
int splitq_decode_wmma(torch::Tensor query, torch::Tensor cache,
                       torch::Tensor block_table, torch::Tensor q_to_req,
                       torch::Tensor q_to_klen, torch::Tensor mid_o,
                       torch::Tensor nope_signs, double sm_scale,
                       int64_t num_kv_splits, int64_t bits) {
  using namespace splitq_wmma;
  static const bool arch_ok = [] {
    const auto* prop = at::cuda::getCurrentDeviceProperties();
    return std::string(prop->gcnArchName).rfind("gfx11", 0) == 0;
  }();
  const int num_q = query.size(0);
  const int num_q_heads = query.size(1);
  const int num_kv_heads = cache.size(1);
  const int hpk = num_q_heads / num_kv_heads;
  const bool eligible = arch_ok && query.size(2) == D && query.stride(2) == 1 &&
                        hpk * QG <= 32 && cache.size(2) % 16 == 0 &&
                        cache.stride(2) % 8 == 0 && num_kv_splits >= 1;
  if (!eligible) return 0;
  const int ns = std::min<int>(kSqWmmaSplits, (int)num_kv_splits);
  if (num_q == 0) return ns;
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  const int nrt = std::min(num_q, QG) * hpk > 16 ? 2 : 1;
  const int spb = 2 * kSqWmmaNsb / nrt;
  dim3 grid((num_q + QG - 1) / QG, num_kv_heads, (ns + spb - 1) / spb);
  #define SQW(B, T)                                                           \
    decode_wmma<B, kSqWmmaNsb, T><<<grid, 64 * kSqWmmaNsb, 0, stream>>>(      \
        (const T*)query.data_ptr(), (const uint8_t*)cache.data_ptr(),         \
        block_table.data_ptr<int>(), q_to_req.data_ptr<int>(),                \
        q_to_klen.data_ptr<int>(), mid_o.data_ptr<float>(),                   \
        nope_signs.data_ptr<int>(), (float)sm_scale, num_q, num_q_heads,      \
        num_kv_heads, cache.size(2), block_table.size(1), block_table.size(0), \
        cache.size(0), ns, kSqWmmaMinTps, nrt, query.stride(0),               \
        query.stride(1),                                                      \
        cache.stride(0), cache.stride(1), cache.stride(2), mid_o.stride(0),   \
        mid_o.stride(1), mid_o.stride(2))
  const bool bf = query.dtype() == at::kBFloat16;
  if (bits == 4) {
    if (bf) SQW(4, __hip_bfloat16); else SQW(4, half);
  } else {
    if (bf) SQW(3, __hip_bfloat16); else SQW(3, half);
  }
  #undef SQW
  return ns;
}

#endif  // USE_ROCM
