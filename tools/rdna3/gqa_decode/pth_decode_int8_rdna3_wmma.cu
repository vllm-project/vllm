// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// INT8 per-token-head decode attention for RDNA3 (gfx1100) on WMMA.
//
// The previous design ran 4 waves in lock-step over a shared LDS stage, with 3
// barriers per 16-token tile and 2.5 waves per SIMD, so every phase of the
// chain was exposed. Here every wave is independent -- one (split, row tile)
// per wave, no barrier in the token loop -- and latency is hidden by occupancy
// (256 VGPRs, no spill in the loop) plus a one-tile software prefetch inside the wave.
//
// Measured on one 7900 XTX, Qwen3.5-27B at TP4 (6 q heads, 1 KV head, head
// size 256), us per call, lock-step design -> this one:
//   284k ctx, 4 rows (1 request, MTP k=3): 263 -> 211
//   284k ctx, 12 rows (3 requests):        678 -> 495
//   2k ctx, 12 rows:                        44 -> 45-50
//
// Two facts about v_wmma_f32_16x16x16_f16 on wave32, measured with a probe:
//   D[i][j] is in lane j + 16*(i&1), element i>>1, and half h of the wave
//   computes the rows of parity h from ITS OWN copy of A and B. So
//   - B must be complete in both halves;
//   - A lane L is only read when (L%16)%2 == L/16. The other lane of each
//     adjacent pair (L, L^1) is free: it loads the other half of the row and
//     hands it over with one DPP swap.
//
// Orientation is transposed so that all per-query state is lane-local:
//   QK:  S^T[token][q] = K (A, rows = tokens) x Q^T (B, from LDS)
//   PV:  O^T[dim][q]   = V^T (A, rows = dims) x P^T (B, columns = q)
// Lane (j, h) owns query vector j: max, sum, rescale and the bias correction
// need no cross-lane traffic except one permlanex16 per tile for the max and
// four to complete P^T in both halves.
//
// PV contracts over tokens in the order tau(k) = 2k (k < 8), 2(k-8)+1 (k >= 8):
// even tokens are the ones half 0 already holds, odd ones half 1.

#include <cstdint>
#include <torch/all.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef _Float16 h16v __attribute__((ext_vector_type(16)));
typedef _Float16 h8v __attribute__((ext_vector_type(8)));
typedef _Float16 h2v __attribute__((ext_vector_type(2)));
typedef float f8v __attribute__((ext_vector_type(8)));
typedef uint32_t u8v __attribute__((ext_vector_type(8)));
typedef const float __attribute__((address_space(4))) cfloat;
typedef const int __attribute__((address_space(4))) cint;

constexpr int HS = 256;
constexpr int G = 4;
constexpr int QROW = HS + 8;  // 528 B rows: 16 lanes x b128 hit 64 distinct banks
constexpr float PSCALE = 4096.0f;
constexpr float BIAS = 1152.0f;

__device__ __forceinline__ uint32_t swap1(uint32_t v) {
  return (uint32_t)__builtin_amdgcn_update_dpp(0, (int)v, 0xB1, 0xF, 0xF, false);
}
__device__ __forceinline__ uint32_t xhalf(uint32_t v) {
  return __builtin_amdgcn_permlanex16(v, v, 0x76543210u, 0xFEDCBA98u, false, false);
}
__device__ __forceinline__ float xhalf(float v) {
  return __uint_as_float(xhalf(__float_as_uint(v)));
}

// Biased bytes (x ^ 0x80) -> fp16 x + 1152, exact.
__device__ __forceinline__ h16v b8_to_h16(uint32_t w0, uint32_t w1, uint32_t w2,
                                          uint32_t w3) {
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

struct u4a {
  uint32_t x, y, z, w;
};
// Token rows are only 4-byte aligned (520 B apart); gfx1100 serves unaligned
// b128 global loads (measured), so the 16-byte claim only picks the opcode.
__device__ __forceinline__ u4a ldu(const int8_t* p) {
  const uint4 v = *(const uint4*)__builtin_assume_aligned(p, 16);
  return u4a{v.x, v.y, v.z, v.w};
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

template <int NSB>
__global__ __launch_bounds__(64 * NSB) void decode_int8_wmma(
    const __half* __restrict__ Q, const int8_t* __restrict__ K_cache,
    const int8_t* __restrict__ V_cache, const float* __restrict__ K_scale,
    const float* __restrict__ V_scale, const int* __restrict__ block_table,
    const int* __restrict__ q_to_req, const int* __restrict__ q_to_klen,
    float* __restrict__ mid_o, float sm_scale, int num_q, int num_q_heads,
    int num_kv_heads, int block_size, int max_blocks, int num_reqs,
    int num_phys_blocks, int num_splits, int min_tps, int64_t sq0, int64_t sq1,
    int64_t skb, int64_t sks, int64_t svb, int64_t svs, int64_t ssb,
    int64_t sss, int64_t svsb, int64_t svss, int64_t smo, int64_t smh,
    int64_t sms) {
  const int grp = blockIdx.x;
  const int tid = threadIdx.x;
  // Wave-uniform, but the compiler cannot tell: without readfirstlane the
  // split, tile position and block bases all live in VGPRs.
  const int w = __builtin_amdgcn_readfirstlane(tid >> 5), lane = tid & 31;
  const int rt = w & 1;
  const int si = blockIdx.z * NSB + (w >> 1);
  const int j = lane & 15, h = lane >> 4;
  const bool useful = (j & 1) == h;  // this lane's A rows are read
  const int rr = useful ? j : (j ^ 1);
  const int hpk = num_q_heads / num_kv_heads;
  const int max_kv = max_blocks * block_size;
  const float sm_log2 = sm_scale * 1.4426950408889634f;
  const uint32_t X = 0x80808080u;

  __shared__ __attribute__((aligned(16))) _Float16 sQ[32][QROW];

  int rreq[G], rlen[G];
#pragma unroll
  for (int r = 0; r < G; ++r) {
    const int qi = grp * G + r;
    int rq = -1, ln = 0;
    if (qi < num_q) {
      rq = q_to_req[qi];
      ln = q_to_klen[qi];
      if (rq < 0 || rq >= num_reqs) { rq = 0; ln = 0; }
      if (ln < 0) ln = 0;
      if (ln > max_kv) ln = max_kv;
    }
    rreq[r] = rq;
    rlen[r] = ln;
  }

  int seg0 = 0;
  while (seg0 < G && rreq[seg0] >= 0) {
    int seg1 = seg0 + 1;
    while (seg1 < G && rreq[seg1] == rreq[seg0]) ++seg1;
    const int req = rreq[seg0];
    int L = 0;
    for (int r = seg0; r < seg1; ++r) L = max(L, rlen[r]);
    const int qv_lo = seg0 * hpk, qv_hi = seg1 * hpk;
    int tps = (L + num_splits - 1) / num_splits;
    if (tps < min_tps) tps = min_tps;
    tps = (tps + 15) & ~15;

    // A block whose splits are all past the context (short contexts) only
    // marks them empty for the reduce: no Q staging, no barrier. The test is
    // block-uniform, so every wave takes the same path.
    if (blockIdx.z * NSB * tps >= L) {
      const bool tlive = qv_hi > rt * 16 && qv_lo < rt * 16 + 16;
      const int qv = rt * 16 + j;
      if (tlive && si < num_splits && h == 0 && qv >= qv_lo && qv < qv_hi) {
        float* op = mid_o + (int64_t)(grp * G + qv / hpk) * smo +
                    (int64_t)(qv % hpk) * smh + (int64_t)si * sms;
        op[HS] = -INFINITY;
        op[HS + 1] = 0.0f;
      }
      seg0 = seg1;
      continue;
    }

    __syncthreads();  // the previous segment is done with sQ
    for (int c = tid; c < 32 * (HS / 8); c += 64 * NSB) {
      const int qv = c >> 5, ch = c & 31;
      int4 val = make_int4(0, 0, 0, 0);
      if (qv >= qv_lo && qv < qv_hi)
        val = *(const int4*)(Q + (int64_t)(grp * G + qv / hpk) * sq0 +
                             (int64_t)(qv % hpk) * sq1 + 8 * ch);
      *(int4*)&sQ[qv][8 * ch] = val;
    }
    __syncthreads();

    const bool tile_live = qv_hi > rt * 16 && qv_lo < rt * 16 + 16;
    if (tile_live && si < num_splits) {
      const int qv = rt * 16 + j;
      const bool qlive = qv >= qv_lo && qv < qv_hi;
      const int qrow = qv / hpk;
      int qlen = 0;
#pragma unroll
      for (int r = 0; r < G; ++r)
        if (qlive && qrow == r) qlen = rlen[r];
      // sum(q) for the bias correction: each half sums 128 dims with fdot2
      // against ones (64 ops instead of 512 per lane), then the halves meet.
      float bq = 0.0f;
      {
        const h2v one = {(_Float16)1.0f, (_Float16)1.0f};
#pragma unroll 4
        for (int s = 0; s < HS / 16; ++s) {
          const h8v x = *(const h8v*)&sQ[qv][128 * h + 8 * s];
#pragma unroll
          for (int i = 0; i < 8; i += 2)
            bq = __builtin_amdgcn_fdot2((h2v){x[i], x[i + 1]}, one, bq, false);
        }
        bq += xhalf(bq);
      }
      bq *= BIAS;

      const int start = si * tps;
      const int end = min(start + tps, L);

      float m = -INFINITY, l = 0.0f, cp = 0.0f;
      f8v O[16];
#pragma unroll
      for (int c = 0; c < 16; ++c) O[c] = (f8v){0, 0, 0, 0, 0, 0, 0, 0};

      // Tiles are 16-aligned and block_size % 16 == 0: a tile never straddles
      // two blocks. Position tracked incrementally (wave-uniform).
      int lb = start / block_size;
      int s0 = start - lb * block_size;
      auto phys = [&](int b) {
        int pb = ((cint*)block_table)[req * max_blocks + b];
        if (pb < 0 || pb >= num_phys_blocks) pb = 0;
        return pb;
      };
      int pb = 0;
      u4a kr[8], vr[8];
      // Uniform 64-bit bases (SGPRs) plus one 32-bit per-lane offset, so the
      // loads use saddr + imm and cost one VGPR of address each. `dep` pins
      // the issue point: the offset passes through an asm that consumes it.
      const uint32_t koff_l = (uint32_t)(rr * (int)sks + (useful ? 0 : 128));
      const uint32_t voff_l = (uint32_t)((useful ? 0 : 1) * (int)svs + 16 * rr);
      auto load_k = [&](int pbx, int s0x, float dep) {
        const int8_t* kb = K_cache + (int64_t)pbx * skb + (int64_t)s0x * sks;
        uint32_t off = koff_l;
        asm volatile("" : "+v"(off) : "v"(dep));
#pragma unroll
        for (int s = 0; s < 8; ++s) kr[s] = ldu(kb + off + 16 * s);
      };
      auto load_v = [&](int pbx, int s0x, float dep) {
        const int8_t* vb = V_cache + (int64_t)pbx * svb + (int64_t)s0x * svs;
        uint32_t off = voff_l;
        asm volatile("" : "+v"(off) : "v"(dep));
#pragma unroll
        for (int i = 0; i < 8; ++i) vr[i] = ldu(vb + (int64_t)(2 * i) * svs + off);
      };
      // Scales: lane L holds token L%16's; element e of lane (j, h) takes
      // token 2e+h's with readlane. No array: a per-lane select between two
      // array elements is folded into a per-lane index, and the array goes to
      // scratch (measured). Loaded one tile ahead, like K.
      float ksl = 0.0f, vsl = 0.0f;
      auto load_s = [&](int pbx, int s0x, float dep) {
        uint32_t o = (uint32_t)j;
        asm volatile("" : "+v"(o) : "v"(dep));
        ksl = K_scale[(int64_t)pbx * ssb + (int64_t)(s0x + (int)o) * sss];
        vsl = V_scale[(int64_t)pbx * svsb + (int64_t)(s0x + (int)o) * svss];
      };
      auto pick = [&](float v, int t) {
        const float ev = __builtin_bit_cast(float, __builtin_amdgcn_readlane(__builtin_bit_cast(int, v), t));
        const float od = __builtin_bit_cast(float, __builtin_amdgcn_readlane(__builtin_bit_cast(int, v), t + 1));
        return h ? od : ev;
      };
      if (start < end) {
        pb = phys(lb);
        load_k(pb, s0, 0.0f);
        load_s(pb, s0, 0.0f);
        load_v(pb, s0, 0.0f);
      }

      for (int base = start; base < end; base += 16) {
        const float ks_c = ksl, vs_c = vsl;

        // QK: S^T = K x Q^T. Lane pair (rr, rr^1): the useful lane loaded
        // k-steps 0-7 of token rr, its partner 8-15.
        f8v S = {0, 0, 0, 0, 0, 0, 0, 0};
        // Q is loop-invariant: without an opaque address the compiler hoists
        // all 16 fragments out of the loop (128 VGPRs) and spills.
        int qoff = qv * QROW;
        asm volatile("" : "+v"(qoff));
        // One k-step of Q look-ahead; the barrier after each WMMA stops the
        // scheduler from pulling all 16 LDS reads up front (13 fragments live
        // at once, measured, and O spilled to make room).
        h16v bq_n = ld16(&sQ[0][0] + qoff);
#pragma unroll
        for (int s = 0; s < 16; ++s) {
          uint32_t a0, a1, a2, a3;
          if (s < 8) {
            a0 = kr[s].x; a1 = kr[s].y; a2 = kr[s].z; a3 = kr[s].w;
          } else {
            a0 = swap1(kr[s - 8].x); a1 = swap1(kr[s - 8].y);
            a2 = swap1(kr[s - 8].z); a3 = swap1(kr[s - 8].w);
          }
          const h16v a = b8_to_h16(a0 ^ X, a1 ^ X, a2 ^ X, a3 ^ X);
          const h16v b = bq_n;
          // The read of step s+1 waits for the WMMA of step s-1: a data
          // dependence the compiler cannot hoist away (sched_barrier alone
          // did not stop it).
          asm volatile("" : "+v"(qoff) : "v"(S[0]));
          if (s < 15) bq_n = ld16(&sQ[0][0] + qoff + 16 * (s + 1));
          S = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, S);
          __builtin_amdgcn_sched_barrier(0);
        }

        // Next tile: position and K prefetch.
        const bool next = base + 16 < end;
        int pbn = pb, s0n = s0 + 16;
        if (s0n >= block_size) { s0n = 0; ++lb; }
        if (next) {
          if (s0n == 0) pbn = phys(lb);
          // K and scales of the next tile now: softmax and PV hide them.
          load_k(pbn, s0n, S[0]);
          load_s(pbn, s0n, S[0]);
        }

        // Softmax over this lane's 8 tokens (2e + h) for query j.
        float sc[8];
        float mloc = -INFINITY;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const int t = base + 2 * e + h;
          const bool ok = t < end && t < qlen;
          const float k_s = pick(ks_c, 2 * e);
          sc[e] = ok ? (S[e] - bq) * k_s * sm_log2 : -INFINITY;
          mloc = fmaxf(mloc, sc[e]);
        }
        const float mn = fmaxf(m, fmaxf(mloc, xhalf(mloc)));
        float alpha = 1.0f;
        if (mn > m) alpha = (m == -INFINITY) ? 0.0f : __builtin_amdgcn_exp2f(m - mn);
        m = mn;
        uint32_t pd[4];
        float psum = 0.0f, lsum = 0.0f;
#pragma unroll
        for (int e = 0; e < 8; e += 2) {
          const float v0 = pick(vs_c, 2 * e);
          const float v1 = pick(vs_c, 2 * e + 2);
          const float p0 = sc[e] == -INFINITY ? 0.0f : __builtin_amdgcn_exp2f(sc[e] - mn);
          const float p1 = sc[e + 1] == -INFINITY ? 0.0f : __builtin_amdgcn_exp2f(sc[e + 1] - mn);
          lsum += p0 + p1;
          const _Float16 h0 = (_Float16)(p0 == 0.0f ? 0.0f : p0 * v0 * PSCALE);
          const _Float16 h1 = (_Float16)(p1 == 0.0f ? 0.0f : p1 * v1 * PSCALE);
          psum += (float)h0 + (float)h1;
          pd[e >> 1] = __builtin_bit_cast(uint32_t, (h2v){h0, h1});
        }
        l = l * alpha + lsum;
        cp = cp * alpha + psum;
        if (__builtin_amdgcn_ballot_w32(alpha != 1.0f)) {
#pragma unroll
          for (int c = 0; c < 16; ++c) O[c] *= alpha;
        }
        // P^T in both halves, contraction order tau.
        u8v pf;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const uint32_t o = xhalf(pd[i]);
          pf[i] = h ? o : pd[i];
          pf[4 + i] = h ? pd[i] : o;
        }
        const h16v pb16 = __builtin_bit_cast(h16v, pf);

        // PV: O^T[16 rr + c][q] over the 16 dim tiles c. Lane pair: the useful
        // lane holds the even tokens of dims [16 rr, +16), its partner the odd.
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          uint32_t t0[4], t1[4];
          tr4(vr[0].x, vr[1].x, vr[2].x, vr[3].x, t0);
          tr4(vr[4].x, vr[5].x, vr[6].x, vr[7].x, t1);
          if (q == 1) {
            tr4(vr[0].y, vr[1].y, vr[2].y, vr[3].y, t0);
            tr4(vr[4].y, vr[5].y, vr[6].y, vr[7].y, t1);
          } else if (q == 2) {
            tr4(vr[0].z, vr[1].z, vr[2].z, vr[3].z, t0);
            tr4(vr[4].z, vr[5].z, vr[6].z, vr[7].z, t1);
          } else if (q == 3) {
            tr4(vr[0].w, vr[1].w, vr[2].w, vr[3].w, t0);
            tr4(vr[4].w, vr[5].w, vr[6].w, vr[7].w, t1);
          }
#pragma unroll
          for (int b = 0; b < 4; ++b) {
            const int c = 4 * q + b;
            uint32_t o0 = t0[b] ^ X, o1 = t1[b] ^ X;
            // Tile c's conversion waits for the WMMA of tile c-2: one tile of
            // look-ahead instead of all 16 A fragments at once (measured).
            if (c >= 2) asm volatile("" : "+v"(o0), "+v"(o1) : "v"(O[c - 2][0]));
            const h16v a = b8_to_h16(o0, o1, swap1(o0), swap1(o1));
            O[c] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, pb16, O[c]);
          }
        }
        if (next) {
          // Issued after the last PV WMMA, not hoisted above the PV.
          load_v(pbn, s0n, O[15][0]);
        }
        pb = pbn;
        s0 = s0n;
      }

      // Merge the two halves' partial sums; m is already common.
      l += xhalf(l);
      cp += xhalf(cp);
      if (qlive) {
        const int hq = qv % hpk;
        float* op = mid_o + (int64_t)(grp * G + qrow) * smo + (int64_t)hq * smh +
                    (int64_t)si * sms;
        const bool any = m != -INFINITY;
        const float corr = BIAS * cp;
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
          op[HS] = m;
          op[HS + 1] = l;
        }
      }
    }
    seg0 = seg1;
  }
}

// Stage 2 for up to 1024 splits: weights of every split at once, then 8 waves
// take the splits round-robin and skip the ones that weigh zero.
__global__ __launch_bounds__(256) void decode_int8_wmma_reduce(
    const float* __restrict__ mid_o, __half* __restrict__ out, int num_splits,
    int64_t smo, int64_t smh, int64_t sms, int64_t soo, int64_t soh) {
  const int qi = blockIdx.x, hi = blockIdx.y;
  const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31;
  const float* base = mid_o + qi * smo + hi * smh;
  __shared__ float sw[1024], swm[8], swl[8];
  __shared__ float so[8][HS];

  float ms[4], ls[4];
  float m = -INFINITY;
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const int s = tid + 256 * k;
    ms[k] = s < num_splits ? base[s * sms + HS] : -INFINITY;
    ls[k] = s < num_splits ? base[s * sms + HS + 1] : 0.0f;
    m = fmaxf(m, ms[k]);
  }
  for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor(m, o));
  if (lane == 0) swm[w] = m;
  __syncthreads();
  float mg = swm[0];
#pragma unroll
  for (int i = 1; i < 8; ++i) mg = fmaxf(mg, swm[i]);
  float l = 0.0f;
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const float a = (ms[k] == -INFINITY) ? 0.0f : exp2f(ms[k] - mg);
    sw[tid + 256 * k] = a;
    l += ls[k] * a;
  }
  for (int o = 16; o > 0; o >>= 1) l += __shfl_xor(l, o);
  if (lane == 0) swl[w] = l;
  __syncthreads();

  float o[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll 4
  for (int s = w; s < num_splits; s += 8) {
    const float as = sw[s];
    if (as == 0.0f) continue;
    const float4* v = (const float4*)(base + s * sms + lane * 8);
    const float4 x = v[0], y = v[1];
    o[0] += x.x * as; o[1] += x.y * as; o[2] += x.z * as; o[3] += x.w * as;
    o[4] += y.x * as; o[5] += y.y * as; o[6] += y.z * as; o[7] += y.w * as;
  }
#pragma unroll
  for (int d = 0; d < 8; ++d) so[w][lane * 8 + d] = o[d];
  __syncthreads();
  float lg = 0.0f;
#pragma unroll
  for (int i = 0; i < 8; ++i) lg += swl[i];
  const float inv = 1.0f / (lg + 1e-10f);
  float acc = 0.0f;
#pragma unroll
  for (int i = 0; i < 8; ++i) acc += so[i][tid];
  out[qi * soo + hi * soh + tid] = __float2half(acc * inv);
}

#ifndef WMMA_NSB
#define WMMA_NSB 2
#endif

// cfg = splits per row group * 1000 + min_tps.
void pth_decode_int8_wmma(torch::Tensor out, torch::Tensor query,
                        torch::Tensor key_cache, torch::Tensor value_cache,
                        torch::Tensor k_scale_cache, torch::Tensor v_scale_cache,
                        torch::Tensor block_table, torch::Tensor q_to_req,
                        torch::Tensor q_to_klen, torch::Tensor mid_o_buf,
                        double sm_scale, int64_t num_kv_splits, int64_t cfg) {
  const int num_q = query.size(0);
  const int num_q_heads = query.size(1);
  const int num_kv_heads = k_scale_cache.size(2);
  const int min_tps = (int)(cfg % 1000);
  int eff = (int)(cfg / 1000);
  TORCH_CHECK(query.size(2) == HS, "wmma: only head_size=256");
  TORCH_CHECK(query.dtype() == at::kHalf, "wmma: fp16 only");
  TORCH_CHECK(query.stride(2) == 1 && query.stride(0) % 8 == 0 &&
                  query.stride(1) % 8 == 0 &&
                  reinterpret_cast<uintptr_t>(query.data_ptr()) % 16 == 0,
              "wmma: query rows must be 16-byte aligned");
  TORCH_CHECK(num_kv_heads == 1, "wmma: one kv head per rank only");
  TORCH_CHECK(num_q_heads * G <= 32, "wmma: rows*heads must fit two row tiles");
  TORCH_CHECK(key_cache.size(1) % 16 == 0, "wmma: block_size % 16 != 0");
  TORCH_CHECK(key_cache.stride(1) % 4 == 0 && value_cache.stride(1) % 4 == 0 &&
                  key_cache.storage_offset() % 4 == 0 &&
                  value_cache.storage_offset() % 4 == 0,
              "wmma: 4-byte aligned token rows");
  const at::cuda::OptionalCUDAGuard guard(device_of(query));
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  int ns = (int)num_kv_splits;
  const int groups = (num_q + G - 1) / G;
  // eff = splits per row group (measured: 192 at 284k, flat for 160-256
  // with 3 groups), capped by the buffer; not divided among the groups.
  if (eff > 0 && eff < ns) ns = eff;
  TORCH_CHECK(ns <= 1024, "wmma: at most 1024 splits");
  if (num_q == 0) return;
  constexpr int NSB = WMMA_NSB;
  dim3 grid1(groups, 1, (ns + NSB - 1) / NSB);
  decode_int8_wmma<NSB><<<grid1, dim3(64 * NSB), 0, stream>>>(
      (const __half*)query.data_ptr(), (const int8_t*)key_cache.data_ptr(),
      (const int8_t*)value_cache.data_ptr(),
      (const float*)k_scale_cache.data_ptr(),
      (const float*)v_scale_cache.data_ptr(),
      (const int*)block_table.data_ptr(), (const int*)q_to_req.data_ptr(),
      (const int*)q_to_klen.data_ptr(), (float*)mid_o_buf.data_ptr(),
      (float)sm_scale, num_q, num_q_heads, num_kv_heads,
      (int)key_cache.size(1), (int)block_table.size(1),
      (int)block_table.size(0), (int)key_cache.size(0), ns, min_tps,
      query.stride(0), query.stride(1), key_cache.stride(0),
      key_cache.stride(1), value_cache.stride(0), value_cache.stride(1),
      k_scale_cache.stride(0), k_scale_cache.stride(1),
      v_scale_cache.stride(0), v_scale_cache.stride(1), mid_o_buf.stride(0),
      mid_o_buf.stride(1), mid_o_buf.stride(2));
  dim3 grid2(num_q, num_q_heads);
  decode_int8_wmma_reduce<<<grid2, dim3(256), 0, stream>>>(
      (const float*)mid_o_buf.data_ptr(), (__half*)out.data_ptr(), ns,
      mid_o_buf.stride(0), mid_o_buf.stride(1), mid_o_buf.stride(2),
      out.stride(0), out.stride(1));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pth_decode_int8_wmma", &pth_decode_int8_wmma,
        "INT8 per-token-head decode attention on WMMA, independent waves ");
}
