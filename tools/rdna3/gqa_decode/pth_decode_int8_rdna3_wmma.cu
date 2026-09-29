// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// INT8 per-token-head decode attention for RDNA3 (gfx1100) on WMMA, with the
// query rows of one request batched into the same tiles.
//
// The scalar GQA kernel is bound by instruction issue: per token and per query
// head it pays 8 FMAs, a 32-lane reduction, an exp and 8 more FMAs, and the k+1
// rows of a speculative-decode request each redo all of it over the same KV.
// Measured at 284k context, 4 rows x 6 heads: 990 us per call, 142 GB/s.
//
// Here the (row, head) query vectors of a run of rows that share a request are
// the M dimension of v_wmma_f32_16x16x16_f16 (24 vectors -> 2 row tiles), and
// 16 KV tokens are its N dimension. QK and PV are one WMMA per 16x16 tile, the
// softmax reductions run once per 16 tokens, and K/V are read and converted
// once for all rows.
//
// WMMA layout on wave32, measured with a probe kernel (gqa4/wmma_layout):
//   A: lane L holds row L%16, 16 halves;  B: lane L holds column L%16
//   D[i][j] is in lane j + 16*(i&1), element i>>1
//
// ⛔Do not store into LDS through __builtin_bit_cast(__half, vec[k]): hipcc
// kept element 0 for every k (measured, gqa4/wmma_layout/sv.cu). The LDS
// arrays are _Float16 for that reason.
//
// Numerics: int8 K and V are exact in fp16. The K scale multiplies the score in
// fp32. The V scale is folded into P before it is rounded to fp16, times 2^12 so
// the product stays out of the subnormal range; O is divided back at the end.
// Accumulation is fp32 throughout.

#include <cstdint>
#include <torch/all.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef _Float16 h16v __attribute__((ext_vector_type(16)));
typedef _Float16 h8v __attribute__((ext_vector_type(8)));
typedef float f8v __attribute__((ext_vector_type(8)));

constexpr int HS = 256;
constexpr int KT = HS / 16;  // 16 k-steps of QK
constexpr int G = 4;         // max rows per group (k+1 with MTP k=3)
constexpr float PSCALE = 4096.0f;
constexpr float BIAS = 1152.0f;

template <int CTRL>
__device__ __forceinline__ float dpp(float v) {
  return __int_as_float(
      __builtin_amdgcn_update_dpp(0, __float_as_int(v), CTRL, 0xF, 0xF, true));
}
// Within each 16-lane half: xor 1, 2 (quad perm), then row_shr 4/8 style
// half-row mirrors. Same sequence as the measured prototype.
__device__ __forceinline__ float red16_max(float v) {
  v = fmaxf(v, dpp<0xB1>(v));
  v = fmaxf(v, dpp<0x4E>(v));
  v = fmaxf(v, dpp<0x141>(v));
  v = fmaxf(v, dpp<0x140>(v));
  return v;
}
__device__ __forceinline__ float red16_add(float v) {
  v += dpp<0xB1>(v);
  v += dpp<0x4E>(v);
  v += dpp<0x141>(v);
  v += dpp<0x140>(v);
  return v;
}

// 16 int8 -> 16 fp16, exact: 0x6400|(x^0x80) is 1024 + x + 128 as a half.
__device__ __forceinline__ h16v i8x16_to_h16(int4 w) {
  // 0x6400 | (x ^ 0x80) is the half 1024 + x + 128; one v_perm places two
  // bytes under two 0x64 high bytes, one packed subtract finishes. Exact on
  // all 256 values (gqa4/wmma_layout/cv.cu).
  typedef _Float16 h2 __attribute__((ext_vector_type(2)));
  h16v r;
  const uint32_t ws[4] = {(uint32_t)w.x, (uint32_t)w.y, (uint32_t)w.z,
                          (uint32_t)w.w};
  const h2 k = {(_Float16)1152.0f, (_Float16)1152.0f};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const uint32_t u = ws[i] ^ 0x80808080u;
    const h2 a = __builtin_bit_cast(h2, __builtin_amdgcn_perm(0x64646464u, u, 0x07010700u)) - k;
    const h2 b = __builtin_bit_cast(h2, __builtin_amdgcn_perm(0x64646464u, u, 0x07030702u)) - k;
    r[4 * i] = a[0];
    r[4 * i + 1] = a[1];
    r[4 * i + 2] = b[0];
    r[4 * i + 3] = b[1];
  }
  return r;
}

// Same, for bytes already biased (x ^ 0x80) when they were staged in LDS.
__device__ __forceinline__ h16v b8x16_to_h16(int4 w) {
  typedef _Float16 h2 __attribute__((ext_vector_type(2)));
  h16v r;
  const uint32_t ws[4] = {(uint32_t)w.x, (uint32_t)w.y, (uint32_t)w.z,
                          (uint32_t)w.w};
  const h2 k = {(_Float16)1152.0f, (_Float16)1152.0f};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const h2 a = __builtin_bit_cast(h2, __builtin_amdgcn_perm(0x64646464u, ws[i], 0x07010700u)) - k;
    const h2 b = __builtin_bit_cast(h2, __builtin_amdgcn_perm(0x64646464u, ws[i], 0x07030702u)) - k;
    r[4 * i] = a[0];
    r[4 * i + 1] = a[1];
    r[4 * i + 2] = b[0];
    r[4 * i + 3] = b[1];
  }
  return r;
}

// Biased bytes -> fp16 values x + 1152 (exact), leaving the bias in.
__device__ __forceinline__ h16v b8x16_to_h16_biased(int4 w) {
  typedef _Float16 h2 __attribute__((ext_vector_type(2)));
  h16v r;
  const uint32_t ws[4] = {(uint32_t)w.x, (uint32_t)w.y, (uint32_t)w.z,
                          (uint32_t)w.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const h2 a = __builtin_bit_cast(h2, __builtin_amdgcn_perm(0x64646464u, ws[i], 0x07010700u));
    const h2 b = __builtin_bit_cast(h2, __builtin_amdgcn_perm(0x64646464u, ws[i], 0x07030702u));
    r[4 * i] = a[0];
    r[4 * i + 1] = a[1];
    r[4 * i + 2] = b[0];
    r[4 * i + 3] = b[1];
  }
  return r;
}

__device__ __forceinline__ h16v ld16(const _Float16* p) {
  const h8v lo = *(const h8v*)p;
  const h8v hi = *(const h8v*)(p + 8);
  return __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                                 12, 13, 14, 15);
}
// 16 bytes from a row that is only 4-byte aligned: the real int8 cache packs
// [K(256) | K scale(4) | V(256) | V scale(4)] per token, so token rows are 520
// bytes apart and V starts at +260.
__device__ __forceinline__ int4 ld16u(const int8_t* p) {
  const uint32_t* q = (const uint32_t*)__builtin_assume_aligned(p, 4);
  return make_int4((int)q[0], (int)q[1], (int)q[2], (int)q[3]);
}

// Grid: (ceil(num_q / G), 1, num_splits). Block: 4 waves = 2 row tiles x 2 dim
// halves. v7 ran the two row tiles as separate blocks, so every KV byte was
// read twice: at 284k that is 2 x 148 MB and a 327 us floor, and v7 sat at 533.
// Here the block loads each 16-token tile once, cooperatively (64 B per
// thread), into a double-buffered LDS stage; the loads of tile n+1 are in flight
// while tile n is computed. K stays int8 in LDS (rows padded to 272 B: the 16
// lanes of a k-step read 16 rows and hit 16 distinct bank groups); V is written
// transposed as fp16. The QK A fragments (Q) live in registers.
constexpr int V8_KROW = HS + 16;

__global__ __launch_bounds__(128) void decode_int8_v8(
    const __half* __restrict__ Q, const int8_t* __restrict__ K_cache,
    const int8_t* __restrict__ V_cache, const float* __restrict__ K_scale,
    const float* __restrict__ V_scale, const int* __restrict__ block_table,
    const int* __restrict__ q_to_req, const int* __restrict__ q_to_klen,
    float* __restrict__ mid_o, float sm_scale, int num_q, int num_q_heads,
    int num_kv_heads, int block_size, int max_blocks, int num_reqs,
    int num_phys_blocks, int num_splits, int min_tps, int64_t sq0, int64_t sq1,
    int64_t skb, int64_t sks, int64_t skh, int64_t svb, int64_t svs,
    int64_t svh, int64_t ssb, int64_t sss, int64_t ssh, int64_t svsb,
    int64_t svss, int64_t svsh, int64_t smo, int64_t smh, int64_t sms) {
  constexpr int CTW = 8;  // column tiles of O per wave (128 dims)
  constexpr int KSW = 8;  // QK k-steps per wave

  const int grp = blockIdx.x;
  const int si = blockIdx.z;
  const int tid = threadIdx.x;
  const int w = tid >> 5;
  const int rt = w >> 1;  // row tile
  const int dh = w & 1;   // dim half / k-step half
  const int lane = tid & 31;
  const int r = lane & 15;
  const int par = lane >> 4;
  const int hpk = num_q_heads / num_kv_heads;
  const int max_kv = max_blocks * block_size;
  const float sm_log2 = sm_scale * 1.4426950408889634f;
  const int kvh = 0;
  // Cooperative tile load: thread -> token lt, bytes [32*lq, +32) of K and V.
  const int lt = tid >> 3, lq = tid & 7;
  const int vd = tid & 63, vt = tid >> 6;

  __shared__ __attribute__((aligned(16))) int8_t sK[2][16][V8_KROW];
  __shared__ uint32_t sVt[2][HS][5];
  __shared__ float sKs[2][16], sVs[2][16];
  __shared__ float sS[4][32][8];
  __shared__ __attribute__((aligned(16))) _Float16 sP[2][16][16];
  __shared__ float sA[2][16], sL[2][16];
  __shared__ float sQc[2][2][16], sCP[2][16];

  int rreq[G], rlen[G];
#pragma unroll
  for (int j = 0; j < G; ++j) {
    const int qi = grp * G + j;
    int rq = -1, ln = 0;
    if (qi < num_q) {
      rq = q_to_req[qi];
      ln = q_to_klen[qi];
      if (rq < 0 || rq >= num_reqs) { rq = 0; ln = 0; }
      if (ln < 0) ln = 0;
      if (ln > max_kv) ln = max_kv;
    }
    rreq[j] = rq;
    rlen[j] = ln;
  }

  int seg0 = 0;
  while (seg0 < G && rreq[seg0] >= 0) {
    int seg1 = seg0 + 1;
    while (seg1 < G && rreq[seg1] == rreq[seg0]) ++seg1;
    const int req = rreq[seg0];
    int L = 0;
    for (int j = seg0; j < seg1; ++j) L = max(L, rlen[j]);
    const int qv_lo = seg0 * hpk, qv_hi = seg1 * hpk;
    // Whether this wave's row tile has any live row (wave-uniform).
    const bool tile_live = qv_hi > rt * 16 && qv_lo < rt * 16 + 16;

    // Q fragments: row r of this row tile, k-steps [8*dh, +8).
    h16v aq[KSW];
    {
      const int qv = rt * 16 + r;
      const bool live = qv >= qv_lo && qv < qv_hi;
      const __half* qp =
          live ? Q + (int64_t)(grp * G + qv / hpk) * sq0 + (int64_t)(qv % hpk) * sq1
               : nullptr;
      // Q rows are 512 B apart and 16-byte aligned: 16-byte loads.
#pragma unroll
      for (int s = 0; s < KSW; ++s) {
        if (live) aq[s] = ld16((const _Float16*)qp + 16 * (KSW * dh + s));
        else aq[s] = h16v{};
      }
    }
    // The K and V fragments are converted without removing their +1152 bias
    // (fp16 products are exact in fp32), so the bias comes off once per row:
    // S -= 1152 * sum(q) here, O -= 1152 * sum(P) at the end.
    {
      float qs = 0.0f;
#pragma unroll
      for (int s = 0; s < KSW; ++s)
#pragma unroll
        for (int i = 0; i < 16; ++i) qs += (float)aq[s][i];
      if (lane < 16) sQc[dh][rt][r] = qs;
      if (tid < 32) sCP[tid >> 4][tid & 15] = 0.0f;
    }

    int frow_len[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const int qv = rt * 16 + 2 * e + par;
      frow_len[e] = (qv >= qv_lo && qv < qv_hi) ? rlen[qv / hpk] : 0;
    }

    int tps = (L + num_splits - 1) / num_splits;
    if (tps < min_tps) tps = min_tps;
    tps = (tps + 15) & ~15;
    const int start = si * tps;
    const int end = min(start + tps, L);

    float m_st[8], l_st[8];
    f8v O[CTW];
#pragma unroll
    for (int e = 0; e < 8; ++e) { m_st[e] = -INFINITY; l_st[e] = 0.0f; }
#pragma unroll
    for (int c = 0; c < CTW; ++c) O[c] = (f8v){0, 0, 0, 0, 0, 0, 0, 0};

    // Global -> registers for one tile (this thread's share). K: token lt,
    // bytes [32*lq, +32). V: dims [4*vd, +4) of tokens [8*vt, +8), one dword
    // per token, so that a 4x4 byte transpose in registers yields dwords that
    // are already token-major for the transposed LDS stage. Loads are
    // unconditional (a tail token reads a valid slot and is masked by -inf in
    // the softmax); only the scales are zeroed, since 0 * NaN would leak.
    int4 pk[2];
    uint32_t pv[8];
    float pks = 0.0f, pvs = 0.0f;
    // Position of the next tile to load, tracked incrementally (tiles are
    // loaded in order, 16 tokens apart): no division per tile, and the block
    // bases are wave-uniform, so their 64-bit products stay scalar. Each token
    // then only picks base A or B and adds a 32-bit in-block offset.
    int c_lb = start / block_size;
    int c_s0 = start - c_lb * block_size;
    auto cargar = [&](int b0) {
      const int lb0 = c_lb, s0 = c_s0;
      c_s0 += 16;
      if (c_s0 >= block_size) { c_s0 -= block_size; ++c_lb; }
      int pbA = block_table[req * max_blocks + lb0];
      if (pbA < 0 || pbA >= num_phys_blocks) pbA = 0;
      int pbB = pbA;
      if (s0 + 16 > block_size && lb0 + 1 < max_blocks) {
        pbB = block_table[req * max_blocks + lb0 + 1];
        if (pbB < 0 || pbB >= num_phys_blocks) pbB = 0;
      }
      {
        int slot = s0 + lt;
        const bool inB = slot >= block_size;
        if (inB) slot -= block_size;
        const int8_t* kb = K_cache + (int64_t)(inB ? pbB : pbA) * skb;
        const int8_t* kp = kb + kvh * skh + slot * (int)sks + 32 * lq;
        pk[0] = ld16u(kp);
        pk[1] = ld16u(kp + 16);
        const bool ok = b0 + lt < end;
        const float* ksb = K_scale + (int64_t)(inB ? pbB : pbA) * ssb;
        const float* vsb = V_scale + (int64_t)(inB ? pbB : pbA) * svsb;
        const float ksv = ksb[slot * (int)sss + kvh * (int)ssh];
        const float vsv = vsb[slot * (int)svss + kvh * (int)svsh];
        pks = ok ? ksv : 0.0f;
        pvs = ok ? vsv : 0.0f;
      }
      const int8_t* vA = V_cache + (int64_t)pbA * svb + kvh * svh + 4 * vd;
      const int8_t* vB = V_cache + (int64_t)pbB * svb + kvh * svh + 4 * vd;
      const int vstr = (int)svs;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        int slot = s0 + 8 * vt + j;
        const bool inB = slot >= block_size;
        if (inB) slot -= block_size;
        pv[j] = *(const uint32_t*)__builtin_assume_aligned((inB ? vB : vA) + slot * vstr, 4);
      }
    };
    // Registers -> LDS buffer b. Bytes are stored biased (x ^ 0x80), which is
    // what the fp16 conversion wants, so consumers skip the xor. V goes in
    // transposed: sVt[dim] holds the 16 tokens of the tile (rows padded to 5
    // dwords against bank conflicts).
    auto guardar = [&](int b) {
      const uint32_t X = 0x80808080u;
      *(int4*)&sK[b][lt][32 * lq] = make_int4(pk[0].x ^ X, pk[0].y ^ X, pk[0].z ^ X, pk[0].w ^ X);
      *(int4*)&sK[b][lt][32 * lq + 16] = make_int4(pk[1].x ^ X, pk[1].y ^ X, pk[1].z ^ X, pk[1].w ^ X);
#pragma unroll
      for (int q = 0; q < 2; ++q) {
        uint32_t a[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) a[i] = pv[4 * q + i] ^ X;
        const uint32_t lo01 = __builtin_amdgcn_perm(a[1], a[0], 0x05010400u);
        const uint32_t hi01 = __builtin_amdgcn_perm(a[1], a[0], 0x07030602u);
        const uint32_t lo23 = __builtin_amdgcn_perm(a[3], a[2], 0x05010400u);
        const uint32_t hi23 = __builtin_amdgcn_perm(a[3], a[2], 0x07030602u);
        sVt[b][4 * vd + 0][2 * vt + q] = __builtin_amdgcn_perm(lo23, lo01, 0x05040100u);
        sVt[b][4 * vd + 1][2 * vt + q] = __builtin_amdgcn_perm(lo23, lo01, 0x07060302u);
        sVt[b][4 * vd + 2][2 * vt + q] = __builtin_amdgcn_perm(hi23, hi01, 0x05040100u);
        sVt[b][4 * vd + 3][2 * vt + q] = __builtin_amdgcn_perm(hi23, hi01, 0x07060302u);
      }
      if (lq == 0) {
        sKs[b][lt] = pks;
        sVs[b][lt] = pvs;
      }
    };

    if (start < end) {
      cargar(start);
      guardar(0);
    }
    __syncthreads();

    auto own = [](int e) { return e >> 2; };  // dh 0 owns e 0..3, dh 1 owns 4..7
    int buf = 0;
    for (int base = start; base < end; base += 16) {
      const bool hay_sig = base + 16 < end;
      if (hay_sig) cargar(base + 16);

      const int t = base + r;
      const bool tok_ok = t < end;
      if (tile_live) {
        f8v S = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
        for (int s = 0; s < KSW; ++s) {
          const int4 kw = *(const int4*)&sK[buf][r][16 * (KSW * dh + s)];
          S = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(aq[s], b8x16_to_h16_biased(kw), S);
        }
#pragma unroll
        for (int e = 0; e < 8; ++e) sS[w][lane][e] = S[e];
      }
      __syncthreads();

      if (tile_live) {
        const float ks = sKs[buf][r], vs = sVs[buf][r];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          if (own(e) != dh || frow_len[e] == 0) continue;
          const int row = 2 * e + par;
          const float acc = sS[2 * rt][lane][e] + sS[2 * rt + 1][lane][e] -
                            BIAS * (sQc[0][rt][row] + sQc[1][rt][row]);
          const bool ok = tok_ok && t < frow_len[e];
          const float sc = ok ? acc * ks * sm_log2 : -INFINITY;
          const float mn = fmaxf(m_st[e], red16_max(sc));
          float alpha = 1.0f;
          if (mn > m_st[e])
            alpha = (m_st[e] == -INFINITY) ? 0.0f : __builtin_amdgcn_exp2f(m_st[e] - mn);
          const float pe = ok ? __builtin_amdgcn_exp2f(sc - mn) : 0.0f;
          l_st[e] = l_st[e] * alpha + pe;
          m_st[e] = mn;
          const _Float16 ph = (_Float16)(pe * vs * PSCALE);
          sP[rt][2 * e + par][r] = ph;
          const float psum = red16_add((float)ph);
          if (r == 0) {
            sA[rt][2 * e + par] = alpha;
            sCP[rt][2 * e + par] = sCP[rt][2 * e + par] * alpha + psum;
          }
        }
      }
      __syncthreads();

      if (tile_live) {
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          if (frow_len[e] == 0) continue;
          const float alpha = sA[rt][2 * e + par];
          if (alpha != 1.0f) {
#pragma unroll
            for (int c = 0; c < CTW; ++c) O[c][e] *= alpha;
          }
        }
        const h16v a = ld16(&sP[rt][r][0]);
#pragma unroll
        for (int c = 0; c < CTW; ++c) {
          const uint32_t* vr = &sVt[buf][128 * dh + 16 * c + r][0];
          const h16v b = b8x16_to_h16_biased(make_int4((int)vr[0], (int)vr[1], (int)vr[2], (int)vr[3]));
          O[c] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, O[c]);
        }
      }
      if (hay_sig) guardar(buf ^ 1);
      __syncthreads();
      buf ^= 1;
    }

    if (tile_live) {
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        if (own(e) != dh) continue;
        const float lsum = red16_add(l_st[e]);
        if (r == 0) {
          sA[rt][2 * e + par] = m_st[e];
          sL[rt][2 * e + par] = lsum;
        }
      }
    }
    __syncthreads();
    if (tile_live) {
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int qv = rt * 16 + 2 * e + par;
        if (!(qv >= qv_lo && qv < qv_hi)) continue;
        const int j = qv / hpk, h = qv % hpk;
        float* op = mid_o + (int64_t)(grp * G + j) * smo + (int64_t)h * smh +
                    (int64_t)si * sms;
        const float mrow = sA[rt][2 * e + par];
        const bool any = mrow != -INFINITY;
#pragma unroll
        for (int c = 0; c < CTW; ++c)
          op[128 * dh + 16 * c + r] =
              any ? (O[c][e] - BIAS * sCP[rt][2 * e + par]) * (1.0f / PSCALE) : 0.0f;
        if (r == 0 && dh == own(e)) {
          op[HS] = mrow;
          op[HS + 1] = sL[rt][2 * e + par];
        }
      }
    }
    __syncthreads();
    seg0 = seg1;
  }
}

// Stage 2: 8 waves split the splits, one lane per 8 dims, then an LDS merge.
// Splits that stage 1 left empty carry m = -inf and weigh exactly zero.
__global__ __launch_bounds__(256) void decode_int8_reduce_v5(
    const float* __restrict__ mid_o, __half* __restrict__ out, int num_splits,
    int64_t smo, int64_t smh, int64_t sms, int64_t soo, int64_t soh) {
  const int qi = blockIdx.x, hi = blockIdx.y;
  const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const float* base = mid_o + qi * smo + hi * smh;
  __shared__ float sm[8], sl[8];
  __shared__ float so[8][HS];

  float m = -INFINITY;
  for (int s = w; s < num_splits; s += 8) m = fmaxf(m, base[s * sms + HS]);
  // wave max
  for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor(m, o));
  if (lane == 0) sm[w] = m;
  __syncthreads();
  float mg = sm[0];
  for (int i = 1; i < 8; ++i) mg = fmaxf(mg, sm[i]);

  float o[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  float l = 0.0f;
  for (int s = w; s < num_splits; s += 8) {
    const float* sp = base + s * sms;
    const float ms = sp[HS];
    const float a = (ms == -INFINITY) ? 0.0f : exp2f(ms - mg);
    if (a == 0.0f) continue;
    const float4* v = (const float4*)(sp + lane * 8);
    const float4 x = v[0], y = v[1];
    o[0] += x.x * a; o[1] += x.y * a; o[2] += x.z * a; o[3] += x.w * a;
    o[4] += y.x * a; o[5] += y.y * a; o[6] += y.z * a; o[7] += y.w * a;
    l += sp[HS + 1] * a;
  }
#pragma unroll
  for (int d = 0; d < 8; ++d) so[w][lane * 8 + d] = o[d];
  if (lane == 0) sl[w] = l;
  __syncthreads();
  if (w == 0) {
    float lg = 0.0f;
    for (int i = 0; i < 8; ++i) lg += sl[i];
    const float inv = 1.0f / (lg + 1e-10f);
#pragma unroll
    for (int d = 0; d < 8; ++d) {
      float acc = 0.0f;
      for (int i = 0; i < 8; ++i) acc += so[i][lane * 8 + d];
      out[qi * soo + hi * soh + lane * 8 + d] = __float2half(acc * inv);
    }
  }
}


// Stage 2, v17: every split's (m, l) is read at once (one thread per split,
// ns <= 256), so the block max and the weights cost one load latency instead
// of ns / 8 dependent ones. Then 8 waves take the splits round-robin and skip
// the ones that weigh zero (empty splits carry m = -inf) without loading them.
__global__ __launch_bounds__(256) void decode_int8_reduce_v17(
    const float* __restrict__ mid_o, __half* __restrict__ out, int num_splits,
    int64_t smo, int64_t smh, int64_t sms, int64_t soo, int64_t soh) {
  const int qi = blockIdx.x, hi = blockIdx.y;
  const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31;
  const float* base = mid_o + qi * smo + hi * smh;
  __shared__ float sw[256], swm[8], swl[8];
  __shared__ float so[8][HS];

  const float ms = tid < num_splits ? base[tid * sms + HS] : -INFINITY;
  const float ls = tid < num_splits ? base[tid * sms + HS + 1] : 0.0f;
  float m = ms;
  for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor(m, o));
  if (lane == 0) swm[w] = m;
  __syncthreads();
  float mg = swm[0];
#pragma unroll
  for (int i = 1; i < 8; ++i) mg = fmaxf(mg, swm[i]);
  const float a = (ms == -INFINITY) ? 0.0f : exp2f(ms - mg);
  sw[tid] = a;
  float l = ls * a;
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
  // 256 threads finish the 256 dims, one each.
  float acc = 0.0f;
#pragma unroll
  for (int i = 0; i < 8; ++i) acc += so[i][tid];
  out[qi * soo + hi * soh + tid] = __float2half(acc * inv);
}

void pth_decode_int8_v8(torch::Tensor out, torch::Tensor query,
                        torch::Tensor key_cache, torch::Tensor value_cache,
                        torch::Tensor k_scale_cache, torch::Tensor v_scale_cache,
                        torch::Tensor block_table, torch::Tensor q_to_req,
                        torch::Tensor q_to_klen, torch::Tensor mid_o_buf,
                        double sm_scale, int64_t num_kv_splits, int64_t cfg) {
  const int num_q = query.size(0);
  const int num_q_heads = query.size(1);
  const int num_kv_heads = k_scale_cache.size(2);
  // cfg = splits * 1000 + min_tps. Fewer effective splits than the buffer
  // holds, so that every block is resident in one round (0 = all of them).
  const int min_tps = (int)(cfg % 1000);
  int eff = (int)(cfg / 1000);
  TORCH_CHECK(query.size(2) == HS, "v8: only head_size=256");
  TORCH_CHECK(query.dtype() == at::kHalf, "v8: fp16 only");
  TORCH_CHECK(query.stride(2) == 1 && query.stride(0) % 8 == 0 &&
                  query.stride(1) % 8 == 0 &&
                  reinterpret_cast<uintptr_t>(query.data_ptr()) % 16 == 0,
              "v17: query rows must be 16-byte aligned");
  TORCH_CHECK(num_kv_heads == 1, "v8: one kv head per rank only");
  TORCH_CHECK(num_q_heads * G <= 32, "v8: rows*heads must fit two row tiles");
  TORCH_CHECK(key_cache.stride(1) % 4 == 0 && value_cache.stride(1) % 4 == 0 &&
                  key_cache.storage_offset() % 4 == 0 &&
                  value_cache.storage_offset() % 4 == 0,
              "v8: 4-byte aligned token rows");
  const at::cuda::OptionalCUDAGuard guard(device_of(query));
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  int ns = (int)num_kv_splits;
  // eff = blocks resident in one round; the row groups of the batch share it.
  const int groups = (num_q + G - 1) / G;
  if (eff > 0) eff = max(eff / max(groups, 1), 16);
  if (eff > 0 && eff < ns) ns = eff;
  TORCH_CHECK(ns <= 256, "v17: at most 256 splits");
  if (num_q == 0) return;
  dim3 grid1((num_q + G - 1) / G, 1, ns);
#define LANZAR()                                                               \
  decode_int8_v8<<<grid1, dim3(128), 0, stream>>>(                   \
      (const __half*)query.data_ptr(), (const int8_t*)key_cache.data_ptr(),    \
      (const int8_t*)value_cache.data_ptr(),                                   \
      (const float*)k_scale_cache.data_ptr(),                                  \
      (const float*)v_scale_cache.data_ptr(),                                  \
      (const int*)block_table.data_ptr(), (const int*)q_to_req.data_ptr(),     \
      (const int*)q_to_klen.data_ptr(), (float*)mid_o_buf.data_ptr(),          \
      (float)sm_scale, num_q, num_q_heads, num_kv_heads,                       \
      (int)key_cache.size(1), (int)block_table.size(1),                        \
      (int)block_table.size(0), (int)key_cache.size(0), ns, min_tps,           \
      query.stride(0), query.stride(1), key_cache.stride(0),                   \
      key_cache.stride(1), key_cache.stride(2), value_cache.stride(0),         \
      value_cache.stride(1), value_cache.stride(2), k_scale_cache.stride(0),   \
      k_scale_cache.stride(1), k_scale_cache.stride(2),                        \
      v_scale_cache.stride(0), v_scale_cache.stride(1),                        \
      v_scale_cache.stride(2), mid_o_buf.stride(0), mid_o_buf.stride(1),       \
      mid_o_buf.stride(2))
  LANZAR();
#undef LANZAR
  dim3 grid2(num_q, num_q_heads);
  decode_int8_reduce_v17<<<grid2, dim3(256), 0, stream>>>(
      (const float*)mid_o_buf.data_ptr(), (__half*)out.data_ptr(), ns,
      mid_o_buf.stride(0), mid_o_buf.stride(1), mid_o_buf.stride(2),
      out.stride(0), out.stride(1));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pth_decode_int8_v8", &pth_decode_int8_v8,
        "INT8 per-token-head decode attention on WMMA, rows of a request batched");
}
