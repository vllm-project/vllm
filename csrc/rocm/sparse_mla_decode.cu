// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

// Sparse MLA decode over fp8_ds_mla caches for gfx942 (MI300X / MI325X).
//
// Shapes are DeepSeek-V4/V4.1's per-rank decode: 16 heads x 512 dims of bf16 Q
// (448 nope + 64 rope) against each query's gathered SWA rows (main cache) and
// top-k rows (extra cache). Both caches use the fp8_ds_mla block layout: per
// block, `block_size` rows of [448 fp8 | 64 bf16], then `block_size` rows of 8
// E8M0 bytes (7 used, one per 64 nope dims).
//
// One wavefront owns one (query, split). S^T = K Q^T is computed with K as the
// MFMA A operand straight from the gathered registers, so P^T lands in the
// B-operand layout that O^T = V^T P^T needs; V^T (keys contiguous per lane) is
// read back from a per-wave LDS copy of the dequantized bf16 tile.
// Dequantization is exact (fp8 * 2^k is representable in bf16), so products
// match the Triton kernel; only the fp32 summation order differs.
//
// NaN handling matches the Triton kernel's scrub of untrusted caches: NaN
// elements of a gathered row count as zero. OCP NaN bytes are zeroed while
// decoding; any other NaN (FNUZ NaN byte, 0 * inf scale) makes the row's score
// NaN, and such tiles are recomputed with every element scrubbed.

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <type_traits>

#if defined(__HIPCC__) && defined(__gfx942__)
  #define __HIP__SPARSE_MLA_DECODE__
#endif

namespace vllm {
namespace sparse_mla_decode {

typedef float f32x4 __attribute__((ext_vector_type(4)));
typedef float f32x2 __attribute__((ext_vector_type(2)));
typedef short s16x4 __attribute__((ext_vector_type(4)));
typedef unsigned u32x4 __attribute__((ext_vector_type(4)));
typedef unsigned u32x2 __attribute__((ext_vector_type(2)));

constexpr int kHeads = 16;
constexpr int kHeadDim = 512;
constexpr int kRowBytes = 576;
constexpr int kWaves = 4;
constexpr float kLog2e = 1.4426950408889634f;

struct Seg {
  const uint8_t* cache;
  const int* idx;
  const int* ptr;
  int64_t stride0;
  int64_t rows;
  int bs_shift;
  int ocp;  // 1: OCP e4m3fn (decoded as FNUZ * 2), 0: e4m3fnuz
};

struct Params {
  const uint16_t* q;
  int64_t q_s0, q_s1;
  Seg seg[2];
  int nseg;
  const float* sink;
  float scale_l2;
  int num_q, splits;
  uint16_t* out;
  int64_t o_s0, o_s1;
  float* part_m;
  float* part_l;
  float* part_acc;
};

#if defined(__HIP__SPARSE_MLA_DECODE__)

constexpr int kTile = 16;
// 512 bf16 per LDS row; 16-byte chunks XOR-swizzled by (row & 7).
constexpr int kLdsRowBytes = 1024;
constexpr int kLdsWave = kTile * kLdsRowBytes;
constexpr int kXcds = 8;
constexpr float kLn2 = 0.6931471805599453f;
constexpr float kNegLarge = -3.4028234663852886e38f;
// log2 headroom before the running max must move.
constexpr float kLazyRescale = 8.0f;

__device__ __forceinline__ unsigned pack_hi(float lo, float hi) {
  return __builtin_amdgcn_perm(__float_as_uint(hi), __float_as_uint(lo),
                               0x07060302u);
}
__device__ __forceinline__ unsigned perm_lo16(unsigned lo, unsigned hi) {
  return __builtin_amdgcn_perm(hi, lo, 0x05040100u);
}
__device__ __forceinline__ unsigned perm_hi16(unsigned lo, unsigned hi) {
  return __builtin_amdgcn_perm(hi, lo, 0x07060302u);
}
__device__ __forceinline__ unsigned bf16_rne(float x) {
  unsigned u = __float_as_uint(x);
  return (u + 0x7fffu + ((u >> 16) & 1u)) >> 16;
}
__device__ __forceinline__ unsigned pack_rne(float lo, float hi) {
  return bf16_rne(lo) | (bf16_rne(hi) << 16);
}
__device__ __forceinline__ s16x4 as_s16x4(unsigned a, unsigned b) {
  u32x2 v = {a, b};
  return __builtin_bit_cast(s16x4, v);
}
// E8M0 byte j of an 8-byte scale word -> 2^(e-127), times 2 for OCP data
// decoded as FNUZ. 0xff saturates to inf, as exp2(255 - 127) does.
__device__ __forceinline__ float e8m0(u32x2 w, int j, int ocp) {
  unsigned word = j < 4 ? w.x : w.y;
  unsigned e = (word >> (8 * (j & 3))) & 0xffu;
  return __uint_as_float(min(e + ocp, 255u) << 23);
}
__device__ __forceinline__ int lds_chunk(int row, int chunk) {
  return row * kLdsRowBytes + ((chunk ^ (row & 7)) << 4);
}
// Four OCP e4m3fn bytes, ready for the FNUZ converter: -0 (0x80, NaN in FNUZ)
// becomes +0 and NaN (0x7f / 0xff) becomes 0. No carries cross bytes since
// 0x7f + 0x7f < 0x100.
__device__ __forceinline__ unsigned ocp_canon(unsigned w) {
  const unsigned mag = w & 0x7f7f7f7fu;
  const unsigned nonzero = (mag + 0x7f7f7f7fu) & w;
  const unsigned nan_bits = (mag + 0x01010101u) & 0x80808080u;
  return ((nonzero & 0x80808080u) | mag) & ~((nan_bits >> 7) * 0xffu);
}
// Two bf16 values with NaNs replaced by 0.
__device__ __forceinline__ unsigned bf16x2_scrub(unsigned w) {
  const unsigned mag = w & 0x7fff7fffu;
  const unsigned nan_bits = (mag + 0x007f007fu) & 0x80008000u;
  return w & ~((nan_bits >> 15) * 0xffffu);
}
__device__ __forceinline__ float scrub(float x) { return x == x ? x : 0.f; }

struct Tile {
  u32x4 d[9];
  u32x2 sc;
  unsigned vmask;
  int ocp;
};

__device__ __forceinline__ int tile_seg(int t, int nt0) {
  return t < nt0 ? 0 : 1;
}
__device__ __forceinline__ Seg pick(const Params& p, int s) {
  return s ? p.seg[1] : p.seg[0];
}

#endif  // __HIP__SPARSE_MLA_DECODE__

template <bool kSplit>
__global__ __launch_bounds__(kWaves * 64,
                             1) void sparse_mla_decode_kernel(Params p) {
#if defined(__HIP__SPARSE_MLA_DECODE__)
  __shared__ __attribute__((aligned(16))) uint8_t smem[kWaves * kLdsWave];
  const int lane = threadIdx.x & 63;
  const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 6);
  int wg = blockIdx.x;
  {
    // Spread consecutive work-groups over the XCDs.
    const int n = gridDim.x, per = n / kXcds, rem = n % kXcds;
    const int grp = wg % kXcds;
    wg = grp * per + min(grp, rem) + wg / kXcds;
  }
  const int item = wg * kWaves + wave;
  if (item >= p.num_q * p.splits) return;
  const int qi = item / p.splits;
  const int si = item - qi * p.splits;
  uint8_t* lds = smem + wave * kLdsWave;
  const int c = lane & 15;
  const int g = lane >> 4;

  // Q^T as the B operand: lane holds Q[head c][64J + 16g + 4qq + i]; the rope
  // chunk uses dims {8g..8g+7} for qq=0,1 and {32+8g..} for qq=2,3 to match the
  // K row loads.
  s16x4 bq[8][4];
  {
    const uint16_t* qrow = p.q + (int64_t)qi * p.q_s0 + (int64_t)c * p.q_s1;
  #pragma unroll
    for (int J = 0; J < 7; ++J) {
      u32x4 v0 = *(const u32x4*)(qrow + 64 * J + 16 * g);
      u32x4 v1 = *(const u32x4*)(qrow + 64 * J + 16 * g + 8);
      bq[J][0] = as_s16x4(v0.x, v0.y);
      bq[J][1] = as_s16x4(v0.z, v0.w);
      bq[J][2] = as_s16x4(v1.x, v1.y);
      bq[J][3] = as_s16x4(v1.z, v1.w);
    }
    u32x4 r0 = *(const u32x4*)(qrow + 448 + 8 * g);
    u32x4 r1 = *(const u32x4*)(qrow + 480 + 8 * g);
    bq[7][0] = as_s16x4(r0.x, r0.y);
    bq[7][1] = as_s16x4(r0.z, r0.w);
    bq[7][2] = as_s16x4(r1.x, r1.y);
    bq[7][3] = as_s16x4(r1.z, r1.w);
  }

  int lo[2] = {0, 0}, hi[2] = {0, 0}, nt[2] = {0, 0};
  #pragma unroll
  for (int s = 0; s < 2; ++s) {
    if (s < p.nseg) {
      const int* ptr = s ? p.seg[1].ptr : p.seg[0].ptr;
      const int beg = __builtin_amdgcn_readfirstlane(ptr[qi]);
      const int end = __builtin_amdgcn_readfirstlane(ptr[qi + 1]);
      const int len = end - beg;
      const int chunk = (len + p.splits - 1) / p.splits;
      const int l0 = min(si * chunk, len);
      const int h0 = min(l0 + chunk, len);
      lo[s] = beg + l0;
      hi[s] = beg + h0;
      nt[s] = (h0 - l0 + kTile - 1) / kTile;
    }
  }
  const int T = nt[0] + nt[1];

  auto tile_pos = [&](int t, int& s) -> int {
    s = tile_seg(t, nt[0]);
    return s ? lo[1] + kTile * (t - nt[0]) : lo[0] + kTile * t;
  };
  auto load_idx = [&](int t) -> int {
    if (t >= T) return -1;
    int s;
    const int pos = tile_pos(t, s) + c;
    const int* idx = s ? p.seg[1].idx : p.seg[0].idx;
    return pos < (s ? hi[1] : hi[0]) ? idx[pos] : -1;
  };
  auto load_tile = [&](int t, int idx) -> Tile {
    Tile r;
    int s;
    (void)tile_pos(t, s);
    const Seg sg = pick(p, s);
    const bool valid = idx >= 0 && (int64_t)idx < sg.rows;
    r.vmask = (unsigned)(__ballot(valid) & 0xffffull);
    r.ocp = sg.ocp;
    if (valid) {
      const int64_t blk = idx >> sg.bs_shift;
      const int pos = idx & ((1 << sg.bs_shift) - 1);
      const uint8_t* base = sg.cache + blk * sg.stride0;
      const uint8_t* row = base + pos * kRowBytes + 16 * g;
  #pragma unroll
      for (int m = 0; m < 9; ++m) r.d[m] = *(const u32x4*)(row + 64 * m);
      r.sc =
          *(const u32x2*)(base + ((int64_t)kRowBytes << sg.bs_shift) + pos * 8);
    } else {
  #pragma unroll
      for (int m = 0; m < 9; ++m) r.d[m] = u32x4{0u, 0u, 0u, 0u};
      r.sc = u32x2{0u, 0u};
    }
    return r;
  };

  f32x4 acc[32];
  #pragma unroll
  for (int k = 0; k < 32; ++k) acc[k] = f32x4{0.f, 0.f, 0.f, 0.f};
  float m_run = -INFINITY;
  float l_lane = 0.f;

  // S^T for one tile; also stashes the dequantized bf16 K in LDS for P V.
  auto qk = [&](Tile& cur, auto ocp_tag, auto scrub_tag) -> f32x4 {
    constexpr bool kOcp = decltype(ocp_tag)::value;
    constexpr bool kScrub = decltype(scrub_tag)::value;
    // Four independent accumulator chains keep the MFMA pipe full.
    f32x4 sp[4];
  #pragma unroll
    for (int qq = 0; qq < 4; ++qq) sp[qq] = f32x4{0.f, 0.f, 0.f, 0.f};
  #pragma unroll
    for (int J = 0; J < 7; ++J) {
      const float sc = e8m0(cur.sc, J, kOcp ? 1 : 0);
      unsigned packed[8];
  #pragma unroll
      for (int qq = 0; qq < 4; ++qq) {
        const unsigned w = kOcp ? ocp_canon(cur.d[J][qq]) : cur.d[J][qq];
        f32x2 a01 = __builtin_amdgcn_cvt_pk_f32_fp8((int)w, false);
        f32x2 a23 = __builtin_amdgcn_cvt_pk_f32_fp8((int)w, true);
        a01 *= sc;
        a23 *= sc;
        if constexpr (kScrub) {
          a01 = f32x2{scrub(a01.x), scrub(a01.y)};
          a23 = f32x2{scrub(a23.x), scrub(a23.y)};
        }
        packed[2 * qq] = pack_hi(a01.x, a01.y);
        packed[2 * qq + 1] = pack_hi(a23.x, a23.y);
        sp[qq] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(
            as_s16x4(packed[2 * qq], packed[2 * qq + 1]), bq[J][qq], sp[qq], 0,
            0, 0);
      }
      *(u32x4*)(lds + lds_chunk(c, 8 * J + 2 * g)) =
          u32x4{packed[0], packed[1], packed[2], packed[3]};
      *(u32x4*)(lds + lds_chunk(c, 8 * J + 2 * g + 1)) =
          u32x4{packed[4], packed[5], packed[6], packed[7]};
    }
    if constexpr (kScrub) {
  #pragma unroll
      for (int m = 7; m < 9; ++m) {
        cur.d[m] = u32x4{bf16x2_scrub(cur.d[m].x), bf16x2_scrub(cur.d[m].y),
                         bf16x2_scrub(cur.d[m].z), bf16x2_scrub(cur.d[m].w)};
      }
    }
    sp[0] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(
        as_s16x4(cur.d[7].x, cur.d[7].y), bq[7][0], sp[0], 0, 0, 0);
    sp[1] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(
        as_s16x4(cur.d[7].z, cur.d[7].w), bq[7][1], sp[1], 0, 0, 0);
    sp[2] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(
        as_s16x4(cur.d[8].x, cur.d[8].y), bq[7][2], sp[2], 0, 0, 0);
    sp[3] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(
        as_s16x4(cur.d[8].z, cur.d[8].w), bq[7][3], sp[3], 0, 0, 0);
    *(u32x4*)(lds + lds_chunk(c, 56 + g)) = cur.d[7];
    *(u32x4*)(lds + lds_chunk(c, 60 + g)) = cur.d[8];
    return (sp[0] + sp[1]) + (sp[2] + sp[3]);
  };

  auto process = [&](Tile& cur) {
    f32x4 s = cur.ocp ? qk(cur, std::true_type{}, std::false_type{})
                      : qk(cur, std::false_type{}, std::false_type{});
    if (__any(s[0] != s[0] || s[1] != s[1] || s[2] != s[2] || s[3] != s[3])) {
      s = cur.ocp ? qk(cur, std::true_type{}, std::true_type{})
                  : qk(cur, std::false_type{}, std::true_type{});
    }
    const unsigned vmask = cur.vmask;

    // Online softmax over S^T[key 4g+i][head c].
    float sv[4];
    float tmax = -INFINITY;
  #pragma unroll
    for (int i = 0; i < 4; ++i) {
      const bool ok = (vmask >> (4 * g + i)) & 1u;
      sv[i] = ok ? s[i] * p.scale_l2 : -INFINITY;
      tmax = fmaxf(tmax, sv[i]);
    }
    tmax = fmaxf(tmax, __shfl_xor(tmax, 16));
    tmax = fmaxf(tmax, __shfl_xor(tmax, 32));
    if (__any(tmax > m_run + kLazyRescale)) {
      const float m_new = fmaxf(m_run, tmax);
      const float alpha =
          m_run == -INFINITY ? 0.f : __builtin_amdgcn_exp2f(m_run - m_new);
  #pragma unroll
      for (int k = 0; k < 32; ++k) acc[k] *= alpha;
      l_lane *= alpha;
      m_run = m_new;
    }
    float pv[4];
  #pragma unroll
    for (int i = 0; i < 4; ++i) {
      pv[i] = sv[i] == -INFINITY ? 0.f : __builtin_amdgcn_exp2f(sv[i] - m_run);
      l_lane += pv[i];
    }
    const s16x4 bp = as_s16x4(pack_rne(pv[0], pv[1]), pack_rne(pv[2], pv[3]));

    // O^T += V^T P^T, V^T read keys-contiguous from LDS.
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
    u32x2 rv[2][4];
  #pragma unroll
    for (int i = 0; i < 4; ++i)
      rv[0][i] =
          *(const u32x2*)(lds + lds_chunk(4 * g + i, (c >> 1)) + 8 * (c & 1));
  #pragma unroll
    for (int u = 0; u < 8; ++u) {
      u32x2* r = rv[u & 1];
      if (u + 1 < 8) {
  #pragma unroll
        for (int i = 0; i < 4; ++i)
          rv[(u + 1) & 1][i] = *(
              const u32x2*)(lds + lds_chunk(4 * g + i, 8 * (u + 1) + (c >> 1)) +
                            8 * (c & 1));
      }
      acc[4 * u + 0] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(
          as_s16x4(perm_lo16(r[0].x, r[1].x), perm_lo16(r[2].x, r[3].x)), bp,
          acc[4 * u + 0], 0, 0, 0);
      acc[4 * u + 1] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(
          as_s16x4(perm_hi16(r[0].x, r[1].x), perm_hi16(r[2].x, r[3].x)), bp,
          acc[4 * u + 1], 0, 0, 0);
      acc[4 * u + 2] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(
          as_s16x4(perm_lo16(r[0].y, r[1].y), perm_lo16(r[2].y, r[3].y)), bp,
          acc[4 * u + 2], 0, 0, 0);
      acc[4 * u + 3] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(
          as_s16x4(perm_hi16(r[0].y, r[1].y), perm_hi16(r[2].y, r[3].y)), bp,
          acc[4 * u + 3], 0, 0, 0);
    }
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
  };

  // Three-tile register ring: while one tile is computed, the next two are in
  // flight. Unrolled by three so buffers rotate by renaming, not copies.
  if (T > 0) {
    Tile A = load_tile(0, load_idx(0));
    Tile B, C;
    if (T > 1) B = load_tile(1, load_idx(1));
    int idx_a = load_idx(2), idx_b = load_idx(3);
    for (int t = 0; t < T; t += 3) {
      {
        const int i0 = idx_a;
        idx_a = idx_b;
        idx_b = load_idx(t + 4);
        if (t + 2 < T) C = load_tile(t + 2, i0);
        process(A);
      }
      if (t + 1 >= T) break;
      {
        const int i0 = idx_a;
        idx_a = idx_b;
        idx_b = load_idx(t + 5);
        if (t + 3 < T) A = load_tile(t + 3, i0);
        process(B);
      }
      if (t + 2 >= T) break;
      {
        const int i0 = idx_a;
        idx_a = idx_b;
        idx_b = load_idx(t + 6);
        if (t + 4 < T) B = load_tile(t + 4, i0);
        process(C);
      }
    }
  }

  float l_head = l_lane;
  l_head += __shfl_xor(l_head, 16);
  l_head += __shfl_xor(l_head, 32);

  // acc[4u+v][i] is O[head c][64u + 16g + 4i + v]
  if constexpr (kSplit) {
    const int64_t base = ((int64_t)qi * p.splits + si) * kHeads + c;
    if (g == 0) {
      p.part_m[base] = m_run == -INFINITY ? kNegLarge : m_run * kLn2;
      p.part_l[base] = l_head;
    }
    float* dst = p.part_acc + base * kHeadDim;
  #pragma unroll
    for (int u = 0; u < 8; ++u) {
  #pragma unroll
      for (int i = 0; i < 4; ++i) {
        *(f32x4*)(dst + 64 * u + 16 * g + 4 * i) =
            f32x4{acc[4 * u + 0][i], acc[4 * u + 1][i], acc[4 * u + 2][i],
                  acc[4 * u + 3][i]};
      }
    }
  } else {
    float m_f = m_run;
    float sink_l2 = -INFINITY;
    if (p.sink != nullptr) {
      sink_l2 = p.sink[c] * kLog2e;
      m_f = fmaxf(m_f, sink_l2);
    }
    const float w =
        m_run == -INFINITY ? 0.f : __builtin_amdgcn_exp2f(m_run - m_f);
    float denom = l_head * w;
    if (p.sink != nullptr) denom += __builtin_amdgcn_exp2f(sink_l2 - m_f);
    const float f = (l_head > 0.f && denom > 0.f) ? w / denom : 0.f;
    uint16_t* dst = p.out + (int64_t)qi * p.o_s0 + (int64_t)c * p.o_s1;
  #pragma unroll
    for (int u = 0; u < 8; ++u) {
      unsigned o[8];
  #pragma unroll
      for (int i = 0; i < 4; ++i) {
        o[2 * i] = pack_rne(acc[4 * u + 0][i] * f, acc[4 * u + 1][i] * f);
        o[2 * i + 1] = pack_rne(acc[4 * u + 2][i] * f, acc[4 * u + 3][i] * f);
      }
      *(u32x4*)(dst + 64 * u + 16 * g) = u32x4{o[0], o[1], o[2], o[3]};
      *(u32x4*)(dst + 64 * u + 16 * g + 8) = u32x4{o[4], o[5], o[6], o[7]};
    }
  }
#endif  // __HIP__SPARSE_MLA_DECODE__
}

// Combines split partials (natural-log running max, unnormalized l and O) with
// the attention sink.
__global__ __launch_bounds__(256) void sparse_mla_decode_reduce_kernel(
    const float* __restrict__ pm, const float* __restrict__ pl,
    const float* __restrict__ pa, const float* __restrict__ sink, uint16_t* out,
    int64_t o_s0, int64_t o_s1, int splits) {
#if defined(__HIP__SPARSE_MLA_DECODE__)
  const int qi = blockIdx.x;
  const int h = threadIdx.x >> 4;
  const int d0 = (threadIdx.x & 15) * 32;
  const int64_t base = (int64_t)qi * splits * kHeads + h;
  float m_f = kNegLarge;
  for (int s = 0; s < splits; ++s)
    m_f = fmaxf(m_f, pm[base + (int64_t)s * kHeads]);
  const float sk = sink != nullptr ? sink[h] : kNegLarge;
  m_f = fmaxf(m_f, sk);
  float acc[32];
  #pragma unroll
  for (int j = 0; j < 32; ++j) acc[j] = 0.f;
  float l = 0.f;
  for (int s = 0; s < splits; ++s) {
    const int64_t row = base + (int64_t)s * kHeads;
    const float w = __expf(pm[row] - m_f);
    l += w * pl[row];
    const f32x4* src = (const f32x4*)(pa + row * kHeadDim + d0);
  #pragma unroll
    for (int j = 0; j < 8; ++j) {
      const f32x4 v = src[j];
      acc[4 * j + 0] += w * v.x;
      acc[4 * j + 1] += w * v.y;
      acc[4 * j + 2] += w * v.z;
      acc[4 * j + 3] += w * v.w;
    }
  }
  if (sink != nullptr) l += __expf(sk - m_f);
  const float f = l > 0.f ? 1.f / fmaxf(l, 1e-30f) : 0.f;
  uint16_t* dst = out + (int64_t)qi * o_s0 + (int64_t)h * o_s1 + d0;
  #pragma unroll
  for (int j = 0; j < 4; ++j) {
    *(u32x4*)(dst + 8 * j) =
        u32x4{pack_rne(acc[8 * j] * f, acc[8 * j + 1] * f),
              pack_rne(acc[8 * j + 2] * f, acc[8 * j + 3] * f),
              pack_rne(acc[8 * j + 4] * f, acc[8 * j + 5] * f),
              pack_rne(acc[8 * j + 6] * f, acc[8 * j + 7] * f)};
  }
#endif  // __HIP__SPARSE_MLA_DECODE__
}

namespace {

int log2_exact(int64_t v) {
  int s = 0;
  while ((int64_t(1) << s) < v) ++s;
  TORCH_CHECK((int64_t(1) << s) == v, "block size must be a power of two, got ",
              v);
  return s;
}

Seg make_seg(const torch::Tensor& cache, const torch::Tensor& idx,
             const torch::Tensor& ptr, bool ocp, int64_t num_q) {
  TORCH_CHECK(cache.dtype() == torch::kUInt8 && cache.dim() == 3,
              "cache must be uint8 [blocks, block_size, bytes]");
  TORCH_CHECK(cache.stride(2) == 1 && cache.stride(1) == cache.size(2),
              "cache rows must be contiguous");
  TORCH_CHECK(cache.size(2) == kRowBytes + 8,
              "fp8_ds_mla rows are 584 bytes, got ", cache.size(2));
  TORCH_CHECK(cache.stride(0) % 16 == 0,
              "cache block stride must be 16-byte aligned");
  TORCH_CHECK(idx.dtype() == torch::kInt32 && ptr.dtype() == torch::kInt32,
              "indices and indptr must be int32");
  TORCH_CHECK(idx.is_contiguous() && ptr.is_contiguous(),
              "indices and indptr must be contiguous");
  TORCH_CHECK(ptr.numel() >= num_q + 1, "indptr needs num_tokens + 1 entries");
  Seg s;
  s.cache = cache.data_ptr<uint8_t>();
  s.idx = idx.data_ptr<int>();
  s.ptr = ptr.data_ptr<int>();
  s.stride0 = cache.stride(0);
  s.rows = cache.size(0) * cache.size(1);
  s.bs_shift = log2_exact(cache.size(1));
  s.ocp = ocp ? 1 : 0;
  return s;
}

}  // namespace
}  // namespace sparse_mla_decode
}  // namespace vllm

void sparse_mla_decode_fp8_ds_mla(
    torch::Tensor& out, const torch::Tensor& q, const torch::Tensor& main_cache,
    const torch::Tensor& main_indices, const torch::Tensor& main_indptr,
    const std::optional<torch::Tensor>& extra_cache,
    const std::optional<torch::Tensor>& extra_indices,
    const std::optional<torch::Tensor>& extra_indptr,
    const std::optional<torch::Tensor>& attn_sink, double scale, bool main_ocp,
    bool extra_ocp, int64_t num_splits) {
  using namespace vllm::sparse_mla_decode;
  TORCH_CHECK(q.dtype() == torch::kBFloat16 && q.dim() == 3,
              "q must be bf16 [num_tokens, 16, 512]");
  TORCH_CHECK(q.size(1) == kHeads && q.size(2) == kHeadDim && q.stride(2) == 1,
              "q must be [num_tokens, 16, 512] with contiguous rows");
  TORCH_CHECK(q.stride(0) % 8 == 0 && q.stride(1) % 8 == 0,
              "q strides must be 16-byte aligned");
  TORCH_CHECK(out.dtype() == torch::kBFloat16 && out.sizes() == q.sizes() &&
                  out.stride(2) == 1 && out.stride(0) % 8 == 0 &&
                  out.stride(1) % 8 == 0,
              "out must be bf16 with q's shape and 16-byte aligned rows");
  TORCH_CHECK(num_splits >= 1, "num_splits must be >= 1");
  const bool has_extra = extra_cache.has_value();
  TORCH_CHECK(has_extra == extra_indices.has_value() &&
                  has_extra == extra_indptr.has_value(),
              "extra_cache, extra_indices and extra_indptr go together");
  const at::cuda::OptionalCUDAGuard device_guard(device_of(q));
  const cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(std::strncmp(prop->gcnArchName, "gfx942", 6) == 0,
              "sparse_mla_decode_fp8_ds_mla is only built for gfx942, got ",
              prop->gcnArchName);

  Params p{};
  p.q = reinterpret_cast<const uint16_t*>(q.data_ptr());
  p.q_s0 = q.stride(0);
  p.q_s1 = q.stride(1);
  p.num_q = static_cast<int>(q.size(0));
  p.seg[0] = make_seg(main_cache, main_indices, main_indptr, main_ocp, p.num_q);
  p.nseg = 1;
  if (has_extra) {
    p.seg[1] = make_seg(*extra_cache, *extra_indices, *extra_indptr, extra_ocp,
                        p.num_q);
    p.nseg = 2;
  } else {
    p.seg[1] = p.seg[0];
  }
  if (attn_sink.has_value()) {
    TORCH_CHECK(attn_sink->dtype() == torch::kFloat32 &&
                    attn_sink->is_contiguous() && attn_sink->numel() >= kHeads,
                "attn_sink must be contiguous fp32 with 16 entries");
    p.sink = attn_sink->data_ptr<float>();
  }
  p.scale_l2 = static_cast<float>(scale) * kLog2e;
  p.splits = static_cast<int>(num_splits);
  p.out = reinterpret_cast<uint16_t*>(out.data_ptr());
  p.o_s0 = out.stride(0);
  p.o_s1 = out.stride(1);
  if (p.num_q == 0) return;

  const int items = p.num_q * p.splits;
  const dim3 grid((items + kWaves - 1) / kWaves);
  const dim3 block(kWaves * 64);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (p.splits == 1) {
    sparse_mla_decode_kernel<false><<<grid, block, 0, stream>>>(p);
  } else {
    const auto f32 = q.options().dtype(torch::kFloat32);
    torch::Tensor part_m = torch::empty({p.num_q, p.splits, kHeads}, f32);
    torch::Tensor part_l = torch::empty({p.num_q, p.splits, kHeads}, f32);
    torch::Tensor part_acc =
        torch::empty({p.num_q, p.splits, kHeads, kHeadDim}, f32);
    p.part_m = part_m.data_ptr<float>();
    p.part_l = part_l.data_ptr<float>();
    p.part_acc = part_acc.data_ptr<float>();
    sparse_mla_decode_kernel<true><<<grid, block, 0, stream>>>(p);
    sparse_mla_decode_reduce_kernel<<<p.num_q, 256, 0, stream>>>(
        p.part_m, p.part_l, p.part_acc, p.sink, p.out, p.o_s0, p.o_s1,
        p.splits);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
