// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Packed prefix phase of chunked prefill, adapted from octave_prefill_rdna3.cu.
// Eight loader waves unpack integer K and FP16 V tiles for ten or twelve
// compute-wave pairs. The first wave in each pair computes QK for 16 query
// vectors. The second consumes the preceding tile's probabilities and
// accumulates PV.

#include "kernels.h"
#include "../octave_format.cuh"
#include <cmath>
#include <type_traits>

namespace {
using octave::D;
using octave::Format;
using octave::LUT_ONE;
using namespace octave_rdna4;
using h8v = _Float16 __attribute__((ext_vector_type(8)));
using h2v = _Float16 __attribute__((ext_vector_type(2)));
using f8v = float __attribute__((ext_vector_type(8)));
using u4v = uint32_t __attribute__((ext_vector_type(4)));
using u2v = uint32_t __attribute__((ext_vector_type(2)));
using i2v = int __attribute__((ext_vector_type(2)));
using i8v = int __attribute__((ext_vector_type(8)));
constexpr int kTile = 16, kLoaderWaves = 8;

constexpr int kKeyStride = D + 16, kValuePadding = 16;
__device__ __forceinline__ int vrow(int row) {
  return row * kTile + (row >> 3) * kValuePadding;
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
__device__ inline __attribute__((convergent)) void tile_barrier() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
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

template <int VB, bool KC, int kPairs>
__global__ __launch_bounds__((2 * kPairs + kLoaderWaves) * 32) void prefix_attn(
    const int8_t* query, const float* query_scale, PagedCache cache,
    const int32_t* block_table, int64_t table_stride, int64_t prefix_capacity,
    const int32_t* starts, const int32_t* lengths, float* partials, float scale,
    int requests, int num_heads, int tokens) {
  using F = Format<VB, KC>;
  const int group = num_heads / cache.heads;
  int seq = 0, hi = requests;
  while (seq + 1 < hi) {
    int mid = (seq + hi) / 2;
    if (int64_t(starts[mid]) * group / (kPairs * kTile) + mid <= blockIdx.x)
      seq = mid;
    else
      hi = mid;
  }
  const int q0 = starts[seq], qlen = starts[seq + 1] - q0;
  const int rowtile =
      blockIdx.x - (int64_t(q0) * group / (kPairs * kTile) + seq);
  const int nqv = qlen * group, kvh = blockIdx.y;
  if (rowtile * kPairs * kTile >= nqv) return;
  const int ctx = bounded_prefix(lengths[seq], qlen, prefix_capacity);
  const int used = prefix_splits(ctx, gridDim.z);
  const int split = blockIdx.z;
  if (split >= used) return;
  const int tiles = 1 + (ctx - 1) / kTile;
  const int k_begin = kTile * (int64_t(tiles) * split / used);
  const int k_stop = split + 1 == used
                         ? ctx
                         : int(kTile * (int64_t(tiles) * (split + 1) / used));
  const int ntiles = 1 + (k_stop - k_begin - 1) / kTile;
  const int tid = threadIdx.x, lane = tid & 31, j = lane & 15, h = lane >> 4;
  const int w = __builtin_amdgcn_readfirstlane(tid >> 5);
  const int block_size = cache.page_size;
  __shared__ __attribute__((aligned(16))) int8_t keys[2][kTile][kKeyStride];
  __shared__ __attribute__((
      aligned(16))) _Float16 values[2][D * kTile + (D / 8) * kValuePadding];
  __shared__ __attribute__((aligned(16))) h8v probability_tile[2][kPairs][32];
  __shared__ float rescale_tile[2][kPairs][16];
  // The QK wave folds each token's V scale into its probabilities.
  __shared__ float key_scales[2][KC ? 1 : 4][kTile];
  __shared__ float value_scales[2][kTile];

  // ---- loader waves
  if (w >= 2 * kPairs) {
    // 256 threads. K: thread lt unpacks dims 16 kp .. 16 kp + 15 of block kq
    // of token tk. V: dims 4 vh .. 4 vh + 3 of the 8-dim group vd of token
    // quad vq.
    const int lt = tid - 2 * kPairs * 32;
    const int tk = lt >> 4, kq = (lt >> 2) & 3, kp = lt & 3;
    const int vu = lt >> 1, vh = lt & 1, vq = vu & 3, vd = (vu >> 2) * 8;
    // Block sizes are multiples of kTile, so a tile sits in one cache block.
    // Tiles past the end read the last one; tokens past k_stop read the last
    // token: their K is masked to -inf and their V meets p = 0.
    const int tlast = k_stop - 1;
    const uint8_t* const head = cache.data + int64_t(kvh) * cache.head_stride;
    const int* const bt = block_table + int64_t(seq) * table_stride;
    // Walk the tiles in order: block index and offset inside it, advanced by
    // kTile per tile and frozen on the last tile.
    struct Pos {
      int start, blk, off;
    };
    auto first = [&]() {
      return Pos{k_begin, k_begin / block_size, k_begin % block_size};
    };
    auto next = [&](Pos p) {
      if (tlast - p.start < kTile) return p;
      p.start += kTile;
      p.off += kTile;
      if (p.off == block_size) {
        p.off = 0;
        ++p.blk;
      }
      return p;
    };
    auto pbase = [&](Pos p, int pb) {
      return head + (int64_t)(pb < 0 ? 0 : pb) * cache.page_stride +
             (int64_t)p.off * cache.token_stride;
    };
    auto tok = [&](int start, int r) {
      return (int64_t)min(r, tlast - start) * cache.token_stride;
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
      uint32_t value_scale_word;
      uint32_t sk0, sk1;
    };
    auto fetch = [&](int TK, const uint8_t* bK, int TV, const uint8_t* bV,
                     Stg& S) {
      const uint8_t* pk = bK + tok(TK, tk);
      if (lt < kTile) {
        const auto* ps =
            reinterpret_cast<const uint32_t*>(bK + tok(TK, lt) + F::OFF_SC);
        S.sk0 = ps[0];
        if constexpr (!KC) S.sk1 = ps[1];
      }
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
      if (lt >= kTile && lt < 2 * kTile) {
        const uint8_t* ps = bK + tok(TK, lt - kTile) + F::OFF_SC + (KC ? 0 : 8);
        S.value_scale_word = *reinterpret_cast<const uint32_t*>(ps);
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
      *reinterpret_cast<u4v*>(&keys[bf][tk][64 * kq + 16 * kp]) = o;
      if (lt < kTile) {
        const h2v a = __builtin_bit_cast(h2v, S.sk0);
        key_scales[bf][0][lt] = float(a[0]);
        if constexpr (!KC) {
          const h2v b = __builtin_bit_cast(h2v, S.sk1);
          key_scales[bf][1][lt] = float(a[1]);
          key_scales[bf][2][lt] = float(b[0]);
          key_scales[bf][3][lt] = float(b[1]);
        }
      }
      if (lt >= kTile && lt < 2 * kTile)
        value_scales[bf][lt - kTile] = (float)__builtin_bit_cast(
            _Float16,
            (uint16_t)(KC ? S.value_scale_word >> 16 : S.value_scale_word));
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
        *(u2v*)&values[bf][vrow(row) + ((4 * vq) ^ sw)] =
            (u2v){__builtin_bit_cast(uint32_t, p01 - kofs),
                  __builtin_bit_cast(uint32_t, p23 - kofs)};
      }
    };
    // Pipeline: K of tile i + 1 and V of tile i are written in iteration i
    // from registers; the block number of a tile is loaded two tiles ahead.
    Stg S;
    {
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
    }
    tile_barrier();
    return;
  }

  const int pair = w >> 1;
  const int qv = rowtile * kPairs * kTile + pair * kTile + j;
  const bool live = qv < nqv;
  const int tok = q0 + (live ? qv / group : 0);
  const int head = kvh * group + (live ? qv % group : 0);
  const int64_t row = int64_t(tok) * num_heads + head;
  float* dest =
      partials + (int64_t(split) * tokens * num_heads + row) * (D + 2);
  if ((w & 1) == 0) {
    const int8_t* qrow = query + row * 2 * D;
    i2v qmain[D / kTile], qres[D / kTile];
#pragma unroll
    for (int f = 0; f < D / kTile; ++f) {
      qmain[f] = *reinterpret_cast<const i2v*>(qrow + kTile * f + 8 * h);
      qres[f] = *reinterpret_cast<const i2v*>(qrow + D + kTile * f + 8 * h);
    }
    const float factor =
        scale * query_scale[row] * (1.4426950408889634f / LUT_ONE);
    float maximum = -INFINITY, mass = 0.f;
    tile_barrier();
    for (int i = 0; i < ntiles; ++i) {
      const int bf = i & 1;
      f8v scores = {};
      constexpr int blocks = KC ? 1 : 4;
      constexpr int fragments = D / (kTile * blocks);
#pragma unroll
      for (int b = 0; b < blocks; ++b) {
        i8v acc = {};
        i2v keys_block[fragments];
#pragma unroll
        for (int step = 0; step < fragments; ++step) {
          const int f = b * fragments + step;
          keys_block[step] =
              *reinterpret_cast<const i2v*>(&keys[bf][j][kTile * f + 8 * h]);
          acc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
              true, keys_block[step], true, qmain[f], acc, false);
        }
        // Blocks contain at most 256 products. Both digits and centroids fit
        // in signed bytes, so folding the high dot stays inside signed int32.
        acc *= 256;
#pragma unroll
        for (int step = 0; step < fragments; ++step) {
          const int f = b * fragments + step;
          acc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
              true, keys_block[step], true, qres[f], acc, false);
        }
#pragma unroll
        for (int e = 0; e < 8; ++e)
          scores[e] =
              fmaf(float(acc[e]), key_scales[bf][b][8 * h + e], scores[e]);
      }
      const int base = k_begin + i * kTile;
      if (k_stop - base < kTile) {
#pragma unroll
        for (int e = 0; e < 8; ++e)
          if (base + 8 * h + e >= k_stop) scores[e] = -INFINITY;
      }
      const float hi = fmaxf(
          fmaxf(fmaxf(scores[0], scores[1]), fmaxf(scores[2], scores[3])),
          fmaxf(fmaxf(scores[4], scores[5]), fmaxf(scores[6], scores[7])));
      const float next = fmaxf(maximum, fmaxf(hi, xhalf(hi)) * factor);
      const float alpha = __builtin_amdgcn_exp2f(maximum - next);
      h8v probabilities;
      float sum = 0.f;
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const float p = __builtin_amdgcn_exp2f(fmaf(scores[e], factor, -next));
        sum += p;
        probabilities[e] = (_Float16)(p * value_scales[bf][8 * h + e]);
      }
      mass = fmaf(mass, alpha, sum);
      maximum = next;
      probability_tile[bf][pair][lane] = probabilities;
      if (!h) rescale_tile[bf][pair][j] = alpha;
      tile_barrier();
    }
    tile_barrier();
    mass += xhalf(mass);
    if (live && !h) {
      dest[D] = maximum;
      dest[D + 1] = mass;
    }
  } else {
    f8v accum[D / kTile] = {};
    // Wait for the first K tile, then for its probabilities and V tile.
    tile_barrier();
    tile_barrier();
    for (int i = 0; i < ntiles; ++i) {
      const int bf = i & 1;
      const h8v probabilities = probability_tile[bf][pair][lane];
      const float alpha = rescale_tile[bf][pair][j];
      if (__builtin_amdgcn_ballot_w32(alpha != 1.f)) {
#pragma unroll
        for (int f = 0; f < D / kTile; ++f) accum[f] *= alpha;
      }
#pragma unroll
      for (int f = 0; f < D / kTile; ++f) {
        const int d = kTile * f + j;
        const h8v v = *reinterpret_cast<const h8v*>(
            &values[bf][vrow(d) + 8 * (h ^ ((d >> 3) & 1))]);
        accum[f] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
            v, probabilities, accum[f]);
      }
      tile_barrier();
    }
    if (!live) return;
#pragma unroll
    for (int f = 0; f < D / kTile; ++f)
#pragma unroll
      for (int e = 0; e < 8; ++e)
        dest[kTile * f + 8 * h + e] = accum[f][e] / LUT_ONE;
  }
}

}  // namespace

int octave_rdna4::launch_prefix_tiles(
    const int8_t* query, const float* query_scale, PagedCache cache,
    const int32_t* table, int64_t table_stride, int64_t prefix_capacity,
    const int32_t* starts, const int32_t* lengths, float* partials, float scale,
    int requests, int heads, int tokens, int splits, int format,
    hipStream_t stream) {
  auto run = [&](auto pairs) {
    constexpr int kPairs = decltype(pairs)::value;
    constexpr int kThreads = (2 * kPairs + kLoaderWaves) * 32;
    const dim3 grid(
        int64_t(tokens) * (heads / cache.heads) / (kPairs * kTile) + requests,
        cache.heads, splits);
    auto launch = [&](auto kernel) {
      kernel<<<grid, kThreads, 0, stream>>>(
          query, query_scale, cache, table, table_stride, prefix_capacity,
          starts, lengths, partials, scale, requests, heads, tokens);
    };
    if (format == 4)
      launch(prefix_attn<4, false, kPairs>);
    else if (format == 3)
      launch(prefix_attn<3, false, kPairs>);
    else
      launch(prefix_attn<3, true, kPairs>);
  };
  if (requests > 1 && tokens / requests >= 192)
    run(std::integral_constant<int, 12>{});
  else
    run(std::integral_constant<int, 10>{});
  return hipGetLastError();
}
