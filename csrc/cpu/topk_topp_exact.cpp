// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Exact top-k/top-p masking of FP32 logits on x86 (AVX-512, or AVX2 with 8-lane
// vectors through the same code), with the contract of
// the Triton kernels: the kept logits are a prefix of (logit descending, index
// ascending), NaN counts as -inf, top-k keeps exactly min(k, finite) and top-p
// the smallest prefix whose FP64 mass reaches p.
//
// Per row: group maxima bound the cut, the logits above the bound are gathered,
// and rounds of 16 sampled pivots narrow them to at most 256, sorted exactly on
// int64 keys. Rows whose nucleus outgrows the gather (flat logits) narrow the
// row.
//
// Dtypes follow topk_topp_triton, with two exceptions: masses that decide a cut
// add FP32 exp weights in FP64 per logit, and the fill compares the FP32 logit
// and int32 index against the cut decoded from its int64 key.

// ATen's 512-bit (or 256-bit) Vectorized, in its own capability namespace as
// ATen's per-ISA builds do, so it cannot clash with the Vectorized other units
// compile.
#if !defined(CPU_CAPABILITY)
  #if defined(__AVX512F__)
    #define CPU_CAPABILITY AVX512
    #define CPU_CAPABILITY_AVX512
  #elif defined(__AVX2__)
    #define CPU_CAPABILITY AVX2
    #define CPU_CAPABILITY_AVX2
  #endif
#endif
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>

#include "cpu_types.hpp"

#include <omp.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#if defined(__AVX512F__) || defined(__AVX2__)

namespace {

using FVec = at::vec::Vectorized<float>;
using IVec = at::vec::Vectorized<int32_t>;
using DVec = at::vec::Vectorized<double>;
constexpr int kVec = FVec::size();  // 16 lanes on AVX-512, 8 on AVX2
constexpr int kGroup = 16;          // logits per group maximum
constexpr int kSuper = 256;         // logits per supergroup maximum
static_assert(IVec::size() == kVec && 2 * DVec::size() == kVec &&
                  kGroup % kVec == 0 && (kSuper / kGroup) % kVec == 0,
              "needs ATen's 512- or 256-bit Vectorized");
constexpr int kMaxSplits = 32;
constexpr int kCand = 32768;      // items a candidate buffer holds
constexpr int kSlack = kVec;      // full-vector stores past the last item
constexpr int kSample = 256;      // items sampled to place the pivots
constexpr int kRowSample = 1024;  // items sampled to bracket the cut in a row
constexpr int kBoundSample = 128;
constexpr int kPivots = 16;
constexpr int kSmall = 256;  // items sorted exactly
constexpr int kMaxRounds = 1024;

// Set by a narrowing that ran out of rounds: errors cannot leave the OpenMP
// region (they would terminate the process), so the op raises after it.
std::atomic<bool>& narrow_failed() {
  static std::atomic<bool> failed{false};
  return failed;
}
constexpr float kInf = std::numeric_limits<float>::infinity();
constexpr int32_t kLastIndex = std::numeric_limits<int32_t>::max();

// ---------------------------------------------------------------- keys

inline uint32_t ordered(float x) {
  const float y = x == 0.0f ? 0.0f : x;  // -0.0 orders as 0.0
  uint32_t bits;
  std::memcpy(&bits, &y, 4);
  return bits >= 0x80000000u ? 0xFFFFFFFFu - bits : bits + 0x80000000u;
}

// Ascending keys: logit descending, then index ascending (as topk_topp_triton).
inline int64_t make_key(float x, int32_t i) {
  const int64_t hi = 0x7FFFFFFFLL - static_cast<int64_t>(ordered(x));
  return static_cast<int64_t>(static_cast<uint64_t>(hi) << 32) |
         static_cast<uint32_t>(i);
}

inline float key_value(int64_t key) {
  const int64_t o = 0x7FFFFFFFLL - (key >> 32);
  const uint32_t bits = o >= 0x80000000LL
                            ? static_cast<uint32_t>(o - 0x80000000LL)
                            : static_cast<uint32_t>(0xFFFFFFFFLL - o);
  float x;
  std::memcpy(&x, &bits, 4);
  return x;
}

inline int32_t key_index(int64_t key) {
  return static_cast<int32_t>(key & 0xFFFFFFFF);
}

inline bool before(float x, int32_t i, float px, int32_t pi) {
  return x > px || (x == px && i < pi);
}

// ---------------------------------------------------------------- ISA helpers
// All the intrinsics of this kernel, one block per ISA with the same interface:
// what Vectorized has no form of (compressing stores, FP32 to FP64 widening,
// the shuffle network taking a vector of group maxima at once), and lane-mask
// operations the hot passes need on mask bits. Vectorized compares and blends
// carry masks as vectors, and converting them back to bits in every pivot step
// made select 3-5x slower.

using Lanes = uint32_t;  // a bit per lane of a vector
constexpr Lanes kAllLanes = (1u << kVec) - 1;

// The lanes of a vector at pos that lie before n.
inline Lanes tail(int pos, int n) {
  const int left = n - pos;
  if (left <= 0) return 0;
  return left >= kVec ? kAllLanes : (1u << left) - 1;
}

  #if defined(__AVX512F__)

// count floats at p, fill past them, in one masked load (Vectorized::loadu
// fills zeros).
inline FVec load_filled(const float* p, int count, float fill) {
  return _mm512_mask_loadu_ps(_mm512_set1_ps(fill),
                              static_cast<__mmask16>(tail(0, count)), p);
}

inline Lanes gt(FVec a, FVec b) { return _mm512_cmp_ps_mask(a, b, _CMP_GT_OQ); }
inline Lanes ge(FVec a, FVec b) { return _mm512_cmp_ps_mask(a, b, _CMP_GE_OQ); }
inline Lanes eq(FVec a, FVec b) { return _mm512_cmp_ps_mask(a, b, _CMP_EQ_OQ); }
inline Lanes lt(IVec a, IVec b) { return _mm512_cmplt_epi32_mask(a, b); }
inline Lanes le(IVec a, IVec b) { return _mm512_cmple_epi32_mask(a, b); }

// acc plus v in the lanes of m.
inline FVec add_lanes(FVec acc, FVec v, Lanes m) {
  return _mm512_mask_add_ps(acc, static_cast<__mmask16>(m), acc, v);
}

// v in the lanes of m, other elsewhere.
inline FVec blend(Lanes m, FVec other, FVec v) {
  return _mm512_mask_blend_ps(static_cast<__mmask16>(m), other, v);
}

// The largest lane: one reduction, where the generic vec_reduce_all shuffles
// through a NaN-aware maximum.
inline float reduce_max(FVec v) { return _mm512_reduce_max_ps(v); }

// The lanes of v in m, stored contiguously at dst (a whole vector is written).
inline void compress_store(float* dst, Lanes m, FVec v) {
  _mm512_storeu_ps(dst, _mm512_maskz_compress_ps(m, v));
}

inline void compress_store(int32_t* dst, Lanes m, IVec v) {
  _mm512_storeu_si512(dst, _mm512_maskz_compress_epi32(m, v));
}

// NaN lanes of x as -inf, NaN being never kept: one masked move, where isnan()
// and blendv pass the mask through vectors (150 us more per row exp pass).
inline FVec drop_nan(FVec x) {
  const __m512 r = x;
  return _mm512_mask_mov_ps(r, _mm512_cmp_ps_mask(r, r, _CMP_UNORD_Q),
                            _mm512_set1_ps(-kInf));
}

// acc plus v in the lanes of m (8 lanes).
inline DVec add_lanes(DVec acc, DVec v, Lanes m) {
  return _mm512_mask_add_pd(acc, static_cast<__mmask8>(m), acc, v);
}

// The low and high 8 lanes of v, in FP64.
inline std::pair<DVec, DVec> widen(FVec v) {
  const __m512 r = v;
  const __m256 hi =
      _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(r), 1));
  return {DVec(_mm512_cvtps_pd(_mm512_castps512_ps256(r))),
          DVec(_mm512_cvtps_pd(hi))};
}

// The maxima of 16 vectors, in some lane order: 15 shuffle steps instead of 16
// horizontal reductions.
inline __m512 max16_unordered(const __m512* v) {
  __m512 u[8], w[4], y[2];
  for (int i = 0; i < 8; ++i) {  // halves: 128-bit chunks 0|2 and 1|3
    u[i] = _mm512_max_ps(_mm512_shuffle_f32x4(v[i], v[i + 8], 0x44),
                         _mm512_shuffle_f32x4(v[i], v[i + 8], 0xEE));
  }
  for (int i = 0; i < 4; ++i) {  // 128-bit chunks 0|1
    w[i] = _mm512_max_ps(_mm512_shuffle_f32x4(u[i], u[i + 4], 0x88),
                         _mm512_shuffle_f32x4(u[i], u[i + 4], 0xDD));
  }
  for (int i = 0; i < 2; ++i) {  // lanes 0|2, 1|3 within each chunk
    y[i] = _mm512_max_ps(_mm512_shuffle_ps(w[i], w[i + 2], 0x44),
                         _mm512_shuffle_ps(w[i], w[i + 2], 0xEE));
  }
  return _mm512_max_ps(_mm512_shuffle_ps(y[0], y[1], 0x88),
                       _mm512_shuffle_ps(y[0], y[1], 0xDD));
}

// The lane order of max16_unordered, found by running it on vector indices.
const __m512i& max16_order() {
  static const __m512i order = [] {
    __m512 v[16];
    for (int j = 0; j < 16; ++j) v[j] = _mm512_set1_ps(static_cast<float>(j));
    alignas(64) float lane_of[16];
    _mm512_store_ps(lane_of, max16_unordered(v));
    alignas(64) int32_t idx[16];
    for (int lane = 0; lane < 16; ++lane) {
      idx[static_cast<int>(lane_of[lane])] = lane;
    }
    return _mm512_load_si512(idx);
  }();
  return order;
}

// The 16 group maxima of a supergroup's 16 vectors: lane j of out[0] holds the
// maximum of x[j].
inline void group_maxima(const FVec* x, FVec* out) {
  __m512 r[16];
  for (int j = 0; j < 16; ++j) r[j] = x[j];
  out[0] = _mm512_permutexvar_ps(max16_order(), max16_unordered(r));
}

  #else  // AVX2

// A vector mask of the lanes in m.
inline __m256i mask_vec(Lanes m) {
  const __m256i bit = _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128);
  return _mm256_cmpeq_epi32(
      _mm256_and_si256(_mm256_set1_epi32(static_cast<int>(m)), bit), bit);
}

inline Lanes bits(__m256 m) {
  return static_cast<Lanes>(_mm256_movemask_ps(m));
}

// count floats at p, fill past them (maskload reads only the lanes it keeps).
inline FVec load_filled(const float* p, int count, float fill) {
  const __m256i m = mask_vec(tail(0, count));
  return _mm256_blendv_ps(_mm256_set1_ps(fill), _mm256_maskload_ps(p, m),
                          _mm256_castsi256_ps(m));
}

inline Lanes gt(FVec a, FVec b) {
  return bits(_mm256_cmp_ps(a, b, _CMP_GT_OQ));
}
inline Lanes ge(FVec a, FVec b) {
  return bits(_mm256_cmp_ps(a, b, _CMP_GE_OQ));
}
inline Lanes eq(FVec a, FVec b) {
  return bits(_mm256_cmp_ps(a, b, _CMP_EQ_OQ));
}
inline Lanes lt(IVec a, IVec b) {
  return bits(_mm256_castsi256_ps(_mm256_cmpgt_epi32(b, a)));
}
inline Lanes le(IVec a, IVec b) {
  return ~bits(_mm256_castsi256_ps(_mm256_cmpgt_epi32(a, b))) & kAllLanes;
}

// acc plus v in the lanes of m (zero elsewhere, which adds nothing).
inline FVec add_lanes(FVec acc, FVec v, Lanes m) {
  return _mm256_add_ps(acc, _mm256_and_ps(v, _mm256_castsi256_ps(mask_vec(m))));
}

// v in the lanes of m, other elsewhere.
inline FVec blend(Lanes m, FVec other, FVec v) {
  return _mm256_blendv_ps(other, v, _mm256_castsi256_ps(mask_vec(m)));
}

inline float reduce_max(FVec v) {
  __m256 r = v;
  r = _mm256_max_ps(r, _mm256_permute2f128_ps(r, r, 1));
  r = _mm256_max_ps(r, _mm256_shuffle_ps(r, r, 0x4E));
  r = _mm256_max_ps(r, _mm256_shuffle_ps(r, r, 0xB1));
  return _mm256_cvtss_f32(r);
}

// For each lane mask, the lanes it selects, first (AVX2 has no compress).
struct CompressTable {
  alignas(32) int32_t idx[256][8];
  CompressTable() {
    for (int m = 0; m < 256; ++m) {
      int k = 0;
      for (int lane = 0; lane < 8; ++lane) {
        if (m >> lane & 1) idx[m][k++] = lane;
      }
      while (k < 8) idx[m][k++] = 0;
    }
  }
};
const CompressTable kCompress;

inline __m256i compress_perm(Lanes m) {
  return _mm256_load_si256(reinterpret_cast<const __m256i*>(kCompress.idx[m]));
}

// The lanes of v in m, stored contiguously at dst (a whole vector is written).
inline void compress_store(float* dst, Lanes m, FVec v) {
  _mm256_storeu_ps(dst, _mm256_permutevar8x32_ps(v, compress_perm(m)));
}

inline void compress_store(int32_t* dst, Lanes m, IVec v) {
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst),
                      _mm256_permutevar8x32_epi32(v, compress_perm(m)));
}

// NaN lanes of x as -inf, NaN being never kept.
inline FVec drop_nan(FVec x) {
  const __m256 r = x;
  return _mm256_blendv_ps(r, _mm256_set1_ps(-kInf),
                          _mm256_cmp_ps(r, r, _CMP_UNORD_Q));
}

// acc plus v in the lanes of m (4 lanes).
inline DVec add_lanes(DVec acc, DVec v, Lanes m) {
  const __m256i bit = _mm256_setr_epi64x(1, 2, 4, 8);
  const __m256i mask = _mm256_cmpeq_epi64(
      _mm256_and_si256(_mm256_set1_epi64x(static_cast<int64_t>(m)), bit), bit);
  return _mm256_add_pd(acc, _mm256_and_pd(v, _mm256_castsi256_pd(mask)));
}

// The low and high 4 lanes of v, in FP64.
inline std::pair<DVec, DVec> widen(FVec v) {
  const __m256 r = v;
  return {DVec(_mm256_cvtps_pd(_mm256_castps256_ps128(r))),
          DVec(_mm256_cvtps_pd(_mm256_extractf128_ps(r, 1)))};
}

// The maxima of 8 vectors, in some lane order: 7 shuffle steps instead of 8
// horizontal reductions.
inline __m256 max8_unordered(const __m256* v) {
  __m256 u[4], w[2];
  for (int i = 0; i < 4; ++i) {  // 128-bit halves
    u[i] = _mm256_max_ps(_mm256_permute2f128_ps(v[i], v[i + 4], 0x20),
                         _mm256_permute2f128_ps(v[i], v[i + 4], 0x31));
  }
  for (int i = 0; i < 2; ++i) {  // lanes 0|2, 1|3 within each half
    w[i] = _mm256_max_ps(_mm256_shuffle_ps(u[i], u[i + 2], 0x44),
                         _mm256_shuffle_ps(u[i], u[i + 2], 0xEE));
  }
  return _mm256_max_ps(_mm256_shuffle_ps(w[0], w[1], 0x88),
                       _mm256_shuffle_ps(w[0], w[1], 0xDD));
}

// The lane order of max8_unordered, found by running it on vector indices.
const __m256i kMax8Order = [] {
  __m256 v[8];
  for (int j = 0; j < 8; ++j) v[j] = _mm256_set1_ps(static_cast<float>(j));
  alignas(32) float lane_of[8];
  _mm256_store_ps(lane_of, max8_unordered(v));
  alignas(32) int32_t idx[8];
  for (int lane = 0; lane < 8; ++lane)
    idx[static_cast<int>(lane_of[lane])] = lane;
  return _mm256_load_si256(reinterpret_cast<const __m256i*>(idx));
}();

// The 16 group maxima of a supergroup's 32 vectors (a group is two): lane j of
// out[h] holds the maximum of group 8h + j.
inline void group_maxima(const FVec* x, FVec* out) {
  __m256 pair[16];
  for (int g = 0; g < 16; ++g) pair[g] = _mm256_max_ps(x[2 * g], x[2 * g + 1]);
  out[0] = _mm256_permutevar8x32_ps(max8_unordered(pair), kMax8Order);
  out[1] = _mm256_permutevar8x32_ps(max8_unordered(pair + 8), kMax8Order);
}

  #endif  // AVX-512 / AVX2

// ---------------------------------------------------------------- vectors

// Indices base..base+15. IVec::arange builds a vector from 16 scalars, too slow
// for every vector of a pass.
inline IVec lane_index(int base) {
  static const IVec iota = IVec::arange(0, 1);
  return IVec(base) + iota;
}

// How many lanes of a vector at pos lie before n.
inline int span(int pos, int n) { return std::clamp(n - pos, 0, kVec); }

// count logits at p; NaN, never kept, and lanes past count load as -inf.
inline FVec load_logits(const float* p, int count) {
  return drop_nan(load_filled(p, count, -kInf));
}

inline Lanes before_v(FVec x, IVec i, float px, int32_t pi) {
  const FVec p(px);
  return gt(x, p) | (eq(x, p) & lt(i, IVec(pi)));
}

inline float reduce_sum(FVec v) {
  return at::vec::vec_reduce_all<float>(std::plus<FVec>(), v);
}

// FP64 sums of FP32 weights, per logit (exception to block FP32 sums: on CPU an
// FP32 block sum adds lane-wise, and the lane of the row max drops the tail).
struct Acc64 {
  DVec lo = DVec(0.0);
  DVec hi = DVec(0.0);
  inline void add(FVec w, Lanes m) {
    const auto [wl, wh] = widen(w);
    lo = add_lanes(lo, wl, m);
    hi = add_lanes(hi, wh, m >> DVec::size());
  }
  inline double sum() const {
    return at::vec::vec_reduce_all<double>(std::plus<DVec>(), lo + hi);
  }
};

// ---------------------------------------------------------------- workspace

struct Buf {
  std::vector<float> x, w;
  std::vector<int32_t> i;
  void reserve(int n) {
    if (static_cast<int>(x.size()) < n) {
      x.resize(n);
      w.resize(n);
      i.resize(n);
    }
  }
};

struct Scratch {
  Buf cand, a, b;
  std::vector<float> row_w;  // the row's weights, cached by a full-row pass
  std::vector<std::pair<int64_t, float>> sorted, sort_tmp;
  void reserve(int vocab) {
    cand.reserve(kCand + kSlack);
    a.reserve(kCand + kSlack);
    b.reserve(kCand + kSlack);
    if (static_cast<int>(row_w.size()) < vocab + kSlack) {
      row_w.resize(vocab + kSlack);
    }
  }
};

// Items to select from: the row itself (indices implicit, weights computed or
// cached) or a buffer of (logit, index, weight).
struct Source {
  const float* x;
  const int32_t* idx;  // nullptr: the row
  const float* w;      // nullptr: computed
  int n;
  const Buf* buf;  // the buffer holding the items, if any
  int base = 0;    // index of x[0], for a chunk of the row
};

// Items before (ux, ui) are above the bracket, items before (lx, li) are in it.
struct Bracket {
  float ux = kInf;
  int32_t ui = -1;
  float lx = -kInf;
  int32_t li = kLastIndex;
};

inline IVec load_index(const Source& s, int pos, int count) {
  if (s.idx == nullptr) return lane_index(s.base + pos);
  return IVec::loadu(s.idx + pos, count);
}

inline FVec load_weights(const Source& s, int pos, int count, FVec x,
                         FVec row_max) {
  if (s.w != nullptr) return FVec::loadu(s.w + pos, count);
  return (x - row_max).exp();
}

inline Lanes inside_v(FVec x, IVec i, const Bracket& b) {
  return ~before_v(x, i, b.ux, b.ui) & before_v(x, i, b.lx, b.li);
}

// ---------------------------------------------------------------- passes

struct Pivots {
  float x[kPivots];
  int32_t i[kPivots];
  int count = 0;
};

// Per pivot, the bracket's items above its logit (or, exact, before it): their
// count and approximate mass, which only pick the next bracket. A full-row pass
// with weights not yet cached caches them. The pivot loop is unrolled at a
// fixed count, so its 16 accumulators stay in registers; missing pivots match
// nothing.
template <bool kMass, bool kExact>
void pivot_pass_t(const Source& s, const Bracket& b, const Pivots& pv,
                  float row_max, float* cache, int64_t* count, double* msum) {
  float px[kPivots];
  int32_t pi[kPivots];
  for (int q = 0; q < kPivots; ++q) {
    px[q] = q < pv.count ? pv.x[q] : kInf;
    pi[q] = q < pv.count ? pv.i[q] : -1;
  }
  const FVec m(row_max);
  FVec acc[kPivots];
  int64_t cnt[kPivots] = {};
  double ms[kPivots] = {};
  #pragma GCC unroll 16
  for (int q = 0; q < kPivots; ++q) acc[q] = FVec(0.0f);
  int flush = 0;
  for (int pos = 0; pos < s.n; pos += kVec) {
    const int c = span(pos, s.n);
    const FVec x = load_logits(s.x + pos, c);
    const IVec i = load_index(s, pos, c);
    const Lanes in = tail(pos, s.n) & inside_v(x, i, b);
    FVec w(0.0f);
    if constexpr (kMass) {
      w = load_weights(s, pos, c, x, m);
      if (cache != nullptr) w.store(cache + pos);
    }
  #pragma GCC unroll 16
    for (int q = 0; q < kPivots; ++q) {
      Lanes hit;
      if constexpr (kExact) {
        hit = in & before_v(x, i, px[q], pi[q]);
      } else {
        hit = in & gt(x, FVec(px[q]));
      }
      cnt[q] += __builtin_popcount(hit);
      if constexpr (kMass) acc[q] = add_lanes(acc[q], w, hit);
    }
    if constexpr (kMass) {
      if (++flush == 16) {  // FP32 lane sums of 16 vectors, then FP64
        flush = 0;
  #pragma GCC unroll 16
        for (int q = 0; q < kPivots; ++q) {
          ms[q] += reduce_sum(acc[q]);
          acc[q] = FVec(0.0f);
        }
      }
    }
  }
  for (int q = 0; q < kPivots; ++q) {
    if constexpr (kMass) ms[q] += reduce_sum(acc[q]);
    count[q] = cnt[q];
    msum[q] = ms[q];
  }
}

void pivot_pass(const Source& s, const Bracket& b, const Pivots& pv,
                float row_max, bool mass, bool exact, float* cache,
                int64_t* count, double* msum) {
  if (mass) {
    exact ? pivot_pass_t<true, true>(s, b, pv, row_max, cache, count, msum)
          : pivot_pass_t<true, false>(s, b, pv, row_max, cache, count, msum);
  } else {
    exact ? pivot_pass_t<false, true>(s, b, pv, row_max, cache, count, msum)
          : pivot_pass_t<false, false>(s, b, pv, row_max, cache, count, msum);
  }
}

struct Split {
  int64_t above = 0;
  double m_above = 0.0;
  int64_t inside = 0;
  double m_inside = 0.0;
};

// Split the bracket's items at the new bracket: count and FP64 mass of those
// above it, and those inside it copied to dst in source order (as many as fit),
// with their count and FP64 mass.
Split compact(const Source& s, const Bracket& b, const Bracket& nb,
              float row_max, bool mass, float* dst_x, int32_t* dst_i,
              float* dst_w, int cap) {
  const FVec m(row_max);
  Acc64 up_acc, in_acc;
  Split r;
  int taken = 0;
  for (int pos = 0; pos < s.n; pos += kVec) {
    const int c = span(pos, s.n);
    const FVec x = load_logits(s.x + pos, c);
    const IVec i = load_index(s, pos, c);
    const Lanes in = tail(pos, s.n) & inside_v(x, i, b);
    if (in == 0) continue;
    const Lanes up = in & before_v(x, i, nb.ux, nb.ui);
    const Lanes keep = in & ~up & before_v(x, i, nb.lx, nb.li);
    r.above += __builtin_popcount(up);
    FVec w(0.0f);
    if (mass) {
      w = load_weights(s, pos, c, x, m);
      up_acc.add(w, up);
      in_acc.add(w, keep);
    }
    if (keep != 0) {
      if (taken <= cap) {
        compress_store(dst_x + taken, keep, x);
        compress_store(dst_i + taken, keep, i);
        if (mass) compress_store(dst_w + taken, keep, w);
      }
      taken += __builtin_popcount(keep);
    }
  }
  r.inside = taken;
  if (mass) {
    r.m_above = up_acc.sum();
    r.m_inside = in_acc.sum();
  }
  return r;
}

Split compact(const Source& s, const Bracket& b, const Bracket& nb,
              float row_max, bool mass, Buf* dst, int cap) {
  return compact(s, b, nb, row_max, mass, dst->x.data(), dst->i.data(),
                 dst->w.data(), cap);
}

// Copy the logits at or above t, in index order, with (mass) their weights:
// how many (-1 past kCand), and their FP64 mass. A group of 16 logits is one
// vector, so the group maxima pick the vectors to load; the weights are
// computed afterwards over the compacted logits, not over every vector with a
// hit.
std::pair<int, double> gather(const float* row, int vocab, const float* gmax,
                              int groups, float t, float row_max, bool mass,
                              Buf* dst) {
  const FVec tv(t);
  int taken = 0;
  for (int g0 = 0; g0 < groups; g0 += kVec) {
    const FVec gm = load_logits(gmax + g0, span(g0, groups));
    // Padding groups past the row load as -inf, which t = -inf would select.
    Lanes hit = tail(g0, groups) & ge(gm, tv);
    while (hit != 0) {
      const int group = (g0 + __builtin_ctz(hit)) * kGroup;
      hit &= hit - 1;
      for (int pos = group; pos < group + kGroup; pos += kVec) {
        const FVec x = load_logits(row + pos, span(pos, vocab));
        const Lanes keep = tail(pos, vocab) & ge(x, tv);
        const int c = __builtin_popcount(keep);
        if (taken + c > kCand) return {-1, 0.0};
        compress_store(dst->x.data() + taken, keep, x);
        compress_store(dst->i.data() + taken, keep, lane_index(pos));
        taken += c;
      }
    }
  }
  if (!mass) return {taken, 0.0};
  const FVec m(row_max);
  Acc64 acc;
  for (int j = 0; j < taken; j += kVec) {
    const FVec w = (FVec::loadu(dst->x.data() + j, span(j, taken)) - m).exp();
    w.store(dst->w.data() + j);
    acc.add(w, tail(j, taken));
  }
  return {taken, acc.sum()};
}

// A logit t with need level maxima at or above it, or (mass) level maxima whose
// weights reach target: as many logits, as much mass, lie at or above t. -inf
// if no sampled pivot qualifies.
float level_bound(const float* lv, int n, int64_t need, double target,
                  float row_max, bool mass) {
  float sample[kBoundSample];
  for (int s = 0; s < kBoundSample; ++s) {
    const float v = lv[static_cast<int64_t>(s) * n / kBoundSample];
    sample[s] = v == v ? v : -kInf;
  }
  std::sort(sample, sample + kBoundSample, std::greater<float>());
  float pv[kPivots];
  for (int r = 0; r < kPivots;
       ++r) {  // ranks spaced quadratically from the best
    pv[r] = sample[std::max(
        0, (r + 1) * (r + 1) * kBoundSample / (kPivots * kPivots) - 1)];
  }
  const FVec m(row_max);
  int64_t count[kPivots] = {};
  double msum[kPivots] = {};
  FVec acc[kPivots];
  for (int q = 0; q < kPivots; ++q) acc[q] = FVec(0.0f);
  int flush = 0;
  for (int pos = 0; pos < n; pos += kVec) {
    const Lanes tm = tail(pos, n);
    const FVec x = load_logits(lv + pos, span(pos, n));
    const FVec w = mass ? (x - m).exp() : FVec(0.0f);
    for (int q = 0; q < kPivots; ++q) {
      const Lanes hit = tm & ge(x, FVec(pv[q]));
      count[q] += __builtin_popcount(hit);
      if (mass) acc[q] = add_lanes(acc[q], w, hit);
    }
    if (mass && ++flush == 16) {
      flush = 0;
      for (int q = 0; q < kPivots; ++q) {
        msum[q] += reduce_sum(acc[q]);
        acc[q] = FVec(0.0f);
      }
    }
  }
  for (int q = 0; q < kPivots; ++q) {
    if (mass) msum[q] += reduce_sum(acc[q]);
    // The margin covers the FP32 sums; the gather checks the mass in FP64.
    if (mass ? msum[q] >= target * (1.0 + 1e-4) : count[q] >= need) {
      return pv[q];
    }
  }
  return -kInf;
}

// Sort (key, weight) items by key: LSD radix over the key bytes, skipping the
// bytes all keys share (a few hundred keys sort in microseconds, not tens).
void sort_items(std::vector<std::pair<int64_t, float>>& items,
                std::vector<std::pair<int64_t, float>>& tmp) {
  const int n = static_cast<int>(items.size());
  if (n < 2) return;
  tmp.resize(n);
  auto* src = items.data();
  auto* dst = tmp.data();
  auto byte_of = [](int64_t key, int shift) {
    return static_cast<uint32_t>(
        ((static_cast<uint64_t>(key) ^ 0x8000000000000000ULL) >> shift) & 0xFF);
  };
  for (int shift = 0; shift < 64; shift += 8) {
    uint32_t start[256] = {};
    for (int j = 0; j < n; ++j) ++start[byte_of(src[j].first, shift)];
    if (start[byte_of(src[0].first, shift)] == static_cast<uint32_t>(n))
      continue;
    uint32_t sum = 0;
    for (auto& c : start) {
      const uint32_t c0 = c;
      c = sum;
      sum += c0;
    }
    for (int j = 0; j < n; ++j)
      dst[start[byte_of(src[j].first, shift)]++] = src[j];
    std::swap(src, dst);
  }
  if (src != items.data()) std::copy(src, src + n, items.data());
}

// A pseudo-random position in [0, n), per sample slot and round: fixed
// strided positions can all miss the bracket, round after round.
inline int sample_pos(int t, int round, int n) {
  uint32_t h = static_cast<uint32_t>(t) * 0x9E3779B1u +
               static_cast<uint32_t>(round) * 0x85EBCA77u + 0x632BE5ABu;
  h ^= h >> 15;
  h *= 0x2C1B3C6Du;
  h ^= h >> 12;
  return static_cast<int>((static_cast<uint64_t>(h) * n) >> 32);
}

// P pivots of the bracket at evenly spaced ranks of a sample of it: random
// positions, or (none of them in the bracket) the bracket's first items.
Pivots sample_pivots(const Source& s, const Bracket& b, int round,
                     Scratch& sc) {
  auto& items = sc.sorted;
  items.clear();
  auto take = [&](int pos) {
    float x = s.x[pos];
    x = x == x ? x : -kInf;
    const int32_t i = s.idx == nullptr ? s.base + pos : s.idx[pos];
    if (!before(x, i, b.ux, b.ui) && before(x, i, b.lx, b.li)) {
      items.push_back({make_key(x, i), 0.0f});
    }
  };
  for (int t = 0; t < kSample; ++t) take(sample_pos(t, round, s.n));
  if (items.empty()) {
    // A uniform reservoir sample of the bracket in one pass: its first items
    // alone would trim only a few hundred per round.
    int64_t seen = 0;
    for (int pos = 0; pos < s.n; ++pos) {
      float x = s.x[pos];
      x = x == x ? x : -kInf;
      const int32_t i = s.idx == nullptr ? s.base + pos : s.idx[pos];
      if (before(x, i, b.ux, b.ui) || !before(x, i, b.lx, b.li)) continue;
      const std::pair<int64_t, float> item{make_key(x, i), 0.0f};
      if (seen < kSample) {
        items.push_back(item);
      } else {
        const int64_t r = sample_pos(static_cast<int>(seen), round,
                                     static_cast<int>(seen + 1));
        if (r < kSample) items[r] = item;
      }
      ++seen;
    }
  }
  sort_items(items, sc.sort_tmp);
  const int unique =
      static_cast<int>(std::unique(items.begin(), items.end(),
                                   [](const auto& l, const auto& r) {
                                     return l.first == r.first;
                                   }) -
                       items.begin());
  Pivots pv;
  for (int r = 0; r < kPivots && unique > 0; ++r) {
    const int at = (r + 1) * unique / (kPivots + 1);
    if (pv.count > 0 &&
        make_key(pv.x[pv.count - 1], pv.i[pv.count - 1]) == items[at].first) {
      continue;
    }
    pv.x[pv.count] = key_value(items[at].first);
    pv.i[pv.count] = key_index(items[at].first);
    ++pv.count;
  }
  return pv;
}

// A bracket of the cut estimated from a sorted strided sample of the row's
// items in b (Floyd-Rivest): about 3 standard errors of the sample around the
// estimate, so one compaction pass over the row usually narrows it. False if
// the sample cannot place it.
bool sampled_bracket(const Source& s, const Bracket& b, int64_t need,
                     double target, int64_t a_count, double a_mass,
                     int64_t size, double set_mass, bool mass, float row_max,
                     Scratch& sc, Bracket* nb) {
  auto& items = sc.sorted;
  items.clear();
  for (int t = 0; t < kRowSample; ++t) {
    const int pos = sample_pos(t, kMaxRounds, s.n);
    float x = s.x[pos];
    x = x == x ? x : -kInf;
    if (!before(x, pos, b.ux, b.ui) && before(x, pos, b.lx, b.li)) {
      items.push_back({make_key(x, pos), 0.0f});
    }
  }
  const int n = static_cast<int>(items.size());
  if (n < 64) return false;
  sort_items(items, sc.sort_tmp);
  // Sample weights: 1 per item, or its exp weight (vector exp, as the passes).
  double total = n, total_sq = n;
  if (mass) {
    alignas(64) float x[kRowSample + kSlack] = {};
    for (int j = 0; j < n; ++j) x[j] = key_value(items[j].first);
    total = total_sq = 0.0;
    const FVec m(row_max);
    for (int j = 0; j < n; j += kVec) {
      (FVec::loadu(x + j) - m).exp().store(x + j);
    }
    for (int j = 0; j < n; ++j) {
      items[j].second = x[j];
      total += x[j];
      total_sq += static_cast<double>(x[j]) * x[j];
    }
    if (!(total > 0.0)) return false;
  } else {
    for (auto& item : items) item.second = 1.0f;
  }
  const double f = mass ? (target - a_mass) / set_mass
                        : static_cast<double>(need - a_count) / size;
  const double n_eff = total * total / total_sq;
  const double delta =
      3.0 * std::sqrt(std::max(f * (1.0 - f), 1.0 / n_eff) / n_eff) +
      2.0 / n_eff;
  const double lo = (f - delta) * total, hi = (f + delta) * total;
  // U: the last sampled item whose sample mass through it stays below lo;
  // L: the first whose sample mass before it reaches hi.
  double run = 0.0;
  *nb = b;
  for (int j = 0; j < n; ++j) {
    const double next = run + items[j].second;
    if (next <= lo) {
      nb->ux = key_value(items[j].first);
      nb->ui = key_index(items[j].first);
    }
    if (run >= hi) {
      nb->lx = key_value(items[j].first);
      nb->li = key_index(items[j].first);
      break;
    }
    run = next;
  }
  return true;
}

// The key of the last kept of n <= kSmall buffered items, after a_count items
// of a_mass: the need-th, or (mass) the first whose mass from the top reaches
// target. Sorted on int64 keys, as topk_topp_triton sorts its window.
int64_t finish(const Buf& buf, int n, int64_t a_count, double a_mass,
               int64_t need, double target, float row_max, bool mass,
               Scratch& sc) {
  if (n == 0) {  // unreachable: the bracket always holds the cut
    narrow_failed().store(true, std::memory_order_relaxed);
    return 0;
  }
  auto& items = sc.sorted;
  items.resize(n);
  float w[kSmall + kSlack] = {};
  if (mass) {  // the passes' vector exp, so the finish weighs items alike
    const FVec m(row_max);
    for (int j = 0; j < n; j += kVec) {
      (FVec::loadu(buf.x.data() + j, span(j, n)) - m).exp().store(w + j);
    }
  }
  for (int j = 0; j < n; ++j) {
    items[j] = {make_key(buf.x[j], buf.i[j]), w[j]};
  }
  sort_items(items, sc.sort_tmp);
  if (!mass) return items[need - a_count - 1].first;
  double before_mass = a_mass;
  int64_t last = items[0].first;
  for (int j = 0; j < n && before_mass < target; ++j) {
    last = items[j].first;
    before_mass += static_cast<double>(items[j].second);
  }
  return last;
}

// The key of the last kept item of the size items of the source in the
// bracket: the need-th, or (mass) the first whose mass from the top reaches
// target, the items holding set_mass, after a_count items of a_mass above the
// bracket. Each round brackets the cut between two
// sampled pivots and compacts the bracket into the other work buffer, so a
// misjudged bracket loses nothing.
int64_t narrow(Source cur, int64_t size, Bracket b, int64_t need, double target,
               double set_mass, bool mass, float row_max, Scratch& sc,
               int64_t a_count = 0, double a_mass = 0.0) {
  const double margin = 1e-5;  // of the bracket's mass, the FP32 sums' error
  bool exact = false;
  bool row_cached = false;
  bool sampled = false;
  bool stale = false;  // cur also holds items outside the bracket
  auto reaches = [&](int64_t count, double m) {
    return mass ? a_mass + m >= target : a_count + count >= need;
  };
  for (int round = 0; round < kMaxRounds; ++round) {
    const bool on_row = cur.idx == nullptr;
    if (!on_row && !stale && size <= kSmall) {
      return finish(*cur.buf, cur.n, a_count, a_mass, need, target, row_max,
                    mass, sc);
    }
    Buf* dst = cur.buf == &sc.a ? &sc.b : &sc.a;
    if ((on_row || stale) && size <= kCand) {  // copy the bracket out
      const Split r = compact(cur, b, b, row_max, mass, dst, kCand);
      cur = {dst->x.data(), dst->i.data(), mass ? dst->w.data() : nullptr,
             static_cast<int>(r.inside), dst};
      stale = false;
      continue;
    }
    Bracket fr;
    if (on_row && !sampled) {  // one pass over the row, if the sample places it
      sampled = true;
      if (sampled_bracket(cur, b, need, target, a_count, a_mass, size, set_mass,
                          mass, row_max, sc, &fr)) {
        const Split r = compact(cur, b, fr, row_max, mass, dst, kCand);
        if (!reaches(r.above, r.m_above) &&
            reaches(r.above + r.inside, r.m_above + r.m_inside) &&
            r.inside <= kCand) {
          a_count += r.above;
          a_mass += r.m_above;
          set_mass = r.m_inside;
          b = fr;
          size = r.inside;
          cur = {dst->x.data(), dst->i.data(), mass ? dst->w.data() : nullptr,
                 static_cast<int>(r.inside), dst};
          continue;
        }
      }
    }
    const Pivots pv = sample_pivots(cur, b, round, sc);
    int64_t count[kPivots];
    double msum[kPivots];
    float* cache = on_row && mass && !row_cached ? sc.row_w.data() : nullptr;
    pivot_pass(cur, b, pv, row_max, mass, exact, cache, count, msum);
    if (cache != nullptr) {
      cur.w = cache;
      row_cached = true;
    }
    const double slack = margin * set_mass;
    Bracket nb = b;
    bool moved = false;
    for (int q = 0; q < pv.count; ++q) {  // pivots are best first, in b
      const bool lower = mass ? a_mass + msum[q] >= target + slack
                              : a_count + count[q] >= need;
      const bool upper =
          mass ? a_mass + msum[q] <= target - slack : a_count + count[q] < need;
      if (upper) {
        nb.ux = pv.x[q];
        nb.ui = pv.i[q];
        moved = true;
      }
      if (lower) {
        nb.lx = pv.x[q];
        nb.li = pv.i[q];
        moved = true;
        break;
      }
    }
    if (!moved && pv.count > 0) {  // the estimates cannot tell: split anyway
      nb.ux = pv.x[pv.count / 2];
      nb.ui = pv.i[pv.count / 2];
    }
    const Split r = compact(cur, b, nb, row_max, mass, dst, kCand);
    // The compaction's exact counts and FP64 masses tell which part of the
    // bracket holds the cut: above the new bracket, inside it, or past it.
    if (reaches(r.above, r.m_above)) {
      b.lx = nb.ux;
      b.li = nb.ui;
      size = r.above;
      set_mass = r.m_above;
      stale = true;
      exact = true;  // ties or the estimates misled this round
    } else if (!reaches(r.above + r.inside, r.m_above + r.m_inside)) {
      a_count += r.above + r.inside;
      a_mass += r.m_above + r.m_inside;
      set_mass = std::max(0.0, set_mass - r.m_above - r.m_inside);
      size -= r.above + r.inside;
      b.ux = nb.lx;
      b.ui = nb.li;
      stale = true;
      exact = true;
    } else {
      a_count += r.above;
      a_mass += r.m_above;
      set_mass = r.m_inside;
      b = nb;
      size = r.inside;
      if (!on_row || r.inside <= kCand) {
        cur = {dst->x.data(), dst->i.data(), mass ? dst->w.data() : nullptr,
               static_cast<int>(r.inside), dst};
        stale = false;
      }
    }
  }
  // Unreachable in practice: every round shrinks the bracket or resamples.
  narrow_failed().store(true, std::memory_order_relaxed);
  return 0;
}

// ---------------------------------------------------------------- phases

struct Shape {
  int rows, vocab, stride, splits, chunk, groups, supergroups;
};

// Row states between the phases.
constexpr uint8_t kIdle = 0;     // left as is
constexpr uint8_t kCut = 1;      // masked at cut
constexpr uint8_t kRowPass = 2;  // top-p row awaiting its split row pass

// A top-p row whose cut is outside its gathered logits: select brackets it from
// a sample, the threads split the pass over the row, and select resumes.
struct RowPass {
  Bracket bracket;
  double target, row_sum;
  float row_max;
};

struct Work {
  std::vector<float> gmax, smax, split_max;
  std::vector<double> split_sum;
  std::vector<int64_t> split_finite, cut;
  std::vector<uint8_t> active;
  std::vector<RowPass> row_pass;
  std::atomic<bool> any_row_pass{false};
  std::vector<Split> segment;  // per split of a row pass
  std::vector<float> seg_x, seg_w;
  std::vector<int32_t> seg_i;
  std::vector<Scratch> scratch;
};

constexpr int64_t kSegment = kCand + kSlack;

void levels(const Shape& sh, float* logits, const int32_t* k, const float* p,
            Work& wk, int row, int split) {
  const int kv = k != nullptr ? k[row] : sh.vocab;
  const float pv = p != nullptr ? p[row] : 1.0f;
  const int slot = row * kMaxSplits + split;
  wk.split_max[slot] = -kInf;
  wk.split_sum[slot] = 0.0;
  wk.split_finite[slot] = 0;
  if (kv >= sh.vocab && pv >= 1.0f) return;
  const float* x_row = logits + static_cast<int64_t>(row) * sh.stride;
  float* gmax = wk.gmax.data() + static_cast<int64_t>(row) * sh.groups;
  float* smax = wk.smax.data() + static_cast<int64_t>(row) * sh.supergroups;
  const int lo = split * sh.chunk;
  const int hi = std::min(lo + sh.chunk, sh.vocab);
  float row_max = -kInf;
  int64_t finite = 0;
  FVec x[kSuper / kVec];
  for (int start = lo; start < hi; start += kSuper) {
    for (int v = 0; v < kSuper / kVec; ++v) {
      const int pos = start + v * kVec;
      x[v] = load_logits(x_row + pos, span(pos, hi));
      finite += __builtin_popcount(gt(x[v], FVec(-kInf)));
    }
    // A supergroup's 16 group maxima, stored up to the row end.
    FVec g[kSuper / kGroup / kVec];
    group_maxima(x, g);
    float block_max = -kInf;
    for (int j = 0; j < kSuper / kGroup / kVec; ++j) {
      const int g0 = start / kGroup + j * kVec;
      g[j].store(gmax + g0, span(g0, sh.groups));
      block_max = std::max(block_max, reduce_max(g[j]));
    }
    smax[start / kSuper] = block_max;
    row_max = std::max(row_max, block_max);
  }
  wk.split_max[slot] = row_max;
  wk.split_finite[slot] = finite;
}

// As topk_topp_triton: only top-p rows without top-k need the row's mass.
inline bool needs_sum(const Shape& sh, const int32_t* k, const float* p,
                      int row) {
  return p != nullptr && p[row] < 1.0f && (k == nullptr || k[row] >= sh.vocab);
}

// One split's share of the row's mass, with the weights the narrowing sums:
// FP32 exp(x - row max), added in FP64. A mass from running maxima differs by
// up to ~5e-7 relative, which leaves a target for p near 1 out of reach.
void row_sums(const Shape& sh, const float* logits, const int32_t* k,
              const float* p, Work& wk, int row, int split) {
  if (!needs_sum(sh, k, p, row)) return;
  float row_max = -kInf;
  for (int s = 0; s < sh.splits; ++s) {
    row_max = std::max(row_max, wk.split_max[row * kMaxSplits + s]);
  }
  double total = 0.0;
  if (row_max > -kInf && row_max < kInf) {
    const float* x_row = logits + static_cast<int64_t>(row) * sh.stride;
    const int lo = split * sh.chunk;
    const int hi = std::min(lo + sh.chunk, sh.vocab);
    const FVec m(row_max);
    Acc64 acc;
    for (int pos = lo; pos < hi; pos += kVec) {
      acc.add((load_logits(x_row + pos, span(pos, hi)) - m).exp(),
              tail(pos, hi));
    }
    total = acc.sum();
  }
  wk.split_sum[row * kMaxSplits + split] = total;
}

// The cut of top-p over a row whose max is +inf, after top-k of k: the +inf
// logits share all the mass equally (the softmax limit), so of the n of them in
// the top-k set the first c in index order are kept, c / n >= p.
int64_t posinf_cut(const float* row, int vocab, int64_t k, float p) {
  const FVec inf(kInf);
  int64_t n_inf = 0;
  for (int pos = 0; pos < vocab; pos += kVec) {
    const FVec x = FVec::loadu(row + pos, span(pos, vocab));
    n_inf += __builtin_popcount(tail(pos, vocab) & eq(x, inf));
  }
  const int64_t n = std::min(k, n_inf);
  int64_t c = static_cast<int64_t>(static_cast<double>(p) * n);
  if (static_cast<double>(c) / n < p) ++c;
  c = std::clamp<int64_t>(c, 1, n);
  for (int pos = 0; pos < vocab; pos += kVec) {
    const FVec x = FVec::loadu(row + pos, span(pos, vocab));
    Lanes hit = tail(pos, vocab) & eq(x, inf);
    const int h = __builtin_popcount(hit);
    if (c > h) {
      c -= h;
      continue;
    }
    while (--c > 0) hit &= hit - 1;
    return make_key(kInf, pos + __builtin_ctz(hit));
  }
  return make_key(kInf, vocab - 1);  // unreachable: c <= n_inf
}

void select_row(const Shape& sh, const float* logits, const int32_t* k,
                const float* p, Work& wk, Scratch& sc, int row) {
  const int64_t kv = k != nullptr ? k[row] : sh.vocab;
  const float pv = p != nullptr ? p[row] : 1.0f;
  wk.active[row] = kIdle;
  if (kv >= sh.vocab && pv >= 1.0f) return;
  float row_max = -kInf;
  int64_t finite = 0;
  for (int s = 0; s < sh.splits; ++s) {
    row_max = std::max(row_max, wk.split_max[row * kMaxSplits + s]);
    finite += wk.split_finite[row * kMaxSplits + s];
  }
  double row_sum = 0.0;
  if (needs_sum(sh, k, p, row)) {
    for (int s = 0; s < sh.splits; ++s) {
      row_sum += wk.split_sum[row * kMaxSplits + s];
    }
  }
  const bool use_k = kv < finite;
  const bool use_p = pv < 1.0f;
  if (finite == 0 || !(use_k || use_p)) return;
  sc.reserve(sh.vocab);
  const float* x_row = logits + static_cast<int64_t>(row) * sh.stride;
  if (use_p && row_max == kInf) {
    // The softmax limit: the n +inf logits of the top-k set share all the mass,
    // so top-p keeps the first c of them with c / n >= p.
    wk.cut[row] = posinf_cut(x_row, sh.vocab, use_k ? kv : sh.vocab, pv);
    wk.active[row] = kCut;
    return;
  }
  const float* gmax = wk.gmax.data() + static_cast<int64_t>(row) * sh.groups;
  const float* smax =
      wk.smax.data() + static_cast<int64_t>(row) * sh.supergroups;
  const Source on_row{x_row, nullptr, nullptr, sh.vocab, nullptr};
  Bracket kept;  // the top-k set, as a bracket
  int64_t cut = 0;
  int gathered = -1;
  if (use_k) {
    // Few enough: the 256-logit maxima bound them, a 16th of the work.
    const float t =
        kv * 4 <= sh.supergroups
            ? level_bound(smax, sh.supergroups, kv, 0.0, row_max, false)
            : level_bound(gmax, sh.groups, kv, 0.0, row_max, false);
    gathered =
        gather(x_row, sh.vocab, gmax, sh.groups, t, row_max, false, &sc.cand)
            .first;
    if (gathered >= 0) {
      const Source cand{sc.cand.x.data(), sc.cand.i.data(), nullptr, gathered,
                        &sc.cand};
      cut = narrow(cand, gathered, Bracket{}, kv, 0.0, 0.0, false, row_max, sc);
    } else {
      cut =
          narrow(on_row, sh.vocab, Bracket{}, kv, 0.0, 0.0, false, row_max, sc);
    }
    kept.lx = key_value(cut);
    kept.li = key_index(cut) + 1;
  } else if (kv < sh.vocab) {  // k reaches every finite logit
    kept.lx = -std::numeric_limits<float>::max();
    kept.li = sh.vocab;
  }
  if (use_p) {
    if (use_k || kv < sh.vocab) {
      // Top-p of the kept set: its mass from the gathered logits or the row.
      const Source src = gathered >= 0
                             ? Source{sc.cand.x.data(), sc.cand.i.data(),
                                      nullptr, gathered, &sc.cand}
                             : on_row;
      const Split r = compact(src, kept, kept, row_max, true, &sc.a, kCand);
      const double target = static_cast<double>(pv) * r.m_inside;
      if (r.inside <= kCand) {
        const Source set{sc.a.x.data(), sc.a.i.data(), sc.a.w.data(),
                         static_cast<int>(r.inside), &sc.a};
        cut = narrow(set, r.inside, Bracket{}, 0, target, r.m_inside, true,
                     row_max, sc);
      } else {
        cut = narrow(on_row, r.inside, kept, 0, target, r.m_inside, true,
                     row_max, sc);
      }
    } else {
      const double target = static_cast<double>(pv) * row_sum;
      // The 256-logit maxima first: a 16th of the work when they suffice.
      float t = level_bound(smax, sh.supergroups, 0, target, row_max, true);
      if (t == -kInf)
        t = level_bound(gmax, sh.groups, 0, target, row_max, true);
      const auto [n, m] =
          gather(x_row, sh.vocab, gmax, sh.groups, t, row_max, true, &sc.cand);
      // The logits at or above t reach target only if the cut is among them;
      // else (rounding, or a flat row) the whole row is narrowed.
      Bracket fr;
      if (n >= 0 && m >= target) {
        const Source cand{sc.cand.x.data(), sc.cand.i.data(), sc.cand.w.data(),
                          n, &sc.cand};
        cut = narrow(cand, n, Bracket{}, 0, target, m, true, row_max, sc);
      } else if (sh.splits > 1 &&
                 sampled_bracket(on_row, Bracket{}, 0, target, 0, 0.0, sh.vocab,
                                 row_sum, true, row_max, sc, &fr)) {
        // Fewer rows than threads: the threads split the pass over the row.
        wk.row_pass[row] = {fr, target, row_sum, row_max};
        wk.active[row] = kRowPass;
        wk.any_row_pass.store(true, std::memory_order_relaxed);
        return;
      } else {
        cut = narrow(on_row, sh.vocab, Bracket{}, 0, target, row_sum, true,
                     row_max, sc);
      }
    }
  }
  wk.cut[row] = cut;
  wk.active[row] = kCut;
}

// One split's share of a row pass: its logits above the sampled bracket
// counted and weighed, those inside it copied to the split's segment.
void row_pass(const Shape& sh, const float* logits, Work& wk, int row,
              int split) {
  if (wk.active[row] != kRowPass) return;
  const RowPass& rp = wk.row_pass[row];
  const int lo = split * sh.chunk;
  const int hi = std::min(lo + sh.chunk, sh.vocab);
  const Source chunk{logits + static_cast<int64_t>(row) * sh.stride + lo,
                     nullptr,
                     nullptr,
                     hi - lo,
                     nullptr,
                     lo};
  const int64_t seg = static_cast<int64_t>(row) * kMaxSplits + split;
  wk.segment[seg] = compact(chunk, Bracket{}, rp.bracket, rp.row_max, true,
                            wk.seg_x.data() + seg * kSegment,
                            wk.seg_i.data() + seg * kSegment,
                            wk.seg_w.data() + seg * kSegment, kCand);
}

// Resume a row after its row pass: join the segments in index order and narrow
// them, or (the sample misjudged the bracket) narrow the row on this thread.
void resume_row(const Shape& sh, const float* logits, Work& wk, Scratch& sc,
                int row) {
  if (wk.active[row] != kRowPass) return;
  sc.reserve(sh.vocab);
  const RowPass& rp = wk.row_pass[row];
  Split r;
  for (int s = 0; s < sh.splits; ++s) {
    const Split& part = wk.segment[static_cast<int64_t>(row) * kMaxSplits + s];
    r.above += part.above;
    r.m_above += part.m_above;
    r.inside += part.inside;
    r.m_inside += part.m_inside;
  }
  const bool ok = r.m_above < rp.target &&
                  rp.target <= r.m_above + r.m_inside && r.inside <= kCand;
  if (ok) {
    int taken = 0;
    for (int s = 0; s < sh.splits; ++s) {
      const int64_t seg = static_cast<int64_t>(row) * kMaxSplits + s;
      const int c = static_cast<int>(wk.segment[seg].inside);
      std::memcpy(sc.a.x.data() + taken, wk.seg_x.data() + seg * kSegment,
                  c * sizeof(float));
      std::memcpy(sc.a.i.data() + taken, wk.seg_i.data() + seg * kSegment,
                  c * sizeof(int32_t));
      std::memcpy(sc.a.w.data() + taken, wk.seg_w.data() + seg * kSegment,
                  c * sizeof(float));
      taken += c;
    }
    const Source set{sc.a.x.data(), sc.a.i.data(), sc.a.w.data(), taken, &sc.a};
    wk.cut[row] = narrow(set, taken, Bracket{}, 0, rp.target, r.m_inside, true,
                         rp.row_max, sc, r.above, r.m_above);
  } else {
    const Source on_row{logits + static_cast<int64_t>(row) * sh.stride, nullptr,
                        nullptr, sh.vocab, nullptr};
    wk.cut[row] = narrow(on_row, sh.vocab, Bracket{}, 0, rp.target, rp.row_sum,
                         true, rp.row_max, sc);
  }
  wk.active[row] = kCut;
}

void fill(const Shape& sh, float* logits, const Work& wk, int row, int split) {
  if (wk.active[row] != kCut) return;
  // Exception to the int64 keys of topk_topp_triton: the cut key, decoded
  // once, orders logits as their keys do (-0.0 == 0.0, NaN never kept).
  const FVec cut(key_value(wk.cut[row]));
  const IVec cut_i(key_index(wk.cut[row]));
  const FVec neg_inf(-kInf);
  float* x_row = logits + static_cast<int64_t>(row) * sh.stride;
  const int lo = split * sh.chunk;
  const int hi = std::min(lo + sh.chunk, sh.vocab);
  for (int pos = lo; pos < hi; pos += kVec) {
    const int c = span(pos, hi);
    const FVec x = FVec::loadu(x_row + pos, c);
    const Lanes keep = gt(x, cut) | (eq(x, cut) & le(lane_index(pos), cut_i));
    blend(keep, neg_inf, x).store(x_row + pos, c);
  }
}

Work& workspace() {
  static Work work;
  return work;
}

}  // namespace

void cpu_topk_topp_exact(torch::Tensor& logits,
                         const std::optional<torch::Tensor>& k,
                         const std::optional<torch::Tensor>& p) {
  TORCH_CHECK(logits.dim() == 2 && logits.is_cpu() &&
                  logits.scalar_type() == torch::kFloat32 &&
                  logits.stride(1) == 1,
              "logits must be a 2D CPU FP32 tensor with unit column stride");
  // Positions are int and keys hold an int32 index.
  TORCH_CHECK(logits.size(1) >= 1 &&
                  logits.size(1) <= std::numeric_limits<int32_t>::max(),
              "vocab size out of range");
  const int rows = static_cast<int>(logits.size(0));
  if (rows == 0 || (!k.has_value() && !p.has_value())) return;
  const int32_t* k_ptr = nullptr;
  const float* p_ptr = nullptr;
  if (k.has_value()) {
    TORCH_CHECK(k->scalar_type() == torch::kInt32 && k->is_contiguous() &&
                k->numel() == rows);
    k_ptr = k->data_ptr<int32_t>();
    TORCH_CHECK(*std::min_element(k_ptr, k_ptr + rows) >= 1, "k must be >= 1");
  }
  if (p.has_value()) {
    TORCH_CHECK(p->scalar_type() == torch::kFloat32 && p->is_contiguous() &&
                p->numel() == rows);
    p_ptr = p->data_ptr<float>();
  }
  Shape sh;
  sh.rows = rows;
  sh.vocab = static_cast<int>(logits.size(1));
  sh.stride = static_cast<int>(logits.stride(0));
  const int threads = omp_get_max_threads();
  int splits = std::clamp((threads + rows - 1) / rows, 1, kMaxSplits);
  const int per = (sh.vocab + splits - 1) / splits;
  sh.chunk = (per + kSuper - 1) / kSuper * kSuper;
  sh.splits = (sh.vocab + sh.chunk - 1) / sh.chunk;
  sh.groups = (sh.vocab + kGroup - 1) / kGroup;
  sh.supergroups = (sh.vocab + kSuper - 1) / kSuper;

  Work& wk = workspace();
  auto grow = [](auto& v, size_t n) {
    if (v.size() < n) v.resize(n);
  };
  grow(wk.gmax, static_cast<size_t>(rows) * sh.groups);
  grow(wk.smax, static_cast<size_t>(rows) * sh.supergroups);
  grow(wk.split_max, static_cast<size_t>(rows) * kMaxSplits);
  grow(wk.split_sum, static_cast<size_t>(rows) * kMaxSplits);
  grow(wk.split_finite, static_cast<size_t>(rows) * kMaxSplits);
  grow(wk.cut, rows);
  grow(wk.active, rows);
  grow(wk.row_pass, rows);
  grow(wk.segment, static_cast<size_t>(rows) * kMaxSplits);
  if (sh.splits > 1) {  // segments of split row passes
    const size_t seg = static_cast<size_t>(rows) * kMaxSplits * kSegment;
    grow(wk.seg_x, seg);
    grow(wk.seg_w, seg);
    grow(wk.seg_i, seg);
  }
  grow(wk.scratch, threads);

  float* x = logits.data_ptr<float>();
  const int items = rows * sh.splits;
  bool any_sum = false;
  for (int row = 0; row < rows && !any_sum; ++row) {
    any_sum = needs_sum(sh, k_ptr, p_ptr, row);
  }
  wk.any_row_pass.store(false, std::memory_order_relaxed);
  narrow_failed().store(false, std::memory_order_relaxed);
  #pragma omp parallel
  {
  #pragma omp for schedule(static)
    for (int it = 0; it < items; ++it) {
      levels(sh, x, k_ptr, p_ptr, wk, it / sh.splits, it % sh.splits);
    }
    if (any_sum) {
  #pragma omp for schedule(static)
      for (int it = 0; it < items; ++it) {
        row_sums(sh, x, k_ptr, p_ptr, wk, it / sh.splits, it % sh.splits);
      }
    }
  #pragma omp for schedule(dynamic, 1)
    for (int row = 0; row < rows; ++row) {
      select_row(sh, x, k_ptr, p_ptr, wk, wk.scratch[omp_get_thread_num()],
                 row);
    }
    // Read after the select loop's barrier: every row has set it by then.
    if (wk.any_row_pass.load(std::memory_order_relaxed)) {
  #pragma omp for schedule(static)
      for (int it = 0; it < items; ++it) {
        row_pass(sh, x, wk, it / sh.splits, it % sh.splits);
      }
  #pragma omp for schedule(dynamic, 1)
      for (int row = 0; row < rows; ++row) {
        resume_row(sh, x, wk, wk.scratch[omp_get_thread_num()], row);
      }
    }
  #pragma omp for schedule(static)
    for (int it = 0; it < items; ++it) {
      fill(sh, x, wk, it / sh.splits, it % sh.splits);
    }
  }
  TORCH_CHECK(!narrow_failed().load(std::memory_order_relaxed),
              "top-k/top-p narrowing did not converge");
}

#endif  // defined(__AVX512F__) || defined(__AVX2__)
