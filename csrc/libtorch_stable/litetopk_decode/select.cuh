// SPDX-License-Identifier: Apache-2.0
// Exact top-K decode selection driven by a coarse score histogram.
// Configurations (DeepSeek-V4.1 + the GLM path):
//   Fp32Top2048             FP32 scores, K = 2048, 64-slot pages: the histogram
//   comes from DeepGEMM's paged MQA logits
//                           kernel (the GLM / DSA production path).
//   Fp32Top512              FP32 scores, K = 512, 128-slot pages (DeepSeek-V4.1
//   full-row index layers): the same
//                           producer histogram (MXFP4 producer).
//   Bf16Top512Page128       BF16 scores, K = 512, 128-slot pages (V4.1 full
//   rows with BF16 logits, DeepGEMM #462): the
//                           histogram from the producer or histogram_bf16.cuh's
//                           pre-pass; the Fused variant counts it.
//   Bf16Top512Block8        BF16 scores, K = 512, 8-score blocks (V4.1
//   consumer-layer candidate rows); also Fused.
// Bf16Top512 is the BF16 base (and histogram_bf16's part split), not a
// configuration of its own.
//
// The histogram counts every live score of a row into 1024 ordered bins (bin 0
// holds the largest scores). The bin where the running count reaches K splits
// the row: S scores above it are selected outright, and the K - S best of its c
// scores are resolved exactly. One launch covers all rows:
//   1. Each row is cut into parts, one CTA each. Up to one row per CTA, each
//   row owns gridDim / rows CTA slots
//      and uses as many as its length fills; more rows are taken whole, round
//      robin. From 16 to 63 rows a CTA has 1024 threads, and only parts long
//      enough to stream faster with 32 warps use them all.
//   2. Each CTA derives its row's split from the histogram and scans its part:
//   scores above the bin and inside
//      it are placed by a block scan (one-tile parts) or staged in shared
//      memory. One atomic on the row then reserves their positions and arrives:
//      the former go straight to the output, the latter to a per-row candidate
//      buffer.
//   3. The part arriving last ranks the candidates: directly when there are
//   few, otherwise through a fine
//      histogram over their ordered keys and an exact rank of the crossing fine
//      bin. Any count mismatch (a stale histogram, candidate overflow) falls
//      back to an exact radix select over the whole row, so the result never
//      depends on the histogram being right.
// The histogram, the hand-off words and the candidate buffers are returned to
// zero.

#pragma once

#include "coarse_bins.cuh"
#include <algorithm>
#include <cstdint>
#include <cuda_bf16.h>
#include <type_traits>

#ifndef LITETOPK_BF16_MIN_PART_UNITS
  #define LITETOPK_BF16_MIN_PART_UNITS 8
#endif

namespace litetopk {

constexpr uint32_t kCoarseBins = 1024;
constexpr uint32_t kVectors = 4;   // 16-byte loads per thread per tile
constexpr uint32_t kPageBits = 6;  // default page: 64 scores (Cfg::kPageBits
                                   // selects the page of a configuration)
constexpr uint32_t kMaxCount =
    1u << 21;  // above any live row length: bounds the histogram sums

// PDL may start this grid before the producer's writes are visible. Ordinary
// global loads after the entry wait keep both the vector scan and scalar
// fallback in the generic memory proxy; read-only cache loads cannot do that.
// Vector loads are kept in L1 so staging can read hits back; evict-first keeps
// the stream from displacing other lines. Measured (FP32): no-allocate slows
// staging, plain allocation slows long DRAM-bound rows.

// FP32 scores, top-2048: the configuration DeepGEMM's coarse histogram serves
struct Fp32Top2048 {
  using Score = float;
  using Vector = float4;
  static constexpr uint32_t kTopK = 2048;
  static constexpr uint32_t kPerVector = 4;
  static constexpr uint32_t kKeyBits =
      32;  // ordered key width; composites hold (key, ~slot)
  static constexpr uint32_t kUnit =
      256;  // partition granularity in scores (1 KiB)
  static constexpr uint32_t kMinPartUnits =
      8;  // measured: parts under 2048 scores only add hand-off traffic
  static constexpr uint32_t kFineBins = 2048;
  static constexpr uint32_t kFineBits = 11;
  static constexpr uint32_t kStage = 2048;  // the crossing fine bin's list
  static constexpr uint32_t kStaged =
      2 * kStage;  // hits a CTA stages per segment
  static constexpr uint32_t kRankCount =
      128;  // crossing bins up to this size are ranked directly
  static constexpr uint32_t kDirectRank =
      512;  // crossing fine bins up to this size are ranked directly
  static constexpr uint32_t kKeyAboveInf = 0xff800001u;
  static constexpr uint32_t kPageBits = litetopk::kPageBits;
  static constexpr uint32_t kLocalScores =
      0;  // rows up to this length are selected by one CTA alone (0: off)
  static constexpr bool kFused =
      false;  // the histogram comes from the producer

  static __device__ __forceinline__ float load(const float* ptr) {
    float value;
    asm volatile("ld.global.f32 %0, [%1];" : "=f"(value) : "l"(ptr) : "memory");
    return value;
  }

  static __device__ __forceinline__ float4 load_vector(const float* ptr) {
    float4 v;
    asm volatile("ld.global.L1::evict_first.v4.f32 {%0, %1, %2, %3}, [%4];"
                 : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
                 : "l"(ptr)
                 : "memory");
    return v;
  }

  static __device__ __forceinline__ float element(const float4& v, uint32_t q) {
    return q == 0 ? v.x : q == 1 ? v.y : q == 2 ? v.z : v.w;
  }

  static __device__ __forceinline__ uint32_t key(float x) {
    return ordered_key(x);
  }

  static __device__ __forceinline__ uint32_t edge(float e) {
    return edge_key(e);
  }

  // Counting FP32 scores without a producer (histogram_fp32.cuh's pre-pass):
  // coarse_bin takes the FP16-RN code of the score, so one packed conversion
  // gives two scores' codes. Word k holds scores 2k (low) and 2k + 1.
  static __device__ __forceinline__ void pair_words(const float4& v,
                                                    uint32_t (&w)[2]) {
    const __half2 a = __floats2half2_rn(v.x, v.y),
                  b = __floats2half2_rn(v.z, v.w);
    w[0] = *reinterpret_cast<const uint32_t*>(&a);
    w[1] = *reinterpret_cast<const uint32_t*>(&b);
  }

  // Byte offsets (4 x bin) of both halves of an FP16 pair: bin = 511 - c for a
  // positive half, 512 + c for a negative one, c = (|h| >> 6); valid where
  // pair_slow is clear
  static __device__ __forceinline__ uint32_t pair_offsets(uint32_t w) {
    uint32_t negative;  // 0xffff per negative half (sign-replicating PRMT)
    asm("prmt.b32 %0, %1, 0, 0xbb99;" : "=r"(negative) : "r"(w));
    return ((w >> 4) & 0x07fc07fcu) ^ 0x07fc07fcu ^ (negative & 0x0ffc0ffcu);
  }

  // Bit 15 / 31 set for a half whose bin needs coarse_bin: code >= 304 (|x| >=
  // 16, inf, NaN) or a zero half (an FP32 -0.0 lies in bin 511, a negative
  // score rounding to -0 in bin 512)
  static __device__ __forceinline__ uint32_t pair_slow(uint32_t w) {
    const uint32_t magnitude = w & 0x7fff7fffu;
    return ((magnitude + 0x34003400u) | ~(magnitude + 0x7fff7fffu)) &
           0x80008000u;
  }

  static __device__ __forceinline__ void count(uint32_t* bins, float x) {
    if (not isnan(x)) atomicAdd(bins + coarse_bin(x), 1u);
  }

  static __device__ __forceinline__ void count_offset(uint32_t* bins,
                                                      uint32_t offset) {
    atomicAdd(reinterpret_cast<uint32_t*>(
                  reinterpret_cast<unsigned char*>(bins) + offset),
              1u);
  }
};

// FP32 scores, top-512: DeepSeek-V4.1's full-row index layers (ratio-2 layers 2
// / 8 / 14 and the ratio-1 candidate source layer 20), whose MXFP4 paged
// producer counts the same coarse histogram. Only the K-dependent stage shrinks
// (as for BF16 / 512: < K scores above the crossing bin plus a segment's share
// of it); the scan, the part split, the fine histogram over 32-bit ordered keys
// and the direct-rank limits do not depend on K. The V4.1 index-K pool pages
// hold 128 slots (SGLang's DSV41_INDEX_PAGE_SIZE with DeepGEMM's sparse MQA
// kernel): a score index i maps to table[row][i >> 7] * 128 + (i & 127),
// SGLang's topk_transform_paged_v2 with page_size 128. Rows of at most one
// 512-thread tile (8192 scores: contexts up to 8K / 16K tokens at ratio 1 / 2)
// are selected by a single CTA without the row hand-off (Selector::local_row):
// such a row fits the registers of one CTA, and the global arrival atomic, the
// candidate round trip and the extra barriers of the multi-part path cost more
// than its scan. Each part loads its row's histogram together with the row's
// length, one round trip earlier (Selector::run; in the 1024-thread kernel only
// a local row does).
#ifndef LITETOPK_LOCAL_SCORES
  #define LITETOPK_LOCAL_SCORES 8192
#endif
struct Fp32Top512 : Fp32Top2048 {
  static constexpr uint32_t kTopK = 512;
  static constexpr uint32_t kStage = 2048;
  static constexpr uint32_t kStaged = 2048;
  static constexpr uint32_t kPageBits = 7;
  static constexpr uint32_t kLocalScores = LITETOPK_LOCAL_SCORES;
};

// BF16 scores, top-512. Relative to FP32, K-dependent sizes shrink by K and the
// 16-byte loads carry 8 scores:
//   kUnit 512       keeps a unit (and so the minimum part, kWideTiles and the
//   tile) at the FP32 byte size: the scan
//                   streams bytes, and its per-byte compare cost is similar
//                   with packed BF16x2 compares.
//   kFineBins 1024  at least the BF16 key width of every coarse bin but the
//   four at +-0 and +-inf, so the fine
//                   histogram ranks exactly on the 16-bit key (shift 0); one
//                   bin per thread of a 1024-thread CTA.
//   kStaged 2048    a segment's hits: < K above the crossing bin plus its share
//   of the bin (4 x K, 16 KiB). kStage 2048     the crossing fine bin (with
//   shift 0: the ties of one BF16 value) shares the staging memory. kRankCount
//   128 and kDirectRank 512 as for FP32: the quadratic ranks cost count
//   iterations whatever K is.
struct Bf16Top512 {
  using Score = __nv_bfloat16;
  using Vector = uint4;
  static constexpr uint32_t kTopK = 512;
  static constexpr uint32_t kPerVector = 8;
  static constexpr uint32_t kKeyBits = 16;
  static constexpr uint32_t kUnit = 512;
  static constexpr uint32_t kMinPartUnits = LITETOPK_BF16_MIN_PART_UNITS;
  static constexpr uint32_t kFineBins = 1024;
  static constexpr uint32_t kFineBits = 10;
  static constexpr uint32_t kStage = 2048;
  static constexpr uint32_t kStaged = 2048;
  static constexpr uint32_t kRankCount = 128;
  static constexpr uint32_t kDirectRank = 512;
  static constexpr uint32_t kKeyAboveInf = 0xff81u;
  static constexpr uint32_t kPageBits = litetopk::kPageBits;
  static constexpr uint32_t kLocalScores = 0;
  static constexpr bool kFused =
      false;  // the histogram comes from histogram_bf16's pre-pass

  static __device__ __forceinline__ float load(const __nv_bfloat16* ptr) {
    unsigned short bits;
    asm volatile("ld.global.u16 %0, [%1];" : "=h"(bits) : "l"(ptr) : "memory");
    return __uint_as_float(static_cast<uint32_t>(bits) << 16);
  }

  static __device__ __forceinline__ uint4
  load_vector(const __nv_bfloat16* ptr) {
    uint4 v;
    asm volatile("ld.global.L1::evict_first.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(ptr)
                 : "memory");
    return v;
  }

  // Score q of a vector: the low half of word q / 2 holds score q = 2 (q / 2)
  static __device__ __forceinline__ float element(const uint4& v, uint32_t q) {
    const uint32_t w = (q >> 1) == 0   ? v.x
                       : (q >> 1) == 1 ? v.y
                       : (q >> 1) == 2 ? v.z
                                       : v.w;
    return __uint_as_float(q & 1 ? w & 0xffff0000u : w << 16);
  }

  // 16-bit ordered key of a BF16 value held in a float (its low 16 bits are
  // zero)
  static __device__ __forceinline__ uint32_t key(float x) {
    return ordered_key(x) >> 16;
  }

  // Smallest key of a BF16 value >= e; a zero edge also admits -0.0
  static __device__ __forceinline__ uint32_t edge(float e) {
    return e == 0.0f
               ? 0x7fffu
               : ordered_key(__bfloat162float(__float2bfloat16_ru(e))) >> 16;
  }

  // Both halves = the smallest BF16 >= t, so that a BF16 x satisfies x >= t (as
  // floats) iff x >= that value
  static __device__ __forceinline__ uint32_t threshold(float t) {
    const auto h = __bfloat16_as_ushort(__float2bfloat16_ru(t));
    return static_cast<uint32_t>(h) * 0x10001u;
  }

  // Bit q of the result: score q of the vector is >= the packed threshold
  static __device__ __forceinline__ uint32_t at_least(const uint4& v,
                                                      uint32_t t2) {
    const auto t = *reinterpret_cast<const __nv_bfloat162*>(&t2);
    const uint32_t m0 =
        __hge2_mask(*reinterpret_cast<const __nv_bfloat162*>(&v.x), t);
    const uint32_t m1 =
        __hge2_mask(*reinterpret_cast<const __nv_bfloat162*>(&v.y), t);
    const uint32_t m2 =
        __hge2_mask(*reinterpret_cast<const __nv_bfloat162*>(&v.z), t);
    const uint32_t m3 =
        __hge2_mask(*reinterpret_cast<const __nv_bfloat162*>(&v.w), t);
    // One flag byte per score (0xff / 0x00), then signed dot products weight
    // them 1, 2, 4, ... 128
    const uint32_t lo = __byte_perm(m0, m1, 0x7531),
                   hi = __byte_perm(m2, m3, 0x7531);
    const int low =
        __dp4a(static_cast<int>(lo), static_cast<int>(0xf8fcfeffu), 0);
    return static_cast<uint32_t>(
        __dp4a(static_cast<int>(hi), static_cast<int>(0x80c0e0f0u), low));
  }

  // The vector's four BF16 pairs (word k: scores 2k, 2k + 1)
  static __device__ __forceinline__ void pair_words(const uint4& v,
                                                    uint32_t (&w)[4]) {
    w[0] = v.x, w[1] = v.y, w[2] = v.z, w[3] = v.w;
  }

  // Byte offsets (4 x bin) of both scores of a BF16 pair (low half = the first)
  // for 2^-14 <= |x| < 16, where FP16 holds a BF16 value exactly: bin = 511 - c
  // for x >= 0, 512 + c for x < 0, with the FP16 code c = (|bits| >> 3) - 1792
  static __device__ __forceinline__ uint32_t pair_offsets(uint32_t w) {
    const uint32_t t =
        ((w >> 1) & 0x3ffc3ffcu) - 0x14001400u;  // 4 c + 0x800 per half
    uint32_t negative;  // 0xffff per negative half (sign-replicating PRMT)
    asm("prmt.b32 %0, %1, 0, 0xbb99;" : "=r"(negative) : "r"(w));
    return (t ^ ~negative) & 0x0ffc0ffcu;
  }

  // Bit 15 / 31 set for a half outside that range: |x| >= 16 (unit bins, inf,
  // NaN) or |x| < 2^-14 (FP16 subnormal)
  static __device__ __forceinline__ uint32_t pair_slow(uint32_t w) {
    const uint32_t magnitude = w & 0x7fff7fffu;
    return ((magnitude + 0x3e803e80u) | ~(magnitude + 0x47804780u)) &
           0x80008000u;
  }

  static __device__ __forceinline__ void count(uint32_t* bins, float x) {
    if (not isnan(x)) atomicAdd(bins + coarse_bin(x), 1u);
  }

  static __device__ __forceinline__ void count_offset(uint32_t* bins,
                                                      uint32_t offset) {
    atomicAdd(reinterpret_cast<uint32_t*>(
                  reinterpret_cast<unsigned char*>(bins) + offset),
              1u);
  }
};

// BF16 scores, top-512 over DeepSeek-V4.1's sparse (candidate) rows: column j
// of the row is slot blocks[row][j >> 3] * 8 + (j & 7) (SGLang's
// topk_transform_sparse: blocks of CANDIDATE_BLOCK_SIZE = 8 scores)
struct Bf16Top512Block8 : Bf16Top512 {
  static constexpr uint32_t kPageBits = 3;
};

// Fused: the coarse histogram is counted by the selector itself. Each part
// counts its scores into shared bins; a one-part row derives its split from
// them directly, the parts of a longer row add them to the row's global bins
// and wait for each other (every part of a row is resident: the grid is one
// wave).
struct Bf16Top512Block8Fused : Bf16Top512Block8 {
  static constexpr bool kFused = true;
};

// BF16 scores, top-512 on 128-slot pages: DeepSeek-V4.1's full-row index layers
// with BF16 logits (DeepGEMM #462, whose paged MQA logits kernel can count the
// coarse histogram of its BF16 output; histogram_bf16.cuh's pre-pass
// otherwise). The index-K pool pages hold 128 slots (DSV41_INDEX_PAGE_SIZE), as
// for Fp32Top512. Rows of (K, LITETOPK_BF16_LOCAL_SCORES] scores (default 16384
// = one 512-thread BF16 tile) are selected by one CTA (Selector::local_row, the
// path Fp32Top512 uses for rows of one FP32 tile; nothing in it depends on K,
// and its 16-bit place counters hold <= 16384 scores per kind). Measured on
// B200, selector alone: 1.20-1.48x over the multi-part path for such rows (B
// 1-4096, L 1024-16384). The fused variant counts its histogram per part
// (process_segment), so it has no local rows.
#ifndef LITETOPK_BF16_LOCAL_SCORES
  #define LITETOPK_BF16_LOCAL_SCORES 16384
#endif
struct Bf16Top512Page128 : Bf16Top512 {
  static constexpr uint32_t kPageBits = 7;
  static constexpr uint32_t kLocalScores = LITETOPK_BF16_LOCAL_SCORES;
};

struct Bf16Top512Page128Fused : Bf16Top512Page128 {
  static constexpr bool kFused = true;
  static constexpr uint32_t kLocalScores = 0;
};

// Per-row hand-off. One atomic per part both reserves output/candidate
// positions and arrives; `word` is zero at rest and `done` equals `done_base`.
struct alignas(16) RowState {
  unsigned long long word;  // [63:48] parts arrived, [47:24] scores above the
                            // crossing bin, [23:0] inside it
  uint32_t done;       // kPartSignal per part whose writes have landed, counted
                       // across calls
  uint32_t done_base;  // `done` at the start of the call: each finalizer
                       // advances it by the other parts' signals
};

template <typename Score>
struct Params {
  const Score* scores;
  int64_t score_stride;
  uint32_t score_width;
  const int32_t* lengths;
  int32_t* histogram;
  int64_t out_stride;
  int32_t* out;
  RowState* state;
  unsigned long long* candidates;
  uint32_t capacity;
  uint32_t rows;
};

// A row's split from the histogram, identical in every CTA that touches the row
struct Split {
  bool certified;
  bool whole;  // every score of the crossing bin is selected
  int32_t bin;
  uint32_t strict;
  uint32_t count;
  float lo;  // scores >= lo lie in bins 0..bin
  float hi;  // scores >= hi lie above the crossing bin (NaN for bin 0)
  uint32_t key_lo;
  uint32_t shift;  // fine bin = (ordered_key - key_lo) >> shift
};

__device__ __forceinline__ uint32_t lane_id() { return threadIdx.x & 31; }

// The shuffle's predicate marks the lanes that have a source: two instructions
// per step
__device__ __forceinline__ uint32_t warp_inclusive_sum(uint32_t v) {
  asm volatile(
      "{\n.reg .pred p;\n.reg .b32 t;\n"
      "shfl.sync.up.b32 t|p, %0, 1, 0, -1;\n@p add.u32 %0, %0, t;\n"
      "shfl.sync.up.b32 t|p, %0, 2, 0, -1;\n@p add.u32 %0, %0, t;\n"
      "shfl.sync.up.b32 t|p, %0, 4, 0, -1;\n@p add.u32 %0, %0, t;\n"
      "shfl.sync.up.b32 t|p, %0, 8, 0, -1;\n@p add.u32 %0, %0, t;\n"
      "shfl.sync.up.b32 t|p, %0, 16, 0, -1;\n@p add.u32 %0, %0, t;\n}"
      : "+r"(v));
  return v;
}

// Reserves and arrives on a row. No ordering is needed: everything the
// finalizer reads is either in the word itself or a candidate entry it waits
// for.
__device__ __forceinline__ unsigned long long arrive(unsigned long long* word,
                                                     unsigned long long add) {
  unsigned long long old;
  asm volatile("atom.relaxed.gpu.global.add.u64 %0, [%1], %2;"
               : "=l"(old)
               : "l"(word), "l"(add)
               : "memory");
  return old;
}

// Marks a part's writes (ordered before it by a barrier) as landed
__device__ __forceinline__ void signal_done(uint32_t* done, uint32_t weight) {
  asm volatile("red.release.gpu.global.add.u32 [%0], %1;" ::"l"(done),
               "r"(weight)
               : "memory");
}

// Every part's warps signal kPartSignal in total, whatever the width of the CTA
// that ran it
constexpr uint32_t kPartSignal = 32;

__device__ __forceinline__ uint32_t load_acquire(const uint32_t* ptr) {
  uint32_t v;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];"
               : "=r"(v)
               : "l"(ptr)
               : "memory");
  return v;
}

__device__ __forceinline__ unsigned long long load_relaxed(
    const unsigned long long* ptr) {
  unsigned long long v;
  asm volatile("ld.relaxed.gpu.global.u64 %0, [%1];"
               : "=l"(v)
               : "l"(ptr)
               : "memory");
  return v;
}

// A candidate entry is never zero; its writer may still be in flight after the
// arrival
__device__ __forceinline__ unsigned long long await_entry(
    const unsigned long long* ptr, unsigned long long v) {
  while (v == 0) v = load_relaxed(ptr);
  return v;
}

// Ordering composite over a score's key and its logical index: larger is
// better, ties go to the lower slot
__device__ __forceinline__ unsigned long long composite(uint32_t key,
                                                        int32_t slot) {
  return (static_cast<unsigned long long>(key) << 32) |
         ~static_cast<uint32_t>(slot);
}

__device__ __forceinline__ int32_t composite_slot(unsigned long long v) {
  return static_cast<int32_t>(~static_cast<uint32_t>(v));
}

__device__ __forceinline__ uint32_t composite_key(unsigned long long v) {
  return static_cast<uint32_t>(v >> 32);
}

template <typename Score>
__device__ __forceinline__ uint32_t row_length(const Params<Score>& p,
                                               uint32_t row) {
  return min(static_cast<uint32_t>(max(p.lengths[row], 0)), p.score_width);
}

// Parts a row of `length` scores uses of its `slots`: none shorter than
// Cfg::kMinPartUnits units
template <typename Cfg>
__device__ __forceinline__ uint32_t parts_for(uint32_t slots, uint32_t length) {
  if constexpr (Cfg::kLocalScores != 0) {
    if (length <= Cfg::kLocalScores)
      return 1;  // Selector::local_row: the whole row on one CTA
  }
  return min(slots, max(1u, (length + Cfg::kUnit - 1) / Cfg::kUnit /
                                Cfg::kMinPartUnits));
}

// Score range [x, y) of part `part` of `parts`: the parts share the row's
// kUnit-score units evenly
template <typename Cfg>
__device__ __forceinline__ uint2 segment(uint32_t length, uint32_t part,
                                         uint32_t parts) {
  const uint32_t units = (length + Cfg::kUnit - 1) / Cfg::kUnit;
  return make_uint2(part * units / parts * Cfg::kUnit,
                    min((part + 1) * units / parts * Cfg::kUnit, length));
}

extern __shared__ __align__(16) unsigned char dynamic_smem[];

// Path coverage counters of the qualification (LITETOPK_COVERAGE builds only;
// otherwise LITETOPK_COV is empty and the code is unchanged): thread 0 of a CTA
// counts each pass through a CTA-uniform decision.
enum Coverage : uint32_t {
  kCovShort,            // row of at most K scores (part 0)
  kCovLocal,            // Selector::local_row
  kCovLocalPacked,      // ... classified with packed FP16 compares
  kCovLocalWhole,       // ... every score of the crossing bin selected
  kCovLocalSmall,       // ... crossing bin <= kRankCount, ballot rank
  kCovLocalFineTake,    // ... fine histogram, crossing fine bin taken whole
  kCovLocalFineBallot,  // ... crossing fine bin <= kRankCount
  kCovLocalFineDirect,  // ... crossing fine bin <= kDirectRank
  kCovLocalFineRadix,   // ... crossing fine bin radix-selected
  kCovLocalFallback,    // ... exact select of the row (stale histogram, list
                        // overflow)
  kCovTile512,      // multi-part path: one-tile part, publish_tile, 512 threads
  kCovSegment512,   // ... scan_segment + publish_segment, 512 threads
  kCovTile1024,     // ... publish_tile, 1024 threads (wide kernel)
  kCovSegment1024,  // ... publish_segment, 1024 threads
  kCovStagingOverflow,   // ... a part staged more hits than kStaged
  kCovFinalUncertified,  // finalizer: no crossing bin in the histogram (stale /
                         // NaN row)
  kCovFinalMismatch,     // ... certified, but the parts' counts differ from it
  kCovFinalCapacity,     // ... crossing bin beyond the candidate capacity
  kCovFinalWhole,        // ... crossing bin taken whole
  kCovFinalSmall,        // resolve_bin: crossing bin <= kRankCount
  kCovFinalFineTake,     // ... crossing fine bin taken whole
  kCovFinalFineDirect,   // ... crossing fine bin <= kDirectRank
  kCovFinalFineRadix,    // ... crossing fine bin radix-selected
  kCovFinalFineFail,     // ... no crossing fine bin / list overflow: fallback
  kCovFinalFallback,     // finalizer: exact select of the row
  kCovFusedOnePart,      // fused: one-part row, split from the shared bins
  kCovFusedMultiPart,    // fused: parts add their bins and wait for each other
  kCovHistBf16,          // histogram_bf16 part counted
  kCovHistFp32,          // histogram_fp32 part counted
  kCovCount
};
#ifdef LITETOPK_COVERAGE
__device__ unsigned long long litetopk_coverage[64];
  #define LITETOPK_COV(point)                                   \
    do {                                                        \
      if (threadIdx.x == 0)                                     \
        atomicAdd(&::litetopk::litetopk_coverage[point], 1ull); \
    } while (0)
#else
  #define LITETOPK_COV(point) \
    do {                      \
    } while (0)
#endif

// Everything that depends on the CTA's thread count. A 1024-thread CTA runs
// either Selector<Cfg, 1024> or, with its upper half exited, Selector<Cfg,
// 512>: barriers count the participating threads explicitly.
template <typename Cfg, uint32_t kThreads>
struct Selector {
  using Score = typename Cfg::Score;
  using Vector = typename Cfg::Vector;
  static constexpr bool kFp32 = std::is_same_v<Score, float>;
  static constexpr uint32_t kTopK = Cfg::kTopK;
  static constexpr uint32_t kFineBins = Cfg::kFineBins;
  static constexpr uint32_t kFineBits = Cfg::kFineBits;
  static constexpr uint32_t kStage = Cfg::kStage;
  static constexpr uint32_t kStaged = Cfg::kStaged;
  static constexpr uint32_t kRankCount = Cfg::kRankCount;
  static constexpr uint32_t kDirectRank = Cfg::kDirectRank;
  static constexpr uint32_t kPerVector = Cfg::kPerVector;
  static constexpr uint32_t kWarps = kThreads / 32;
  static constexpr uint32_t kTile = kThreads * kVectors * kPerVector;
  static constexpr uint32_t kCoarsePerThread =
      kCoarseBins / kThreads;  // 2 or 1
  static constexpr uint32_t kFinePerThread = kFineBins / kThreads;
  static_assert(kFineBins == 1u << kFineBits);
  static_assert(kStaged >= kTopK,
                "a CTA's scores above the crossing bin always fit its stage");
  static_assert(kStage <= kStaged,
                "the crossing fine bin's list shares the staging memory");
  static_assert(kVectors * kPerVector <= 32,
                "a thread's tile scores index one 32-bit mask");
  static_assert(kCoarseBins % kThreads == 0 and kFineBins % kThreads == 0 and
                kStaged % kThreads == 0);
  static_assert(kPartSignal % kWarps == 0);
  static_assert(not(Cfg::kFused and Cfg::kLocalScores != 0),
                "a fused configuration counts its histogram per part");
  using P = Params<Score>;

  struct Shared {
    uint32_t scan[kWarps];
    uint32_t radix[256];
    union {
      unsigned long long
          staged[kStaged];       // (score bits, index) of the segment's hits
      uint32_t fine[kFineBins];  // the crossing bin's fine histogram
      unsigned long long list[kStage];  // the crossing fine bin's composites
      uint32_t coarse[kCoarseBins];     // Cfg::kFused: the segment's coarse
                                        // histogram, until the split
    };
    uint32_t staged_count;  // hits of the segment, possibly beyond kStaged
    uint32_t strict_count;  // those of them above the crossing bin
    unsigned long long reserved;
    unsigned long long
        row_word;  // the finalizer's view of its row's hand-off word
    int32_t bin;
    uint32_t strict;
    uint32_t count;
    uint32_t emitted;
    uint32_t listed;
  };

  static __device__ __forceinline__ Shared& sm() {
    return *reinterpret_cast<Shared*>(dynamic_smem);
  }

  static __device__ __forceinline__ void sync() {
    asm volatile("bar.sync 1, %0;" ::"n"(kThreads) : "memory");
  }

  // Exclusive prefix in thread order. Contains one barrier; `scratch` is free
  // again after the next barrier.
  static __device__ __forceinline__ uint32_t
  block_exclusive_sum(uint32_t v, uint32_t* scratch, uint32_t& total) {
    const uint32_t inclusive = warp_inclusive_sum(v);
    if (lane_id() == 31) scratch[threadIdx.x >> 5] = inclusive;
    sync();
    const uint32_t warp_total = lane_id() < kWarps ? scratch[lane_id()] : 0u;
    const uint32_t warp_inclusive = warp_inclusive_sum(warp_total);
    total = __shfl_sync(0xffffffffu, warp_inclusive, kWarps - 1);
    return __shfl_sync(0xffffffffu, warp_inclusive - warp_total,
                       threadIdx.x >> 5) +
           inclusive - v;
  }

  // This thread's tile vectors that start before `end`. Nothing else touches
  // the destination registers, so the loads stay in flight until the scores are
  // compared; vectors past `end` keep stale values that `live_bits` masks.
  static __device__ __forceinline__ void load_tile(Vector (&v)[kVectors],
                                                   const Score* row,
                                                   uint32_t base,
                                                   uint32_t end) {
    if (base + kTile <= end) {
#pragma unroll
      for (uint32_t u = 0; u < kVectors; ++u)
        v[u] = Cfg::load_vector(row + base +
                                (u * kThreads + threadIdx.x) * kPerVector);
      return;
    }
#pragma unroll
    for (uint32_t u = 0; u < kVectors; ++u) {
      const uint32_t first = base + (u * kThreads + threadIdx.x) * kPerVector;
      if (first < end) v[u] = Cfg::load_vector(row + first);
    }
  }

  // Bit u * kPerVector + q is set when score q of this thread's vector u lies
  // before `end`
  static __device__ __forceinline__ uint32_t live_bits(uint32_t base,
                                                       uint32_t end) {
    uint32_t bits = 0;
#pragma unroll
    for (uint32_t u = 0; u < kVectors; ++u) {
      const uint32_t first = base + (u * kThreads + threadIdx.x) * kPerVector;
      const uint32_t live = end > first ? min(end - first, kPerVector) : 0u;
      bits |= ((1u << live) - 1) << (u * kPerVector);
    }
    return bits;
  }

  // Bit u * kPerVector + q: score q of vector u is >= the BF16 threshold `t2`
  // (Bf16Top512::threshold)
  static __device__ __forceinline__ uint32_t
  tile_at_least(const Vector (&v)[kVectors], uint32_t t2) {
    uint32_t bits = 0;
#pragma unroll
    for (uint32_t u = 0; u < kVectors; ++u)
      bits |= Cfg::at_least(v[u], t2) << (u * kPerVector);
    return bits;
  }

  // This thread's coarse bins (kCoarsePerThread of them, in .x then .y)
  static __device__ __forceinline__ int2 load_histogram(const P& p,
                                                        uint32_t row) {
    const int32_t* bins = p.histogram + static_cast<size_t>(row) * kCoarseBins;
    if constexpr (kCoarsePerThread == 2)
      return __ldcg(reinterpret_cast<const int2*>(bins) + threadIdx.x);
    return make_int2(__ldcg(bins + threadIdx.x), 0);
  }

  static __device__ __forceinline__ void clear_histogram(const P& p,
                                                         uint32_t row) {
    int32_t* bins = p.histogram + static_cast<size_t>(row) * kCoarseBins;
    if constexpr (kCoarsePerThread == 2) {
      reinterpret_cast<int2*>(bins)[threadIdx.x] = make_int2(0, 0);
    } else {
      bins[threadIdx.x] = 0;
    }
  }

  // BF16: counts this thread's live scores of the tile at `base` into `bins`;
  // `next` (between the offsets and the shared adds) issues the next tile's
  // loads
  template <typename Next>
  static __device__ __forceinline__ void count_tile(uint32_t* bins,
                                                    const Vector (&v)[kVectors],
                                                    uint32_t base, uint32_t end,
                                                    const Next& next) {
    constexpr uint32_t kPairs =
        kPerVector / 2;  // 32-bit words of two scores' bins (Cfg::pair_words)
    uint32_t offsets[kVectors][kPairs];
    uint32_t fast =
        0;  // bit u: vector u is whole and all its scores take the integer path
#pragma unroll
    for (uint32_t u = 0; u < kVectors; ++u) {
      const uint32_t first = base + (u * kThreads + threadIdx.x) * kPerVector;
      uint32_t w[kPairs];
      Cfg::pair_words(v[u], w);
      uint32_t slow = 0;
#pragma unroll
      for (uint32_t k = 0; k < kPairs; ++k) {
        offsets[u][k] = Cfg::pair_offsets(w[k]);
        slow |= Cfg::pair_slow(w[k]);
      }
      const bool whole = first + kPerVector <= end;
      fast |= static_cast<uint32_t>(whole and slow == 0) << u;
      if (not(whole and slow == 0) and first < end) {
        // Rare: a row's last partial vector, or scores outside the integer
        // path's range
#pragma unroll
        for (uint32_t q = 0; q < kPerVector; ++q) {
          if (first + q < end) Cfg::count(bins, Cfg::element(v[u], q));
        }
      }
    }
    next();
#pragma unroll
    for (uint32_t u = 0; u < kVectors; ++u) {
      if (fast >> u & 1) {
#pragma unroll
        for (uint32_t k = 0; k < kPairs; ++k) {
          Cfg::count_offset(bins, offsets[u][k] & 0xffffu);
          Cfg::count_offset(bins, offsets[u][k] >> 16);
        }
      }
    }
  }

  // Cfg::kFused: counts [begin, end) with `v` holding the first tile, and
  // leaves the first tile in `v` again
  static __device__ __forceinline__ void count_segment(const Score* x,
                                                       uint32_t begin,
                                                       uint32_t end,
                                                       Vector (&v)[kVectors]) {
    for (uint32_t i = threadIdx.x; i < kCoarseBins; i += kThreads)
      sm().coarse[i] = 0;
    sync();
    uint32_t* bins = sm().coarse;
    const auto none = [] {};
    if (end - begin <= kTile) {
      count_tile(bins, v, begin, end, none);
      return;
    }
    // Two register tiles in turn: the next tile's loads are issued before the
    // shared adds of the current one
    Vector b[kVectors] = {};
    load_tile(b, x, begin + kTile, end);
    count_tile(bins, v, begin, end, none);
    for (uint32_t base = begin + kTile;; base += 2 * kTile) {
      count_tile(bins, b, base, end, [&] {
        if (base + kTile < end) load_tile(v, x, base + kTile, end);
      });
      if (base + kTile >= end) break;
      count_tile(bins, v, base + kTile, end, [&] {
        if (base + 2 * kTile < end) load_tile(b, x, base + 2 * kTile, end);
      });
      if (base + 2 * kTile >= end) break;
    }
    load_tile(
        v, x, begin,
        end);  // the scan starts over from the first tile (now in L1 or L2)
  }

  // Cfg::kFused: the row's coarse bins of this thread. A one-part row keeps
  // them in shared memory; the parts of a longer row add theirs to the row's
  // global bins, arrive on the row's counter and wait for all of its parts.
  static __device__ __forceinline__ int2 fused_bins(const P& p, uint32_t row,
                                                    uint32_t parts) {
    sync();
    if (parts == 1) {
      LITETOPK_COV(kCovFusedOnePart);
      if constexpr (kCoarsePerThread == 2)
        return reinterpret_cast<const int2*>(sm().coarse)[threadIdx.x];
      return make_int2(static_cast<int32_t>(sm().coarse[threadIdx.x]), 0);
    }
    LITETOPK_COV(kCovFusedMultiPart);
    int32_t* global = p.histogram + static_cast<size_t>(row) * kCoarseBins;
    if constexpr (kCoarsePerThread == 2) {
      const uint2 c = reinterpret_cast<const uint2*>(sm().coarse)[threadIdx.x];
      if ((c.x | c.y) != 0)
        atomicAdd(reinterpret_cast<unsigned long long*>(global) + threadIdx.x,
                  static_cast<unsigned long long>(c.x) |
                      (static_cast<unsigned long long>(c.y) << 32));
    } else {
      const uint32_t c = sm().coarse[threadIdx.x];
      if (c != 0) atomicAdd(global + threadIdx.x, static_cast<int32_t>(c));
    }
    sync();
    if (threadIdx.x == 0) {
      // Per row after the candidates: {parts counted, counted at the start of
      // the call}; the finalizer advances the second by the row's parts once
      // every part has passed this wait. The barrier above orders the CTA's
      // histogram adds before the release.
      uint32_t* counter =
          reinterpret_cast<uint32_t*>(
              p.candidates + static_cast<size_t>(p.rows) * p.capacity) +
          2 * row;
      const uint32_t base = load_acquire(counter + 1);
      signal_done(counter, 1);
      while (load_acquire(counter) - base != parts) {
      }
    }
    sync();
    return load_histogram(p, row);
  }

  static __device__ Split derive_split(int2 bins, uint32_t length) {
    if (threadIdx.x == 0) sm().bin = -1;
    const uint32_t a = min(static_cast<uint32_t>(bins.x), kMaxCount);
    const uint32_t b = min(static_cast<uint32_t>(bins.y), kMaxCount);
    uint32_t total;
    const uint32_t before = block_exclusive_sum(a + b, sm().scan, total);
    // The crossing bin holds the K-th largest score: before < K <= before +
    // count
    if (total == length) {
      if (before < kTopK and kTopK <= before + a) {
        sm().bin = kCoarsePerThread * threadIdx.x, sm().strict = before,
        sm().count = a;
      } else if (before + a < kTopK and kTopK <= before + a + b) {
        sm().bin = kCoarsePerThread * threadIdx.x + 1, sm().strict = before + a,
        sm().count = b;
      }
    }
    sync();
    Split s;
    s.bin = sm().bin;
    s.certified = s.bin >= 0;
    s.strict = sm().strict;
    s.count = sm().count;
    s.whole = s.certified and s.count == kTopK - s.strict;
    const float nan = __int_as_float(0x7fc00000);
    s.lo = s.certified ? coarse_lower_edge(s.bin) : nan;
    s.hi = s.bin > 0 ? coarse_lower_edge(s.bin - 1) : nan;
    s.key_lo = Cfg::edge(s.lo);
    const uint32_t key_hi =
        s.bin > 0 ? Cfg::edge(s.hi) : Cfg::kKeyAboveInf;  // just above +inf
    const uint32_t width = key_hi - s.key_lo;
    s.shift = width > kFineBins ? 32 - __clz(width - 1) - kFineBits : 0;
    return s;
  }

  // Score j of this thread's tile vectors, without indexing registers
  // dynamically
  static __device__ __forceinline__ float pick(const Vector (&v)[kVectors],
                                               uint32_t j) {
    static_assert(kVectors == 4);
    if constexpr (kFp32) {
      const uint32_t u = j >> 2, q = j & 3;
      float4 w = v[0];
      w = u == 1 ? v[1] : w;
      w = u == 2 ? v[2] : w;
      w = u == 3 ? v[3] : w;
      float x = w.x;
      x = q == 1 ? w.y : x;
      x = q == 2 ? w.z : x;
      x = q == 3 ? w.w : x;
      return x;
    } else {
      const uint32_t u = j / kPerVector, q = (j % kPerVector) >> 1;
      Vector w = v[0];
      w = u == 1 ? v[1] : w;
      w = u == 2 ? v[2] : w;
      w = u == 3 ? v[3] : w;
      uint32_t x = w.x;
      x = q == 1 ? w.y : x;
      x = q == 2 ? w.z : x;
      x = q == 3 ? w.w : x;
      return __uint_as_float(j & 1 ? x & 0xffff0000u : x << 16);
    }
  }

  // Stages one warp's hits of a tile in shared memory as (score bits, index)
  // and counts this thread's hits above the crossing bin. `hits` marks this
  // thread's scores >= lo; their scores are read back from L1, where the tile's
  // loads left them.
  static __device__ __forceinline__ void stage_hits(const Score* x,
                                                    uint32_t hits,
                                                    uint32_t base, float hi,
                                                    uint32_t& strict_mine) {
    const uint32_t mine = __popc(hits);
    const uint32_t inclusive = warp_inclusive_sum(mine);
    uint32_t first = 0;
    if (lane_id() == 31) first = atomicAdd(&sm().staged_count, inclusive);
    uint32_t pos = __shfl_sync(0xffffffffu, first, 31) + inclusive - mine;
    for (uint32_t left = hits; left != 0; left &= left - 1, ++pos) {
      const uint32_t j = __ffs(left) - 1;
      const uint32_t i =
          base + ((j / kPerVector) * kThreads + threadIdx.x) * kPerVector +
          (j % kPerVector);
      const float score = Cfg::load(x + i);
      strict_mine += score >= hi;
      if (pos < kStaged)
        sm().staged[pos] =
            (static_cast<unsigned long long>(__float_as_uint(score)) << 32) | i;
    }
  }

  // Publishes the segment's staged hits with one atomic on the row that
  // reserves their positions and arrives; returns whether this part arrived
  // last. Candidates are written after the arrival: the finalizer waits for
  // each entry to turn nonzero. Positions past the certified counts only arise
  // from an inconsistent histogram, which the finalizer detects.
  static __device__ bool publish_segment(const P& p, const Split& s,
                                         uint32_t row, uint32_t parts,
                                         int32_t* out_row) {
    constexpr uint32_t kPerThread = kStaged / kThreads;

    sync();
    const uint32_t staged = min(sm().staged_count, kStaged);
    unsigned long long before = 0, add = 0;
    if (threadIdx.x == 0) {
      // A part that staged more hits than it holds adds K to the strict count:
      // the finalizer then sees a mismatch
      const bool overflow = sm().staged_count > kStaged;
      if (overflow) LITETOPK_COV(kCovStagingOverflow);
      add = (1ull << 48) |
            (static_cast<unsigned long long>(sm().strict_count +
                                             (overflow ? kTopK : 0u))
             << 24) |
            (sm().staged_count - sm().strict_count);
      before = arrive(
          &p.state[row].word,
          add);  // read only after the split below: the round trip overlaps it
    }
    // Each thread splits its staged hits at the crossing bin, as staging
    // counted them, and looks up their pages
    uint32_t live = 0, strict = 0;
    int32_t slot[kPerThread];
#pragma unroll
    for (uint32_t k = 0; k < kPerThread; ++k) {
      const uint32_t i = k * kThreads + threadIdx.x;
      if (i < staged) {
        const unsigned long long entry = sm().staged[i];
        live |= 1u << k;
        strict |=
            static_cast<uint32_t>(
                __uint_as_float(static_cast<uint32_t>(entry >> 32)) >= s.hi)
            << k;
        slot[k] = static_cast<int32_t>(static_cast<uint32_t>(entry));
      }
    }
    uint32_t total;
    const uint32_t offset = block_exclusive_sum(
        __popc(strict) | (__popc(live ^ strict) << 16), sm().scan, total);
    if (threadIdx.x == 0) sm().reserved = before, sm().row_word = before + add;
    sync();
    before = sm().reserved;
    uint32_t strict_pos =
        (static_cast<uint32_t>(before >> 24) & 0xffffffu) + (offset & 0xffffu);
    uint32_t inside_pos =
        (static_cast<uint32_t>(before) & 0xffffffu) + (offset >> 16);
    const bool last = (before >> 48) + 1 == parts;
    unsigned long long* candidates =
        p.candidates + static_cast<size_t>(row) * p.capacity;
#pragma unroll
    for (uint32_t k = 0; k < kPerThread; ++k) {
      if (not(live >> k & 1)) continue;
      if (strict >> k & 1) {
        if (strict_pos < s.strict) out_row[strict_pos] = slot[k];
        ++strict_pos;
      } else {
        if (s.whole) {
          if (inside_pos < s.count) out_row[s.strict + inside_pos] = slot[k];
        } else if (inside_pos < p.capacity) {
          const auto bits = static_cast<uint32_t>(
              sm().staged[k * kThreads + threadIdx.x] >> 32);
          candidates[inside_pos] =
              composite(Cfg::key(__uint_as_float(bits)), slot[k]);
        }
        ++inside_pos;
      }
    }
    if (threadIdx.x == 0) sm().staged_count = 0, sm().strict_count = 0;
    // The finalizer reuses the stage's shared memory
    if (last) sync();
    // Each warp's writes land before its own completion signal
    __syncwarp();
    if (lane_id() == 0 and not last)
      signal_done(&p.state[row].done, kPartSignal / kWarps);
    return last;
  }

  // Publishes a one-tile part without staging: a block scan places every
  // thread's hits, one atomic reserves and arrives, and each thread writes its
  // own hits. Returns whether this part arrived last.
  static __device__ bool publish_tile(const P& p, const Split& s, uint32_t row,
                                      uint32_t parts, int32_t* out_row,
                                      const Vector (&v)[kVectors],
                                      uint32_t base, uint32_t end) {
    uint32_t hits = 0, strict = 0;
    if constexpr (kFp32) {
#pragma unroll
      for (uint32_t u = 0; u < kVectors; ++u) {
#pragma unroll
        for (uint32_t q = 0; q < 4; ++q) {
          hits |= static_cast<uint32_t>(Cfg::element(v[u], q) >= s.lo)
                  << (u * 4 + q);
          strict |= static_cast<uint32_t>(Cfg::element(v[u], q) >= s.hi)
                    << (u * 4 + q);
        }
      }
    } else {
      hits = tile_at_least(v, Cfg::threshold(s.lo));
      strict = tile_at_least(v, Cfg::threshold(s.hi));
    }
    const uint32_t live = live_bits(base, end);
    hits &= live, strict &= live;
    // A tile holds at most kTile hits of each kind: both counts fit 16 bits
    const uint32_t mine = __popc(strict) | (__popc(hits ^ strict) << 16);
    uint32_t total;
    const uint32_t offset = block_exclusive_sum(mine, sm().scan, total);

    if (threadIdx.x == 0) {
      const unsigned long long add =
          (1ull << 48) |
          (static_cast<unsigned long long>(total & 0xffffu) << 24) |
          (total >> 16);
      const unsigned long long before = arrive(&p.state[row].word, add);
      sm().reserved = before;
      sm().row_word = before + add;
    }
    sync();
    const unsigned long long before = sm().reserved;
    uint32_t strict_pos =
        (static_cast<uint32_t>(before >> 24) & 0xffffffu) + (offset & 0xffffu);
    uint32_t inside_pos =
        (static_cast<uint32_t>(before) & 0xffffffu) + (offset >> 16);
    const bool last = (before >> 48) + 1 == parts;
    const auto index = [&](uint32_t j) {
      return base + ((j / kPerVector) * kThreads + threadIdx.x) * kPerVector +
             (j % kPerVector);
    };
    for (uint32_t left = strict; left != 0; left &= left - 1, ++strict_pos) {
      if (strict_pos < s.strict)
        out_row[strict_pos] = static_cast<int32_t>(index(__ffs(left) - 1));
    }
    unsigned long long* candidates =
        p.candidates + static_cast<size_t>(row) * p.capacity;
    for (uint32_t left = hits ^ strict; left != 0;
         left &= left - 1, ++inside_pos) {
      const uint32_t j = __ffs(left) - 1;
      const int32_t slot = static_cast<int32_t>(index(j));
      if (s.whole) {
        if (inside_pos < s.count) out_row[s.strict + inside_pos] = slot;
      } else if (inside_pos < p.capacity) {
        candidates[inside_pos] = composite(Cfg::key(pick(v, j)), slot);
      }
    }
    __syncwarp();
    if (lane_id() == 0 and not last)
      signal_done(&p.state[row].done, kPartSignal / kWarps);
    return last;
  }

  // Exact top-`need` of `n` distinct composites, 8 bits per pass from the top
  // of the key. `load(i)` returns entry i and `emit(v)` receives each selected
  // entry once.
  template <typename Load, typename Emit>
  static __device__ void radix_select(uint32_t n, uint32_t need,
                                      const Load& load, const Emit& emit) {
    unsigned long long prefix = 0, mask = 0;
    for (int shift = 24 + Cfg::kKeyBits; shift >= 0; shift -= 8) {
      if (threadIdx.x < 256) sm().radix[threadIdx.x] = 0;
      sync();
      for (uint32_t i = threadIdx.x; i < n; i += kThreads) {
        const unsigned long long v = load(i);
        if ((v & mask) == prefix)
          atomicAdd(&sm().radix[(v >> shift) & 255], 1u);
      }
      sync();
      // Thread t holds digit 255 - t, so the prefix runs from the best digit
      // down
      const uint32_t count =
          threadIdx.x < 256 ? sm().radix[(255 - threadIdx.x) & 255] : 0u;
      uint32_t total;
      const uint32_t above = block_exclusive_sum(count, sm().scan, total);
      if (threadIdx.x < 256 and above < need and need <= above + count)
        sm().bin = 255 - threadIdx.x, sm().strict = above, sm().count = count;
      sync();
      const uint32_t digit = sm().bin, digit_above = sm().strict,
                     digit_count = sm().count;
      const bool take = digit_count == need - digit_above;
      for (uint32_t i = threadIdx.x; i < n; i += kThreads) {
        const unsigned long long v = load(i);
        if ((v & mask) == prefix) {
          const uint32_t d = (v >> shift) & 255;
          if (d > digit or (take and d == digit)) emit(v);
        }
      }
      if (take) return;
      need -= digit_above;
      prefix |= static_cast<unsigned long long>(digit) << shift;
      mask |= 0xffull << shift;
      sync();
    }
  }

  // Exact top-K of a whole row from its scores
  static __device__ void select_row_exact(const P& p, uint32_t row,
                                          uint32_t length, int32_t* out_row) {
    const Score* x = p.scores + static_cast<size_t>(row) * p.score_stride;
    if (threadIdx.x == 0) sm().emitted = 0;
    sync();
    const auto load = [&](uint32_t i) {
      return composite(Cfg::key(Cfg::load(x + i)), static_cast<int32_t>(i));
    };
    const auto emit = [&](unsigned long long v) {
      out_row[atomicAdd(&sm().emitted, 1u)] = composite_slot(v);
    };
    radix_select(length, kTopK, load, emit);
  }

  // Resolves the crossing bin from the row's candidates: a shared fine
  // histogram over their ordered keys finds the crossing fine bin, bins above
  // it are emitted and that bin is ranked exactly. Every candidate entry is
  // read (waiting for late writers) and returned to zero.
  static __device__ bool resolve_bin(const P& p, const Split& s, uint32_t row,
                                     int32_t* out_row) {
    constexpr uint32_t kHeld =
        4;  // candidates per thread kept in registers between the two passes
    unsigned long long* candidates =
        p.candidates + static_cast<size_t>(row) * p.capacity;
    const uint32_t need = kTopK - s.strict;
    unsigned long long held[kHeld];
#pragma unroll
    for (uint32_t k = 0; k < kHeld; ++k) {
      const uint32_t i = k * kThreads + threadIdx.x;
      held[k] = i < s.count ? load_relaxed(candidates + i) : 1ull;
    }
    if (s.count <= kRankCount) {
      LITETOPK_COV(kCovFinalSmall);
      // Small bin: every candidate ranks itself against the others; composites
      // are distinct
      if (threadIdx.x < s.count) {
        held[0] = await_entry(candidates + threadIdx.x, held[0]);
        candidates[threadIdx.x] = 0;
        sm().list[threadIdx.x] = held[0];
      }
      sync();
      if (threadIdx.x < s.count) {
        uint32_t rank = 0;
        for (uint32_t j = 0; j < s.count; ++j) rank += sm().list[j] > held[0];
        if (rank < need) out_row[s.strict + rank] = composite_slot(held[0]);
      }
      return true;
    }
    uint32_t* fine = sm().fine;
    for (uint32_t i = threadIdx.x; i < kFineBins; i += kThreads) fine[i] = 0;
    if (threadIdx.x == 0) sm().bin = -1, sm().emitted = 0, sm().listed = 0;
    sync();
    const auto fine_bin = [&](unsigned long long v) {
      return min((composite_key(v) - s.key_lo) >> s.shift, kFineBins - 1);
    };
#pragma unroll
    for (uint32_t k = 0; k < kHeld; ++k) {
      const uint32_t i = k * kThreads + threadIdx.x;
      if (i < s.count) {
        held[k] = await_entry(candidates + i, held[k]);
        candidates[i] = 0;
        atomicAdd(&fine[fine_bin(held[k])], 1u);
      }
    }
    for (uint32_t i = kHeld * kThreads + threadIdx.x; i < s.count;
         i += kThreads)
      atomicAdd(&fine[fine_bin(
                    await_entry(candidates + i, load_relaxed(candidates + i)))],
                1u);
    sync();
    // Thread t holds the fine bins just below kFineBins - kFinePerThread * t,
    // so the prefix runs from the best bin down
    uint32_t counts[kFinePerThread], sum = 0;
#pragma unroll
    for (uint32_t j = 0; j < kFinePerThread; ++j)
      sum += counts[j] = fine[kFineBins - 1 - kFinePerThread * threadIdx.x - j];
    uint32_t total;
    uint32_t acc = block_exclusive_sum(sum, sm().scan, total);
#pragma unroll
    for (uint32_t j = 0; j < kFinePerThread; ++j) {
      if (acc < need and need <= acc + counts[j])
        sm().bin = kFineBins - 1 - kFinePerThread * threadIdx.x - j,
        sm().strict = acc, sm().count = counts[j];
      acc += counts[j];
    }
    sync();
    const bool found = sm().bin >= 0;
    const uint32_t crossing = sm().bin, above = sm().strict,
                   in_bin = sm().count;
    const uint32_t rest = need - above;
    const bool take_bin = in_bin == rest;
    const auto place = [&](unsigned long long v) {
      const uint32_t fb = fine_bin(v);
      if (fb > crossing or (take_bin and fb == crossing)) {
        out_row[s.strict + atomicAdd(&sm().emitted, 1u)] = composite_slot(v);
      } else if (fb == crossing) {
        const uint32_t j = atomicAdd(&sm().listed, 1u);
        if (j < kStage) sm().list[j] = v;
      }
    };
    // The second pass reads the entries beyond the held ones again, then
    // restores them to zero
    for (uint32_t i = kHeld * kThreads + threadIdx.x; i < s.count;
         i += kThreads) {
      const unsigned long long v = load_relaxed(candidates + i);
      candidates[i] = 0;
      if (found) place(v);
    }
    if (not found) {
      LITETOPK_COV(kCovFinalFineFail);
      return false;
    }
#pragma unroll
    for (uint32_t k = 0; k < kHeld; ++k) {
      if (k * kThreads + threadIdx.x < s.count) place(held[k]);
    }
    sync();
    if (take_bin) {
      LITETOPK_COV(kCovFinalFineTake);
      return true;
    }
    if (sm().listed != in_bin or in_bin > kStage) {
      LITETOPK_COV(kCovFinalFineFail);
      return false;
    }
    int32_t* out_bin = out_row + s.strict + above;
    const unsigned long long* list = sm().list;
    if (in_bin <= kDirectRank) {
      LITETOPK_COV(kCovFinalFineDirect);
      // Composites are distinct, so ranks are a permutation of 0..in_bin-1
      if (threadIdx.x < in_bin) {
        const unsigned long long mine = list[threadIdx.x];
        uint32_t rank = 0;
        for (uint32_t j = 0; j < in_bin; ++j) rank += list[j] > mine;
        if (rank < rest) out_bin[rank] = composite_slot(mine);
      }
      return true;
    }
    LITETOPK_COV(kCovFinalFineRadix);
    if (threadIdx.x == 0) sm().emitted = 0;
    sync();
    radix_select(
        in_bin, rest, [&](uint32_t i) { return list[i]; },
        [&](unsigned long long v) {
          out_bin[atomicAdd(&sm().emitted, 1u)] = composite_slot(v);
        });
    return true;
  }

  // Waits until the row's other parts have finished writing
  static __device__ __forceinline__ void await_parts(const P& p, uint32_t row,
                                                     uint32_t parts) {
    if (threadIdx.x == 0) {
      const uint32_t base = p.state[row].done_base;
      while (load_acquire(&p.state[row].done) - base !=
             (parts - 1) * kPartSignal) {
      }
    }
    sync();
  }

  static __device__ void finalize_row(const P& p, const Split& s, uint32_t row,
                                      uint32_t length, uint32_t parts,
                                      int32_t* out_row) {
    const unsigned long long word = sm().row_word;
    const auto strict_total = static_cast<uint32_t>(word >> 24) & 0xffffffu;
    const auto inside_total = static_cast<uint32_t>(word) & 0xffffffu;
    bool ok = s.certified and strict_total == s.strict and
              inside_total == s.count and (s.whole or s.count <= p.capacity);
    const bool drained =
        ok and not s.whole;  // resolve_bin reads and zeroes every candidate
#ifdef LITETOPK_COVERAGE
    if (not s.certified) {
      LITETOPK_COV(kCovFinalUncertified);
    } else if (not(strict_total == s.strict and inside_total == s.count)) {
      LITETOPK_COV(kCovFinalMismatch);
    } else if (not ok) {
      LITETOPK_COV(kCovFinalCapacity);
    } else if (s.whole) {
      LITETOPK_COV(kCovFinalWhole);
    }
#endif
    if (drained) ok = resolve_bin(p, s, row, out_row);
    // Every thread has read its shared state; the fallback reuses it
    sync();
    if (not ok) {
      // Late writes of the other parts must land first: candidates are restored
      // to zero, outputs rewritten
      await_parts(p, row, parts);
      if (s.certified and not s.whole and not drained) {
        unsigned long long* candidates =
            p.candidates + static_cast<size_t>(row) * p.capacity;
        for (uint32_t i = threadIdx.x; i < min(inside_total, p.capacity);
             i += kThreads)
          candidates[i] = 0;
      }
      LITETOPK_COV(kCovFinalFallback);
      select_row_exact(p, row, length, out_row);
    }
    clear_histogram(p, row);
    if (threadIdx.x == 0) {
      // Every part has arrived, so the word can be reset; their `done` signals
      // may still be in flight
      p.state[row].word = 0;
      atomicAdd(
          &p.state[row].done_base,
          (parts - 1) *
              kPartSignal);  // no return value: a fire-and-forget reduction
      if constexpr (Cfg::kFused) {
        uint32_t* counter =
            reinterpret_cast<uint32_t*>(
                p.candidates + static_cast<size_t>(p.rows) * p.capacity) +
            2 * row;
        if (parts > 1) atomicAdd(counter + 1, parts);
      }
    }
  }

  // Scans [begin, end) with `v` holding the first tile. A tile's next loads
  // reuse its registers as soon as its scores are compared, so no copy waits on
  // them; only the lower edge of the crossing bin is tested here.
  static __device__ __forceinline__ void scan_segment(const Split& s,
                                                      const Score* x,
                                                      uint32_t begin,
                                                      uint32_t end,
                                                      Vector (&v)[kVectors]) {
    uint32_t strict_mine = 0;
    uint32_t lo2 = 0;
    if constexpr (not kFp32) lo2 = Cfg::threshold(s.lo);
    for (uint32_t base = begin; base < end; base += kTile) {
      uint32_t hits = 0;
      if constexpr (kFp32) {
#pragma unroll
        for (uint32_t u = 0; u < kVectors; ++u) {
#pragma unroll
          for (uint32_t q = 0; q < 4; ++q)
            hits |= static_cast<uint32_t>(Cfg::element(v[u], q) >= s.lo)
                    << (u * 4 + q);
        }
      } else {
        hits = tile_at_least(v, lo2);
      }
      if (base + kTile > end) hits &= live_bits(base, end);
      if (base + kTile < end) load_tile(v, x, base + kTile, end);
      if (__any_sync(0xffffffffu, hits != 0))
        stage_hits(x, hits, base, s.hi, strict_mine);
    }
    // The counts let publishing reserve before it has split the staged hits
    const uint32_t warp_strict = __reduce_add_sync(0xffffffffu, strict_mine);
    if (lane_id() == 0 and warp_strict != 0)
      atomicAdd(&sm().strict_count, warp_strict);
  }

  // derive_split for a local row: the warp totals' prefix comes from two warp
  // reductions instead of a second scan. Also zeroes the fine histogram (before
  // the split's barriers) and the local place counter, and waits for the row's
  // page slice before the second barrier (which then makes it visible to every
  // thread).
  static __device__ Split local_split(int2 bins, uint32_t length) {
    for (uint32_t i = threadIdx.x; i < kFineBins; i += kThreads)
      sm().fine[i] = 0;
    if (threadIdx.x == 0) sm().bin = -1, sm().staged_count = 0;
    const uint32_t a = min(static_cast<uint32_t>(bins.x), kMaxCount);
    const uint32_t b = min(static_cast<uint32_t>(bins.y), kMaxCount);
    const uint32_t lane = lane_id(), warp = threadIdx.x >> 5;
    const uint32_t inclusive = warp_inclusive_sum(a + b);
    if (lane == 31) sm().scan[warp] = inclusive;
    sync();
    const uint32_t warp_total = lane < kWarps ? sm().scan[lane] : 0u;
    const uint32_t total = __reduce_add_sync(0xffffffffu, warp_total);
    const uint32_t before =
        __reduce_add_sync(0xffffffffu, lane < warp ? warp_total : 0u) +
        inclusive - (a + b);
    // The crossing bin holds the K-th largest score: before < K <= before +
    // count
    if (total == length) {
      if (before < kTopK and kTopK <= before + a) {
        sm().bin = kCoarsePerThread * threadIdx.x, sm().strict = before,
        sm().count = a;
      } else if (before + a < kTopK and kTopK <= before + a + b) {
        sm().bin = kCoarsePerThread * threadIdx.x + 1, sm().strict = before + a,
        sm().count = b;
      }
    }

    sync();
    Split s;
    s.bin = sm().bin;
    s.certified = s.bin >= 0;
    s.strict = sm().strict;
    s.count = sm().count;
    s.whole = s.certified and s.count == kTopK - s.strict;
    const float nan = __int_as_float(0x7fc00000);
    s.lo = s.certified ? coarse_lower_edge(s.bin) : nan;
    s.hi = s.bin > 0 ? coarse_lower_edge(s.bin - 1) : nan;
    s.key_lo = Cfg::edge(s.lo);
    const uint32_t key_hi =
        s.bin > 0 ? Cfg::edge(s.hi) : Cfg::kKeyAboveInf;  // just above +inf
    const uint32_t width = key_hi - s.key_lo;
    s.shift = width > kFineBins ? 32 - __clz(width - 1) - kFineBits : 0;
    return s;
  }

  // FP16 bits of the lower edge of coarse bin t, for t in [207, 510] and [512,
  // 815]: positive bins hold the FP16-RN codes (511 - t) << 6 and up, negative
  // bins 512 + c the magnitudes up to (c << 6) | 63
  static __device__ __forceinline__ uint32_t local_half_edge(int32_t t) {
    return t < 512 ? static_cast<uint32_t>(511 - t) << 6
                   : 0x8000u | (static_cast<uint32_t>(t - 512) << 6) | 63u;
  }

  // Exact ranks of n <= kRankCount distinct composites list[0, n): an entry
  // whose rank (entries above it) is below `need` goes to out[rank]. Lanes hold
  // the entries as columns (a composite is never zero, so missing entries rank
  // nothing), warps take the entries in turn: one ballot per column instead of
  // n comparisons per entry.
  static __device__ __forceinline__ void local_ballot_rank(
      const unsigned long long* list, uint32_t n, uint32_t need, int32_t* out) {
    static_assert(kRankCount == 128, "four 32-entry columns");
    const uint32_t lane = lane_id();
    unsigned long long column[4];
#pragma unroll
    for (uint32_t c = 0; c < 4; ++c)
      column[c] = c * 32 + lane < n ? list[c * 32 + lane] : 0ull;
#pragma unroll 1
    for (uint32_t i = threadIdx.x >> 5; i < n; i += kWarps) {
      const unsigned long long target = list[i];
      uint32_t rank = 0;
#pragma unroll
      for (uint32_t c = 0; c < 4; ++c)
        rank += __popc(__ballot_sync(0xffffffffu, column[c] > target));
      if (lane == 0 and rank < need) out[rank] = composite_slot(target);
    }
  }

  // A row of at most one tile (Cfg::kLocalScores), all of it on this CTA: no
  // arrival atomic, no global candidates, no hand-off state. The scores stay in
  // registers (vectors past the row are skipped by the whole CTA); the split
  // comes from the histogram (`bins`, loaded together with the length) with two
  // barriers; the scores are classified with packed FP16 compares where both
  // edges of the crossing bin are FP16 code edges; each warp reserves the
  // places of its scores with one shared atomic: scores above the crossing bin
  // go straight to the output, a small bin is listed in shared memory and
  // ranked with warp ballots, a larger one is resolved through a fine histogram
  // over its keys. Counts that disagree with the histogram (or a bin beyond the
  // list) fall back to the exact select of the whole row.
  static __device__ void local_row(const P& p, uint32_t row, uint32_t length,
                                   int2 bins) {
    static_assert(Cfg::kLocalScores <= kTile,
                  "a local row is one tile of this CTA");
    LITETOPK_COV(kCovLocal);
    int32_t* out_row = p.out + static_cast<size_t>(row) * p.out_stride;
    const Score* x = p.scores + static_cast<size_t>(row) * p.score_stride;
    Vector v[kVectors] = {};
    load_tile(v, x, 0, length);  // in flight while the split is derived
    const Split s = local_split(bins, length);
    uint32_t hits = 0, strict = 0;
    if constexpr (kFp32) {
      if ((s.bin >= 208 and s.bin <= 510) or (s.bin >= 513 and s.bin <= 815)) {
        LITETOPK_COV(kCovLocalPacked);
        // Both edges of these bins are FP16-RN code edges: x >= edge <=>
        // half_rn(x) >= H (selector/v4/tools/ check_half_edges.py), so packed
        // FP16 compares classify two scores at a time
        const uint32_t lo2 = local_half_edge(s.bin) * 0x10001u,
                       hi2 = local_half_edge(s.bin - 1) * 0x10001u;
#pragma unroll
        for (uint32_t u = 0; u < kVectors; ++u) {
          if (u * kThreads * kPerVector <
              length) {  // the whole CTA skips vectors past the row
            const __half2 a = __floats2half2_rn(v[u].x, v[u].y),
                          b = __floats2half2_rn(v[u].z, v[u].w);
            const __half2 lo = *reinterpret_cast<const __half2*>(&lo2),
                          hi = *reinterpret_cast<const __half2*>(&hi2);
            // One flag byte per score (0xff / 0x00), then signed dot products
            // weight them 1, 2, 4, 8
            const uint32_t lo_flags =
                __byte_perm(__hge2_mask(a, lo), __hge2_mask(b, lo), 0x6420);
            const uint32_t hi_flags =
                __byte_perm(__hge2_mask(a, hi), __hge2_mask(b, hi), 0x6420);
            hits |=
                static_cast<uint32_t>(__dp4a(static_cast<int>(lo_flags),
                                             static_cast<int>(0xf8fcfeffu), 0))
                << (4 * u);
            strict |=
                static_cast<uint32_t>(__dp4a(static_cast<int>(hi_flags),
                                             static_cast<int>(0xf8fcfeffu), 0))
                << (4 * u);
          }
        }
      } else {
#pragma unroll
        for (uint32_t u = 0; u < kVectors; ++u) {
          if (u * kThreads * kPerVector < length) {
#pragma unroll
            for (uint32_t q = 0; q < 4; ++q) {
              hits |= static_cast<uint32_t>(Cfg::element(v[u], q) >= s.lo)
                      << (u * 4 + q);
              strict |= static_cast<uint32_t>(Cfg::element(v[u], q) >= s.hi)
                        << (u * 4 + q);
            }
          }
        }
      }
    } else {
      hits = tile_at_least(v, Cfg::threshold(s.lo));
      strict = tile_at_least(v, Cfg::threshold(s.hi));
    }
    const uint32_t live = live_bits(0, length);
    hits &= live, strict &= live;
    // Each warp reserves the places of its scores with one shared atomic;
    // places past the histogram's counts only arise from an inconsistent
    // histogram, which the counts check after the next barrier (writes made
    // before it are then overwritten by the exact select)
    const uint32_t mine = __popc(strict) | (__popc(hits ^ strict) << 16);
    const uint32_t warp_inclusive = warp_inclusive_sum(mine);
    uint32_t warp_base = 0;
    if (lane_id() == 31)
      warp_base = atomicAdd(&sm().staged_count, warp_inclusive);
    const uint32_t offset =
        __shfl_sync(0xffffffffu, warp_base, 31) + warp_inclusive - mine;
    bool done = false;
    if (s.certified) {
      const auto index = [&](uint32_t j) {
        return ((j / kPerVector) * kThreads + threadIdx.x) * kPerVector +
               (j % kPerVector);
      };
      uint32_t strict_pos = offset & 0xffffu, inside_pos = offset >> 16;
      for (uint32_t left = strict; left != 0; left &= left - 1, ++strict_pos) {
        if (strict_pos < s.strict)
          out_row[strict_pos] = static_cast<int32_t>(index(__ffs(left) - 1));
      }
      const uint32_t inside = hits ^ strict;
      const uint32_t need = kTopK - s.strict;
      unsigned long long* list = sm().list;
      const auto fine_bin = [&](uint32_t key) {
        return min((key - s.key_lo) >> s.shift, kFineBins - 1);
      };
      if (s.whole) {
        for (uint32_t left = inside; left != 0;
             left &= left - 1, ++inside_pos) {
          if (inside_pos < s.count)
            out_row[s.strict + inside_pos] =
                static_cast<int32_t>(index(__ffs(left) - 1));
        }
      } else if (s.count <= kRankCount) {
        // Small bin: listed at the reserved positions
        for (uint32_t left = inside; left != 0;
             left &= left - 1, ++inside_pos) {
          const uint32_t j = __ffs(left) - 1;
          if (inside_pos < kRankCount)
            list[inside_pos] =
                composite(Cfg::key(pick(v, j)), static_cast<int32_t>(index(j)));
        }
      } else {
        // Larger bin: counted into the fine histogram (zeroed by local_split)
        for (uint32_t left = inside; left != 0; left &= left - 1)
          atomicAdd(&sm().fine[fine_bin(Cfg::key(pick(v, __ffs(left) - 1)))],
                    1u);
      }
      if (threadIdx.x == 0) sm().emitted = 0, sm().listed = 0;
      sync();
      // The same in every thread: the counts must be the histogram's
      const uint32_t placed = sm().staged_count;
      const bool ok =
          (placed & 0xffffu) == s.strict and (placed >> 16) == s.count;
      if (not ok) {
      } else if (s.whole) {
        LITETOPK_COV(kCovLocalWhole);
        done = true;
      } else if (s.count <= kRankCount) {
        LITETOPK_COV(kCovLocalSmall);
        local_ballot_rank(list, s.count, need, out_row + s.strict);
        done = true;
      } else {
        // Larger bin, as resolve_bin but from the registers: the fine histogram
        // over the ordered keys finds the crossing fine bin, fine bins above it
        // go to the output, that bin is listed and ranked exactly
        uint32_t* fine = sm().fine;
        // Thread t holds the fine bins just below kFineBins - kFinePerThread *
        // t: the prefix runs from the best bin down
        uint32_t counts[kFinePerThread], sum = 0;
#pragma unroll
        for (uint32_t j = 0; j < kFinePerThread; ++j)
          sum += counts[j] =
              fine[kFineBins - 1 - kFinePerThread * threadIdx.x - j];
        if (threadIdx.x == 0) sm().bin = -1;
        const uint32_t lane = lane_id(), warp = threadIdx.x >> 5;
        const uint32_t fine_inclusive = warp_inclusive_sum(sum);
        if (lane == 31) sm().scan[warp] = fine_inclusive;
        sync();
        const uint32_t warp_total = lane < kWarps ? sm().scan[lane] : 0u;
        uint32_t acc =
            __reduce_add_sync(0xffffffffu, lane < warp ? warp_total : 0u) +
            fine_inclusive - sum;
#pragma unroll
        for (uint32_t j = 0; j < kFinePerThread; ++j) {
          if (acc < need and need <= acc + counts[j])
            sm().bin = kFineBins - 1 - kFinePerThread * threadIdx.x - j,
            sm().strict = acc, sm().count = counts[j];
          acc += counts[j];
        }
        sync();
        const bool found = sm().bin >= 0;
        const uint32_t crossing = sm().bin, above = sm().strict,
                       in_bin = sm().count;
        const uint32_t rest = need - above;
        const bool take_bin = in_bin == rest;
        if (found) {
          for (uint32_t left = inside; left != 0; left &= left - 1) {
            const uint32_t j = __ffs(left) - 1;
            const uint32_t key = Cfg::key(pick(v, j));
            const uint32_t fb = fine_bin(key);
            if (fb > crossing or (take_bin and fb == crossing)) {
              out_row[s.strict + atomicAdd(&sm().emitted, 1u)] =
                  static_cast<int32_t>(index(j));
            } else if (fb == crossing) {
              const uint32_t at = atomicAdd(&sm().listed, 1u);
              if (at < kStage)
                list[at] = composite(key, static_cast<int32_t>(index(j)));
            }
          }
          sync();
          done = take_bin;
          if (take_bin) LITETOPK_COV(kCovLocalFineTake);
          if (not take_bin and sm().listed == in_bin and in_bin <= kStage) {
            int32_t* out_bin = out_row + s.strict + above;
            if (in_bin <= kRankCount) {
              LITETOPK_COV(kCovLocalFineBallot);
              local_ballot_rank(list, in_bin, rest, out_bin);
            } else if (in_bin <= kDirectRank) {
              LITETOPK_COV(kCovLocalFineDirect);
              if (threadIdx.x < in_bin) {
                const unsigned long long mine = list[threadIdx.x];
                uint32_t rank = 0;
                for (uint32_t j = 0; j < in_bin; ++j) rank += list[j] > mine;
                if (rank < rest) out_bin[rank] = composite_slot(mine);
              }
            } else {
              LITETOPK_COV(kCovLocalFineRadix);
              if (threadIdx.x == 0) sm().emitted = 0;
              sync();
              radix_select(
                  in_bin, rest, [&](uint32_t i) { return list[i]; },
                  [&](unsigned long long c) {
                    out_bin[atomicAdd(&sm().emitted, 1u)] = composite_slot(c);
                  });
            }
            done = true;
          }
        }
      }
    }
    if (not done) {
      // A histogram that disagrees with the scores, or a bin beyond the list:
      // exact select of the whole row
      LITETOPK_COV(kCovLocalFallback);
      sync();
      select_row_exact(p, row, length, out_row);
    }
    clear_histogram(p, row);
    sync();  // the next row of this CTA reuses the shared memory
  }

  // Scores [begin, end) of a row, part `part` of its `parts`; `early` holds the
  // row's histogram bins when the caller loaded them already (`use_early`)
  static __device__ __forceinline__ void process_segment(
      const P& p, uint32_t row, uint32_t length, uint32_t begin, uint32_t end,
      uint32_t part, uint32_t parts, int2 early = make_int2(0, 0),
      bool use_early = false) {
    int32_t* out_row = p.out + static_cast<size_t>(row) * p.out_stride;
    if (length <= kTopK) {
      // Every live score is selected: no split, no hand-off. Part 0 pads and
      // clears the histogram.
      if (part == 0) LITETOPK_COV(kCovShort);
      for (uint32_t i = begin + threadIdx.x; i < end; i += kThreads)
        out_row[i] = static_cast<int32_t>(i);
      if (part == 0) {
        for (uint32_t i = length + threadIdx.x; i < kTopK; i += kThreads)
          out_row[i] = -1;
        clear_histogram(p, row);
      }
      return;
    }
    const Score* x = p.scores + static_cast<size_t>(row) * p.score_stride;
    Vector current[kVectors] = {};
    load_tile(current, x, begin, end);  // in flight while the split is derived
    int2 bins;
    if constexpr (Cfg::kFused) {
      count_segment(x, begin, end, current);
      bins = fused_bins(p, row, parts);
    } else {
      bins = use_early ? early : load_histogram(p, row);
    }
    const Split s = derive_split(bins, length);
    bool last;
    if (s.certified and end - begin <= kTile) {
      LITETOPK_COV(kThreads == 512 ? kCovTile512 : kCovTile1024);
      last = publish_tile(p, s, row, parts, out_row, current, begin, end);
    } else {
      LITETOPK_COV(kThreads == 512 ? kCovSegment512 : kCovSegment1024);
      if (s.certified) scan_segment(s, x, begin, end, current);
      last = publish_segment(p, s, row, parts, out_row);
    }
    if (last) {
      finalize_row(p, s, row, length, parts, out_row);
      sync();
    }
    sync();
  }

  // Part `part` of `parts` of a row, scores [begin, end)
  static __device__ void run_part(const P& p, uint32_t row, uint32_t length,
                                  uint32_t begin, uint32_t end, uint32_t part,
                                  uint32_t parts, int2 early = make_int2(0, 0),
                                  bool use_early = false) {
    if (threadIdx.x == 0)
      sm().staged_count = 0, sm().strict_count = 0, sm().reserved = 0;
    sync();
    process_segment(p, row, length, begin, end, part, parts, early, use_early);
  }

  // One launch's work for this CTA. Up to one row per CTA, each row owns
  // gridDim / rows CTA slots and uses as many as its own length fills; slots
  // are numbered part-major, so the live parts of all rows take distinct SMs
  // before any SM hosts two. More rows are taken whole, round robin.
  static __device__ void run(const P& p) {
    const uint32_t slots = max(gridDim.x / p.rows, 1u);
    const uint32_t part = blockIdx.x / p.rows;
    // Slots no row could fill leave before reading the length words
    if (part >= parts_for<Cfg>(slots, p.score_width)) return;
    for (uint32_t row = blockIdx.x % p.rows; row < p.rows; row += gridDim.x) {
      if constexpr (Cfg::kLocalScores != 0) {
        // The row's histogram is loaded together with its length (its address
        // does not depend on the length), for a local row and for every part of
        // a longer one
        const int2 bins = load_histogram(p, row);
        const uint32_t length = row_length(p, row);
        const uint32_t parts = parts_for<Cfg>(slots, length);
        if (part < parts) {
          if (length > kTopK and length <= Cfg::kLocalScores) {
            local_row(p, row, length, bins);
            continue;
          }
          const uint2 seg = segment<Cfg>(length, part, parts);
          run_part(p, row, length, seg.x, seg.y, part, parts, bins, true);
        }
      } else {
        const uint32_t length = row_length(p, row);
        const uint32_t parts = parts_for<Cfg>(slots, length);
        if (part < parts) {
          const uint2 seg = segment<Cfg>(length, part, parts);
          run_part(p, row, length, seg.x, seg.y, part, parts);
        }
      }
    }
  }
};

// kMinBlocks 512-thread CTAs per SM: one CTA per SM may use twice the
// registers, which spares the spills
template <typename Cfg, uint32_t kMinBlocks>
__global__ void __launch_bounds__(512, kMinBlocks)
    select_kernel(const __grid_constant__ Params<typename Cfg::Score> p) {
  // Every reader waits before any histogram, score, length or persistent-state
  // access.
  asm volatile("griddepcontrol.wait;" ::: "memory");
  // Dependents may launch now; they wait for this grid's completion before
  // reading its outputs
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  Selector<Cfg, 512>::run(p);
}

// Measured on B200 (FP32): parts over 12 tiles of a 512-thread CTA stream
// faster with 32 warps; shorter ones are dominated by fixed per-part steps that
// more warps only slow down. A BF16 tile has the same bytes.
constexpr uint32_t kWideTiles = 12;

// Rows 16..63 on one 1024-thread CTA per SM, in the slots Selector<Cfg,
// 512>::run gives them. A long part is scanned by all 32 warps; a short one
// leaves the upper half idle. The narrow path comes first: its code is the hot
// one.
template <typename Cfg>
__global__ void __launch_bounds__(1024, 1)
    select_kernel_wide(const __grid_constant__ Params<typename Cfg::Score> p) {
  // Every reader waits before any histogram, score, length or persistent-state
  // access.
  asm volatile("griddepcontrol.wait;" ::: "memory");
  // Dependents may launch now; they wait for this grid's completion before
  // reading its outputs
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  const uint32_t slots = gridDim.x / p.rows;
  const uint32_t row = blockIdx.x % p.rows, part = blockIdx.x / p.rows;
  if (part >= parts_for<Cfg>(slots, p.score_width)) return;
  int2 bins = make_int2(0, 0);
  if constexpr (Cfg::kLocalScores != 0) {
    // A local row's histogram is loaded together with its length (see
    // Selector::run)
    if (part == 0 and threadIdx.x < 512)
      bins = Selector<Cfg, 512>::load_histogram(p, row);
  }
  const uint32_t length = row_length(p, row);
  const uint32_t parts = parts_for<Cfg>(slots, length);
  if (part >= parts) return;
  const uint2 seg = segment<Cfg>(length, part, parts);
  if constexpr (Cfg::kLocalScores != 0) {
    if (length > Cfg::kTopK and length <= Cfg::kLocalScores) {
      if (threadIdx.x < 512)
        Selector<Cfg, 512>::local_row(p, row, length, bins);
      return;
    }
  }
  const bool wide = __builtin_expect(
      length > Cfg::kTopK and
          seg.y - seg.x > kWideTiles * Selector<Cfg, 512>::kTile,
      0);
  if (not wide) {
    if (threadIdx.x < 512)
      Selector<Cfg, 512>::run_part(p, row, length, seg.x, seg.y, part, parts);
  } else {
    Selector<Cfg, 1024>::run_part(p, row, length, seg.x, seg.y, part, parts);
  }
}

}  // namespace litetopk
