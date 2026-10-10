// SPDX-License-Identifier: Apache-2.0
// Exact FP32 top-2048 decode selection driven by DeepGEMM's coarse score
// histogram.
//
// The producer counts every live score of a row into 1024 ordered bins (bin 0
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
//   few, otherwise through a 2048-bin
//      fine histogram over their ordered keys and an exact rank of the crossing
//      fine bin. Any count mismatch (a stale histogram, candidate overflow)
//      falls back to an exact radix select over the whole row, so the result
//      never depends on the histogram being right.
// The histogram, the hand-off words and the candidate buffers are returned to
// zero.

#include "coarse_bins.cuh"
#include <algorithm>
#include <cstdint>

namespace litetopk {

constexpr uint32_t kTopK = 2048;
constexpr uint32_t kCoarseBins = 1024;
constexpr uint32_t kFineBins = 2048;
constexpr uint32_t kFineBits = 11;
constexpr uint32_t kUnit = 256;           // partition granularity in scores
constexpr uint32_t kVectors = 4;          // float4 loads per thread per tile
constexpr uint32_t kStage = 2048;         // the crossing fine bin's list
constexpr uint32_t kStaged = 2 * kStage;  // hits a CTA stages per segment
constexpr uint32_t kPageBits = 6;
constexpr uint32_t kMinPartUnits =
    8;  // measured: parts under 2048 scores only add hand-off traffic
constexpr uint32_t kRankCount =
    128;  // crossing bins up to this size are ranked directly (quadratic, one
          // barrier)
constexpr uint32_t kDirectRank =
    512;  // crossing fine bins up to this size are ranked directly, whatever
          // the CTA width
constexpr uint32_t kMaxCount =
    1u << 21;  // above any live row length: bounds the histogram sums

static_assert(kFineBins == 1u << kFineBits);
static_assert(kStaged >= kTopK,
              "a CTA's scores above the crossing bin always fit its stage");

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

struct Params {
  const float* scores;
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

// Kept in L1 so staging can read hits back; evict-first keeps the stream from
// displacing other lines. Measured: no-allocate slows staging, plain allocation
// slows long DRAM-bound rows. PDL may start this grid before the producer's
// writes are visible. Ordinary global loads after the entry wait keep both the
// vector scan and scalar fallback in the generic memory proxy; read-only cache
// loads cannot do that.
__device__ __forceinline__ float load_score(const float* ptr) {
  float value;
  asm volatile("ld.global.f32 %0, [%1];" : "=f"(value) : "l"(ptr) : "memory");
  return value;
}

__device__ __forceinline__ float4 load_scores(const float* ptr) {
  float4 v;
  asm volatile("ld.global.L1::evict_first.v4.f32 {%0, %1, %2, %3}, [%4];"
               : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
               : "l"(ptr)
               : "memory");
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

__device__ __forceinline__ float element(const float4& v, uint32_t q) {
  return q == 0 ? v.x : q == 1 ? v.y : q == 2 ? v.z : v.w;
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

__device__ __forceinline__ uint32_t row_length(const Params& p, uint32_t row) {
  return min(static_cast<uint32_t>(max(p.lengths[row], 0)), p.score_width);
}

// Parts a row of `length` scores uses of its `slots`: none shorter than
// kMinPartUnits units
__device__ __forceinline__ uint32_t parts_for(uint32_t slots, uint32_t length) {
  return min(slots, max(1u, (length + kUnit - 1) / kUnit / kMinPartUnits));
}

// Score range [x, y) of part `part` of `parts`: the parts share the row's
// kUnit-score units evenly
__device__ __forceinline__ uint2 segment(uint32_t length, uint32_t part,
                                         uint32_t parts) {
  const uint32_t units = (length + kUnit - 1) / kUnit;
  return make_uint2(part * units / parts * kUnit,
                    min((part + 1) * units / parts * kUnit, length));
}

extern __shared__ __align__(16) unsigned char dynamic_smem[];

// Everything that depends on the CTA's thread count. A 1024-thread CTA runs
// either Selector<1024> or, with its upper half exited, Selector<512>: barriers
// count the participating threads explicitly.
template <uint32_t kThreads>
struct Selector {
  static constexpr uint32_t kWarps = kThreads / 32;
  static constexpr uint32_t kTile = kThreads * kVectors * 4;
  static constexpr uint32_t kCoarsePerThread =
      kCoarseBins / kThreads;                                       // 2 or 1
  static constexpr uint32_t kFinePerThread = kFineBins / kThreads;  // 4 or 2
  static_assert(kCoarseBins % kThreads == 0 and kFineBins % kThreads == 0 and
                kStaged % kThreads == 0);
  static_assert(kPartSignal % kWarps == 0);

  struct Shared {
    uint32_t scan[kWarps];
    uint32_t radix[256];
    union {
      unsigned long long
          staged[kStaged];       // (score bits, index) of the segment's hits
      uint32_t fine[kFineBins];  // the crossing bin's fine histogram
      unsigned long long list[kStage];  // the crossing fine bin's composites
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
  static __device__ __forceinline__ void load_tile(float4 (&v)[kVectors],
                                                   const float* row,
                                                   uint32_t base,
                                                   uint32_t end) {
    if (base + kTile <= end) {
#pragma unroll
      for (uint32_t u = 0; u < kVectors; ++u)
        v[u] = load_scores(row + base + (u * kThreads + threadIdx.x) * 4);
      return;
    }
#pragma unroll
    for (uint32_t u = 0; u < kVectors; ++u) {
      const uint32_t first = base + (u * kThreads + threadIdx.x) * 4;
      if (first < end) v[u] = load_scores(row + first);
    }
  }

  // Bit u * 4 + q is set when score q of this thread's vector u lies before
  // `end`
  static __device__ __forceinline__ uint32_t live_bits(uint32_t base,
                                                       uint32_t end) {
    uint32_t bits = 0;
#pragma unroll
    for (uint32_t u = 0; u < kVectors; ++u) {
      const uint32_t first = base + (u * kThreads + threadIdx.x) * 4;
      const uint32_t live = end > first ? min(end - first, 4u) : 0u;
      bits |= ((1u << live) - 1) << (u * 4);
    }
    return bits;
  }

  // This thread's coarse bins (kCoarsePerThread of them, in .x then .y)
  static __device__ __forceinline__ int2 load_histogram(const Params& p,
                                                        uint32_t row) {
    const int32_t* bins = p.histogram + static_cast<size_t>(row) * kCoarseBins;
    if constexpr (kCoarsePerThread == 2)
      return __ldcg(reinterpret_cast<const int2*>(bins) + threadIdx.x);
    return make_int2(__ldcg(bins + threadIdx.x), 0);
  }

  static __device__ __forceinline__ void clear_histogram(const Params& p,
                                                         uint32_t row) {
    int32_t* bins = p.histogram + static_cast<size_t>(row) * kCoarseBins;
    if constexpr (kCoarsePerThread == 2) {
      reinterpret_cast<int2*>(bins)[threadIdx.x] = make_int2(0, 0);
    } else {
      bins[threadIdx.x] = 0;
    }
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
    s.key_lo = edge_key(s.lo);
    const uint32_t key_hi =
        s.bin > 0 ? edge_key(s.hi) : 0xff800001u;  // just above +inf
    const uint32_t width = key_hi - s.key_lo;
    s.shift = width > kFineBins ? 32 - __clz(width - 1) - kFineBits : 0;
    return s;
  }

  // Score j of this thread's tile vectors, without indexing registers
  // dynamically
  static __device__ __forceinline__ float pick(const float4 (&v)[kVectors],
                                               uint32_t j) {
    static_assert(kVectors == 4);
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
  }

  // Stages one warp's hits of a tile in shared memory as (score bits, index)
  // and counts this thread's hits above the crossing bin. `hits` marks this
  // thread's scores >= lo; their scores are read back from L1, where the tile's
  // loads left them.
  static __device__ __forceinline__ void stage_hits(const float* x,
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
          base + ((j >> 2) * kThreads + threadIdx.x) * 4 + (j & 3);
      const float score = load_score(x + i);
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
  static __device__ bool publish_segment(const Params& p, const Split& s,
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
              composite(ordered_key(__uint_as_float(bits)), slot[k]);
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
  static __device__ bool publish_tile(const Params& p, const Split& s,
                                      uint32_t row, uint32_t parts,
                                      int32_t* out_row,
                                      const float4 (&v)[kVectors],
                                      uint32_t base, uint32_t end) {
    uint32_t hits = 0, strict = 0;
#pragma unroll
    for (uint32_t u = 0; u < kVectors; ++u) {
#pragma unroll
      for (uint32_t q = 0; q < 4; ++q) {
        hits |= static_cast<uint32_t>(element(v[u], q) >= s.lo) << (u * 4 + q);
        strict |= static_cast<uint32_t>(element(v[u], q) >= s.hi)
                  << (u * 4 + q);
      }
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
      return base + ((j >> 2) * kThreads + threadIdx.x) * 4 + (j & 3);
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
        candidates[inside_pos] = composite(ordered_key(pick(v, j)), slot);
      }
    }
    __syncwarp();
    if (lane_id() == 0 and not last)
      signal_done(&p.state[row].done, kPartSignal / kWarps);
    return last;
  }

  // Exact top-`need` of `n` distinct composites, 8 bits per pass from the top.
  // `load(i)` returns entry i and `emit(v)` receives each selected entry once.
  template <typename Load, typename Emit>
  static __device__ void radix_select(uint32_t n, uint32_t need,
                                      const Load& load, const Emit& emit) {
    unsigned long long prefix = 0, mask = 0;
    for (int shift = 56; shift >= 0; shift -= 8) {
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
  static __device__ void select_row_exact(const Params& p, uint32_t row,
                                          uint32_t length, int32_t* out_row) {
    const float* x = p.scores + static_cast<size_t>(row) * p.score_stride;
    if (threadIdx.x == 0) sm().emitted = 0;
    sync();
    const auto load = [&](uint32_t i) {
      return composite(ordered_key(load_score(x + i)), static_cast<int32_t>(i));
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
  static __device__ bool resolve_bin(const Params& p, const Split& s,
                                     uint32_t row, int32_t* out_row) {
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
    // Thread t holds the fine bins just below 2048 - kFinePerThread * t, so the
    // prefix runs from the best bin down
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
    if (not found) return false;
#pragma unroll
    for (uint32_t k = 0; k < kHeld; ++k) {
      if (k * kThreads + threadIdx.x < s.count) place(held[k]);
    }
    sync();
    if (take_bin) return true;
    if (sm().listed != in_bin or in_bin > kStage) return false;
    int32_t* out_bin = out_row + s.strict + above;
    const unsigned long long* list = sm().list;
    if (in_bin <= kDirectRank) {
      // Composites are distinct, so ranks are a permutation of 0..in_bin-1
      if (threadIdx.x < in_bin) {
        const unsigned long long mine = list[threadIdx.x];
        uint32_t rank = 0;
        for (uint32_t j = 0; j < in_bin; ++j) rank += list[j] > mine;
        if (rank < rest) out_bin[rank] = composite_slot(mine);
      }
      return true;
    }
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
  static __device__ __forceinline__ void await_parts(const Params& p,
                                                     uint32_t row,
                                                     uint32_t parts) {
    if (threadIdx.x == 0) {
      const uint32_t base = p.state[row].done_base;
      while (load_acquire(&p.state[row].done) - base !=
             (parts - 1) * kPartSignal) {
      }
    }
    sync();
  }

  static __device__ void finalize_row(const Params& p, const Split& s,
                                      uint32_t row, uint32_t length,
                                      uint32_t parts, int32_t* out_row) {
    const unsigned long long word = sm().row_word;
    const auto strict_total = static_cast<uint32_t>(word >> 24) & 0xffffffu;
    const auto inside_total = static_cast<uint32_t>(word) & 0xffffffu;
    bool ok = s.certified and strict_total == s.strict and
              inside_total == s.count and (s.whole or s.count <= p.capacity);
    const bool drained =
        ok and not s.whole;  // resolve_bin reads and zeroes every candidate
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
    }
  }

  // Scans [begin, end) with `v` holding the first tile. A tile's next loads
  // reuse its registers as soon as its scores are compared, so no copy waits on
  // them; only the lower edge of the crossing bin is tested here.
  static __device__ __forceinline__ void scan_segment(const Split& s,
                                                      const float* x,
                                                      uint32_t begin,
                                                      uint32_t end,
                                                      float4 (&v)[kVectors]) {
    uint32_t strict_mine = 0;
    for (uint32_t base = begin; base < end; base += kTile) {
      uint32_t hits = 0;
#pragma unroll
      for (uint32_t u = 0; u < kVectors; ++u) {
#pragma unroll
        for (uint32_t q = 0; q < 4; ++q)
          hits |= static_cast<uint32_t>(element(v[u], q) >= s.lo)
                  << (u * 4 + q);
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

  // Scores [begin, end) of a row, part `part` of its `parts`
  static __device__ __forceinline__ void process_segment(
      const Params& p, uint32_t row, uint32_t length, uint32_t begin,
      uint32_t end, uint32_t part, uint32_t parts) {
    int32_t* out_row = p.out + static_cast<size_t>(row) * p.out_stride;
    if (length <= kTopK) {
      // Every live score is selected: no split, no hand-off. Part 0 pads and
      // clears the histogram.
      for (uint32_t i = begin + threadIdx.x; i < end; i += kThreads)
        out_row[i] = static_cast<int32_t>(i);
      if (part == 0) {
        for (uint32_t i = length + threadIdx.x; i < kTopK; i += kThreads)
          out_row[i] = -1;
        clear_histogram(p, row);
      }
      return;
    }
    const float* x = p.scores + static_cast<size_t>(row) * p.score_stride;
    float4 current[kVectors] = {};
    load_tile(current, x, begin, end);  // in flight while the split is derived
    const Split s = derive_split(load_histogram(p, row), length);
    bool last;
    if (s.certified and end - begin <= kTile) {
      last = publish_tile(p, s, row, parts, out_row, current, begin, end);
    } else {
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
  static __device__ void run_part(const Params& p, uint32_t row,
                                  uint32_t length, uint32_t begin, uint32_t end,
                                  uint32_t part, uint32_t parts) {
    if (threadIdx.x == 0)
      sm().staged_count = 0, sm().strict_count = 0, sm().reserved = 0;
    sync();
    process_segment(p, row, length, begin, end, part, parts);
  }

  // One launch's work for this CTA. Up to one row per CTA, each row owns
  // gridDim / rows CTA slots and uses as many as its own length fills; slots
  // are numbered part-major, so the live parts of all rows take distinct SMs
  // before any SM hosts two. More rows are taken whole, round robin.
  static __device__ void run(const Params& p) {
    const uint32_t slots = max(gridDim.x / p.rows, 1u);
    const uint32_t part = blockIdx.x / p.rows;
    // Slots no row could fill leave before reading the length words
    if (part >= parts_for(slots, p.score_width)) return;
    for (uint32_t row = blockIdx.x % p.rows; row < p.rows; row += gridDim.x) {
      const uint32_t length = row_length(p, row);
      const uint32_t parts = parts_for(slots, length);
      if (part < parts) {
        const uint2 seg = segment(length, part, parts);
        run_part(p, row, length, seg.x, seg.y, part, parts);
      }
    }
  }
};

// kMinBlocks 512-thread CTAs per SM: one CTA per SM may use twice the
// registers, which spares the spills
template <uint32_t kMinBlocks>
__global__ void __launch_bounds__(512, kMinBlocks)
    select_kernel(const __grid_constant__ Params p) {
  // Every reader waits before any histogram, score, length or persistent-state
  // access.
  asm volatile("griddepcontrol.wait;" ::: "memory");
  // Dependents may launch now; they wait for this grid's completion before
  // reading its outputs
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  Selector<512>::run(p);
}

// Measured on B200: parts over 12 tiles of a 512-thread CTA stream faster with
// 32 warps; shorter ones are dominated by fixed per-part steps that more warps
// only slow down
constexpr uint32_t kWideTiles = 12;

// Rows 16..63 on one 1024-thread CTA per SM, in the slots Selector<512>::run
// gives them. A long part is scanned by all 32 warps; a short one leaves the
// upper half idle. The narrow path comes first: its code is the hot one.
__global__ void __launch_bounds__(1024, 1)
    select_kernel_wide(const __grid_constant__ Params p) {
  // Every reader waits before any histogram, score, length or persistent-state
  // access.
  asm volatile("griddepcontrol.wait;" ::: "memory");
  // Dependents may launch now; they wait for this grid's completion before
  // reading its outputs
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  const uint32_t slots = gridDim.x / p.rows;
  const uint32_t row = blockIdx.x % p.rows, part = blockIdx.x / p.rows;
  if (part >= parts_for(slots, p.score_width)) return;
  const uint32_t length = row_length(p, row);
  const uint32_t parts = parts_for(slots, length);
  if (part >= parts) return;
  const uint2 seg = segment(length, part, parts);
  const bool wide = __builtin_expect(
      length > kTopK and seg.y - seg.x > kWideTiles * Selector<512>::kTile, 0);
  if (not wide) {
    if (threadIdx.x < 512)
      Selector<512>::run_part(p, row, length, seg.x, seg.y, part, parts);
  } else {
    Selector<1024>::run_part(p, row, length, seg.x, seg.y, part, parts);
  }
}

}  // namespace litetopk
