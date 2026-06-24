/*
 * Shared 4096-bin single-CTA TopK helpers.
 */

#ifndef TOPK_HISTOGRAM_4096_CUH_
#define TOPK_HISTOGRAM_4096_CUH_

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace vllm {
namespace topk_histogram_4096 {

constexpr uint32_t kBlockSize = 1024;
constexpr uint32_t RADIX = 256;
constexpr uint32_t kMaxTies = 1024;
static_assert(kMaxTies <= kBlockSize,
              "tie_handle requires kMaxTies <= kBlockSize");
constexpr uint32_t kWarpSize = 32;
constexpr uint32_t kNumWarps = kBlockSize / kWarpSize;

// Register path
constexpr uint32_t kHist4096VecsPerThread = 4;
constexpr uint32_t kHist4096MaxLen =
    kHist4096VecsPerThread * 4 * kBlockSize;  // 16384

struct alignas(16) MatchBin {
  uint32_t bin, above_count, equal_count;
};
struct alignas(8) Tie {
  uint32_t idx;
  float score;
};

__device__ __forceinline__ void load_float4_predicated(const float* ptr,
                                                       int base, int seq_len,
                                                       float& v0, float& v1,
                                                       float& v2, float& v3) {
  uint32_t r0, r1, r2, r3;
  const int p0 = (base < seq_len);
  const int p1 = (base + 1 < seq_len);
  const int p2 = (base + 2 < seq_len);
  const int p3 = (base + 3 < seq_len);
  asm volatile(
      "{\n"
      "  .reg .pred pr0, pr1, pr2, pr3;\n"
      "  setp.ne.u32 pr0, %4, 0;\n"
      "  setp.ne.u32 pr1, %5, 0;\n"
      "  setp.ne.u32 pr2, %6, 0;\n"
      "  setp.ne.u32 pr3, %7, 0;\n"
      "  mov.u32 %0, 0xFF800000;\n"
      "  mov.u32 %1, 0xFF800000;\n"
      "  mov.u32 %2, 0xFF800000;\n"
      "  mov.u32 %3, 0xFF800000;\n"
      "  @pr0 ld.global.cg.u32 %0, [%8];\n"
      "  @pr1 ld.global.cg.u32 %1, [%8+4];\n"
      "  @pr2 ld.global.cg.u32 %2, [%8+8];\n"
      "  @pr3 ld.global.cg.u32 %3, [%8+12];\n"
      "}\n"
      : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
      : "r"(p0), "r"(p1), "r"(p2), "r"(p3), "l"(ptr));
  v0 = __uint_as_float(r0);
  v1 = __uint_as_float(r1);
  v2 = __uint_as_float(r2);
  v3 = __uint_as_float(r3);
}

// converts the float32 score to a 32-bit ordered unsigned integer — the full
// precision key for radix sorting
__device__ __forceinline__ auto convert_to_uint32_v2(float x) -> uint32_t {
  uint32_t bits = __float_as_uint(x);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

// Converts each score to a 12-bit bin (FP16 sign-magnitude -> top 12 bits ->
// bin 0-4095)
template <uint32_t kBits>
__device__ __forceinline__ uint32_t extract_coarse_bin_N(float x) {
  __half h = __float2half_rn(x);
  uint16_t bits = __half_as_ushort(h);
  uint16_t key = (bits & 0x8000) ? static_cast<uint16_t>(~bits)
                                 : static_cast<uint16_t>(bits | 0x8000);
  return key >> (16 - kBits);
}

// running sum within each warp — thread 0 gets its own value, thread 1 gets
// thread 0 + thread 1, thread 2 gets threads 0+1+2, etc.
__device__ __forceinline__ uint32_t warp_inclusive_sum(uint32_t lane,
                                                       uint32_t v) {
#pragma unroll
  for (uint32_t o = 1; o < 32; o *= 2) {
    uint32_t n = __shfl_up_sync(0xFFFFFFFF, v, o);
    if (lane >= o) v += n;
  }
  return v;
}

// Returns the sum of a value across all 32 threads in the warp, and every
// thread gets the same result SM90+ PTX instruction that does a hardware
// warp-wide reduction in a single instruction w.r.t. warp::reduce_sum(), which
// uses a __shfl_xor_sync butterfly tree (5 shuffles for 32 lanes)
__device__ __forceinline__ uint32_t warp_reduce_sum_full(uint32_t v) {
  uint32_t r;
  asm("redux.sync.add.u32 %0, %1, 0xFFFFFFFF;" : "=r"(r) : "r"(v));
  return r;
}

// ============================================================================
// Tie refinement (single CTA): 4-round radix-256 topK on the full FP32 ordered
// key Each round narrows by 8 bits until ties are fully resolved
// ============================================================================

template <uint32_t TopK>
__device__ void tie_handle(const Tie* ties, uint32_t num_ties,
                           uint32_t num_above, int32_t* output, void* _smem) {
  struct TS {
    alignas(128) uint32_t counter;
    alignas(128) MatchBin match;
    uint32_t histogram[RADIX];
    uint32_t warp_sum[kNumWarps];
  };
  auto* s = static_cast<TS*>(_smem);
  const auto tx = threadIdx.x;
  const auto li = tx % kWarpSize, wi = tx / kWarpSize;

  // Each thread loads one tie element.
  const bool has = tx < num_ties;
  const auto tie = has ? ties[tx] : Tie{0, 0.0f};
  const uint32_t key = convert_to_uint32_v2(tie.score);

  bool active = has;  // tracks whether this thread's tie is still a candidate.
  uint32_t remain =
      TopK - num_above;  // decreases each round as ties are resolved.
  uint32_t wpos = TopK;  // wpos will hold the final output position.
  s->counter = 0;
  __syncthreads();

  // The 4-round radix loop - each round narrows by 8 bits until ties are fully
  // resolved
#pragma unroll
  for (int r = 0; r < 4; r++) {
    uint32_t sh = 24 - r * 8;  // round 0: bits 31-24, round 1: 23-16, etc.
    uint32_t bin = (key >> sh) & 0xFF;  // this tie's 8-bit bin for this round

    // Step 1: Build 256-bin histogram.
    if (tx < RADIX) s->histogram[tx] = 0;
    __syncthreads();
    if (active) atomicAdd(&s->histogram[bin], 1);
    __syncthreads();

    // Step 2: Prefix scan to find threshold
    uint32_t hv = 0, wi2 = 0;
    if (tx < RADIX) {
      hv = s->histogram[tx];
      wi2 = warp_inclusive_sum(li, hv);
      if (li == kWarpSize - 1) s->warp_sum[wi] = wi2;
    }
    __syncthreads();

    if (tx < RADIX) {
      auto tmp = (li < RADIX / kWarpSize) ? s->warp_sum[li] : 0;
      auto tot = warp_reduce_sum_full(tmp);
      auto inter = warp_reduce_sum_full(li < wi ? tmp : 0);
      auto above = tot - (inter + wi2);
      if (above < remain && above + hv >= remain) {
        s->match = {tx, above, remain - above};
      }
    }
    __syncthreads();

    // Step 3: Scatter
    auto [thr, na, _] = s->match;  // threshold bin, num above, unused
    if (active) {
      if (bin > thr) {
        wpos = num_above +
               atomicAdd(&s->counter, 1);  // above -> place in output directly
        active = false;
      } else if (bin < thr)
        active = false;  // below -> discard
      else if (r == 3)
        wpos = TopK - atomicAdd(&s->match.equal_count,
                                -1u);  // last round: place remaining
    }
    remain -= na;
    if (!remain) break;  // all ties resolved early
  }
  // Final write
  if (wpos < TopK) output[wpos] = tie.idx;
}

// Extended tie_handle for TopK > kBlockSize (e.g. TopK=2048).
// tie_handle assumes 1 tie per thread (max 1024).
// This version handles 2 ties per thread via kPerThread=2
template <uint32_t TopK>
__device__ void tie_handle_large(const Tie* ties, uint32_t num_ties,
                                 uint32_t num_above, int32_t* output,
                                 void* _smem) {
  static_assert(TopK > kBlockSize);
  struct TS {
    alignas(128) uint32_t counter;
    alignas(128) MatchBin match;
    uint32_t histogram[RADIX];
    uint32_t warp_sum[kNumWarps];
  };
  auto* s = static_cast<TS*>(_smem);
  const auto tx = threadIdx.x;
  const auto li = tx % kWarpSize;
  const auto wi = tx / kWarpSize;

  constexpr uint32_t kPerThread = (TopK + kBlockSize - 1) / kBlockSize;
  Tie my_ties[kPerThread];
  uint32_t keys[kPerThread];
  bool active[kPerThread];

  for (uint32_t e = 0; e < kPerThread; e++) {
    uint32_t idx = e * kBlockSize + tx;
    if (idx < num_ties) {
      my_ties[e] = ties[idx];
      keys[e] = convert_to_uint32_v2(ties[idx].score);
      active[e] = true;
    } else {
      my_ties[e] = {0, 0.0f};
      keys[e] = 0;
      active[e] = false;
    }
  }

  uint32_t remain = TopK - num_above;
  s->counter = 0;
  __syncthreads();

  for (int r = 0; r < 4; r++) {
    uint32_t sh = 24 - r * 8;
    if (tx < RADIX) {
      s->histogram[tx] = 0;
    }
    __syncthreads();

    for (uint32_t e = 0; e < kPerThread; e++) {
      if (active[e]) {
        atomicAdd(&s->histogram[(keys[e] >> sh) & 0xFF], 1);
      }
    }
    __syncthreads();

    uint32_t hv = 0;
    if (tx < RADIX) {
      hv = s->histogram[tx];
      auto wi2 = warp_inclusive_sum(li, hv);
      if (li == kWarpSize - 1) {
        s->warp_sum[wi] = wi2;
      }
    }
    __syncthreads();
    if (tx < RADIX) {
      auto tmp2 = (li < RADIX / kWarpSize) ? s->warp_sum[li] : 0;
      auto total = warp_reduce_sum_full(tmp2);
      auto inter = warp_reduce_sum_full(li < wi ? tmp2 : 0);
      auto wi2 = warp_inclusive_sum(li, hv);
      auto above = total - (inter + wi2);
      if (above < remain && above + hv >= remain) {
        s->match = {
            .bin = tx, .above_count = above, .equal_count = remain - above};
      }
    }
    __syncthreads();

    auto thr = s->match.bin;
    auto na = s->match.above_count;

    for (uint32_t e = 0; e < kPerThread; e++) {
      if (!active[e]) {
        continue;
      }
      uint32_t bin = (keys[e] >> sh) & 0xFF;
      if (bin > thr) {
        uint32_t wpos = num_above + atomicAdd(&s->counter, 1);
        if (wpos < TopK) {
          output[wpos] = my_ties[e].idx;
        }
        active[e] = false;
      } else if (bin < thr) {
        active[e] = false;
      } else if (r == 3) {
        uint32_t wpos = TopK - atomicAdd(&s->match.equal_count, -1u);
        if (wpos < TopK) {
          output[wpos] = my_ties[e].idx;
        }
      }
    }

    num_above += na;
    remain -= na;
    __syncthreads();
    s->counter = 0;
    __syncthreads();
  }
}

// ============================================================================
// Register-based single-CTA fast path for seq_len <= 16384
// 4 float4 per thread × 1024 threads = 16384 elements max
// Uses 4096-bin (12-bit) histogram for better precision
// ============================================================================

template <uint32_t TopK, uint32_t HIST_BITS>
struct Histogram4096Smem {
  static constexpr uint32_t HIST_BINS = 1 << HIST_BITS;
  static constexpr uint32_t TIE_CAPACITY = TopK > kMaxTies ? TopK : kMaxTies;
  alignas(128) uint32_t counter_gt;
  alignas(128) uint32_t counter_eq;
  MatchBin match;
  uint32_t warp_sum[kNumWarps];
  union {
    uint32_t histogram[HIST_BINS];
    Tie tie_buffer[TIE_CAPACITY];
  };
};

template <uint32_t TopK, uint32_t HIST_BITS,
          uint32_t VECS_PER_THREAD = kHist4096VecsPerThread,
          bool UsePredicatedLoads = false>
__device__ void histogram_4096_topk(const float* __restrict__ scores,
                                    int32_t* __restrict__ output,
                                    uint32_t length, void* _smem) {
  constexpr uint32_t HIST_BINS = 1 << HIST_BITS;
  constexpr uint32_t ITEMS_PER_THREAD = HIST_BINS / kBlockSize;
  static_assert(HIST_BINS >= kBlockSize,
                "HIST_BITS must give >= kBlockSize bins");

  using Smem = Histogram4096Smem<TopK, HIST_BITS>;
  auto* smem = static_cast<Smem*>(_smem);
  const auto tx = threadIdx.x;
  const auto lane_id = tx % kWarpSize;
  const auto warp_id = tx / kWarpSize;

  // Phase 1: Load all data into RF + build histogram
  float4
      vecs[VECS_PER_THREAD];  // 4 vectors x 4 floats = 16 elements per thread
  if constexpr (ITEMS_PER_THREAD >= 4) {
    // Zero the histogram (SMEM writes)
    for (uint32_t i = 0; i < ITEMS_PER_THREAD / 4; i++)
      reinterpret_cast<uint4*>(
          smem->histogram)[tx * (ITEMS_PER_THREAD / 4) + i] =
          make_uint4(0, 0, 0, 0);
  } else {
    if (tx < HIST_BINS) smem->histogram[tx] = 0;
  }
  if (tx == 0) {
    smem->counter_gt = 0;
    smem->counter_eq = 0;
  }
  if constexpr (UsePredicatedLoads) {
    const bool row_aligned = (reinterpret_cast<uintptr_t>(scores) & 0xFu) == 0;
#pragma unroll
    for (uint32_t v = 0; v < VECS_PER_THREAD; v++) {
      const uint32_t base = (tx + v * kBlockSize) * 4;
      if (base < length) {
        if (row_aligned && base + 3 < length) {
          vecs[v] = *reinterpret_cast<const float4*>(scores + base);
        } else {
          load_float4_predicated(scores + base, static_cast<int>(base),
                                 static_cast<int>(length), vecs[v].x, vecs[v].y,
                                 vecs[v].z, vecs[v].w);
        }
      }
    }
  } else {
#pragma unroll
    for (uint32_t v = 0; v < VECS_PER_THREAD; v++) {
      const uint32_t base = (tx + v * kBlockSize) * 4;
      if (base < length) {
        vecs[v] = *reinterpret_cast<const float4*>(scores + base);
      }
    }
  }
  __syncthreads();

  // Build histogram from RF via atomic adds into the shared histogram
  bool done = false;
#pragma unroll
  for (uint32_t v = 0; v < VECS_PER_THREAD && !done; v++) {
    const float* elems = reinterpret_cast<const float*>(&vecs[v]);
#pragma unroll
    for (uint32_t e = 0; e < 4 && !done; e++) {
      const uint32_t idx = (tx + v * kBlockSize) * 4 + e;
      if (idx >= length) {
        done = true;
      } else {
        atomicAdd(&smem->histogram[extract_coarse_bin_N<HIST_BITS>(elems[e])],
                  1);
      }
    }
  }
  __syncthreads();

  // Phase 2: Prefix scan to find threshold bin
  // Multi-element scan (4096 bins: 4 per thread)
  uint32_t orig[ITEMS_PER_THREAD];
  uint32_t local_sum = 0;

  // Step 1: Each thread sums its 4 bins
#pragma unroll
  for (uint32_t i = 0; i < ITEMS_PER_THREAD; i++) {
    orig[i] = smem->histogram[tx * ITEMS_PER_THREAD + i];
    local_sum += orig[i];
  }

  // Step 2: Warp-level inclusive prefix sum on local_sum
  const auto warp_inc = warp_inclusive_sum(lane_id, local_sum);
  if (lane_id == kWarpSize - 1) smem->warp_sum[warp_id] = warp_inc;
  __syncthreads();

  // Step 3: Inter-warp prefix via redux.sync
  const auto tmp = smem->warp_sum[lane_id];
  uint32_t prefix = warp_reduce_sum_full(
      lane_id < warp_id ? tmp : 0);  // sum of all prior warps
  prefix +=
      warp_inc - local_sum;  // exclusive prefix within this thread's position

  // Step 4: Find threshold - scan 4 bins, accumulate prefix
#pragma unroll
  for (uint32_t i = 0; i < ITEMS_PER_THREAD; i++) {
    prefix += orig[i];
    const auto above = length - prefix;  // elements in bins ABOVE this one
    if (above < TopK && above + orig[i] >= TopK) {
      smem->match = {.bin = tx * ITEMS_PER_THREAD + i,
                     .above_count = above,
                     .equal_count = orig[i]};
    }
  }

  __syncthreads();

  // Phase 3: Scatter from registers
  const auto [thr_bin, num_above, num_equal] = smem->match;
  const bool need_tie = (num_equal + num_above > TopK);

  done = false;
#pragma unroll
  for (uint32_t v = 0; v < VECS_PER_THREAD && !done; v++) {
    const float* elems = reinterpret_cast<const float*>(&vecs[v]);
#pragma unroll
    for (uint32_t e = 0; e < 4 && !done; e++) {
      const uint32_t idx = (tx + v * kBlockSize) * 4 + e;
      if (idx >= length) {
        done = true;
      } else {
        const uint32_t bin = extract_coarse_bin_N<HIST_BITS>(elems[e]);
        if (bin > thr_bin) {
          output[atomicAdd(&smem->counter_gt, 1)] =
              idx;  // above -> output directly
        } else if (bin == thr_bin) {
          const auto pos = atomicAdd(&smem->counter_eq, 1);
          if (!need_tie) {
            if (pos + num_above < TopK) {
              output[pos + num_above] = idx;  // all fit
            }
          } else {
            if (pos < TopK) {
              smem->tie_buffer[pos] = {idx, elems[e]};  // store for refirement
            }
          }
        }
        // else: bin < thr_bin - discard (not in top-k)
      }
    }
  }

  // Phase 4: Tie-breaking
  if (!need_tie) return;
  __syncthreads();

  // Fast warp-ballot tie-breaking for small tie counts
  const uint32_t num_ties = min(num_equal, static_cast<uint32_t>(TopK));
  const uint32_t topk_remain =
      TopK - num_above;  // pick exactly remaining elements to fill topK

  auto is_greater = [](const Tie& a, const Tie& b) {
    return (a.score > b.score) || (a.score == b.score && a.idx < b.idx);
  };

  if (num_ties <= kWarpSize) {
    // <=32 ties - Use warp ballot
    // All-to-all comparison in one __ballot_sync. 32 ties x 32 warps = 1024
    // comparisons in one instruction per warp. O(1) work.
    const auto lane_id = tx % kWarpSize;
    const auto warp_id = tx / kWarpSize;
    if (lane_id >= num_ties || warp_id >= num_ties) return;
    const uint32_t mask = (1ull << num_ties) - 1u;
    const auto tie = smem->tie_buffer[lane_id];  // each lane holds one tie
    const auto target =
        smem->tie_buffer[warp_id];  // each warp evaluates one candidate
    const bool pred =
        is_greater(tie, target);  // compare all ties against target
    const auto rank = static_cast<uint32_t>(
        __popc(__ballot_sync(mask, pred)));  // count how many are greater
    if (lane_id == 0 && rank < topk_remain) {
      output[num_above + rank] = target.idx;  // place at correct position
    }
  } else if (num_ties <=
             kWarpSize *
                 2) {  // TODO (roberto): try to refactor this with <=32 case
    //  Same idea but each thread handles 2 tie elements
    const auto lane_id = tx % kWarpSize;
    const auto warp_id = tx / kWarpSize;
    const auto lane1 = lane_id + kWarpSize;
    const auto warp1 = warp_id + kWarpSize;
    const auto invalid = Tie{0xFFFFFFFF, -__FLT_MAX__};
    const auto tie0 = smem->tie_buffer[lane_id];
    const auto tie1 = lane1 < num_ties ? smem->tie_buffer[lane1] : invalid;
    if (warp_id < num_ties) {
      const auto target = smem->tie_buffer[warp_id];
      const auto r0 =
          __popc(__ballot_sync(0xFFFFFFFF, is_greater(tie0, target)));
      const auto r1 =
          __popc(__ballot_sync(0xFFFFFFFF, is_greater(tie1, target)));
      if (lane_id == 0 && r0 + r1 < topk_remain)
        output[num_above + r0 + r1] = target.idx;
    }
    if (warp1 < num_ties) {
      const auto target = smem->tie_buffer[warp1];
      const auto r0 =
          __popc(__ballot_sync(0xFFFFFFFF, is_greater(tie0, target)));
      const auto r1 =
          __popc(__ballot_sync(0xFFFFFFFF, is_greater(tie1, target)));
      if (lane_id == 0 && r0 + r1 < topk_remain)
        output[num_above + r0 + r1] = target.idx;
    }
  } else {
    // Large tie count: fall back to 4-round radix-256 sort
    if constexpr (TopK <= kBlockSize) {
      tie_handle<TopK>(smem->tie_buffer, num_ties, num_above, output, smem);
    } else {
      tie_handle_large<TopK>(smem->tie_buffer, num_ties, num_above, output,
                             smem);
    }
  }
}

template <uint32_t TopK, uint32_t HIST_BITS,
          uint32_t VECS_PER_THREAD = kHist4096VecsPerThread>
__device__ __noinline__ void histogram_4096_topk_predicated(
    const float* __restrict__ scores, int32_t* __restrict__ output,
    uint32_t length, void* _smem) {
  histogram_4096_topk<TopK, HIST_BITS, VECS_PER_THREAD, true>(scores, output,
                                                              length, _smem);
}

}  // namespace topk_histogram_4096
}  // namespace vllm

#endif  // TOPK_HISTOGRAM_4096_CUH_
