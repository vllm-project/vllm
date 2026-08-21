/*
 * Shared 4096-bin single-CTA TopK helpers.
 */

#ifndef TOPK_HISTOGRAM_4096_CUH_
#define TOPK_HISTOGRAM_4096_CUH_

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <type_traits>

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
  uint32_t key;
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

__device__ __forceinline__ uint32_t score_to_ordered(float x) {
  uint32_t bits = __float_as_uint(x);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

__device__ __forceinline__ uint32_t score_to_ordered(__half x) {
  const uint16_t bits = __half_as_ushort(x);
  return (bits & 0x8000) ? static_cast<uint16_t>(~bits)
                         : static_cast<uint16_t>(bits | 0x8000);
}

__device__ __forceinline__ uint32_t fp16x2_bits_to_ordered(uint32_t bits) {
  const uint32_t negative = (bits & 0x80008000u) >> 15;
  return bits ^ (0x80008000u | negative * 0xFFFFu);
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

template <uint32_t kBits>
__device__ __forceinline__ uint32_t extract_coarse_bin_N(__half x) {
  return score_to_ordered(x) >> (16 - kBits);
}

template <uint32_t kBits, typename InputType>
__device__ __forceinline__ uint32_t extract_input_coarse_bin(InputType x) {
  return extract_coarse_bin_N<kBits>(x);
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
// thread gets the same result. SM80+ uses redux.sync.add.u32, a single PTX
// instruction for hardware warp-wide reduction. Older targets use the
// __shfl_xor_sync butterfly tree, like warp::reduce_sum() (5 shuffles for 32
// lanes).
__device__ __forceinline__ uint32_t warp_reduce_sum_full(uint32_t v) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  uint32_t r;
  asm("redux.sync.add.u32 %0, %1, 0xFFFFFFFF;" : "=r"(r) : "r"(v));
  return r;
#else
  #pragma unroll
  for (uint32_t mask = kWarpSize >> 1; mask > 0; mask >>= 1) {
    v += __shfl_xor_sync(0xFFFFFFFF, v, mask);
  }
  return v;
#endif
}

// ============================================================================
// Tie refinement on ordered score keys. FP16 scans only the native key bits
// below its coarse histogram. FP32 needs four radix passes because its coarse
// histogram is based on an FP16 projection.
// ============================================================================

template <uint32_t TopK, uint32_t FineBits>
__device__ void tie_handle_fp16(const Tie* ties, uint32_t num_ties,
                                uint32_t num_above, int32_t* output,
                                void* _smem) {
  constexpr uint32_t kFineBins = 1 << FineBits;
  constexpr uint32_t kFineWarps = (kFineBins + kWarpSize - 1) / kWarpSize;
  constexpr uint32_t kScanThreads = kFineWarps * kWarpSize;
  constexpr uint32_t kPerThread = (TopK + kBlockSize - 1) / kBlockSize;
  struct TS {
    alignas(128) uint32_t counter;
    alignas(128) MatchBin match;
    uint32_t histogram[kFineBins];
    uint32_t warp_sum[kFineWarps];
  };
  auto* s = static_cast<TS*>(_smem);
  const auto tx = threadIdx.x;
  const auto lane = tx % kWarpSize;

  Tie my_ties[kPerThread];
  bool active[kPerThread];
#pragma unroll
  for (uint32_t e = 0; e < kPerThread; e++) {
    const uint32_t pos = e * kBlockSize + tx;
    active[e] = pos < num_ties;
    my_ties[e] = active[e] ? ties[pos] : Tie{0, 0};
  }

  if (tx < kFineBins) s->histogram[tx] = 0;
  if (tx == 0) s->counter = 0;
  __syncthreads();

#pragma unroll
  for (uint32_t e = 0; e < kPerThread; e++) {
    if (active[e])
      atomicAdd(&s->histogram[my_ties[e].key & (kFineBins - 1)], 1);
  }
  __syncthreads();

  const uint32_t value = tx < kFineBins ? s->histogram[tx] : 0;
  const uint32_t inclusive =
      tx < kScanThreads ? warp_inclusive_sum(lane, value) : 0;
  const uint32_t warp = tx / kWarpSize;
  if (tx < kScanThreads) {
    if (lane == kWarpSize - 1) s->warp_sum[warp] = inclusive;
  }
  __syncthreads();
  if (tx < kScanThreads) {
    uint32_t total = 0;
    uint32_t prior = 0;
#pragma unroll
    for (uint32_t w = 0; w < kFineWarps; w++) {
      total += s->warp_sum[w];
      if (w < warp) prior += s->warp_sum[w];
    }
    const uint32_t above = total - prior - inclusive;
    const uint32_t remaining = TopK - num_above;
    if (tx < kFineBins && above < remaining && above + value >= remaining) {
      s->match = {
          .bin = tx, .above_count = above, .equal_count = remaining - above};
    }
  }
  __syncthreads();

  const auto threshold = s->match.bin;
#pragma unroll
  for (uint32_t e = 0; e < kPerThread; e++) {
    if (!active[e]) continue;
    const uint32_t bin = my_ties[e].key & (kFineBins - 1);
    uint32_t pos = TopK;
    if (bin > threshold) {
      pos = num_above + atomicAdd(&s->counter, 1);
    } else if (bin == threshold) {
      pos = TopK - atomicAdd(&s->match.equal_count, -1u);
    }
    if (pos < TopK) output[pos] = my_ties[e].idx;
  }
}

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
  const auto tie = has ? ties[tx] : Tie{0, 0};
  const uint32_t key = tie.key;

  constexpr int kRadixRounds = 4;

  bool active = has;  // tracks whether this thread's tie is still a candidate.
  uint32_t remain =
      TopK - num_above;  // decreases each round as ties are resolved.
  uint32_t wpos = TopK;  // wpos will hold the final output position.
  s->counter = 0;
  __syncthreads();

#pragma unroll
  for (int r = 0; r < kRadixRounds; r++) {
    uint32_t sh = (kRadixRounds - 1 - r) * 8;
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
      else if (r == kRadixRounds - 1)
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
      keys[e] = ties[idx].key;
      active[e] = true;
    } else {
      my_ties[e] = {0, 0};
      keys[e] = 0;
      active[e] = false;
    }
  }

  constexpr int kRadixRounds = 4;

  uint32_t remain = TopK - num_above;
  s->counter = 0;
  __syncthreads();

  for (int r = 0; r < kRadixRounds; r++) {
    uint32_t sh = (kRadixRounds - 1 - r) * 8;
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
      } else if (r == kRadixRounds - 1) {
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

template <typename InputType, uint32_t TopK, uint32_t HIST_BITS,
          uint32_t VECS_PER_THREAD = kHist4096VecsPerThread,
          bool UsePredicatedLoads = false>
__device__ void histogram_4096_topk(const InputType* __restrict__ scores,
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

  static_assert(std::is_same_v<InputType, float> ||
                std::is_same_v<InputType, __half>);
  constexpr uint32_t ELEMS_PER_VEC = sizeof(uint4) / sizeof(InputType);
  union InputVec {
    uint4 packed;
    InputType elems[ELEMS_PER_VEC];
    uint32_t fp16_pairs[sizeof(uint4) / sizeof(uint32_t)];
  };

  // Phase 1: Load all data into RF + build histogram
  InputVec vecs[VECS_PER_THREAD];
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
    static_assert(std::is_same_v<InputType, float>);
    const bool row_aligned = (reinterpret_cast<uintptr_t>(scores) & 0xFu) == 0;
#pragma unroll
    for (uint32_t v = 0; v < VECS_PER_THREAD; v++) {
      const uint32_t base = (tx + v * kBlockSize) * ELEMS_PER_VEC;
      if (base < length) {
        if (row_aligned && base + ELEMS_PER_VEC - 1 < length) {
          vecs[v].packed = *reinterpret_cast<const uint4*>(scores + base);
        } else {
          load_float4_predicated(scores + base, static_cast<int>(base),
                                 static_cast<int>(length), vecs[v].elems[0],
                                 vecs[v].elems[1], vecs[v].elems[2],
                                 vecs[v].elems[3]);
        }
      }
    }
  } else {
#pragma unroll
    for (uint32_t v = 0; v < VECS_PER_THREAD; v++) {
      const uint32_t base = (tx + v * kBlockSize) * ELEMS_PER_VEC;
      if (base < length) {
        vecs[v].packed = *reinterpret_cast<const uint4*>(scores + base);
      }
    }
  }
  __syncthreads();

  // Build histogram from RF via atomic adds into the shared histogram
  bool done = false;
  if constexpr (std::is_same_v<InputType, __half>) {
#pragma unroll
    for (uint32_t v = 0; v < VECS_PER_THREAD && !done; v++) {
#pragma unroll
      for (uint32_t p = 0; p < ELEMS_PER_VEC / 2 && !done; p++) {
        const uint32_t idx = (tx + v * kBlockSize) * ELEMS_PER_VEC + p * 2;
        if (idx >= length) {
          done = true;
          continue;
        }
        const uint32_t keys = fp16x2_bits_to_ordered(vecs[v].fp16_pairs[p]);
        atomicAdd(&smem->histogram[(keys & 0xFFFFu) >> (16 - HIST_BITS)], 1);
        if (idx + 1 < length) {
          atomicAdd(&smem->histogram[keys >> (32 - HIST_BITS)], 1);
        } else {
          done = true;
        }
      }
    }
  } else {
#pragma unroll
    for (uint32_t v = 0; v < VECS_PER_THREAD && !done; v++) {
#pragma unroll
      for (uint32_t e = 0; e < ELEMS_PER_VEC && !done; e++) {
        const uint32_t idx = (tx + v * kBlockSize) * ELEMS_PER_VEC + e;
        if (idx >= length) {
          done = true;
        } else {
          const uint32_t bin =
              extract_input_coarse_bin<HIST_BITS>(vecs[v].elems[e]);
          atomicAdd(&smem->histogram[bin], 1);
        }
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

  // Step 3: Inter-warp prefix across warp sums.
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
  if constexpr (std::is_same_v<InputType, __half>) {
#pragma unroll
    for (uint32_t v = 0; v < VECS_PER_THREAD && !done; v++) {
#pragma unroll
      for (uint32_t p = 0; p < ELEMS_PER_VEC / 2 && !done; p++) {
        const uint32_t base = (tx + v * kBlockSize) * ELEMS_PER_VEC + p * 2;
        if (base >= length) {
          done = true;
          continue;
        }
        const uint32_t keys = fp16x2_bits_to_ordered(vecs[v].fp16_pairs[p]);
#pragma unroll
        for (uint32_t e = 0; e < 2; e++) {
          const uint32_t idx = base + e;
          if (idx >= length) continue;
          const uint32_t key = (keys >> (e * 16)) & 0xFFFFu;
          const uint32_t bin = key >> (16 - HIST_BITS);
          if (bin > thr_bin) {
            output[atomicAdd(&smem->counter_gt, 1)] = idx;
          } else if (bin == thr_bin) {
            const auto pos = atomicAdd(&smem->counter_eq, 1);
            if (!need_tie) {
              if (pos + num_above < TopK) output[pos + num_above] = idx;
            } else if (pos < TopK) {
              smem->tie_buffer[pos] = {idx, key};
            }
          }
        }
      }
    }
  } else {
#pragma unroll
    for (uint32_t v = 0; v < VECS_PER_THREAD && !done; v++) {
#pragma unroll
      for (uint32_t e = 0; e < ELEMS_PER_VEC && !done; e++) {
        const uint32_t idx = (tx + v * kBlockSize) * ELEMS_PER_VEC + e;
        if (idx >= length) {
          done = true;
        } else {
          const auto raw_score = vecs[v].elems[e];
          const uint32_t bin = extract_input_coarse_bin<HIST_BITS>(raw_score);
          if (bin > thr_bin) {
            output[atomicAdd(&smem->counter_gt, 1)] = idx;
          } else if (bin == thr_bin) {
            const auto pos = atomicAdd(&smem->counter_eq, 1);
            if (!need_tie) {
              if (pos + num_above < TopK) output[pos + num_above] = idx;
            } else if (pos < TopK) {
              smem->tie_buffer[pos] = {idx, score_to_ordered(raw_score)};
            }
          }
        }
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
    return (a.key > b.key) || (a.key == b.key && a.idx < b.idx);
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
    const auto invalid = Tie{0xFFFFFFFF, 0};
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
    if constexpr (std::is_same_v<InputType, __half>) {
      tie_handle_fp16<TopK, 16 - HIST_BITS>(smem->tie_buffer, num_ties,
                                            num_above, output, smem);
    } else if constexpr (TopK <= kBlockSize) {
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
  histogram_4096_topk<float, TopK, HIST_BITS, VECS_PER_THREAD, true>(
      scores, output, length, _smem);
}

}  // namespace topk_histogram_4096
}  // namespace vllm

#endif  // TOPK_HISTOGRAM_4096_CUH_
