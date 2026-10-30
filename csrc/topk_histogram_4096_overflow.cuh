#ifndef TOPK_HISTOGRAM_4096_OVERFLOW_CUH_
#define TOPK_HISTOGRAM_4096_OVERFLOW_CUH_

#include <cub/block/block_scan.cuh>

#include "topk_histogram_4096.cuh"

namespace vllm {
namespace topk_histogram_4096 {

// Visit a row with vectorized loads and a scalar tail.
template <uint32_t BlockSize, uint32_t VecSize, typename Visit>
__device__ __forceinline__ void scan_scores(const float* __restrict__ scores,
                                            uint32_t length, Visit visit) {
  static_assert(VecSize == 1 || VecSize == 2 || VecSize == 4);
  const uint32_t tx = threadIdx.x;

  if constexpr (VecSize == 1) {
    for (uint32_t idx = tx; idx < length; idx += BlockSize) {
      visit(idx, scores[idx]);
    }
    return;
  }

  const uint32_t aligned_length = length - length % VecSize;

  for (uint32_t base = tx * VecSize; base < aligned_length;
       base += BlockSize * VecSize) {
    if constexpr (VecSize == 4) {
      const float4 values = *reinterpret_cast<const float4*>(scores + base);
      visit(base, values.x);
      visit(base + 1, values.y);
      visit(base + 2, values.z);
      visit(base + 3, values.w);
    } else {
      const float2 values = *reinterpret_cast<const float2*>(scores + base);
      visit(base, values.x);
      visit(base + 1, values.y);
    }
  }

  for (uint32_t idx = aligned_length + tx; idx < length; idx += BlockSize) {
    visit(idx, scores[idx]);
  }
}

// Wide radix layout: 11 + 11 + 10 bits over the ordered FP32 key.
struct ExactRadixTraits {
  static constexpr uint32_t kBins = 2048;
  static constexpr uint32_t kRounds = 3;

  __device__ static constexpr uint32_t shift(uint32_t round) {
    return round == 0 ? 21 : (round == 1 ? 10 : 0);
  }

  __device__ static constexpr uint32_t digit_mask(uint32_t round) {
    return round < 2 ? 0x7FFu : 0x3FFu;
  }

  __device__ static constexpr uint32_t prefix_mask(uint32_t round) {
    return round == 0 ? 0u : (round == 1 ? 0xFFE00000u : 0xFFFFFC00u);
  }
};

// Final-round candidates are staged past the rescan's own shared state.
constexpr uint32_t kExactCandidateOffset = 16 * 1024;

// Exact bounded-memory radix selection. Each round rescans the row, so memory
// use and correctness are independent of coarse-bin occupancy. Recovery can
// resume at StartRound from a known prefix, and with FinalCandidateCapacity
// the last round stages its candidates so the output pass skips a row scan.
template <uint32_t TopK, uint32_t BlockSize, uint32_t VecSize = 1,
          uint32_t StartRound = 0, uint32_t FinalCandidateCapacity = 0>
__device__ void exact_topk_rescan(const float* __restrict__ scores,
                                  int32_t* __restrict__ output, uint32_t length,
                                  void* _smem, uint32_t initial_prefix = 0,
                                  uint32_t initial_remaining = TopK) {
  static_assert(VecSize == 1 || VecSize == 2 || VecSize == 4);
  using Radix = ExactRadixTraits;
  constexpr uint32_t kHistogramBins = Radix::kBins;
  static_assert(kHistogramBins % BlockSize == 0);
  static_assert(StartRound < Radix::kRounds);
  static_assert(BlockSize == kWarpSize * kWarpSize, "one warp sum per lane");
  struct ExactSmem {
    uint32_t histogram[kHistogramBins];
    uint32_t warp_sum[BlockSize / kWarpSize];
    uint32_t prefix;
    uint32_t remaining;
    uint32_t output_counter;
    uint32_t candidate_count;
    uint32_t pivot_count;
  };
  static_assert(FinalCandidateCapacity == 0 ||
                sizeof(ExactSmem) <= kExactCandidateOffset);
  static_assert(FinalCandidateCapacity == 0 || TopK <= FinalCandidateCapacity);

  auto* smem = static_cast<ExactSmem*>(_smem);
  auto* final_candidates = reinterpret_cast<int32_t*>(
      static_cast<uint8_t*>(_smem) + kExactCandidateOffset);
  const uint32_t tx = threadIdx.x;

  if (tx == 0) {
    smem->prefix = initial_prefix;
    smem->remaining = initial_remaining;
  }
  __syncthreads();

#pragma unroll
  for (uint32_t round = StartRound; round < Radix::kRounds; ++round) {
    for (uint32_t bin = tx; bin < kHistogramBins; bin += BlockSize) {
      smem->histogram[bin] = 0;
    }
    __syncthreads();

    const uint32_t shift = Radix::shift(round);
    const uint32_t digit_mask = Radix::digit_mask(round);
    const uint32_t prefix_mask = Radix::prefix_mask(round);
    const uint32_t prefix = smem->prefix;
    const bool final_round = round == Radix::kRounds - 1;

    if constexpr (FinalCandidateCapacity > 0) {
      if (final_round) {
        if (tx == 0) {
          smem->output_counter = 0;
          smem->candidate_count = 0;
        }
        __syncthreads();
      }
    }

    scan_scores<BlockSize, VecSize>(
        scores, length, [&](uint32_t idx, float score) {
          const uint32_t ordered = convert_to_uint32_v2(score);
          const uint32_t ordered_prefix = ordered & prefix_mask;
          if (ordered_prefix == prefix) {
            atomicAdd(&smem->histogram[(ordered >> shift) & digit_mask], 1);
            if constexpr (FinalCandidateCapacity > 0) {
              if (final_round) {
                const uint32_t pos = atomicAdd(&smem->candidate_count, 1);
                if (pos < FinalCandidateCapacity) {
                  final_candidates[pos] = static_cast<int32_t>(idx);
                }
              }
            }
          } else if constexpr (FinalCandidateCapacity > 0) {
            if (final_round && ordered_prefix > prefix) {
              const uint32_t pos = atomicAdd(&smem->output_counter, 1);
              output[pos] = static_cast<int32_t>(idx);
            }
          }
        });
    __syncthreads();

    constexpr uint32_t kBinsPerThread = kHistogramBins / BlockSize;
    uint32_t counts[kBinsPerThread];
    uint32_t local_sum = 0;
#pragma unroll
    for (uint32_t i = 0; i < kBinsPerThread; ++i) {
      counts[i] = smem->histogram[tx * kBinsPerThread + i];
      local_sum += counts[i];
    }

    // Shuffle scan: lighter on registers than a CUB block scan. Snapshot
    // remaining first; the barrier below precedes any update.
    const uint32_t remaining = smem->remaining;
    const uint32_t lane = tx % kWarpSize;
    const uint32_t warp = tx / kWarpSize;
    const uint32_t warp_inclusive = warp_inclusive_sum(lane, local_sum);
    if (lane == kWarpSize - 1) smem->warp_sum[warp] = warp_inclusive;
    __syncthreads();
    const uint32_t warp_total = smem->warp_sum[lane];
    uint32_t count_above =
        warp_reduce_sum_full(lane > warp ? warp_total : 0) +
        __shfl_sync(0xFFFFFFFFu, warp_inclusive, kWarpSize - 1) -
        warp_inclusive;
#pragma unroll
    for (int i = static_cast<int>(kBinsPerThread) - 1; i >= 0; --i) {
      const uint32_t count = counts[i];
      if (count_above < remaining && count_above + count >= remaining) {
        const uint32_t bin = tx * kBinsPerThread + i;
        smem->remaining = remaining - count_above;
        smem->prefix = prefix | (bin << shift);
        if constexpr (FinalCandidateCapacity > 0) smem->pivot_count = count;
      }
      count_above += count;
    }
    __syncthreads();
  }

  const uint32_t pivot = smem->prefix;
  if (tx == 0) {
    if constexpr (FinalCandidateCapacity == 0) smem->output_counter = 0;
    smem->histogram[0] = 0;
  }
  __syncthreads();

  const uint32_t equal_count = smem->remaining;
  const uint32_t equal_base = TopK - equal_count;
  const auto collect = [&](uint32_t idx, float score) {
    const uint32_t ordered = convert_to_uint32_v2(score);
    if (ordered > pivot) {
      const uint32_t pos = atomicAdd(&smem->output_counter, 1);
      output[pos] = static_cast<int32_t>(idx);
    } else if (ordered == pivot) {
      const uint32_t pos = atomicAdd(&smem->histogram[0], 1);
      if (pos < equal_count) {
        output[equal_base + pos] = static_cast<int32_t>(idx);
      }
    }
  };

  if constexpr (FinalCandidateCapacity > 0) {
    // Keys above the final round's prefix are already in the output.
    const uint32_t candidate_count = smem->candidate_count;
    if (candidate_count <= FinalCandidateCapacity) {
      for (uint32_t i = tx; i < candidate_count; i += BlockSize) {
        const uint32_t idx = final_candidates[i];
        collect(idx, scores[idx]);
      }
    } else if (smem->pivot_count == candidate_count) {
      // Every key under the final prefix equals the pivot (heavy ties), so
      // any equal_count staged candidates complete the output.
      for (uint32_t i = tx; i < equal_count; i += BlockSize) {
        output[equal_base + i] = final_candidates[i];
      }
    } else {
      constexpr uint32_t kFinalPrefixMask =
          Radix::prefix_mask(Radix::kRounds - 1);
      scan_scores<BlockSize, VecSize>(
          scores, length, [&](uint32_t idx, float score) {
            if ((convert_to_uint32_v2(score) & kFinalPrefixMask) ==
                (pivot & kFinalPrefixMask)) {
              collect(idx, score);
            }
          });
    }
  } else {
    scan_scores<BlockSize, VecSize>(scores, length, collect);
  }
  __syncthreads();
}

template <uint32_t TopK, uint32_t BlockSize, uint32_t VecSize>
__device__ __noinline__ void exact_topk_rescan_cold(
    const float* __restrict__ scores, int32_t* __restrict__ output,
    uint32_t length, void* smem) {
  exact_topk_rescan<TopK, BlockSize, VecSize>(scores, output, length, smem);
}

struct CoarseRefineRange {
  uint32_t base_key;
  bool finite;
};

template <uint32_t HistBits>
__device__ __forceinline__ CoarseRefineRange
coarse_refine_range(uint32_t coarse_bin) {
  static_assert(HistBits == 10 || HistBits == 12);
  const uint16_t lower_half_key =
      static_cast<uint16_t>(coarse_bin << (16 - HistBits));
  const uint16_t lower_half_bits =
      (lower_half_key & 0x8000u)
          ? static_cast<uint16_t>(lower_half_key & 0x7FFFu)
          : static_cast<uint16_t>(~lower_half_key);
  const uint32_t lower_key =
      convert_to_uint32_v2(__half2float(__ushort_as_half(lower_half_bits)));
  return {.base_key = lower_key > 8192 ? lower_key - 8192 : 0,
          .finite = (lower_half_bits & 0x7C00u) != 0x7C00u};
}

constexpr uint32_t kCoarseRefineBins = 2048;
constexpr uint32_t kCoarseRefineDigitBits = 11;
using CoarseRefineScan = cub::BlockScan<uint32_t, kBlockSize>;

struct CoarseRefineSmem {
  uint32_t histogram[kCoarseRefineBins];
  CoarseRefineScan::TempStorage scan;
  uint32_t prefix;
  uint32_t remaining;
  uint32_t output_counter;
  uint32_t equal_counter;
  uint32_t valid;
  uint32_t key_min;
  uint32_t key_max;
};

__device__ __forceinline__ uint32_t warp_reduce_min_full(uint32_t v) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  return __reduce_min_sync(0xFFFFFFFFu, v);
#else
  #pragma unroll
  for (uint32_t mask = kWarpSize >> 1; mask > 0; mask >>= 1) {
    v = min(v, __shfl_xor_sync(0xFFFFFFFFu, v, mask));
  }
  return v;
#endif
}

__device__ __forceinline__ uint32_t warp_reduce_max_full(uint32_t v) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  return __reduce_max_sync(0xFFFFFFFFu, v);
#else
  #pragma unroll
  for (uint32_t mask = kWarpSize >> 1; mask > 0; mask >>= 1) {
    v = max(v, __shfl_xor_sync(0xFFFFFFFFu, v, mask));
  }
  return v;
#endif
}

__device__ __forceinline__ void select_coarse_refine_digit(
    CoarseRefineSmem* smem, uint32_t prefix, uint32_t shift) {
  const uint32_t bin0 = 2 * threadIdx.x;
  const uint32_t count0 = smem->histogram[bin0];
  const uint32_t count1 = smem->histogram[bin0 + 1];
  const uint32_t local_sum = count0 + count1;
  const uint32_t remaining = smem->remaining;
  uint32_t lower_prefix;
  uint32_t total;
  CoarseRefineScan(smem->scan).ExclusiveSum(local_sum, lower_prefix, total);
  uint32_t count_above = total - lower_prefix - local_sum;
  if (count_above < remaining && count_above + count1 >= remaining) {
    smem->remaining = remaining - count_above;
    smem->prefix = prefix | ((bin0 + 1) << shift);
  }
  count_above += count1;
  if (count_above < remaining && count_above + count0 >= remaining) {
    smem->remaining = remaining - count_above;
    smem->prefix = prefix | (bin0 << shift);
  }
}

// Refine an overflowing FP16 histogram bin from values already resident in
// registers. A 12-bit FP16 bin spans at most 22 ordered-FP32 bits for finite
// values, so two radix-2048 passes select the exact pivot. A bin holding a
// single key (e.g. all-equal scores) needs no refinement: that key is the
// pivot, so the first pass also tracks the bin's min and max key.
template <uint32_t TopK, uint32_t HistBits, uint32_t VecsPerThread>
__device__ __forceinline__ bool exact_topk_refine_registers(
    const float4 (&vecs)[VecsPerThread], int32_t* __restrict__ output,
    uint32_t length, uint32_t coarse_bin, uint32_t num_above, void* _smem) {
  constexpr uint32_t kBins = kCoarseRefineBins;
  constexpr uint32_t kDigitBits = kCoarseRefineDigitBits;
  constexpr uint32_t kDigitMask = kBins - 1;
  constexpr uint32_t kDeltaLimit = 1u << (2 * kDigitBits);
  static_assert(HistBits == 12);
  static_assert(kBins == 2 * kBlockSize);
  auto* smem = static_cast<CoarseRefineSmem*>(_smem);
  const uint32_t tx = threadIdx.x;
  const auto range = coarse_refine_range<HistBits>(coarse_bin);
  const uint32_t base_key = range.base_key;

  if (tx == 0) {
    smem->prefix = 0;
    smem->remaining = TopK - num_above;
    smem->valid = range.finite;
    smem->key_min = 0xFFFFFFFFu;
    smem->key_max = 0;
  }
  __syncthreads();

#pragma unroll
  for (uint32_t round = 0; round < 2; ++round) {
    for (uint32_t bin = tx; bin < kBins; bin += kBlockSize) {
      smem->histogram[bin] = 0;
    }
    __syncthreads();

    const uint32_t prefix = smem->prefix;
    const uint32_t shift = round == 0 ? kDigitBits : 0;
    uint32_t key_min = 0xFFFFFFFFu;
    uint32_t key_max = 0;
    bool done = false;
#pragma unroll
    for (uint32_t v = 0; v < VecsPerThread && !done; ++v) {
      const float* values = reinterpret_cast<const float*>(&vecs[v]);
#pragma unroll
      for (uint32_t e = 0; e < 4 && !done; ++e) {
        const uint32_t idx = (tx + v * kBlockSize) * 4 + e;
        if (idx >= length) {
          done = true;
        } else if (extract_coarse_bin_N<HistBits>(values[e]) == coarse_bin) {
          const uint32_t key = convert_to_uint32_v2(values[e]);
          const uint32_t delta = key - base_key;
          if (round == 0) {
            key_min = min(key_min, key);
            key_max = max(key_max, key);
          }
          if (key < base_key || delta >= kDeltaLimit) {
            atomicExch(&smem->valid, 0);
          } else if (round == 0 || (delta & ~kDigitMask) == prefix) {
            atomicAdd(&smem->histogram[(delta >> shift) & kDigitMask], 1);
          }
        }
      }
    }
    if (round == 0) {
      key_min = warp_reduce_min_full(key_min);
      key_max = warp_reduce_max_full(key_max);
      if (tx % kWarpSize == 0) {
        atomicMin(&smem->key_min, key_min);
        atomicMax(&smem->key_max, key_max);
      }
    }
    __syncthreads();

    if (smem->valid == 0) return false;

    if (round == 0 && smem->key_min == smem->key_max) {
      if (tx == 0) smem->prefix = smem->key_min - base_key;
      __syncthreads();
      break;
    }

    select_coarse_refine_digit(smem, prefix, shift);
    __syncthreads();
  }

  if (tx == 0) {
    smem->output_counter = 0;
    smem->equal_counter = 0;
  }
  __syncthreads();

  const uint32_t pivot = base_key + smem->prefix;
  const uint32_t equal_base = TopK - smem->remaining;
  bool done = false;
#pragma unroll
  for (uint32_t v = 0; v < VecsPerThread && !done; ++v) {
    const float* values = reinterpret_cast<const float*>(&vecs[v]);
#pragma unroll
    for (uint32_t e = 0; e < 4 && !done; ++e) {
      const uint32_t idx = (tx + v * kBlockSize) * 4 + e;
      if (idx >= length) {
        done = true;
      } else {
        const uint32_t key = convert_to_uint32_v2(values[e]);
        if (key > pivot) {
          const uint32_t pos = atomicAdd(&smem->output_counter, 1);
          output[pos] = static_cast<int32_t>(idx);
        } else if (key == pivot) {
          const uint32_t pos = atomicAdd(&smem->equal_counter, 1);
          if (equal_base + pos < TopK) {
            output[equal_base + pos] = static_cast<int32_t>(idx);
          }
        }
      }
    }
  }
  __syncthreads();
  return true;
}

}  // namespace topk_histogram_4096
}  // namespace vllm

#endif  // TOPK_HISTOGRAM_4096_OVERFLOW_CUH_
