#ifndef TOPK_HISTOGRAM_4096_OVERFLOW_CUH_
#define TOPK_HISTOGRAM_4096_OVERFLOW_CUH_

#include <cub/block/block_scan.cuh>

#include "topk_histogram_4096.cuh"

namespace vllm {
namespace topk_histogram_4096 {

// Visit a row with vectorized loads and a scalar tail.
template <uint32_t BlockSize, uint32_t VecSize, typename Visit>
__device__ __forceinline__ void scan_scores(
    const float* __restrict__ scores, uint32_t length, Visit visit) {
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
      const float4 values =
          *reinterpret_cast<const float4*>(scores + base);
      visit(base, values.x);
      visit(base + 1, values.y);
      visit(base + 2, values.z);
      visit(base + 3, values.w);
    } else {
      const float2 values =
          *reinterpret_cast<const float2*>(scores + base);
      visit(base, values.x);
      visit(base + 1, values.y);
    }
  }

  for (uint32_t idx = aligned_length + tx; idx < length; idx += BlockSize) {
    visit(idx, scores[idx]);
  }
}

template <bool Wide>
struct ExactRadixTraits;

template <>
struct ExactRadixTraits<false> {
  static constexpr uint32_t kBins = 256;
  static constexpr uint32_t kRounds = 4;

  __device__ static constexpr uint32_t shift(uint32_t round) {
    return 24 - round * 8;
  }

  __device__ static constexpr uint32_t digit_mask(uint32_t) { return 0xFFu; }

  __device__ static constexpr uint32_t prefix_mask(uint32_t round) {
    return round == 0 ? 0u : (~0u << (32 - round * 8));
  }
};

template <>
struct ExactRadixTraits<true> {
  static constexpr uint32_t kBins = 2048;
  static constexpr uint32_t kRounds = 3;

  __device__ static constexpr uint32_t shift(uint32_t round) {
    return round == 0 ? 21 : (round == 1 ? 10 : 0);
  }

  __device__ static constexpr uint32_t digit_mask(uint32_t round) {
    return round < 2 ? 0x7FFu : 0x3FFu;
  }

  __device__ static constexpr uint32_t prefix_mask(uint32_t round) {
    return round == 0 ? 0u
                      : (round == 1 ? 0xFFE00000u : 0xFFFFFC00u);
  }
};

// Exact bounded-memory radix selection. Each round rescans the row, so memory
// use and correctness are independent of coarse-bin occupancy.
template <uint32_t TopK, uint32_t BlockSize, bool FuseOutputScan = false,
          uint32_t VecSize = 1, bool UseWideRadix = false,
          uint32_t StartRound = 0, uint32_t FinalCandidateCapacity = 0>
__device__ void exact_topk_rescan(
    const float* __restrict__ scores, int32_t* __restrict__ output,
    uint32_t length, void* _smem, uint32_t initial_prefix = 0,
    uint32_t initial_remaining = TopK) {
  static_assert(BlockSize >= RADIX);
  static_assert(VecSize == 1 || VecSize == 2 || VecSize == 4);
  using Radix = ExactRadixTraits<UseWideRadix>;
  constexpr uint32_t kHistogramBins = Radix::kBins;
  constexpr uint32_t kRadixRounds = Radix::kRounds;
  static_assert(StartRound < kRadixRounds);
  static_assert(!UseWideRadix || kHistogramBins % BlockSize == 0);
  static_assert(FinalCandidateCapacity == 0 ||
                (FuseOutputScan && UseWideRadix));
  using BlockScan = cub::BlockScan<uint32_t, BlockSize>;
  struct ExactSmem {
    uint32_t histogram[kHistogramBins];
    typename BlockScan::TempStorage scan;
    uint32_t prefix;
    uint32_t remaining;
    uint32_t output_counter;
    uint32_t candidate_count;
  };
  static_assert(FinalCandidateCapacity == 0 ||
                sizeof(ExactSmem) <= kExactCandidateOffset);

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
  for (uint32_t round = StartRound; round < kRadixRounds; ++round) {
    for (uint32_t bin = tx; bin < kHistogramBins; bin += BlockSize) {
      smem->histogram[bin] = 0;
    }
    __syncthreads();

    const uint32_t shift = Radix::shift(round);
    const uint32_t digit_mask = Radix::digit_mask(round);
    const uint32_t prefix_mask = Radix::prefix_mask(round);
    const uint32_t prefix = smem->prefix;

    if constexpr (FinalCandidateCapacity > 0) {
      if (round == kRadixRounds - 1) {
        if (tx == 0) {
          smem->output_counter = 0;
          smem->candidate_count = 0;
        }
        __syncthreads();
      }
    }

    const auto add_to_histogram = [&](uint32_t idx, float score) {
      const uint32_t ordered = convert_to_uint32_v2(score);
      const uint32_t ordered_prefix = ordered & prefix_mask;
      if (ordered_prefix == prefix) {
        atomicAdd(&smem->histogram[(ordered >> shift) & digit_mask], 1);
        if constexpr (FinalCandidateCapacity > 0) {
          if (round == kRadixRounds - 1) {
            const uint32_t pos = atomicAdd(&smem->candidate_count, 1);
            if (pos < FinalCandidateCapacity) {
              final_candidates[pos] = static_cast<int32_t>(idx);
            }
          }
        }
      } else if constexpr (FinalCandidateCapacity > 0) {
        if (round == kRadixRounds - 1 && ordered_prefix > prefix) {
          const uint32_t pos = atomicAdd(&smem->output_counter, 1);
          output[pos] = static_cast<int32_t>(idx);
        }
      }
    };

    scan_scores<BlockSize, VecSize>(scores, length, add_to_histogram);
    __syncthreads();

    if constexpr (UseWideRadix) {
      constexpr uint32_t kBinsPerThread = kHistogramBins / BlockSize;
      uint32_t counts[kBinsPerThread];
      uint32_t local_sum = 0;
#pragma unroll
      for (uint32_t i = 0; i < kBinsPerThread; ++i) {
        counts[i] = smem->histogram[tx * kBinsPerThread + i];
        local_sum += counts[i];
      }

      uint32_t lower_prefix;
      uint32_t total;
      // Snapshot before the block scan; its barrier precedes any update.
      const uint32_t remaining = smem->remaining;
      BlockScan(smem->scan).ExclusiveSum(local_sum, lower_prefix, total);
      uint32_t count_above = total - lower_prefix - local_sum;
#pragma unroll
      for (int i = static_cast<int>(kBinsPerThread) - 1; i >= 0; --i) {
        const uint32_t count = counts[i];
        if (count_above < remaining && count_above + count >= remaining) {
          const uint32_t bin = tx * kBinsPerThread + i;
          smem->remaining = remaining - count_above;
          smem->prefix = prefix | (bin << shift);
        }
        count_above += count;
      }
    } else if (tx == 0) {
      uint32_t count_above = 0;
      for (int bin = RADIX - 1; bin >= 0; --bin) {
        const uint32_t count = smem->histogram[bin];
        if (count_above + count >= smem->remaining) {
          smem->remaining -= count_above;
          smem->prefix |= static_cast<uint32_t>(bin) << shift;
          break;
        }
        count_above += count;
      }
    }
    __syncthreads();
  }

  const uint32_t pivot = smem->prefix;
  if constexpr (FuseOutputScan) {
    if (tx == 0) {
      if constexpr (FinalCandidateCapacity == 0) {
        smem->output_counter = 0;
      }
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
      const uint32_t candidate_count = smem->candidate_count;
      if (candidate_count <= FinalCandidateCapacity) {
        for (uint32_t i = tx; i < candidate_count; i += BlockSize) {
          const uint32_t idx = final_candidates[i];
          collect(idx, scores[idx]);
        }
      } else {
        scan_scores<BlockSize, VecSize>(
            scores, length, [&](uint32_t idx, float score) {
              if ((convert_to_uint32_v2(score) & 0xFFFFFC00u) ==
                  (pivot & 0xFFFFFC00u)) {
                collect(idx, score);
              }
            });
      }
    } else {
      scan_scores<BlockSize, VecSize>(scores, length, collect);
    }
    __syncthreads();
  } else {
    if (tx == 0) smem->output_counter = 0;
    __syncthreads();

    scan_scores<BlockSize, VecSize>(
        scores, length, [&](uint32_t idx, float score) {
          if (convert_to_uint32_v2(score) > pivot) {
            const uint32_t pos = atomicAdd(&smem->output_counter, 1);
            if (pos < TopK) output[pos] = static_cast<int32_t>(idx);
          }
        });
    __syncthreads();

    scan_scores<BlockSize, VecSize>(
        scores, length, [&](uint32_t idx, float score) {
          if (convert_to_uint32_v2(score) == pivot) {
            const uint32_t pos = atomicAdd(&smem->output_counter, 1);
            if (pos < TopK) output[pos] = static_cast<int32_t>(idx);
          }
        });
    __syncthreads();
  }
}

template <uint32_t TopK, uint32_t BlockSize, uint32_t VecSize>
__device__ __noinline__ void exact_topk_rescan_cold(
    const float* __restrict__ scores, int32_t* __restrict__ output,
    uint32_t length, void* smem) {
  exact_topk_rescan<TopK, BlockSize, true, VecSize, true>(scores, output,
                                                          length, smem);
}

struct CoarseRefineRange {
  uint32_t base_key;
  bool finite;
};

template <uint32_t HistBits>
__device__ __forceinline__ CoarseRefineRange coarse_refine_range(
    uint32_t coarse_bin) {
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

}  // namespace topk_histogram_4096
}  // namespace vllm

#endif  // TOPK_HISTOGRAM_4096_OVERFLOW_CUH_
