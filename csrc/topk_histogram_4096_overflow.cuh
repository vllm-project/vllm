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

// Exact bounded-memory radix selection. Each round rescans the row, so memory
// use and correctness are independent of coarse-bin occupancy.
template <uint32_t TopK, uint32_t BlockSize, uint32_t VecSize = 1>
__device__ void exact_topk_rescan(const float* __restrict__ scores,
                                  int32_t* __restrict__ output, uint32_t length,
                                  void* _smem) {
  static_assert(VecSize == 1 || VecSize == 2 || VecSize == 4);
  using Radix = ExactRadixTraits;
  constexpr uint32_t kHistogramBins = Radix::kBins;
  static_assert(kHistogramBins % BlockSize == 0);
  using BlockScan = cub::BlockScan<uint32_t, BlockSize>;
  struct ExactSmem {
    uint32_t histogram[kHistogramBins];
    typename BlockScan::TempStorage scan;
    uint32_t prefix;
    uint32_t remaining;
    uint32_t output_counter;
  };

  auto* smem = static_cast<ExactSmem*>(_smem);
  const uint32_t tx = threadIdx.x;

  if (tx == 0) {
    smem->prefix = 0;
    smem->remaining = TopK;
  }
  __syncthreads();

#pragma unroll
  for (uint32_t round = 0; round < Radix::kRounds; ++round) {
    for (uint32_t bin = tx; bin < kHistogramBins; bin += BlockSize) {
      smem->histogram[bin] = 0;
    }
    __syncthreads();

    const uint32_t shift = Radix::shift(round);
    const uint32_t digit_mask = Radix::digit_mask(round);
    const uint32_t prefix_mask = Radix::prefix_mask(round);
    const uint32_t prefix = smem->prefix;

    scan_scores<BlockSize, VecSize>(scores, length, [&](uint32_t, float score) {
      const uint32_t ordered = convert_to_uint32_v2(score);
      if ((ordered & prefix_mask) == prefix) {
        atomicAdd(&smem->histogram[(ordered >> shift) & digit_mask], 1);
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
    __syncthreads();
  }

  const uint32_t pivot = smem->prefix;
  if (tx == 0) {
    smem->output_counter = 0;
    smem->histogram[0] = 0;
  }
  __syncthreads();

  const uint32_t equal_count = smem->remaining;
  const uint32_t equal_base = TopK - equal_count;
  scan_scores<BlockSize, VecSize>(
      scores, length, [&](uint32_t idx, float score) {
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
      });
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

}  // namespace topk_histogram_4096
}  // namespace vllm

#endif  // TOPK_HISTOGRAM_4096_OVERFLOW_CUH_
