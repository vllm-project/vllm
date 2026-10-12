#ifndef PERSISTENT_TOPK_OVERFLOW_CUH_
#define PERSISTENT_TOPK_OVERFLOW_CUH_

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include "topk_histogram_4096.cuh"

namespace vllm::persistent::overflow {

// The medium histogram has already selected the coarse FP32 prefix.
struct MediumFallbackState {
  uint32_t prefix;
  uint32_t remaining;
};

__device__ __forceinline__ bool prepare_medium_fallback(
    int threshold_bin_count, int capacity, int threshold_bin, int remaining,
    void* smem) {
  if (!__builtin_expect(threshold_bin_count > capacity, 0)) return false;
  if (threadIdx.x == 0) {
    auto* state = static_cast<MediumFallbackState*>(smem);
    state->prefix = static_cast<uint32_t>(threshold_bin) << 21;
    state->remaining = static_cast<uint32_t>(remaining);
  }
  __syncthreads();
  return true;
}

template <int TopK, int BlockSize, int VecSize>
__device__ __forceinline__ void recover_medium(
    const float* scores, int32_t* output, uint32_t length, void* smem) {
  const auto* state = static_cast<const MediumFallbackState*>(smem);
  const uint32_t prefix = state->prefix;
  const uint32_t remaining = state->remaining;
  __syncthreads();
  topk_histogram_4096::exact_topk_rescan<
      TopK, BlockSize, true, VecSize, true, 1, 4096>(
      scores, output, length, smem, prefix, remaining);
}

}  // namespace vllm::persistent::overflow

namespace vllm::filtered_topk {

template <typename T, size_t N>
struct vec_t;

template <typename DType, bool UseWideCoarse>
struct FilteredTopKTraits;

namespace overflow {

constexpr uint32_t kSamplingMinRows = 96;
constexpr uint32_t kSamplingMinLength = 192 * 1024;

// Sample a pivot, bound the candidate set, then select it exactly by radix.
template <typename DType, typename IdType, int VecSize, uint32_t BlockSize,
          uint32_t CandidateCapacity>
__device__ bool try_sampled_exact_topk(
    const DType* __restrict__ scores, IdType* __restrict__ output,
    uint32_t length, uint32_t top_k, void* dynamic_smem, int* histogram,
    int* candidate_count, int* output_counter, int* radix_prefix,
    int* radix_remaining) {
  using Traits = FilteredTopKTraits<DType, false>;
  using Radix = topk_histogram_4096::ExactRadixTraits<false>;
  constexpr uint32_t kItemsPerThread = 2;
  constexpr uint32_t kSampleSize = BlockSize * kItemsPerThread;
  using SampleSort =
      cub::BlockRadixSort<uint32_t, BlockSize, kItemsPerThread>;
  static_assert(sizeof(typename SampleSort::TempStorage) <=
                2 * CandidateCapacity * sizeof(int));

  const uint32_t tx = threadIdx.x;
  uint32_t sample_keys[kItemsPerThread];
#pragma unroll
  for (uint32_t i = 0; i < kItemsPerThread; ++i) {
    const uint32_t sample_idx = tx * kItemsPerThread + i;
    const uint32_t score_idx = static_cast<uint32_t>(
        (static_cast<uint64_t>(2 * sample_idx + 1) * length) /
        (2 * kSampleSize));
    sample_keys[i] = Traits::ToOrdered(scores[score_idx]);
  }

  auto* sort_smem =
      static_cast<typename SampleSort::TempStorage*>(dynamic_smem);
  SampleSort(*sort_smem).SortDescending(sample_keys);
  __syncthreads();

  const uint32_t target_candidates =
      min(CandidateCapacity / 2, (top_k < 2048) ? 4 * top_k : 2 * top_k);
  uint32_t sample_rank = static_cast<uint32_t>(
      (static_cast<uint64_t>(target_candidates) * kSampleSize + length - 1) /
      length);
  sample_rank = max(1u, min(sample_rank, kSampleSize));
  const uint32_t rank_index = sample_rank - 1;
  if (tx == rank_index / kItemsPerThread) {
    *radix_prefix = static_cast<int>(sample_keys[rank_index % kItemsPerThread]);
  }
  if (tx == 0) *candidate_count = 0;
  __syncthreads();

  auto* candidates = static_cast<int*>(dynamic_smem);
  const uint32_t proposed_pivot = static_cast<uint32_t>(*radix_prefix);
  const auto collect_candidate = [&](uint32_t idx, DType score) {
    if (Traits::ToOrdered(score) >= proposed_pivot) {
      const int pos = atomicAdd(candidate_count, 1);
      if (pos < static_cast<int>(CandidateCapacity)) candidates[pos] = idx;
    }
  };

  vec_t<DType, VecSize> score_vec;
  const uint32_t aligned_length = length / VecSize * VecSize;
  for (uint32_t base = tx * VecSize; base < aligned_length;
       base += BlockSize * VecSize) {
    score_vec.cast_load(scores + base);
#pragma unroll
    for (uint32_t i = 0; i < VecSize; ++i) {
      collect_candidate(base + i, score_vec[i]);
    }
  }
  for (uint32_t idx = aligned_length + tx; idx < length; idx += BlockSize) {
    collect_candidate(idx, scores[idx]);
  }
  __syncthreads();

  const int num_candidates = *candidate_count;
  if (num_candidates < static_cast<int>(top_k) ||
      num_candidates > static_cast<int>(CandidateCapacity)) {
    return false;
  }

  auto* candidate_keys = candidates + CandidateCapacity;
  for (int i = tx; i < num_candidates; i += BlockSize) {
    candidate_keys[i] =
        static_cast<int>(Traits::ToOrdered(scores[candidates[i]]));
  }

  if (tx == 0) {
    *radix_prefix = 0;
    *radix_remaining = static_cast<int>(top_k);
  }
  __syncthreads();

#pragma unroll
  for (uint32_t round = 0; round < Radix::kRounds; ++round) {
    if (tx < Radix::kBins) histogram[tx] = 0;
    __syncthreads();

    const uint32_t prefix = static_cast<uint32_t>(*radix_prefix);
    const uint32_t prefix_mask = Radix::prefix_mask(round);
    const uint32_t shift = Radix::shift(round);
    for (int i = tx; i < num_candidates; i += BlockSize) {
      const uint32_t ordered = static_cast<uint32_t>(candidate_keys[i]);
      if ((ordered & prefix_mask) == prefix) {
        atomicAdd(&histogram[(ordered >> shift) & Radix::digit_mask(round)], 1);
      }
    }
    __syncthreads();

    if (tx == 0) {
      int count_above = 0;
      const int remaining = *radix_remaining;
      for (int bin = static_cast<int>(Radix::kBins) - 1; bin >= 0; --bin) {
        const int count = histogram[bin];
        if (count_above + count >= remaining) {
          *radix_remaining = remaining - count_above;
          *radix_prefix =
              static_cast<int>(prefix | (static_cast<uint32_t>(bin) << shift));
          break;
        }
        count_above += count;
      }
    }
    __syncthreads();
  }

  if (tx == 0) {
    *output_counter = 0;
    histogram[0] = 0;
  }
  __syncthreads();

  const uint32_t pivot = static_cast<uint32_t>(*radix_prefix);
  const int equal_count = *radix_remaining;
  const int equal_base = static_cast<int>(top_k) - equal_count;
  for (int i = tx; i < num_candidates; i += BlockSize) {
    const int idx = candidates[i];
    const uint32_t ordered = static_cast<uint32_t>(candidate_keys[i]);
    if (ordered > pivot) {
      output[atomicAdd(output_counter, 1)] = static_cast<IdType>(idx);
    } else if (ordered == pivot) {
      const int pos = atomicAdd(&histogram[0], 1);
      if (pos < equal_count) output[equal_base + pos] = static_cast<IdType>(idx);
    }
  }
  __syncthreads();
  return true;
}

template <typename DType, typename IdType, int VecSize, uint32_t BlockSize,
          uint32_t MaxK, uint32_t CandidateCapacity, bool UseWideCoarse>
__device__ __forceinline__ bool recover_coarse_if_needed(
    const DType* scores, IdType* output, uint32_t length, uint32_t top_k,
    uint32_t num_rows, uint32_t threshold_bin, uint32_t remaining,
    const int* coarse_histogram, void* dynamic_smem, int* refine_histogram,
    int* candidate_count, int* output_counter, int* radix_prefix,
    int* radix_remaining) {
  const bool coarse_overflow =
      coarse_histogram[threshold_bin] -
          coarse_histogram[threshold_bin + 1] >
      CandidateCapacity;
  __syncthreads();
  if (!__builtin_expect(coarse_overflow, 0)) return false;

  // Sampling amortizes its fixed block-sort cost only for long, busy batches.
  if (num_rows >= kSamplingMinRows && length >= kSamplingMinLength &&
      try_sampled_exact_topk<DType, IdType, VecSize, BlockSize,
                             CandidateCapacity>(
          scores, output, length, top_k, dynamic_smem, refine_histogram,
          candidate_count, output_counter, radix_prefix, radix_remaining)) {
    return true;
  }

  if constexpr (UseWideCoarse) {
    const uint32_t high11_bin = threshold_bin >> 1;
    const uint32_t count_above_high11 =
        coarse_histogram[(high11_bin + 1) << 1];
    topk_histogram_4096::exact_topk_rescan<
        MaxK, BlockSize, true, VecSize, true, 1, CandidateCapacity>(
        scores, output, length, dynamic_smem, high11_bin << 21,
        top_k - count_above_high11);
  } else {
    topk_histogram_4096::exact_topk_rescan<
        MaxK, BlockSize, true, VecSize, true, 1, CandidateCapacity>(
        scores, output, length, dynamic_smem, threshold_bin << 21, remaining);
  }
  return true;
}

template <uint32_t MaxK, uint32_t BlockSize, uint32_t VecSize,
          uint32_t CandidateCapacity, typename DType, typename IdType>
__device__ __forceinline__ bool recover_refinement_if_needed(
    uint32_t buffered_count, const DType* scores, IdType* output,
    uint32_t length, void* smem) {
  if (buffered_count <= CandidateCapacity) return false;
  topk_histogram_4096::exact_topk_rescan<MaxK, BlockSize, true, VecSize, true>(
      scores, output, length, smem);
  return true;
}

}  // namespace overflow
}  // namespace vllm::filtered_topk

#endif  // PERSISTENT_TOPK_OVERFLOW_CUH_
