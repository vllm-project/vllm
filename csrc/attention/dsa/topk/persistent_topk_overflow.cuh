#ifndef PERSISTENT_TOPK_OVERFLOW_CUH_
#define PERSISTENT_TOPK_OVERFLOW_CUH_

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include "topk_histogram_4096.cuh"
#include "topk_histogram_4096_overflow.cuh"

namespace vllm::persistent::overflow {

// Where exact recovery resumes after the medium path's coarse histogram:
// the selected 11-bit FP32 prefix and how many keys are still needed in it.
struct MediumResume {
  bool needed = false;
  uint32_t prefix = 0;
  uint32_t remaining = 0;
};

template <int TopK, int BlockSize, int VecSize, typename InT>
__device__ __noinline__ void recover_medium(const InT* scores, int32_t* output,
                                            uint32_t length, void* smem,
                                            uint32_t prefix,
                                            uint32_t remaining) {
  topk_histogram_4096::exact_topk_rescan<TopK, BlockSize, VecSize, 1, 4096>(
      scores, output, length, smem, prefix, remaining);
}

}  // namespace vllm::persistent::overflow

namespace vllm::filtered_topk {

template <typename T, size_t N>
struct vec_t;

template <typename DType, bool UseWideCoarse>
struct FilteredTopKTraits;

namespace overflow {

// Sampled recovery pays off only for long rows in large batches. topk.cu
// currently routes such batches to sampled_topk_kernel, so this matters for
// direct FilteredTopKRaggedTransform callers and future dispatch changes.
constexpr uint32_t kSamplingMinRows = 96;
constexpr uint32_t kSamplingMinLength = 192 * 1024;

// 8-bit digits over the ordered key: the sampled candidate set is small, so a
// 256-bin histogram selected by one thread is cheaper than a block scan.
struct SampledRadixDigits {
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

// Sample a pivot, bound the candidate set, then select it exactly by radix.
template <typename DType, typename IdType, int VecSize, uint32_t BlockSize,
          uint32_t CandidateCapacity>
__device__ bool try_sampled_exact_topk(const DType* __restrict__ scores,
                                       IdType* __restrict__ output,
                                       uint32_t length, uint32_t top_k,
                                       void* dynamic_smem, int* scratch) {
  using Traits =
      FilteredTopKTraits<decltype(topk_histogram_4096::to_float(DType{})),
                         false>;
  using Radix = SampledRadixDigits;
  constexpr uint32_t kItemsPerThread = 2;
  constexpr uint32_t kSampleSize = BlockSize * kItemsPerThread;
  using SampleSort = cub::BlockRadixSort<uint32_t, BlockSize, kItemsPerThread>;
  static_assert(sizeof(typename SampleSort::TempStorage) <=
                2 * CandidateCapacity * sizeof(int));

  // scratch: the radix histogram followed by four counters.
  int* histogram = scratch;
  int* candidate_count = scratch + Radix::kBins;
  int* output_counter = candidate_count + 1;
  int* radix_prefix = candidate_count + 2;
  int* radix_remaining = candidate_count + 3;
  const uint32_t tx = threadIdx.x;
  uint32_t sample_keys[kItemsPerThread];
#pragma unroll
  for (uint32_t i = 0; i < kItemsPerThread; ++i) {
    const uint32_t sample_idx = tx * kItemsPerThread + i;
    const uint32_t score_idx = static_cast<uint32_t>(
        (static_cast<uint64_t>(2 * sample_idx + 1) * length) /
        (2 * kSampleSize));
    sample_keys[i] =
        Traits::ToOrdered(topk_histogram_4096::to_float(scores[score_idx]));
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
  // Every key tied with the pivot is a candidate. When the sample already
  // predicts more candidates than fit, skip the doomed collection scan.
  uint32_t samples_at_or_above = 0;
#pragma unroll
  for (uint32_t i = 0; i < kItemsPerThread; ++i) {
    samples_at_or_above +=
        __syncthreads_count(sample_keys[i] >= proposed_pivot);
  }
  if (static_cast<uint64_t>(samples_at_or_above) * length >
      static_cast<uint64_t>(CandidateCapacity) * kSampleSize) {
    return false;
  }
  const auto collect_candidate = [&](uint32_t idx, float score) {
    if (Traits::ToOrdered(score) >= proposed_pivot) {
      const int pos = atomicAdd(candidate_count, 1);
      if (pos < static_cast<int>(CandidateCapacity)) candidates[pos] = idx;
    }
  };

  vec_t<float, VecSize> score_vec;
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
    collect_candidate(idx, topk_histogram_4096::to_float(scores[idx]));
  }
  __syncthreads();

  const int num_candidates = *candidate_count;
  if (num_candidates < static_cast<int>(top_k) ||
      num_candidates > static_cast<int>(CandidateCapacity)) {
    return false;
  }

  auto* candidate_keys = candidates + CandidateCapacity;
  for (int i = tx; i < num_candidates; i += BlockSize) {
    candidate_keys[i] = static_cast<int>(Traits::ToOrdered(
        topk_histogram_4096::to_float(scores[candidates[i]])));
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
      if (pos < equal_count)
        output[equal_base + pos] = static_cast<IdType>(idx);
    }
  }
  __syncthreads();
  return true;
}

// Exact selection after the filtered path's coarse bin overflowed the
// candidate stash. The coarse histogram holds suffix counts over the top
// CoarseBits of the ordered key; recovery resumes from its top 11 bits.
// scratch must hold Radix::kBins + 4 ints for the sampled path.
template <typename DType, typename IdType, int VecSize, uint32_t BlockSize,
          uint32_t MaxK, uint32_t CandidateCapacity, int CoarseBits>
__device__ __noinline__ void recover_coarse(const DType* scores, IdType* output,
                                            uint32_t length, uint32_t top_k,
                                            uint32_t num_rows,
                                            const int* coarse_suffix,
                                            uint32_t threshold_bin,
                                            void* dynamic_smem, int* scratch) {
  static_assert(CoarseBits >= 11);
  constexpr uint32_t kShift = CoarseBits - 11;
  const uint32_t prefix_bin = threshold_bin >> kShift;
  const uint32_t remaining =
      top_k - static_cast<uint32_t>(coarse_suffix[(prefix_bin + 1) << kShift]);
  // The coarse histogram shares dynamic smem with the recovery buffers.
  __syncthreads();

  // Sampling amortizes its fixed block-sort cost only for long, busy batches.
  if (num_rows >= kSamplingMinRows && length >= kSamplingMinLength &&
      try_sampled_exact_topk<DType, IdType, VecSize, BlockSize,
                             CandidateCapacity>(scores, output, length, top_k,
                                                dynamic_smem, scratch)) {
    return;
  }
  topk_histogram_4096::exact_topk_rescan<MaxK, BlockSize, VecSize, 1,
                                         CandidateCapacity>(
      scores, output, length, dynamic_smem, prefix_bin << 21, remaining);
}

}  // namespace overflow
}  // namespace vllm::filtered_topk

#endif  // PERSISTENT_TOPK_OVERFLOW_CUH_
