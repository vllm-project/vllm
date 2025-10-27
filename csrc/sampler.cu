#include "dispatch_utils.h"

#include <torch/cuda.h>
#include <c10/cuda/CUDAGuard.h>

#ifndef USE_ROCM
  #include <cub/cub.cuh>
#else
  #include <hipcub/hipcub.hpp>
#endif

namespace vllm {

template <typename scalar_t>
__global__ void apply_repetition_penalties_kernel(
    scalar_t* __restrict__ logits,         // [num_seqs, vocab_size]
    const bool* __restrict__ prompt_mask,  // [num_seqs, vocab_size]
    const bool* __restrict__ output_mask,  // [num_seqs, vocab_size]
    const scalar_t* __restrict__ repetition_penalties,  // [num_seqs]
    const int num_seqs, const int vocab_size, const int tile_size) {
  // Each block handles one sequence and a tile of vocab
  const int seq_idx = blockIdx.x;
  if (seq_idx >= num_seqs) return;

  const int tile_start = blockIdx.y * tile_size;
  const int tile_end = min(tile_start + tile_size, vocab_size);

  // Load repetition penalty for this sequence
  const scalar_t penalty = repetition_penalties[seq_idx];

  // Each thread processes multiple vocab items within the tile
  for (int vocab_idx = tile_start + threadIdx.x; vocab_idx < tile_end;
       vocab_idx += blockDim.x) {
    const int64_t idx = static_cast<int64_t>(seq_idx) * vocab_size + vocab_idx;
    const bool is_repeated = prompt_mask[idx] || output_mask[idx];
    if (is_repeated) {
      scalar_t logit = logits[idx];
      if (logit > 0) {
        logits[idx] = logit / penalty;
      } else {
        logits[idx] = logit * penalty;
      }
    }
  }
}

template <int step>
static inline __device__ uint32_t extractBinIdx(float x) {
  uint32_t bits = __float_as_uint(x);
  bits = (bits & 0x80000000) ? bits : ~bits & 0x7fffffff;

  if constexpr (step == 0) {
    return bits >> 21;
  } else if constexpr (step == 1) {
    return (bits >> 10) & 0x7ff;
  } else {
    return bits & 0x3ff;
  }
}

template <int shift>
static inline __device__ bool isPartialMatch(float x, uint32_t pattern) {
  if constexpr (shift == 0) {
    return true;
  }
  uint32_t bits = __float_as_uint(x);
  bits = (bits & 0x80000000) ? bits : ~bits & 0x7fffffff;
  return (bits ^ pattern) >> shift == 0;
}

template <int step, int kNumThreadsPerBlock, int kNumBins, int kTopK,
          int kNumFinalItems, typename SmemFinalType>
__device__ bool processHistogramStep(
    const float* logits, int rowEnd, uint32_t& logitPattern,
    int& thresholdBinIdx, int* smemHistogram, int* smemIndices,
    int* smemThresholdBinIdx, int* smemFinalDstIdx, int* smemFinalBinSize,
    int* smemFoundTopKValues, SmemFinalType& smemFinal, int stride1,
    int rowStart) {
  // Clear the histogram.
#pragma unroll
  for (int idx = threadIdx.x; idx < kNumBins; idx += kNumThreadsPerBlock) {
    smemHistogram[idx] = 0;
  }

  // Make sure the histogram is ready.
  __syncthreads();

  // Update pattern
  constexpr auto patternShift = step == 0 ? 0 : step == 1 ? 21 : 10;
  if constexpr (step == 1) {
    logitPattern = static_cast<uint32_t>(thresholdBinIdx & 0x7ff)
                   << patternShift;
  } else if constexpr (step == 2) {
    logitPattern |= static_cast<uint32_t>(thresholdBinIdx & 0x7ff)
                    << patternShift;
  }

  // Fetch elements one-by-one.
  for (int idx = rowStart + threadIdx.x; idx < rowEnd;
       idx += kNumThreadsPerBlock) {
    float logit = logits[idx * stride1];
    if (isPartialMatch<patternShift>(logit, logitPattern)) {
      uint32_t binIdx = extractBinIdx<step>(logit);
      atomicAdd(&smemHistogram[binIdx], 1);
    }
  }

  // Make sure the histogram is ready.
  __syncthreads();

  // Reads the value of the starting position in the smemIndices array
  int lastValue = smemFoundTopKValues[0];

  for (int round = 0; round < kNumBins / kNumThreadsPerBlock; round++) {
    // Read the values from SMEM.
    int idx = threadIdx.x + kNumThreadsPerBlock * round;
    int binCount{0};
    binCount = smemHistogram[idx];

    // Make sure each thread has read its value.
    __syncthreads();

    // Compute the prefix sum.
    int prefixSum{0}, totalSum{0};
    using Scan = cub::BlockScan<int, kNumThreadsPerBlock>;
    Scan(smemFinal.smemScan).ExclusiveSum(binCount, prefixSum, totalSum);

    // Update the histogram with the prefix sums.
    prefixSum += lastValue;
    totalSum += lastValue;
    smemHistogram[idx] = prefixSum;

    // Make sure the data is in shared memory.
    __syncthreads();

    // Find the last valid bin.
    bool foundThreshold = false;
    if (prefixSum < kTopK) {
      int nextPrefixSum = threadIdx.x == kNumThreadsPerBlock - 1
                              ? totalSum
                              : smemHistogram[idx + 1];

      if (nextPrefixSum >= kTopK) {
        smemThresholdBinIdx[0] = idx;
        smemFinalBinSize[0] = nextPrefixSum - prefixSum;
        smemFoundTopKValues[0] = prefixSum;
        foundThreshold = true;
      }
    }

    // Early exit: if any thread found the threshold, we can skip remaining
    // rounds
    if (__syncthreads_or(foundThreshold)) {
      break;
    }

    lastValue = totalSum;
  }

  // Make sure the data is in shared memory.
  __syncthreads();

  // The threshold bin.
  thresholdBinIdx = smemThresholdBinIdx[0];

  // Fetch elements one-by-one and populate the shared memory buffers.
  for (int idx = rowStart + threadIdx.x; idx < rowEnd;
       idx += kNumThreadsPerBlock) {
    float logit = logits[idx * stride1];
    if (isPartialMatch<patternShift>(logit, logitPattern)) {
      uint32_t binIdx = extractBinIdx<step>(logit);
      if (binIdx < thresholdBinIdx) {
        int dstIdx = atomicAdd(&smemHistogram[binIdx], 1);
        smemIndices[dstIdx] = idx;
      }
      if constexpr (step < 2) {
        // Only fill the final items if the threshold bin fits
        if (binIdx == thresholdBinIdx &&
            smemFinalBinSize[0] <= kNumFinalItems) {
          int dstIdx = atomicAdd(&smemFinalDstIdx[0], 1);
          smemFinal.items.logits[dstIdx] = logit;
          smemFinal.items.indices[dstIdx] = idx;
        }
      } else {
        if (binIdx == thresholdBinIdx) {
          int dstIdx = atomicAdd(&smemHistogram[binIdx], 1);
          if (dstIdx < kTopK) {
            smemIndices[dstIdx] = idx;
          }
        }
      }
    }
  }

  // Make sure the elements are in shared memory.
  __syncthreads();

  // Check if we should continue to next step
  return smemFinalBinSize[0] > kNumFinalItems;
}

// Follows 11 - 11 - 10 bit iterations
template <int kNumThreadsPerBlock = 512, int kNumBins = 2048, int kTopK = 2048,
          bool useRadixSort = true, bool sortResultLogitDescending = false>
static __device__ void topKPerRowJob(const float* logits, int rowStart,
                                     int rowEnd, int* outIndices, int stride1) {
  // The number of slots for the final pass.
  static constexpr int kNumFinalItems = 2048;
  // The number of elements per thread for the final sort.
  static constexpr int kNumFinalItemsPerThread =
      kNumFinalItems / kNumThreadsPerBlock;
  // The class to sort the elements during the final pass.
  using FinalSort = cub::BlockRadixSort<float, kNumThreadsPerBlock,
                                        kNumFinalItemsPerThread, int>;
  // The class to compute the inclusive prefix-sum over the histogram.
  using Scan = cub::BlockScan<int, kNumThreadsPerBlock>;

  // The structure to store the final items (for the final pass).
  struct FinalItems {
    // Shared memory to store the indices for the final pass.
    int indices[kNumFinalItems];
    // Shared memory to store the logits for the final pass.
    float logits[kNumFinalItems];
  };

  // Shared memory to compute the block sort.
  __shared__ union {
    FinalItems items;
    typename FinalSort::TempStorage finalSort;
    typename Scan::TempStorage smemScan;
  } smemFinal;

  // Shared memory to store the histogram.
  __shared__ int smemHistogram[kNumBins];
  // Shared memory to store the selected indices.
  __shared__ int smemIndices[kTopK];
  // Shared memory to store the threshold bin.
  __shared__ int smemThresholdBinIdx[1];
  // Shared memory counter to register the candidates for the final phase.
  __shared__ int smemFinalDstIdx[1];
  // Shared memory to determine if the threshold bin fits in the final items.
  __shared__ int smemFinalBinSize[1];
  // Shared memory to keep track of the top-k values found so far by the
  // previous iterations
  __shared__ int smemFoundTopKValues[1];

  // The length of the row.
  int rowLen = rowEnd - rowStart;

  // Shortcut if the length of the row is smaller than Top-K. Indices are not
  // sorted by their corresponding logit.
  if (rowLen <= kTopK) {
    for (int rowIt = threadIdx.x; rowIt < rowLen;
         rowIt += kNumThreadsPerBlock) {
      outIndices[rowIt] = rowIt;
    }
    for (int rowIt = rowLen + threadIdx.x; rowIt < kTopK;
         rowIt += kNumThreadsPerBlock) {
      outIndices[rowIt] = -1;
    }
    return;
  }
  // Initialize values
  if (threadIdx.x == 0) {
    smemFinalDstIdx[0] = 0;
    smemFoundTopKValues[0] = 0;
  }
  __syncthreads();
  int thresholdBinIdx = -1;
  uint32_t logitPattern = 0;

  // Step 0: Process first 11 bits
  bool continueToNextStep =
      processHistogramStep<0, kNumThreadsPerBlock, kNumBins, kTopK,
                           kNumFinalItems>(
          logits, rowEnd, logitPattern, thresholdBinIdx, smemHistogram,
          smemIndices, smemThresholdBinIdx, smemFinalDstIdx, smemFinalBinSize,
          smemFoundTopKValues, smemFinal, stride1, rowStart);

  if (continueToNextStep) {
    // Step 1: Process next 11 bits
    continueToNextStep = processHistogramStep<1, kNumThreadsPerBlock, kNumBins,
                                              kTopK, kNumFinalItems>(
        logits, rowEnd, logitPattern, thresholdBinIdx, smemHistogram,
        smemIndices, smemThresholdBinIdx, smemFinalDstIdx, smemFinalBinSize,
        smemFoundTopKValues, smemFinal, stride1, rowStart);

    if (continueToNextStep) {
      // Step 2: Process final 10 bits
      processHistogramStep<2, kNumThreadsPerBlock, kNumBins, kTopK,
                           kNumFinalItems>(
          logits, rowEnd, logitPattern, thresholdBinIdx, smemHistogram,
          smemIndices, smemThresholdBinIdx, smemFinalDstIdx, smemFinalBinSize,
          smemFoundTopKValues, smemFinal, stride1, rowStart);
    }
  }

  if (!continueToNextStep) {
    // The histogram did not proceed to the final 10 bits, therefore we need to
    // sort the final items The logits of the elements to be sorted in the final
    // pass.
    if constexpr (useRadixSort) {
      // Sorting with radix sort
      float finalLogits[kNumFinalItemsPerThread];
      // The indices of the elements to be sorted in the final pass.
      int finalIndices[kNumFinalItemsPerThread];

#pragma unroll
      for (int ii = 0; ii < kNumFinalItemsPerThread; ++ii) {
        finalLogits[ii] = -FLT_MAX;
      }

      // Read the elements from SMEM.
#pragma unroll
      for (int ii = 0; ii < kNumFinalItemsPerThread; ++ii) {
        int srcIdx = ii * kNumThreadsPerBlock + threadIdx.x;
        if (srcIdx < smemFinalDstIdx[0]) {
          finalLogits[ii] = smemFinal.items.logits[srcIdx];
          finalIndices[ii] = smemFinal.items.indices[srcIdx];
        }
      }
      // Make sure the shared memory has been read.
      __syncthreads();

      // Sort the elements.
      FinalSort(smemFinal.finalSort)
          .SortDescendingBlockedToStriped(finalLogits, finalIndices);

      // Copy the data back to the shared memory storage.
      int baseIdx = smemFoundTopKValues[0];

#pragma unroll
      for (int ii = 0; ii < kNumFinalItemsPerThread; ++ii) {
        int srcIdx = ii * kNumThreadsPerBlock + threadIdx.x;
        int dstIdx = baseIdx + srcIdx;

        if (dstIdx < kTopK) {
          smemIndices[dstIdx] = finalIndices[ii];
        }
      }
    } else {
      // Sorting with insertion sort
      auto baseIdx = smemFoundTopKValues[0];
      for (int i = threadIdx.x; i < smemFinalDstIdx[0];
           i += kNumThreadsPerBlock) {
        int outIndex = 0;
        auto logit = smemFinal.items.logits[i];
        for (int j = 0; j < smemFinalDstIdx[0]; j++) {
          auto otherLogit = smemFinal.items.logits[j];
          if (logit < otherLogit || (logit == otherLogit && i < j)) {
            outIndex++;
          }
        }
        // Store if outIndex is in bounds
        if (outIndex + baseIdx < kTopK) {
          smemIndices[outIndex + baseIdx] = smemFinal.items.indices[i];
        }
      }
    }
    __syncthreads();
  }

  if constexpr (sortResultLogitDescending) {
    // Sorting with radix sort
    float finalLogits[kNumFinalItemsPerThread];
    // The indices of the elements to be sorted in the final pass.
    int finalIndices[kNumFinalItemsPerThread];

// Read the elements from SMEM.
#pragma unroll
    for (int ii = 0; ii < kNumFinalItemsPerThread; ++ii) {
      int srcIdx = ii * kNumThreadsPerBlock + threadIdx.x;
      const auto index = smemIndices[srcIdx];
      const auto logit = logits[rowStart + index * stride1];
      finalLogits[ii] = logit;
      finalIndices[ii] = index;
    }

    // Make sure the shared memory has been read.
    __syncthreads();

    // Sort the elements.
    FinalSort(smemFinal.finalSort)
        .SortDescendingBlockedToStriped(finalLogits, finalIndices);

    // Store to global memory
#pragma unroll
    for (int ii = 0; ii < kNumFinalItemsPerThread; ++ii) {
      int srcIdx = ii * kNumThreadsPerBlock + threadIdx.x;
      outIndices[srcIdx] = finalIndices[ii];
    }
  }

  if constexpr (!sortResultLogitDescending) {
    // Store to global memory.
#pragma unroll
    for (int i = threadIdx.x; i < kTopK; i += kNumThreadsPerBlock) {
      outIndices[i] = smemIndices[i];
    }
  }
}

template <int kNumThreadsPerBlock = 512, bool useRadixSort>
static __global__ __launch_bounds__(kNumThreadsPerBlock) void topKPerRow(
    const float* logits, const int* rowStarts, const int* rowEnds,
    int* outIndices, int stride0, int stride1, const int offsetIndex) {
  // The number of bins in the histogram.
  static constexpr int kNumBins = 2048;

  // The top-k width.
  static constexpr int kTopK = 2048;

  // The row computed by this block.
  int rowIdx = blockIdx.x + offsetIndex;

  // The range of logits within the row.
  int rowStart = rowStarts[rowIdx];
  int rowEnd = rowEnds[rowIdx];

  // Local pointers to this block
  auto outIndicesLocal = outIndices + rowIdx * kTopK;
  auto logitsLocal = logits + rowIdx * stride0;

  topKPerRowJob<kNumThreadsPerBlock, kNumBins, kTopK, useRadixSort>(
      logitsLocal, rowStart, rowEnd, outIndicesLocal, stride1);
}

template <int kNumThreadsPerBlock = 512, bool useRadixSort>
static __global__ __launch_bounds__(kNumThreadsPerBlock) void topKPerRowDecode(
    const float* logits, const int* seqLens, int* outIndices, int stride0,
    int stride1, int next_n) {
  // The number of bins in the histogram.
  static constexpr int kNumBins = 2048;

  // The top-k width.
  static constexpr int kTopK = 2048;

  // The row computed by this block.
  int rowIdx = blockIdx.x;

  // The range of logits within the row.
  int rowStart = 0;
  int seq_len = seqLens[rowIdx / next_n];
  int rowEnd = seq_len - next_n + (rowIdx % next_n) + 1;

  // Local pointers to this block
  auto outIndicesLocal = outIndices + rowIdx * kTopK;
  auto logitsLocal = logits + rowIdx * stride0;

  topKPerRowJob<kNumThreadsPerBlock, kNumBins, kTopK, useRadixSort>(
      logitsLocal, rowStart, rowEnd, outIndicesLocal, stride1);
}

}  // namespace vllm

void apply_repetition_penalties_(
    torch::Tensor& logits,             // [num_seqs, vocab_size], in-place
    const torch::Tensor& prompt_mask,  // [num_seqs, vocab_size]
    const torch::Tensor& output_mask,  // [num_seqs, vocab_size]
    const torch::Tensor& repetition_penalties) {  // [num_seqs]
  TORCH_CHECK(logits.is_contiguous());
  TORCH_CHECK(prompt_mask.is_contiguous());
  TORCH_CHECK(output_mask.is_contiguous());
  TORCH_CHECK(repetition_penalties.is_contiguous());

  int vocab_size = logits.size(-1);
  int num_seqs = logits.size(0);

  if (num_seqs == 0) return;

  // Get number of SMs on the current device
  int sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount,
                         logits.get_device());

  // Compute tile_num and tile_size
  int tile_num =
      std::min(vocab_size, std::max(1, (sms + num_seqs - 1) / num_seqs));
  int tile_size = (vocab_size + tile_num - 1) / tile_num;

  // Each block handles one sequence and a tile of vocab
  dim3 grid(num_seqs, tile_num);
  dim3 block(std::min(tile_size, 1024));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(logits));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(
      logits.scalar_type(), "apply_repetition_penalties_kernel", [&] {
        vllm::apply_repetition_penalties_kernel<scalar_t>
            <<<grid, block, 0, stream>>>(
                logits.data_ptr<scalar_t>(), prompt_mask.data_ptr<bool>(),
                output_mask.data_ptr<bool>(),
                repetition_penalties.data_ptr<scalar_t>(), num_seqs, vocab_size,
                tile_size);
      });
}

void top_k_per_row_decode(const torch::Tensor& logits, int64_t next_n,
                          const torch::Tensor& seqLens, torch::Tensor& indices,
                          int64_t numRows, int64_t stride0, int64_t stride1) {
  constexpr int kSortingAlgorithmThreshold = 12288;
  constexpr int kNumThreadsPerBlock = 512;
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const auto numColumns = logits.size(1);

  if (numColumns < kSortingAlgorithmThreshold) {
    vllm::topKPerRowDecode<kNumThreadsPerBlock, false>
        <<<numRows, kNumThreadsPerBlock, 0, stream>>>(
            logits.data_ptr<float>(), seqLens.data_ptr<int>(),
            indices.data_ptr<int>(), static_cast<int>(stride0),
            static_cast<int>(stride1), static_cast<int>(next_n));
  } else {
    vllm::topKPerRowDecode<kNumThreadsPerBlock, true>
        <<<numRows, kNumThreadsPerBlock, 0, stream>>>(
            logits.data_ptr<float>(), seqLens.data_ptr<int>(),
            indices.data_ptr<int>(), static_cast<int>(stride0),
            static_cast<int>(stride1), static_cast<int>(next_n));
  }
}

void top_k_per_row(const torch::Tensor& logits, const torch::Tensor& rowStarts,
                   const torch::Tensor& rowEnds, torch::Tensor& indices,
                   int64_t numRows, int64_t stride0, int64_t stride1) {
  constexpr int kSortingAlgorithmThreshold = 12288;
  constexpr int kNumThreadsPerBlock = 512;
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int numInsertionBlocks =
      std::min(static_cast<int>(numRows), kSortingAlgorithmThreshold);
  vllm::topKPerRow<kNumThreadsPerBlock, false>
      <<<numInsertionBlocks, kNumThreadsPerBlock, 0, stream>>>(
          logits.data_ptr<float>(), rowStarts.data_ptr<int>(),
          rowEnds.data_ptr<int>(), indices.data_ptr<int>(),
          static_cast<int>(stride0), static_cast<int>(stride1), 0);

  if (numRows > kSortingAlgorithmThreshold) {
    int numRadixBlocks = numRows - kSortingAlgorithmThreshold;
    vllm::topKPerRow<kNumThreadsPerBlock, true>
        <<<numRadixBlocks, kNumThreadsPerBlock, 0, stream>>>(
            logits.data_ptr<float>(), rowStarts.data_ptr<int>(),
            rowEnds.data_ptr<int>(), indices.data_ptr<int>(),
            static_cast<int>(stride0), static_cast<int>(stride1),
            kSortingAlgorithmThreshold);
  }
}
