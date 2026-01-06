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

__device__ __forceinline__ auto convert_to_uint32(float x) -> uint32_t {
  uint32_t bits = __float_as_uint(x);
  return (bits & 0x80000000) ? bits : ~bits & 0x7fffffff;
}

template <int step>
static inline __device__ uint32_t extractBinIdx(float x) {
  if constexpr (step == 0) {
    __half hx = __float2half(x);
    uint16_t bits = __half_as_ushort(hx);
    bits = (bits & 0x8000) ? bits : ~bits & 0x7fff;
    return bits >> 5;
  } else {
    uint32_t bits = __float_as_uint(x);
    bits = (bits & 0x80000000) ? bits : ~bits & 0x7fffffff;

    if constexpr (step == 1) {
      return bits >> 21;
    } else if constexpr (step == 2) {
      return (bits >> 10) & 0x7ff;
    } else if constexpr (step == 3) {
      return bits & 0x3ff;
    }
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

/**
 * Map a Func over the input data, using vectorized load instructions if
 * possible.
 *
 * @tparam T element type
 * @tparam IdxT indexing type
 * @tparam Func void (T x, IdxT idx)
 *
 * @param thread_rank rank of the calling thread among all participating threads
 * @param num_threads number of the threads that participate in processing
 * @param in the input data
 * @param len the number of elements to read
 * @param f the lambda taking two arguments (T x, IdxT idx)
 */
template <typename T, typename idxT, typename Func>
__device__ void vectorized_process(size_t thread_rank, size_t num_threads,
                                   const T* in, idxT len, Func f) {
  constexpr int WARP_SIZE = 32;
  using WideT = float4;
  if constexpr (sizeof(T) >= sizeof(WideT)) {
    for (idxT i = thread_rank; i < len; i += num_threads) {
      f(in[i], i);
    }
  } else {
    static_assert(sizeof(WideT) % sizeof(T) == 0);
    constexpr int items_per_scalar = sizeof(WideT) / sizeof(T);
    // TODO: it's UB
    union {
      WideT scalar;
      T array[items_per_scalar];
    } wide;

    int skip_cnt =
        (reinterpret_cast<size_t>(in) % sizeof(WideT))
            ? ((sizeof(WideT) - reinterpret_cast<size_t>(in) % sizeof(WideT)) /
               sizeof(T))
            : 0;
    if (skip_cnt > len) {
      skip_cnt = len;
    }
    const WideT* in_cast = reinterpret_cast<decltype(in_cast)>(in + skip_cnt);
    const idxT len_cast = (len - skip_cnt) / items_per_scalar;

    for (idxT i = thread_rank; i < len_cast; i += num_threads) {
      wide.scalar = in_cast[i];
      const idxT real_i = skip_cnt + i * items_per_scalar;
#pragma unroll
      for (int j = 0; j < items_per_scalar; ++j) {
        f(wide.array[j], real_i + j);
      }
    }

    static_assert(WARP_SIZE >= items_per_scalar);
    // and because items_per_scalar > skip_cnt, WARP_SIZE > skip_cnt
    // no need to use loop
    if (thread_rank < skip_cnt) {
      f(in[thread_rank], thread_rank);
    }
    // because len_cast = (len - skip_cnt) / items_per_scalar,
    // len_cast * items_per_scalar + items_per_scalar > len - skip_cnt;
    // and so
    // len - (skip_cnt + len_cast * items_per_scalar) < items_per_scalar <=
    // WARP_SIZE no need to use loop
    const idxT remain_i = skip_cnt + len_cast * items_per_scalar + thread_rank;
    if (remain_i < len) {
      f(in[remain_i], remain_i);
    }
  }
}

template <int step, int kNumThreadsPerBlock, int kNumBins, int kNumFinalItems,
          bool multipleBlocksPerRow, bool mergeBlocks, typename SmemFinalType,
          typename SmemOutputType>
__device__ bool processHistogramStep(
    const int* indices, const float* logits, int rowEnd, uint32_t& logitPattern,
    int& thresholdBinIdx, SmemOutputType& smemOutput, int* smemThresholdBinIdx,
    int* smemFinalDstIdx, int* smemFinalBinSize, int* smemFoundTopKValues,
    SmemFinalType& smemFinal, int stride1, int rowStart, int topK) {
  // Clear the histogram.
#pragma unroll
  for (int idx = threadIdx.x; idx < kNumBins; idx += kNumThreadsPerBlock) {
    smemFinal.histo.data[idx] = 0;
  }

  // Make sure the histogram is ready.
  __syncthreads();

  // Update pattern
  constexpr auto patternShift = step < 2 ? 0 : step == 2 ? 21 : 10;
  if constexpr (step == 2) {
    logitPattern = static_cast<uint32_t>(thresholdBinIdx & 0x7ff)
                   << patternShift;
  } else if constexpr (step == 3) {
    logitPattern |= static_cast<uint32_t>(thresholdBinIdx & 0x7ff)
                    << patternShift;
  }

  auto distributeToBins = [&](float logit, int /* idx */ = 0) {
    if (isPartialMatch<patternShift>(logit, logitPattern)) {
      uint32_t binIdx = extractBinIdx<step>(logit);
      atomicAdd(&smemFinal.histo.data[binIdx], 1);
    }
  };

  // Distribute the elements to the histogram bins.
  if (stride1 == 1) {
    vectorized_process(threadIdx.x, kNumThreadsPerBlock, logits + rowStart,
                       rowEnd - rowStart, distributeToBins);
  } else {
    for (int idx = rowStart + threadIdx.x; idx < rowEnd;
         idx += kNumThreadsPerBlock) {
      float logit = logits[idx * stride1];
      distributeToBins(logit, idx);
    }
  }
  // Make sure the histogram is ready.
  __syncthreads();

  // Reads the value of the starting position in the smemOutput array
  int lastValue = smemFoundTopKValues[0];

  for (int round = 0; round < kNumBins / kNumThreadsPerBlock; round++) {
    // Read the values from SMEM.
    int idx = threadIdx.x + kNumThreadsPerBlock * round;
    int binCount{0};
    binCount = smemFinal.histo.data[idx];

    // Make sure each thread has read its value.
    __syncthreads();

    // Compute the prefix sum.
    int prefixSum{0}, totalSum{0};
    using Scan = cub::BlockScan<int, kNumThreadsPerBlock>;
    Scan(smemFinal.histo.scan).ExclusiveSum(binCount, prefixSum, totalSum);

    // Update the histogram with the prefix sums.
    prefixSum += lastValue;
    totalSum += lastValue;
    smemFinal.histo.data[idx] = prefixSum;

    // Make sure the data is in shared memory.
    __syncthreads();

    // Find the last valid bin.
    bool foundThreshold = false;
    if (prefixSum < topK) {
      int nextPrefixSum = threadIdx.x == kNumThreadsPerBlock - 1
                              ? totalSum
                              : smemFinal.histo.data[idx + 1];

      if (nextPrefixSum >= topK) {
        smemThresholdBinIdx[0] = idx;
        smemFinalBinSize[0] = nextPrefixSum - prefixSum;
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

  auto processBins = [&](float logit, int idx) {
    if (isPartialMatch<patternShift>(logit, logitPattern)) {
      uint32_t binIdx = extractBinIdx<step>(logit);
      if (binIdx < thresholdBinIdx) {
        // The element is part of the top-k selection
        int dstIdx = atomicAdd(&smemFoundTopKValues[0], 1);

        if constexpr (mergeBlocks) {
          smemOutput[dstIdx] = indices[idx];
        } else if constexpr (multipleBlocksPerRow) {
          smemOutput[dstIdx] = idx + rowStart;
          reinterpret_cast<float*>(smemOutput + topK)[dstIdx] = logit;
        } else {
          smemOutput[dstIdx] = idx;
        }
      }
      if constexpr (step < 3) {
        // Only fill the final items for sorting if the threshold bin fits
        if (binIdx == thresholdBinIdx &&
            smemFinalBinSize[0] <= kNumFinalItems) {
          int dstIdx = atomicAdd(&smemFinalDstIdx[0], 1);
          smemFinal.items.logits[dstIdx] = logit;
          if constexpr (mergeBlocks) {
            smemFinal.items.indices[dstIdx] = indices[idx];
          } else if constexpr (multipleBlocksPerRow) {
            smemFinal.items.indices[dstIdx] = idx + rowStart;
          } else {
            smemFinal.items.indices[dstIdx] = idx;
          }
        }
      } else {
        if (binIdx == thresholdBinIdx) {
          // The elements in the threshold bin share the same 32 bits at step 3
          int dstIdx = atomicAdd(&smemFinal.histo.data[binIdx], 1);
          if (dstIdx < topK) {
            if constexpr (mergeBlocks) {
              smemOutput[dstIdx] = indices[idx];
            } else if constexpr (multipleBlocksPerRow) {
              smemOutput[dstIdx] = idx + rowStart;
              reinterpret_cast<float*>(smemOutput + topK)[dstIdx] = logit;
            } else {
              smemOutput[dstIdx] = idx;
            }
          }
        }
      }
    }
  };

  if (stride1 == 1) {
    vectorized_process(threadIdx.x, kNumThreadsPerBlock, logits + rowStart,
                       rowEnd - rowStart, processBins);
  } else {
    for (int idx = rowStart + threadIdx.x; idx < rowEnd;
         idx += kNumThreadsPerBlock) {
      float logit = logits[idx * stride1];
      processBins(logit, idx);
    }
  }

  // Make sure the elements are in shared memory.
  __syncthreads();

  // Check if we should continue to next step
  return smemFinalBinSize[0] > kNumFinalItems;
}

// Follows half - 11 - 11 - 10 bit iterations
template <int kNumThreadsPerBlock, int kNumBins, bool useRadixSort,
          bool multipleBlocksPerRow = false, bool mergeBlocks = false>
static __device__ void topKPerRowJob(const int* indices, const float* logits,
                                     int rowStart, int rowEnd, int* outIndices,
                                     float* outLogits, int stride1, int topK) {
  // The number of slots for the final pass.
  static constexpr int kNumFinalItems = 2048;
  // The number of elements per thread for the final sort.
  static constexpr int kNumFinalItemsPerThread =
      kNumFinalItems / kNumThreadsPerBlock;
  // The class to sort the elements during the final pass.
  using FinalSort = cub::BlockRadixSort<float, kNumThreadsPerBlock,
                                        kNumFinalItemsPerThread, int>;
  using FinalSortTempStorage =
      std::conditional_t<useRadixSort, typename FinalSort::TempStorage, int>;
  // The class to compute the inclusive prefix-sum over the histogram.
  using Scan = cub::BlockScan<int, kNumThreadsPerBlock>;

  // The structure to store the final items (for the final pass).
  struct FinalItems {
    // Shared memory to store the indices for the final pass.
    int indices[kNumFinalItems];
    // Shared memory to store the logits for the final pass.
    float logits[kNumFinalItems];
  };

  struct Histogram {
    typename Scan::TempStorage scan;
    int data[kNumBins];
  };

  // Shared memory to compute the block sort.
  __shared__ union {
    FinalItems items;
    FinalSortTempStorage finalSort;
    Histogram histo;
  } smemFinal;

  // Shared memory to store the selected indices.
  // If we are processing using multiple blocks, we need to store the logits and
  // indices.
  extern __shared__ int32_t smemOutput[];

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
  if (rowLen <= topK) {
    for (int rowIt = threadIdx.x; rowIt < rowLen;
         rowIt += kNumThreadsPerBlock) {
      if constexpr (multipleBlocksPerRow) {
        outIndices[rowIt] = rowIt + rowStart;
        outLogits[rowIt] = logits[rowIt + rowStart];
      } else {
        outIndices[rowIt] = rowIt;
      }
    }
    for (int rowIt = rowLen + threadIdx.x; rowIt < topK;
         rowIt += kNumThreadsPerBlock) {
      outIndices[rowIt] = -1;
      if constexpr (multipleBlocksPerRow) {
        outLogits[rowIt] = -FLT_MAX;
      }
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

  // Step 0: Process first 11 bits of half representation
  bool continueToNextStep =
      processHistogramStep<0, kNumThreadsPerBlock, kNumBins, kNumFinalItems,
                           multipleBlocksPerRow, mergeBlocks>(
          indices, logits, rowEnd, logitPattern, thresholdBinIdx, smemOutput,
          smemThresholdBinIdx, smemFinalDstIdx, smemFinalBinSize,
          smemFoundTopKValues, smemFinal, stride1, rowStart, topK);

  if (continueToNextStep) {
    // Step 1: Process next 11 bits
    continueToNextStep =
        processHistogramStep<1, kNumThreadsPerBlock, kNumBins, kNumFinalItems,
                             multipleBlocksPerRow, mergeBlocks>(
            indices, logits, rowEnd, logitPattern, thresholdBinIdx, smemOutput,
            smemThresholdBinIdx, smemFinalDstIdx, smemFinalBinSize,
            smemFoundTopKValues, smemFinal, stride1, rowStart, topK);
  }

  if (continueToNextStep) {
    // Step 2: Process next 11 bits
    continueToNextStep =
        processHistogramStep<2, kNumThreadsPerBlock, kNumBins, kNumFinalItems,
                             multipleBlocksPerRow, mergeBlocks>(
            indices, logits, rowEnd, logitPattern, thresholdBinIdx, smemOutput,
            smemThresholdBinIdx, smemFinalDstIdx, smemFinalBinSize,
            smemFoundTopKValues, smemFinal, stride1, rowStart, topK);
  }

  if (continueToNextStep) {
    // Step 3: Process last 10 bits
    processHistogramStep<3, kNumThreadsPerBlock, kNumBins, kNumFinalItems,
                         multipleBlocksPerRow, mergeBlocks>(
        indices, logits, rowEnd, logitPattern, thresholdBinIdx, smemOutput,
        smemThresholdBinIdx, smemFinalDstIdx, smemFinalBinSize,
        smemFoundTopKValues, smemFinal, stride1, rowStart, topK);
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

        if (dstIdx < topK) {
          smemOutput[dstIdx] = finalIndices[ii];
          if constexpr (multipleBlocksPerRow) {
            reinterpret_cast<float*>(smemOutput + topK)[dstIdx] =
                finalLogits[ii];
          }
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
        if (outIndex + baseIdx < topK) {
          smemOutput[outIndex + baseIdx] = smemFinal.items.indices[i];
          if constexpr (multipleBlocksPerRow) {
            reinterpret_cast<float*>(smemOutput + topK)[outIndex + baseIdx] =
                smemFinal.items.logits[i];
          }
        }
      }
    }
    __syncthreads();
  }

  // Store to global memory.
  for (int i = threadIdx.x; i < topK; i += kNumThreadsPerBlock) {
    if constexpr (multipleBlocksPerRow) {
      outIndices[i] = smemOutput[i];
      outLogits[i] = reinterpret_cast<float*>(smemOutput + topK)[i];
    } else {
      if (stride1 == 1) {
        // stride1 == 1 will use vectorized_process, which indexes already skip
        // the rowStart.
        outIndices[i] = smemOutput[i];
      } else {
        outIndices[i] = smemOutput[i] - rowStart;
      }
    }
  }
}

template <int kNumThreadsPerBlock, bool useRadixSort>
static __global__ __launch_bounds__(kNumThreadsPerBlock) void topKPerRowPrefill(
    const float* logits, const int* rowStarts, const int* rowEnds,
    int* outIndices, int stride0, int stride1, const int topK,
    const int offsetIndex) {
  // The number of bins in the histogram.
  static constexpr int kNumBins = 2048;

  // The row computed by this block.
  int rowIdx = blockIdx.x + offsetIndex;

  // The range of logits within the row.
  int rowStart = rowStarts[rowIdx];
  int rowEnd = rowEnds[rowIdx];

  // Local pointers to this block
  outIndices += static_cast<int64_t>(rowIdx) * topK;
  logits += static_cast<int64_t>(rowIdx) * stride0;

  topKPerRowJob<kNumThreadsPerBlock, kNumBins, useRadixSort>(
      nullptr, logits, rowStart, rowEnd, outIndices, nullptr, stride1, topK);
}

template <int kNumThreadsPerBlock, bool useRadixSort,
          bool multipleBlocksPerRow = false, bool mergeBlocks = false>
static __global__ __launch_bounds__(kNumThreadsPerBlock) void topKPerRowDecode(
    const float* logits, const int* seqLens, int* outIndices, int stride0,
    int stride1, const int topK, int next_n, float* outLogits = nullptr,
    const int numBlocksToMerge = 0, const int* indices = nullptr) {
  // The number of bins in the histogram.
  static constexpr int kNumBins = 2048;

  // The row computed by this block.
  int rowIdx = blockIdx.x;

  // The range of logits within the row.
  int rowStart = 0;
  int seq_len = seqLens[rowIdx / next_n];
  int rowEnd = seq_len - next_n + (rowIdx % next_n) + 1;

  // Local pointers to this block
  if constexpr (!multipleBlocksPerRow && !mergeBlocks) {
    outIndices += static_cast<int64_t>(rowIdx) * topK;
  } else if constexpr (multipleBlocksPerRow) {
    const auto blockSize = rowEnd / gridDim.y;  // 16384 / 2 = 8192
    rowStart = blockSize * blockIdx.y;          // 8192 * 1 = 8192
    rowEnd = gridDim.y == blockIdx.y + 1 ? rowEnd : rowStart + blockSize;
    outIndices +=
        static_cast<int64_t>(rowIdx) * gridDim.y * topK + blockIdx.y * topK;
    outLogits +=
        static_cast<int64_t>(rowIdx) * gridDim.y * topK + blockIdx.y * topK;
  } else if constexpr (mergeBlocks) {
    rowEnd = numBlocksToMerge * topK;
    indices += static_cast<int64_t>(rowIdx) * numBlocksToMerge * topK;
    outIndices += static_cast<int64_t>(rowIdx) * topK;
  }
  logits += static_cast<int64_t>(rowIdx) * stride0;

  topKPerRowJob<kNumThreadsPerBlock, kNumBins, useRadixSort,
                multipleBlocksPerRow, mergeBlocks>(
      indices, logits, rowStart, rowEnd, outIndices, outLogits, stride1, topK);
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
                          int64_t numRows, int64_t stride0, int64_t stride1,
                          int64_t topK) {
  constexpr int kSortingAlgorithmThreshold = 12288;
  constexpr int kSplitWorkThreshold = 200 * 1000;
  constexpr int kNumThreadsPerBlock = 512;
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const auto numColumns = logits.size(1);

  if (numColumns < kSortingAlgorithmThreshold) {
    // Use insertion sort
    vllm::topKPerRowDecode<kNumThreadsPerBlock, false>
        <<<numRows, kNumThreadsPerBlock, topK * sizeof(int32_t), stream>>>(
            logits.data_ptr<float>(), seqLens.data_ptr<int>(),
            indices.data_ptr<int>(), static_cast<int>(stride0),
            static_cast<int>(stride1), static_cast<int>(topK),
            static_cast<int>(next_n));
  } else if (numColumns < kSplitWorkThreshold) {
    // From this threshold, use radix sort instead
    vllm::topKPerRowDecode<kNumThreadsPerBlock, true>
        <<<numRows, kNumThreadsPerBlock, topK * sizeof(int32_t), stream>>>(
            logits.data_ptr<float>(), seqLens.data_ptr<int>(),
            indices.data_ptr<int>(), static_cast<int>(stride0),
            static_cast<int>(stride1), static_cast<int>(topK),
            static_cast<int>(next_n));
  } else {
    // Long sequences are run in two steps
    constexpr auto multipleBlocksPerRowConfig = 10;

    const auto outIndicesAux =
        torch::empty({numRows, multipleBlocksPerRowConfig, topK},
                     torch::dtype(torch::kInt32).device(logits.device()));
    const auto outLogitsAux =
        torch::empty({numRows, multipleBlocksPerRowConfig, topK},
                     torch::dtype(torch::kFloat).device(logits.device()));

    vllm::topKPerRowDecode<kNumThreadsPerBlock, true, true>
        <<<dim3(numRows, multipleBlocksPerRowConfig), kNumThreadsPerBlock,
           2 * topK * sizeof(int32_t), stream>>>(
            logits.data_ptr<float>(), seqLens.data_ptr<int>(),
            outIndicesAux.data_ptr<int>(), static_cast<int>(stride0),
            static_cast<int>(stride1), static_cast<int>(topK),
            static_cast<int>(next_n), outLogitsAux.data_ptr<float>());

    constexpr int kNumThreadsPerBlockMerge = 1024;
    vllm::topKPerRowDecode<kNumThreadsPerBlockMerge, true, false, true>
        <<<numRows, kNumThreadsPerBlockMerge, topK * sizeof(int32_t), stream>>>(
            outLogitsAux.data_ptr<float>(), seqLens.data_ptr<int>(),
            indices.data_ptr<int>(), multipleBlocksPerRowConfig * topK, 1,
            static_cast<int>(topK), static_cast<int>(next_n), nullptr,
            multipleBlocksPerRowConfig, outIndicesAux.data_ptr<int>());
  }
}

void top_k_per_row_prefill(const torch::Tensor& logits,
                           const torch::Tensor& rowStarts,
                           const torch::Tensor& rowEnds, torch::Tensor& indices,
                           int64_t numRows, int64_t stride0, int64_t stride1,
                           int64_t topK) {
  constexpr int kSortingAlgorithmThreshold = 12288;
  constexpr int kNumThreadsPerBlock = 512;
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int numInsertionBlocks =
      std::min(static_cast<int>(numRows), kSortingAlgorithmThreshold);
  vllm::topKPerRowPrefill<kNumThreadsPerBlock, false>
      <<<numInsertionBlocks, kNumThreadsPerBlock, topK * sizeof(int32_t),
         stream>>>(logits.data_ptr<float>(), rowStarts.data_ptr<int>(),
                   rowEnds.data_ptr<int>(), indices.data_ptr<int>(),
                   static_cast<int>(stride0), static_cast<int>(stride1),
                   static_cast<int>(topK), 0);

  if (numRows > kSortingAlgorithmThreshold) {
    int numRadixBlocks = numRows - kSortingAlgorithmThreshold;
    vllm::topKPerRowPrefill<kNumThreadsPerBlock, true>
        <<<numRadixBlocks, kNumThreadsPerBlock, topK * sizeof(int32_t),
           stream>>>(logits.data_ptr<float>(), rowStarts.data_ptr<int>(),
                     rowEnds.data_ptr<int>(), indices.data_ptr<int>(),
                     static_cast<int>(stride0), static_cast<int>(stride1),
                     static_cast<int>(topK), kSortingAlgorithmThreshold);
  }
}
