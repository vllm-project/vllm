#include "../cuda_compat.h"
#include "dispatch_utils.h"
#include "torch_utils.h"

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
  // Use dynamic WARP_SIZE from cuda_compat.h to support both
  // Wave64 (MI300X/gfx942) and Wave32 (Strix Halo/gfx1151) architectures
  constexpr int kWarpSize = WARP_SIZE;
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

    static_assert(kWarpSize >= items_per_scalar);
    // and because items_per_scalar > skip_cnt, kWarpSize > skip_cnt
    // no need to use loop
    if (thread_rank < skip_cnt) {
      f(in[thread_rank], thread_rank);
    }
    // because len_cast = (len - skip_cnt) / items_per_scalar,
    // len_cast * items_per_scalar + items_per_scalar > len - skip_cnt;
    // and so
    // len - (skip_cnt + len_cast * items_per_scalar) < items_per_scalar <=
    // kWarpSize no need to use loop
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
      // Only write elements with binIdx < thresholdBinIdx when:
      // 1. This is step 0 and the threshold bin is small enough (no step 1)
      // 2. This is step >= 1 (where pattern matching filters correctly)
      // This prevents duplicates when step 0 and step 1 both run.
      bool shouldWriteDirectly =
          (step == 0 && smemFinalBinSize[0] <= kNumFinalItems) || (step >= 1);
      if (binIdx < thresholdBinIdx && shouldWriteDirectly) {
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
    int stride1, const int topK, int next_n, int seqLensIs2D = 0,
    float* outLogits = nullptr, const int numBlocksToMerge = 0,
    const int* indices = nullptr) {
  // The number of bins in the histogram.
  static constexpr int kNumBins = 2048;

  // The row computed by this block.
  int rowIdx = blockIdx.x;

  // The range of logits within the row.
  int rowStart = 0;
  int batch_idx = rowIdx / next_n;
  int next_n_idx = rowIdx % next_n;
  // seqLensIs2D=0: 1D seqLens — all rows in a batch share the same seq_len;
  //               kernel computes per-row effective length via offset.
  // seqLensIs2D=1: 2D seqLens — each logit row has its own pre-computed
  //               effective length (flat index rowIdx = b*next_n + j maps
  //               directly to seqLens[b, j] in C-contiguous layout).
  int seq_len = seqLensIs2D ? seqLens[rowIdx] : seqLens[batch_idx];
  int rowEnd =
      seqLensIs2D ? max(0, seq_len) : max(0, seq_len - next_n + next_n_idx + 1);

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

static constexpr int kStableTopKThreads = 512;
static constexpr int kStableTopKMaxK = 2048;
static constexpr int kStableTopKBins = 2048;
static constexpr int kStableTopKItemsPerThread =
    kStableTopKMaxK / kStableTopKThreads;

__device__ __forceinline__ uint32_t stable_ordered_float_bits(float score) {
  uint32_t bits = __float_as_uint(score);
  uint32_t mask = (bits & 0x80000000u) ? 0xffffffffu : 0x80000000u;
  return bits ^ mask;
}

__device__ __forceinline__ uint64_t stable_topk_key(float score,
                                                    int32_t token_id) {
  if (token_id < 0) {
    return 0;
  }
  const uint64_t score_key =
      static_cast<uint64_t>(stable_ordered_float_bits(score)) << 32;
  const uint64_t id_key =
      static_cast<uint64_t>(~static_cast<uint32_t>(token_id));
  return score_key | id_key;
}

__device__ __forceinline__ bool stable_prefix_matches(uint64_t key,
                                                      uint64_t prefix,
                                                      int prefix_bits) {
  if (prefix_bits == 0) {
    return true;
  }
  return (key >> (64 - prefix_bits)) == (prefix >> (64 - prefix_bits));
}

template <bool HAS_TOKEN_IDS>
__device__ __forceinline__ uint64_t load_stable_topk_key(
    const float* __restrict__ scores, const int32_t* __restrict__ token_ids,
    const int32_t* __restrict__ seq_lens, int row, int col,
    int64_t score_stride0, int64_t score_stride1, int64_t id_stride0,
    int64_t id_stride1) {
  int32_t token_id;
  if constexpr (HAS_TOKEN_IDS) {
    token_id = token_ids[static_cast<int64_t>(row) * id_stride0 +
                         static_cast<int64_t>(col) * id_stride1];
  } else {
    const int32_t seq_len = seq_lens[row];
    token_id = col < seq_len ? col : -1;
  }
  const float score = scores[static_cast<int64_t>(row) * score_stride0 +
                             static_cast<int64_t>(col) * score_stride1];
  return stable_topk_key(score, token_id);
}

template <bool HAS_TOKEN_IDS>
__device__ __forceinline__ int32_t load_stable_topk_token_id(
    const int32_t* __restrict__ token_ids, const int32_t* __restrict__ seq_lens,
    int col, int row, int64_t id_stride0, int64_t id_stride1) {
  if constexpr (HAS_TOKEN_IDS) {
    return token_ids[static_cast<int64_t>(row) * id_stride0 +
                     static_cast<int64_t>(col) * id_stride1];
  } else {
    return col < seq_lens[row] ? col : -1;
  }
}

template <bool HAS_TOKEN_IDS>
static __global__
__launch_bounds__(kStableTopKThreads) void stableTopKByKeyKernel(
    const float* __restrict__ scores, const int32_t* __restrict__ token_ids,
    const int32_t* __restrict__ seq_lens, int32_t* __restrict__ out_indices,
    int num_cols, int64_t score_stride0, int64_t score_stride1,
    int64_t id_stride0, int64_t id_stride1, int topK) {
  using Sort = cub::BlockRadixSort<uint64_t, kStableTopKThreads,
                                   kStableTopKItemsPerThread, int32_t>;
  using Scan = cub::BlockScan<int, kStableTopKThreads>;

  struct HistogramScratch {
    typename Scan::TempStorage scan;
    int data[kStableTopKBins];
  };
  __shared__ union {
    HistogramScratch hist;
    typename Sort::TempStorage sort;
  } scratch;
  __shared__ uint64_t final_keys[kStableTopKMaxK];
  __shared__ int32_t final_ids[kStableTopKMaxK];
  __shared__ int selected_count;
  __shared__ int final_count;
  __shared__ int threshold_bin;
  __shared__ int threshold_count;
  __shared__ uint64_t prefix;
  __shared__ int prefix_bits;

  const int row = blockIdx.x;
  const int tid = threadIdx.x;

  if constexpr (!HAS_TOKEN_IDS) {
    const int row_len = min(max(seq_lens[row], 0), num_cols);
    if (row_len <= topK) {
      for (int i = tid; i < topK; i += kStableTopKThreads) {
        out_indices[static_cast<int64_t>(row) * topK + i] =
            i < row_len ? i : -1;
      }
      return;
    }
  }

  if (tid == 0) {
    selected_count = 0;
    prefix = 0;
    prefix_bits = 0;
  }
  __syncthreads();

  for (int step = 0; step < 6; ++step) {
    const int bits = step == 5 ? 9 : 11;
    const int num_bins = 1 << bits;
    const int shift = 64 - prefix_bits - bits;
    const uint64_t bin_mask = static_cast<uint64_t>(num_bins - 1);

    for (int i = tid; i < kStableTopKBins; i += kStableTopKThreads) {
      scratch.hist.data[i] = 0;
    }
    if (tid == 0) {
      final_count = 0;
      threshold_bin = 0;
      threshold_count = 0;
    }
    __syncthreads();

    for (int col = tid; col < num_cols; col += kStableTopKThreads) {
      const uint64_t key = load_stable_topk_key<HAS_TOKEN_IDS>(
          scores, token_ids, seq_lens, row, col, score_stride0, score_stride1,
          id_stride0, id_stride1);
      if (stable_prefix_matches(key, prefix, prefix_bits)) {
        const int bin = static_cast<int>((key >> shift) & bin_mask);
        atomicAdd(&scratch.hist.data[bin], 1);
      }
    }
    __syncthreads();

    int running = selected_count;
    for (int round = 0;
         round < (num_bins + kStableTopKThreads - 1) / kStableTopKThreads;
         ++round) {
      const int descending_idx = round * kStableTopKThreads + tid;
      const int bin = num_bins - 1 - descending_idx;
      const int count = bin >= 0 ? scratch.hist.data[bin] : 0;
      int prefix_sum = 0;
      int round_total = 0;
      Scan(scratch.hist.scan).ExclusiveSum(count, prefix_sum, round_total);
      prefix_sum += running;
      round_total += running;

      const bool found =
          bin >= 0 && prefix_sum < topK && prefix_sum + count >= topK;
      if (found) {
        threshold_bin = bin;
        threshold_count = count;
      }
      if (__syncthreads_or(found)) {
        break;
      }
      running = round_total;
      __syncthreads();
    }
    __syncthreads();

    const bool finish = threshold_count <= kStableTopKMaxK || step == 5;
    for (int col = tid; col < num_cols; col += kStableTopKThreads) {
      const uint64_t key = load_stable_topk_key<HAS_TOKEN_IDS>(
          scores, token_ids, seq_lens, row, col, score_stride0, score_stride1,
          id_stride0, id_stride1);
      if (!stable_prefix_matches(key, prefix, prefix_bits)) {
        continue;
      }
      const int bin = static_cast<int>((key >> shift) & bin_mask);
      if (bin > threshold_bin) {
        const int dst = atomicAdd(&selected_count, 1);
        if (dst < topK) {
          out_indices[static_cast<int64_t>(row) * topK + dst] =
              load_stable_topk_token_id<HAS_TOKEN_IDS>(
                  token_ids, seq_lens, col, row, id_stride0, id_stride1);
        }
      } else if (bin == threshold_bin && finish) {
        const int dst = atomicAdd(&final_count, 1);
        if (dst < kStableTopKMaxK) {
          final_keys[dst] = key;
          final_ids[dst] = load_stable_topk_token_id<HAS_TOKEN_IDS>(
              token_ids, seq_lens, col, row, id_stride0, id_stride1);
        }
      }
    }
    __syncthreads();

    if (finish) {
      uint64_t thread_keys[kStableTopKItemsPerThread];
      int32_t thread_ids[kStableTopKItemsPerThread];
#pragma unroll
      for (int ii = 0; ii < kStableTopKItemsPerThread; ++ii) {
        const int src = ii * kStableTopKThreads + tid;
        if (src < final_count && src < kStableTopKMaxK) {
          thread_keys[ii] = final_keys[src];
          thread_ids[ii] = final_ids[src];
        } else {
          thread_keys[ii] = 0;
          thread_ids[ii] = -1;
        }
      }
      Sort(scratch.sort)
          .SortDescendingBlockedToStriped(thread_keys, thread_ids);
      __syncthreads();

      const int base = selected_count;
      const int need = max(0, topK - base);
#pragma unroll
      for (int ii = 0; ii < kStableTopKItemsPerThread; ++ii) {
        const int rank = ii * kStableTopKThreads + tid;
        if (rank < need && rank < final_count) {
          out_indices[static_cast<int64_t>(row) * topK + base + rank] =
              thread_ids[ii];
        }
      }
      __syncthreads();

      if (tid == 0) {
        selected_count = min(topK, base + min(need, final_count));
      }
      __syncthreads();

      for (int i = selected_count + tid; i < topK; i += kStableTopKThreads) {
        out_indices[static_cast<int64_t>(row) * topK + i] = -1;
      }
      return;
    }

    if (tid == 0) {
      prefix |= static_cast<uint64_t>(threshold_bin) << shift;
      prefix_bits += bits;
    }
    __syncthreads();
  }
}

}  // namespace vllm

void apply_repetition_penalties_(
    torch::stable::Tensor& logits,  // [num_seqs, vocab_size], in-place
    const torch::stable::Tensor& prompt_mask,  // [num_seqs, vocab_size]
    const torch::stable::Tensor& output_mask,  // [num_seqs, vocab_size]
    const torch::stable::Tensor& repetition_penalties) {  // [num_seqs]
  STD_TORCH_CHECK(logits.is_contiguous());
  STD_TORCH_CHECK(prompt_mask.is_contiguous());
  STD_TORCH_CHECK(output_mask.is_contiguous());
  STD_TORCH_CHECK(repetition_penalties.is_contiguous());

  int vocab_size = logits.size(-1);
  int num_seqs = logits.size(0);

  if (num_seqs == 0) return;

  // Get number of SMs on the current device
  int sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount,
                         logits.get_device_index());

  // Compute tile_num and tile_size
  int tile_num =
      std::min(vocab_size, std::max(1, (sms + num_seqs - 1) / num_seqs));
  int tile_size = (vocab_size + tile_num - 1) / tile_num;

  // Each block handles one sequence and a tile of vocab
  dim3 grid(num_seqs, tile_num);
  dim3 block(std::min(tile_size, 1024));
  const torch::stable::accelerator::DeviceGuard device_guard(
      logits.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream();
  VLLM_STABLE_DISPATCH_FLOATING_TYPES(
      logits.scalar_type(), "apply_repetition_penalties_kernel", [&] {
        vllm::apply_repetition_penalties_kernel<scalar_t>
            <<<grid, block, 0, stream>>>(
                logits.mutable_data_ptr<scalar_t>(),
                prompt_mask.const_data_ptr<bool>(),
                output_mask.const_data_ptr<bool>(),
                repetition_penalties.const_data_ptr<scalar_t>(), num_seqs,
                vocab_size, tile_size);
      });
}

void top_k_per_row_decode(const torch::stable::Tensor& logits, int64_t next_n,
                          const torch::stable::Tensor& seqLens,
                          torch::stable::Tensor& indices, int64_t numRows,
                          int64_t stride0, int64_t stride1, int64_t topK) {
  constexpr int kSortingAlgorithmThreshold = 12288;
  constexpr int kSplitWorkThreshold = 200 * 1000;
  constexpr int kNumThreadsPerBlock = 512;
  const cudaStream_t stream = get_current_cuda_stream();
  const auto numColumns = logits.size(1);

  // True if seqLens is 2D (B, next_n): each logit row has its own pre-computed
  // effective seq_len. False if seqLens is 1D (B,): all rows in a batch share
  // the same seq_len and the kernel computes the per-row offset itself.
  int seqLensIs2D = seqLens.dim() == 2 ? 1 : 0;

  if (numColumns < kSortingAlgorithmThreshold) {
    // Use insertion sort
    vllm::topKPerRowDecode<kNumThreadsPerBlock, false>
        <<<numRows, kNumThreadsPerBlock, topK * sizeof(int32_t), stream>>>(
            logits.const_data_ptr<float>(), seqLens.const_data_ptr<int>(),
            indices.mutable_data_ptr<int>(), static_cast<int>(stride0),
            static_cast<int>(stride1), static_cast<int>(topK),
            static_cast<int>(next_n), seqLensIs2D);
  } else if (numColumns < kSplitWorkThreshold) {
    // From this threshold, use radix sort instead
    vllm::topKPerRowDecode<kNumThreadsPerBlock, true>
        <<<numRows, kNumThreadsPerBlock, topK * sizeof(int32_t), stream>>>(
            logits.const_data_ptr<float>(), seqLens.const_data_ptr<int>(),
            indices.mutable_data_ptr<int>(), static_cast<int>(stride0),
            static_cast<int>(stride1), static_cast<int>(topK),
            static_cast<int>(next_n), seqLensIs2D);
  } else {
    // Long sequences are run in two steps
    constexpr auto multipleBlocksPerRowConfig = 10;

    const auto outIndicesAux = torch::stable::empty(
        {numRows, multipleBlocksPerRowConfig, topK},
        torch::headeronly::ScalarType::Int, std::nullopt, logits.device());
    const auto outLogitsAux = torch::stable::empty(
        {numRows, multipleBlocksPerRowConfig, topK},
        torch::headeronly::ScalarType::Float, std::nullopt, logits.device());

    vllm::topKPerRowDecode<kNumThreadsPerBlock, true, true>
        <<<dim3(numRows, multipleBlocksPerRowConfig), kNumThreadsPerBlock,
           2 * topK * sizeof(int32_t), stream>>>(
            logits.const_data_ptr<float>(), seqLens.const_data_ptr<int>(),
            outIndicesAux.mutable_data_ptr<int>(), static_cast<int>(stride0),
            static_cast<int>(stride1), static_cast<int>(topK),
            static_cast<int>(next_n), seqLensIs2D,
            outLogitsAux.mutable_data_ptr<float>());

    constexpr int kNumThreadsPerBlockMerge = 1024;
    vllm::topKPerRowDecode<kNumThreadsPerBlockMerge, true, false, true>
        <<<numRows, kNumThreadsPerBlockMerge, topK * sizeof(int32_t), stream>>>(
            outLogitsAux.const_data_ptr<float>(), seqLens.const_data_ptr<int>(),
            indices.mutable_data_ptr<int>(), multipleBlocksPerRowConfig * topK,
            1, static_cast<int>(topK), static_cast<int>(next_n), seqLensIs2D,
            nullptr, multipleBlocksPerRowConfig,
            outIndicesAux.const_data_ptr<int>());
  }
}

void stable_top_k_per_row(const torch::stable::Tensor& scores,
                          const torch::stable::Tensor& seq_lens,
                          torch::stable::Tensor& indices, int64_t numRows,
                          int64_t stride0, int64_t stride1, int64_t topK) {
#ifndef USE_ROCM
  STD_TORCH_CHECK(scores.scalar_type() == torch::headeronly::ScalarType::Float,
                  "stable_top_k_per_row expects fp32 scores");
  STD_TORCH_CHECK(seq_lens.scalar_type() == torch::headeronly::ScalarType::Int,
                  "stable_top_k_per_row expects int32 seq_lens");
  STD_TORCH_CHECK(indices.scalar_type() == torch::headeronly::ScalarType::Int,
                  "stable_top_k_per_row expects int32 output indices");
  STD_TORCH_CHECK(topK > 0 && topK <= vllm::kStableTopKMaxK,
                  "stable_top_k_per_row supports 1 <= k <= ",
                  vllm::kStableTopKMaxK, ", got k=", topK);
  if (numRows == 0) {
    return;
  }
  const auto num_cols = scores.size(1);
  const cudaStream_t stream = get_current_cuda_stream();
  vllm::stableTopKByKeyKernel<false>
      <<<static_cast<int>(numRows), vllm::kStableTopKThreads, 0, stream>>>(
          scores.const_data_ptr<float>(), nullptr,
          seq_lens.const_data_ptr<int32_t>(),
          indices.mutable_data_ptr<int32_t>(), static_cast<int>(num_cols),
          stride0, stride1, 0, 0, static_cast<int>(topK));
#else
  STD_TORCH_CHECK(false, "stable_top_k_per_row is not supported on ROCm");
#endif
}

void stable_top_k_from_candidates(const torch::stable::Tensor& scores,
                                  const torch::stable::Tensor& token_ids,
                                  torch::stable::Tensor& indices,
                                  int64_t numRows, int64_t score_stride0,
                                  int64_t score_stride1, int64_t id_stride0,
                                  int64_t id_stride1, int64_t topK) {
#ifndef USE_ROCM
  STD_TORCH_CHECK(scores.scalar_type() == torch::headeronly::ScalarType::Float,
                  "stable_top_k_from_candidates expects fp32 scores");
  STD_TORCH_CHECK(token_ids.scalar_type() == torch::headeronly::ScalarType::Int,
                  "stable_top_k_from_candidates expects int32 token_ids");
  STD_TORCH_CHECK(indices.scalar_type() == torch::headeronly::ScalarType::Int,
                  "stable_top_k_from_candidates expects int32 output indices");
  STD_TORCH_CHECK(topK > 0 && topK <= vllm::kStableTopKMaxK,
                  "stable_top_k_from_candidates supports 1 <= k <= ",
                  vllm::kStableTopKMaxK, ", got k=", topK);
  if (numRows == 0) {
    return;
  }
  const auto num_cols = scores.size(1);
  const cudaStream_t stream = get_current_cuda_stream();
  vllm::stableTopKByKeyKernel<true>
      <<<static_cast<int>(numRows), vllm::kStableTopKThreads, 0, stream>>>(
          scores.const_data_ptr<float>(), token_ids.const_data_ptr<int32_t>(),
          nullptr, indices.mutable_data_ptr<int32_t>(),
          static_cast<int>(num_cols), score_stride0, score_stride1, id_stride0,
          id_stride1, static_cast<int>(topK));
#else
  STD_TORCH_CHECK(false,
                  "stable_top_k_from_candidates is not supported on ROCm");
#endif
}

void top_k_per_row_prefill(const torch::stable::Tensor& logits,
                           const torch::stable::Tensor& rowStarts,
                           const torch::stable::Tensor& rowEnds,
                           torch::stable::Tensor& indices, int64_t numRows,
                           int64_t stride0, int64_t stride1, int64_t topK) {
  constexpr int kSortingAlgorithmThreshold = 12288;
  constexpr int kNumThreadsPerBlock = 512;
  const cudaStream_t stream = get_current_cuda_stream();

  int numInsertionBlocks =
      std::min(static_cast<int>(numRows), kSortingAlgorithmThreshold);
  vllm::topKPerRowPrefill<kNumThreadsPerBlock, false>
      <<<numInsertionBlocks, kNumThreadsPerBlock, topK * sizeof(int32_t),
         stream>>>(logits.const_data_ptr<float>(),
                   rowStarts.const_data_ptr<int>(),
                   rowEnds.const_data_ptr<int>(),
                   indices.mutable_data_ptr<int>(), static_cast<int>(stride0),
                   static_cast<int>(stride1), static_cast<int>(topK), 0);

  if (numRows > kSortingAlgorithmThreshold) {
    int numRadixBlocks = numRows - kSortingAlgorithmThreshold;
    vllm::topKPerRowPrefill<kNumThreadsPerBlock, true>
        <<<numRadixBlocks, kNumThreadsPerBlock, topK * sizeof(int32_t),
           stream>>>(
            logits.const_data_ptr<float>(), rowStarts.const_data_ptr<int>(),
            rowEnds.const_data_ptr<int>(), indices.mutable_data_ptr<int>(),
            static_cast<int>(stride0), static_cast<int>(stride1),
            static_cast<int>(topK), kSortingAlgorithmThreshold);
  }
}
