// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// The sparse indexer's decode top-512 for gfx942 (MI300X and MI325X).
//
// The helpers, processHistogramStep and topKPerRowJob in namespace vllm are
// copied unchanged from vLLM's csrc/libtorch_stable/sampler.cu (lines 48 to
// 571, the same from commit 92044241a02f to 72d59adc2c76). The installed vLLM
// package has no csrc directory, so the extension that torch builds at run
// time needs the whole source in this file. topK512Decode is vLLM's
// topKPerRowDecode512DeviceLengthAware with two changes. vLLM compiles its body
// for gfx950 only, so this file drops that guard. The number of blocks that a
// row is split into comes from the launch, not from vLLM's gfx950 table.
//
// Why this exists: on gfx942, vLLM's top_k_per_row_decode takes its generic
// 10-block path. That path adds every logit to a shared-memory histogram bin
// with an atomic. The DSpark candidate mask leaves about 7 of every 8 logits of
// a layer 24 to 36 row at -inf, and they all hit one bin, so at 128k context
// that call takes 100 us for 6 rows. The top-512 path counts -inf logits in a
// register and adds them to the bin once.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>

#include <cfloat>
#include <cub/cub.cuh>

// gfx942 runs 64-lane waves. vLLM takes this from cuda_compat.h.
#define WARP_SIZE 64

namespace vllm {

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
          bool multipleBlocksPerRow, bool mergeBlocks,
          bool useTopK512Optimization, typename SmemFinalType,
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

  int negativeInfinityCount = 0;
  auto distributeToBins = [&](float logit, int /* idx */ = 0) {
    if (isPartialMatch<patternShift>(logit, logitPattern)) {
      uint32_t binIdx = extractBinIdx<step>(logit);
      if (useTopK512Optimization && __float_as_uint(logit) == 0xff800000u) {
        ++negativeInfinityCount;
      } else {
        atomicAdd(&smemFinal.histo.data[binIdx], 1);
      }
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
  if constexpr (useTopK512Optimization) {
    if (negativeInfinityCount > 0) {
      const auto bin = extractBinIdx<step>(-INFINITY);
      atomicAdd(&smemFinal.histo.data[bin], negativeInfinityCount);
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
  // Resolve a -inf cutoff with exact tie emission, keeping sort padding out.
  const bool finalBinFits =
      smemFinalBinSize[0] <= kNumFinalItems &&
      !(useTopK512Optimization && step < 3 &&
        isPartialMatch<patternShift>(-INFINITY, logitPattern) &&
        thresholdBinIdx == extractBinIdx<step>(-INFINITY));

  auto processBins = [&](float logit, int idx) {
    if (isPartialMatch<patternShift>(logit, logitPattern)) {
      uint32_t binIdx = extractBinIdx<step>(logit);
      // Only write elements with binIdx < thresholdBinIdx when:
      // 1. This is step 0 and the threshold bin is small enough (no step 1)
      // 2. This is step >= 1 (where pattern matching filters correctly)
      // This prevents duplicates when step 0 and step 1 both run.
      bool shouldWriteDirectly = (step == 0 && finalBinFits) || (step >= 1);
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
        if (binIdx == thresholdBinIdx && finalBinFits) {
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
  return !finalBinFits;
}

// Follows half - 11 - 11 - 10 bit iterations
// Keep the adaptive kernel's device instantiation separate from legacy calls.
template <int kNumThreadsPerBlock, int kNumBins, bool useRadixSort,
          bool multipleBlocksPerRow = false, bool mergeBlocks = false,
          bool deviceLengthAware = false, bool useTopK512Optimization = false>
static __device__ void topKPerRowJob(const int* indices, const float* logits,
                                     int rowStart, int rowEnd, int* outIndices,
                                     float* outLogits, int stride1, int topK) {
  static_assert(!deviceLengthAware || multipleBlocksPerRow != mergeBlocks);
  static_assert(!useTopK512Optimization || kNumThreadsPerBlock == 1024);
  // The number of slots for the final pass.
  static constexpr int kNumFinalItems = useTopK512Optimization ? 1024 : 2048;
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
        outLogits[rowIt] = useTopK512Optimization ? -INFINITY : -FLT_MAX;
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
                           multipleBlocksPerRow, mergeBlocks,
                           useTopK512Optimization>(
          indices, logits, rowEnd, logitPattern, thresholdBinIdx, smemOutput,
          smemThresholdBinIdx, smemFinalDstIdx, smemFinalBinSize,
          smemFoundTopKValues, smemFinal, stride1, rowStart, topK);

  if (continueToNextStep) {
    // Step 1: Process next 11 bits
    continueToNextStep =
        processHistogramStep<1, kNumThreadsPerBlock, kNumBins, kNumFinalItems,
                             multipleBlocksPerRow, mergeBlocks,
                             useTopK512Optimization>(
            indices, logits, rowEnd, logitPattern, thresholdBinIdx, smemOutput,
            smemThresholdBinIdx, smemFinalDstIdx, smemFinalBinSize,
            smemFoundTopKValues, smemFinal, stride1, rowStart, topK);
  }

  if (continueToNextStep) {
    // Step 2: Process next 11 bits
    continueToNextStep =
        processHistogramStep<2, kNumThreadsPerBlock, kNumBins, kNumFinalItems,
                             multipleBlocksPerRow, mergeBlocks,
                             useTopK512Optimization>(
            indices, logits, rowEnd, logitPattern, thresholdBinIdx, smemOutput,
            smemThresholdBinIdx, smemFinalDstIdx, smemFinalBinSize,
            smemFoundTopKValues, smemFinal, stride1, rowStart, topK);
  }

  if (continueToNextStep) {
    // Step 3: Process last 10 bits
    processHistogramStep<3, kNumThreadsPerBlock, kNumBins, kNumFinalItems,
                         multipleBlocksPerRow, mergeBlocks,
                         useTopK512Optimization>(
        indices, logits, rowEnd, logitPattern, thresholdBinIdx, smemOutput,
        smemThresholdBinIdx, smemFinalDstIdx, smemFinalBinSize,
        smemFoundTopKValues, smemFinal, stride1, rowStart, topK);
  }

  if (!continueToNextStep) {
    // The histogram did not proceed to the final 10 bits, therefore we need to
    // sort the final items The logits of the elements to be sorted in the final
    // pass.
    auto insertionSort = [&]() {
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
    };
    if constexpr (useRadixSort) {
      if (useTopK512Optimization && smemFinalDstIdx[0] <= 128) {
        insertionSort();
      } else {
        // Sorting with radix sort
        float finalLogits[kNumFinalItemsPerThread];
        // The indices of the elements to be sorted in the final pass.
        int finalIndices[kNumFinalItemsPerThread];

#pragma unroll
        for (int ii = 0; ii < kNumFinalItemsPerThread; ++ii) {
          finalLogits[ii] = useTopK512Optimization ? -INFINITY : -FLT_MAX;
          if constexpr (useTopK512Optimization) finalIndices[ii] = -1;
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
      }
    } else {
      insertionSort();
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

}  // namespace vllm

namespace topk942 {

constexpr int kTopK = 512;
constexpr int kNumBins = 2048;
constexpr int kNumThreads = 1024;

// The number of blocks that one row is split into: one block for every
// minChunk logits, at most maxBlocks. The merge kernel reads 512 entries from
// each block, so every block must hold at least 512 logits of the row.
__device__ int activeBlocks(int rowLength, int maxBlocks, int minChunk) {
  const int blocks = (rowLength + minChunk - 1) / minChunk;
  return min(min(blocks, maxBlocks), max(1, rowLength / kTopK));
}

// Row r ends where vLLM's top_k_per_row_decode ends it. A 2D seqLens holds
// each row's own end. A 1D seqLens holds the request's length, and row r is
// draft position r % next_n of request r / next_n.
__device__ int rowEndOf(const int* seqLens, int rowIdx, int next_n,
                        int seqLensIs2D) {
  return seqLensIs2D
             ? max(0, seqLens[rowIdx])
             : max(0, seqLens[rowIdx / next_n] - next_n + rowIdx % next_n + 1);
}

// Without mergeBlocks, block (r, b) writes the top 512 of its part of row r
// to outIndicesAux and outLogitsAux. A row that fits one block writes its
// result to outIndices directly. With mergeBlocks, block r takes the top 512
// of row r's parts.
template <bool mergeBlocks>
__global__ __launch_bounds__(kNumThreads) void topK512Decode(
    const float* logits, const int* seqLens, int* outIndices,
    int* outIndicesAux, float* outLogitsAux, int stride0, int outStride,
    int next_n, int seqLensIs2D, int maxBlocks, int minChunk) {
  const int rowIdx = blockIdx.x;
  const int rowEnd = rowEndOf(seqLens, rowIdx, next_n, seqLensIs2D);
  const int blocks = activeBlocks(rowEnd, maxBlocks, minChunk);
  outIndices += static_cast<int64_t>(rowIdx) * outStride;
  if constexpr (mergeBlocks) {
    if (blocks == 1) return;
    const int64_t offset = static_cast<int64_t>(rowIdx) * maxBlocks * kTopK;
    vllm::topKPerRowJob<kNumThreads, kNumBins, true, false, true, false, true>(
        outIndicesAux + offset, outLogitsAux + offset, 0, blocks * kTopK,
        outIndices, nullptr, 1, kTopK);
  } else {
    if (blockIdx.y >= blocks) return;
    logits += static_cast<int64_t>(rowIdx) * stride0;
    if (blocks == 1) {
      vllm::topKPerRowJob<kNumThreads, kNumBins, true, false, false, false,
                          true>(nullptr, logits, 0, rowEnd, outIndices, nullptr,
                                1, kTopK);
      return;
    }
    const int blockSize = rowEnd / blocks;
    const int rowStart = blockSize * blockIdx.y;
    const int blockRowEnd =
        blockIdx.y + 1 == blocks ? rowEnd : rowStart + blockSize;
    const int64_t offset =
        (static_cast<int64_t>(rowIdx) * maxBlocks + blockIdx.y) * kTopK;
    vllm::topKPerRowJob<kNumThreads, kNumBins, true, true, false, false, true>(
        nullptr, logits, rowStart, blockRowEnd, outIndicesAux + offset,
        outLogitsAux + offset, 1, kTopK);
  }
}

// Layers 24 to 36 take their top 512 only from the candidate blocks that
// layer 20 chose for the row. vLLM sets every other logit of the row to -inf
// and then runs its top-k over the whole row. This kernel instead copies
// the candidate logits of row r into a compact row: position i holds column
// blockSize * cand[r][i / blockSize] + i % blockSize with that column as its
// id, or -inf with id -1 when the block entry is -1 or the column is at or
// past the row's end. The compact row has the same finite logits as vLLM's
// masked row, so its top 512 are the same columns.
__global__ void gatherCandidates(const float* logits, const int* seqLens,
                                 const int* cand, float* compactLogits,
                                 int* compactIds, int stride0, int candStride,
                                 int len, int blockSize, int next_n,
                                 int seqLensIs2D) {
  const int rowIdx = blockIdx.y;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= len) return;
  const int rowEnd = rowEndOf(seqLens, rowIdx, next_n, seqLensIs2D);
  const int block =
      cand[static_cast<int64_t>(rowIdx) * candStride + i / blockSize];
  const int64_t col = static_cast<int64_t>(block) * blockSize + i % blockSize;
  const bool live = block >= 0 && col < rowEnd;
  const int64_t dst = static_cast<int64_t>(rowIdx) * len + i;
  compactLogits[dst] =
      live ? logits[static_cast<int64_t>(rowIdx) * stride0 + col] : -INFINITY;
  compactIds[dst] = live ? static_cast<int>(col) : -1;
}

// The top 512 of a compact row, written as column ids by topKPerRowJob with
// mergeBlocks, which reports the id stored next to each logit. A row that ends
// at or before 512 gets columns 0 to its end - 1 and then -1. topK512Decode
// writes such a row the same way, whatever its logits are, and the readers
// of the indices take only the first min(end, 512) entries. A longer row
// always has at least 512 live candidates (layer 20 lists 2048 blocks of 8,
// or every block of a row shorter than that), so no -1 id is chosen there.
__global__ __launch_bounds__(kNumThreads) void topK512Candidates(
    const float* compactLogits, const int* compactIds, const int* seqLens,
    int* outIndices, int len, int outStride, int next_n, int seqLensIs2D) {
  const int rowIdx = blockIdx.x;
  const int rowEnd = rowEndOf(seqLens, rowIdx, next_n, seqLensIs2D);
  outIndices += static_cast<int64_t>(rowIdx) * outStride;
  if (rowEnd <= kTopK) {
    for (int k = threadIdx.x; k < kTopK; k += kNumThreads) {
      outIndices[k] = k < rowEnd ? k : -1;
    }
    return;
  }
  const int64_t offset = static_cast<int64_t>(rowIdx) * len;
  vllm::topKPerRowJob<kNumThreads, kNumBins, true, false, true, false, true>(
      compactIds + offset, compactLogits + offset, 0, len, outIndices, nullptr,
      1, kTopK);
}

// The top 512 of rows that fit in the registers of one block
// (selectFromRegisters). Each thread holds its items' order-preserving keys
// and ids in registers, so the row is read from memory once. vLLM's radix
// job reads the row again in each of its up to 4 histogram passes.
//
// One CU does all the work of a row, and a CU retires 64 lanes of an
// instruction a cycle, so each instruction per item costs about 256 cycles
// for a row of 16384. So the code keeps the instructions per item few.
// Spreading a row over many blocks is slower: the device-scope fence that
// hands the counts to the row's last block costs about 16 us on gfx942, more
// than the whole select.

// A larger float has a larger key. +NaN is above +inf, as torch.topk ranks
// NaN. A key of 0 is below every float's key, so it can pad a row.
__device__ inline uint32_t floatKey(float x) {
  const uint32_t u = __float_as_uint(x);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

struct SelectSmem {
  int hist[kNumBins];
  int waveSum[kNumThreads / 64];
  int bin;
  int above;
  int inBin;
};

// The first pass of selectFromRegisters bins a key by the sign, the exponent
// and the top 5 mantissa bits of its value, as FP16 does: 32 bins an octave
// for magnitudes from 2^-15 to 2^17 on each side of zero. Bin 0 holds the
// largest values. The indexer's logits are almost all negative (a row of
// 65539 had 2 positive values, with its 512th largest value at -0.87), so
// the bins must not depend on the row's largest value. A bin's key range
// follows from its number, so the later passes need no block-wide minimum
// and maximum, and "taken" is one comparison with the cut's upper key.
//
// The magnitude bits of a key: key - 2^31 for a value >= +0, and
// 2^31 - 1 - key for a value <= -0. Both are key XOR (2^31 - 1 + sign bit).
constexpr int kBinShift = 18;
constexpr int kBinBias = (127 - 15) << (23 - kBinShift);

__device__ inline uint32_t magnitudeOf(uint32_t key) {
  return key ^ (0x7FFFFFFFu + (key >> 31));
}

__device__ inline int signedBin(uint32_t key) {
  const int e =
      min(max(static_cast<int>(magnitudeOf(key) >> kBinShift) - kBinBias, 0),
          kNumBins / 2 - 1);
  return key >> 31 ? kNumBins / 2 - 1 - e : kNumBins / 2 + e;
}

// The keys of signedBin's bin b: [*lo, *hi]. The bins of the smallest and
// largest magnitudes also hold everything closer to zero and further away.
__device__ inline void signedBinKeys(int bin, uint32_t* lo, uint32_t* hi) {
  const bool positive = bin < kNumBins / 2;
  const int e = positive ? kNumBins / 2 - 1 - bin : bin - kNumBins / 2;
  const uint32_t magLo =
      e == 0 ? 0u : static_cast<uint32_t>(e + kBinBias) << kBinShift;
  const uint32_t magHi =
      e == kNumBins / 2 - 1
          ? 0x7FFFFFFFu
          : (static_cast<uint32_t>(e + kBinBias + 1) << kBinShift) - 1u;
  if (positive) {
    *lo = 0x80000000u | magLo;
    *hi = 0x80000000u | magHi;
  } else {
    *lo = 0x7FFFFFFFu - magHi;
    *hi = 0x7FFFFFFFu - magLo;
  }
}

// Adds 1 to hist[bin] for each active lane. LDS does the atomics of lanes
// with the same address one at a time, so when all active lanes of the wave
// have the same bin (equal keys, or padding), one lane adds their count.
// The first pass of selectFromRegisters uses this. Its far keys, -inf included,
// are not counted, so its waves rarely share a bin only in part.
__device__ inline void addToBin(int* hist, int bin, bool active) {
  const uint64_t lanes = __ballot(active);
  if (lanes == 0) return;
  const int firstLane = __ffsll(static_cast<unsigned long long>(lanes)) - 1;
  const int firstBin = __builtin_amdgcn_readlane(bin, firstLane);
  if (__ballot(active && bin == firstBin) == lanes) {
    if (static_cast<int>(threadIdx.x % 64) == firstLane) {
      atomicAdd(&hist[firstBin], static_cast<int>(__popcll(lanes)));
    }
  } else if (active) {
    atomicAdd(&hist[bin], 1);
  }
}

// addToBin for the range passes, where a wave can hold many equal keys among
// a few others, as -inf among a few finite logits when the cut is in the
// last bin. The first active lane adds the count of all active lanes in its
// bin, and only the lanes in other bins add 1 each. That costs one more
// atomic instruction, which the first pass would pay on every item.
__device__ inline void addToBinMostlyEqual(int* hist, int bin, bool active) {
  const uint64_t lanes = __ballot(active);
  if (lanes == 0) return;
  const int firstLane = __ffsll(static_cast<unsigned long long>(lanes)) - 1;
  const int firstBin = __builtin_amdgcn_readlane(bin, firstLane);
  const bool inFirstBin = active && bin == firstBin;
  const uint64_t sameLanes = __ballot(inFirstBin);
  if (static_cast<int>(threadIdx.x % 64) == firstLane) {
    atomicAdd(&hist[firstBin], static_cast<int>(__popcll(sameLanes)));
  }
  if (active && !inFirstBin) atomicAdd(&hist[bin], 1);
}

// The exclusive prefix sum of v over the block's threads in thread order.
// *total gets the sum over the block.
__device__ inline uint32_t blockExclusiveSum(uint32_t v, SelectSmem& s,
                                             uint32_t* total) {
  const int lane = threadIdx.x % 64;
  const int wave = threadIdx.x / 64;
  uint32_t inclusive = v;
#pragma unroll
  for (int d = 1; d < 64; d <<= 1) {
    const uint32_t x = __shfl_up(inclusive, d, 64);
    if (lane >= d) inclusive += x;
  }
  if (lane == 63) s.waveSum[wave] = static_cast<int>(inclusive);
  __syncthreads();
  uint32_t before = 0;
  uint32_t all = 0;
#pragma unroll
  for (int w = 0; w < kNumThreads / 64; ++w) {
    const uint32_t x = static_cast<uint32_t>(s.waveSum[w]);
    before += w < wave ? x : 0u;
    all += x;
  }
  *total = all;
  return before + inclusive - v;
}

// After a pass's histogram, whose bin 0 holds the largest keys: thread t
// holds bins 2t and 2t + 1. Writes the bin that holds the
// needed-th largest key, the count of the bins before it and its own count.
// With lastBinOf > 0 the last bin was not counted, and its count is
// lastBinOf minus the counts of all other bins.
__device__ __noinline__ void findCutBin(SelectSmem& s, int needed,
                                        int lastBinOf) {
  const int t = threadIdx.x;
  const int lane = t % 64;
  const int wave = t / 64;
  const int first = s.hist[2 * t];
  int second = s.hist[2 * t + 1];
  int sum = first + second;
  int inclusive = sum;
#pragma unroll
  for (int d = 1; d < 64; d <<= 1) {
    const int v = __shfl_up(inclusive, d, 64);
    if (lane >= d) inclusive += v;
  }
  if (lane == 63) s.waveSum[wave] = inclusive;
  __syncthreads();
#pragma unroll
  for (int w = 0; w < kNumThreads / 64; ++w) {
    inclusive += w < wave ? s.waveSum[w] : 0;
  }
  const int exclusive = inclusive - sum;
  if (lastBinOf > 0 && t == kNumThreads - 1) {
    second = lastBinOf - exclusive - first;
    inclusive = lastBinOf;
  }
  if (exclusive < needed && inclusive >= needed) {
    if (exclusive + first >= needed) {
      s.bin = 2 * t;
      s.above = exclusive;
      s.inBin = first;
    } else {
      s.bin = 2 * t + 1;
      s.above = exclusive + first;
      s.inBin = second;
    }
  }
  __syncthreads();
}

// Writes the ids of the k largest of the block's keys to out[0, k), in no
// particular order, and their keys to outKeys[0, k) when outKeys is not
// null. There must be at least k keys, padding included. Of the keys equal
// to the cut, the ones at the lowest (thread, item) places are taken.
// kStop < 3 only exists to time the phases: 0 stops after the load, 1 after
// the first histogram and 2 after the cut.
template <int kItems, int kStop = 3>
__device__ void selectFromRegisters(const uint32_t (&keys)[kItems],
                                    const int (&ids)[kItems], int k,
                                    SelectSmem& s, int* out,
                                    uint32_t* outKeys = nullptr) {
  for (int b = threadIdx.x; b < kNumBins; b += kNumThreads) s.hist[b] = 0;
  if constexpr (kStop == 0) {
    uint32_t m = 0;
#pragma unroll
    for (int j = 0; j < kItems; ++j) {
      m ^= keys[j] ^ static_cast<uint32_t>(ids[j]);
    }
    if (m == 0x12345678u) out[threadIdx.x % k] = 0;
    return;
  }
  __syncthreads();
  // The last bin holds -inf, the padding keys and negative values below
  // -2^17, all far below the cut in practice. They are not counted, so that
  // a wave of a few finite logits among -inf does not add to one bin lane by
  // lane, and findCutBin takes the last bin's count as all keys
  // minus the other bins.
#pragma unroll
  for (int j = 0; j < kItems; ++j) {
    const int bin = signedBin(keys[j]);
    addToBin(s.hist, bin, bin != kNumBins - 1);
  }
  __syncthreads();
  if constexpr (kStop == 1) {
    uint32_t m = s.hist[threadIdx.x];
#pragma unroll
    for (int j = 0; j < kItems; ++j) m ^= static_cast<uint32_t>(ids[j]);
    if (m == 0x12345678u) out[threadIdx.x % k] = 0;
    return;
  }
  findCutBin(s, k, kItems * kNumThreads);
  const int bin = s.bin;
  int needed = k - s.above;
  bool takeAll = s.inBin == needed;
  uint32_t lo;
  uint32_t hi;
  signedBinKeys(bin, &lo, &hi);
  while (!takeAll && hi != lo) {
    const int shift = max(0, 32 - __clz(static_cast<int>(hi - lo)) - 11);
    for (int b = threadIdx.x; b < kNumBins; b += kNumThreads) s.hist[b] = 0;
    __syncthreads();
#pragma unroll
    for (int j = 0; j < kItems; ++j) {
      const bool inRange = keys[j] >= lo && keys[j] <= hi;
      addToBinMostlyEqual(s.hist, static_cast<int>((hi - keys[j]) >> shift),
                          inRange);
    }
    __syncthreads();
    findCutBin(s, needed, 0);
    needed -= s.above;
    takeAll = s.inBin == needed;
    // Bin c of this pass holds hi - key in
    // [c * 2^shift, (c + 1) * 2^shift - 1].
    const uint32_t newHi = hi - (static_cast<uint32_t>(s.bin) << shift);
    lo = static_cast<uint32_t>(max<int64_t>(
        static_cast<int64_t>(lo),
        static_cast<int64_t>(newHi) - ((int64_t{1} << shift) - 1)));
    hi = newHi;
  }
  if constexpr (kStop == 2) {
    uint32_t m = hi ^ lo;
#pragma unroll
    for (int j = 0; j < kItems; ++j) m ^= static_cast<uint32_t>(ids[j]);
    if (m == 0x12345678u) out[threadIdx.x % k] = needed;
    return;
  }
  // The k - needed keys above hi go to out[0, k - needed) and the first
  // needed keys in [lo, hi] to out[k - needed, k). One prefix sum over the
  // block gives every thread its places. The low 16 bits count keys above
  // hi (at most 511 in all) and the high 16 bits keys in [lo, hi] (at most
  // 32768 in all), so the two sums do not mix.
  uint32_t counts = 0;
#pragma unroll
  for (int j = 0; j < kItems; ++j) {
    counts += keys[j] > hi ? 1u : (keys[j] >= lo ? 0x10000u : 0u);
  }
  uint32_t total;
  const uint32_t before = blockExclusiveSum(counts, s, &total);
  int aboveAt = static_cast<int>(before & 0xFFFFu);
  int tieAt = static_cast<int>(before >> 16);
  const int tieBase = static_cast<int>(total & 0xFFFFu);
#pragma unroll
  for (int j = 0; j < kItems; ++j) {
    const uint32_t key = keys[j];
    int place = -1;
    if (key > hi) {
      place = aboveAt++;
    } else if (key >= lo) {
      if (tieAt < needed) place = tieBase + tieAt;
      ++tieAt;
    }
    if (place >= 0) {
      out[place] = ids[j];
      if (outKeys != nullptr) outKeys[place] = key;
    }
  }
}

// The decode top 512 of dense logits rows with selectFromRegisters, in two
// kernels. A row of n columns is split into ceil(n / 16384) chunks of equal
// size, and block (r, c) of topK512DecodeChunks takes the top 512 of chunk c
// of row r. A row of one chunk writes its result to outIndices. A longer row
// writes the chunk's ids and keys to auxIds and auxKeys, and
// topK512MergeChunks takes the row's top 512 of them. The top 512 of all
// chunks' top 512 are the row's top 512, except for ties at the cut. Every
// chunk of a longer row holds more than 8192 columns, so no chunk list
// holds padding.
constexpr int kDecodeChunk = 16 * kNumThreads;

template <int kItems>
__device__ void loadKeys(const float* row, int start, int end,
                         uint32_t (&keys)[kItems], int (&ids)[kItems]) {
#pragma unroll
  for (int v = 0; v < kItems / 4; ++v) {
    const int pos =
        start + 4 * (v * kNumThreads + static_cast<int>(threadIdx.x));
    if (pos + 3 < end) {
      const float4 x = *reinterpret_cast<const float4*>(row + pos);
      keys[4 * v] = floatKey(x.x);
      keys[4 * v + 1] = floatKey(x.y);
      keys[4 * v + 2] = floatKey(x.z);
      keys[4 * v + 3] = floatKey(x.w);
#pragma unroll
      for (int c = 0; c < 4; ++c) ids[4 * v + c] = pos + c;
    } else {
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        keys[4 * v + c] = pos + c < end ? floatKey(row[pos + c]) : 0u;
        ids[4 * v + c] = pos + c < end ? pos + c : -1;
      }
    }
  }
}

__global__ __launch_bounds__(kNumThreads) void topK512DecodeChunks(
    const float* logits, const int* seqLens, int* outIndices, int* auxIds,
    uint32_t* auxKeys, int stride0, int outStride, int next_n, int seqLensIs2D,
    int maxChunks) {
  __shared__ SelectSmem s;
  const int rowIdx = blockIdx.x;
  const int rowEnd = rowEndOf(seqLens, rowIdx, next_n, seqLensIs2D);
  const int chunks = (rowEnd + kDecodeChunk - 1) / kDecodeChunk;
  if (static_cast<int>(blockIdx.y) >= max(1, chunks)) return;
  outIndices += static_cast<int64_t>(rowIdx) * outStride;
  if (rowEnd <= kTopK) {
    for (int k = threadIdx.x; k < kTopK; k += kNumThreads) {
      outIndices[k] = k < rowEnd ? k : -1;
    }
    return;
  }
  const float* row = logits + static_cast<int64_t>(rowIdx) * stride0;
  // The select's time follows the items per thread, padding included, so a
  // short row takes 4 or 8 items a thread instead of 16.
  if (rowEnd <= 4 * kNumThreads) {
    uint32_t keys[4];
    int ids[4];
    loadKeys(row, 0, rowEnd, keys, ids);
    selectFromRegisters(keys, ids, kTopK, s, outIndices);
    return;
  }
  if (rowEnd <= 8 * kNumThreads) {
    uint32_t keys[8];
    int ids[8];
    loadKeys(row, 0, rowEnd, keys, ids);
    selectFromRegisters(keys, ids, kTopK, s, outIndices);
    return;
  }
  // The chunks share the row equally, so that no chunk is mostly padding. A
  // chunk with fewer than 512 columns would have its cut among the padding
  // keys, which takes three more passes.
  const int chunkLen = ((rowEnd + chunks - 1) / chunks + 3) / 4 * 4;
  const int start = blockIdx.y * chunkLen;
  uint32_t keys[16];
  int ids[16];
  loadKeys(row, start, min(start + chunkLen, rowEnd), keys, ids);
  if (chunks == 1) {
    selectFromRegisters(keys, ids, kTopK, s, outIndices);
  } else {
    const int64_t offset =
        (static_cast<int64_t>(rowIdx) * maxChunks + blockIdx.y) * kTopK;
    selectFromRegisters(keys, ids, kTopK, s, auxIds + offset, auxKeys + offset);
  }
}

template <int kItems>
__device__ void mergeChunks(const int* auxIds, const uint32_t* auxKeys, int n,
                            SelectSmem& s, int* out) {
  uint32_t keys[kItems];
  int ids[kItems];
#pragma unroll
  for (int v = 0; v < kItems / 4; ++v) {
    const int pos = 4 * (v * kNumThreads + static_cast<int>(threadIdx.x));
    if (pos < n) {
      const uint4 x = *reinterpret_cast<const uint4*>(auxKeys + pos);
      const int4 i = *reinterpret_cast<const int4*>(auxIds + pos);
      keys[4 * v] = x.x;
      keys[4 * v + 1] = x.y;
      keys[4 * v + 2] = x.z;
      keys[4 * v + 3] = x.w;
      ids[4 * v] = i.x;
      ids[4 * v + 1] = i.y;
      ids[4 * v + 2] = i.z;
      ids[4 * v + 3] = i.w;
    } else {
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        keys[4 * v + c] = 0u;
        ids[4 * v + c] = -1;
      }
    }
  }
  selectFromRegisters(keys, ids, kTopK, s, out);
}

// The merge takes chunks * 512 entries. Its items per thread follow the
// row's chunk count, so a row of 16 chunks or fewer (256k columns) does not
// pay for the 32 items that a row of 64 chunks needs.
__global__ __launch_bounds__(kNumThreads) void topK512MergeChunks(
    const int* seqLens, int* outIndices, const int* auxIds,
    const uint32_t* auxKeys, int outStride, int next_n, int seqLensIs2D,
    int maxChunks) {
  __shared__ SelectSmem s;
  const int rowIdx = blockIdx.x;
  const int rowEnd = rowEndOf(seqLens, rowIdx, next_n, seqLensIs2D);
  const int chunks = (rowEnd + kDecodeChunk - 1) / kDecodeChunk;
  if (chunks <= 1) return;
  const int n = chunks * kTopK;
  const int64_t offset = static_cast<int64_t>(rowIdx) * maxChunks * kTopK;
  int* out = outIndices + static_cast<int64_t>(rowIdx) * outStride;
  if (n <= 8 * kNumThreads) {
    mergeChunks<8>(auxIds + offset, auxKeys + offset, n, s, out);
  } else if (n <= 16 * kNumThreads) {
    mergeChunks<16>(auxIds + offset, auxKeys + offset, n, s, out);
  } else {
    mergeChunks<32>(auxIds + offset, auxKeys + offset, n, s, out);
  }
}

// topK512Candidates with selectFromRegisters. Thread t holds the compact row's
// positions 4 * (v * kNumThreads + t) + c for v < kItems / 4 and c < 4, so
// each v is one coalesced float4 load of logits and one int4 load of ids.
// Positions past len get key 0, below every logit.
template <int kItems, int kStop = 3>
__global__ __launch_bounds__(kNumThreads) void topK512CompactRegs(
    const float* compactLogits, const int* compactIds, const int* seqLens,
    int* outIndices, int len, int outStride, int next_n, int seqLensIs2D) {
  __shared__ SelectSmem s;
  const int rowIdx = blockIdx.x;
  const int rowEnd = rowEndOf(seqLens, rowIdx, next_n, seqLensIs2D);
  outIndices += static_cast<int64_t>(rowIdx) * outStride;
  if (rowEnd <= kTopK) {
    for (int k = threadIdx.x; k < kTopK; k += kNumThreads) {
      outIndices[k] = k < rowEnd ? k : -1;
    }
    return;
  }
  const int64_t offset = static_cast<int64_t>(rowIdx) * len;
  const float* row = compactLogits + offset;
  const int* rowIds = compactIds + offset;
  uint32_t keys[kItems];
  int ids[kItems];
#pragma unroll
  for (int v = 0; v < kItems / 4; ++v) {
    const int pos = 4 * (v * kNumThreads + threadIdx.x);
    if (pos + 3 < len) {
      const float4 x = *reinterpret_cast<const float4*>(row + pos);
      const int4 i = *reinterpret_cast<const int4*>(rowIds + pos);
      keys[4 * v] = floatKey(x.x);
      keys[4 * v + 1] = floatKey(x.y);
      keys[4 * v + 2] = floatKey(x.z);
      keys[4 * v + 3] = floatKey(x.w);
      ids[4 * v] = i.x;
      ids[4 * v + 1] = i.y;
      ids[4 * v + 2] = i.z;
      ids[4 * v + 3] = i.w;
    } else {
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        keys[4 * v + c] = pos + c < len ? floatKey(row[pos + c]) : 0u;
        ids[4 * v + c] = pos + c < len ? rowIds[pos + c] : -1;
      }
    }
  }
  selectFromRegisters<kItems, kStop>(keys, ids, kTopK, s, outIndices);
}

// The larger of two values, or NaN when either one is NaN, as Triton's
// tl.maximum with propagate_nan=ALL in vLLM's block score kernel.
__device__ inline float maxPropagatingNan(float a, float b) {
  return (a != a || b != b) ? NAN : fmaxf(a, b);
}

// Layer 20 lists, for each decode row, the blocks of blockSize compressed
// positions that layers 24 to 36 take their top 512 from. A block's score is
// the largest of its logits before the row's end, the row's newest block
// always scores +inf, and the list holds the topkBlocks blocks with the
// largest scores, -1 for a pick whose score is -inf. vLLM does this with
// three launches and torch.topk, which also sorts its picks
// (select_candidate_blocks in vllm/model_executor/kernels/attention/dsa/
// candidate_blocks.py). This kernel does it in one launch with one block per
// row. Each thread scores whole blocks. With blocks of 8 and a 16-byte
// aligned row, a block is two float4 loads, and the loop is unrolled so that
// a thread has several blocks' loads in flight. The scores of the blocks
// before the row's end go to the scores scratch, and vLLM's topKPerRowJob
// picks the top topkBlocks of them. vLLM also scores the blocks past the
// row's end, as -inf, so it only picks them when fewer than topkBlocks blocks
// are left, and then writes -1 for them. Here those places get -1 directly.
// So the list holds the same block ids as vLLM's, in no particular order,
// unless two scores are equal at the cut. blockSize must be at least 1.
__global__ __launch_bounds__(kNumThreads) void candidateBlocks(
    const float* logits, const int* visible, float* scores, int* out,
    int stride0, int outStride, int width, int scoresStride, int topkBlocks,
    int blockSize, int rowRepeat) {
  const int rowIdx = blockIdx.x;
  const int end = max(0, visible[rowIdx / rowRepeat]);
  const int liveEnd = min(end, width);
  const int liveBlocks = (liveEnd + blockSize - 1) / blockSize;
  const int newest = end > 0 ? (end - 1) / blockSize : -1;
  const float* row = logits + static_cast<int64_t>(rowIdx) * stride0;
  float* rowScores = scores + static_cast<int64_t>(rowIdx) * scoresStride;
  int* outRow = out + static_cast<int64_t>(rowIdx) * outStride;
  const bool vec8 =
      blockSize == 8 && reinterpret_cast<uintptr_t>(row) % sizeof(float4) == 0;
  // The blocks that lie wholly before liveEnd. The last block can be cut by
  // the row's end and is scored element by element below.
  const int wholeBlocks = vec8 ? liveEnd / 8 : 0;
  const float4* row4 = reinterpret_cast<const float4*>(row);
#pragma unroll 4
  for (int b = threadIdx.x; b < wholeBlocks; b += kNumThreads) {
    const float4 lo = row4[2 * b];
    const float4 hi = row4[2 * b + 1];
    float v = maxPropagatingNan(maxPropagatingNan(lo.x, lo.y),
                                maxPropagatingNan(lo.z, lo.w));
    v = maxPropagatingNan(v, maxPropagatingNan(maxPropagatingNan(hi.x, hi.y),
                                               maxPropagatingNan(hi.z, hi.w)));
    rowScores[b] = b == newest ? INFINITY : v;
  }
  for (int b = wholeBlocks + threadIdx.x; b < liveBlocks; b += kNumThreads) {
    float v = -INFINITY;
    const int c1 = min((b + 1) * blockSize, liveEnd);
    for (int c = b * blockSize; c < c1; ++c) {
      v = maxPropagatingNan(v, row[c]);
    }
    rowScores[b] = b == newest ? INFINITY : v;
  }
  __syncthreads();
  vllm::topKPerRowJob<kNumThreads, kNumBins, true, false, false, false, true>(
      nullptr, rowScores, 0, liveBlocks, outRow, nullptr, 1, topkBlocks);
  __syncthreads();
  for (int i = threadIdx.x; i < topkBlocks; i += kNumThreads) {
    const int b = outRow[i];
    if (b >= 0 && rowScores[b] == -INFINITY) {
      outRow[i] = -1;
    }
  }
}

}  // namespace topk942

// The same contract as vLLM's ops.top_k_per_row_decode with topK = 512 and
// stride1 = 1: indices[r, :512] gets row r's top 512 column indices, in no
// particular order, and -1 past the row's length when the row is shorter.
void top_k_per_row_decode_512(const torch::Tensor& logits, int64_t next_n,
                              const torch::Tensor& seq_lens,
                              torch::Tensor& indices, int64_t max_blocks,
                              int64_t min_chunk) {
  TORCH_CHECK(logits.scalar_type() == at::kFloat && logits.stride(1) == 1,
              "logits must be float32 with unit column stride");
  TORCH_CHECK(seq_lens.scalar_type() == at::kInt && seq_lens.is_contiguous(),
              "seq_lens must be contiguous int32");
  TORCH_CHECK(indices.scalar_type() == at::kInt && indices.stride(1) == 1 &&
                  indices.size(1) >= topk942::kTopK,
              "indices must be int32 with unit column stride and 512 columns");
  TORCH_CHECK(max_blocks >= 1 && min_chunk >= topk942::kTopK,
              "max_blocks must be >= 1 and min_chunk >= 512");
  const int numRows = static_cast<int>(logits.size(0));
  if (numRows == 0) return;
  const at::cuda::OptionalCUDAGuard guard(logits.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto aux_indices = torch::empty({numRows, max_blocks, topk942::kTopK},
                                  logits.options().dtype(at::kInt));
  auto aux_logits =
      torch::empty({numRows, max_blocks, topk942::kTopK}, logits.options());
  const int seqLensIs2D = seq_lens.dim() == 2 ? 1 : 0;
  topk942::topK512Decode<false>
      <<<dim3(numRows, max_blocks), topk942::kNumThreads,
         2 * topk942::kTopK * sizeof(int32_t), stream>>>(
          logits.data_ptr<float>(), seq_lens.data_ptr<int>(),
          indices.data_ptr<int>(), aux_indices.data_ptr<int>(),
          aux_logits.data_ptr<float>(), static_cast<int>(logits.stride(0)),
          static_cast<int>(indices.stride(0)), static_cast<int>(next_n),
          seqLensIs2D, static_cast<int>(max_blocks),
          static_cast<int>(min_chunk));
  topk942::topK512Decode<true><<<numRows, topk942::kNumThreads,
                                 topk942::kTopK * sizeof(int32_t), stream>>>(
      nullptr, seq_lens.data_ptr<int>(), indices.data_ptr<int>(),
      aux_indices.data_ptr<int>(), aux_logits.data_ptr<float>(), 0,
      static_cast<int>(indices.stride(0)), static_cast<int>(next_n),
      seqLensIs2D, static_cast<int>(max_blocks), static_cast<int>(min_chunk));
}

// top_k_per_row_decode_512 with selectFromRegisters (topK512DecodeChunks and
// topK512MergeChunks). The same contract and the same index sets, except for
// ties at the cut, for rows of at most 64 chunks of 16384 columns.
void top_k_per_row_decode_512_regs(const torch::Tensor& logits, int64_t next_n,
                                   const torch::Tensor& seq_lens,
                                   torch::Tensor& indices) {
  TORCH_CHECK(logits.scalar_type() == at::kFloat && logits.stride(1) == 1,
              "logits must be float32 with unit column stride");
  TORCH_CHECK(logits.stride(0) % 4 == 0 &&
                  reinterpret_cast<uintptr_t>(logits.data_ptr()) % 16 == 0,
              "the logits rows must start at 16-byte boundaries");
  TORCH_CHECK(logits.size(1) <= 64 * topk942::kDecodeChunk,
              "a row may hold at most 64 chunks of 16384 columns");
  TORCH_CHECK(seq_lens.scalar_type() == at::kInt && seq_lens.is_contiguous(),
              "seq_lens must be contiguous int32");
  TORCH_CHECK(indices.scalar_type() == at::kInt && indices.stride(1) == 1 &&
                  indices.size(1) >= topk942::kTopK,
              "indices must be int32 with unit column stride and 512 columns");
  const int numRows = static_cast<int>(logits.size(0));
  if (numRows == 0) return;
  const at::cuda::OptionalCUDAGuard guard(logits.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int maxChunks = static_cast<int>(std::max<int64_t>(
      1, (logits.size(1) + topk942::kDecodeChunk - 1) / topk942::kDecodeChunk));
  auto aux_ids = torch::empty({numRows, maxChunks, topk942::kTopK},
                              logits.options().dtype(at::kInt));
  auto aux_keys = torch::empty({numRows, maxChunks, topk942::kTopK},
                               logits.options().dtype(at::kInt));
  const int seqLensIs2D = seq_lens.dim() == 2 ? 1 : 0;
  topk942::topK512DecodeChunks<<<dim3(numRows, maxChunks), topk942::kNumThreads,
                                 0, stream>>>(
      logits.data_ptr<float>(), seq_lens.data_ptr<int>(),
      indices.data_ptr<int>(), aux_ids.data_ptr<int>(),
      reinterpret_cast<uint32_t*>(aux_keys.data_ptr<int>()),
      static_cast<int>(logits.stride(0)), static_cast<int>(indices.stride(0)),
      static_cast<int>(next_n), seqLensIs2D, maxChunks);
  topk942::topK512MergeChunks<<<numRows, topk942::kNumThreads, 0, stream>>>(
      seq_lens.data_ptr<int>(), indices.data_ptr<int>(),
      aux_ids.data_ptr<int>(),
      reinterpret_cast<const uint32_t*>(aux_keys.data_ptr<int>()),
      static_cast<int>(indices.stride(0)), static_cast<int>(next_n),
      seqLensIs2D, maxChunks);
}

// The top 512 of each row among its candidate blocks, without masking the
// logits: candidates[r] holds row r's block ids (int32, -1 padded), and block
// b keeps columns block_size * b to block_size * b + block_size - 1. The
// result is that of vLLM's candidate mask followed by top_k_per_row_decode,
// as sets, with the same row ends and the same -1 padding.
void candidate_top_k_512(const torch::Tensor& logits, int64_t next_n,
                         const torch::Tensor& seq_lens,
                         const torch::Tensor& candidates, int64_t block_size,
                         torch::Tensor& indices) {
  TORCH_CHECK(logits.scalar_type() == at::kFloat && logits.stride(1) == 1,
              "logits must be float32 with unit column stride");
  TORCH_CHECK(seq_lens.scalar_type() == at::kInt && seq_lens.is_contiguous(),
              "seq_lens must be contiguous int32");
  TORCH_CHECK(candidates.scalar_type() == at::kInt &&
                  candidates.stride(1) == 1 &&
                  candidates.size(0) >= logits.size(0),
              "candidates must be int32 with unit column stride, one row "
              "for each logits row");
  TORCH_CHECK(indices.scalar_type() == at::kInt && indices.stride(1) == 1 &&
                  indices.size(1) >= topk942::kTopK &&
                  indices.size(0) >= logits.size(0),
              "indices must be int32 with unit column stride and 512 columns");
  const int64_t len = candidates.size(1) * block_size;
  TORCH_CHECK(block_size >= 1 && len > topk942::kTopK && len < (1 << 30),
              "the candidate blocks must hold more than 512 columns");
  const int numRows = static_cast<int>(logits.size(0));
  if (numRows == 0) return;
  const at::cuda::OptionalCUDAGuard guard(logits.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto compact_logits = torch::empty({numRows, len}, logits.options());
  auto compact_ids =
      torch::empty({numRows, len}, logits.options().dtype(at::kInt));
  const int seqLensIs2D = seq_lens.dim() == 2 ? 1 : 0;
  constexpr int kGatherThreads = 256;
  topk942::gatherCandidates<<<
      dim3(static_cast<unsigned>((len + kGatherThreads - 1) / kGatherThreads),
           numRows),
      kGatherThreads, 0, stream>>>(
      logits.data_ptr<float>(), seq_lens.data_ptr<int>(),
      candidates.data_ptr<int>(), compact_logits.data_ptr<float>(),
      compact_ids.data_ptr<int>(), static_cast<int>(logits.stride(0)),
      static_cast<int>(candidates.stride(0)), static_cast<int>(len),
      static_cast<int>(block_size), static_cast<int>(next_n), seqLensIs2D);
  topk942::topK512Candidates<<<numRows, topk942::kNumThreads,
                               topk942::kTopK * sizeof(int32_t), stream>>>(
      compact_logits.data_ptr<float>(), compact_ids.data_ptr<int>(),
      seq_lens.data_ptr<int>(), indices.data_ptr<int>(), static_cast<int>(len),
      static_cast<int>(indices.stride(0)), static_cast<int>(next_n),
      seqLensIs2D);
}

// The top 512 of compact candidate rows that are already written: logits and
// column ids of each row's candidate positions, -inf and -1 where a candidate
// is not live, as gatherCandidates writes them. cand_logits.py computes the
// candidate logits of layers 24 to 36 straight into such rows, so this is
// candidate_top_k_512 without its gather.
void compact_top_k_512(const torch::Tensor& compact_logits,
                       const torch::Tensor& compact_ids,
                       const torch::Tensor& seq_lens, int64_t next_n,
                       torch::Tensor& indices) {
  TORCH_CHECK(compact_logits.scalar_type() == at::kFloat &&
                  compact_logits.is_contiguous(),
              "compact_logits must be contiguous float32");
  TORCH_CHECK(compact_ids.scalar_type() == at::kInt &&
                  compact_ids.is_contiguous() &&
                  compact_ids.sizes() == compact_logits.sizes(),
              "compact_ids must be contiguous int32 of the logits' shape");
  TORCH_CHECK(seq_lens.scalar_type() == at::kInt && seq_lens.is_contiguous(),
              "seq_lens must be contiguous int32");
  TORCH_CHECK(indices.scalar_type() == at::kInt && indices.stride(1) == 1 &&
                  indices.size(1) >= topk942::kTopK &&
                  indices.size(0) >= compact_logits.size(0),
              "indices must be int32 with unit column stride and 512 columns");
  const int64_t len = compact_logits.size(1);
  TORCH_CHECK(len > topk942::kTopK && len < (1 << 30),
              "a compact row must hold more than 512 columns");
  const int numRows = static_cast<int>(compact_logits.size(0));
  if (numRows == 0) return;
  const at::cuda::OptionalCUDAGuard guard(compact_logits.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int seqLensIs2D = seq_lens.dim() == 2 ? 1 : 0;
  topk942::topK512Candidates<<<numRows, topk942::kNumThreads,
                               topk942::kTopK * sizeof(int32_t), stream>>>(
      compact_logits.data_ptr<float>(), compact_ids.data_ptr<int>(),
      seq_lens.data_ptr<int>(), indices.data_ptr<int>(), static_cast<int>(len),
      static_cast<int>(indices.stride(0)), static_cast<int>(next_n),
      seqLensIs2D);
}

// compact_top_k_512 with selectFromRegisters instead of vLLM's topKPerRowJob.
// The same arguments and the same index sets, except for ties at the cut.
void compact_top_k_512_regs(const torch::Tensor& compact_logits,
                            const torch::Tensor& compact_ids,
                            const torch::Tensor& seq_lens, int64_t next_n,
                            torch::Tensor& indices) {
  TORCH_CHECK(compact_logits.scalar_type() == at::kFloat &&
                  compact_logits.is_contiguous(),
              "compact_logits must be contiguous float32");
  TORCH_CHECK(compact_ids.scalar_type() == at::kInt &&
                  compact_ids.is_contiguous() &&
                  compact_ids.sizes() == compact_logits.sizes(),
              "compact_ids must be contiguous int32 of the logits' shape");
  TORCH_CHECK(seq_lens.scalar_type() == at::kInt && seq_lens.is_contiguous(),
              "seq_lens must be contiguous int32");
  TORCH_CHECK(indices.scalar_type() == at::kInt && indices.stride(1) == 1 &&
                  indices.size(1) >= topk942::kTopK &&
                  indices.size(0) >= compact_logits.size(0),
              "indices must be int32 with unit column stride and 512 columns");
  const int64_t len = compact_logits.size(1);
  TORCH_CHECK(
      len > topk942::kTopK && len <= 32 * topk942::kNumThreads && len % 4 == 0,
      "a compact row must hold 516 to 32768 positions, a multiple of 4");
  const int numRows = static_cast<int>(compact_logits.size(0));
  if (numRows == 0) return;
  const at::cuda::OptionalCUDAGuard guard(compact_logits.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int seqLensIs2D = seq_lens.dim() == 2 ? 1 : 0;
  auto launch = [&](auto kernel) {
    kernel<<<numRows, topk942::kNumThreads, 0, stream>>>(
        compact_logits.data_ptr<float>(), compact_ids.data_ptr<int>(),
        seq_lens.data_ptr<int>(), indices.data_ptr<int>(),
        static_cast<int>(len), static_cast<int>(indices.stride(0)),
        static_cast<int>(next_n), seqLensIs2D);
  };
  if (len <= 16 * topk942::kNumThreads) {
    launch(topk942::topK512CompactRegs<16>);
  } else {
    launch(topk942::topK512CompactRegs<32>);
  }
}

// Runs topK512CompactRegs<16> only up to a phase (see selectFromRegisters's
// kStop), to time it.
void compact_top_k_512_regs_phase(const torch::Tensor& compact_logits,
                                  const torch::Tensor& compact_ids,
                                  const torch::Tensor& seq_lens,
                                  torch::Tensor& indices, int64_t stop) {
  TORCH_CHECK(compact_logits.size(1) == 16 * topk942::kNumThreads,
              "the phase timing takes rows of 16384");
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int numRows = static_cast<int>(compact_logits.size(0));
  auto launch = [&](auto kernel) {
    kernel<<<numRows, topk942::kNumThreads, 0, stream>>>(
        compact_logits.data_ptr<float>(), compact_ids.data_ptr<int>(),
        seq_lens.data_ptr<int>(), indices.data_ptr<int>(),
        static_cast<int>(compact_logits.size(1)),
        static_cast<int>(indices.stride(0)), 1, seq_lens.dim() == 2 ? 1 : 0);
  };
  if (stop == 0) launch(topk942::topK512CompactRegs<16, 0>);
  if (stop == 1) launch(topk942::topK512CompactRegs<16, 1>);
  if (stop == 2) launch(topk942::topK512CompactRegs<16, 2>);
  if (stop == 3) launch(topk942::topK512CompactRegs<16, 3>);
}

// Layer 20's candidate blocks for each decode row (topk942::candidateBlocks):
// out[r, :out.size(1)] gets row r's block ids, -1 padded, in no particular
// order. visible holds the row ends that vLLM's select_candidate_blocks
// takes: one for each row, or one for each request of row_repeat rows.
void candidate_blocks(const torch::Tensor& logits, const torch::Tensor& visible,
                      int64_t row_repeat, int64_t block_size,
                      torch::Tensor& out) {
  TORCH_CHECK(logits.scalar_type() == at::kFloat && logits.stride(1) == 1,
              "logits must be float32 with unit column stride");
  TORCH_CHECK(visible.scalar_type() == at::kInt && visible.is_contiguous(),
              "visible must be contiguous int32");
  TORCH_CHECK(out.scalar_type() == at::kInt && out.stride(1) == 1 &&
                  out.size(0) >= logits.size(0),
              "out must be int32 with unit column stride, one row for each "
              "logits row");
  TORCH_CHECK(block_size >= 1, "block_size must be at least 1");
  TORCH_CHECK(row_repeat >= 1 && visible.numel() * row_repeat >= logits.size(0),
              "visible must hold an end for every logits row");
  const int numRows = static_cast<int>(logits.size(0));
  const int topkBlocks = static_cast<int>(out.size(1));
  if (numRows == 0 || topkBlocks == 0) return;
  const at::cuda::OptionalCUDAGuard guard(logits.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int width = static_cast<int>(logits.size(1));
  const int nblocks =
      (width + static_cast<int>(block_size) - 1) / static_cast<int>(block_size);
  auto scores = torch::empty({numRows, max(nblocks, 1)}, logits.options());
  topk942::candidateBlocks<<<numRows, topk942::kNumThreads,
                             topkBlocks * sizeof(int32_t), stream>>>(
      logits.data_ptr<float>(), visible.data_ptr<int>(),
      scores.data_ptr<float>(), out.data_ptr<int>(),
      static_cast<int>(logits.stride(0)), static_cast<int>(out.stride(0)),
      width, static_cast<int>(scores.stride(0)), topkBlocks,
      static_cast<int>(block_size), static_cast<int>(row_repeat));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("top_k_per_row_decode_512", &top_k_per_row_decode_512,
        "The sparse indexer's decode top-512 for gfx942");
  m.def("top_k_per_row_decode_512_regs", &top_k_per_row_decode_512_regs,
        "top_k_per_row_decode_512 with the register select");
  m.def("candidate_top_k_512", &candidate_top_k_512,
        "The decode top-512 among the DSpark candidate blocks for gfx942");
  m.def("candidate_blocks", &candidate_blocks,
        "Layer 20's DSpark candidate blocks for gfx942");
  m.def("compact_top_k_512", &compact_top_k_512,
        "The decode top-512 of compact DSpark candidate rows for gfx942");
  m.def("compact_top_k_512_regs", &compact_top_k_512_regs,
        "compact_top_k_512 with the register select");
  m.def("compact_top_k_512_regs_phase", &compact_top_k_512_regs_phase,
        "The register select up to a phase, to time it");
}
