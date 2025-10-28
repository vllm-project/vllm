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

static inline __device__ uint16_t extractBinIdx(float x) {
  union {
    __half h;
    uint16_t u16;
  } tmp;
  tmp.h = __float2half_rn(x);
  tmp.u16 = (x < 0.f) ? (~tmp.u16 & 0xffff) : (tmp.u16 | 0x8000);
  return 511 - (tmp.u16 >> 7);
}

template <int kNumThreadsPerBlock = 512, int kNumBins = 512, int kTopK = 2048>
__device__ void topKPerRowJob(const float* logits, const int rowStart,
                              const int rowEnd, const int rowIdx,
                              int* outIndices, int stride0, int stride1) {
  // The number of elements per thread for the final top-k sort.
  static constexpr int kNumTopKItemsPerThread = kTopK / kNumThreadsPerBlock;
  // The class to sort the elements during the final top-k sort.
  using TopKSort = cub::BlockRadixSort<float, kNumThreadsPerBlock,
                                       kNumTopKItemsPerThread, int>;

  // The number of slots for the final pass.
  static constexpr int kNumFinalItems = 3072;
  // The number of elements per thread for the final sort.
  static constexpr int kNumFinalItemsPerThread =
      kNumFinalItems / kNumThreadsPerBlock;
  // The class to sort the elements during the final pass.
  using FinalSort = cub::BlockRadixSort<float, kNumThreadsPerBlock,
                                        kNumFinalItemsPerThread, int>;

  // The class to compute the inclusive prefix-sum over the histogram.
  using Scan = cub::BlockScan<int, kNumThreadsPerBlock>;

  // Shared memory to compute the block scan.
  __shared__ typename Scan::TempStorage smemScan;

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
    typename TopKSort::TempStorage topKSort;
  } smemFinal;

  // Shared memory to store the histogram.
  __shared__ int smemHistogram[kNumBins];
  // Shared memory to store the selected indices.
  __shared__ int smemIndices[kTopK];
  // Shared memory to store the threshold bin.
  __shared__ int smemThresholdBinIdx[1];
  // Shared memory counter to register the candidates for the final phase.
  __shared__ int smemFinalDstIdx[1];

  // The length of the row.
  int rowLen = rowEnd - rowStart;

  // Shortcut if the length of the row is smaller than Top-K. Indices are not
  // sorted by their corresponding logit.
  if (rowLen <= kTopK) {
    for (int rowIt = threadIdx.x; rowIt < rowLen;
         rowIt += kNumThreadsPerBlock) {
      int idx = rowStart + rowIt;
      outIndices[rowIdx * kTopK + rowIt] = idx - rowStart;
    }
    for (int rowIt = rowLen + threadIdx.x; rowIt < kTopK;
         rowIt += kNumThreadsPerBlock) {
      outIndices[rowIdx * kTopK + rowIt] = -1;
    }
    return;
  }

  // Clear the histogram.
  if (threadIdx.x < kNumBins) {
    smemHistogram[threadIdx.x] = 0;
  }

  // Make sure the histogram is ready.
  __syncthreads();

  // Fetch elements one-by-one.
  for (int rowIt = rowStart + threadIdx.x; rowIt < rowEnd;
       rowIt += kNumThreadsPerBlock) {
    uint16_t idx = extractBinIdx(logits[rowIdx * stride0 + rowIt * stride1]);
    atomicAdd(&smemHistogram[idx], 1);
  }

  // Make sure the histogram is ready.
  __syncthreads();

  // Read the values from SMEM.
  int binCount{0};
  if (threadIdx.x < kNumBins) {
    binCount = smemHistogram[threadIdx.x];
  }

  // Make sure each thread has read its value.
  __syncthreads();

  // Compute the prefix sum.
  int prefixSum{0}, totalSum{0};
  Scan(smemScan).ExclusiveSum(binCount, prefixSum, totalSum);

  // Update the histogram with the prefix sums.
  if (threadIdx.x < kNumBins) {
    smemHistogram[threadIdx.x] = prefixSum;
  }

  // Make sure the data is in shared memory.
  __syncthreads();

  // Find the last valid bin.
  if (threadIdx.x < kNumBins) {
    int nextPrefixSum =
        threadIdx.x == kNumBins - 1 ? totalSum : smemHistogram[threadIdx.x + 1];
    if (prefixSum < kTopK && nextPrefixSum >= kTopK) {
      smemThresholdBinIdx[0] = threadIdx.x;
    }
  }

  // Clear the counter to store the items for the final phase.
  if (threadIdx.x == 0) {
    smemFinalDstIdx[0] = 0;
  }

  // Make sure the data is in shared memory.
  __syncthreads();

  // The threshold bin.
  int thresholdBinIdx = smemThresholdBinIdx[0];

  // Fetch elements one-by-one and populate the shared memory buffers.
  for (int rowIt = rowStart + threadIdx.x; rowIt < rowEnd;
       rowIt += kNumThreadsPerBlock) {
    float logit = logits[rowIdx * stride0 + rowIt * stride1];
    uint16_t idx = extractBinIdx(logit);
    if (idx < thresholdBinIdx) {
      int dstIdx = atomicAdd(&smemHistogram[idx], 1);
      smemIndices[dstIdx] = rowIt;
    } else if (idx == thresholdBinIdx) {
      int dstIdx = atomicAdd(&smemFinalDstIdx[0], 1);
      if (dstIdx < kNumFinalItems) {
        smemFinal.items.logits[dstIdx] = logit;
        smemFinal.items.indices[dstIdx] = rowIt;
      }
    }
  }

  // Make sure the elements are in shared memory.
  __syncthreads();

  // The logits of the elements to be sorted in the final pass.
  float finalLogits[kNumFinalItemsPerThread];
  // The indices of the elements to be sorted in the final pass.
  int finalIndices[kNumFinalItemsPerThread];

// Init.
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
  int baseIdx = thresholdBinIdx > 0 ? smemHistogram[thresholdBinIdx - 1] : 0;
#pragma unroll
  for (int ii = 0; ii < kNumFinalItemsPerThread; ++ii) {
    int srcIdx = ii * kNumThreadsPerBlock + threadIdx.x;
    int dstIdx = baseIdx + srcIdx;
    if (dstIdx < kTopK) {
      smemIndices[dstIdx] = finalIndices[ii];
    }
  }

  // Make sure the data is in shared memory.
  __syncthreads();

// Store to global memory.
#pragma unroll
  for (int ii = 0; ii < kNumTopKItemsPerThread; ++ii) {
    int offset = rowIdx * kTopK + ii * kNumThreadsPerBlock + threadIdx.x;
    outIndices[offset] =
        smemIndices[ii * kNumThreadsPerBlock + threadIdx.x] - rowStart;
  }
}

template <int kNumThreadsPerBlock = 512>
static __global__ void topKPerRow(const float* logits, const int* rowStarts,
                                  const int* rowEnds, int* outIndices,
                                  int stride0, int stride1) {
  // The number of bins in the histogram.
  static constexpr int kNumBins = 512;

  // The top-k width.
  static constexpr int kTopK = 2048;

  // The row computed by this block.
  int rowIdx = blockIdx.x;

  // The range of logits within the row.
  int rowStart = rowStarts[rowIdx];
  int rowEnd = rowEnds[rowIdx];

  topKPerRowJob<kNumThreadsPerBlock, kNumBins, kTopK>(
      logits, rowStart, rowEnd, rowIdx, outIndices, stride0, stride1);
}

template <int kNumThreadsPerBlock = 512>
static __global__ void topKPerRowDecode(const float* logits, const int* seqLens,
                                        int* outIndices, int stride0,
                                        int stride1, int next_n) {
  // The number of bins in the histogram.
  static constexpr int kNumBins = 512;

  // The top-k width.
  static constexpr int kTopK = 2048;

  // The row computed by this block.
  int rowIdx = blockIdx.x;

  // The range of logits within the row.
  int rowStart = 0;
  int seq_len = seqLens[rowIdx / next_n];
  int rowEnd = seq_len - next_n + (rowIdx % next_n) + 1;

  topKPerRowJob<kNumThreadsPerBlock, kNumBins, kTopK>(
      logits, rowStart, rowEnd, rowIdx, outIndices, stride0, stride1);
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
  // Compute the results on the device.
  constexpr int kNumThreadsPerBlock = 512;
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  vllm::topKPerRowDecode<kNumThreadsPerBlock>
      <<<numRows, kNumThreadsPerBlock, 0, stream>>>(
          logits.data_ptr<float>(), seqLens.data_ptr<int>(),
          indices.data_ptr<int>(), static_cast<int>(stride0),
          static_cast<int>(stride1), static_cast<int>(next_n));
}

void top_k_per_row(const torch::Tensor& logits, const torch::Tensor& rowStarts,
                   const torch::Tensor& rowEnds, torch::Tensor& indices,
                   int64_t numRows, int64_t stride0, int64_t stride1) {
  // Compute the results on the device.
  constexpr int kNumThreadsPerBlock = 512;
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  vllm::topKPerRow<kNumThreadsPerBlock>
      <<<numRows, kNumThreadsPerBlock, 0, stream>>>(
          logits.data_ptr<float>(), rowStarts.data_ptr<int>(),
          rowEnds.data_ptr<int>(), indices.data_ptr<int>(),
          static_cast<int>(stride0), static_cast<int>(stride1));
}
