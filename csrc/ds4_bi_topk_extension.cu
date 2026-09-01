// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <algorithm>
#include <cstdint>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cub/block/block_radix_sort.cuh>
#include <torch/library.h>

namespace {

constexpr int kThreads = 512;
constexpr int kMaxTopK = 2048;
constexpr int kItemsPerThread = kMaxTopK / kThreads;
constexpr int kRadixBits = 8;
constexpr int kRadixBuckets = 1 << kRadixBits;

__device__ __forceinline__ uint32_t ordered_float_bits(float value) {
  const uint32_t bits = __float_as_uint(value);
  return (bits & 0x80000000U) ? ~bits : (bits ^ 0x80000000U);
}

__global__ __launch_bounds__(kThreads) void deterministic_top_k_per_row_prefill(
    const float* logits, const int* row_starts, const int* row_ends,
    int* output, int64_t stride0, int64_t stride1, int top_k) {
  using ScoreSort =
      cub::BlockRadixSort<uint64_t, kThreads, kItemsPerThread>;
  union SharedStorage {
    uint32_t histogram[kRadixBuckets];
    uint64_t selected[kMaxTopK];
    typename ScoreSort::TempStorage sort;
  };
  __shared__ SharedStorage shared;
  __shared__ uint64_t selected_prefix;
  __shared__ int rank;
  __shared__ int selected_count;

  const int row = blockIdx.x;
  const int row_start = row_starts[row];
  const int row_end = row_ends[row];
  const int row_length = max(0, row_end - row_start);
  const int effective_top_k = min(top_k, row_length);
  uint64_t score_keys[kItemsPerThread];

  if (effective_top_k == 0) {
    for (int output_column = threadIdx.x; output_column < top_k;
         output_column += kThreads) {
      output[static_cast<int64_t>(row) * top_k + output_column] = -1;
    }
    return;
  }

  if (threadIdx.x == 0) {
    selected_prefix = 0;
    // One-indexed rank of the key that bounds the final Top-K set.
    rank = effective_top_k;
  }
  __syncthreads();

  // Select the exact Kth-largest 64-bit (score, local-index) key eight radix
  // bits at a time.  Unlike the old 4096-element register sort, each pass can
  // scan a row of any length while retaining the same deterministic tie break.
#pragma unroll
  for (int shift = 64 - kRadixBits; shift >= 0; shift -= kRadixBits) {
    if (threadIdx.x < kRadixBuckets) {
      shared.histogram[threadIdx.x] = 0;
    }
    __syncthreads();

    for (int local_index = threadIdx.x; local_index < row_length;
         local_index += kThreads) {
      const int absolute_index = row_start + local_index;
      const float score =
          logits[static_cast<int64_t>(row) * stride0 +
                 static_cast<int64_t>(absolute_index) * stride1];
      // Descending score, then descending request-local source index,
      // matching vLLM's insertion-sort tie and output semantics. The key is
      // unique, so neither candidate selection nor output depends on warp
      // scheduling or the request's offset in a packed buffer.
      const uint64_t key =
          (static_cast<uint64_t>(ordered_float_bits(score)) << 32) |
          static_cast<uint32_t>(local_index);
      const uint64_t upper_prefix =
          shift == 64 - kRadixBits ? 0 : key >> (shift + kRadixBits);
      if (upper_prefix == selected_prefix) {
        atomicAdd(&shared.histogram[(key >> shift) & (kRadixBuckets - 1)],
                  1U);
      }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      int remaining_rank = rank;
      int selected_bucket = 0;
      for (int bucket = kRadixBuckets - 1; bucket >= 0; --bucket) {
        const int bucket_count = static_cast<int>(shared.histogram[bucket]);
        if (remaining_rank <= bucket_count) {
          selected_bucket = bucket;
          break;
        }
        remaining_rank -= bucket_count;
      }
      selected_prefix =
          (selected_prefix << kRadixBits) | selected_bucket;
      rank = remaining_rank;
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    selected_count = 0;
  }
  __syncthreads();

  // Composite keys are unique because their low bits contain the row-local
  // index, so exactly effective_top_k keys are >= the selected threshold.
  for (int local_index = threadIdx.x; local_index < row_length;
       local_index += kThreads) {
    const int absolute_index = row_start + local_index;
    const float score =
        logits[static_cast<int64_t>(row) * stride0 +
               static_cast<int64_t>(absolute_index) * stride1];
    const uint64_t key =
        (static_cast<uint64_t>(ordered_float_bits(score)) << 32) |
        static_cast<uint32_t>(local_index);
    if (key >= selected_prefix) {
      const int slot = atomicAdd(&selected_count, 1);
      if (slot < effective_top_k) {
        shared.selected[slot] = key;
      }
    }
  }
  __syncthreads();

#pragma unroll
  for (int item = 0; item < kItemsPerThread; ++item) {
    const int slot = item * kThreads + threadIdx.x;
    score_keys[item] =
        slot < effective_top_k ? shared.selected[slot] : uint64_t{0};
  }
  // All reads from the selected-key view must finish before CUB reuses the
  // same union storage for its sort scratch space.
  __syncthreads();
  ScoreSort(shared.sort).SortDescendingBlockedToStriped(score_keys);
  __syncthreads();

#pragma unroll
  for (int item = 0; item < kItemsPerThread; ++item) {
    const int output_column = item * kThreads + threadIdx.x;
    if (output_column < top_k) {
      output[static_cast<int64_t>(row) * top_k + output_column] =
          output_column < effective_top_k
              ? static_cast<int>(static_cast<uint32_t>(score_keys[item]))
              : -1;
    }
  }
}

void ds4_top_k_per_row_prefill(
    const at::Tensor& logits, const at::Tensor& row_starts,
    const at::Tensor& row_ends, at::Tensor& indices, int64_t num_rows,
    int64_t stride0, int64_t stride1, int64_t top_k) {
  TORCH_CHECK(logits.is_cuda(), "logits must be CUDA");
  TORCH_CHECK(logits.scalar_type() == at::kFloat, "logits must be float32");
  TORCH_CHECK(row_starts.scalar_type() == at::kInt,
              "row_starts must be int32");
  TORCH_CHECK(row_ends.scalar_type() == at::kInt, "row_ends must be int32");
  TORCH_CHECK(indices.scalar_type() == at::kInt, "indices must be int32");
  TORCH_CHECK(logits.dim() == 2, "logits must be rank 2");
  TORCH_CHECK(top_k > 0 && top_k <= kMaxTopK,
              "DS4 deterministic Top-K requires 0 < top_k <= ", kMaxTopK,
              ", got ", top_k);
  TORCH_CHECK(indices.size(0) >= num_rows && indices.size(1) >= top_k,
              "indices output is too small");
  if (num_rows == 0) {
    return;
  }

  const c10::cuda::CUDAGuard device_guard(logits.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  deterministic_top_k_per_row_prefill<<<num_rows, kThreads, 0, stream>>>(
      logits.const_data_ptr<float>(), row_starts.const_data_ptr<int>(),
      row_ends.const_data_ptr<int>(), indices.mutable_data_ptr<int>(), stride0,
      stride1, static_cast<int>(top_k));
}

}  // namespace

TORCH_LIBRARY(ds4_bi, m) {
  m.def(
      "top_k_per_row_prefill(Tensor logits, Tensor row_starts, "
      "Tensor row_ends, Tensor(a!) indices, int num_rows, int stride0, "
      "int stride1, int top_k) -> ()");
}

TORCH_LIBRARY_IMPL(ds4_bi, CUDA, m) {
  m.impl("top_k_per_row_prefill", &ds4_top_k_per_row_prefill);
}
