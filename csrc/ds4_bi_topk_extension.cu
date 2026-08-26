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
constexpr int kMaxColumns = 4096;
constexpr int kItemsPerThread = kMaxColumns / kThreads;
constexpr int kMaxTopK = 2048;

__device__ __forceinline__ uint32_t ordered_float_bits(float value) {
  const uint32_t bits = __float_as_uint(value);
  return (bits & 0x80000000U) ? ~bits : (bits ^ 0x80000000U);
}

__global__ __launch_bounds__(kThreads) void deterministic_top_k_per_row_prefill(
    const float* logits, const int* row_starts, const int* row_ends,
    int* output, int64_t stride0, int64_t stride1, int top_k) {
  using ScoreSort =
      cub::BlockRadixSort<uint64_t, kThreads, kItemsPerThread>;
  __shared__ typename ScoreSort::TempStorage temp;

  const int row = blockIdx.x;
  const int row_start = row_starts[row];
  const int row_end = row_ends[row];
  uint64_t score_keys[kItemsPerThread];

#pragma unroll
  for (int item = 0; item < kItemsPerThread; ++item) {
    const int local_index = item * kThreads + threadIdx.x;
    if (row_start + local_index < row_end) {
      const int absolute_index = row_start + local_index;
      const float score =
          logits[static_cast<int64_t>(row) * stride0 +
                 static_cast<int64_t>(absolute_index) * stride1];
      // Descending score, then descending source index, matching vLLM's
      // insertion-sort tie semantics. The key is unique, so
      // neither candidate selection nor output depends on warp scheduling.
      score_keys[item] =
          (static_cast<uint64_t>(ordered_float_bits(score)) << 32) |
          static_cast<uint32_t>(absolute_index);
    } else {
      score_keys[item] = 0;
    }
  }
  ScoreSort(temp).SortDescendingBlockedToStriped(score_keys);
  __syncthreads();

#pragma unroll
  for (int item = 0; item < kItemsPerThread; ++item) {
    const int output_column = item * kThreads + threadIdx.x;
    if (output_column < top_k) {
      output[static_cast<int64_t>(row) * top_k + output_column] =
          score_keys[item] == 0
              ? -1
              : static_cast<int>(static_cast<uint32_t>(score_keys[item]));
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
  TORCH_CHECK(logits.size(1) <= kMaxColumns,
              "DS4 deterministic Top-K supports at most ", kMaxColumns,
              " packed columns, got ", logits.size(1));
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
