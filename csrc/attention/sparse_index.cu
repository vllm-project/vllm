// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <algorithm>

#include "../torch_utils.h"

#include <cub/block/block_radix_sort.cuh>

namespace vllm::batch_invariant {

namespace {

constexpr int kCombineThreads = 256;

template <int Capacity>
__global__
__launch_bounds__(Capacity ? 128 : 256) void combine_swa_decode_kernel(
    int* combined_indices, int* combined_lens, const int* topk_indices,
    const int* seq_lens, const bool* is_valid, int64_t output_stride,
    int64_t topk_stride, int output_width, int M, int N, int top_k,
    int compress_ratio, int window_size) {
  constexpr int threads = Capacity ? 128 : 256;
  constexpr int items = Capacity ? Capacity / threads : 1;
  const int row = blockIdx.x;
  const int seq_len = seq_lens[row];
  const int topk_len = min(seq_len / compress_ratio, top_k);
  const int swa_len = min(seq_len, window_size);
  const int row_base = M * row;
  unsigned sorted_indices[items];

  if constexpr (Capacity) {
    using Sort =
        cub::BlockRadixSort<unsigned, threads, items, cub::NullType, 5>;
    __shared__ typename Sort::TempStorage sort_storage;
#pragma unroll
    for (int item = 0; item < items; ++item) {
      const int column = threadIdx.x * items + item;
      sorted_indices[item] =
          column < topk_len
              ? static_cast<unsigned>(
                    topk_indices[static_cast<int64_t>(row) * topk_stride +
                                 column]) +
                    1U
              : 0U;
    }
    if (topk_len > 1) {
      const int end_bit = 32 - __clz(static_cast<unsigned>(max(N, 1)));
      Sort(sort_storage)
          .SortDescendingBlockedToStriped(sorted_indices, 0, end_bit);
    }
  }

  for (int column = threadIdx.x; column < output_width; column += blockDim.x) {
    int value = -1;
    if (column < topk_len) {
      if constexpr (Capacity) {
        value =
            static_cast<int>(sorted_indices[column / threads]) - 1 + row_base;
      } else {
        value = row_base + topk_len - 1 - column;
      }
    } else if (column < topk_len + swa_len) {
      value = row_base + N + column - topk_len;
    }
    combined_indices[static_cast<int64_t>(row) * output_stride + column] =
        value;
  }
  if (threadIdx.x == 0) {
    combined_lens[row] = is_valid[row] ? topk_len + swa_len : 0;
  }
}

}  // namespace

void combine_topk_swa_decode(torch::stable::Tensor& combined_indices,
                             torch::stable::Tensor& combined_lens,
                             const torch::stable::Tensor& topk_indices,
                             const torch::stable::Tensor& seq_lens,
                             const torch::stable::Tensor& is_valid, int64_t M,
                             int64_t N, int64_t top_k, int64_t compress_ratio,
                             int64_t window_size) {
  STD_TORCH_CHECK(
      combined_indices.device().is_cuda() && combined_lens.device().is_cuda() &&
          topk_indices.device().is_cuda() && seq_lens.device().is_cuda() &&
          is_valid.device().is_cuda(),
      "decode tensors must be CUDA");
  STD_TORCH_CHECK(
      combined_indices.scalar_type() == torch::headeronly::ScalarType::Int &&
          combined_lens.scalar_type() == torch::headeronly::ScalarType::Int &&
          topk_indices.scalar_type() == torch::headeronly::ScalarType::Int &&
          seq_lens.scalar_type() == torch::headeronly::ScalarType::Int,
      "decode index tensors must be int32");
  STD_TORCH_CHECK(is_valid.scalar_type() == torch::headeronly::ScalarType::Bool,
                  "is_valid must be bool");
  STD_TORCH_CHECK(combined_indices.dim() == 2 && topk_indices.dim() == 2,
                  "index tensors must be rank 2");
  const int64_t num_rows = seq_lens.numel();
  STD_TORCH_CHECK(combined_indices.size(0) == num_rows &&
                      combined_lens.numel() == num_rows &&
                      topk_indices.size(0) == num_rows &&
                      is_valid.numel() == num_rows,
                  "decode tensors must have the same row count");
  STD_TORCH_CHECK(
      combined_indices.stride(1) == 1 && topk_indices.stride(1) == 1,
      "index rows must be contiguous");
  STD_TORCH_CHECK(top_k >= 0 && top_k <= topk_indices.size(1) && top_k <= 512,
                  "fused decode combine supports top_k <= 512");
  STD_TORCH_CHECK(compress_ratio > 0 && window_size >= 0 && N >= 0,
                  "invalid sparse attention dimensions");
  if (num_rows == 0) {
    return;
  }

  const torch::stable::accelerator::DeviceGuard device_guard(
      combined_indices.get_device_index());
  const cudaStream_t stream =
      get_current_cuda_stream(combined_indices.get_device_index());
  const auto launch = [&]<int Capacity>() {
    combine_swa_decode_kernel<Capacity><<<num_rows, 128, 0, stream>>>(
        combined_indices.mutable_data_ptr<int>(),
        combined_lens.mutable_data_ptr<int>(),
        topk_indices.const_data_ptr<int>(), seq_lens.const_data_ptr<int>(),
        is_valid.const_data_ptr<bool>(), combined_indices.stride(0),
        topk_indices.stride(0), combined_indices.size(1), static_cast<int>(M),
        static_cast<int>(N), static_cast<int>(top_k),
        static_cast<int>(compress_ratio), static_cast<int>(window_size));
  };
  if (top_k <= 128) {
    launch.template operator()<128>();
  } else if (top_k <= 256) {
    launch.template operator()<256>();
  } else {
    launch.template operator()<512>();
  }
}

void combine_c128_swa_decode(torch::stable::Tensor& combined_indices,
                             torch::stable::Tensor& combined_lens,
                             const torch::stable::Tensor& seq_lens,
                             const torch::stable::Tensor& is_valid, int64_t M,
                             int64_t N, int64_t top_k, int64_t compress_ratio,
                             int64_t window_size) {
  STD_TORCH_CHECK(
      combined_indices.device().is_cuda() && combined_lens.device().is_cuda() &&
          seq_lens.device().is_cuda() && is_valid.device().is_cuda(),
      "decode tensors must be CUDA");
  STD_TORCH_CHECK(
      combined_indices.scalar_type() == torch::headeronly::ScalarType::Int &&
          combined_lens.scalar_type() == torch::headeronly::ScalarType::Int &&
          seq_lens.scalar_type() == torch::headeronly::ScalarType::Int,
      "decode index tensors must be int32");
  STD_TORCH_CHECK(is_valid.scalar_type() == torch::headeronly::ScalarType::Bool,
                  "is_valid must be bool");
  const int64_t num_rows = seq_lens.numel();
  STD_TORCH_CHECK(
      combined_indices.dim() == 2 && combined_indices.size(0) == num_rows &&
          combined_lens.numel() == num_rows && is_valid.numel() == num_rows,
      "decode tensors must have the same row count");
  STD_TORCH_CHECK(combined_indices.stride(1) == 1,
                  "index rows must be contiguous");
  STD_TORCH_CHECK(top_k >= 0 && top_k <= combined_indices.size(1),
                  "top_k exceeds the output width");
  STD_TORCH_CHECK(compress_ratio > 0 && window_size >= 0,
                  "invalid sparse attention dimensions");
  if (num_rows == 0) {
    return;
  }

  const torch::stable::accelerator::DeviceGuard device_guard(
      combined_indices.get_device_index());
  const cudaStream_t stream =
      get_current_cuda_stream(combined_indices.get_device_index());
  combine_swa_decode_kernel<0><<<num_rows, kCombineThreads, 0, stream>>>(
      combined_indices.mutable_data_ptr<int>(),
      combined_lens.mutable_data_ptr<int>(), nullptr,
      seq_lens.const_data_ptr<int>(), is_valid.const_data_ptr<bool>(),
      combined_indices.stride(0), 0, combined_indices.size(1),
      static_cast<int>(M), static_cast<int>(N), static_cast<int>(top_k),
      static_cast<int>(compress_ratio), static_cast<int>(window_size));
}

}  // namespace vllm::batch_invariant
