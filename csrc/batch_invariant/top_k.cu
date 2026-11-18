// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <algorithm>
#include <cstdint>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cub/block/block_radix_sort.cuh>
#include <torch/library.h>

namespace vllm::batch_invariant {
namespace {

constexpr int kThreads = 512;
constexpr int kMaxTopK = 2048;
constexpr int kItemsPerThread = kMaxTopK / kThreads;
constexpr int kRadixBits = 8;
constexpr int kRadixBuckets = 1 << kRadixBits;

__device__ __forceinline__ uint32_t ordered_float_bits(float value) {
  uint32_t bits = __float_as_uint(value);
  // Invalid scores sort with padding; signed zeros share the index tie rule.
  if ((bits & 0x7fffffffU) > 0x7f800000U) bits = 0xff800000U;
  if ((bits & 0x7fffffffU) == 0) bits = 0;
  return (bits & 0x80000000U) ? ~bits : (bits ^ 0x80000000U);
}

__global__ __launch_bounds__(kThreads) void reference_top_k_per_row_prefill(
    const float* logits, const int* row_starts, const int* row_ends,
    int* output, int64_t stride0, int64_t stride1, int top_k) {
  using ScoreSort = cub::BlockRadixSort<uint64_t, kThreads, kItemsPerThread>;
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
      // independent of the generic selector's atomic-slot tie order. The key
      // is unique, so neither candidate selection nor output depends on warp
      // scheduling or the request's offset in a packed buffer.
      const uint64_t key =
          (static_cast<uint64_t>(ordered_float_bits(score)) << 32) |
          static_cast<uint32_t>(local_index);
      const uint64_t upper_prefix =
          shift == 64 - kRadixBits ? 0 : key >> (shift + kRadixBits);
      if (upper_prefix == selected_prefix) {
        atomicAdd(&shared.histogram[(key >> shift) & (kRadixBuckets - 1)], 1U);
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
      selected_prefix = (selected_prefix << kRadixBits) | selected_bucket;
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
    const float score = logits[static_cast<int64_t>(row) * stride0 +
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
          output_column < effective_top_k &&
                  (score_keys[item] >> 32) > ordered_float_bits(-INFINITY)
              ? static_cast<int>(static_cast<uint32_t>(score_keys[item]))
              : -1;
    }
  }
}

// Same output as reference_top_k_per_row_prefill (descending (score,
// local-index) keys, -1 padding), selected faster: the row's score bits are
// staged once in shared memory, histograms are warp-private and
// match-aggregated, buckets are chosen by a parallel scan, the radix passes
// stop as soon as the boundary bucket is taken whole, and rows no longer than
// top_k skip selection. With kIndexOrder the selected indices are written in
// ascending index order instead (-1 padding last), which is what the BI
// sparse attention consumes; this replaces a separate where/sort/copy pass.
constexpr int kWarps = kThreads / 32;
constexpr int kStagedScores = 16384;

template <int kItems, bool kIndexOrder>
__global__ __launch_bounds__(kThreads) void deterministic_top_k_per_row_prefill(
    const float* logits, const int* row_starts, const int* row_ends,
    int* output, int64_t stride0, int64_t stride1, int top_k) {
  using ScoreSort = cub::BlockRadixSort<uint64_t, kThreads, kItems>;
  using IndexSort = cub::BlockRadixSort<uint32_t, kThreads, kItems>;
  constexpr int kCapacity = kThreads * kItems;
  union SharedStorage {
    uint32_t histogram[kWarps][kRadixBuckets];
    uint64_t selected[kCapacity];
    typename ScoreSort::TempStorage sort;
    typename IndexSort::TempStorage index_sort;
  };
  __shared__ SharedStorage shared;
  __shared__ uint32_t warp_sums[kRadixBuckets / 32];
  __shared__ uint64_t selected_prefix;
  __shared__ int rank;
  __shared__ int done_shift;
  __shared__ int selected_count;
  extern __shared__ uint32_t staged[];

  const int row = blockIdx.x;
  const int row_start = row_starts[row];
  const int row_length = max(0, row_ends[row] - row_start);
  const int effective_top_k = min(top_k, row_length);
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;

  if (effective_top_k == 0) {
    for (int c = threadIdx.x; c < top_k; c += kThreads) {
      output[static_cast<int64_t>(row) * top_k + c] = -1;
    }
    return;
  }

  const bool is_staged = row_length <= kStagedScores;
  const float* const row_logits = logits + static_cast<int64_t>(row) * stride0 +
                                  static_cast<int64_t>(row_start) * stride1;
  auto score_bits = [&](int i) -> uint32_t {
    return is_staged ? staged[i]
                     : ordered_float_bits(
                           row_logits[static_cast<int64_t>(i) * stride1]);
  };
  if (is_staged) {
    for (int i = threadIdx.x; i < row_length; i += kThreads) {
      staged[i] =
          ordered_float_bits(row_logits[static_cast<int64_t>(i) * stride1]);
    }
  }
  if (threadIdx.x == 0) {
    selected_prefix = 0;
    rank = effective_top_k;
    done_shift = row_length == effective_top_k ? 64 : -1;
  }
  __syncthreads();

  for (int shift = 64 - kRadixBits; shift >= 0 && done_shift < 0;
       shift -= kRadixBits) {
    for (int b = threadIdx.x; b < kWarps * kRadixBuckets; b += kThreads) {
      (&shared.histogram[0][0])[b] = 0;
    }
    // Read the pass state before the scan phase below rewrites it.
    const uint64_t prefix = selected_prefix;
    const uint32_t need = static_cast<uint32_t>(rank);
    __syncthreads();
    const int iters = (row_length + kThreads - 1) / kThreads;
    for (int it = 0; it < iters; ++it) {
      const int i = it * kThreads + threadIdx.x;
      int bucket = -1;
      if (i < row_length) {
        const uint64_t key = (static_cast<uint64_t>(score_bits(i)) << 32) |
                             static_cast<uint32_t>(i);
        const uint64_t upper =
            shift == 64 - kRadixBits ? 0 : key >> (shift + kRadixBits);
        if (upper == prefix) {
          bucket = static_cast<int>((key >> shift) & (kRadixBuckets - 1));
        }
      }
      const unsigned peers = __match_any_sync(0xffffffffU, bucket);
      if (bucket >= 0 && lane == __ffs(peers) - 1) {
        shared.histogram[warp][bucket] += __popc(peers);
      }
    }
    __syncthreads();

    // Thread t owns bucket 255 - t; an inclusive scan over t gives the number
    // of candidate keys in this bucket or a larger one.
    uint32_t count = 0, inclusive = 0;
    if (threadIdx.x < kRadixBuckets) {
      const int bucket = kRadixBuckets - 1 - threadIdx.x;
#pragma unroll
      for (int w = 0; w < kWarps; ++w) count += shared.histogram[w][bucket];
      inclusive = count;
#pragma unroll
      for (int d = 1; d < 32; d <<= 1) {
        const uint32_t v = __shfl_up_sync(0xffffffffU, inclusive, d);
        if (lane >= d) inclusive += v;
      }
      if (lane == 31) warp_sums[warp] = inclusive;
    }
    __syncthreads();
    if (threadIdx.x < kRadixBuckets) {
      for (int w = 0; w < warp; ++w) inclusive += warp_sums[w];
      if (count > 0 && inclusive >= need && inclusive - count < need) {
        const int bucket = kRadixBuckets - 1 - threadIdx.x;
        const uint32_t remaining = need - (inclusive - count);
        selected_prefix = (prefix << kRadixBits) | bucket;
        rank = static_cast<int>(remaining);
        if (remaining == count) done_shift = shift;
      }
    }
    __syncthreads();
  }

  // Keys at or above the threshold are exactly the effective_top_k largest.
  const uint64_t threshold =
      done_shift >= 64 ? 0 : selected_prefix << done_shift;
  if (threadIdx.x == 0) selected_count = 0;
  __syncthreads();
  for (int i = threadIdx.x; i < row_length; i += kThreads) {
    const uint64_t key =
        (static_cast<uint64_t>(score_bits(i)) << 32) | static_cast<uint32_t>(i);
    if (key >= threshold) {
      const int slot = atomicAdd(&selected_count, 1);
      if (slot < effective_top_k) shared.selected[slot] = key;
    }
  }
  __syncthreads();

  if constexpr (kIndexOrder) {
    // Valid indices ascending, then -1 for -inf scores and padding. Rows
    // shorter than 2^16 sort only the low 16 bits (indices < 0xffff).
    const bool short_row = row_length < 0xffff;
    const uint32_t kInvalid = short_row ? 0xffffU : 0xffffffffU;
    uint32_t indices[kItems];
#pragma unroll
    for (int item = 0; item < kItems; ++item) {
      const int slot = item * kThreads + threadIdx.x;
      const uint64_t key =
          slot < effective_top_k ? shared.selected[slot] : uint64_t{0};
      indices[item] =
          slot < effective_top_k && (key >> 32) > ordered_float_bits(-INFINITY)
              ? static_cast<uint32_t>(key)
              : kInvalid;
    }
    __syncthreads();
    if (short_row) {
      IndexSort(shared.index_sort).SortBlockedToStriped(indices, 0, 16);
    } else {
      IndexSort(shared.index_sort).SortBlockedToStriped(indices);
    }
#pragma unroll
    for (int item = 0; item < kItems; ++item) {
      const int c = item * kThreads + threadIdx.x;
      if (c < top_k) {
        output[static_cast<int64_t>(row) * top_k + c] =
            indices[item] == kInvalid ? -1 : static_cast<int>(indices[item]);
      }
    }
    return;
  }

  // Rows shorter than 2^16 sort a packed (score << 16 | index) key over 48
  // bits; the order is the same as the full 64-bit key's.
  const bool packed = row_length <= (1 << 16);
  uint64_t keys[kItems];
#pragma unroll
  for (int item = 0; item < kItems; ++item) {
    const int slot = item * kThreads + threadIdx.x;
    uint64_t key = slot < effective_top_k ? shared.selected[slot] : uint64_t{0};
    if (packed) key = ((key >> 32) << 16) | (key & 0xffffU);
    keys[item] = key;
  }
  __syncthreads();
  if (packed) {
    ScoreSort(shared.sort).SortDescendingBlockedToStriped(keys, 0, 48);
#pragma unroll
    for (int item = 0; item < kItems; ++item) {
      keys[item] = ((keys[item] >> 16) << 32) | (keys[item] & 0xffffU);
    }
  } else {
    ScoreSort(shared.sort).SortDescendingBlockedToStriped(keys);
  }

#pragma unroll
  for (int item = 0; item < kItems; ++item) {
    const int c = item * kThreads + threadIdx.x;
    if (c < top_k) {
      output[static_cast<int64_t>(row) * top_k + c] =
          c < effective_top_k &&
                  (keys[item] >> 32) > ordered_float_bits(-INFINITY)
              ? static_cast<int>(static_cast<uint32_t>(keys[item]))
              : -1;
    }
  }
}

}  // namespace

namespace {
bool check_top_k_args(const at::Tensor& logits, const at::Tensor& row_starts,
                      const at::Tensor& row_ends, const at::Tensor& indices,
                      int64_t num_rows, int64_t stride0, int64_t stride1,
                      int64_t top_k) {
  TORCH_CHECK(logits.is_cuda(), "logits must be CUDA");
  TORCH_CHECK(logits.scalar_type() == at::kFloat, "logits must be float32");
  TORCH_CHECK(row_starts.scalar_type() == at::kInt, "row_starts must be int32");
  TORCH_CHECK(row_ends.scalar_type() == at::kInt, "row_ends must be int32");
  TORCH_CHECK(indices.scalar_type() == at::kInt, "indices must be int32");
  TORCH_CHECK(logits.dim() == 2, "logits must be rank 2");
  TORCH_CHECK(row_starts.device() == logits.device() &&
                  row_ends.device() == logits.device() &&
                  indices.device() == logits.device(),
              "Top-K tensors must share a CUDA device");
  TORCH_CHECK(num_rows >= 0 && num_rows <= logits.size(0),
              "invalid query row count");
  TORCH_CHECK(row_starts.dim() == 1 && row_ends.dim() == 1 &&
                  row_starts.is_contiguous() && row_ends.is_contiguous() &&
                  row_starts.numel() >= num_rows &&
                  row_ends.numel() >= num_rows,
              "row bounds must be contiguous int32 vectors");
  TORCH_CHECK(stride0 == logits.stride(0) && stride1 == logits.stride(1) &&
                  stride0 > 0 && stride1 > 0,
              "Top-K input strides must match the tensor");
  TORCH_CHECK(top_k > 0 && top_k <= kMaxTopK,
              "DS4 deterministic Top-K requires 0 < top_k <= ", kMaxTopK,
              ", got ", top_k);
  TORCH_CHECK(indices.dim() == 2 && indices.is_contiguous() &&
                  indices.size(0) >= num_rows && indices.size(1) == top_k,
              "indices must be contiguous with top_k columns");
  if (num_rows == 0) {
    return false;
  }

  return true;
}

template <int kItems, bool kIndexOrder>
void launch_top_k(const at::Tensor& logits, const at::Tensor& row_starts,
                  const at::Tensor& row_ends, at::Tensor& indices,
                  int64_t num_rows, int64_t stride0, int64_t stride1,
                  int64_t top_k, cudaStream_t stream) {
  constexpr int kSmem = kStagedScores * sizeof(uint32_t);
  static const bool configured = [] {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        deterministic_top_k_per_row_prefill<kItems, kIndexOrder>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
    return true;
  }();
  (void)configured;
  deterministic_top_k_per_row_prefill<kItems, kIndexOrder>
      <<<num_rows, kThreads, kSmem, stream>>>(
          logits.const_data_ptr<float>(), row_starts.const_data_ptr<int>(),
          row_ends.const_data_ptr<int>(), indices.mutable_data_ptr<int>(),
          stride0, stride1, static_cast<int>(top_k));
}
template <bool kIndexOrder>
void run_top_k(const at::Tensor& logits, const at::Tensor& row_starts,
               const at::Tensor& row_ends, at::Tensor& indices,
               int64_t num_rows, int64_t stride0, int64_t stride1,
               int64_t top_k) {
  if (!check_top_k_args(logits, row_starts, row_ends, indices, num_rows,
                        stride0, stride1, top_k)) {
    return;
  }
  const c10::cuda::CUDAGuard device_guard(logits.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (top_k <= kThreads) {
    launch_top_k<1, kIndexOrder>(logits, row_starts, row_ends, indices,
                                 num_rows, stride0, stride1, top_k, stream);
  } else {
    launch_top_k<kItemsPerThread, kIndexOrder>(logits, row_starts, row_ends,
                                               indices, num_rows, stride0,
                                               stride1, top_k, stream);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}  // namespace

void top_k_per_row_prefill(const at::Tensor& logits,
                           const at::Tensor& row_starts,
                           const at::Tensor& row_ends, at::Tensor& indices,
                           int64_t num_rows, int64_t stride0, int64_t stride1,
                           int64_t top_k) {
  run_top_k<false>(logits, row_starts, row_ends, indices, num_rows, stride0,
                   stride1, top_k);
}

void top_k_per_row_prefill_index_order(const at::Tensor& logits,
                                       const at::Tensor& row_starts,
                                       const at::Tensor& row_ends,
                                       at::Tensor& indices, int64_t num_rows,
                                       int64_t stride0, int64_t stride1,
                                       int64_t top_k) {
  run_top_k<true>(logits, row_starts, row_ends, indices, num_rows, stride0,
                  stride1, top_k);
}

void top_k_per_row_prefill_reference(const at::Tensor& logits,
                                     const at::Tensor& row_starts,
                                     const at::Tensor& row_ends,
                                     at::Tensor& indices, int64_t num_rows,
                                     int64_t stride0, int64_t stride1,
                                     int64_t top_k) {
  if (!check_top_k_args(logits, row_starts, row_ends, indices, num_rows,
                        stride0, stride1, top_k)) {
    return;
  }
  const c10::cuda::CUDAGuard device_guard(logits.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  reference_top_k_per_row_prefill<<<num_rows, kThreads, 0, stream>>>(
      logits.const_data_ptr<float>(), row_starts.const_data_ptr<int>(),
      row_ends.const_data_ptr<int>(), indices.mutable_data_ptr<int>(), stride0,
      stride1, static_cast<int>(top_k));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace vllm::batch_invariant
