// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Direct symmetric-memory DCP A2A LSE reduction.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>
#include <torch/csrc/stable/library.h>
#include <torch/headeronly/core/ScalarType.h>

#include <cstdio>
#include <optional>
#include <type_traits>

#include "dcp_direct_common.cuh"

namespace {

using vllm::direct_dcp::check_cuda_launch;
using vllm::direct_dcp::get_peer_ptr;
using vllm::direct_dcp::increment_epoch_kernel;
using vllm::direct_dcp::store_release_system;
using vllm::direct_dcp::wait_for_epoch;

__device__ __forceinline__ int64_t find_sequence(const int32_t* query_start_loc,
                                                 int64_t token_idx,
                                                 int64_t num_seqs) {
  int64_t left = 0;
  int64_t right = num_seqs;
  while (left < right) {
    int64_t mid = (left + right) / 2;
    if (query_start_loc[mid] <= token_idx) {
      left = mid + 1;
    } else {
      right = mid;
    }
  }
  return left - 1;
}

template <typename scalar_t>
__device__ __forceinline__ float to_float(scalar_t value) {
  if constexpr (std::is_same_v<scalar_t, float>) {
    return value;
  } else if constexpr (std::is_same_v<scalar_t, __half>) {
    return __half2float(value);
  } else {
    return __bfloat162float(value);
  }
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t from_float(float value) {
  if constexpr (std::is_same_v<scalar_t, __half>) {
    return __float2half_rn(value);
  } else {
    return __float2bfloat16_rn(value);
  }
}

// Send each destination's head slice and LSE to its symmetric buffer. Mask
// undefined empty-shard rows at the source so peers give them zero weight.
template <typename lse_t>
__global__ void dispatch_output_lse_kernel(
    const uint4* partial_output, const lse_t* partial_lse,
    const int32_t* seq_lens, const int32_t* query_start_loc,
    const int64_t* peer_output_ptrs, const int64_t* peer_lse_ptrs,
    const int64_t* epoch_ptr, int64_t world_size, int64_t rank,
    int64_t num_tokens, int64_t max_num_tokens, int64_t num_seqs,
    int64_t heads_per_rank, int64_t head_dim, int64_t output_token_stride,
    int64_t lse_token_stride) {
  int64_t item = static_cast<int64_t>(blockIdx.x);
  int64_t destination_rank = item / num_tokens;
  int64_t token_idx = item - destination_rank * num_tokens;

  __shared__ bool empty_kv;
  if (threadIdx.x == 0) {
    empty_kv = false;
    if (seq_lens != nullptr) {
      int64_t seq_idx = find_sequence(query_start_loc, token_idx, num_seqs);
      empty_kv = seq_lens[seq_idx] == 0;
    }
  }
  __syncthreads();

  uint32_t epoch = static_cast<uint32_t>(epoch_ptr[0]);
  int64_t parity = static_cast<int64_t>(epoch & 1u);
  int64_t destination_item =
      ((parity * world_size + rank) * max_num_tokens + token_idx) *
      heads_per_rank;
  int64_t source_head = destination_rank * heads_per_rank;

  if (!empty_kv) {
    uint4* peer_output =
        get_peer_ptr<uint4>(peer_output_ptrs, destination_rank);
    int64_t vectors_per_item = heads_per_rank * head_dim / 8;
    int64_t source_vector =
        (token_idx * output_token_stride + source_head * head_dim) / 8;
    int64_t destination_vector = destination_item * head_dim / 8;
    for (int64_t vector_idx = threadIdx.x; vector_idx < vectors_per_item;
         vector_idx += blockDim.x) {
      peer_output[destination_vector + vector_idx] =
          partial_output[source_vector + vector_idx];
    }
  }

  float* peer_lse = get_peer_ptr<float>(peer_lse_ptrs, destination_rank);
  int64_t source_lse = token_idx * lse_token_stride + source_head;
  for (int64_t head_idx = threadIdx.x; head_idx < heads_per_rank;
       head_idx += blockDim.x) {
    peer_lse[destination_item + head_idx] =
        empty_kv ? -CUDART_INF_F : to_float(partial_lse[source_lse + head_idx]);
  }
}

// Publish completion after the stream-ordered payload writes.
__global__ void signal_kernel(const int64_t* peer_signal_ptrs,
                              const int64_t* epoch_ptr, int64_t world_size,
                              int64_t rank) {
  int64_t destination_rank =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (destination_rank >= world_size) {
    return;
  }

  uint32_t epoch = static_cast<uint32_t>(epoch_ptr[0]);
  int64_t parity = static_cast<int64_t>(epoch & 1u);
  uint32_t* peer_signal =
      get_peer_ptr<uint32_t>(peer_signal_ptrs, destination_rank);
  int64_t signal_item = parity * world_size + rank;
  store_release_system(peer_signal + signal_item, epoch);
}

// Form stable base-2 LSE weights and combine in FP32.
template <typename scalar_t>
__global__ void wait_lse_combine_kernel(
    const scalar_t* received_output, const float* received_lse,
    const uint32_t* received_signal, const int64_t* epoch_ptr,
    scalar_t* combined_output, int64_t world_size, int64_t num_tokens,
    int64_t max_num_tokens, int64_t heads_per_rank, int64_t head_dim,
    bool is_lse_base_on_e) {
  extern __shared__ float weights[];

  int64_t item = static_cast<int64_t>(blockIdx.x);
  int64_t token_idx = item / heads_per_rank;
  int64_t head_idx = item - token_idx * heads_per_rank;
  uint32_t epoch = static_cast<uint32_t>(epoch_ptr[0]);
  int64_t parity = static_cast<int64_t>(epoch & 1u);

  if (threadIdx.x == 0) {
    float lse_max = -CUDART_INF_F;
    for (int64_t source_rank = 0; source_rank < world_size; ++source_rank) {
      int64_t source_item =
          ((parity * world_size + source_rank) * max_num_tokens + token_idx) *
              heads_per_rank +
          head_idx;
      int64_t signal_item = parity * world_size + source_rank;
      if (!wait_for_epoch(received_signal + signal_item, epoch)) {
        printf(
            "direct DCP A2A timeout source=%lld token=%lld head=%lld "
            "epoch=%u\n",
            static_cast<long long>(source_rank),
            static_cast<long long>(token_idx), static_cast<long long>(head_idx),
            epoch);
        asm volatile("trap;");
      }
      float value = received_lse[source_item];
      if (isnan(value) || value == CUDART_INF_F) {
        value = -CUDART_INF_F;
      }
      if (is_lse_base_on_e) {
        value *= CUDART_L2E_F;
      }
      weights[source_rank] = value;
      lse_max = fmaxf(lse_max, value);
    }
    if (lse_max == -CUDART_INF_F) {
      lse_max = 0.0f;
    }

    float lse_sum = 0.0f;
    for (int64_t source_rank = 0; source_rank < world_size; ++source_rank) {
      lse_sum += exp2f(weights[source_rank] - lse_max);
    }
    float inverse_lse_sum = lse_sum > 0.0f ? 1.0f / lse_sum : 0.0f;
    for (int64_t source_rank = 0; source_rank < world_size; ++source_rank) {
      weights[source_rank] =
          exp2f(weights[source_rank] - lse_max) * inverse_lse_sum;
    }
  }
  __syncthreads();

  for (int64_t dim_idx = threadIdx.x; dim_idx < head_dim;
       dim_idx += blockDim.x) {
    float accumulator = 0.0f;
    for (int64_t source_rank = 0; source_rank < world_size; ++source_rank) {
      // Empty shards leave stale payloads, so never read zero-weight sources.
      float weight = weights[source_rank];
      if (weight == 0.0f) {
        continue;
      }
      int64_t source_item =
          ((parity * world_size + source_rank) * max_num_tokens + token_idx) *
              heads_per_rank +
          head_idx;
      accumulator +=
          to_float(received_output[source_item * head_dim + dim_idx]) * weight;
    }
    combined_output[item * head_dim + dim_idx] =
        from_float<scalar_t>(accumulator);
  }
}

void direct_dcp_a2a_lse_reduce(
    const torch::stable::Tensor& partial_output,
    const torch::stable::Tensor& partial_lse,
    const std::optional<torch::stable::Tensor>& seq_lens,
    const std::optional<torch::stable::Tensor>& query_start_loc,
    const torch::stable::Tensor& peer_output_ptrs,
    const torch::stable::Tensor& peer_lse_ptrs,
    const torch::stable::Tensor& peer_signal_ptrs,
    torch::stable::Tensor& received_output, torch::stable::Tensor& received_lse,
    torch::stable::Tensor& received_signal, torch::stable::Tensor& epoch,
    torch::stable::Tensor& combined_output, int64_t world_size, int64_t rank,
    int64_t max_num_tokens, bool is_lse_base_on_e) {
  STD_TORCH_CHECK(partial_output.is_cuda() && partial_lse.is_cuda(),
                  "partial output and LSE must be CUDA tensors");
  auto output_dtype = partial_output.scalar_type();
  STD_TORCH_CHECK(output_dtype == torch::headeronly::ScalarType::Half ||
                      output_dtype == torch::headeronly::ScalarType::BFloat16,
                  "direct DCP A2A only supports FP16 and BF16 output");
  auto lse_dtype = partial_lse.scalar_type();
  STD_TORCH_CHECK(lse_dtype == torch::headeronly::ScalarType::Float ||
                      lse_dtype == torch::headeronly::ScalarType::Half ||
                      lse_dtype == torch::headeronly::ScalarType::BFloat16,
                  "partial LSE must be FP32, FP16, or BF16");
  STD_TORCH_CHECK(partial_output.dim() == 3 && partial_lse.dim() == 2,
                  "expected output [T,H,D] and LSE [T,H]");
  STD_TORCH_CHECK(world_size > 1, "world_size must be greater than 1");
  STD_TORCH_CHECK(rank >= 0 && rank < world_size, "invalid rank");

  int64_t num_tokens = partial_output.size(0);
  int64_t total_heads = partial_output.size(1);
  int64_t head_dim = partial_output.size(2);
  int64_t output_token_stride = partial_output.stride(0);
  int64_t lse_token_stride = partial_lse.stride(0);
  STD_TORCH_CHECK(
      partial_output.stride(2) == 1 && partial_output.stride(1) == head_dim &&
          output_token_stride >= total_heads * head_dim &&
          output_token_stride % 8 == 0,
      "partial output must have packed heads and an aligned token stride");
  STD_TORCH_CHECK(partial_lse.stride(1) == 1 && lse_token_stride >= total_heads,
                  "partial LSE must have packed heads");
  STD_TORCH_CHECK(num_tokens > 0 && num_tokens <= max_num_tokens,
                  "token count exceeds symmetric buffer capacity");
  STD_TORCH_CHECK(total_heads % world_size == 0,
                  "attention heads must divide evenly across DCP ranks");
  STD_TORCH_CHECK(
      partial_lse.size(0) == num_tokens && partial_lse.size(1) == total_heads,
      "LSE shape must match attention output");
  STD_TORCH_CHECK(head_dim % 8 == 0,
                  "head_dim must be divisible by 8 for 16-byte stores");
  int64_t heads_per_rank = total_heads / world_size;
  STD_TORCH_CHECK(combined_output.scalar_type() == output_dtype &&
                      combined_output.is_contiguous() &&
                      combined_output.is_cuda(),
                  "combined output must match the contiguous CUDA input");
  STD_TORCH_CHECK(combined_output.size(0) == num_tokens &&
                      combined_output.size(1) == heads_per_rank &&
                      combined_output.size(2) == head_dim,
                  "combined output has the wrong shape");
  STD_TORCH_CHECK(
      received_output.scalar_type() == output_dtype &&
          received_lse.scalar_type() == torch::headeronly::ScalarType::Float &&
          received_signal.scalar_type() == torch::headeronly::ScalarType::Int,
      "invalid symmetric staging dtypes");
  STD_TORCH_CHECK(
      peer_output_ptrs.scalar_type() == torch::headeronly::ScalarType::Long &&
          peer_lse_ptrs.scalar_type() == torch::headeronly::ScalarType::Long &&
          peer_signal_ptrs.scalar_type() == torch::headeronly::ScalarType::Long,
      "peer pointer tables must be int64");
  const int32_t* seq_lens_ptr = nullptr;
  const int32_t* query_start_loc_ptr = nullptr;
  int64_t num_seqs = 0;
  STD_TORCH_CHECK(seq_lens.has_value() == query_start_loc.has_value(),
                  "seq_lens and query_start_loc must be provided together");
  if (seq_lens.has_value() && query_start_loc.has_value()) {
    STD_TORCH_CHECK(
        seq_lens->is_cuda() && seq_lens->dim() == 1 &&
            seq_lens->stride(0) == 1 &&
            seq_lens->scalar_type() == torch::headeronly::ScalarType::Int,
        "seq_lens must be a contiguous 1-D int32 CUDA tensor");
    STD_TORCH_CHECK(
        query_start_loc->is_cuda() && query_start_loc->dim() == 1 &&
            query_start_loc->stride(0) == 1 &&
            query_start_loc->scalar_type() ==
                torch::headeronly::ScalarType::Int,
        "query_start_loc must be a contiguous 1-D int32 CUDA tensor");
    num_seqs = seq_lens->size(0);
    STD_TORCH_CHECK(num_seqs > 0 && query_start_loc->size(0) == num_seqs + 1,
                    "query_start_loc must contain one boundary per sequence");
    seq_lens_ptr = seq_lens->const_data_ptr<int32_t>();
    query_start_loc_ptr = query_start_loc->const_data_ptr<int32_t>();
  }

  const torch::stable::accelerator::DeviceGuard device_guard(
      partial_output.get_device_index());
  cudaStream_t stream = get_current_cuda_stream();
  constexpr int kExchangeThreads = 256;
  constexpr int kCombineThreads = 128;

  increment_epoch_kernel<<<1, 1, 0, stream>>>(
      epoch.mutable_data_ptr<int64_t>());
  check_cuda_launch("direct DCP A2A");
  int64_t dispatch_blocks = world_size * num_tokens;
  auto launch_dispatch = [&]<typename lse_t>() {
    dispatch_output_lse_kernel<lse_t>
        <<<dispatch_blocks, kExchangeThreads, 0, stream>>>(
            reinterpret_cast<const uint4*>(partial_output.data_ptr()),
            reinterpret_cast<const lse_t*>(partial_lse.data_ptr()),
            seq_lens_ptr, query_start_loc_ptr,
            peer_output_ptrs.const_data_ptr<int64_t>(),
            peer_lse_ptrs.const_data_ptr<int64_t>(),
            epoch.const_data_ptr<int64_t>(), world_size, rank, num_tokens,
            max_num_tokens, num_seqs, heads_per_rank, head_dim,
            output_token_stride, lse_token_stride);
  };
  if (lse_dtype == torch::headeronly::ScalarType::Float) {
    launch_dispatch.operator()<float>();
  } else if (lse_dtype == torch::headeronly::ScalarType::BFloat16) {
    launch_dispatch.operator()<__nv_bfloat16>();
  } else {
    launch_dispatch.operator()<__half>();
  }
  check_cuda_launch("direct DCP A2A");
  int64_t signal_items = world_size;
  int64_t signal_blocks =
      (signal_items + kExchangeThreads - 1) / kExchangeThreads;
  signal_kernel<<<signal_blocks, kExchangeThreads, 0, stream>>>(
      peer_signal_ptrs.const_data_ptr<int64_t>(),
      epoch.const_data_ptr<int64_t>(), world_size, rank);
  check_cuda_launch("direct DCP A2A");
  int64_t combine_blocks = num_tokens * heads_per_rank;
  size_t shared_memory_bytes = world_size * sizeof(float);
  auto launch_combine = [&]<typename scalar_t>() {
    wait_lse_combine_kernel<scalar_t>
        <<<combine_blocks, kCombineThreads, shared_memory_bytes, stream>>>(
            reinterpret_cast<const scalar_t*>(received_output.data_ptr()),
            received_lse.const_data_ptr<float>(),
            reinterpret_cast<const uint32_t*>(
                received_signal.const_data_ptr<int32_t>()),
            epoch.const_data_ptr<int64_t>(),
            reinterpret_cast<scalar_t*>(combined_output.mutable_data_ptr()),
            world_size, num_tokens, max_num_tokens, heads_per_rank, head_dim,
            is_lse_base_on_e);
  };
  if (output_dtype == torch::headeronly::ScalarType::BFloat16) {
    launch_combine.operator()<__nv_bfloat16>();
  } else {
    launch_combine.operator()<__half>();
  }
  check_cuda_launch("direct DCP A2A");
}

}  // namespace

STABLE_TORCH_LIBRARY_FRAGMENT(_C, direct_dcp_a2a_ops) {
  direct_dcp_a2a_ops.def(
      "direct_dcp_a2a_lse_reduce("
      "Tensor partial_output, Tensor partial_lse, Tensor? seq_lens, "
      "Tensor? query_start_loc, Tensor peer_output_ptrs, "
      "Tensor peer_lse_ptrs, Tensor peer_signal_ptrs, Tensor! received_output, "
      "Tensor! received_lse, Tensor! received_signal, Tensor! epoch, "
      "Tensor! combined_output, int world_size, int rank, "
      "int max_num_tokens, bool is_lse_base_on_e) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, direct_dcp_a2a_ops) {
  direct_dcp_a2a_ops.impl("direct_dcp_a2a_lse_reduce",
                          TORCH_BOX(&direct_dcp_a2a_lse_reduce));
}
