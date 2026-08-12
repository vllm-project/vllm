// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Direct symmetric-memory DCP query gather into the consumer-final buffer.

#include <torch/csrc/stable/library.h>
#include <torch/headeronly/core/ScalarType.h>

#include <cstdio>

#include "dcp_direct_common.cuh"

namespace {

using vllm::direct_dcp::check_cuda_launch;
using vllm::direct_dcp::increment_epoch_kernel;
using vllm::direct_dcp::multimem_store_16;
using vllm::direct_dcp::multimem_store_release_system;
using vllm::direct_dcp::wait_for_epoch;

// Multicast each rank's head slice directly into every consumer's final query
// buffer. Reuse is ordered by the downstream DCP output synchronization.
__global__ void direct_dcp_q_gather_multimem_kernel(
    const uint4* local_query, uint4* mc_final_query, uint32_t* mc_signal,
    const uint32_t* received_signal, int64_t* epoch_ptr, uint32_t* completion,
    int64_t world_size, int64_t rank, int64_t num_tokens,
    int64_t bytes_per_token, int64_t query_token_stride_bytes,
    int64_t destination_token_stride_bytes) {
  // The common one-token decode case uses one block. Fold the epoch update
  // into that publication kernel to avoid an otherwise separate tiny launch.
  if (gridDim.x == 1 && threadIdx.x == 0) {
    epoch_ptr[0] += 1;
  }
  __syncthreads();
  uint32_t epoch = static_cast<uint32_t>(epoch_ptr[0]);
  int64_t signal_slot = static_cast<int64_t>(epoch & 1u);

  int64_t items_per_token = bytes_per_token / sizeof(uint4);
  int64_t source_token_stride = query_token_stride_bytes / sizeof(uint4);
  int64_t destination_token_stride =
      destination_token_stride_bytes / sizeof(uint4);
  int64_t destination_head_offset = rank * items_per_token;
  for (int64_t token_idx = blockIdx.x; token_idx < num_tokens;
       token_idx += gridDim.x) {
    for (int64_t token_item = threadIdx.x; token_item < items_per_token;
         token_item += blockDim.x) {
      multimem_store_16(
          mc_final_query + token_idx * destination_token_stride +
              destination_head_offset + token_item,
          local_query[token_idx * source_token_stride + token_item]);
    }
  }

  // Publish all multicast writes before incrementing completion.
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x != 0) {
    return;
  }

  if (gridDim.x > 1) {
    uint32_t completed = atomicAdd(completion, 1u);
    if (completed + 1u != gridDim.x) {
      return;
    }
    atomicExch(completion, 0u);
  }

  multimem_store_release_system(mc_signal + signal_slot * world_size + rank,
                                epoch);

  for (int64_t source_rank = 0; source_rank < world_size; ++source_rank) {
    int64_t signal_item = signal_slot * world_size + source_rank;
    if (!wait_for_epoch(received_signal + signal_item, epoch)) {
      printf("direct DCP q-gather multimem timeout source=%lld epoch=%u\n",
             static_cast<long long>(source_rank), epoch);
      asm volatile("trap;");
    }
  }
}

void direct_dcp_q_gather(const torch::stable::Tensor& local_query,
                         torch::stable::Tensor& final_query,
                         torch::stable::Tensor& received_signal,
                         torch::stable::Tensor& completion,
                         torch::stable::Tensor& epoch, int64_t world_size,
                         int64_t rank, int64_t max_num_tokens,
                         int64_t padded_num_heads, int64_t query_mc_ptr,
                         int64_t signal_mc_ptr) {
  using torch::headeronly::ScalarType;

  STD_TORCH_CHECK(local_query.is_cuda(), "local query must be a CUDA tensor");
  ScalarType dtype = local_query.scalar_type();
  STD_TORCH_CHECK(local_query.dim() == 3,
                  "local query must have shape [T,H,D]");
  STD_TORCH_CHECK(world_size > 1, "world_size must be greater than 1");
  STD_TORCH_CHECK(rank >= 0 && rank < world_size, "invalid rank");

  int64_t num_tokens = local_query.size(0);
  int64_t heads_per_rank = local_query.size(1);
  int64_t head_dim = local_query.size(2);
  int64_t gathered_num_heads = world_size * heads_per_rank;
  int64_t element_size = local_query.element_size();
  STD_TORCH_CHECK(num_tokens > 0 && num_tokens <= max_num_tokens,
                  "token count exceeds symmetric q-gather buffer capacity");
  STD_TORCH_CHECK(heads_per_rank > 0 && head_dim > 0,
                  "query head dimensions must be positive");
  STD_TORCH_CHECK(padded_num_heads >= gathered_num_heads,
                  "padded query heads must cover all gathered heads");
  STD_TORCH_CHECK(local_query.stride(2) == 1 &&
                      local_query.stride(1) == head_dim &&
                      local_query.stride(0) >= heads_per_rank * head_dim,
                  "local query must have packed heads");

  STD_TORCH_CHECK(
      final_query.is_cuda() && final_query.scalar_type() == dtype &&
          final_query.is_contiguous() && final_query.dim() == 3 &&
          final_query.size(0) == num_tokens &&
          final_query.size(1) == gathered_num_heads &&
          final_query.size(2) == head_dim,
      "final query must be contiguous with shape [T,world_size*H,D]");
  STD_TORCH_CHECK(
      received_signal.is_cuda() && received_signal.is_contiguous() &&
          received_signal.scalar_type() == ScalarType::Int &&
          received_signal.dim() == 2 && received_signal.size(0) == 2 &&
          received_signal.size(1) == world_size,
      "received signal has the wrong symmetric buffer layout");
  STD_TORCH_CHECK(completion.is_cuda() && completion.is_contiguous() &&
                      completion.scalar_type() == ScalarType::Int &&
                      completion.numel() == 1,
                  "completion counter must be one CUDA int32 tensor");
  STD_TORCH_CHECK(epoch.is_cuda() && epoch.is_contiguous() &&
                      epoch.scalar_type() == ScalarType::Long &&
                      epoch.numel() == 1,
                  "epoch must be a one-element CUDA int64 tensor");
  int64_t device_index = local_query.get_device_index();
  STD_TORCH_CHECK(
      final_query.get_device_index() == device_index &&
          received_signal.get_device_index() == device_index &&
          completion.get_device_index() == device_index &&
          epoch.get_device_index() == device_index,
      "direct DCP q-gather tensors must be on the same CUDA device");

  int64_t query_token_stride_bytes = local_query.stride(0) * element_size;
  int64_t bytes_per_token = heads_per_rank * head_dim * element_size;
  int64_t gathered_token_stride_bytes =
      gathered_num_heads * head_dim * element_size;
  bool vectorized =
      reinterpret_cast<uintptr_t>(local_query.data_ptr()) % alignof(uint4) ==
          0 &&
      reinterpret_cast<uintptr_t>(final_query.data_ptr()) % alignof(uint4) ==
          0 &&
      query_token_stride_bytes % sizeof(uint4) == 0 &&
      bytes_per_token % sizeof(uint4) == 0;
  STD_TORCH_CHECK(
      vectorized,
      "direct DCP q-gather requires 16-byte-aligned pointers and strides");
  STD_TORCH_CHECK(query_mc_ptr != 0 && signal_mc_ptr != 0,
                  "direct DCP q-gather requires multicast pointers");

  const torch::stable::accelerator::DeviceGuard device_guard(device_index);
  cudaStream_t stream = get_current_cuda_stream();
  constexpr int kThreads = 256;
  int64_t blocks = num_tokens < world_size ? num_tokens : world_size;
  if (blocks > 1) {
    increment_epoch_kernel<<<1, 1, 0, stream>>>(
        epoch.mutable_data_ptr<int64_t>());
    check_cuda_launch("direct DCP q-gather epoch");
  }

  direct_dcp_q_gather_multimem_kernel<<<blocks, kThreads, 0, stream>>>(
      reinterpret_cast<const uint4*>(local_query.data_ptr()),
      reinterpret_cast<uint4*>(static_cast<uintptr_t>(query_mc_ptr)),
      reinterpret_cast<uint32_t*>(static_cast<uintptr_t>(signal_mc_ptr)),
      reinterpret_cast<const uint32_t*>(
          received_signal.const_data_ptr<int32_t>()),
      epoch.mutable_data_ptr<int64_t>(),
      reinterpret_cast<uint32_t*>(completion.mutable_data_ptr<int32_t>()),
      world_size, rank, num_tokens, bytes_per_token, query_token_stride_bytes,
      gathered_token_stride_bytes);
  check_cuda_launch("direct DCP q-gather");
}

}  // namespace

STABLE_TORCH_LIBRARY_FRAGMENT(_C, direct_dcp_q_gather_ops) {
  direct_dcp_q_gather_ops.def(
      "direct_dcp_q_gather("
      "Tensor local_query, Tensor! final_query, Tensor! received_signal, "
      "Tensor! completion, Tensor! epoch, "
      "int world_size, int rank, int max_num_tokens, "
      "int padded_num_heads, int query_mc_ptr, int signal_mc_ptr) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, direct_dcp_q_gather_ops) {
  direct_dcp_q_gather_ops.impl("direct_dcp_q_gather",
                               TORCH_BOX(&direct_dcp_q_gather));
}
