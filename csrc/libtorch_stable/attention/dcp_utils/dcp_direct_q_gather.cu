// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Direct symmetric-memory DCP query gather.

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

// Multicast each rank's head slice and completion epoch to every replica.
__global__ void direct_dcp_q_gather_multimem_kernel(
    const uint4* local_query, uint4* mc_query, uint32_t* mc_signal,
    const uint32_t* received_signal, const int64_t* epoch_ptr,
    uint32_t* completion, int64_t world_size, int64_t rank, int64_t num_tokens,
    int64_t bytes_per_token, int64_t query_token_stride_bytes,
    int64_t destination_token_stride_bytes, int64_t slot_stride_bytes) {
  uint32_t epoch = static_cast<uint32_t>(epoch_ptr[0]);
  int64_t buffer_slot = static_cast<int64_t>(epoch & 1u);
  mc_query += buffer_slot * slot_stride_bytes / sizeof(uint4);

  int64_t items_per_token = bytes_per_token / sizeof(uint4);
  int64_t source_token_stride = query_token_stride_bytes / sizeof(uint4);
  int64_t destination_token_stride =
      destination_token_stride_bytes / sizeof(uint4);
  int64_t destination_head_offset = rank * items_per_token;
  int64_t item_count = num_tokens * items_per_token;
  int64_t item_stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t item =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       item < item_count; item += item_stride) {
    int64_t token_idx = item / items_per_token;
    int64_t token_item = item - token_idx * items_per_token;
    multimem_store_16(
        mc_query + token_idx * destination_token_stride +
            destination_head_offset + token_item,
        local_query[token_idx * source_token_stride + token_item]);
  }

  // Publish all multicast writes before incrementing completion.
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x != 0) {
    return;
  }

  uint32_t completed = atomicAdd(completion + buffer_slot, 1u);
  if (completed + 1u != gridDim.x) {
    return;
  }
  atomicExch(completion + buffer_slot, 0u);

  multimem_store_release_system(mc_signal + buffer_slot * world_size + rank,
                                epoch);

  for (int64_t source_rank = 0; source_rank < world_size; ++source_rank) {
    int64_t signal_item = buffer_slot * world_size + source_rank;
    if (!wait_for_epoch(received_signal + signal_item, epoch)) {
      printf("direct DCP q-gather multimem timeout source=%lld epoch=%u\n",
             static_cast<long long>(source_rank), epoch);
      asm volatile("trap;");
    }
  }
}

// Materialize the acquired slot at the stable downstream address.
template <typename copy_t>
__global__ void materialize_q_gather_kernel(const copy_t* received_query,
                                            const int64_t* epoch_ptr,
                                            copy_t* gathered_query,
                                            int64_t gathered_item_count,
                                            int64_t slot_stride_items) {
  uint32_t epoch = static_cast<uint32_t>(epoch_ptr[0]);
  int64_t buffer_slot = static_cast<int64_t>(epoch & 1u);
  received_query += buffer_slot * slot_stride_items;
  int64_t item = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t item_stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (; item < gathered_item_count; item += item_stride) {
    gathered_query[item] = received_query[item];
  }
}

void direct_dcp_q_gather(const torch::stable::Tensor& local_query,
                         torch::stable::Tensor& received_query,
                         torch::stable::Tensor& received_signal,
                         torch::stable::Tensor& completion,
                         torch::stable::Tensor& epoch,
                         torch::stable::Tensor& gathered_query,
                         int64_t world_size, int64_t rank,
                         int64_t max_num_tokens, int64_t padded_num_heads,
                         int64_t query_mc_ptr, int64_t signal_mc_ptr) {
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
      received_query.is_cuda() && received_query.scalar_type() == dtype &&
          received_query.is_contiguous() && received_query.dim() == 4 &&
          received_query.size(0) == 2 &&
          received_query.size(1) == max_num_tokens &&
          received_query.size(2) == padded_num_heads &&
          received_query.size(3) == head_dim,
      "received query has the wrong symmetric buffer layout");
  STD_TORCH_CHECK(
      received_signal.is_cuda() && received_signal.is_contiguous() &&
          received_signal.scalar_type() == ScalarType::Int &&
          received_signal.dim() == 2 && received_signal.size(0) == 2 &&
          received_signal.size(1) == world_size,
      "received signal has the wrong symmetric buffer layout");
  STD_TORCH_CHECK(completion.is_cuda() && completion.is_contiguous() &&
                      completion.scalar_type() == ScalarType::Int &&
                      completion.numel() == 2,
                  "completion counter must be a two-element CUDA int32 tensor");
  STD_TORCH_CHECK(epoch.is_cuda() && epoch.is_contiguous() &&
                      epoch.scalar_type() == ScalarType::Long &&
                      epoch.numel() == 1,
                  "epoch must be a one-element CUDA int64 tensor");
  STD_TORCH_CHECK(
      gathered_query.is_cuda() && gathered_query.scalar_type() == dtype &&
          gathered_query.is_contiguous() && gathered_query.dim() == 3 &&
          gathered_query.size(0) == num_tokens &&
          gathered_query.size(1) == gathered_num_heads &&
          gathered_query.size(2) == head_dim,
      "gathered query must be contiguous with shape [T,world_size*H,D]");
  int64_t device_index = local_query.get_device_index();
  STD_TORCH_CHECK(
      received_query.get_device_index() == device_index &&
          received_signal.get_device_index() == device_index &&
          completion.get_device_index() == device_index &&
          epoch.get_device_index() == device_index &&
          gathered_query.get_device_index() == device_index,
      "direct DCP q-gather tensors must be on the same CUDA device");

  int64_t query_token_stride_bytes = local_query.stride(0) * element_size;
  int64_t bytes_per_token = heads_per_rank * head_dim * element_size;
  int64_t gathered_token_stride_bytes =
      gathered_num_heads * head_dim * element_size;
  int64_t slot_stride_bytes =
      max_num_tokens * padded_num_heads * head_dim * element_size;
  bool vectorized =
      reinterpret_cast<uintptr_t>(local_query.data_ptr()) % alignof(uint4) ==
          0 &&
      reinterpret_cast<uintptr_t>(received_query.data_ptr()) % alignof(uint4) ==
          0 &&
      reinterpret_cast<uintptr_t>(gathered_query.data_ptr()) % alignof(uint4) ==
          0 &&
      query_token_stride_bytes % sizeof(uint4) == 0 &&
      bytes_per_token % sizeof(uint4) == 0 &&
      slot_stride_bytes % sizeof(uint4) == 0;
  STD_TORCH_CHECK(
      vectorized,
      "direct DCP q-gather requires 16-byte-aligned pointers and strides");
  STD_TORCH_CHECK(query_mc_ptr != 0 && signal_mc_ptr != 0,
                  "direct DCP q-gather requires multicast pointers");

  const torch::stable::accelerator::DeviceGuard device_guard(device_index);
  cudaStream_t stream = get_current_cuda_stream();
  constexpr int kThreads = 256;
  increment_epoch_kernel<<<1, 1, 0, stream>>>(
      epoch.mutable_data_ptr<int64_t>());
  check_cuda_launch("direct DCP q-gather");

  direct_dcp_q_gather_multimem_kernel<<<world_size, kThreads, 0, stream>>>(
      reinterpret_cast<const uint4*>(local_query.data_ptr()),
      reinterpret_cast<uint4*>(static_cast<uintptr_t>(query_mc_ptr)),
      reinterpret_cast<uint32_t*>(static_cast<uintptr_t>(signal_mc_ptr)),
      reinterpret_cast<const uint32_t*>(
          received_signal.const_data_ptr<int32_t>()),
      epoch.const_data_ptr<int64_t>(),
      reinterpret_cast<uint32_t*>(completion.mutable_data_ptr<int32_t>()),
      world_size, rank, num_tokens, bytes_per_token, query_token_stride_bytes,
      gathered_token_stride_bytes, slot_stride_bytes);
  check_cuda_launch("direct DCP q-gather");

  int64_t gathered_item_count =
      num_tokens * gathered_token_stride_bytes / sizeof(uint4);
  int64_t copy_blocks = (gathered_item_count + kThreads - 1) / kThreads;
  constexpr int64_t kMaxCopyBlocks = 1024;
  copy_blocks = copy_blocks < kMaxCopyBlocks ? copy_blocks : kMaxCopyBlocks;
  materialize_q_gather_kernel<uint4><<<copy_blocks, kThreads, 0, stream>>>(
      reinterpret_cast<const uint4*>(received_query.data_ptr()),
      epoch.const_data_ptr<int64_t>(),
      reinterpret_cast<uint4*>(gathered_query.mutable_data_ptr()),
      gathered_item_count, slot_stride_bytes / sizeof(uint4));
  check_cuda_launch("direct DCP q-gather");
}

}  // namespace

STABLE_TORCH_LIBRARY_FRAGMENT(_C, direct_dcp_q_gather_ops) {
  direct_dcp_q_gather_ops.def(
      "direct_dcp_q_gather("
      "Tensor local_query, Tensor! received_query, Tensor! received_signal, "
      "Tensor! completion, "
      "Tensor! epoch, Tensor! gathered_query, "
      "int world_size, int rank, int max_num_tokens, "
      "int padded_num_heads, int query_mc_ptr, int signal_mc_ptr) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, direct_dcp_q_gather_ops) {
  direct_dcp_q_gather_ops.impl("direct_dcp_q_gather",
                               TORCH_BOX(&direct_dcp_q_gather));
}
