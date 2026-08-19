// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Direct symmetric-memory DCP KV gather.

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

constexpr int kThreads = 256;
// KV chunks need many blocks in flight to saturate the fabric.
constexpr int64_t kMaxMulticastBlocks = 128;

// Multicast each rank's valid local rows directly into compact, request-major
// kv_c and k_pe planes. dst_rows maps every padded local input row to its final
// output row, or -1 for padding. Destination rows are disjoint across source
// ranks, so all ranks can publish concurrently without atomics.
__global__ void direct_dcp_kv_gather_multimem_kernel(
    const uint4* local_kv, const int32_t* dst_rows, uint4* mc_kv,
    uint32_t* mc_signal, const uint32_t* received_signal,
    const int64_t* epoch_ptr, uint32_t* completion, int64_t world_size,
    int64_t rank, int64_t num_tokens, int64_t items_per_row,
    int64_t kv_c_items_per_row, int64_t output_tokens,
    int64_t max_gathered_tokens, int64_t buffer_slot,
    int64_t slot_stride_items) {
  uint32_t epoch = static_cast<uint32_t>(epoch_ptr[0]);
  mc_kv += buffer_slot * slot_stride_items;

  int64_t item_count = num_tokens * items_per_row;
  int64_t item_stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t item =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       item < item_count; item += item_stride) {
    int64_t src_row = item / items_per_row;
    int32_t dst_row = dst_rows[src_row];
    if (dst_row < 0) {
      continue;
    }
    if (dst_row >= output_tokens) {
      printf(
          "direct DCP final-layout kv-gather destination out of bounds "
          "source=%lld dst=%d output_tokens=%lld\n",
          static_cast<long long>(rank), dst_row,
          static_cast<long long>(output_tokens));
      asm volatile("trap;");
    }
    int64_t row_item = item - src_row * items_per_row;
    int64_t dst_item;
    if (row_item < kv_c_items_per_row) {
      dst_item = static_cast<int64_t>(dst_row) * kv_c_items_per_row + row_item;
    } else {
      int64_t k_pe_items_per_row = items_per_row - kv_c_items_per_row;
      dst_item = max_gathered_tokens * kv_c_items_per_row +
                 static_cast<int64_t>(dst_row) * k_pe_items_per_row + row_item -
                 kv_c_items_per_row;
    }
    multimem_store_16(mc_kv + dst_item, local_kv[item]);
  }

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
      printf("direct DCP final-layout kv-gather timeout source=%lld epoch=%u\n",
             static_cast<long long>(source_rank), epoch);
      asm volatile("trap;");
    }
  }
}

void direct_dcp_kv_gather(const torch::stable::Tensor& local_kv,
                          const torch::stable::Tensor& dst_rows,
                          torch::stable::Tensor& received_kv,
                          torch::stable::Tensor& received_signal,
                          torch::stable::Tensor& completion,
                          torch::stable::Tensor& epoch, int64_t output_tokens,
                          int64_t plane_split_dim, int64_t buffer_slot,
                          int64_t world_size, int64_t rank,
                          int64_t max_gathered_tokens, int64_t kv_mc_ptr,
                          int64_t signal_mc_ptr) {
  using torch::headeronly::ScalarType;

  STD_TORCH_CHECK(local_kv.is_cuda(), "local kv must be a CUDA tensor");
  ScalarType dtype = local_kv.scalar_type();
  STD_TORCH_CHECK(dtype == ScalarType::Half || dtype == ScalarType::BFloat16 ||
                      dtype == ScalarType::Float8_e4m3fn,
                  "direct DCP final-layout kv-gather only supports FP16, "
                  "BF16, and FP8");
  STD_TORCH_CHECK(local_kv.dim() == 2 && local_kv.is_contiguous(),
                  "local kv must be a contiguous [T,D] tensor");
  STD_TORCH_CHECK(dst_rows.is_cuda() && dst_rows.is_contiguous() &&
                      dst_rows.scalar_type() == ScalarType::Int &&
                      dst_rows.dim() == 1 &&
                      dst_rows.numel() == local_kv.size(0),
                  "final-layout destination rows must be CUDA int32 [T]");
  STD_TORCH_CHECK(world_size > 1, "world_size must be greater than 1");
  STD_TORCH_CHECK(rank >= 0 && rank < world_size, "invalid rank");
  STD_TORCH_CHECK(buffer_slot == 0 || buffer_slot == 1,
                  "final-layout buffer slot must be 0 or 1");
  STD_TORCH_CHECK(output_tokens > 0 && output_tokens <= max_gathered_tokens,
                  "final-layout output exceeds symmetric buffer capacity");

  int64_t num_tokens = local_kv.size(0);
  int64_t token_dim = local_kv.size(1);
  int64_t element_size = local_kv.element_size();
  STD_TORCH_CHECK(num_tokens > 0 && token_dim > 0,
                  "local kv dimensions must be positive");
  STD_TORCH_CHECK(plane_split_dim > 0 && plane_split_dim < token_dim,
                  "final-layout plane split must be within the token row");
  STD_TORCH_CHECK(num_tokens * world_size <= max_gathered_tokens,
                  "padded gathered kv exceeds symmetric buffer capacity");

  STD_TORCH_CHECK(received_kv.is_cuda() && received_kv.scalar_type() == dtype &&
                      received_kv.is_contiguous() && received_kv.dim() == 3 &&
                      received_kv.size(0) == 2 &&
                      received_kv.size(1) == max_gathered_tokens &&
                      received_kv.size(2) == token_dim,
                  "received kv has the wrong symmetric buffer layout");
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

  int64_t device_index = local_kv.get_device_index();
  STD_TORCH_CHECK(dst_rows.get_device_index() == device_index &&
                      received_kv.get_device_index() == device_index &&
                      received_signal.get_device_index() == device_index &&
                      completion.get_device_index() == device_index &&
                      epoch.get_device_index() == device_index,
                  "direct DCP final-layout tensors must share a CUDA device");

  int64_t row_bytes = token_dim * element_size;
  int64_t kv_c_row_bytes = plane_split_dim * element_size;
  int64_t k_pe_row_bytes = row_bytes - kv_c_row_bytes;
  int64_t slot_stride_bytes = max_gathered_tokens * row_bytes;
  bool vectorized =
      reinterpret_cast<uintptr_t>(local_kv.data_ptr()) % alignof(uint4) == 0 &&
      reinterpret_cast<uintptr_t>(received_kv.data_ptr()) % alignof(uint4) ==
          0 &&
      row_bytes % sizeof(uint4) == 0 && kv_c_row_bytes % sizeof(uint4) == 0 &&
      k_pe_row_bytes % sizeof(uint4) == 0 &&
      slot_stride_bytes % sizeof(uint4) == 0;
  STD_TORCH_CHECK(vectorized,
                  "direct DCP final-layout kv-gather requires 16-byte-aligned "
                  "planes, rows, and pointers");
  STD_TORCH_CHECK(kv_mc_ptr != 0 && signal_mc_ptr != 0,
                  "direct DCP final-layout kv-gather requires multicast "
                  "pointers");

  const torch::stable::accelerator::DeviceGuard device_guard(device_index);
  cudaStream_t stream = get_current_cuda_stream();
  increment_epoch_kernel<<<1, 1, 0, stream>>>(
      epoch.mutable_data_ptr<int64_t>());
  check_cuda_launch("direct DCP final-layout kv-gather");

  int64_t items_per_row = row_bytes / sizeof(uint4);
  int64_t kv_c_items_per_row = kv_c_row_bytes / sizeof(uint4);
  int64_t item_count = num_tokens * items_per_row;
  int64_t blocks = (item_count + kThreads - 1) / kThreads;
  blocks = blocks < kMaxMulticastBlocks ? blocks : kMaxMulticastBlocks;
  direct_dcp_kv_gather_multimem_kernel<<<blocks, kThreads, 0, stream>>>(
      reinterpret_cast<const uint4*>(local_kv.data_ptr()),
      dst_rows.const_data_ptr<int32_t>(),
      reinterpret_cast<uint4*>(static_cast<uintptr_t>(kv_mc_ptr)),
      reinterpret_cast<uint32_t*>(static_cast<uintptr_t>(signal_mc_ptr)),
      reinterpret_cast<const uint32_t*>(
          received_signal.const_data_ptr<int32_t>()),
      epoch.const_data_ptr<int64_t>(),
      reinterpret_cast<uint32_t*>(completion.mutable_data_ptr<int32_t>()),
      world_size, rank, num_tokens, items_per_row, kv_c_items_per_row,
      output_tokens, max_gathered_tokens, buffer_slot,
      slot_stride_bytes / sizeof(uint4));
  check_cuda_launch("direct DCP final-layout kv-gather");
}

}  // namespace

STABLE_TORCH_LIBRARY_FRAGMENT(_C, direct_dcp_kv_gather_ops) {
  direct_dcp_kv_gather_ops.def(
      "direct_dcp_kv_gather("
      "Tensor local_kv, Tensor dst_rows, Tensor! received_kv, "
      "Tensor! received_signal, Tensor! completion, Tensor! epoch, "
      "int output_tokens, int plane_split_dim, int buffer_slot, "
      "int world_size, int rank, int max_gathered_tokens, "
      "int kv_mc_ptr, int signal_mc_ptr) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, direct_dcp_kv_gather_ops) {
  direct_dcp_kv_gather_ops.impl("direct_dcp_kv_gather",
                                TORCH_BOX(&direct_dcp_kv_gather));
}
