// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <torch/csrc/stable/library.h>
#include <torch/headeronly/core/ScalarType.h>

#include <cstdio>

#include "dcp_direct_common.cuh"

namespace {

using vllm::direct_dcp::check_cuda_launch;
using vllm::direct_dcp::get_peer_ptr;
using vllm::direct_dcp::increment_epoch_kernel;
using vllm::direct_dcp::multimem_store_16;
using vllm::direct_dcp::multimem_store_release_system;
using vllm::direct_dcp::store_release_system;
using vllm::direct_dcp::wait_for_epoch;

constexpr int kThreads = 256;
// The payload is MB-scale (a whole context-chunk KV slice), so unlike the
// q-gather the copy needs many blocks in flight to reach fabric bandwidth.
constexpr int64_t kMaxMulticastBlocks = 128;
constexpr int64_t kMaxBlocksPerPeer = 16;
constexpr int64_t kMaxCopyBlocks = 1024;

// Unicast fallback: blocks are striped across destination ranks
// (peer = blockIdx.x % world_size) so every peer gets gridDim.x / world_size
// blocks copying this rank's contiguous KV slice into the peer's staging slot
// at the rank offset. The last completing block publishes this rank's epoch to
// every peer, then waits until every source rank's slice has arrived locally.
template <typename copy_t>
__global__ void direct_dcp_kv_gather_kernel(
    const copy_t* local_kv, const int64_t* peer_kv_ptrs,
    const int64_t* peer_signal_ptrs, const uint32_t* received_signal,
    const int64_t* epoch_ptr, uint32_t* completion, int64_t world_size,
    int64_t rank, int64_t slice_bytes, int64_t slot_stride_bytes) {
  uint32_t epoch = static_cast<uint32_t>(epoch_ptr[0]);
  int64_t buffer_slot = static_cast<int64_t>(epoch & 1u);
  int64_t item_count = slice_bytes / sizeof(copy_t);
  int64_t destination_rank = static_cast<int64_t>(blockIdx.x) % world_size;
  copy_t* peer_kv = get_peer_ptr<copy_t>(peer_kv_ptrs, destination_rank);
  peer_kv += buffer_slot * slot_stride_bytes / sizeof(copy_t);
  peer_kv += rank * item_count;

  int64_t blocks_per_peer = static_cast<int64_t>(gridDim.x) / world_size;
  int64_t block_in_peer = static_cast<int64_t>(blockIdx.x) / world_size;
  int64_t item_stride = blocks_per_peer * blockDim.x;
  for (int64_t item = block_in_peer * blockDim.x + threadIdx.x;
       item < item_count; item += item_stride) {
    peer_kv[item] = local_kv[item];
  }

  // Every lane must publish its peer writes before thread 0 contributes this
  // block to the completion count.
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

  for (int64_t peer = 0; peer < world_size; ++peer) {
    uint32_t* peer_signal = get_peer_ptr<uint32_t>(peer_signal_ptrs, peer);
    store_release_system(peer_signal + buffer_slot * world_size + rank, epoch);
  }

  for (int64_t source_rank = 0; source_rank < world_size; ++source_rank) {
    int64_t signal_item = buffer_slot * world_size + source_rank;
    if (!wait_for_epoch(received_signal + signal_item, epoch)) {
      printf("direct DCP kv-gather timeout source=%lld epoch=%u\n",
             static_cast<long long>(source_rank), epoch);
      asm volatile("trap;");
    }
  }
}

// NVLS multicast variant: each rank stores its contiguous KV slice once
// through the multicast pointer at the rank offset and the fabric replicates
// it into every rank's staging slot — one store stream instead of one per
// peer. The completion signal is a single release-ordered multicast store of
// the epoch to this rank's signal word on all replicas.
__global__ void direct_dcp_kv_gather_multimem_kernel(
    const uint4* local_kv, uint4* mc_kv, uint32_t* mc_signal,
    const uint32_t* received_signal, const int64_t* epoch_ptr,
    uint32_t* completion, int64_t world_size, int64_t rank, int64_t slice_bytes,
    int64_t slot_stride_bytes) {
  uint32_t epoch = static_cast<uint32_t>(epoch_ptr[0]);
  int64_t buffer_slot = static_cast<int64_t>(epoch & 1u);
  int64_t item_count = slice_bytes / sizeof(uint4);
  mc_kv += buffer_slot * slot_stride_bytes / sizeof(uint4);
  mc_kv += rank * item_count;

  int64_t item_stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t item =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       item < item_count; item += item_stride) {
    multimem_store_16(mc_kv + item, local_kv[item]);
  }

  // Every lane must publish its multicast writes before thread 0 contributes
  // this block to the completion count.
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
      printf("direct DCP kv-gather multimem timeout source=%lld epoch=%u\n",
             static_cast<long long>(source_rank), epoch);
      asm volatile("trap;");
    }
  }
}

// The stream-ordered exchange above completes its acquire waits before this
// kernel materializes the selected slot into the caller's workspace slice.
template <typename copy_t>
__global__ void materialize_kv_gather_kernel(const copy_t* received_kv,
                                             const int64_t* epoch_ptr,
                                             copy_t* gathered_kv,
                                             int64_t gathered_item_count,
                                             int64_t slot_stride_items) {
  uint32_t epoch = static_cast<uint32_t>(epoch_ptr[0]);
  int64_t buffer_slot = static_cast<int64_t>(epoch & 1u);
  received_kv += buffer_slot * slot_stride_items;
  int64_t item = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t item_stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (; item < gathered_item_count; item += item_stride) {
    gathered_kv[item] = received_kv[item];
  }
}

void direct_dcp_kv_gather(
    const torch::stable::Tensor& local_kv,
    const torch::stable::Tensor& peer_kv_ptrs,
    const torch::stable::Tensor& peer_signal_ptrs,
    torch::stable::Tensor& received_kv, torch::stable::Tensor& received_signal,
    torch::stable::Tensor& completion, torch::stable::Tensor& epoch,
    torch::stable::Tensor& gathered_kv, int64_t world_size, int64_t rank,
    int64_t max_gathered_tokens, int64_t kv_mc_ptr, int64_t signal_mc_ptr) {
  using torch::headeronly::ScalarType;

  STD_TORCH_CHECK(local_kv.is_cuda(), "local kv must be a CUDA tensor");
  ScalarType dtype = local_kv.scalar_type();
  STD_TORCH_CHECK(dtype == ScalarType::Half || dtype == ScalarType::BFloat16 ||
                      dtype == ScalarType::Float8_e4m3fn,
                  "direct DCP kv-gather only supports FP16, BF16, and FP8");
  STD_TORCH_CHECK(local_kv.dim() == 2 && local_kv.is_contiguous(),
                  "local kv must be a contiguous [T,D] tensor");
  STD_TORCH_CHECK(world_size > 1, "world_size must be greater than 1");
  STD_TORCH_CHECK(rank >= 0 && rank < world_size, "invalid rank");

  int64_t num_tokens = local_kv.size(0);
  int64_t token_dim = local_kv.size(1);
  int64_t element_size = local_kv.element_size();
  STD_TORCH_CHECK(num_tokens > 0 && token_dim > 0,
                  "local kv dimensions must be positive");
  STD_TORCH_CHECK(num_tokens * world_size <= max_gathered_tokens,
                  "gathered kv exceeds symmetric kv-gather buffer capacity");

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
  STD_TORCH_CHECK(gathered_kv.is_cuda() && gathered_kv.scalar_type() == dtype &&
                      gathered_kv.is_contiguous() && gathered_kv.dim() == 2 &&
                      gathered_kv.size(0) == num_tokens * world_size &&
                      gathered_kv.size(1) == token_dim,
                  "gathered kv must be contiguous with shape [world_size*T,D]");
  STD_TORCH_CHECK(
      peer_kv_ptrs.is_cuda() && peer_kv_ptrs.is_contiguous() &&
          peer_kv_ptrs.scalar_type() == ScalarType::Long &&
          peer_kv_ptrs.numel() == world_size && peer_signal_ptrs.is_cuda() &&
          peer_signal_ptrs.is_contiguous() &&
          peer_signal_ptrs.scalar_type() == ScalarType::Long &&
          peer_signal_ptrs.numel() == world_size,
      "peer pointer tables must be CUDA int64 tensors of world_size entries");

  int64_t device_index = local_kv.get_device_index();
  STD_TORCH_CHECK(
      received_kv.get_device_index() == device_index &&
          received_signal.get_device_index() == device_index &&
          completion.get_device_index() == device_index &&
          epoch.get_device_index() == device_index &&
          gathered_kv.get_device_index() == device_index &&
          peer_kv_ptrs.get_device_index() == device_index &&
          peer_signal_ptrs.get_device_index() == device_index,
      "direct DCP kv-gather tensors must be on the same CUDA device");

  int64_t slice_bytes = num_tokens * token_dim * element_size;
  int64_t slot_stride_bytes = max_gathered_tokens * token_dim * element_size;
  bool vectorized =
      reinterpret_cast<uintptr_t>(local_kv.data_ptr()) % alignof(uint4) == 0 &&
      reinterpret_cast<uintptr_t>(received_kv.data_ptr()) % alignof(uint4) ==
          0 &&
      reinterpret_cast<uintptr_t>(gathered_kv.data_ptr()) % alignof(uint4) ==
          0 &&
      slice_bytes % sizeof(uint4) == 0 &&
      slot_stride_bytes % sizeof(uint4) == 0;

  const torch::stable::accelerator::DeviceGuard device_guard(device_index);
  cudaStream_t stream = get_current_cuda_stream();
  increment_epoch_kernel<<<1, 1, 0, stream>>>(
      epoch.mutable_data_ptr<int64_t>());
  check_cuda_launch("direct DCP kv-gather");

  bool use_multimem = vectorized && kv_mc_ptr != 0 && signal_mc_ptr != 0;
  auto launch = [&]<typename copy_t>() {
    int64_t item_count = slice_bytes / sizeof(copy_t);
    if (use_multimem) {
      int64_t blocks = (item_count + kThreads - 1) / kThreads;
      blocks = blocks < kMaxMulticastBlocks ? blocks : kMaxMulticastBlocks;
      direct_dcp_kv_gather_multimem_kernel<<<blocks, kThreads, 0, stream>>>(
          reinterpret_cast<const uint4*>(local_kv.data_ptr()),
          reinterpret_cast<uint4*>(static_cast<uintptr_t>(kv_mc_ptr)),
          reinterpret_cast<uint32_t*>(static_cast<uintptr_t>(signal_mc_ptr)),
          reinterpret_cast<const uint32_t*>(
              received_signal.const_data_ptr<int32_t>()),
          epoch.const_data_ptr<int64_t>(),
          reinterpret_cast<uint32_t*>(completion.mutable_data_ptr<int32_t>()),
          world_size, rank, slice_bytes, slot_stride_bytes);
    } else {
      int64_t blocks_per_peer = (item_count + kThreads - 1) / kThreads;
      blocks_per_peer = blocks_per_peer < kMaxBlocksPerPeer ? blocks_per_peer
                                                            : kMaxBlocksPerPeer;
      direct_dcp_kv_gather_kernel<copy_t>
          <<<world_size * blocks_per_peer, kThreads, 0, stream>>>(
              reinterpret_cast<const copy_t*>(local_kv.data_ptr()),
              peer_kv_ptrs.const_data_ptr<int64_t>(),
              peer_signal_ptrs.const_data_ptr<int64_t>(),
              reinterpret_cast<const uint32_t*>(
                  received_signal.const_data_ptr<int32_t>()),
              epoch.const_data_ptr<int64_t>(),
              reinterpret_cast<uint32_t*>(
                  completion.mutable_data_ptr<int32_t>()),
              world_size, rank, slice_bytes, slot_stride_bytes);
    }
    check_cuda_launch("direct DCP kv-gather");

    int64_t gathered_item_count = world_size * item_count;
    int64_t copy_blocks = (gathered_item_count + kThreads - 1) / kThreads;
    copy_blocks = copy_blocks < kMaxCopyBlocks ? copy_blocks : kMaxCopyBlocks;
    materialize_kv_gather_kernel<copy_t><<<copy_blocks, kThreads, 0, stream>>>(
        reinterpret_cast<const copy_t*>(received_kv.data_ptr()),
        epoch.const_data_ptr<int64_t>(),
        reinterpret_cast<copy_t*>(gathered_kv.mutable_data_ptr()),
        gathered_item_count, slot_stride_bytes / sizeof(copy_t));
  };
  if (vectorized) {
    launch.operator()<uint4>();
  } else {
    launch.operator()<uint8_t>();
  }
  check_cuda_launch("direct DCP kv-gather");
}

}  // namespace

STABLE_TORCH_LIBRARY_FRAGMENT(_C, direct_dcp_kv_gather_ops) {
  direct_dcp_kv_gather_ops.def(
      "direct_dcp_kv_gather("
      "Tensor local_kv, Tensor peer_kv_ptrs, Tensor peer_signal_ptrs, "
      "Tensor! received_kv, Tensor! received_signal, Tensor! completion, "
      "Tensor! epoch, Tensor! gathered_kv, "
      "int world_size, int rank, int max_gathered_tokens, "
      "int kv_mc_ptr, int signal_mc_ptr) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, direct_dcp_kv_gather_ops) {
  direct_dcp_kv_gather_ops.impl("direct_dcp_kv_gather",
                                TORCH_BOX(&direct_dcp_kv_gather));
}
