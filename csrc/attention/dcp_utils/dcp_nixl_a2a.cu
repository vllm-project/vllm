// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <cuda/atomic>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <gpu/nixl_device.cuh>
#include <math_constants.h>
#include <torch/csrc/stable/library.h>
#include <torch/headeronly/core/ScalarType.h>

#include <algorithm>
#include <optional>
#include <type_traits>

#include "dcp_direct_common.cuh"

#if !UCT_CUDA_IPC_SUPPORTED && !UCT_RC_MLX5_GDA_SUPPORTED
  #error "NIXL DCP requires a UCX GPU device transport"
#endif

namespace {

using torch::headeronly::ScalarType;
using torch::stable::Tensor;
using vllm::direct_dcp::check_cuda_launch;
constexpr int kThreads = 256;
constexpr int kBatchPuts = 4;
constexpr int kCounterStride = 128;
constexpr uint64_t kTimeoutNs = 30000000000ULL;

__host__ __device__ size_t align_bytes(size_t value, size_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

__device__ uint64_t time_ns() {
  uint64_t value;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
  return value;
}

__device__ void fail(const char* phase, int peer, int channel, int status) {
  printf("NIXL DCP %s failed: peer=%d channel=%d status=%d\n", phase, peer,
         channel, status);
  asm volatile("trap;");
}

template <nixl_gpu_level_t Level>
__device__ void complete(nixl_status_t status, nixlGpuXferStatusH& request,
                         const char* phase, int peer, int channel) {
  if (status != NIXL_IN_PROG) {
    fail(phase, peer, channel, status);
  }
  const uint64_t start = time_ns();
  do {
    status = nixlGpuGetXferStatus<Level>(request);
    if (time_ns() - start > kTimeoutNs) {
      fail(phase, peer, channel, NIXL_IN_PROG);
    }
  } while (status == NIXL_IN_PROG);
  if (status != NIXL_SUCCESS) {
    fail(phase, peer, channel, status);
  }
}

// Cache optional peer mappings during the existing setup probe. The receive
// payload and its counters belong to the same registered memory element.
__global__ void peer_info_kernel(nixlMemViewH remote, int64_t* info,
                                 int64_t* peer_ptrs, int peers, int channels) {
  int item = blockIdx.x * blockDim.x + threadIdx.x;
  if (item >= peers * channels) {
    return;
  }
  int peer = item / channels;
  int channel = item % channels;
  auto view = static_cast<ucp_device_remote_mem_list_h>(remote);
  auto* result = info + item * 4;
  result[0] = -1;
  result[1] = result[2] = result[3] = 0;
  if (channel == 0) {
    peer_ptrs[peer] = 0;
  }
  if (ucp_device_check_params(view, peer) != UCS_OK || view->num_lanes == 0) {
    return;
  }
  if (channel == 0) {
    peer_ptrs[peer] = reinterpret_cast<int64_t>(nixlGetPtr(remote, peer));
  }
  unsigned lane = channel % view->num_lanes;
  unsigned cid = channel / view->num_lanes;
  auto* element = UCP_DEVICE_GET_ELEM(view, peer);
  auto* ep = element->tl[lane].ep;
  result[1] = reinterpret_cast<uintptr_t>(ep);
#if UCT_CUDA_IPC_SUPPORTED
  if (ep->uct_tl_id == UCT_DEVICE_TL_CUDA_IPC) {
    result[0] = 1;
    return;
  }
#endif
#if UCT_RC_MLX5_GDA_SUPPORTED
  if (ep->uct_tl_id == UCT_DEVICE_TL_RC_MLX5_GDA) {
    auto* gda = reinterpret_cast<uct_rc_gdaki_dev_ep_t*>(ep);
    result[0] = 2;
    result[2] = cid & gda->channel_mask;
    result[3] = gda->sq_wqe_num;
  }
#endif
}

__global__ void nixl_epoch_kernel(uint64_t* state, int active_channels) {
  int index = threadIdx.x;
  if (index <= active_channels) {
    if (state[index] == UINT64_MAX) {
      fail("epoch overflow", -1, index - 1, NIXL_ERR_INVALID_PARAM);
    }
    ++state[index];
  }
}

template <typename T>
__device__ float to_float(T value) {
  if constexpr (std::is_same_v<T, float>) {
    return value;
  } else if constexpr (std::is_same_v<T, __half>) {
    return __half2float(value);
  } else {
    return __bfloat162float(value);
  }
}

__device__ bool empty_row(int64_t token, const int32_t* seq_lens,
                          const int32_t* query_start, int num_seqs) {
  if (seq_lens == nullptr) {
    return false;
  }
  if (token >= query_start[num_seqs]) {
    return true;
  }
  int left = 0, right = num_seqs;
  while (left < right) {
    int mid = (left + right) / 2;
    if (query_start[mid + 1] <= token) {
      left = mid + 1;
    } else {
      right = mid;
    }
  }
  return seq_lens[left] == 0;
}

template <typename Lse>
__global__ void nixl_pack_put_signal_kernel(
    const uint16_t* output, const Lse* lse, const int32_t* seq_lens,
    const int32_t* query_start, int num_seqs, int64_t output_token_stride,
    int64_t output_head_stride, int64_t output_dim_stride,
    int64_t lse_token_stride, int64_t lse_head_stride, char* send,
    char* receive, const uint64_t* state, const int64_t* peer_ptrs,
    nixlMemViewH local, nixlMemViewH remote, int world_size, int rank,
    int ubatch, int num_tokens, int heads, int head_dim, int tile_tokens,
    int tile_stride, size_t peer_bytes, size_t receive_ubatch_bytes,
    int active_channels, bool vectorized) {
  const int peer = blockIdx.y;
  const int channel = blockIdx.x;
  // Remote-view indices omit our own rank. A null mapping uses NIXL puts;
  // otherwise every packing thread writes directly to the peer allocation.
  char* peer_receive =
      peer == rank ? receive
                   : reinterpret_cast<char*>(peer_ptrs[peer - (peer > rank)]);
  const int tiles = (num_tokens + tile_tokens - 1) / tile_tokens;
  const size_t slot = (state[0] & 1) * world_size * peer_bytes;
  const size_t rx_base = ubatch * receive_ubatch_bytes;
  const size_t tx_base = (ubatch * world_size + peer) * peer_bytes;
  const size_t counter_offset = rx_base + 2 * world_size * peer_bytes +
                                rank * kCounterStride +
                                channel * sizeof(uint64_t);
  extern __shared__ uint8_t row_empty[];
  __shared__ nixlGpuXferStatusH request;
  int batch_count = 0;

  for (int tile = channel; tile < tiles; tile += active_channels) {
    const int first = tile * tile_tokens;
    const int rows = min(tile_tokens, num_tokens - first);
    const size_t offset = static_cast<size_t>(tile) * tile_stride;
    const size_t dst_offset = rx_base + slot + rank * peer_bytes + offset;
    char* packed = peer_receive != nullptr ? peer_receive + dst_offset
                                           : send + tx_base + offset;
    const size_t output_bytes =
        static_cast<size_t>(rows) * heads * head_dim * 2;
    const size_t lse_offset = align_bytes(output_bytes, 16);
    const size_t bytes = lse_offset + static_cast<size_t>(rows) * heads * 4;
    for (int row = threadIdx.x; row < rows; row += blockDim.x) {
      row_empty[row] = empty_row(first + row, seq_lens, query_start, num_seqs);
    }
    __syncthreads();

    if (vectorized) {
      const int vectors_per_head = head_dim / 8;
      const int count = rows * heads * vectors_per_head;
      for (int item = threadIdx.x; item < count; item += blockDim.x) {
        int head_item = item / vectors_per_head;
        int row = head_item / heads;
        int head = head_item % heads;
        int dim = (item % vectors_per_head) * 8;
        auto* src = output + (first + row) * output_token_stride +
                    (peer * heads + head) * output_head_stride + dim;
        reinterpret_cast<uint4*>(packed)[item] =
            row_empty[row] ? make_uint4(0, 0, 0, 0)
                           : *reinterpret_cast<const uint4*>(src);
      }
    } else {
      const int count = rows * heads * head_dim;
      for (int item = threadIdx.x; item < count; item += blockDim.x) {
        int head_item = item / head_dim;
        int row = head_item / heads;
        int head = head_item % heads;
        int dim = item % head_dim;
        reinterpret_cast<uint16_t*>(packed)[item] =
            row_empty[row] ? 0
                           : output[(first + row) * output_token_stride +
                                    (peer * heads + head) * output_head_stride +
                                    dim * output_dim_stride];
      }
    }
    for (int item = threadIdx.x; item < rows * heads; item += blockDim.x) {
      int row = item / heads;
      int head = item % heads;
      reinterpret_cast<float*>(packed + lse_offset)[item] =
          row_empty[row]
              ? -CUDART_INF_F
              : to_float(lse[(first + row) * lse_token_stride +
                             (peer * heads + head) * lse_head_stride]);
    }
    // Publish every producer's stores before either the NIC reads staging or
    // the CTA leader signals a directly mapped receiver.
    __threadfence_system();
    __syncthreads();
    ++batch_count;
    const bool batch_end =
        batch_count == kBatchPuts || tile + active_channels >= tiles;
    if (peer_receive == nullptr && threadIdx.x < 32) {
      auto status = nixlPut<nixl_gpu_level_t::WARP>(
          {local, 0, tx_base + offset},
          {remote, static_cast<size_t>(peer - (peer > rank)), dst_offset},
          bytes, channel, batch_end ? 0 : nixl_gpu_flags::defer,
          batch_end ? &request : nullptr);
      if (status != NIXL_IN_PROG) {
        fail("put", peer, channel, status);
      }
      if (batch_end) {
        complete<nixl_gpu_level_t::WARP>(status, request, "payload", peer,
                                         channel);
      }
      // CUDA IPC copies are performed by this warp; their completion alone
      // does not publish every writer's stores to the receiving GPU.
      __threadfence_system();
    }
    __syncthreads();
    if (batch_end) {
      batch_count = 0;
    }
  }
  if (threadIdx.x == 0) {
    if (peer_receive != nullptr) {
      auto* counter =
          reinterpret_cast<uint64_t*>(peer_receive + counter_offset);
      cuda::atomic_ref<uint64_t, cuda::thread_scope_system>(*counter).store(
          state[channel + 1], cuda::memory_order_release);
    } else {
      __threadfence_system();
      auto status = nixlAtomicAdd<nixl_gpu_level_t::THREAD>(
          1,
          {remote, static_cast<size_t>(peer - (peer > rank)), counter_offset},
          channel, 0, &request);
      complete<nixl_gpu_level_t::THREAD>(status, request, "ready", peer,
                                         channel);
    }
  }
}

void check_workspace_tensor(const Tensor& tensor, const Tensor& reference,
                            ScalarType dtype) {
  STD_TORCH_CHECK(tensor.is_cuda() && tensor.is_contiguous() &&
                      tensor.scalar_type() == dtype &&
                      tensor.get_device_index() == reference.get_device_index(),
                  "NIXL DCP workspace must be contiguous on the input GPU");
}

void nixl_dcp_peer_info(int64_t remote_view, Tensor& info, Tensor& peer_ptrs) {
  check_workspace_tensor(info, info, ScalarType::Long);
  check_workspace_tensor(peer_ptrs, info, ScalarType::Long);
  STD_TORCH_CHECK(remote_view != 0 && info.dim() == 3 && info.size(0) > 0 &&
                      info.size(1) > 0 && info.size(1) <= 8 &&
                      info.size(2) == 4,
                  "invalid NIXL DCP peer-info dimensions");
  STD_TORCH_CHECK(peer_ptrs.dim() == 1 && peer_ptrs.size(0) == info.size(0),
                  "invalid NIXL DCP peer-pointer dimensions");
  const torch::stable::accelerator::DeviceGuard guard(info.get_device_index());
  int items = info.size(0) * info.size(1);
  peer_info_kernel<<<(items + 31) / 32, 32, 0, get_current_cuda_stream()>>>(
      reinterpret_cast<nixlMemViewH>(remote_view),
      info.mutable_data_ptr<int64_t>(), peer_ptrs.mutable_data_ptr<int64_t>(),
      info.size(0), info.size(1));
  check_cuda_launch("NIXL DCP peer info");
}

void nixl_dcp_a2a_dispatch(const Tensor& output, const Tensor& lse,
                           const std::optional<Tensor>& seq_lens,
                           const std::optional<Tensor>& query_start_loc,
                           Tensor& send, Tensor& receive, Tensor& state,
                           const Tensor& peer_ptrs, int64_t local_view,
                           int64_t remote_view, int64_t rank, int64_t ubatch,
                           int64_t tile_tokens) {
  STD_TORCH_CHECK(output.is_cuda() && lse.is_cuda() && output.dim() == 3 &&
                      lse.dim() == 2 &&
                      output.get_device_index() == lse.get_device_index(),
                  "NIXL DCP requires output [T,H,D] and LSE [T,H] on one GPU");
  STD_TORCH_CHECK(output.scalar_type() == ScalarType::Half ||
                      output.scalar_type() == ScalarType::BFloat16,
                  "NIXL DCP supports FP16 and BF16 output");
  auto lse_type = lse.scalar_type();
  STD_TORCH_CHECK(lse_type == ScalarType::Float ||
                      lse_type == ScalarType::Half ||
                      lse_type == ScalarType::BFloat16,
                  "NIXL DCP requires floating-point LSE");
  check_workspace_tensor(send, output, ScalarType::Byte);
  check_workspace_tensor(receive, output, ScalarType::Byte);
  check_workspace_tensor(state, output, ScalarType::Long);
  check_workspace_tensor(peer_ptrs, output, ScalarType::Long);
  STD_TORCH_CHECK(send.dim() == 3 && receive.dim() == 2 && state.dim() == 2,
                  "invalid NIXL DCP workspace dimensions");
  int world = send.size(1);
  int channels = state.size(1) - 1;
  int64_t tokens = output.size(0), dim = output.size(2);
  STD_TORCH_CHECK(
      world > 1 && rank >= 0 && rank < world && ubatch >= 0 &&
          ubatch < send.size(0) && receive.size(0) == send.size(0) &&
          state.size(0) == send.size(0) && channels >= 1 && channels <= 8 &&
          tile_tokens >= 1 && tile_tokens <= 16384 && tokens > 0 && dim > 0 &&
          output.size(1) > 0 && output.size(1) % world == 0 &&
          lse.size(0) == tokens && lse.size(1) == output.size(1) &&
          peer_ptrs.dim() == 1 && peer_ptrs.size(0) == world - 1 &&
          local_view != 0 && remote_view != 0,
      "invalid NIXL DCP geometry or views");
  int heads = output.size(1) / world;
  size_t tile_stride = align_bytes(
      align_bytes(tile_tokens * heads * dim * 2, 16) + tile_tokens * heads * 4,
      16);
  size_t peer_bytes = send.size(2);
  int tiles = (tokens + tile_tokens - 1) / tile_tokens;
  STD_TORCH_CHECK(peer_bytes % 128 == 0 && tiles * tile_stride <= peer_bytes &&
                      static_cast<size_t>(receive.size(1)) ==
                          2 * world * peer_bytes + world * kCounterStride,
                  "NIXL DCP input exceeds registered capacity");
  STD_TORCH_CHECK(seq_lens.has_value() == query_start_loc.has_value(),
                  "NIXL DCP needs both empty-shard mask tensors");
  const int32_t* seq = nullptr;
  const int32_t* starts = nullptr;
  int num_seqs = 0;
  if (seq_lens.has_value()) {
    check_workspace_tensor(*seq_lens, output, ScalarType::Int);
    check_workspace_tensor(*query_start_loc, output, ScalarType::Int);
    STD_TORCH_CHECK(seq_lens->dim() == 1 && seq_lens->size(0) > 0 &&
                        query_start_loc->dim() == 1 &&
                        query_start_loc->size(0) == seq_lens->size(0) + 1,
                    "invalid NIXL DCP empty-shard mask dimensions");
    seq = seq_lens->const_data_ptr<int32_t>();
    starts = query_start_loc->const_data_ptr<int32_t>();
    num_seqs = seq_lens->size(0);
  }
  const torch::stable::accelerator::DeviceGuard guard(
      output.get_device_index());
  auto stream = get_current_cuda_stream();
  auto* state_ptr =
      reinterpret_cast<uint64_t*>(state.mutable_data_ptr<int64_t>()) +
      ubatch * (channels + 1);
  int active_channels = std::min(channels, tiles);
  bool vectorized = dim % 8 == 0 && output.stride(2) == 1 &&
                    output.stride(1) % 8 == 0 && output.stride(0) % 8 == 0 &&
                    reinterpret_cast<uintptr_t>(output.data_ptr()) % 16 == 0;
  nixl_epoch_kernel<<<1, 32, 0, stream>>>(state_ptr, active_channels);
  check_cuda_launch("NIXL DCP begin");
  auto launch = [&]<typename Lse>() {
    nixl_pack_put_signal_kernel<Lse>
        <<<dim3(active_channels, world), kThreads, tile_tokens, stream>>>(
            reinterpret_cast<const uint16_t*>(output.data_ptr()),
            reinterpret_cast<const Lse*>(lse.data_ptr()), seq, starts, num_seqs,
            output.stride(0), output.stride(1), output.stride(2), lse.stride(0),
            lse.stride(1), static_cast<char*>(send.mutable_data_ptr()),
            static_cast<char*>(receive.mutable_data_ptr()), state_ptr,
            peer_ptrs.const_data_ptr<int64_t>(),
            reinterpret_cast<nixlMemViewH>(local_view),
            reinterpret_cast<nixlMemViewH>(remote_view), world, rank, ubatch,
            tokens, heads, dim, tile_tokens, tile_stride, peer_bytes,
            receive.size(1), active_channels, vectorized);
  };
  if (lse_type == ScalarType::Float) {
    launch.operator()<float>();
  } else if (lse_type == ScalarType::Half) {
    launch.operator()<__half>();
  } else {
    launch.operator()<__nv_bfloat16>();
  }
  check_cuda_launch("NIXL DCP pack/put/signal");
}

}  // namespace

STABLE_TORCH_LIBRARY_FRAGMENT(_C, nixl_dcp_ops) {
  nixl_dcp_ops.def(
      "nixl_dcp_peer_info(int remote_view, Tensor! info, Tensor! peer_ptrs) -> "
      "()");
  nixl_dcp_ops.def(
      "nixl_dcp_a2a_dispatch(Tensor partial_output, Tensor partial_lse, "
      "Tensor? seq_lens, Tensor? query_start_loc, Tensor! send, Tensor! "
      "receive, "
      "Tensor! state, Tensor peer_ptrs, int local_view, int remote_view, "
      "int rank, int ubatch, "
      "int tile_tokens) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, nixl_dcp_ops) {
  nixl_dcp_ops.impl("nixl_dcp_peer_info", TORCH_BOX(&nixl_dcp_peer_info));
  nixl_dcp_ops.impl("nixl_dcp_a2a_dispatch", TORCH_BOX(&nixl_dcp_a2a_dispatch));
}
