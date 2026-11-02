// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Direct peer-to-peer DCP A2A LSE reduction for ROCm (gfx942 / gfx950).
//
// Same result as dcp_direct_a2a_lse_reduce.cu, but pull-based: each rank
// publishes its whole partial output into its own staging buffer, then every
// rank reads its heads from all ranks and combines them in the same pass, so
// the payload crosses xGMI once. Staging is uncached IPC memory, so no cache
// can hold a stale copy and no per-access cache maintenance is needed. That
// matters on gfx94x/95x, where a system-scope release writes back and an
// acquire invalidates a whole XCD's L2: issued per thread or per spin they cost
// more than the exchange, so each call publishes completion once, from the last
// publishing block.

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <torch/csrc/stable/library.h>
#include <torch/headeronly/core/ScalarType.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <optional>
#include <string>
#include <type_traits>

#include "../../torch_utils.h"

namespace {

constexpr uint64_t kSpinLimit = 100000000;
constexpr float kLog2E = 1.4426950408889634f;

template <typename T>
__device__ __forceinline__ T* peer_ptr(const int64_t* peer_ptrs, int64_t peer) {
  return reinterpret_cast<T*>(static_cast<uintptr_t>(peer_ptrs[peer]));
}

__device__ __forceinline__ void store_release_system(uint32_t* ptr,
                                                     uint32_t value) {
  __hip_atomic_store(ptr, value, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__ __forceinline__ uint32_t load_relaxed_system(const uint32_t* ptr) {
  return __hip_atomic_load(ptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

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
    return __float2bfloat16(value);
  }
}

constexpr int kMaxWorldSize = 8;

// Staging per rank and ubatch: output [2 slots][max_tokens][total_heads]
// [head_dim], lse [2][max_tokens][total_heads], and signal [2][world] that
// peers write their flags into. The epoch's low bit picks the slot. A rank
// republishes a slot only after its combine of the intervening call has seen
// every peer's next flag, which each peer raises after finishing the combine
// that read the slot, so a slot is never overwritten while being read.
//
// Grid (token, head chunk); the last block publishes the epoch.
template <typename lse_t>
__global__ void publish_signal_kernel(
    const uint4* partial_output, const lse_t* partial_lse,
    const int32_t* seq_lens, const int32_t* query_start_loc,
    uint4* staged_output, float* staged_lse, const int64_t* peer_signal_ptrs,
    int64_t* epoch_ptr, uint32_t* completion, int64_t world_size, int64_t rank,
    int64_t max_num_tokens, int64_t num_seqs, int64_t total_heads,
    int64_t head_dim, int64_t output_token_stride, int64_t lse_token_stride) {
  int64_t token_idx = static_cast<int64_t>(blockIdx.x);
  int64_t heads_per_chunk = total_heads / gridDim.y;
  int64_t first_head = static_cast<int64_t>(blockIdx.y) * heads_per_chunk;

  __shared__ bool empty_kv;
  __shared__ bool is_last_block;
  if (threadIdx.x == 0) {
    empty_kv = false;
    if (seq_lens != nullptr) {
      int64_t seq_idx = find_sequence(query_start_loc, token_idx, num_seqs);
      empty_kv = seq_lens[seq_idx] == 0;
    }
  }
  __syncthreads();

  uint32_t epoch = static_cast<uint32_t>(epoch_ptr[0]) + 1u;
  int64_t parity = static_cast<int64_t>(epoch & 1u);
  int64_t row =
      (parity * max_num_tokens + token_idx) * total_heads + first_head;

  if (!empty_kv) {
    int64_t vectors = heads_per_chunk * head_dim / 8;
    int64_t source_vector =
        (token_idx * output_token_stride + first_head * head_dim) / 8;
    int64_t destination_vector = row * head_dim / 8;
    for (int64_t vector_idx = threadIdx.x; vector_idx < vectors;
         vector_idx += blockDim.x) {
      staged_output[destination_vector + vector_idx] =
          partial_output[source_vector + vector_idx];
    }
  }
  int64_t source_lse = token_idx * lse_token_stride + first_head;
  for (int64_t head_idx = threadIdx.x; head_idx < heads_per_chunk;
       head_idx += blockDim.x) {
    staged_lse[row + head_idx] =
        empty_kv ? -INFINITY : to_float(partial_lse[source_lse + head_idx]);
  }

  __builtin_amdgcn_s_waitcnt(0);
  __syncthreads();
  if (threadIdx.x == 0) {
    is_last_block = atomicAdd(completion, 1u) == gridDim.x * gridDim.y - 1;
  }
  __syncthreads();
  if (!is_last_block) {
    return;
  }

  __threadfence_system();
  for (int64_t peer = threadIdx.x; peer < world_size; peer += blockDim.x) {
    uint32_t* signal = peer_ptr<uint32_t>(peer_signal_ptrs, peer);
    store_release_system(signal + parity * world_size + rank, epoch);
  }
  if (threadIdx.x == 0) {
    *completion = 0u;
    epoch_ptr[0] = static_cast<int64_t>(epoch);
  }
}

// Persistent pull combine, laid out like AITER's allgather_lastdim: one block
// per CU, each split into one thread group per rank so every wave reads one
// contiguous row from a single peer. A block checks the ready flags once, then
// walks (token, local head) items; per item each group scales its rank's row
// by that rank's LSE weight and the groups are summed through LDS.
constexpr int kPullThreads = 512;
constexpr int kPullBlocks = 256;

template <typename scalar_t>
__global__ void __launch_bounds__(kPullThreads)
    pull_combine_kernel(const int64_t* peer_output_ptrs,
                        const int64_t* peer_lse_ptrs,
                        const uint32_t* received_signal,
                        const int64_t* epoch_ptr, scalar_t* combined_output,
                        int64_t world_size, int64_t rank, int64_t num_items,
                        int64_t max_num_tokens, int64_t heads_per_rank,
                        int64_t head_dim, bool is_lse_base_on_e) {
  __shared__ float lse[kMaxWorldSize];
  __shared__ float partial[kPullThreads * 8];

  uint32_t epoch = static_cast<uint32_t>(epoch_ptr[0]);
  int64_t parity = static_cast<int64_t>(epoch & 1u);
  int64_t threads_per_rank = kPullThreads / world_size;
  int64_t source = threadIdx.x / threads_per_rank;
  int64_t lane = threadIdx.x - source * threads_per_rank;
  int64_t vectors = head_dim / 8;
  int64_t total_heads = heads_per_rank * world_size;

  if (threadIdx.x < world_size) {
    const uint32_t* signal =
        received_signal + parity * world_size + threadIdx.x;
    uint64_t spins = 0;
    while (load_relaxed_system(signal) != epoch) {
      if (++spins == kSpinLimit) {
        printf("direct DCP A2A timeout source=%u epoch=%u\n", threadIdx.x,
               epoch);
        __builtin_trap();
      }
    }
  }
  // Staging is uncached: ordering the reads after the flags is enough.
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
  __syncthreads();

  const uint4* rows = peer_ptr<const uint4>(peer_output_ptrs, source);
  const float* lses = peer_ptr<const float>(peer_lse_ptrs, source);

  for (int64_t item = blockIdx.x; item < num_items; item += gridDim.x) {
    int64_t token_idx = item / heads_per_rank;
    int64_t head_idx = item - token_idx * heads_per_rank;
    int64_t row = (parity * max_num_tokens + token_idx) * total_heads +
                  rank * heads_per_rank + head_idx;
    if (lane == 0) {
      float value = lses[row];
      if (isnan(value) || value == INFINITY) {
        value = -INFINITY;
      }
      lse[source] = is_lse_base_on_e ? value * kLog2E : value;
    }
    __syncthreads();

    float lse_max = -INFINITY;
    for (int64_t s = 0; s < world_size; ++s) {
      lse_max = fmaxf(lse_max, lse[s]);
    }
    if (lse_max == -INFINITY) {
      lse_max = 0.0f;
    }
    float lse_sum = 0.0f;
    for (int64_t s = 0; s < world_size; ++s) {
      lse_sum += exp2f(lse[s] - lse_max);
    }
    float weight =
        lse_sum > 0.0f ? exp2f(lse[source] - lse_max) / lse_sum : 0.0f;

    if (lane < vectors) {
      float* mine = partial + threadIdx.x * 8;
      // Empty shards leave stale payloads, so never read zero-weight sources.
      if (weight != 0.0f) {
        uint4 packed = rows[row * vectors + lane];
        const scalar_t* values = reinterpret_cast<const scalar_t*>(&packed);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          mine[i] = to_float(values[i]) * weight;
        }
      } else {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          mine[i] = 0.0f;
        }
      }
    }
    __syncthreads();

    if (source == 0 && lane < vectors) {
      float accumulator[8] = {};
      for (int64_t s = 0; s < world_size; ++s) {
        const float* theirs = partial + (s * threads_per_rank + lane) * 8;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          accumulator[i] += theirs[i];
        }
      }
      uint4 result;
      scalar_t* out_values = reinterpret_cast<scalar_t*>(&result);
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        out_values[i] = from_float<scalar_t>(accumulator[i]);
      }
      reinterpret_cast<uint4*>(combined_output)[item * vectors + lane] = result;
    }
    __syncthreads();
  }
}

void check_launch(const char* operation) {
  hipError_t error = hipGetLastError();
  STD_TORCH_CHECK(error == hipSuccess,
                  std::string(operation) +
                      " kernel launch failed: " + hipGetErrorString(error));
}

// partial_output [T, world_size * H, D] (fp16/bf16, D % 8 == 0) and partial_lse
// [T, world_size * H] (fp32/fp16/bf16, unit head stride) hold this rank's
// partial attention for every head; combined_output [T, H, D] receives the
// LSE-weighted merge for this rank's H heads, with T <= max_num_tokens.
// seq_lens / query_start_loc (int32) mark tokens whose local KV shard is empty.
// peer_*_ptrs (int64 [world_size]) hold one ubatch slice of every rank's
// staging buffer, including this rank's own at index `rank`; received_*_ptr are
// that own slice. epoch (int64 [1]) and completion (int32 [1]) persist across
// calls.
void direct_dcp_a2a_lse_reduce_rocm(
    const torch::stable::Tensor& partial_output,
    const torch::stable::Tensor& partial_lse,
    const std::optional<torch::stable::Tensor>& seq_lens,
    const std::optional<torch::stable::Tensor>& query_start_loc,
    const torch::stable::Tensor& peer_output_ptrs,
    const torch::stable::Tensor& peer_lse_ptrs,
    const torch::stable::Tensor& peer_signal_ptrs, int64_t received_output_ptr,
    int64_t received_lse_ptr, int64_t received_signal_ptr,
    torch::stable::Tensor& epoch, torch::stable::Tensor& completion,
    torch::stable::Tensor& combined_output, int64_t world_size, int64_t rank,
    int64_t max_num_tokens, bool is_lse_base_on_e) {
  STD_TORCH_CHECK(partial_output.is_cuda() && partial_lse.is_cuda(),
                  "partial output and LSE must be device tensors");
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
  STD_TORCH_CHECK(world_size > 1 && world_size <= kMaxWorldSize,
                  "world_size must be in [2, 8]");
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
                  "token count exceeds the staging buffer capacity");
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
                  "combined output must match the contiguous device input");
  STD_TORCH_CHECK(combined_output.size(0) == num_tokens &&
                      combined_output.size(1) == heads_per_rank &&
                      combined_output.size(2) == head_dim,
                  "combined output has the wrong shape");
  STD_TORCH_CHECK(
      peer_output_ptrs.scalar_type() == torch::headeronly::ScalarType::Long &&
          peer_lse_ptrs.scalar_type() == torch::headeronly::ScalarType::Long &&
          peer_signal_ptrs.scalar_type() == torch::headeronly::ScalarType::Long,
      "peer pointer tables must be int64");
  STD_TORCH_CHECK(
      epoch.scalar_type() == torch::headeronly::ScalarType::Long &&
          completion.scalar_type() == torch::headeronly::ScalarType::Int,
      "epoch must be int64 and completion int32");
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
        "seq_lens must be a contiguous 1-D int32 device tensor");
    STD_TORCH_CHECK(
        query_start_loc->is_cuda() && query_start_loc->dim() == 1 &&
            query_start_loc->stride(0) == 1 &&
            query_start_loc->scalar_type() ==
                torch::headeronly::ScalarType::Int,
        "query_start_loc must be a contiguous 1-D int32 device tensor");
    num_seqs = seq_lens->size(0);
    STD_TORCH_CHECK(num_seqs > 0 && query_start_loc->size(0) == num_seqs + 1,
                    "query_start_loc must contain one boundary per sequence");
    seq_lens_ptr = seq_lens->const_data_ptr<int32_t>();
    query_start_loc_ptr = query_start_loc->const_data_ptr<int32_t>();
  }

  const torch::stable::accelerator::DeviceGuard device_guard(
      partial_output.get_device_index());
  hipStream_t stream =
      get_current_cuda_stream(partial_output.get_device_index());
  constexpr int kExchangeThreads = 256;

  STD_TORCH_CHECK(head_dim / 8 <= kPullThreads / world_size,
                  "pull combine needs head_dim / 8 <= 512 / world_size");
  dim3 publish_grid(static_cast<unsigned>(num_tokens),
                    static_cast<unsigned>(world_size));
  auto launch_publish = [&]<typename lse_t>() {
    publish_signal_kernel<lse_t><<<publish_grid, kExchangeThreads, 0, stream>>>(
        reinterpret_cast<const uint4*>(partial_output.data_ptr()),
        reinterpret_cast<const lse_t*>(partial_lse.data_ptr()), seq_lens_ptr,
        query_start_loc_ptr,
        reinterpret_cast<uint4*>(static_cast<uintptr_t>(received_output_ptr)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(received_lse_ptr)),
        peer_signal_ptrs.const_data_ptr<int64_t>(),
        epoch.mutable_data_ptr<int64_t>(),
        reinterpret_cast<uint32_t*>(completion.mutable_data_ptr<int32_t>()),
        world_size, rank, max_num_tokens, num_seqs, total_heads, head_dim,
        output_token_stride, lse_token_stride);
  };
  if (lse_dtype == torch::headeronly::ScalarType::Float) {
    launch_publish.operator()<float>();
  } else if (lse_dtype == torch::headeronly::ScalarType::BFloat16) {
    launch_publish.operator()<__hip_bfloat16>();
  } else {
    launch_publish.operator()<__half>();
  }
  check_launch("direct DCP A2A publish");

  int64_t num_items = num_tokens * heads_per_rank;
  unsigned pull_blocks =
      static_cast<unsigned>(std::min<int64_t>(num_items, kPullBlocks));
  auto launch_pull = [&]<typename scalar_t>() {
    pull_combine_kernel<scalar_t><<<pull_blocks, kPullThreads, 0, stream>>>(
        peer_output_ptrs.const_data_ptr<int64_t>(),
        peer_lse_ptrs.const_data_ptr<int64_t>(),
        reinterpret_cast<const uint32_t*>(
            static_cast<uintptr_t>(received_signal_ptr)),
        epoch.const_data_ptr<int64_t>(),
        reinterpret_cast<scalar_t*>(combined_output.mutable_data_ptr()),
        world_size, rank, num_items, max_num_tokens, heads_per_rank, head_dim,
        is_lse_base_on_e);
  };
  if (output_dtype == torch::headeronly::ScalarType::BFloat16) {
    launch_pull.operator()<__hip_bfloat16>();
  } else {
    launch_pull.operator()<__half>();
  }
  check_launch("direct DCP A2A pull combine");
}

}  // namespace

STABLE_TORCH_LIBRARY_FRAGMENT(_C, direct_dcp_a2a_rocm_ops) {
  direct_dcp_a2a_rocm_ops.def(
      "direct_dcp_a2a_lse_reduce_rocm("
      "Tensor partial_output, Tensor partial_lse, Tensor? seq_lens, "
      "Tensor? query_start_loc, Tensor peer_output_ptrs, "
      "Tensor peer_lse_ptrs, Tensor peer_signal_ptrs, int received_output_ptr, "
      "int received_lse_ptr, int received_signal_ptr, Tensor! epoch, "
      "Tensor! completion, Tensor! combined_output, int world_size, int rank, "
      "int max_num_tokens, bool is_lse_base_on_e) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, direct_dcp_a2a_rocm_ops) {
  direct_dcp_a2a_rocm_ops.impl("direct_dcp_a2a_lse_reduce_rocm",
                               TORCH_BOX(&direct_dcp_a2a_lse_reduce_rocm));
}
