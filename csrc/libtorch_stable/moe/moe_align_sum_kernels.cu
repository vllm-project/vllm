#include <array>
#include <limits>
#include <cub/cub.cuh>

#include <cuda_runtime.h>
#include <torch/csrc/stable/macros.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include "../../cuda_compat.h"
#include "libtorch_stable/core/math.hpp"
#include "libtorch_stable/dispatch_utils.h"
#ifndef USE_ROCM
  #include "libtorch_stable/moe/permute_unpermute_kernels/moe_permute_unpermute_kernel.h"
#endif
#include "libtorch_stable/quantization/vectorization.cuh"
#include "libtorch_stable/torch_utils.h"

#define CEILDIV(x, y) (((x) + (y) - 1) / (y))

namespace vllm {
namespace moe {
namespace batched_moe_align_block_size {

// Note num_threads needs to be 1024 for BlockScan Reduction in the kernel.
static constexpr int32_t num_threads = 1024;
static constexpr int32_t num_blocks = 1;
__global__ void batched_moe_align_block_size_kernel(
    int32_t const num_batches, int32_t const max_tokens_per_batch,
    int32_t const block_size, int32_t const* __restrict__ batch_num_tokens,
    int32_t* __restrict__ sorted_ids, int32_t* __restrict__ block_ids,
    int32_t* __restrict__ num_tokens_post_pad) {
  // TODO(varun): This is a naive implementation. Could be optimized.

  size_t const batch_id = threadIdx.x;
  size_t const stride = blockDim.x * gridDim.x;
  int32_t const num_blocks_per_batch =
      CEILDIV(max_tokens_per_batch, block_size);
  int32_t const sorted_ids_size =
      num_blocks_per_batch * num_batches * block_size;
  int32_t const block_ids_size = sorted_ids_size / block_size;
  int32_t const SENTINEL =
      num_batches * max_tokens_per_batch;  // To denote invalid entries.
  // Initialize sorted_ids
  for (size_t i = threadIdx.x; i < sorted_ids_size; i += stride) {
    sorted_ids[i] = SENTINEL;
  }
  // Initialize expert_ids with -1
  for (size_t i = threadIdx.x; i < block_ids_size; i += stride) {
    block_ids[i] = -1;
  }

  int32_t b_num_tokens = 0;
  if (batch_id < num_batches) {
    b_num_tokens = batch_num_tokens[batch_id];
  }
  int32_t const ceil_b_num_tokens =
      CEILDIV(b_num_tokens, block_size) * block_size;

  // Compute prefix sum over token counts per expert
  using BlockScan = cub::BlockScan<int32_t, 1024>;
  __shared__ typename BlockScan::TempStorage temp_storage;
  int cumsum_val;
  BlockScan(temp_storage).ExclusiveSum(ceil_b_num_tokens, cumsum_val);
  __syncthreads();

  bool const is_last_batch = batch_id == (num_batches - 1);
  if (is_last_batch) {
    *num_tokens_post_pad = cumsum_val + ceil_b_num_tokens;
  }

  if (batch_id < num_batches) {
    int32_t const batch_offset = batch_id * max_tokens_per_batch;
    for (size_t i = 0; i < b_num_tokens; ++i) {
      sorted_ids[cumsum_val + i] = batch_offset + i;
    }

    int32_t const block_start = cumsum_val / block_size;
    int32_t const num_blocks = ceil_b_num_tokens / block_size;
    for (size_t i = 0; i < num_blocks; ++i) {
      block_ids[block_start + i] = batch_id;
    }
  }
}
}  // namespace batched_moe_align_block_size

template <typename scalar_t>
__device__ void _moe_align_block_size(
    const scalar_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids, int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ total_tokens_post_pad,
    int32_t* __restrict__ expert_map, int32_t num_experts,
    int32_t padded_num_experts, int32_t experts_per_warp, int32_t block_size,
    size_t numel, int32_t* __restrict__ cumsum, int32_t max_num_tokens_padded,
    int32_t max_num_m_blocks, int32_t model_offset, int32_t inactive_expert_id,
    int32_t topk_num, int32_t* token_mask, bool has_expert_map) {
  extern __shared__ int32_t shared_counts[];

  // Compute input buffer offsets. Typically these will all be 0, except when
  // using Multi LoRA.
  int sorted_token_ids_offset = max_num_tokens_padded * model_offset;
  int expert_ids_offset = max_num_m_blocks * model_offset;
  int cumsum_offset = (num_experts + 1) * model_offset;

  // Use separate threadblocks to fill sorted_token_ids.
  // This is safe since the current kernel does not use sorted_token_ids.
  if (blockIdx.x % 2) {
    // Initialize sorted_token_ids with numel
    for (size_t it = threadIdx.x; it < max_num_tokens_padded;
         it += blockDim.x) {
      sorted_token_ids[sorted_token_ids_offset + it] = numel;
    }
    return;
  }

  const int warp_id = threadIdx.x / WARP_SIZE;
  const int my_expert_start = warp_id * experts_per_warp;

  for (int i = 0; i < experts_per_warp; ++i) {
    if (my_expert_start + i < padded_num_experts) {
      shared_counts[warp_id * experts_per_warp + i] = 0;
    }
  }

  __syncthreads();

  const size_t tid = threadIdx.x;
  const size_t stride = blockDim.x;

  for (size_t i = tid; i < numel; i += stride) {
    int expert_id = topk_ids[i];
    if (expert_id >= num_experts) {
      continue;
    }
    if (has_expert_map) {
      expert_id = expert_map[expert_id];
      // filter invalid experts
      if (expert_id == -1) continue;
    }
    int warp_idx = expert_id / experts_per_warp;
    int expert_offset = expert_id % experts_per_warp;
    int mask = token_mask == nullptr ? 1 : token_mask[i / topk_num];
    atomicAdd(&shared_counts[warp_idx * experts_per_warp + expert_offset],
              mask);
  }

  __syncthreads();

  // Compute prefix sum over token counts per expert
  using BlockScan = cub::BlockScan<int32_t, 1024>;
  __shared__ typename BlockScan::TempStorage temp_storage;

  int expert_count = 0;
  int expert_id = threadIdx.x;
  if (expert_id < num_experts) {
    int warp_idx = expert_id / experts_per_warp;
    int expert_offset = expert_id % experts_per_warp;
    expert_count = shared_counts[warp_idx * experts_per_warp + expert_offset];
    expert_count = CEILDIV(expert_count, block_size) * block_size;
  }

  int cumsum_val;
  BlockScan(temp_storage).ExclusiveSum(expert_count, cumsum_val);
  if (expert_id <= num_experts) {
    cumsum[cumsum_offset + expert_id] = cumsum_val;
  }

  if (expert_id == num_experts) {
    total_tokens_post_pad[model_offset] = cumsum_val;
  }

  __syncthreads();

  if (threadIdx.x < num_experts) {
    for (int i = cumsum[cumsum_offset + threadIdx.x];
         i < cumsum[cumsum_offset + threadIdx.x + 1]; i += block_size) {
      expert_ids[expert_ids_offset + i / block_size] = threadIdx.x;
    }
  }

  // Fill remaining expert_ids with -1
  const size_t fill_start_idx =
      cumsum[cumsum_offset + num_experts] / block_size + threadIdx.x;
  for (size_t i = fill_start_idx; i < max_num_m_blocks; i += blockDim.x) {
    expert_ids[expert_ids_offset + i] = inactive_expert_id;
  }
}

template <typename scalar_t, int32_t fill_threads>
__device__ void _moe_align_block_size_small_batch_expert(
    const scalar_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids, int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ total_tokens_post_pad,
    int32_t* __restrict__ expert_map, int32_t num_experts, int32_t block_size,
    size_t numel, int32_t max_num_tokens_padded, int32_t max_num_m_blocks,
    int32_t inactive_expert_id, int32_t model_offset, int32_t topk_num,
    int32_t* token_mask, bool has_expert_map) {
  // Compute input buffer offsets. Typically these will all be 0, except when
  // using Multi LoRA.
  int sorted_token_ids_offset = max_num_tokens_padded * model_offset;
  int expert_ids_offset = max_num_m_blocks * model_offset;

  // Use an additional group of threads to fill sorted_token_ids.
  // Since the current kernel will use sorted_token_ids afterward,
  // we fill sorted_token_ids within the same threadblock to make
  // synchronization easier.
  if (threadIdx.x < fill_threads) {
    // Initialize sorted_token_ids with numel
    for (size_t it = threadIdx.x; it < max_num_tokens_padded;
         it += fill_threads) {
      sorted_token_ids[sorted_token_ids_offset + it] = numel;
    }
    // Three __syncthreads() corresponding to the other threads
    __syncthreads();
    __syncthreads();
    __syncthreads();
    return;
  }

  const size_t tid = threadIdx.x - fill_threads;
  const size_t stride = blockDim.x - fill_threads;

  extern __shared__ int32_t shared_mem[];
  int32_t* cumsum = shared_mem;
  int32_t* tokens_cnts = (int32_t*)(shared_mem + num_experts + 1);

  for (int i = 0; i < num_experts; ++i) {
    tokens_cnts[(tid + 1) * num_experts + i] = 0;
  }

  for (size_t i = tid; i < numel; i += stride) {
    int32_t expert_id = topk_ids[i];
    if (has_expert_map) {
      expert_id = expert_map[expert_id];
      // filter invalid expert
      if (expert_id == -1) continue;
    }
    int mask = token_mask == nullptr ? 1 : token_mask[i / topk_num];
    tokens_cnts[(tid + 1) * num_experts + expert_id] += mask;
  }

  __syncthreads();

  if (tid < num_experts) {
    tokens_cnts[tid] = 0;
    for (int i = 1; i <= stride; ++i) {
      tokens_cnts[i * num_experts + tid] +=
          tokens_cnts[(i - 1) * num_experts + tid];
    }
  }

  __syncthreads();

  if (tid == 0) {
    cumsum[0] = 0;
    for (int i = 1; i <= num_experts; ++i) {
      cumsum[i] =
          cumsum[i - 1] +
          CEILDIV(tokens_cnts[stride * num_experts + i - 1], block_size) *
              block_size;
    }
    total_tokens_post_pad[model_offset] =
        static_cast<int32_t>(cumsum[num_experts]);
  }

  __syncthreads();

  if (tid < num_experts) {
    for (int i = cumsum[tid]; i < cumsum[tid + 1]; i += block_size) {
      expert_ids[expert_ids_offset + i / block_size] = tid;
    }
  }

  // Fill remaining expert_ids with -1
  const size_t fill_start_idx = cumsum[num_experts] / block_size + tid;
  for (size_t i = fill_start_idx; i < max_num_m_blocks; i += stride) {
    expert_ids[expert_ids_offset + i] = inactive_expert_id;
  }

  for (size_t i = tid; i < numel; i += stride) {
    int32_t expert_id = topk_ids[i];
    if (has_expert_map) {
      expert_id = expert_map[expert_id];
      // filter invalid expert
      if (expert_id == -1) continue;
    }
    int32_t rank_post_pad =
        tokens_cnts[tid * num_experts + expert_id] + cumsum[expert_id];

    if (token_mask == nullptr || token_mask[i / topk_num]) {
      sorted_token_ids[sorted_token_ids_offset + rank_post_pad] = i;
      ++tokens_cnts[tid * num_experts + expert_id];
    }
  }
}

template <typename scalar_t>
__device__ void _count_and_sort_expert_tokens(
    const scalar_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids, int32_t* __restrict__ cumsum_buffer,
    int32_t* __restrict__ expert_map, size_t numel, int32_t num_experts,
    int32_t max_num_tokens_padded, int32_t* __restrict__ token_mask,
    int32_t model_offset, int32_t topk_num, bool has_expert_map) {
  const size_t tid = blockIdx.y * blockDim.x + threadIdx.x;
  const size_t stride = blockDim.x * gridDim.y;

  for (size_t i = tid; i < numel; i += stride) {
    int32_t expert_id = topk_ids[i];
    if (expert_id >= num_experts) {
      continue;
    }

    if (has_expert_map) {
      expert_id = expert_map[expert_id];
      // filter invalid experts
      if (expert_id == -1) continue;
    }

    if (token_mask == nullptr || token_mask[i / topk_num]) {
      int32_t rank_post_pad = atomicAdd(
          &cumsum_buffer[(model_offset * (num_experts + 1)) + expert_id], 1);
      sorted_token_ids[max_num_tokens_padded * model_offset + rank_post_pad] =
          i;
    }
  }
}

template <typename scalar_t>
__global__ void moe_align_block_size_kernel(
    const scalar_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids, int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ total_tokens_post_pad,
    int32_t* __restrict__ expert_map, int32_t num_experts,
    int32_t padded_num_experts, int32_t experts_per_warp, int32_t block_size,
    size_t numel, int32_t* __restrict__ cumsum, int32_t max_num_tokens_padded,
    int32_t topk_num, bool has_expert_map) {
  _moe_align_block_size(
      topk_ids, sorted_token_ids, expert_ids, total_tokens_post_pad, expert_map,
      num_experts, padded_num_experts, experts_per_warp, block_size, numel,
      cumsum, max_num_tokens_padded, CEILDIV(max_num_tokens_padded, block_size),
      0, -1, topk_num, nullptr, has_expert_map);
}

template <typename scalar_t>
__global__ void count_and_sort_expert_tokens_kernel(
    const scalar_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids, int32_t* __restrict__ cumsum_buffer,
    int32_t* __restrict__ expert_map, size_t numel, int32_t num_experts,
    int32_t max_num_tokens_padded, int32_t topk_num, bool has_expert_map) {
  _count_and_sort_expert_tokens(
      topk_ids, sorted_token_ids, cumsum_buffer, expert_map, numel, num_experts,
      max_num_tokens_padded, nullptr, 0, topk_num, has_expert_map);
}

// Decode-sized inputs fit in one block. Shared-memory counts and offsets build
// the padded expert ranges; comparing only earlier routes gives a stable rank.
// The O(num_routes^2 + num_experts) work is explicitly capped by max_routes.
template <typename scalar_t, int32_t max_routes>
__global__ void stable_small_route_align_kernel(
    const scalar_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids, int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ total_tokens_post_pad,
    const int32_t* __restrict__ expert_map, int32_t num_experts,
    int32_t block_size, int32_t numel, int32_t max_num_tokens_padded,
    int32_t max_num_m_blocks, bool has_expert_map) {
  __shared__ int32_t expert_counts[1024];
  __shared__ int32_t expert_offsets[1024];
  __shared__ int32_t route_experts[max_routes];
  using BlockScan = cub::BlockScan<int32_t, 1024>;
  __shared__ typename BlockScan::TempStorage scan_storage;

  const int32_t tid = threadIdx.x;
  for (int32_t index = tid; index < max_num_tokens_padded;
       index += blockDim.x) {
    sorted_token_ids[index] = numel;
  }
  if (tid < num_experts) {
    expert_counts[tid] = 0;
  }
  __syncthreads();

  if (tid < numel) {
    const int32_t global_expert_id = static_cast<int32_t>(topk_ids[tid]);
    int32_t output_expert_id = -1;
    if (global_expert_id >= 0 && global_expert_id < num_experts) {
      output_expert_id =
          has_expert_map ? expert_map[global_expert_id] : global_expert_id;
    }
    route_experts[tid] = output_expert_id;
    if (output_expert_id >= 0) {
      atomicAdd(&expert_counts[output_expert_id], 1);
    }
  }
  __syncthreads();

  int32_t padded_count = 0;
  if (tid < num_experts) {
    padded_count = CEILDIV(expert_counts[tid], block_size) * block_size;
  }
  int32_t padded_offset;
  BlockScan(scan_storage).ExclusiveSum(padded_count, padded_offset);
  if (tid <= num_experts) {
    expert_offsets[tid] = padded_offset;
  }
  if (tid == num_experts) {
    total_tokens_post_pad[0] = padded_offset;
  }
  __syncthreads();

  if (tid < num_experts) {
    for (int32_t index = expert_offsets[tid]; index < expert_offsets[tid + 1];
         index += block_size) {
      expert_ids[index / block_size] = tid;
    }
  }
  const int32_t first_inactive_block = expert_offsets[num_experts] / block_size;
  for (int32_t index = first_inactive_block + tid; index < max_num_m_blocks;
       index += blockDim.x) {
    expert_ids[index] = -1;
  }

  if (tid < numel) {
    const int32_t output_expert_id = route_experts[tid];
    if (output_expert_id >= 0) {
      int32_t rank = 0;
      for (int32_t previous = 0; previous < tid; ++previous) {
        rank += route_experts[previous] == output_expert_id;
      }
      sorted_token_ids[expert_offsets[output_expert_id] + rank] = tid;
    }
  }
}

template <typename scalar_t>
__global__ void prepare_radix_sort_keys_kernel(
    const scalar_t* __restrict__ topk_ids, int32_t* __restrict__ keys,
    const int32_t* __restrict__ expert_map, size_t numel, int32_t num_experts,
    bool has_expert_map) {
  for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < numel;
       index += blockDim.x * gridDim.x) {
    const int32_t expert_id = static_cast<int32_t>(topk_ids[index]);
    if (expert_id < 0 || expert_id >= num_experts) {
      keys[index] = 2 * num_experts - 1;
    } else if (has_expert_map) {
      const int32_t local_expert_id = expert_map[expert_id];
      keys[index] =
          local_expert_id < 0 ? num_experts + expert_id : local_expert_id;
    } else {
      keys[index] = expert_id;
    }
  }
}

__global__ void copy_compact_tokens_to_padded_kernel(
    const int32_t* __restrict__ compact_sorted_token_ids,
    int32_t* __restrict__ sorted_token_ids,
    const int32_t* __restrict__ padded_expert_offsets,
    const int64_t* __restrict__ unpadded_expert_offsets, int32_t num_experts) {
  const int32_t expert_id = blockIdx.x;
  if (expert_id >= num_experts) {
    return;
  }

  const int64_t compact_start = unpadded_expert_offsets[expert_id];
  const int64_t compact_end = unpadded_expert_offsets[expert_id + 1];
  const int32_t padded_start = padded_expert_offsets[expert_id];
  for (int64_t index = compact_start + threadIdx.x; index < compact_end;
       index += blockDim.x) {
    sorted_token_ids[padded_start + index - compact_start] =
        compact_sorted_token_ids[index];
  }
}

__global__ void build_padded_metadata_from_offsets_kernel(
    int32_t* __restrict__ sorted_token_ids, int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ total_tokens_post_pad,
    int32_t* __restrict__ padded_expert_offsets,
    const int64_t* __restrict__ unpadded_expert_offsets, int32_t num_experts,
    int32_t block_size, int32_t numel, int32_t max_num_tokens_padded,
    int32_t max_num_m_blocks) {
  if (blockIdx.x == 1) {
    for (int32_t index = threadIdx.x; index < max_num_tokens_padded;
         index += blockDim.x) {
      sorted_token_ids[index] = numel;
    }
    return;
  }

  using BlockScan = cub::BlockScan<int32_t, 1024>;
  __shared__ typename BlockScan::TempStorage scan_storage;

  const int32_t expert_id = threadIdx.x;
  int32_t padded_count = 0;
  if (expert_id < num_experts) {
    const int64_t count = unpadded_expert_offsets[expert_id + 1] -
                          unpadded_expert_offsets[expert_id];
    padded_count = CEILDIV(count, block_size) * block_size;
  }

  int32_t padded_offset;
  BlockScan(scan_storage).ExclusiveSum(padded_count, padded_offset);
  if (expert_id <= num_experts) {
    padded_expert_offsets[expert_id] = padded_offset;
  }
  if (expert_id == num_experts) {
    total_tokens_post_pad[0] = padded_offset;
  }
  __syncthreads();

  if (expert_id < num_experts) {
    for (int32_t index = padded_expert_offsets[expert_id];
         index < padded_expert_offsets[expert_id + 1]; index += block_size) {
      expert_ids[index / block_size] = expert_id;
    }
  }
  const int32_t fill_start =
      padded_expert_offsets[num_experts] / block_size + threadIdx.x;
  for (int32_t index = fill_start; index < max_num_m_blocks;
       index += blockDim.x) {
    expert_ids[index] = -1;
  }
}

// Reduce the topk expert outputs per token (summed in fp32). The output is
// dense [num_tokens, d]; the input is addressed by its strides so non-
// contiguous inputs work without a copy. A 16B-vectorized path is used when
// the hidden dim is contiguous (innermost stride 1) and aligned; otherwise a
// scalar kernel reads via arbitrary strides. topk is a compile-time constant
// for common values and runtime otherwise.

// Elements per 16-byte vector (8 for bf16/fp16, 4 for fp32).
template <typename scalar_t>
constexpr int MOE_SUM_VEC = 16 / sizeof(scalar_t);

template <typename scalar_t, int TOPK>
__global__ void moe_sum_vec_kernel(
    scalar_t* __restrict__ out,          // [num_tokens, d], contiguous
    const scalar_t* __restrict__ input,  // [num_tokens, topk, d], d contiguous
    const int64_t num_tokens, const int d, const int64_t stride_token,
    const int64_t stride_topk) {
  using vec_t = vllm::vec_n_t<scalar_t, MOE_SUM_VEC<scalar_t>>;  // 16-byte pack
  constexpr int VEC = MOE_SUM_VEC<scalar_t>;
  const int64_t n_vec = d / VEC;
  const int64_t total = num_tokens * n_vec;
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total;
       i += (int64_t)gridDim.x * blockDim.x) {
    const int64_t token = i / n_vec;
    const int64_t v = i % n_vec;
    const scalar_t* in_tok = input + token * stride_token + v * VEC;

    float acc[VEC];
#pragma unroll
    for (int j = 0; j < VEC; ++j) acc[j] = 0.f;

#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
      vec_t packed = *reinterpret_cast<const vec_t*>(in_tok + k * stride_topk);
#pragma unroll
      for (int j = 0; j < VEC; ++j) acc[j] += static_cast<float>(packed.val[j]);
    }

    vec_t outp;
#pragma unroll
    for (int j = 0; j < VEC; ++j) outp.val[j] = static_cast<scalar_t>(acc[j]);
    *reinterpret_cast<vec_t*>(out + token * d + v * VEC) = outp;
  }
}

// Runtime-topk variant of the above.
template <typename scalar_t>
__global__ void moe_sum_vec_dynamic_kernel(
    scalar_t* __restrict__ out,          // [num_tokens, d], contiguous
    const scalar_t* __restrict__ input,  // [num_tokens, topk, d], d contiguous
    const int64_t num_tokens, const int d, const int topk,
    const int64_t stride_token, const int64_t stride_topk) {
  using vec_t = vllm::vec_n_t<scalar_t, MOE_SUM_VEC<scalar_t>>;
  constexpr int VEC = MOE_SUM_VEC<scalar_t>;
  const int64_t n_vec = d / VEC;
  const int64_t total = num_tokens * n_vec;
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total;
       i += (int64_t)gridDim.x * blockDim.x) {
    const int64_t token = i / n_vec;
    const int64_t v = i % n_vec;
    const scalar_t* in_tok = input + token * stride_token + v * VEC;

    float acc[VEC];
#pragma unroll
    for (int j = 0; j < VEC; ++j) acc[j] = 0.f;

    for (int k = 0; k < topk; ++k) {
      vec_t packed = *reinterpret_cast<const vec_t*>(in_tok + k * stride_topk);
#pragma unroll
      for (int j = 0; j < VEC; ++j) acc[j] += static_cast<float>(packed.val[j]);
    }

    vec_t outp;
#pragma unroll
    for (int j = 0; j < VEC; ++j) outp.val[j] = static_cast<scalar_t>(acc[j]);
    *reinterpret_cast<vec_t*>(out + token * d + v * VEC) = outp;
  }
}

// Stride-aware scalar fallback: handles unaligned/non-vectorizable hidden dims
// (including a non-contiguous hidden stride) via per-element strided reads.
template <typename scalar_t>
__global__ void moe_sum_scalar_kernel(
    scalar_t* __restrict__ out,          // [num_tokens, d], contiguous
    const scalar_t* __restrict__ input,  // [num_tokens, topk, d]
    const int d, const int topk, const int64_t stride_token,
    const int64_t stride_topk, const int64_t stride_hidden) {
  const int64_t token_idx = blockIdx.x;
  const scalar_t* in_tok = input + token_idx * stride_token;
  for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
    float x = 0.f;
    for (int k = 0; k < topk; ++k) {
      x += static_cast<float>(
          VLLM_LDG(&in_tok[k * stride_topk + idx * stride_hidden]));
    }
    out[token_idx * d + idx] = static_cast<scalar_t>(x);
  }
}

template <typename scalar_t, int32_t fill_threads>
__global__ void moe_align_block_size_small_batch_expert_kernel(
    const scalar_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids, int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ total_tokens_post_pad,
    int32_t* __restrict__ expert_map, int32_t num_experts, int32_t block_size,
    size_t numel, int32_t max_num_tokens_padded, int32_t topk_num,
    bool has_expert_map) {
  _moe_align_block_size_small_batch_expert<scalar_t, fill_threads>(
      topk_ids, sorted_token_ids, expert_ids, total_tokens_post_pad, expert_map,
      num_experts, block_size, numel, max_num_tokens_padded,
      CEILDIV(max_num_tokens_padded, block_size), -1, 0, topk_num, nullptr,
      has_expert_map);
}

template <typename scalar_t>
__global__ void moe_lora_align_block_size_kernel(
    scalar_t* __restrict__ topk_ids, int32_t* __restrict__ token_lora_mapping,
    int64_t block_size, int32_t* __restrict__ expert_map, int num_experts,
    int max_loras, size_t numel, int max_num_tokens_padded,
    int max_num_m_blocks, int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ expert_ids, int32_t topk_num,
    int32_t* total_tokens_post_pad, int32_t* adapter_enabled,
    int32_t* __restrict__ cumsum, int32_t experts_per_warp,
    int32_t padded_num_experts, int32_t* lora_ids,
    int32_t* __restrict__ token_mask, bool has_expert_map) {
  int lora_idx = blockIdx.x / 2;
  int lora_id = lora_ids[lora_idx];
  // Output buffers are indexed by lora_id (in [0, max_loras)). The grid
  // iterates one extra slot to accommodate the "-1" entry that
  // active_lora_ids may hold in position 0 for mixed base + LoRA batches;
  // guard against any other unexpected lora_id >= max_loras to avoid
  // out-of-bounds writes. This mirrors the `lora_id >= max_loras` guard in
  // the Triton _fused_moe_lora_kernel.
  if (lora_id == -1 || lora_id >= max_loras || adapter_enabled[lora_id] == 0) {
    return;
  }

  // Populate the token_mask based on the token-LoRA mapping
  int num_tokens = numel / topk_num;
  if (threadIdx.x == 0) {
    total_tokens_post_pad[lora_id] = 0;

    for (int i = 0; i < num_tokens; i++) {
      token_mask[(lora_id * num_tokens) + i] =
          (int)token_lora_mapping[i] == lora_id;
    }
  }

  __syncthreads();

  _moe_align_block_size(
      topk_ids, sorted_token_ids, expert_ids, total_tokens_post_pad, expert_map,
      num_experts, padded_num_experts, experts_per_warp, block_size, numel,
      cumsum, max_num_tokens_padded, max_num_m_blocks, lora_id, -1, topk_num,
      &token_mask[(lora_id * num_tokens)], has_expert_map);
}

template <typename scalar_t>
__global__ void lora_count_and_sort_expert_tokens_kernel(
    const scalar_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids, int32_t* __restrict__ cumsum_buffer,
    int32_t* __restrict__ expert_map, size_t numel, int32_t num_experts,
    int32_t max_num_tokens_padded, int32_t topk_num, int32_t* token_mask,
    int32_t max_loras, int32_t* lora_ids, int32_t* adapter_enabled,
    bool has_expert_map) {
  int lora_idx = blockIdx.x;
  int lora_id = lora_ids[lora_idx];
  // Same guard rationale as moe_lora_align_block_size_kernel. Additionally
  // skip disabled adapter slots: moe_lora_align_block_size_kernel early-returns
  // for them and leaves token_mask[lora_id, :] uninitialized (token_mask is
  // allocated with torch::empty), so running the sort loop here would traverse
  // garbage mask bits and pollute this slot's rows of sorted_token_ids and
  // cumsum_buffer. Downstream consumers already skip disabled slots, so the
  // pollution is dormant today, but the check keeps behavior symmetric with
  // the other two align kernels and avoids O(numel) wasted work per disabled
  // slot. Short-circuit evaluation ensures adapter_enabled is only indexed
  // after lora_id is confirmed to be in [0, max_loras).
  if (lora_id == -1 || lora_id >= max_loras || adapter_enabled[lora_id] == 0) {
    return;
  }

  int num_tokens = numel / topk_num;

  _count_and_sort_expert_tokens(
      topk_ids, sorted_token_ids, cumsum_buffer, expert_map, numel, num_experts,
      max_num_tokens_padded, &token_mask[(lora_id * num_tokens)], lora_id,
      topk_num, has_expert_map);
}

template <typename scalar_t, int32_t fill_threads>
__global__ void moe_lora_align_block_size_small_batch_expert_kernel(
    scalar_t* __restrict__ topk_ids, int32_t* token_lora_mapping,
    int64_t block_size, int32_t* __restrict__ expert_map, int num_experts,
    int max_loras, size_t numel, int max_num_tokens_padded,
    int max_num_m_blocks, int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ expert_ids, int topk_num,
    int32_t* total_tokens_post_pad, int32_t* adapter_enabled, int32_t* lora_ids,
    int32_t* token_mask, bool has_expert_map) {
  int lora_idx = blockIdx.x;
  int lora_id = lora_ids[lora_idx];
  // Same guard rationale as moe_lora_align_block_size_kernel.
  if (lora_id == -1 || lora_id >= max_loras || adapter_enabled[lora_id] == 0) {
    return;
  }

  int num_tokens = numel / topk_num;
  if (threadIdx.x == 0) {
    total_tokens_post_pad[lora_id] = 0;

    for (int i = 0; i < num_tokens; i++) {
      token_mask[(lora_id * num_tokens) + i] =
          (int)token_lora_mapping[i] == lora_id;
    }
  }

  __syncthreads();

  _moe_align_block_size_small_batch_expert<scalar_t, fill_threads>(
      topk_ids, sorted_token_ids, expert_ids, total_tokens_post_pad, expert_map,
      num_experts, block_size, numel, max_num_tokens_padded, max_num_m_blocks,
      -1, lora_id, topk_num, &token_mask[(lora_id * num_tokens)],
      has_expert_map);
}

}  // namespace moe
}  // namespace vllm

// taken from
// https://github.com/sgl-project/sglang/blob/8b5f83ed3b7d2a49ad5c5cd5aa61c5d502f47dbc
static void moe_align_block_size_impl(
    torch::stable::Tensor topk_ids, int64_t num_experts, int64_t block_size,
    torch::stable::Tensor sorted_token_ids, torch::stable::Tensor experts_ids,
    torch::stable::Tensor num_tokens_post_pad,
    std::optional<torch::stable::Tensor> maybe_expert_map,
    bool stable_token_order) {
  const torch::stable::accelerator::DeviceGuard device_guard(
      topk_ids.get_device_index());
  const cudaStream_t stream =
      get_current_cuda_stream(topk_ids.get_device_index());

  int64_t padded_num_experts =
      ((num_experts + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
  int experts_per_warp = WARP_SIZE;
  int threads = 1024;
  threads = ((threads + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;

  // BlockScan uses 1024 threads and assigns one thread per expert.
  STD_TORCH_CHECK(padded_num_experts < 1024,
                  "padded_num_experts must be less than 1024");
  STD_TORCH_CHECK(!stable_token_order || topk_ids.numel() <= 256,
                  "stable alignment supports at most 256 routed entries; "
                  "use moe_align_block_size_radix for larger inputs");
  bool has_expert_map = maybe_expert_map.has_value();
  torch::stable::Tensor expert_map;
  if (has_expert_map) {
    expert_map = maybe_expert_map.value();
  } else {
    expert_map = torch::stable::new_empty(topk_ids, {0},
                                          torch::headeronly::ScalarType::Int);
  }

  VLLM_STABLE_DISPATCH_INTEGRAL_AND_UNSIGNED_TYPES(
      topk_ids.scalar_type(), "moe_align_block_size_kernel", [&] {
        constexpr int32_t stable_small_route_limit = 256;
        if (stable_token_order &&
            topk_ids.numel() <= stable_small_route_limit) {
          vllm::moe::stable_small_route_align_kernel<
              scalar_t, stable_small_route_limit><<<1, 1024, 0, stream>>>(
              reinterpret_cast<const scalar_t*>(topk_ids.const_data_ptr()),
              reinterpret_cast<int32_t*>(sorted_token_ids.mutable_data_ptr()),
              reinterpret_cast<int32_t*>(experts_ids.mutable_data_ptr()),
              reinterpret_cast<int32_t*>(
                  num_tokens_post_pad.mutable_data_ptr()),
              reinterpret_cast<const int32_t*>(expert_map.const_data_ptr()),
              num_experts, block_size, topk_ids.numel(),
              sorted_token_ids.size(0), experts_ids.size(0), has_expert_map);
          return;
        }

        // calc needed amount of shared mem for `cumsum` tensors
        bool small_batch_expert_mode =
            (topk_ids.numel() < 1024) && (num_experts <= 64);

        if (small_batch_expert_mode) {
          const int32_t threads = max((int32_t)num_experts, WARP_SIZE);
          const int32_t shared_mem_size =
              ((threads + 1) * num_experts + (num_experts + 1)) *
              sizeof(int32_t);

          // threadIdx.x >= fill_threads: counting experts and aligning
          // threadIdx.x < fill_threads: filling sorted_token_ids
          constexpr int32_t fill_threads = 256;
          auto small_batch_expert_kernel =
              vllm::moe::moe_align_block_size_small_batch_expert_kernel<
                  scalar_t, fill_threads>;
          small_batch_expert_kernel<<<1, fill_threads + threads,
                                      shared_mem_size, stream>>>(
              reinterpret_cast<const scalar_t*>(topk_ids.const_data_ptr()),
              reinterpret_cast<int32_t*>(sorted_token_ids.mutable_data_ptr()),
              reinterpret_cast<int32_t*>(experts_ids.mutable_data_ptr()),
              reinterpret_cast<int32_t*>(
                  num_tokens_post_pad.mutable_data_ptr()),
              reinterpret_cast<int32_t*>(expert_map.mutable_data_ptr()),
              num_experts, block_size, topk_ids.numel(),
              sorted_token_ids.size(0), topk_ids.size(1), has_expert_map);
        } else {
          torch::stable::Tensor cumsum_buffer = torch::stable::new_empty(
              topk_ids, {num_experts + 1}, torch::headeronly::ScalarType::Int);
          auto align_kernel = vllm::moe::moe_align_block_size_kernel<scalar_t>;

          size_t num_warps = CEILDIV(padded_num_experts, experts_per_warp);
          size_t shared_mem_size =
              num_warps * experts_per_warp * sizeof(int32_t);

          // launch two threadblocks
          // blockIdx.x == 0: counting experts and aligning
          // blockIdx.x == 1: filling sorted_token_ids
          align_kernel<<<2, threads, shared_mem_size, stream>>>(
              reinterpret_cast<const scalar_t*>(topk_ids.const_data_ptr()),
              reinterpret_cast<int32_t*>(sorted_token_ids.mutable_data_ptr()),
              reinterpret_cast<int32_t*>(experts_ids.mutable_data_ptr()),
              reinterpret_cast<int32_t*>(
                  num_tokens_post_pad.mutable_data_ptr()),
              reinterpret_cast<int32_t*>(expert_map.mutable_data_ptr()),
              num_experts, padded_num_experts, experts_per_warp, block_size,
              topk_ids.numel(),
              reinterpret_cast<int32_t*>(cumsum_buffer.mutable_data_ptr()),
              sorted_token_ids.size(0), topk_ids.size(1), has_expert_map);

          const int block_threads = std::min(256, (int)threads);
          const int num_blocks =
              (topk_ids.numel() + block_threads - 1) / block_threads;
          const int max_blocks = 65535;
          const int actual_blocks = std::min(num_blocks, max_blocks);
          dim3 gridDims(1, actual_blocks);

          auto sort_kernel =
              vllm::moe::count_and_sort_expert_tokens_kernel<scalar_t>;
          sort_kernel<<<gridDims, block_threads, 0, stream>>>(
              reinterpret_cast<const scalar_t*>(topk_ids.const_data_ptr()),
              reinterpret_cast<int32_t*>(sorted_token_ids.mutable_data_ptr()),
              reinterpret_cast<int32_t*>(cumsum_buffer.mutable_data_ptr()),
              reinterpret_cast<int32_t*>(expert_map.mutable_data_ptr()),
              topk_ids.numel(), num_experts, sorted_token_ids.size(0),
              topk_ids.size(1), has_expert_map);
        }
      });
}

void moe_align_block_size(
    torch::stable::Tensor topk_ids, int64_t num_experts, int64_t block_size,
    torch::stable::Tensor sorted_token_ids, torch::stable::Tensor experts_ids,
    torch::stable::Tensor num_tokens_post_pad,
    std::optional<torch::stable::Tensor> maybe_expert_map) {
  moe_align_block_size_impl(topk_ids, num_experts, block_size, sorted_token_ids,
                            experts_ids, num_tokens_post_pad, maybe_expert_map,
                            false);
}

void moe_align_block_size_stable_small(
    torch::stable::Tensor topk_ids, int64_t num_experts, int64_t block_size,
    torch::stable::Tensor sorted_token_ids, torch::stable::Tensor experts_ids,
    torch::stable::Tensor num_tokens_post_pad,
    std::optional<torch::stable::Tensor> maybe_expert_map) {
  moe_align_block_size_impl(topk_ids, num_experts, block_size, sorted_token_ids,
                            experts_ids, num_tokens_post_pad, maybe_expert_map,
                            true);
}

void moe_align_block_size_radix(
    torch::stable::Tensor topk_ids, int64_t num_experts, int64_t block_size,
    torch::stable::Tensor sorted_token_ids, torch::stable::Tensor experts_ids,
    torch::stable::Tensor num_tokens_post_pad,
    torch::stable::Tensor sort_workspace,
    torch::stable::Tensor sorted_expert_ids,
    torch::stable::Tensor compact_sorted_token_ids,
    torch::stable::Tensor token_indices,
    torch::stable::Tensor topk_ids_for_sort,
    torch::stable::Tensor padded_expert_offsets,
    torch::stable::Tensor unpadded_expert_offsets,
    std::optional<torch::stable::Tensor> maybe_expert_map) {
#ifdef USE_ROCM
  STD_TORCH_CHECK(false, "moe_align_block_size_radix is not supported on ROCm");
#else
  const torch::stable::accelerator::DeviceGuard device_guard(
      topk_ids.get_device_index());
  const cudaStream_t stream =
      get_current_cuda_stream(topk_ids.get_device_index());
  const int64_t numel = topk_ids.numel();

  STD_TORCH_CHECK(num_experts > 0 && num_experts < 1024,
                  "num_experts must be in [1, 1024)");
  STD_TORCH_CHECK(block_size > 0, "block_size must be positive");
  STD_TORCH_CHECK(numel > 0, "topk_ids must not be empty");
  STD_TORCH_CHECK(numel <= std::numeric_limits<int32_t>::max(),
                  "topk_ids contains too many entries for int32 token indices");
  const int64_t max_padding = num_experts * (block_size - 1);
  STD_TORCH_CHECK(numel <= std::numeric_limits<int32_t>::max() - max_padding,
                  "padded token count exceeds the int32 alignment ABI");
  STD_TORCH_CHECK(topk_ids.is_contiguous(), "topk_ids must be contiguous");

  auto check_scratch = [&](const torch::stable::Tensor& tensor,
                           torch::headeronly::ScalarType dtype,
                           const char* name) {
    STD_TORCH_CHECK(tensor.device() == topk_ids.device(), name,
                    " must be on the same device as topk_ids");
    STD_TORCH_CHECK(tensor.scalar_type() == dtype, name,
                    " has an unexpected dtype");
    STD_TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  };
  check_scratch(sort_workspace, torch::headeronly::ScalarType::Char,
                "sort_workspace");
  check_scratch(sorted_expert_ids, torch::headeronly::ScalarType::Int,
                "sorted_expert_ids");
  check_scratch(compact_sorted_token_ids, torch::headeronly::ScalarType::Int,
                "compact_sorted_token_ids");
  check_scratch(token_indices, torch::headeronly::ScalarType::Int,
                "token_indices");
  check_scratch(topk_ids_for_sort, torch::headeronly::ScalarType::Int,
                "topk_ids_for_sort");
  check_scratch(padded_expert_offsets, torch::headeronly::ScalarType::Int,
                "padded_expert_offsets");
  check_scratch(unpadded_expert_offsets, torch::headeronly::ScalarType::Long,
                "unpadded_expert_offsets");
  STD_TORCH_CHECK(sorted_expert_ids.numel() >= numel,
                  "sorted_expert_ids is too small");
  STD_TORCH_CHECK(compact_sorted_token_ids.numel() >= numel,
                  "compact_sorted_token_ids is too small");
  STD_TORCH_CHECK(token_indices.numel() >= numel, "token_indices is too small");
  STD_TORCH_CHECK(topk_ids_for_sort.numel() >= numel,
                  "topk_ids_for_sort is too small");
  STD_TORCH_CHECK(padded_expert_offsets.numel() >= num_experts + 1,
                  "padded_expert_offsets is too small");
  STD_TORCH_CHECK(unpadded_expert_offsets.numel() >= num_experts + 1,
                  "unpadded_expert_offsets is too small");

  bool has_expert_map = maybe_expert_map.has_value();
  if (has_expert_map) {
    check_scratch(maybe_expert_map.value(), torch::headeronly::ScalarType::Int,
                  "expert_map");
    STD_TORCH_CHECK(maybe_expert_map.value().numel() >= num_experts,
                    "expert_map is too small");
  }
  const int32_t* expert_map_ptr =
      has_expert_map ? reinterpret_cast<const int32_t*>(
                           maybe_expert_map.value().const_data_ptr())
                     : nullptr;

  VLLM_STABLE_DISPATCH_INTEGRAL_AND_UNSIGNED_TYPES(
      topk_ids.scalar_type(), "moe_align_block_size_radix", [&] {
        constexpr int32_t key_threads = 256;
        const int32_t key_blocks =
            std::min<int64_t>(CEILDIV(numel, key_threads), 65535);
        vllm::moe::prepare_radix_sort_keys_kernel<scalar_t>
            <<<key_blocks, key_threads, 0, stream>>>(
                reinterpret_cast<const scalar_t*>(topk_ids.const_data_ptr()),
                reinterpret_cast<int32_t*>(
                    topk_ids_for_sort.mutable_data_ptr()),
                expert_map_ptr, numel, num_experts, has_expert_map);
      });

  // CUB radix sort is stable. Monotonic input values therefore preserve
  // flattened route order within equal expert keys without a composite key.
  CubKeyValueSorter sorter(num_experts);
  sorter.run(
      sort_workspace.mutable_data_ptr(), sort_workspace.numel(),
      reinterpret_cast<const int32_t*>(topk_ids_for_sort.const_data_ptr()),
      reinterpret_cast<int32_t*>(sorted_expert_ids.mutable_data_ptr()),
      reinterpret_cast<const int32_t*>(token_indices.const_data_ptr()),
      reinterpret_cast<int32_t*>(compact_sorted_token_ids.mutable_data_ptr()),
      numel, stream);

  computeExpertFirstTokenOffset(
      reinterpret_cast<const int32_t*>(sorted_expert_ids.const_data_ptr()),
      numel, num_experts,
      reinterpret_cast<int64_t*>(unpadded_expert_offsets.mutable_data_ptr()),
      stream);

  vllm::moe::build_padded_metadata_from_offsets_kernel<<<2, 1024, 0, stream>>>(
      reinterpret_cast<int32_t*>(sorted_token_ids.mutable_data_ptr()),
      reinterpret_cast<int32_t*>(experts_ids.mutable_data_ptr()),
      reinterpret_cast<int32_t*>(num_tokens_post_pad.mutable_data_ptr()),
      reinterpret_cast<int32_t*>(padded_expert_offsets.mutable_data_ptr()),
      reinterpret_cast<const int64_t*>(
          unpadded_expert_offsets.const_data_ptr()),
      num_experts, block_size, numel, sorted_token_ids.size(0),
      experts_ids.size(0));

  vllm::moe::
      copy_compact_tokens_to_padded_kernel<<<num_experts, 256, 0, stream>>>(
          reinterpret_cast<const int32_t*>(
              compact_sorted_token_ids.const_data_ptr()),
          reinterpret_cast<int32_t*>(sorted_token_ids.mutable_data_ptr()),
          reinterpret_cast<const int32_t*>(
              padded_expert_offsets.const_data_ptr()),
          reinterpret_cast<const int64_t*>(
              unpadded_expert_offsets.const_data_ptr()),
          num_experts);
#endif
}

void batched_moe_align_block_size(int64_t max_tokens_per_batch,
                                  int64_t block_size,
                                  const torch::stable::Tensor& batch_num_tokens,
                                  torch::stable::Tensor sorted_ids,
                                  torch::stable::Tensor batch_ids,
                                  torch::stable::Tensor num_tokens_post_pad) {
  namespace batched_kernel = vllm::moe::batched_moe_align_block_size;

  const torch::stable::accelerator::DeviceGuard device_guard(
      batch_num_tokens.get_device_index());
  const cudaStream_t stream =
      get_current_cuda_stream(batch_num_tokens.get_device_index());
  int32_t const B = batch_num_tokens.size(0);
  int32_t const num_blocks_per_batch =
      round_to_next_multiple_of(max_tokens_per_batch, block_size) / block_size;
  int32_t const num_blocks = num_blocks_per_batch * B;
  int64_t const sorted_ids_size = num_blocks * block_size;

  STD_TORCH_CHECK(sorted_ids.size(0) == sorted_ids_size);
  STD_TORCH_CHECK(batch_ids.size(0) == sorted_ids_size / block_size);
  STD_TORCH_CHECK(num_tokens_post_pad.size(0) == 1);
  STD_TORCH_CHECK(B <= batched_kernel::num_threads);

  batched_kernel::batched_moe_align_block_size_kernel<<<
      batched_kernel::num_blocks, batched_kernel::num_threads, 0, stream>>>(
      B, max_tokens_per_batch, block_size,
      reinterpret_cast<const int32_t*>(batch_num_tokens.const_data_ptr()),
      reinterpret_cast<int32_t*>(sorted_ids.mutable_data_ptr()),
      reinterpret_cast<int32_t*>(batch_ids.mutable_data_ptr()),
      reinterpret_cast<int32_t*>(num_tokens_post_pad.mutable_data_ptr()));
}

void moe_sum(torch::stable::Tensor& input,   // [num_tokens, topk, hidden_size]
             torch::stable::Tensor& output)  // [num_tokens, hidden_size]
{
  // Output is dense and written in place, so it must be contiguous. The input
  // is read by its strides (no copy); only the hidden dim needs to be
  // contiguous to take the vectorized path.
  STD_TORCH_CHECK(output.is_contiguous(),
                  "moe_sum expects a contiguous output");

  const int hidden_size = input.size(-1);
  const int64_t num_tokens = output.numel() / hidden_size;
  const int topk = input.size(1);
  const int64_t stride_token = input.stride(0);
  const int64_t stride_topk = input.stride(1);
  const int64_t stride_hidden = input.stride(2);

  const torch::stable::accelerator::DeviceGuard device_guard(
      output.get_device_index());
  const cudaStream_t stream =
      get_current_cuda_stream(output.get_device_index());

#define LAUNCH_MOE_SUM_VEC(TOPK)                \
  vllm::moe::moe_sum_vec_kernel<scalar_t, TOPK> \
      <<<grid, dim3(block), 0, stream>>>(       \
          out_ptr, in_ptr, num_tokens, hidden_size, stride_token, stride_topk)

  VLLM_STABLE_DISPATCH_FLOATING_TYPES(input.scalar_type(), "moe_sum", [&] {
    constexpr int VEC = vllm::moe::MOE_SUM_VEC<scalar_t>;
    constexpr int WIDTH = VEC * sizeof(scalar_t);  // 16 bytes
    auto* out_ptr = reinterpret_cast<scalar_t*>(output.mutable_data_ptr());
    auto* in_ptr = reinterpret_cast<const scalar_t*>(input.const_data_ptr());

    // Vectorize along hidden only when it is contiguous (innermost stride 1),
    // a whole number of vectors, and every row offset stays 16B-aligned.
    const bool can_vec = (stride_hidden == 1) && (hidden_size % VEC == 0) &&
                         (stride_token % VEC == 0) &&
                         (stride_topk % VEC == 0) &&
                         (reinterpret_cast<uintptr_t>(in_ptr) % WIDTH == 0) &&
                         (reinterpret_cast<uintptr_t>(out_ptr) % WIDTH == 0);
    if (can_vec) {
      const int64_t n_vec = hidden_size / VEC;
      const int64_t total = num_tokens * n_vec;
      const int block = 256;
      const dim3 grid(std::min<int64_t>((total + block - 1) / block, 65535));
      switch (topk) {
        case 1:
          LAUNCH_MOE_SUM_VEC(1);
          break;
        case 2:
          LAUNCH_MOE_SUM_VEC(2);
          break;
        case 4:
          LAUNCH_MOE_SUM_VEC(4);
          break;
        case 6:
          LAUNCH_MOE_SUM_VEC(6);
          break;
        case 8:
          LAUNCH_MOE_SUM_VEC(8);
          break;
        case 9:
          LAUNCH_MOE_SUM_VEC(9);
          break;
        default:
          vllm::moe::moe_sum_vec_dynamic_kernel<scalar_t>
              <<<grid, dim3(block), 0, stream>>>(out_ptr, in_ptr, num_tokens,
                                                 hidden_size, topk,
                                                 stride_token, stride_topk);
          break;
      }
    } else {
      dim3 grid(num_tokens);
      dim3 block(std::min(hidden_size, 1024));
      vllm::moe::moe_sum_scalar_kernel<scalar_t><<<grid, block, 0, stream>>>(
          out_ptr, in_ptr, hidden_size, topk, stride_token, stride_topk,
          stride_hidden);
    }
  });
#undef LAUNCH_MOE_SUM_VEC
}

void moe_lora_align_block_size(
    torch::stable::Tensor topk_ids, torch::stable::Tensor token_lora_mapping,
    int64_t num_experts, int64_t block_size, int64_t max_loras,
    int64_t max_num_tokens_padded, int64_t max_num_m_blocks,
    torch::stable::Tensor sorted_token_ids, torch::stable::Tensor expert_ids,
    torch::stable::Tensor num_tokens_post_pad,
    torch::stable::Tensor adapter_enabled, torch::stable::Tensor lora_ids,
    std::optional<torch::stable::Tensor> maybe_expert_map) {
  const int topk_num = topk_ids.size(1);

  STD_TORCH_CHECK(block_size > 0, "block_size should be greater than 0. ");

  int device_max_shared_mem;
  int dev = topk_ids.get_device_index();
  const torch::stable::accelerator::DeviceGuard device_guard(dev);
  cudaDeviceGetAttribute(&device_max_shared_mem,
                         cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  const cudaStream_t stream = get_current_cuda_stream(dev);

  int64_t padded_num_experts =
      ((num_experts + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;

  // BlockScan uses 1024 threads and assigns one thread per expert.
  STD_TORCH_CHECK(padded_num_experts < 1024,
                  "padded_num_experts must be less than 1024");

  torch::stable::Tensor token_mask =
      torch::stable::new_empty(topk_ids, {max_loras * topk_ids.size(0)},
                               torch::headeronly::ScalarType::Int);
  bool has_expert_map = maybe_expert_map.has_value();
  torch::stable::Tensor expert_map;
  if (has_expert_map) {
    expert_map = maybe_expert_map.value();
  } else {
    expert_map = torch::stable::new_empty(topk_ids, {0},
                                          torch::headeronly::ScalarType::Int);
  }

  VLLM_STABLE_DISPATCH_INTEGRAL_TYPES(
      topk_ids.scalar_type(), "moe_lora_align_sum_kernel", [&] {
        bool small_batch_expert_mode =
            (topk_ids.numel() < 1024) && (num_experts <= 64);

        if (small_batch_expert_mode) {
          const int32_t num_thread = max((int32_t)num_experts, 128);
          const int32_t shared_mem =
              (num_thread + 1) * num_experts * sizeof(int32_t) +
              (num_experts + 1) * sizeof(int32_t);
          if (shared_mem > device_max_shared_mem) {
            STD_TORCH_CHECK(false, "Shared memory usage exceeds device limit.");
          }

          // threadIdx.x >= fill_threads: counting experts and aligning
          // threadIdx.x < fill_threads: filling sorted_token_ids
          constexpr int32_t fill_threads = 256;

          dim3 blockDim(num_thread + fill_threads);
          auto kernel =
              vllm::moe::moe_lora_align_block_size_small_batch_expert_kernel<
                  scalar_t, fill_threads>;
          STD_CUDA_CHECK(VLLM_DevFuncAttribute_SET_MaxDynamicSharedMemorySize(
              (void*)kernel, shared_mem));
          // Grid size is (max_loras + 1) because active_lora_ids has length
          // max_loras + 1: sorted-unique values of token_lora_mapping, which
          // can include -1 (base-model tokens) in addition to up to max_loras
          // real LoRA slots. Using max_loras would drop the real LoRA slot
          // when -1 is present at position 0 and leave output buffers
          // uninitialized, causing illegal memory accesses in downstream
          // MoE-LoRA kernels. This mirrors the fix made for the Triton
          // _fused_moe_lora_kernel grid in vllm-project/vllm#32277.
          kernel<<<max_loras + 1, blockDim, shared_mem, stream>>>(
              reinterpret_cast<scalar_t*>(topk_ids.mutable_data_ptr()),
              reinterpret_cast<int32_t*>(token_lora_mapping.mutable_data_ptr()),
              block_size,
              reinterpret_cast<int32_t*>(expert_map.mutable_data_ptr()),
              num_experts, max_loras, topk_ids.numel(), max_num_tokens_padded,
              max_num_m_blocks,
              reinterpret_cast<int32_t*>(sorted_token_ids.mutable_data_ptr()),
              reinterpret_cast<int32_t*>(expert_ids.mutable_data_ptr()),
              topk_num,
              reinterpret_cast<int32_t*>(
                  num_tokens_post_pad.mutable_data_ptr()),
              reinterpret_cast<int32_t*>(adapter_enabled.mutable_data_ptr()),
              reinterpret_cast<int32_t*>(lora_ids.mutable_data_ptr()),
              reinterpret_cast<int32_t*>(token_mask.mutable_data_ptr()),
              has_expert_map);
        } else {
          int num_thread = 1024;
          dim3 blockDim(num_thread);
          size_t num_warps = CEILDIV(padded_num_experts, WARP_SIZE);

          size_t shared_mem_size = num_warps * WARP_SIZE * sizeof(int32_t);

          // cumsum buffer
          torch::stable::Tensor cumsum = torch::stable::new_zeros(
              topk_ids, {max_loras * (num_experts + 1)},
              torch::headeronly::ScalarType::Int);

          auto align_kernel =
              vllm::moe::moe_lora_align_block_size_kernel<scalar_t>;

          // Launch two threadblocks per LoRA slot, across max_loras + 1 slots
          // to cover the extra "-1" (base-model tokens) entry that
          // active_lora_ids may contain in addition to up to max_loras real
          // LoRA slots. Using max_loras would drop the real LoRA slot when -1
          // occupies position 0 and leave the output buffers uninitialized,
          // causing illegal memory accesses downstream. Mirrors the grid fix
          // applied to _fused_moe_lora_kernel in vllm-project/vllm#32277.
          // blockIdx.x % 2 == 0: counting experts and aligning
          // blockIdx.x % 2 == 1: filling sorted_token_ids
          align_kernel<<<(max_loras + 1) * 2, blockDim, shared_mem_size,
                         stream>>>(
              reinterpret_cast<scalar_t*>(topk_ids.mutable_data_ptr()),
              reinterpret_cast<int32_t*>(token_lora_mapping.mutable_data_ptr()),
              block_size,
              reinterpret_cast<int32_t*>(expert_map.mutable_data_ptr()),
              num_experts, max_loras, topk_ids.numel(), max_num_tokens_padded,
              max_num_m_blocks,
              reinterpret_cast<int32_t*>(sorted_token_ids.mutable_data_ptr()),
              reinterpret_cast<int32_t*>(expert_ids.mutable_data_ptr()),
              topk_num,
              reinterpret_cast<int32_t*>(
                  num_tokens_post_pad.mutable_data_ptr()),
              reinterpret_cast<int32_t*>(adapter_enabled.mutable_data_ptr()),
              reinterpret_cast<int32_t*>(cumsum.mutable_data_ptr()), WARP_SIZE,
              padded_num_experts,
              reinterpret_cast<int32_t*>(lora_ids.mutable_data_ptr()),
              reinterpret_cast<int32_t*>(token_mask.mutable_data_ptr()),
              has_expert_map);

          const int block_threads = std::min(256, (int)num_thread);
          const int num_blocks =
              (topk_ids.numel() + block_threads - 1) / block_threads;

          const int max_blocks = 65535;
          const int actual_blocks = std::min(num_blocks, max_blocks);

          // Same rationale as align_kernel above: iterate over max_loras + 1
          // slots so the sort kernel processes the real LoRA slot even when
          // active_lora_ids has -1 at position 0.
          dim3 gridDims(max_loras + 1, actual_blocks);
          auto sort_kernel =
              vllm::moe::lora_count_and_sort_expert_tokens_kernel<scalar_t>;

          sort_kernel<<<gridDims, block_threads, 0, stream>>>(
              reinterpret_cast<const scalar_t*>(topk_ids.const_data_ptr()),
              reinterpret_cast<int32_t*>(sorted_token_ids.mutable_data_ptr()),
              reinterpret_cast<int32_t*>(cumsum.mutable_data_ptr()),
              reinterpret_cast<int32_t*>(expert_map.mutable_data_ptr()),
              topk_ids.numel(), num_experts, max_num_tokens_padded, topk_num,
              reinterpret_cast<int32_t*>(token_mask.mutable_data_ptr()),
              max_loras,
              reinterpret_cast<int32_t*>(lora_ids.mutable_data_ptr()),
              reinterpret_cast<int32_t*>(adapter_enabled.mutable_data_ptr()),
              has_expert_map);
        }
      });
}
