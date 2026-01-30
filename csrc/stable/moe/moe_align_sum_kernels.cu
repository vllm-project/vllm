#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>

#include <cub/cub.cuh>

#include "cuda_compat.h"
#include "core/math.hpp"
#include "stable/dispatch_utils.h"
#include "stable/torch_utils.h"

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
  // Intialize sorted_ids
  for (size_t i = threadIdx.x; i < sorted_ids_size; i += stride) {
    sorted_ids[i] = SENTINEL;
  }
  // Intialize expert_ids with -1
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

  // Fill remaining expert_ids with 0
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

  // Fill remaining expert_ids with 0
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
      0, 0, topk_num, nullptr, has_expert_map);
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

template <typename scalar_t, int TOPK>
__global__ void moe_sum_kernel(
    scalar_t* __restrict__ out,          // [..., d]
    const scalar_t* __restrict__ input,  // [..., topk, d]
    const int d) {
  const int64_t token_idx = blockIdx.x;
  for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
    scalar_t x = 0.0;
#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
      x += VLLM_LDG(&input[token_idx * TOPK * d + k * d + idx]);
    }
    out[token_idx * d + idx] = x;
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
      CEILDIV(max_num_tokens_padded, block_size), 0, 0, topk_num, nullptr,
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
  if (lora_id == -1 || adapter_enabled[lora_id] == 0) {
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
    int32_t* lora_ids, bool has_expert_map) {
  int lora_idx = blockIdx.x;
  int lora_id = lora_ids[lora_idx];
  if (lora_id == -1) {
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
  if (lora_id == -1 || adapter_enabled[lora_id] == 0) {
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
void moe_align_block_size(
    torch::stable::Tensor topk_ids, int64_t num_experts, int64_t block_size,
    torch::stable::Tensor sorted_token_ids, torch::stable::Tensor experts_ids,
    torch::stable::Tensor num_tokens_post_pad,
    std::optional<torch::stable::Tensor> maybe_expert_map) {
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

  bool has_expert_map = maybe_expert_map.has_value();
  torch::stable::Tensor expert_map;
  if (has_expert_map) {
    expert_map = maybe_expert_map.value();
  } else {
    expert_map = torch::stable::empty({0}, torch::headeronly::ScalarType::Int,
                                      std::nullopt, topk_ids.device());
  }

  VLLM_STABLE_DISPATCH_INTEGRAL_AND_UNSIGNED_TYPES(
      topk_ids.scalar_type(), "moe_align_block_size_kernel", [&] {
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
              topk_ids.const_data_ptr<scalar_t>(),
              sorted_token_ids.mutable_data_ptr<int32_t>(),
              experts_ids.mutable_data_ptr<int32_t>(),
              num_tokens_post_pad.mutable_data_ptr<int32_t>(),
              expert_map.mutable_data_ptr<int32_t>(), num_experts, block_size,
              topk_ids.numel(), sorted_token_ids.size(0), topk_ids.size(1),
              has_expert_map);
        } else {
          torch::stable::Tensor cumsum_buffer = torch::stable::empty(
              {num_experts + 1}, torch::headeronly::ScalarType::Int,
              std::nullopt, topk_ids.device());
          auto align_kernel = vllm::moe::moe_align_block_size_kernel<scalar_t>;

          size_t num_warps = CEILDIV(padded_num_experts, experts_per_warp);
          size_t shared_mem_size =
              num_warps * experts_per_warp * sizeof(int32_t);

          // launch two threadblocks
          // blockIdx.x == 0: counting experts and aligning
          // blockIdx.x == 1: filling sorted_token_ids
          align_kernel<<<2, threads, shared_mem_size, stream>>>(
              topk_ids.const_data_ptr<scalar_t>(),
              sorted_token_ids.mutable_data_ptr<int32_t>(),
              experts_ids.mutable_data_ptr<int32_t>(),
              num_tokens_post_pad.mutable_data_ptr<int32_t>(),
              expert_map.mutable_data_ptr<int32_t>(), num_experts,
              padded_num_experts, experts_per_warp, block_size,
              topk_ids.numel(), cumsum_buffer.mutable_data_ptr<int32_t>(),
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
              topk_ids.const_data_ptr<scalar_t>(),
              sorted_token_ids.mutable_data_ptr<int32_t>(),
              cumsum_buffer.mutable_data_ptr<int32_t>(),
              expert_map.mutable_data_ptr<int32_t>(), topk_ids.numel(),
              num_experts, sorted_token_ids.size(0), topk_ids.size(1),
              has_expert_map);
        }
      });
}

void batched_moe_align_block_size(int64_t max_tokens_per_batch,
                                  int64_t block_size,
                                  torch::stable::Tensor const& batch_num_tokens,
                                  torch::stable::Tensor sorted_ids,
                                  torch::stable::Tensor batch_ids,
                                  torch::stable::Tensor num_tokens_post_pad) {
  namespace batched_kernel = vllm::moe::batched_moe_align_block_size;

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
      batch_num_tokens.const_data_ptr<int32_t>(),
      sorted_ids.mutable_data_ptr<int32_t>(),
      batch_ids.mutable_data_ptr<int32_t>(),
      num_tokens_post_pad.mutable_data_ptr<int32_t>());
}

void moe_sum(torch::stable::Tensor& input,   // [num_tokens, topk, hidden_size]
             torch::stable::Tensor& output)  // [num_tokens, hidden_size]
{
  const int hidden_size = input.size(-1);
  const auto num_tokens = output.numel() / hidden_size;
  const int topk = input.size(1);

  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, 1024));
  torch::stable::accelerator::DeviceGuard device_guard(
      output.get_device_index());
  const cudaStream_t stream =
      get_current_cuda_stream(output.get_device_index());

  switch (topk) {
    case 2:
      VLLM_STABLE_DISPATCH_FLOATING_TYPES(
          input.scalar_type(), "moe_sum_kernel", [&] {
            vllm::moe::moe_sum_kernel<scalar_t, 2><<<grid, block, 0, stream>>>(
                output.mutable_data_ptr<scalar_t>(),
                input.const_data_ptr<scalar_t>(), hidden_size);
          });
      break;

    case 3:
      VLLM_STABLE_DISPATCH_FLOATING_TYPES(
          input.scalar_type(), "moe_sum_kernel", [&] {
            vllm::moe::moe_sum_kernel<scalar_t, 3><<<grid, block, 0, stream>>>(
                output.mutable_data_ptr<scalar_t>(),
                input.const_data_ptr<scalar_t>(), hidden_size);
          });
      break;

    case 4:
      VLLM_STABLE_DISPATCH_FLOATING_TYPES(
          input.scalar_type(), "moe_sum_kernel", [&] {
            vllm::moe::moe_sum_kernel<scalar_t, 4><<<grid, block, 0, stream>>>(
                output.mutable_data_ptr<scalar_t>(),
                input.const_data_ptr<scalar_t>(), hidden_size);
          });
      break;

    default:
      int64_t dim = 1;
      torch::stable::sum_out(output, input, dim);
      break;
  }
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

  STD_TORCH_CHECK(block_size > 0, "block_size should be greater than 0.");

  int device_max_shared_mem;
  auto dev = topk_ids.get_device_index();
  cudaDeviceGetAttribute(&device_max_shared_mem,
                         cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  const cudaStream_t stream = get_current_cuda_stream(dev);

  int64_t padded_num_experts =
      ((num_experts + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;

  // BlockScan uses 1024 threads and assigns one thread per expert.
  STD_TORCH_CHECK(padded_num_experts < 1024,
                  "padded_num_experts must be less than 1024");

  torch::stable::Tensor token_mask = torch::stable::empty(
      {max_loras * topk_ids.size(0)}, torch::headeronly::ScalarType::Int,
      std::nullopt, topk_ids.device());
  bool has_expert_map = maybe_expert_map.has_value();
  torch::stable::Tensor expert_map;
  if (has_expert_map) {
    expert_map = maybe_expert_map.value();
  } else {
    expert_map = torch::stable::empty({0}, torch::headeronly::ScalarType::Int,
                                      std::nullopt, topk_ids.device());
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
          kernel<<<max_loras, blockDim, shared_mem, stream>>>(
              topk_ids.mutable_data_ptr<scalar_t>(),
              token_lora_mapping.mutable_data_ptr<int32_t>(), block_size,
              expert_map.mutable_data_ptr<int32_t>(), num_experts, max_loras,
              topk_ids.numel(), max_num_tokens_padded, max_num_m_blocks,
              sorted_token_ids.mutable_data_ptr<int32_t>(),
              expert_ids.mutable_data_ptr<int32_t>(), topk_num,
              num_tokens_post_pad.mutable_data_ptr<int32_t>(),
              adapter_enabled.mutable_data_ptr<int32_t>(),
              lora_ids.mutable_data_ptr<int32_t>(),
              token_mask.mutable_data_ptr<int32_t>(), has_expert_map);
        } else {
          int num_thread = 1024;
          dim3 blockDim(num_thread);
          size_t num_warps = CEILDIV(padded_num_experts, WARP_SIZE);

          size_t shared_mem_size = num_warps * WARP_SIZE * sizeof(int32_t);

          // cumsum buffer
          torch::stable::Tensor cumsum =
              torch::stable::empty({max_loras * (num_experts + 1)},
                                   torch::headeronly::ScalarType::Int,
                                   std::nullopt, topk_ids.device());
          torch::stable::zero_(cumsum);

          auto align_kernel =
              vllm::moe::moe_lora_align_block_size_kernel<scalar_t>;

          // launch two threadblocks for each lora
          // blockIdx.x % 2 == 0: counting experts and aligning
          // blockIdx.x % 2 == 1: filling sorted_token_ids
          align_kernel<<<max_loras * 2, blockDim, shared_mem_size, stream>>>(
              topk_ids.mutable_data_ptr<scalar_t>(),
              token_lora_mapping.mutable_data_ptr<int32_t>(), block_size,
              expert_map.mutable_data_ptr<int32_t>(), num_experts, max_loras,
              topk_ids.numel(), max_num_tokens_padded, max_num_m_blocks,
              sorted_token_ids.mutable_data_ptr<int32_t>(),
              expert_ids.mutable_data_ptr<int32_t>(), topk_num,
              num_tokens_post_pad.mutable_data_ptr<int32_t>(),
              adapter_enabled.mutable_data_ptr<int32_t>(),
              cumsum.mutable_data_ptr<int32_t>(), WARP_SIZE, padded_num_experts,
              lora_ids.mutable_data_ptr<int32_t>(),
              token_mask.mutable_data_ptr<int32_t>(), has_expert_map);

          const int block_threads = std::min(256, (int)num_thread);
          const int num_blocks =
              (topk_ids.numel() + block_threads - 1) / block_threads;

          const int max_blocks = 65535;
          const int actual_blocks = std::min(num_blocks, max_blocks);

          dim3 gridDims(max_loras, actual_blocks);
          auto sort_kernel =
              vllm::moe::lora_count_and_sort_expert_tokens_kernel<scalar_t>;

          sort_kernel<<<gridDims, block_threads, 0, stream>>>(
              topk_ids.const_data_ptr<scalar_t>(),
              sorted_token_ids.mutable_data_ptr<int32_t>(),
              cumsum.mutable_data_ptr<int32_t>(),
              expert_map.mutable_data_ptr<int32_t>(), topk_ids.numel(),
              num_experts, max_num_tokens_padded, topk_num,
              token_mask.mutable_data_ptr<int32_t>(),
              lora_ids.mutable_data_ptr<int32_t>(), has_expert_map);
        }
      });
}
