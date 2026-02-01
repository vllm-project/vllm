#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cub/cub.cuh>

#include <ATen/ATen.h>
#include <ATen/cuda/Atomic.cuh>

#include "../cuda_compat.h"
#include "../dispatch_utils.h"
#include "core/math.hpp"

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
__global__ void moe_align_block_size_kernel(
    const scalar_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids, int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ total_tokens_post_pad,
    int32_t* __restrict__ expert_map, int32_t num_experts,
    int32_t padded_num_experts, int32_t experts_per_warp, int32_t block_size,
    size_t numel, int32_t* __restrict__ cumsum, int32_t max_num_tokens_padded,
    int32_t topk_num, bool has_expert_map) {
  extern __shared__ int32_t shared_counts[];

  const int32_t max_num_m_blocks = CEILDIV(max_num_tokens_padded, block_size);

  // Use separate threadblocks to fill sorted_token_ids.
  // This is safe since the current kernel does not use sorted_token_ids.
  if (blockIdx.x % 2) {
    // Initialize sorted_token_ids with numel
    for (size_t it = threadIdx.x; it < max_num_tokens_padded;
         it += blockDim.x) {
      sorted_token_ids[it] = numel;
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
    atomicAdd(&shared_counts[warp_idx * experts_per_warp + expert_offset], 1);
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
    cumsum[expert_id] = cumsum_val;
  }

  if (expert_id == num_experts) {
    total_tokens_post_pad[0] = cumsum_val;
  }

  __syncthreads();

  if (threadIdx.x < num_experts) {
    for (int i = cumsum[threadIdx.x]; i < cumsum[threadIdx.x + 1];
         i += block_size) {
      expert_ids[i / block_size] = threadIdx.x;
    }
  }

  // Fill remaining expert_ids with 0
  const size_t fill_start_idx = cumsum[num_experts] / block_size + threadIdx.x;
  for (size_t i = fill_start_idx; i < max_num_m_blocks; i += blockDim.x) {
    expert_ids[i] = 0;
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
  const int32_t max_num_m_blocks = CEILDIV(max_num_tokens_padded, block_size);

  // Use an additional group of threads to fill sorted_token_ids.
  // Since the current kernel will use sorted_token_ids afterward,
  // we fill sorted_token_ids within the same threadblock to make
  // synchronization easier.
  if (threadIdx.x < fill_threads) {
    // Initialize sorted_token_ids with numel
    for (size_t it = threadIdx.x; it < max_num_tokens_padded;
         it += fill_threads) {
      sorted_token_ids[it] = numel;
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
    ++tokens_cnts[(tid + 1) * num_experts + expert_id];
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
    total_tokens_post_pad[0] = static_cast<int32_t>(cumsum[num_experts]);
  }

  __syncthreads();

  if (tid < num_experts) {
    for (int i = cumsum[tid]; i < cumsum[tid + 1]; i += block_size) {
      expert_ids[i / block_size] = tid;
    }
  }

  // Fill remaining expert_ids with 0
  const size_t fill_start_idx = cumsum[num_experts] / block_size + tid;
  for (size_t i = fill_start_idx; i < max_num_m_blocks; i += stride) {
    expert_ids[i] = 0;
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

    sorted_token_ids[rank_post_pad] = i;
    ++tokens_cnts[tid * num_experts + expert_id];
  }
}

template <typename scalar_t>
__global__ void count_and_sort_expert_tokens_kernel(
    const scalar_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids, int32_t* __restrict__ cumsum_buffer,
    int32_t* __restrict__ expert_map, size_t numel, int32_t num_experts,
    int32_t max_num_tokens_padded, int32_t topk_num, bool has_expert_map) {
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

    int32_t rank_post_pad = atomicAdd(&cumsum_buffer[expert_id], 1);
    sorted_token_ids[rank_post_pad] = i;
  }
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

template <typename scalar_t>
__global__ void moe_lora_align_block_size_kernel(
    const scalar_t* __restrict__ topk_ids,
    const int32_t* __restrict__ token_lora_mapping,
    const int32_t* __restrict__ lora_ids,
    const int32_t* __restrict__ adapter_enabled,
    int32_t* __restrict__ sorted_token_ids, int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ num_tokens_post_pad,
    const int32_t* __restrict__ expert_map, int32_t num_experts,
    int32_t num_virtual_experts, int32_t max_loras,
    int32_t padded_num_virtual_experts, int32_t experts_per_warp,
    int32_t block_size, size_t numel, int32_t* __restrict__ cumsum,
    int32_t max_num_tokens_padded, int32_t topk_num, bool has_expert_map) {
  const int32_t max_num_m_blocks = CEILDIV(max_num_tokens_padded, block_size);

  extern __shared__ int32_t
      shared_counts[];  // size = padded_num_virtual_experts

  // block 1: init sorted_token_ids
  if (blockIdx.x & 1) {
    for (size_t it = threadIdx.x; it < (size_t)max_num_tokens_padded;
         it += blockDim.x) {
      sorted_token_ids[it] = (int32_t)numel;  // sentinel
    }
    return;
  }

  // init shared_counts for padded_num_virtual_experts using linear indexing
  // This handles any size of padded_num_virtual_experts (including > 1024)
  for (int32_t e = threadIdx.x; e < padded_num_virtual_experts;
       e += blockDim.x) {
    shared_counts[e] = 0;
  }
  __syncthreads();

  const size_t tid = threadIdx.x;
  const size_t stride = blockDim.x;

  // count tokens per virtual expert
  for (size_t i = tid; i < numel; i += stride) {
    int32_t expert = (int32_t)topk_ids[i];

    // topk flattened indexing: token index = i / topk_num
    int32_t token_idx = (int32_t)(i / topk_num);
    int32_t lora = token_lora_mapping[token_idx];

    // reject invalid
    if (expert < 0 || expert >= num_experts) continue;
    if (lora < 0 || lora >= max_loras || adapter_enabled[lora] == 0) continue;

    // apply expert_map on physical expert id if needed
    if (has_expert_map) {
      expert = expert_map[expert];
      if (expert == -1) continue;
      // still physical expert range after mapping
      if (expert < 0 || expert >= num_experts) continue;
    }

    // virtual expert id in [0, num_virtual_experts)
    int32_t ve = lora * num_experts + expert;

    // safety
    if (ve < 0 || ve >= num_virtual_experts) continue;

    // Use linear indexing for shared_counts (simpler and supports large
    // num_virtual_experts)
    atomicAdd(&shared_counts[ve], 1);
  }

  __syncthreads();

  // =========================================================================
  // Iterative prefix sum for num_virtual_experts >= 1024
  // For num_virtual_experts > 1024, we iterate with stride 1024
  // =========================================================================
  using BlockScan = cub::BlockScan<int32_t, 1024>;
  __shared__ typename BlockScan::TempStorage temp_storage;
  __shared__ int32_t running_total;

  constexpr int32_t TILE_SIZE = 1024;
  const int32_t num_iterations = CEILDIV(num_virtual_experts, TILE_SIZE);

  // Initialize running total
  if (tid == 0) {
    running_total = 0;
  }
  __syncthreads();

  // Iterate over tiles of 1024 experts
  for (int32_t iter = 0; iter < num_iterations; ++iter) {
    const int32_t ve_start = iter * TILE_SIZE;
    const int32_t ve_idx = ve_start + (int32_t)tid;

    // Load expert count for this thread's expert (using linear indexing)
    int32_t expert_count = 0;
    if (ve_idx < num_virtual_experts) {
      expert_count = shared_counts[ve_idx];
      expert_count = CEILDIV(expert_count, block_size) * block_size;
    }

    // Perform block-wide exclusive prefix sum WITH AGGREGATE
    int32_t local_cumsum = 0;
    int32_t block_aggregate = 0;
    BlockScan(temp_storage)
        .ExclusiveSum(expert_count, local_cumsum, block_aggregate);

    // Sync to ensure temp_storage is done before we read running_total
    __syncthreads();

    // Write cumsum with offset from previous iterations
    if (ve_idx < num_virtual_experts) {
      cumsum[ve_idx] = running_total + local_cumsum;
    }

    // Update running total for next iteration (only thread 0)
    if (tid == 0) {
      running_total = running_total + block_aggregate;
    }

    // Sync before next iteration
    __syncthreads();
  }

  // Write the final cumsum[num_virtual_experts] and num_tokens_post_pad
  if (tid == 0) {
    cumsum[num_virtual_experts] = running_total;
    num_tokens_post_pad[0] = running_total;
  }

  __syncthreads();

  // =========================================================================
  // Fill expert_ids with stride loop for large num_virtual_experts
  // =========================================================================
  for (int32_t ve = (int32_t)tid; ve < num_virtual_experts;
       ve += (int32_t)blockDim.x) {
    for (int32_t i = cumsum[ve]; i < cumsum[ve + 1]; i += block_size) {
      expert_ids[i / block_size] = ve;
    }
  }

  __syncthreads();

  // fill remaining expert_ids with num_virtual_experts (sentinel/inactive)
  const size_t fill_start_idx =
      (size_t)(cumsum[num_virtual_experts] / block_size) + tid;
  for (size_t i = fill_start_idx; i < (size_t)max_num_m_blocks;
       i += blockDim.x) {
    expert_ids[i] = num_virtual_experts;
  }
}

template <typename scalar_t>
__global__ void moe_lora_count_and_sort_kernel(
    const scalar_t* __restrict__ topk_ids,
    const int32_t* __restrict__ token_lora_mapping,
    const int32_t* __restrict__ adapter_enabled,
    int32_t* __restrict__ sorted_token_ids, int32_t* __restrict__ cumsum_buffer,
    const int32_t* __restrict__ expert_map, size_t numel, int32_t num_experts,
    int32_t num_virtual_experts, int32_t max_loras,
    int32_t max_num_tokens_padded, int32_t topk_num, bool has_expert_map) {
  const size_t tid = blockIdx.y * blockDim.x + threadIdx.x;
  const size_t stride = blockDim.x * gridDim.y;

  for (size_t i = tid; i < numel; i += stride) {
    int32_t expert = (int32_t)topk_ids[i];
    if (expert < 0 || expert >= num_experts) continue;

    int32_t token_idx = (int32_t)(i / topk_num);
    int32_t lora = token_lora_mapping[token_idx];

    if (lora < 0 || lora >= max_loras || adapter_enabled[lora] == 0) continue;

    if (has_expert_map) {
      expert = expert_map[expert];
      if (expert == -1) continue;
      if (expert < 0 || expert >= num_experts) continue;
    }

    int32_t virtual_expert = lora * num_experts + expert;

    if (virtual_expert < 0 || virtual_expert >= num_virtual_experts) continue;

    int32_t rank_post_pad = atomicAdd(&cumsum_buffer[virtual_expert], 1);
    sorted_token_ids[rank_post_pad] = (int32_t)i;
  }
}

}  // namespace moe
}  // namespace vllm

// taken from
// https://github.com/sgl-project/sglang/blob/8b5f83ed3b7d2a49ad5c5cd5aa61c5d502f47dbc
void moe_align_block_size(torch::Tensor topk_ids, int64_t num_experts,
                          int64_t block_size, torch::Tensor sorted_token_ids,
                          torch::Tensor experts_ids,
                          torch::Tensor num_tokens_post_pad,
                          std::optional<torch::Tensor> maybe_expert_map) {
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int64_t padded_num_experts =
      ((num_experts + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
  int experts_per_warp = WARP_SIZE;
  int threads = 1024;
  threads = ((threads + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;

  // BlockScan uses 1024 threads and assigns one thread per expert.
  TORCH_CHECK(padded_num_experts < 1024,
              "padded_num_experts must be less than 1024");
  auto options_int =
      torch::TensorOptions().dtype(torch::kInt).device(topk_ids.device());
  bool has_expert_map = maybe_expert_map.has_value();
  torch::Tensor expert_map;
  if (has_expert_map) {
    expert_map = maybe_expert_map.value();
  } else {
    expert_map = torch::empty({0}, options_int);
  }

  VLLM_DISPATCH_INTEGRAL_AND_UNSIGNED_TYPES(
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
              topk_ids.data_ptr<scalar_t>(),
              sorted_token_ids.data_ptr<int32_t>(),
              experts_ids.data_ptr<int32_t>(),
              num_tokens_post_pad.data_ptr<int32_t>(),
              expert_map.data_ptr<int32_t>(), num_experts, block_size,
              topk_ids.numel(), sorted_token_ids.size(0), topk_ids.size(1),
              has_expert_map);
        } else {
          torch::Tensor cumsum_buffer =
              torch::empty({num_experts + 1}, options_int);
          auto align_kernel = vllm::moe::moe_align_block_size_kernel<scalar_t>;

          size_t num_warps = CEILDIV(padded_num_experts, experts_per_warp);
          size_t shared_mem_size =
              num_warps * experts_per_warp * sizeof(int32_t);

          // launch two threadblocks
          // blockIdx.x == 0: counting experts and aligning
          // blockIdx.x == 1: filling sorted_token_ids
          align_kernel<<<2, threads, shared_mem_size, stream>>>(
              topk_ids.data_ptr<scalar_t>(),
              sorted_token_ids.data_ptr<int32_t>(),
              experts_ids.data_ptr<int32_t>(),
              num_tokens_post_pad.data_ptr<int32_t>(),
              expert_map.data_ptr<int32_t>(), num_experts, padded_num_experts,
              experts_per_warp, block_size, topk_ids.numel(),
              cumsum_buffer.data_ptr<int32_t>(), sorted_token_ids.size(0),
              topk_ids.size(1), has_expert_map);

          const int block_threads = std::min(256, (int)threads);
          const int num_blocks =
              (topk_ids.numel() + block_threads - 1) / block_threads;
          const int max_blocks = 65535;
          const int actual_blocks = std::min(num_blocks, max_blocks);
          dim3 gridDims(1, actual_blocks);

          auto sort_kernel =
              vllm::moe::count_and_sort_expert_tokens_kernel<scalar_t>;
          sort_kernel<<<gridDims, block_threads, 0, stream>>>(
              topk_ids.data_ptr<scalar_t>(),
              sorted_token_ids.data_ptr<int32_t>(),
              cumsum_buffer.data_ptr<int32_t>(), expert_map.data_ptr<int32_t>(),
              topk_ids.numel(), num_experts, sorted_token_ids.size(0),
              topk_ids.size(1), has_expert_map);
        }
      });
}

void batched_moe_align_block_size(int64_t max_tokens_per_batch,
                                  int64_t block_size,
                                  torch::Tensor const& batch_num_tokens,
                                  torch::Tensor sorted_ids,
                                  torch::Tensor batch_ids,
                                  torch::Tensor num_tokens_post_pad) {
  namespace batched_kernel = vllm::moe::batched_moe_align_block_size;

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  int32_t const B = batch_num_tokens.size(0);
  int32_t const num_blocks_per_batch =
      round_to_next_multiple_of(max_tokens_per_batch, block_size) / block_size;
  int32_t const num_blocks = num_blocks_per_batch * B;
  int64_t const sorted_ids_size = num_blocks * block_size;

  TORCH_CHECK(sorted_ids.size(0) == sorted_ids_size);
  TORCH_CHECK(batch_ids.size(0) == sorted_ids_size / block_size);
  TORCH_CHECK(num_tokens_post_pad.size(0) == 1);
  TORCH_CHECK(B <= batched_kernel::num_threads);

  batched_kernel::batched_moe_align_block_size_kernel<<<
      batched_kernel::num_blocks, batched_kernel::num_threads, 0, stream>>>(
      B, max_tokens_per_batch, block_size, batch_num_tokens.data_ptr<int32_t>(),
      sorted_ids.data_ptr<int32_t>(), batch_ids.data_ptr<int32_t>(),
      num_tokens_post_pad.data_ptr<int32_t>());
}

void moe_sum(torch::Tensor& input,   // [num_tokens, topk, hidden_size]
             torch::Tensor& output)  // [num_tokens, hidden_size]
{
  const int hidden_size = input.size(-1);
  const auto num_tokens = output.numel() / hidden_size;
  const int topk = input.size(1);

  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, 1024));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(output));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  switch (topk) {
    case 2:
      VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "moe_sum_kernel", [&] {
        vllm::moe::moe_sum_kernel<scalar_t, 2><<<grid, block, 0, stream>>>(
            output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
            hidden_size);
      });
      break;

    case 3:
      VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "moe_sum_kernel", [&] {
        vllm::moe::moe_sum_kernel<scalar_t, 3><<<grid, block, 0, stream>>>(
            output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
            hidden_size);
      });
      break;

    case 4:
      VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "moe_sum_kernel", [&] {
        vllm::moe::moe_sum_kernel<scalar_t, 4><<<grid, block, 0, stream>>>(
            output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
            hidden_size);
      });
      break;

    default:
      at::sum_out(output, input, 1);
      break;
  }
}

void moe_lora_align_block_size(torch::Tensor topk_ids, torch::Tensor lora_ids,
                               torch::Tensor adapter_enabled,
                               torch::Tensor token_lora_mapping,
                               int64_t num_virtual_experts, int64_t max_loras,
                               int64_t block_size,
                               torch::Tensor sorted_token_ids,
                               torch::Tensor expert_ids,
                               torch::Tensor num_tokens_post_pad,
                               std::optional<torch::Tensor> maybe_expert_map) {
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int64_t num_experts = num_virtual_experts / max_loras;
  int64_t topk_num = topk_ids.size(1);

  int64_t padded_num_virtual_experts =
      ((num_virtual_experts + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;

  int experts_per_warp = WARP_SIZE;

  int threads = 1024;
  threads = ((threads + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;

  // No longer limited to 1024 - iterative BlockScan handles larger counts
  // Add upper bound for sanity (shared memory limit)
  TORCH_CHECK(num_virtual_experts <= 65536,
              "num_virtual_experts must be <= 65536");

  auto options_int =
      torch::TensorOptions().dtype(torch::kInt).device(topk_ids.device());

  bool has_expert_map = maybe_expert_map.has_value();
  torch::Tensor expert_map = has_expert_map ? maybe_expert_map.value()
                                            : torch::empty({0}, options_int);

  // IMPORTANT: cumsum must be sized by num_virtual_experts + 1
  torch::Tensor cumsum_buffer =
      torch::zeros({num_virtual_experts + 1}, options_int);

  VLLM_DISPATCH_INTEGRAL_AND_UNSIGNED_TYPES(
      topk_ids.scalar_type(), "moe_lora_align_block_size_kernel", [&] {
        auto align_kernel =
            vllm::moe::moe_lora_align_block_size_kernel<scalar_t>;

        size_t num_warps =
            CEILDIV(padded_num_virtual_experts, experts_per_warp);
        size_t shared_mem_size = num_warps * experts_per_warp * sizeof(int32_t);

        // Check shared memory limits for large num_virtual_experts
        int device_max_shared_mem;
        cudaDeviceGetAttribute(&device_max_shared_mem,
                               cudaDevAttrMaxSharedMemoryPerBlockOptin,
                               topk_ids.get_device());
        TORCH_CHECK(shared_mem_size <= (size_t)device_max_shared_mem,
                    "Required shared memory (", shared_mem_size,
                    " bytes) exceeds device limit (", device_max_shared_mem,
                    " bytes)");

        // 2 blocks: [0]=count+align, [1]=init sorted_token_ids
        align_kernel<<<2, threads, shared_mem_size, stream>>>(
            topk_ids.data_ptr<scalar_t>(),
            token_lora_mapping.data_ptr<int32_t>(),
            lora_ids.data_ptr<int32_t>(), adapter_enabled.data_ptr<int32_t>(),
            sorted_token_ids.data_ptr<int32_t>(),
            expert_ids.data_ptr<int32_t>(),
            num_tokens_post_pad.data_ptr<int32_t>(),
            expert_map.data_ptr<int32_t>(), (int32_t)num_experts,
            (int32_t)num_virtual_experts, (int32_t)max_loras,
            (int32_t)padded_num_virtual_experts, (int32_t)experts_per_warp,
            (int32_t)block_size, (size_t)topk_ids.numel(),
            cumsum_buffer.data_ptr<int32_t>(),
            (int32_t)sorted_token_ids.size(0), (int32_t)topk_num,
            has_expert_map);

        // Launch sorting kernel
        const int block_threads = std::min(256, threads);
        const int num_blocks =
            ((int)topk_ids.numel() + block_threads - 1) / block_threads;
        const int max_blocks = 65535;
        const int actual_blocks = std::min(num_blocks, max_blocks);
        dim3 gridDims(1, actual_blocks);

        auto sort_kernel = vllm::moe::moe_lora_count_and_sort_kernel<scalar_t>;
        sort_kernel<<<gridDims, block_threads, 0, stream>>>(
            topk_ids.data_ptr<scalar_t>(),
            token_lora_mapping.data_ptr<int32_t>(),
            adapter_enabled.data_ptr<int32_t>(),
            sorted_token_ids.data_ptr<int32_t>(),
            cumsum_buffer.data_ptr<int32_t>(), expert_map.data_ptr<int32_t>(),
            (size_t)topk_ids.numel(), (int32_t)num_experts,
            (int32_t)num_virtual_experts, (int32_t)max_loras,
            (int32_t)sorted_token_ids.size(0), (int32_t)topk_num,
            has_expert_map);
      });
}
