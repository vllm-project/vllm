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
    int32_t* __restrict__ total_tokens_post_pad, int32_t num_experts,
    int32_t padded_num_experts, int32_t experts_per_warp, int32_t block_size,
    size_t numel, int32_t* __restrict__ cumsum, int32_t max_num_tokens_padded) {
  extern __shared__ int32_t shared_counts[];

  // Initialize sorted_token_ids with numel
  for (size_t it = threadIdx.x; it < max_num_tokens_padded; it += blockDim.x) {
    sorted_token_ids[it] = numel;
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
    *total_tokens_post_pad = cumsum_val;
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
  const size_t expert_ids_size = CEILDIV(max_num_tokens_padded, block_size);
  for (size_t i = fill_start_idx; i < expert_ids_size; i += blockDim.x) {
    expert_ids[i] = 0;
  }
}

template <typename scalar_t>
__global__ void count_and_sort_expert_tokens_kernel(
    const scalar_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids, int32_t* __restrict__ cumsum_buffer,
    size_t numel, int32_t num_experts) {
  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = blockDim.x * gridDim.x;

  for (size_t i = tid; i < numel; i += stride) {
    int32_t expert_id = topk_ids[i];
    if (expert_id >= num_experts) {
      continue;
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
__global__ void moe_align_block_size_small_batch_expert_kernel(
    const scalar_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids, int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ total_tokens_post_pad, int32_t num_experts,
    int32_t block_size, size_t numel, int32_t max_num_tokens_padded) {
  // Initialize sorted_token_ids with numel
  for (size_t it = threadIdx.x; it < max_num_tokens_padded; it += blockDim.x) {
    sorted_token_ids[it] = numel;
  }

  const size_t tid = threadIdx.x;
  const size_t stride = blockDim.x;

  extern __shared__ int32_t shared_mem[];
  int32_t* cumsum = shared_mem;
  int32_t* tokens_cnts = (int32_t*)(shared_mem + num_experts + 1);

  for (int i = 0; i < num_experts; ++i) {
    tokens_cnts[(threadIdx.x + 1) * num_experts + i] = 0;
  }

  for (size_t i = tid; i < numel; i += stride) {
    ++tokens_cnts[(threadIdx.x + 1) * num_experts + topk_ids[i]];
  }

  __syncthreads();

  if (threadIdx.x < num_experts) {
    tokens_cnts[threadIdx.x] = 0;
    for (int i = 1; i <= blockDim.x; ++i) {
      tokens_cnts[i * num_experts + threadIdx.x] +=
          tokens_cnts[(i - 1) * num_experts + threadIdx.x];
    }
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    cumsum[0] = 0;
    for (int i = 1; i <= num_experts; ++i) {
      cumsum[i] =
          cumsum[i - 1] +
          CEILDIV(tokens_cnts[blockDim.x * num_experts + i - 1], block_size) *
              block_size;
    }
    *total_tokens_post_pad = static_cast<int32_t>(cumsum[num_experts]);
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
  const size_t expert_ids_size = CEILDIV(max_num_tokens_padded, block_size);
  for (size_t i = fill_start_idx; i < expert_ids_size; i += blockDim.x) {
    expert_ids[i] = 0;
  }

  for (size_t i = tid; i < numel; i += stride) {
    int32_t expert_id = topk_ids[i];
    int32_t rank_post_pad =
        tokens_cnts[threadIdx.x * num_experts + expert_id] + cumsum[expert_id];
    sorted_token_ids[rank_post_pad] = i;
    ++tokens_cnts[threadIdx.x * num_experts + expert_id];
  }
}

}  // namespace moe
}  // namespace vllm

// taken from
// https://github.com/sgl-project/sglang/blob/8b5f83ed3b7d2a49ad5c5cd5aa61c5d502f47dbc
void moe_align_block_size(torch::Tensor topk_ids, int64_t num_experts,
                          int64_t block_size, torch::Tensor sorted_token_ids,
                          torch::Tensor experts_ids,
                          torch::Tensor num_tokens_post_pad) {
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int64_t padded_num_experts =
      ((num_experts + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
  int experts_per_warp = WARP_SIZE;
  int threads = 1024;
  threads = ((threads + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;

  // BlockScan uses 1024 threads and assigns one thread per expert.
  TORCH_CHECK(padded_num_experts < 1024,
              "padded_num_experts must be less than 1024");

  VLLM_DISPATCH_INTEGRAL_AND_UNSIGNED_TYPES(
      topk_ids.scalar_type(), "moe_align_block_size_kernel", [&] {
        // calc needed amount of shared mem for `cumsum` tensors
        auto options_int =
            torch::TensorOptions().dtype(torch::kInt).device(topk_ids.device());
        torch::Tensor cumsum_buffer =
            torch::empty({num_experts + 1}, options_int);
        bool small_batch_expert_mode =
            (topk_ids.numel() < 1024) && (num_experts <= 64);

        if (small_batch_expert_mode) {
          const int32_t threads = max((int32_t)num_experts, WARP_SIZE);
          const int32_t shared_mem_size =
              ((threads + 1) * num_experts + (num_experts + 1)) *
              sizeof(int32_t);

          auto small_batch_expert_kernel =
              vllm::moe::moe_align_block_size_small_batch_expert_kernel<
                  scalar_t>;
          small_batch_expert_kernel<<<1, threads, shared_mem_size, stream>>>(
              topk_ids.data_ptr<scalar_t>(),
              sorted_token_ids.data_ptr<int32_t>(),
              experts_ids.data_ptr<int32_t>(),
              num_tokens_post_pad.data_ptr<int32_t>(), num_experts, block_size,
              topk_ids.numel(), sorted_token_ids.size(0));
        } else {
          auto align_kernel = vllm::moe::moe_align_block_size_kernel<scalar_t>;

          size_t num_warps = CEILDIV(padded_num_experts, experts_per_warp);
          size_t shared_mem_size =
              num_warps * experts_per_warp * sizeof(int32_t);

          align_kernel<<<1, threads, shared_mem_size, stream>>>(
              topk_ids.data_ptr<scalar_t>(),
              sorted_token_ids.data_ptr<int32_t>(),
              experts_ids.data_ptr<int32_t>(),
              num_tokens_post_pad.data_ptr<int32_t>(), num_experts,
              padded_num_experts, experts_per_warp, block_size,
              topk_ids.numel(), cumsum_buffer.data_ptr<int32_t>(),
              sorted_token_ids.size(0));

          const int block_threads = std::min(256, (int)threads);
          const int num_blocks =
              (topk_ids.numel() + block_threads - 1) / block_threads;
          const int max_blocks = 65535;
          const int actual_blocks = std::min(num_blocks, max_blocks);

          auto sort_kernel =
              vllm::moe::count_and_sort_expert_tokens_kernel<scalar_t>;
          sort_kernel<<<actual_blocks, block_threads, 0, stream>>>(
              topk_ids.data_ptr<scalar_t>(),
              sorted_token_ids.data_ptr<int32_t>(),
              cumsum_buffer.data_ptr<int32_t>(), topk_ids.numel(), num_experts);
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
