#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <ATen/ATen.h>
#include <ATen/cuda/Atomic.cuh>

#include "../cuda_compat.h"
#include "../dispatch_utils.h"
#include "core/math.hpp"

namespace {

__device__ __forceinline__ int32_t index(int32_t total_col, int32_t row,
                                         int32_t col) {
  return row * total_col + col;
}

}  // namespace

// TODO: Refactor common parts with moe_align_sum_kernels
template <typename scalar_t, typename token_cnts_t>
__global__ void moe_lora_align_sum_kernel(
    scalar_t* __restrict__ topk_ids, int32_t* token_lora_mapping,
    int64_t block_size, int num_experts, int max_loras, size_t numel,
    int max_num_tokens_padded, int max_num_m_blocks,
    int32_t* __restrict__ sorted_token_ids, int32_t* __restrict__ expert_ids,
    int topk_num, int32_t* total_tokens_post_pad, int32_t* adapter_enabled,
    int32_t* lora_ids) {
  const size_t tokens_per_thread = div_ceil(numel, blockDim.x);
  const size_t start_idx = threadIdx.x * tokens_per_thread;

  int lora_idx = blockIdx.x;
  int lora_id = lora_ids[lora_idx];
  if (lora_id == -1 || adapter_enabled[lora_id] == 0) {
    return;
  }
  extern __shared__ int32_t shared_mem[];
  int32_t* cumsum = shared_mem;
  token_cnts_t* tokens_cnts = (token_cnts_t*)(shared_mem + num_experts + 1);

  // Initialize sorted_token_ids with numel
  for (size_t it = threadIdx.x; it < max_num_tokens_padded; it += blockDim.x) {
    sorted_token_ids[lora_id * max_num_tokens_padded + it] = numel;
  }

  // Initialize expert_ids with -1
  for (size_t it = threadIdx.x; it < max_num_m_blocks; it += blockDim.x) {
    expert_ids[lora_id * max_num_m_blocks + it] = -1;
  }

  // Initialize total_tokens_post_pad with 0
  if (threadIdx.x == 0) {
    total_tokens_post_pad[lora_id] = 0;
  }

  for (int i = 0; i < num_experts; ++i) {
    tokens_cnts[index(num_experts, threadIdx.x + 1, i)] = 0;
  }

  for (int i = start_idx; i < numel && i < start_idx + tokens_per_thread; ++i) {
    int mask = token_lora_mapping[i / topk_num] == lora_id;
    int idx = index(num_experts, threadIdx.x + 1, topk_ids[i]);
    tokens_cnts[idx] += mask;
  }

  __syncthreads();

  // For each expert we accumulate the token counts from the different threads.
  if (threadIdx.x < num_experts) {
    tokens_cnts[index(num_experts, 0, threadIdx.x)] = 0;
    for (int i = 1; i <= blockDim.x; ++i) {
      tokens_cnts[index(num_experts, i, threadIdx.x)] +=
          tokens_cnts[index(num_experts, i - 1, threadIdx.x)];
    }
  }

  __syncthreads();

  // We accumulate the token counts of all experts in thread 0.
  if (threadIdx.x == 0) {
    cumsum[0] = 0;
    for (int i = 1; i <= num_experts; ++i) {
      cumsum[i] = cumsum[i - 1] +
                  div_ceil(tokens_cnts[index(num_experts, blockDim.x, i - 1)],
                           block_size) *
                      block_size;
    }
    total_tokens_post_pad[lora_id] = static_cast<int32_t>(cumsum[num_experts]);
  }

  __syncthreads();

  /**
   * For each expert, each thread processes the tokens of the corresponding
   * blocks and stores the corresponding expert_id for each block.
   */
  if (threadIdx.x < num_experts) {
    for (int i = cumsum[threadIdx.x]; i < cumsum[threadIdx.x + 1];
         i += block_size) {
      expert_ids[index(max_num_m_blocks, lora_id, i / block_size)] =
          threadIdx.x;
    }
  }

  for (int i = start_idx; i < numel && i < start_idx + tokens_per_thread; ++i) {
    int32_t expert_id = topk_ids[i];
    /** The cumsum[expert_id] stores the starting index of the tokens that the
     * expert with expert_id needs to process, and
     * tokens_cnts[threadIdx.x][expert_id] stores the indices of the tokens
     * processed by the expert with expert_id within the current thread's token
     * shard.
     */
    int32_t rank_post_pad =
        tokens_cnts[index(num_experts, threadIdx.x, expert_id)] +
        cumsum[expert_id];

    int mask = (int)token_lora_mapping[i / topk_num] == lora_id;
    atomicAdd(
        &sorted_token_ids[index(max_num_tokens_padded, lora_id, rank_post_pad)],
        (i - numel) * mask);
    tokens_cnts[index(num_experts, threadIdx.x, expert_id)] += mask;
  }
}

void moe_lora_align_block_size(
    torch::Tensor topk_ids, torch::Tensor token_lora_mapping,
    int64_t num_experts, int64_t block_size, int64_t max_loras,
    int64_t max_num_tokens_padded, int64_t max_num_m_blocks,
    torch::Tensor sorted_token_ids, torch::Tensor expert_ids,
    torch::Tensor num_tokens_post_pad, torch::Tensor adapter_enabled,
    torch::Tensor lora_ids) {
  const int topk_num = topk_ids.size(1);

  TORCH_CHECK(block_size > 0, "block_size should be greater than 0. ");

  int device_max_shared_mem;
  auto dev = topk_ids.get_device();
  cudaDeviceGetAttribute(&device_max_shared_mem,
                         cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int32_t num_thread = max((int32_t)num_experts, 128);  // WARP_SIZE,
  TORCH_CHECK(num_thread <= 1024,
              "num_thread must be less than 1024, "
              "and fallback is not implemented yet.");
  const int32_t shared_mem = (num_thread + 1) * num_experts * sizeof(int32_t) +
                             (num_experts + 1) * sizeof(int32_t);

  if (shared_mem > device_max_shared_mem) {
    TORCH_CHECK(false,
                "Shared memory usage exceeds device limit, and global memory "
                "fallback is not implemented yet.");
  }

  VLLM_DISPATCH_INTEGRAL_TYPES(
      topk_ids.scalar_type(), "moe_lora_align_sum_kernel", [&] {
        dim3 blockDim(num_thread);
        auto kernel = moe_lora_align_sum_kernel<scalar_t, int32_t>;
        AT_CUDA_CHECK(VLLM_DevFuncAttribute_SET_MaxDynamicSharedMemorySize(
            (void*)kernel, shared_mem));
        kernel<<<max_loras, blockDim, shared_mem, stream>>>(
            topk_ids.data_ptr<scalar_t>(),
            token_lora_mapping.data_ptr<int32_t>(), block_size, num_experts,
            max_loras, topk_ids.numel(), max_num_tokens_padded,
            max_num_m_blocks, sorted_token_ids.data_ptr<int32_t>(),
            expert_ids.data_ptr<int32_t>(), topk_num,
            num_tokens_post_pad.data_ptr<int32_t>(),
            adapter_enabled.data_ptr<int32_t>(), lora_ids.data_ptr<int32_t>());
      });
}