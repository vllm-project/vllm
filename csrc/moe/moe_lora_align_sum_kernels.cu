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

#define CEILDIV(x, y) (((x) + (y) - 1) / (y))

namespace {
__device__ __forceinline__ int32_t dim3_index(int32_t page, int32_t total_row,
                                              int32_t total_col, int32_t row,
                                              int32_t col) {
  return page * total_row * total_col + row * total_col + col;
}

__device__ __forceinline__ int32_t dim2_index(int32_t total_col, int32_t row,
                                              int32_t col) {
  return row * total_col + col;
}

}  // namespace

int round_up(int value, int multiple) {
  if (multiple == 0) return value;
  return ((value + multiple - 1) / multiple) * multiple;
}

template <typename scalar_t, typename token_cnts_t>
__global__ void moe_lora_align_sum_kernel(
    scalar_t* __restrict__ topk_ids, scalar_t* __restrict__ token_lora_mapping,
    int64_t block_size, int num_experts, int max_loras, size_t numel,
    int max_num_tokens_padded, int max_num_m_blocks,
    int32_t* __restrict__ sorted_token_ids, int32_t* __restrict__ expert_ids,
    int topk_num, int32_t* total_tokens_post_pad) {  //
  int tid = threadIdx.x;

  const size_t tokens_per_thread = CEILDIV(numel, blockDim.x);
  const size_t start_idx = threadIdx.x * tokens_per_thread;

  int lora_id = threadIdx.y;
  extern __shared__ int32_t shared_mem[];
  int32_t* cumsum =
      shared_mem;  // 2d tensor with shape (max_loras, num_experts + 1)
  token_cnts_t* tokens_cnts =
      (token_cnts_t*)(shared_mem +
                      (num_experts + 1) *
                          max_loras);  // 3d tensor with shape (max_loras,
                                       // blockDim.x + 1, num_experts)

  for (int i = 0; i < num_experts; ++i) {
    tokens_cnts[dim3_index(lora_id, blockDim.x + 1, num_experts,
                           threadIdx.x + 1, i)] = 0;
  }

  for (int i = start_idx; i < numel && i < start_idx + tokens_per_thread; ++i) {
    // if (token_lora_mapping[i%topk_num]==lora_id){
    //   ++tokens_cnts[dim3_index(lora_id,blockDim.x+1,num_experts, threadIdx.x
    //   + 1, topk_ids[i])];
    // }

    int mask = token_lora_mapping[i / topk_num] == lora_id;
    int idx = dim3_index(lora_id, blockDim.x + 1, num_experts, threadIdx.x + 1,
                         topk_ids[i]);
    // atomicAdd(&tokens_cnts[idx], mask);
    tokens_cnts[idx] += mask;
  }

  __syncthreads();

  // For each expert we accumulate the token counts from the different threads.
  if (threadIdx.x < num_experts) {
    tokens_cnts[dim3_index(lora_id, blockDim.x + 1, num_experts, 0,
                           threadIdx.x)] = 0;
    for (int i = 1; i <= blockDim.x; ++i) {
      tokens_cnts[dim3_index(lora_id, blockDim.x + 1, num_experts, i,
                             threadIdx.x)] +=
          tokens_cnts[dim3_index(lora_id, blockDim.x + 1, num_experts, i - 1,
                                 threadIdx.x)];
    }
  }

  __syncthreads();

  // We accumulate the token counts of all experts in thread 0.
  if (threadIdx.x == 0) {
    cumsum[dim2_index(blockDim.x + 1, lora_id, 0)] = 0;
    for (int i = 1; i <= num_experts; ++i) {
      cumsum[dim2_index(blockDim.x + 1, lora_id, i)] =
          cumsum[dim2_index(blockDim.x + 1, lora_id, i - 1)] +
          CEILDIV(tokens_cnts[dim3_index(lora_id, blockDim.x + 1, num_experts,
                                         blockDim.x, i - 1)],
                  block_size) *
              block_size;
    }
    total_tokens_post_pad[lora_id] = static_cast<int32_t>(
        cumsum[dim2_index(blockDim.x + 1, lora_id, num_experts)]);
  }

  __syncthreads();

  /**
   * For each expert, each thread processes the tokens of the corresponding
   * blocks and stores the corresponding expert_id for each block.ßßß
   */
  if (threadIdx.x < num_experts) {
    for (int i = cumsum[dim2_index(blockDim.x + 1, lora_id, threadIdx.x)];
         i < cumsum[dim2_index(blockDim.x + 1, lora_id, threadIdx.x + 1)];
         i += block_size) {
      expert_ids[dim2_index(max_num_m_blocks, lora_id, i / block_size)] =
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
        tokens_cnts[dim3_index(lora_id, blockDim.x + 1, num_experts,
                               threadIdx.x, expert_id)] +
        cumsum[dim2_index(blockDim.x + 1, lora_id, expert_id)];
    int mask = token_lora_mapping[i / topk_num] == lora_id;
    // sorted_token_ids[dim2_index(max_num_tokens_padded,lora_id,rank_post_pad)]
    // = i;
    // ++tokens_cnts[dim3_index(lora_id,blockDim.x+1,num_experts, threadIdx.x,
    // expert_id)];
    sorted_token_ids[dim2_index(max_num_tokens_padded, lora_id,
                                rank_post_pad)] =
        sorted_token_ids[dim2_index(max_num_tokens_padded, lora_id,
                                    rank_post_pad)] *
            (1 - mask) +
        i * mask;
    tokens_cnts[dim3_index(lora_id, blockDim.x + 1, num_experts, threadIdx.x,
                           expert_id)] += mask;
  }
}

void moe_lora_align_block_size(torch::Tensor topk_ids,
                               torch::Tensor token_lora_mapping,
                               int64_t num_experts, int64_t block_size,
                               int64_t max_loras,
                               torch::Tensor sorted_token_ids,
                               torch::Tensor expert_ids,
                               torch::Tensor num_tokens_post_pad) {
  // const int topk_num = 6;
  const int topk_num = topk_ids.size(1);

  int max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1);
  max_num_tokens_padded = round_up(max_num_tokens_padded, block_size);
  int max_num_m_blocks = CEILDIV(max_num_tokens_padded, block_size);

  int device_max_shared_mem;
  auto dev = topk_ids.get_device();
  cudaDeviceGetAttribute(&device_max_shared_mem,
                         cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int32_t num_thread = max((int32_t)num_experts, WARP_SIZE);

  const int32_t shared_mem_i16 =
      ((num_thread + 1) * num_experts) * max_loras * sizeof(uint16_t) +
      (num_experts + 1) * max_loras * sizeof(int32_t);

  const int32_t shared_mem_i32 = ((num_experts + 1) * num_experts * max_loras +
                                  (num_experts + 1) * max_loras) *
                                 sizeof(int32_t);

  dim3 blockDim(num_experts, max_loras);

  bool use_global_memory = false;
  bool use_i16 = false;  // Use uint16_t for shared memory token counts
  if (shared_mem_i32 < device_max_shared_mem) {
    // Do nothing in this case. We're all set to use int32_t token counts
  } else if (shared_mem_i16 <
             device_max_shared_mem) {  //&& topk_ids.numel() <= 65535
    // when nelements of topk_ids is smaller than 65535 (max value of uint16),
    // element value of token_cnts would also smaller than 65535,
    // so we can use uint16 as dtype of token_cnts
    use_i16 = true;
  } else {
    use_global_memory = true;
  }

  // TODO
  //  if (use_i16) {

  // }else{
  //     moe_lora_align_sum_kernel<<<1,
  //     blockDim,shared_mem_i32,stream>>>(d_topk_ids,
  //     d_token_lora_mapping,num_experts,max_loras);
  // }

  VLLM_DISPATCH_INTEGRAL_TYPES(
      topk_ids.scalar_type(), "moe_lora_align_sum_kernel", [&] {
        auto kernel = moe_lora_align_sum_kernel<scalar_t, uint16_t>;
        int32_t shared_mem_i32a = ((num_experts + 1) * num_experts * max_loras +
                                   (num_experts + 1) * max_loras) *
                                  sizeof(int32_t);
        AT_CUDA_CHECK(VLLM_DevFuncAttribute_SET_MaxDynamicSharedMemorySize(
            (void*)kernel, shared_mem_i16));
        kernel<<<1, blockDim, shared_mem_i16, stream>>>(
            topk_ids.data_ptr<scalar_t>(),
            token_lora_mapping.data_ptr<scalar_t>(), block_size, num_experts,
            max_loras, topk_ids.numel(), max_num_tokens_padded,
            max_num_m_blocks, sorted_token_ids.data_ptr<int32_t>(),
            expert_ids.data_ptr<int32_t>(), topk_num,
            num_tokens_post_pad.data_ptr<int32_t>());
      });
}
