/*
 * Modified by Neural Magic
 * Copyright (C) Marlin.2024 Elias Frantar
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <torch/all.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <iostream>

#include "core/exception.hpp"
#include "core/scalar_type.hpp"
#include "marlin_kernels/marlin_moe_kernel_ku4b8.h"
#include "marlin_kernels/marlin_moe_kernel_ku8b128.h"

template <typename T>
inline std::string str(T x) {
  return std::to_string(x);
}

namespace marlin_moe {

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800

// For a given "a" of size [M,K] performs a permutation of the K columns based
// on the given "perm" indices.
__global__ void permute_cols_kernel(int4 const* __restrict__ a_int4_ptr,
                                    int const* __restrict__ perm_int_ptr,
                                    int4* __restrict__ out_int4_ptr, int size_m,
                                    int size_k, int block_rows) {
  int start_row = block_rows * blockIdx.x;
  int finish_row = start_row + block_rows;
  if (finish_row > size_m) {
    finish_row = size_m;
  }
  int cur_block_rows = finish_row - start_row;

  int row_stride = size_k * sizeof(half) / 16;

  auto permute_row = [&](int row) {
    int iters = size_k / blockDim.x;
    int rest = size_k % blockDim.x;

    int offset = row * row_stride;

    half const* a_row_half = reinterpret_cast<half const*>(a_int4_ptr + offset);
    half* out_half = reinterpret_cast<half*>(out_int4_ptr + offset);

    int base_k = 0;

    for (int i = 0; i < iters; i++) {
      int cur_k = base_k + threadIdx.x;
      int src_pos = perm_int_ptr[cur_k];

      out_half[cur_k] = a_row_half[src_pos];

      base_k += blockDim.x;
    }

    if (rest) {
      if (threadIdx.x < rest) {
        int cur_k = base_k + threadIdx.x;
        int src_pos = perm_int_ptr[cur_k];

        out_half[cur_k] = a_row_half[src_pos];
      }
    }
  };

  for (int i = 0; i < cur_block_rows; i++) {
    int cur_row = start_row + i;
    if (cur_row < size_m) {
      permute_row(cur_row);
    }
  }
}

__global__ void compute_expert_offsets(int const* __restrict__ topk_ids,
                                       int* __restrict__ expert_offsets,
                                       int topk_length, int block_size) {
  int expert_id = threadIdx.x;
  int num_experts = blockDim.x;

  int occurrences = 0;
  for (int i = 0; i < topk_length; ++i) {
    occurrences += (topk_ids[i] == expert_id);
  }
  expert_offsets[expert_id + 1] = occurrences;
  __syncthreads();

  if (threadIdx.x == 0) {
    int tot_offset = 0;
    expert_offsets[0] = 0;
    for (int i = 0; i < num_experts; ++i) {
      tot_offset += ceildiv(expert_offsets[i + 1], block_size) * block_size;
      expert_offsets[i + 1] = tot_offset;
    }
  }
  __syncthreads();
}

#else

__global__ void permute_cols_kernel(int4 const* __restrict__ a_int4_ptr,
                                    int const* __restrict__ perm_int_ptr,
                                    int4* __restrict__ out_int4_ptr, int size_m,
                                    int size_k, int block_rows) {
  // Marlin is not implemented yet for SM < 8.0
  assert(false);
  return;
}

__global__ void compute_expert_offsets(int const* __restrict__ topk_ids,
                                       int* __restrict__ expert_offsets,
                                       int topk_length, int block_size) {
  // Marlin is not implemented yet for SM < 8.0
  assert(false);
  return;
}

#endif

typedef struct {
  int thread_k;
  int thread_n;
  int num_threads;
} thread_config_t;

typedef struct {
  int max_m_blocks;
  thread_config_t tb_cfg;
} exec_config_t;

thread_config_t small_batch_thread_configs[] = {
    // Ordered by priority

    // thread_k, thread_n, num_threads
    {128, 128, 256},  // Default
    {128, 64, 128},   // Reduce N 2X, same K
    {64, 256, 256},   // Reduce K 2X, increase N 2X
    {64, 128, 128},   // Reduce K 2X, same N
};

thread_config_t large_batch_thread_configs[] = {
    // Ordered by priority

    // thread_k, thread_n, num_threads
    {64, 256, 256},   // Default
    {128, 128, 256},  // Reduce N 2X, increase K 2X
    {64, 128, 128},   // Reduce N 2X, same K
    {128, 64, 128},   // Reduce N 4X, increase K 2X
};

int get_scales_cache_size(thread_config_t const& th_config, int prob_m,
                          int prob_n, int prob_k, int num_bits, int group_size,
                          bool has_act_order, bool is_k_full) {
  bool cache_scales_chunk = has_act_order && !is_k_full;

  int tb_n = th_config.thread_n;
  int tb_k = th_config.thread_k;

  // Get max scale groups per thread-block
  int tb_groups;
  if (group_size == -1) {
    tb_groups = 1;
  } else if (group_size == 0) {
    tb_groups = ceildiv(tb_k, 32);  // Worst case is 32 group size
  } else {
    tb_groups = ceildiv(tb_k, group_size);
  }

  if (cache_scales_chunk) {
    int load_groups =
        tb_groups * STAGES * 2;          // Chunk size is 2x pipeline over dim K
    load_groups = max(load_groups, 32);  // We load at least 32 scale groups
    return load_groups * tb_n * 4;

  } else {
    int tb_scales = tb_groups * tb_n * 2;

    return tb_scales * STAGES;
  }
}

bool is_valid_cache_size(thread_config_t const& th_config, int max_m_blocks,
                         int prob_m, int prob_n, int prob_k, int num_bits,
                         int scales_cache_size, int max_shared_mem) {
  int pack_factor = 32 / num_bits;

  // Get B size
  int tb_k = th_config.thread_k;
  int tb_n = th_config.thread_n;

  int b_size = (tb_k * tb_n / pack_factor) * 4;

  // Get A size
  int m_blocks = ceildiv(prob_m, 16);
  int tb_max_m = 16;

  while (true) {
    if (m_blocks >= max_m_blocks) {
      tb_max_m *= max_m_blocks;
      break;
    }

    max_m_blocks--;
    if (max_m_blocks == 0) {
      TORCH_CHECK(false, "Unexpected m_blocks = ", m_blocks);
    }
  }

  int a_size = (tb_max_m * tb_k) * 2;

  float pipe_size = (a_size + b_size) * STAGES;

  TORCH_CHECK(max_shared_mem / 2 > scales_cache_size);  // Sanity

  return pipe_size < 0.95f * (max_shared_mem - scales_cache_size);
}

bool is_valid_config(thread_config_t const& th_config, int max_m_blocks,
                     int prob_m, int prob_n, int prob_k, int num_bits,
                     int group_size, bool has_act_order, bool is_k_full,
                     int max_shared_mem) {
  // Sanity
  if (th_config.thread_k == -1 || th_config.thread_n == -1 ||
      th_config.num_threads == -1) {
    return false;
  }

  // Verify K/N are divisible by thread K/N
  if (prob_k % th_config.thread_k != 0 || prob_n % th_config.thread_n != 0) {
    return false;
  }

  // thread_k can be only 128 or 64 (because it must be less than groupsize
  // which is 128)
  if (th_config.thread_k != 128 && th_config.thread_k != 64) {
    return false;
  }

  // Verify min for thread K/N
  if (th_config.thread_n < min_thread_n || th_config.thread_k < min_thread_k) {
    return false;
  }

  // num_threads must be at least 128 (= 4 warps)
  if (th_config.num_threads < 128) {
    return false;
  }

  //  Determine cache for scales
  int scales_cache_size =
      get_scales_cache_size(th_config, prob_m, prob_n, prob_k, num_bits,
                            group_size, has_act_order, is_k_full);

  // Check that pipeline fits into cache
  if (!is_valid_cache_size(th_config, max_m_blocks, prob_m, prob_n, prob_k,
                           num_bits, scales_cache_size, max_shared_mem)) {
    return false;
  }

  return true;
}

exec_config_t determine_thread_config(int prob_m, int prob_n, int prob_k,
                                      int num_bits, int group_size,
                                      bool has_act_order, bool is_k_full,
                                      int max_shared_mem) {
  int max_m_blocks = 4;
  while (max_m_blocks > 0) {
    if (prob_m <= 16) {
      for (auto th_config : small_batch_thread_configs) {
        if (is_valid_config(th_config, max_m_blocks, prob_m, prob_n, prob_k,
                            num_bits, group_size, has_act_order, is_k_full,
                            max_shared_mem)) {
          return exec_config_t{max_m_blocks, th_config};
        }
      }
    } else {
      for (auto th_config : large_batch_thread_configs) {
        if (is_valid_config(th_config, max_m_blocks, prob_m, prob_n, prob_k,
                            num_bits, group_size, has_act_order, is_k_full,
                            max_shared_mem)) {
          return exec_config_t{max_m_blocks, th_config};
        }
      }
    }

    max_m_blocks--;  // Process less M blocks per invocation to reduce cache
                     // usage
  }

  return exec_config_t{0, {-1, -1, -1}};
}

#define CALL_MOE_KERNEL_FUNCTION(KERNEL_FUNCTION)                              \
  else if (KERNEL_FUNCTION(q_type, thread_n_blocks, thread_k_blocks,           \
                           has_act_order, group_blocks, num_threads, blocks,   \
                           max_shared_mem, stream, A_ptr, B_ptr, C_ptr,        \
                           sorted_ids_ptr, topk_weights_ptr, s_ptr, g_idx_ptr, \
                           expert_offsets_ptr, num_groups, expert_idx,         \
                           num_experts, topk, prob_m, prob_n, prob_k, tot_m,   \
                           locks, replicate_input, apply_weights, m_block,     \
                           max_par, exec_cfg.max_m_blocks)) {                  \
  }

void marlin_mm_moe(const void* A, const void* B, void* C,
                   const void* sorted_ids, const void* topk_weights,
                   const void* topk_ids, const void* s, const void* g_idx,
                   const void* perm, void* a_tmp, void* expert_offsets,
                   int prob_m, int prob_n, int prob_k, void* workspace,
                   vllm::ScalarType const& q_type, bool has_act_order,
                   bool is_k_full, int num_groups, int group_size,
                   int num_experts, int topk, int moe_block_size, int dev,
                   cudaStream_t stream, int thread_k, int thread_n, int sms,
                   int max_par, bool replicate_input, bool apply_weights) {
  TORCH_CHECK(prob_m > 0 && prob_n > 0 && prob_k > 0, "Invalid MNK = [", prob_m,
              ", ", prob_n, ", ", prob_k, "]");

  if (sms == -1) {
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  }

  int max_shared_mem = 0;
  cudaDeviceGetAttribute(&max_shared_mem,
                         cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  TORCH_CHECK(max_shared_mem > 0);

  int num_bits = q_type.size_bits();

  // Set thread config
  exec_config_t exec_cfg;
  if (thread_k != -1 && thread_n != -1) {
    // User-defined config
    exec_cfg =
        exec_config_t{4, thread_config_t{thread_k, thread_n, USER_THREADS}};
  } else {
    // Auto config
    exec_cfg =
        determine_thread_config(prob_m, prob_n, prob_k, num_bits, group_size,
                                has_act_order, is_k_full, max_shared_mem);
  }

  TORCH_CHECK(exec_cfg.max_m_blocks > 0 &&
                  is_valid_config(exec_cfg.tb_cfg, exec_cfg.max_m_blocks,
                                  prob_m, prob_n, prob_k, num_bits, group_size,
                                  has_act_order, is_k_full, max_shared_mem),
              "Invalid thread config: max_m_blocks = ", exec_cfg.max_m_blocks,
              ", thread_k = ", exec_cfg.tb_cfg.thread_k,
              ", thread_n = ", exec_cfg.tb_cfg.thread_n,
              ", num_threads = ", exec_cfg.tb_cfg.num_threads, " for MKN = [",
              prob_m, ", ", prob_k, ", ", prob_n, "] and num_bits = ", num_bits,
              ", group_size = ", group_size,
              ", has_act_order = ", has_act_order, ", is_k_full = ", is_k_full,
              ", max_shared_mem = ", max_shared_mem);

  int num_threads = exec_cfg.tb_cfg.num_threads;
  thread_k = exec_cfg.tb_cfg.thread_k;
  thread_n = exec_cfg.tb_cfg.thread_n;

  int thread_k_blocks = thread_k / 16;
  int thread_n_blocks = thread_n / 16;

  int blocks = sms;

  TORCH_CHECK(prob_n % thread_n == 0, "prob_n = ", prob_n,
              " is not divisible by thread_n = ", thread_n);
  TORCH_CHECK(prob_k % thread_k == 0, "prob_k = ", prob_k,
              " is not divisible by thread_k = ", thread_k);

  int group_blocks = 0;
  if (has_act_order) {
    if (is_k_full) {
      TORCH_CHECK(group_size != -1);
      group_blocks = group_size / 16;
      TORCH_CHECK(prob_k % group_blocks == 0, "prob_k = ", prob_k,
                  " is not divisible by group_blocks = ", group_blocks);
    } else {
      TORCH_CHECK(group_size == 0);
      group_blocks = 0;
    }

  } else {
    if (group_size == -1) {
      group_blocks = -1;
    } else {
      group_blocks = group_size / 16;
      TORCH_CHECK(prob_k % group_blocks == 0, "prob_k = ", prob_k,
                  " is not divisible by group_blocks = ", group_blocks);
    }
  }

  int tot_m = prob_m;

  const int* topk_ids_ptr = (const int*)topk_ids;
  int* expert_offsets_ptr = (int*)expert_offsets;
  compute_expert_offsets<<<1, num_experts, 0, stream>>>(
      topk_ids_ptr, expert_offsets_ptr, tot_m * topk, moe_block_size);

  bool do_permute_a = has_act_order;

  // If we have a full K, then we can run the non-act-order version of Marlin
  // (since the weight rows are reordered by increasing group ids, and by
  // having a full K, we have full original groups)
  if (is_k_full) {
    has_act_order = false;
  }

  int pack_factor = 32 / q_type.size_bits();

  for (int expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
    const int4* A_ptr = (const int4*)A;
    int4* a_tmp_ptr = (int4*)a_tmp;
    const int4* B_ptr =
        (const int4*)B + (prob_n * prob_k / (pack_factor * 4)) * expert_idx;
    int4* C_ptr = (int4*)C;
    const float* topk_weights_ptr = (const float*)topk_weights;
    const int* sorted_ids_ptr = (const int*)sorted_ids;
    const int4* s_ptr = (const int4*)s + num_groups * prob_n / 8 * expert_idx;
    const int* g_idx_ptr = (const int*)g_idx + prob_k * expert_idx;
    const int* perm_ptr = (const int*)perm + prob_k * expert_idx;
    int* locks = (int*)workspace;

    if (do_permute_a) {
      // Permute A columns
      int topk_rows = replicate_input ? tot_m : tot_m * topk;
      int block_rows = ceildiv(topk_rows, blocks);
      permute_cols_kernel<<<blocks, num_threads, 0, stream>>>(
          A_ptr, perm_ptr, a_tmp_ptr, topk_rows, prob_k, block_rows);
      A_ptr = a_tmp_ptr;
    }

    int tot_m_blocks = ceildiv(tot_m, 16);
    for (int m_block = 0; m_block < tot_m_blocks;
         m_block += 4 * exec_cfg.max_m_blocks) {
      if (false) {
      }
      CALL_MOE_KERNEL_FUNCTION(call_marlin_moe_kernel_ku4b8)
      CALL_MOE_KERNEL_FUNCTION(call_marlin_moe_kernel_ku8b128)
      else {
        TORCH_CHECK(false, "Unsupported shapes: MNK = [" + str(prob_m) + ", " +
                               str(prob_n) + ", " + str(prob_k) + "]" +
                               ", has_act_order = " + str(has_act_order) +
                               ", num_groups = " + str(num_groups) +
                               ", group_size = " + str(group_size) +
                               ", thread_n_blocks = " + str(thread_n_blocks) +
                               ", thread_k_blocks = " + str(thread_k_blocks));
      }
    }
  }
}

}  // namespace marlin_moe

torch::Tensor marlin_gemm_moe(
    const torch::Tensor& a, const torch::Tensor& b_q_weights,
    const torch::Tensor& sorted_ids, const torch::Tensor& topk_weights,
    const torch::Tensor& topk_ids, const torch::Tensor& b_scales,
    const torch::Tensor& g_idx, const torch::Tensor& perm,
    torch::Tensor& workspace, vllm::ScalarTypeTorchPtr const& b_q_type,
    int64_t size_m, int64_t size_n, int64_t size_k, bool is_k_full,
    int64_t num_experts, int64_t topk, int64_t moe_block_size,
    bool replicate_input, bool apply_weights) {
  TORCH_CHECK(*b_q_type == vllm::kU4B8 || *b_q_type == vllm::kU8B128,
              "b_q_type must be uint4b8 or uint8b128. Got = ", b_q_type->str());

  int pack_factor = 32 / b_q_type->size_bits();

  int max_par = 4;

  int dev = a.get_device();

  auto options_dtype =
      torch::TensorOptions().dtype(a.dtype()).device(a.device());
  auto options_int =
      torch::TensorOptions().dtype(torch::kInt).device(a.device());
  torch::Tensor c = torch::zeros({size_m, topk, size_n}, options_dtype);
  torch::Tensor a_tmp =
      replicate_input ? torch::zeros({size_m, size_k}, options_dtype)
                      : torch::zeros({size_m, topk, size_k}, options_dtype);
  torch::Tensor expert_offsets = torch::empty({num_experts + 1}, options_int);

  // thread_k: `k` size of a thread_tile in `weights` (can usually be left as
  // auto -1)
  int thread_k = -1;
  // thread_n: `n` size of a thread_tile in `weights` (can usually be left as
  // auto -1)
  int thread_n = -1;
  // sms: number of SMs to use for the kernel (can usually be left as auto -1)
  int sms = -1;

  // Detect groupsize and act_order
  int num_groups = -1;
  int group_size = -1;
  bool has_act_order = g_idx.size(1) != 0;

  int b_rank = b_scales.sizes().size();
  TORCH_CHECK(b_rank == 3, "b_scales rank = ", b_rank, " is not 3");
  TORCH_CHECK(b_scales.size(2) == size_n, "b_scales dim 2 = ", b_scales.size(2),
              " is not size_n = ", size_n);
  num_groups = b_scales.size(1);

  TORCH_CHECK(VLLM_IMPLIES(!is_k_full, has_act_order),
              "if is_k_full is false, has_act_order must be true");

  if (has_act_order) {
    if (is_k_full) {
      TORCH_CHECK(num_groups > 1, "For act_order, num_groups must be > 1");
      TORCH_CHECK(size_k % num_groups == 0, "size_k = ", size_k,
                  ", is not divisible by num_groups = ", num_groups);
      group_size = size_k / num_groups;
    } else {
      group_size = 0;
    }

  } else {
    if (num_groups > 1) {
      TORCH_CHECK(
          size_k % num_groups == 0, "size_k = ", size_k,
          ", is not divisible by b_scales.size(0) = ", b_scales.size(0));
      group_size = size_k / num_groups;
    } else {
      group_size = -1;
    }
  }

  marlin_moe::marlin_mm_moe(
      a.data_ptr(), b_q_weights.data_ptr(), c.data_ptr(), sorted_ids.data_ptr(),
      topk_weights.data_ptr(), topk_ids.data_ptr(), b_scales.data_ptr(),
      g_idx.data_ptr(), perm.data_ptr(), a_tmp.data_ptr(),
      expert_offsets.data_ptr(), size_m, size_n, size_k, workspace.data_ptr(),
      *b_q_type, has_act_order, is_k_full, num_groups, group_size, num_experts,
      topk, moe_block_size, dev, at::cuda::getCurrentCUDAStream(dev), thread_k,
      thread_n, sms, max_par, replicate_input, apply_weights);
  return c;
}
