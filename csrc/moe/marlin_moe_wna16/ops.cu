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

/*
 * Adapted from https://github.com/IST-DASLab/marlin
 */

#ifndef MARLIN_NAMESPACE_NAME
  #define MARLIN_NAMESPACE_NAME marlin_moe_wna16
#endif

#include "kernel.h"
#include "core/registration.h"

#define STATIC_ASSERT_SCALAR_TYPE_VALID(scalar_t)               \
  static_assert(std::is_same<scalar_t, half>::value ||          \
                    std::is_same<scalar_t, nv_bfloat16>::value, \
                "only float16 and bfloat16 is supported");

namespace MARLIN_NAMESPACE_NAME {

__global__ void MarlinDefault(MARLIN_KERNEL_PARAMS){};

using MarlinFuncPtr = void (*)(MARLIN_KERNEL_PARAMS);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800

template <int moe_block_size>
__global__ void permute_cols_kernel(
    int4 const* __restrict__ a_int4_ptr, int const* __restrict__ perm_int_ptr,
    int4* __restrict__ out_int4_ptr,
    const int32_t* __restrict__ sorted_token_ids_ptr,
    const int32_t* __restrict__ expert_ids_ptr,
    const int32_t* __restrict__ num_tokens_past_padded_ptr, int size_m,
    int size_k, int top_k) {};

}  // namespace marlin

torch::Tensor moe_wna16_marlin_gemm(
    torch::Tensor& a, std::optional<torch::Tensor> const& c_or_none,
    torch::Tensor& b_q_weight, torch::Tensor& b_scales,
    std::optional<torch::Tensor> const& b_zeros_or_none,
    std::optional<torch::Tensor> const& g_idx_or_none,
    std::optional<torch::Tensor> const& perm_or_none, torch::Tensor& workspace,
    torch::Tensor& sorted_token_ids, torch::Tensor& expert_ids,
    torch::Tensor& num_tokens_past_padded, torch::Tensor& topk_weights,
    int64_t moe_block_size, int64_t top_k, bool mul_topk_weights, bool is_ep,
    vllm::ScalarTypeId const& b_q_type_id, int64_t size_m, int64_t size_n,
    int64_t size_k, bool is_k_full, bool use_atomic_add, bool use_fp32_reduce,
    bool is_zp_float) {
  TORCH_CHECK_NOT_IMPLEMENTED(false,
                              "marlin_gemm(..) requires CUDA_ARCH >= 8.0");
  return torch::empty({1, 1});
}

#else

// For a given "a" of size [M,K] performs a permutation of the K columns based
// on the given "perm" indices.
template <int moe_block_size>
__global__ void permute_cols_kernel(
    int4 const* __restrict__ a_int4_ptr, int const* __restrict__ perm_int_ptr,
    int4* __restrict__ out_int4_ptr,
    const int32_t* __restrict__ sorted_token_ids_ptr,
    const int32_t* __restrict__ expert_ids_ptr,
    const int32_t* __restrict__ num_tokens_past_padded_ptr, int size_m,
    int size_k, int top_k) {
  int num_tokens_past_padded = num_tokens_past_padded_ptr[0];
  int num_moe_blocks = div_ceil(num_tokens_past_padded, moe_block_size);
  int32_t block_sorted_ids[moe_block_size];
  int block_num_valid_tokens = 0;
  int64_t old_expert_id = 0;
  int64_t expert_id = 0;
  int row_stride = size_k * sizeof(half) / 16;

  auto read_moe_block_data = [&](int block_id) {
    block_num_valid_tokens = moe_block_size;
    int4* tmp_block_sorted_ids = reinterpret_cast<int4*>(block_sorted_ids);
    for (int i = 0; i < moe_block_size / 4; i++) {
      tmp_block_sorted_ids[i] =
          ((int4*)sorted_token_ids_ptr)[block_id * moe_block_size / 4 + i];
    }
    for (int i = 0; i < moe_block_size; i++) {
      if (block_sorted_ids[i] >= size_m * top_k) {
        block_num_valid_tokens = i;
        break;
      };
    }
  };

  auto permute_row = [&](int row) {
    int iters = size_k / default_threads;
    int rest = size_k % default_threads;

    int in_offset = (row / top_k) * row_stride;
    int out_offset = row * row_stride;

    half const* a_row_half =
        reinterpret_cast<half const*>(a_int4_ptr + in_offset);
    half* out_half = reinterpret_cast<half*>(out_int4_ptr + out_offset);

    int base_k = 0;

    for (int i = 0; i < iters; i++) {
      int cur_k = base_k + threadIdx.x;
      int src_pos = perm_int_ptr[cur_k];

      out_half[cur_k] = a_row_half[src_pos];

      base_k += default_threads;
    }

    if (rest) {
      if (threadIdx.x < rest) {
        int cur_k = base_k + threadIdx.x;
        int src_pos = perm_int_ptr[cur_k];

        out_half[cur_k] = a_row_half[src_pos];
      }
    }
  };

  for (int index = blockIdx.x; index < num_moe_blocks; index += gridDim.x) {
    old_expert_id = expert_id;
    int tmp_expert_id = expert_ids_ptr[index];
    if (tmp_expert_id == -1) continue;
    expert_id = tmp_expert_id;
    perm_int_ptr += (expert_id - old_expert_id) * size_k;
    read_moe_block_data(index);

    for (int i = 0; i < block_num_valid_tokens; i++)
      permute_row(block_sorted_ids[i]);
  }
}

typedef struct {
  int thread_k;
  int thread_n;
  int num_threads;
} thread_config_t;

thread_config_t small_batch_thread_configs[] = {
    // Ordered by priority

    // thread_k, thread_n, num_threads
    {128, 128, 256},
    {64, 128, 128}};

thread_config_t large_batch_thread_configs[] = {
    // Ordered by priority

    // thread_k, thread_n, num_threads
    {64, 256, 256},
    {64, 128, 128}};

typedef struct {
  int blocks_per_sm;
  thread_config_t tb_cfg;
} exec_config_t;

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
    tb_groups = div_ceil(tb_k, 32);  // Worst case is 32 group size
  } else {
    tb_groups = div_ceil(tb_k, group_size);
  }

  if (cache_scales_chunk) {
    int load_groups =
        tb_groups * pipe_stages * 2;     // Chunk size is 2x pipeline over dim K
    load_groups = max(load_groups, 32);  // We load at least 32 scale groups
    return load_groups * tb_n * 2;

  } else {
    int tb_scales = tb_groups * tb_n * 2;

    return tb_scales * pipe_stages;
  }
}

int get_kernel_cache_size(thread_config_t const& th_config, int thread_m_blocks,
                          int prob_m, int prob_n, int prob_k, int num_bits,
                          int group_size, bool has_act_order, bool is_k_full,
                          int has_zp, int is_zp_float) {
  int pack_factor = 32 / num_bits;

  // Get B size
  int tb_k = th_config.thread_k;
  int tb_n = th_config.thread_n;
  int tb_m = thread_m_blocks * 16;

  // shm size for block_sorted_ids/block_topk_weights
  // both of them requires tb_m * 4 bytes (tb_m * int32 or tb_m * float32)
  int sh_block_meta_size = tb_m * 4 * 2;
  int sh_a_size = pipe_stages * (tb_m * tb_k) * 2;
  int sh_b_size = pipe_stages * (tb_k * tb_n / pack_factor) * 4;
  int sh_s_size =
      get_scales_cache_size(th_config, prob_m, prob_n, prob_k, num_bits,
                            group_size, has_act_order, is_k_full);
  int sh_g_idx_size = has_act_order && !is_k_full ? pipe_stages * tb_k / 4 : 0;
  int sh_zp_size = 0;
  if (has_zp) {
    if (is_zp_float)
      sh_zp_size = sh_s_size;
    else if (num_bits == 4)
      sh_zp_size = sh_s_size / 4;
    else if (num_bits == 8)
      sh_zp_size = sh_s_size / 2;
  }

  int total_size = sh_a_size + sh_b_size + sh_s_size + sh_zp_size +
                   sh_g_idx_size + sh_block_meta_size;

  return total_size;
}

bool is_valid_config(thread_config_t const& th_config, int thread_m_blocks,
                     int prob_m, int prob_n, int prob_k, int num_bits,
                     int group_size, bool has_act_order, bool is_k_full,
                     int has_zp, int is_zp_float, int max_shared_mem) {
  // Sanity
  if (th_config.thread_k == -1 || th_config.thread_n == -1 ||
      th_config.num_threads == -1) {
    return false;
  }

  // Verify K/N are divisible by thread K/N
  if (prob_k % th_config.thread_k != 0 || prob_n % th_config.thread_n != 0) {
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

  // Check that pipeline fits into cache
  int cache_size = get_kernel_cache_size(
      th_config, thread_m_blocks, prob_m, prob_n, prob_k, num_bits, group_size,
      has_act_order, is_k_full, has_zp, is_zp_float);
  return cache_size <= max_shared_mem;
}

  #define __GET_IF(W_TYPE, THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS, \
                   M_BLOCK_SIZE_8, HAS_ACT_ORDER, HAS_ZP, GROUP_BLOCKS,       \
                   NUM_THREADS, IS_ZP_FLOAT)                                  \
    else if (q_type == W_TYPE && thread_m_blocks == THREAD_M_BLOCKS &&        \
             thread_n_blocks == THREAD_N_BLOCKS &&                            \
             thread_k_blocks == THREAD_K_BLOCKS &&                            \
             m_block_size_8 == M_BLOCK_SIZE_8 &&                              \
             has_act_order == HAS_ACT_ORDER && has_zp == HAS_ZP &&            \
             group_blocks == GROUP_BLOCKS && num_threads == NUM_THREADS &&    \
             is_zp_float == IS_ZP_FLOAT) {                                    \
      kernel = Marlin<scalar_t, W_TYPE.id(), NUM_THREADS, THREAD_M_BLOCKS,    \
                      THREAD_N_BLOCKS, THREAD_K_BLOCKS, M_BLOCK_SIZE_8,       \
                      pipe_stages, HAS_ACT_ORDER, HAS_ZP, GROUP_BLOCKS,       \
                      IS_ZP_FLOAT>;                                           \
    }

  #define GPTQ_GET_IF_M1(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS)              \
    __GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, true, true, false, 0, NUM_THREADS, \
             false)                                                            \
    __GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, true, false, 0,             \
             NUM_THREADS, false)                                               \
                                                                               \
    __GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, true, false, false, -1,            \
             NUM_THREADS, false)                                               \
    __GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, true, false, false, 2,             \
             NUM_THREADS, false)                                               \
    __GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, true, false, false, 4,             \
             NUM_THREADS, false)                                               \
    __GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, true, false, false, 8,             \
             NUM_THREADS, false)                                               \
    __GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, false, false, -1,           \
             NUM_THREADS, false)                                               \
    __GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, false, false, 2,            \
             NUM_THREADS, false)                                               \
    __GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, false, false, 4,            \
             NUM_THREADS, false)                                               \
    __GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, false, false, 8,            \
             NUM_THREADS, false)

  #define GPTQ_GET_IF_M234(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS)  \
    __GET_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, true, false, 0,   \
             NUM_THREADS, false)                                     \
    __GET_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, true, false, 0,   \
             NUM_THREADS, false)                                     \
    __GET_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, true, false, 0,   \
             NUM_THREADS, false)                                     \
                                                                     \
    __GET_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, false, false, -1, \
             NUM_THREADS, false)                                     \
    __GET_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, false, false, 2,  \
             NUM_THREADS, false)                                     \
    __GET_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, false, false, 4,  \
             NUM_THREADS, false)                                     \
    __GET_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, false, false, 8,  \
             NUM_THREADS, false)                                     \
                                                                     \
    __GET_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, false, false, -1, \
             NUM_THREADS, false)                                     \
    __GET_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, false, false, 2,  \
             NUM_THREADS, false)                                     \
    __GET_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, false, false, 4,  \
             NUM_THREADS, false)                                     \
    __GET_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, false, false, 8,  \
             NUM_THREADS, false)                                     \
                                                                     \
    __GET_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, false, false, -1, \
             NUM_THREADS, false)                                     \
    __GET_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, false, false, 2,  \
             NUM_THREADS, false)                                     \
    __GET_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, false, false, 4,  \
             NUM_THREADS, false)                                     \
    __GET_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, false, false, 8,  \
             NUM_THREADS, false)

  #define AWQ_GET_IF_M1(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS)               \
    __GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, true, false, true, -1,             \
             NUM_THREADS, false)                                               \
    __GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, true, false, true, 2, NUM_THREADS, \
             false)                                                            \
    __GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, true, false, true, 4, NUM_THREADS, \
             false)                                                            \
    __GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, true, false, true, 8, NUM_THREADS, \
             false)                                                            \
    __GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, false, true, -1,            \
             NUM_THREADS, false)                                               \
    __GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, false, true, 2,             \
             NUM_THREADS, false)                                               \
    __GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, false, true, 4,             \
             NUM_THREADS, false)                                               \
    __GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, false, true, 8,             \
             NUM_THREADS, false)

  #define AWQ_GET_IF_M234(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS)  \
    __GET_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, false, true, -1, \
             NUM_THREADS, false)                                    \
    __GET_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, false, true, 2,  \
             NUM_THREADS, false)                                    \
    __GET_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, false, true, 4,  \
             NUM_THREADS, false)                                    \
    __GET_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, false, true, 8,  \
             NUM_THREADS, false)                                    \
                                                                    \
    __GET_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, false, true, -1, \
             NUM_THREADS, false)                                    \
    __GET_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, false, true, 2,  \
             NUM_THREADS, false)                                    \
    __GET_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, false, true, 4,  \
             NUM_THREADS, false)                                    \
    __GET_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, false, true, 8,  \
             NUM_THREADS, false)                                    \
                                                                    \
    __GET_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, false, true, -1, \
             NUM_THREADS, false)                                    \
    __GET_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, false, true, 2,  \
             NUM_THREADS, false)                                    \
    __GET_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, false, true, 4,  \
             NUM_THREADS, false)                                    \
    __GET_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, false, true, 8,  \
             NUM_THREADS, false)

  // We currently have 4-bit models only with group_blocks == 4
  #define HQQ_GET_IF(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS)                  \
    __GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, true, false, true, 4, NUM_THREADS, \
             true)                                                             \
    __GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, false, true, 4,             \
             NUM_THREADS, true)                                                \
    __GET_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, false, true, 4,             \
             NUM_THREADS, true)                                                \
    __GET_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, false, true, 4,             \
             NUM_THREADS, true)                                                \
    __GET_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, false, true, 4,             \
             NUM_THREADS, true)

template <typename scalar_t>
MarlinFuncPtr get_marlin_kernel(const vllm::ScalarType q_type,
                                int thread_m_blocks, int thread_n_blocks,
                                int thread_k_blocks, bool m_block_size_8,
                                bool has_act_order, bool has_zp,
                                int group_blocks, int num_threads,
                                bool is_zp_float) {
  int num_bits = q_type.size_bits();
  auto kernel = MarlinDefault;
  if (false) {
  }
  GPTQ_GET_IF_M1(vllm::kU4B8, 8, 8, 256)
  GPTQ_GET_IF_M1(vllm::kU4B8, 8, 4, 128)

  GPTQ_GET_IF_M234(vllm::kU4B8, 16, 4, 256)
  GPTQ_GET_IF_M234(vllm::kU4B8, 8, 4, 128)

  GPTQ_GET_IF_M1(vllm::kU8B128, 8, 8, 256)
  GPTQ_GET_IF_M1(vllm::kU8B128, 8, 4, 128)

  GPTQ_GET_IF_M234(vllm::kU8B128, 16, 4, 256)
  GPTQ_GET_IF_M234(vllm::kU8B128, 8, 4, 128)

  AWQ_GET_IF_M1(vllm::kU4, 8, 8, 256)
  AWQ_GET_IF_M1(vllm::kU4, 8, 4, 128)

  AWQ_GET_IF_M234(vllm::kU4, 16, 4, 256)
  AWQ_GET_IF_M234(vllm::kU4, 8, 4, 128)

  return kernel;
}

template <typename scalar_t>
exec_config_t determine_exec_config(const vllm::ScalarType& q_type, int prob_m,
                                    int prob_n, int prob_k, int thread_m_blocks,
                                    bool m_block_size_8, int num_bits,
                                    int group_size, bool has_act_order,
                                    bool is_k_full, bool has_zp,
                                    bool is_zp_float, int max_shared_mem) {
  exec_config_t exec_cfg = exec_config_t{1, thread_config_t{-1, -1, -1}};
  thread_config_t* thread_configs = thread_m_blocks > 1
                                        ? large_batch_thread_configs
                                        : small_batch_thread_configs;
  int thread_configs_size =
      thread_m_blocks > 1
          ? sizeof(large_batch_thread_configs) / sizeof(thread_config_t)
          : sizeof(small_batch_thread_configs) / sizeof(thread_config_t);

  int count = 0;
  constexpr int device_max_reg_size = 255 * 1024;
  for (int i = 0; i < thread_configs_size; i++) {
    thread_config_t th_config = thread_configs[i];

    if (!is_valid_config(th_config, thread_m_blocks, prob_m, prob_n, prob_k,
                         num_bits, group_size, has_act_order, is_k_full, has_zp,
                         is_zp_float, max_shared_mem)) {
      continue;
    }

    int cache_size = get_kernel_cache_size(
        th_config, thread_m_blocks, prob_m, prob_n, prob_k, num_bits,
        group_size, has_act_order, is_k_full, has_zp, is_zp_float);

    int group_blocks = 0;
    if (!has_act_order) {
      group_blocks = group_size == -1 ? -1 : group_size / 16;
    }

    auto kernel = get_marlin_kernel<scalar_t>(
        q_type, thread_m_blocks, th_config.thread_n / 16,
        th_config.thread_k / 16, m_block_size_8, has_act_order, has_zp,
        group_blocks, th_config.num_threads, is_zp_float);

    if (kernel == MarlinDefault) continue;

    if (thread_m_blocks > 1) {
      exec_cfg = {1, th_config};
      break;
    } else {
      cudaFuncAttributes attr;
      cudaFuncGetAttributes(&attr, kernel);
      int reg_size = max(attr.numRegs, 1) * th_config.num_threads * 4;
      int allow_count = min(device_max_reg_size / reg_size,
                            max_shared_mem / (cache_size + 1024));
      allow_count = max(min(allow_count, 4), 1);
      if (allow_count > count) {
        count = allow_count;
        exec_cfg = {count, th_config};
      };
    }
  }

  return exec_cfg;
}

template <typename scalar_t>
void marlin_mm(const void* A, const void* B, void* C, void* C_tmp, void* s,
               void* zp, void* g_idx, void* perm, void* a_tmp,
               void* sorted_token_ids, void* expert_ids,
               void* num_tokens_past_padded, void* topk_weights,
               int moe_block_size, int top_k, bool mul_topk_weights, bool is_ep,
               int prob_m, int prob_n, int prob_k, void* workspace,
               vllm::ScalarType const& q_type, bool has_act_order,
               bool is_k_full, bool has_zp, int num_groups, int group_size,
               int dev, cudaStream_t stream, int thread_k, int thread_n,
               int sms, bool use_atomic_add, bool use_fp32_reduce,
               bool is_zp_float) {
  int thread_m_blocks = div_ceil(moe_block_size, 16);
  bool m_block_size_8 = moe_block_size == 8;

  if (has_zp) {
    TORCH_CHECK(
        q_type == vllm::kU4 || q_type == vllm::kU8,
        "q_type must be u4 or u8 when has_zp = True. Got = ", q_type.str());
  } else {
    TORCH_CHECK(
        q_type == vllm::kU4B8 || q_type == vllm::kU8B128,
        "q_type must be uint4b8 or uint8b128 when has_zp = False. Got = ",
        q_type.str());
  }

  TORCH_CHECK(prob_m > 0 && prob_n > 0 && prob_k > 0, "Invalid MNK = [", prob_m,
              ", ", prob_n, ", ", prob_k, "]");

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

  int num_bits = q_type.size_bits();
  const int4* A_ptr = (const int4*)A;
  const int4* B_ptr = (const int4*)B;
  int4* C_ptr = (int4*)C;
  int4* C_tmp_ptr = (int4*)C_tmp;
  const int4* s_ptr = (const int4*)s;
  const int4* zp_ptr = (const int4*)zp;
  const int* g_idx_ptr = (const int*)g_idx;
  const int* perm_ptr = (const int*)perm;
  int4* a_tmp_ptr = (int4*)a_tmp;
  const int32_t* sorted_token_ids_ptr = (const int32_t*)sorted_token_ids;
  const int32_t* expert_ids_ptr = (const int32_t*)expert_ids;
  const int32_t* num_tokens_past_padded_ptr =
      (const int32_t*)num_tokens_past_padded;
  const float* topk_weights_ptr = (const float*)topk_weights;
  int* locks = (int*)workspace;

  if (has_act_order) {
    // Permute A columns
    auto kernel = permute_cols_kernel<8>;
    if (moe_block_size == 8) {
    } else if (moe_block_size == 16)
      kernel = permute_cols_kernel<16>;
    else if (moe_block_size == 32)
      kernel = permute_cols_kernel<32>;
    else if (moe_block_size == 48)
      kernel = permute_cols_kernel<48>;
    else if (moe_block_size == 64)
      kernel = permute_cols_kernel<64>;
    else
      TORCH_CHECK(false, "unsupported moe_block_size ", moe_block_size);

    // avoid ">>>" being formatted to "> > >"
    // clang-format off
    kernel<<<sms, default_threads, 0, stream>>>(
        A_ptr, perm_ptr, a_tmp_ptr, sorted_token_ids_ptr, expert_ids_ptr,
        num_tokens_past_padded_ptr, prob_m, prob_k, top_k);
    // clang-format on
    A_ptr = a_tmp_ptr;
    prob_m = prob_m * top_k;
    top_k = 1;

    // If we have a full K, then we can run the non-act-order version of Marlin
    // (since the weight rows are reordered by increasing group ids, and by
    // having a full K, we have full original groups)
    if (is_k_full) has_act_order = false;
  }

  int max_shared_mem = 0;
  cudaDeviceGetAttribute(&max_shared_mem,
                         cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  TORCH_CHECK(max_shared_mem > 0);

  // Set thread config
  exec_config_t exec_cfg;
  thread_config_t thread_tfg;
  if (thread_k != -1 && thread_n != -1) {
    thread_tfg = thread_config_t{thread_k, thread_n, default_threads};
    exec_cfg = exec_config_t{1, thread_tfg};
    TORCH_CHECK(prob_n % thread_n == 0, "prob_n = ", prob_n,
                " is not divisible by thread_n = ", thread_n);
    TORCH_CHECK(prob_k % thread_k == 0, "prob_k = ", prob_k,
                " is not divisible by thread_k = ", thread_k);
  } else {
    // Auto config
    exec_cfg = determine_exec_config<scalar_t>(
        q_type, prob_m, prob_n, prob_k, thread_m_blocks, m_block_size_8,
        num_bits, group_size, has_act_order, is_k_full, has_zp, is_zp_float,
        max_shared_mem);
    thread_tfg = exec_cfg.tb_cfg;
  }

  int num_threads = thread_tfg.num_threads;
  thread_k = thread_tfg.thread_k;
  thread_n = thread_tfg.thread_n;
  int blocks = sms * exec_cfg.blocks_per_sm;
  if (exec_cfg.blocks_per_sm > 1)
    max_shared_mem = max_shared_mem / exec_cfg.blocks_per_sm - 1024;

  int thread_k_blocks = thread_k / 16;
  int thread_n_blocks = thread_n / 16;

  TORCH_CHECK(is_valid_config(thread_tfg, thread_m_blocks, prob_m, prob_n,
                              prob_k, num_bits, group_size, has_act_order,
                              is_k_full, has_zp, is_zp_float, max_shared_mem),
              "Invalid thread config: thread_m_blocks = ", thread_m_blocks,
              ", thread_k = ", thread_tfg.thread_k,
              ", thread_n = ", thread_tfg.thread_n,
              ", num_threads = ", thread_tfg.num_threads, " for MKN = [",
              prob_m, ", ", prob_k, ", ", prob_n, "] and num_bits = ", num_bits,
              ", group_size = ", group_size,
              ", has_act_order = ", has_act_order, ", is_k_full = ", is_k_full,
              ", has_zp = ", has_zp, ", is_zp_float = ", is_zp_float,
              ", max_shared_mem = ", max_shared_mem);

  auto kernel = get_marlin_kernel<scalar_t>(
      q_type, thread_m_blocks, thread_n_blocks, thread_k_blocks, m_block_size_8,
      has_act_order, has_zp, group_blocks, num_threads, is_zp_float);

  if (kernel == MarlinDefault) {
    TORCH_CHECK(false, "Unsupported shapes: MNK = [", prob_m, ", ", prob_n,
                ", ", prob_k, "]", ", has_act_order = ", has_act_order,
                ", num_groups = ", num_groups, ", group_size = ", group_size,
                ", thread_m_blocks = ", thread_m_blocks,
                ", thread_n_blocks = ", thread_n_blocks,
                ", thread_k_blocks = ", thread_k_blocks,
                ", num_bits = ", num_bits);
  }

  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       max_shared_mem);
  // avoid ">>>" being formatted to "> > >"
  // clang-format off
  kernel<<<blocks, num_threads, max_shared_mem, stream>>>(
      A_ptr, B_ptr, C_ptr, C_tmp_ptr, s_ptr, zp_ptr, g_idx_ptr,
      sorted_token_ids_ptr, expert_ids_ptr, num_tokens_past_padded_ptr,
      topk_weights_ptr, top_k, mul_topk_weights, is_ep, num_groups, prob_m,
      prob_n, prob_k, locks, use_atomic_add, use_fp32_reduce);
  // clang-format on
}

}  // namespace MARLIN_NAMESPACE_NAME

torch::Tensor moe_wna16_marlin_gemm(
    torch::Tensor& a, std::optional<torch::Tensor> const& c_or_none,
    torch::Tensor& b_q_weight, torch::Tensor& b_scales,
    std::optional<torch::Tensor> const& b_zeros_or_none,
    std::optional<torch::Tensor> const& g_idx_or_none,
    std::optional<torch::Tensor> const& perm_or_none, torch::Tensor& workspace,
    torch::Tensor& sorted_token_ids, torch::Tensor& expert_ids,
    torch::Tensor& num_tokens_past_padded, torch::Tensor& topk_weights,
    int64_t moe_block_size, int64_t top_k, bool mul_topk_weights, bool is_ep,
    vllm::ScalarTypeId const& b_q_type_id, int64_t size_m, int64_t size_n,
    int64_t size_k, bool is_k_full, bool use_atomic_add, bool use_fp32_reduce,
    bool is_zp_float) {
  vllm::ScalarType const b_q_type = vllm::ScalarType::from_id(b_q_type_id);
  int pack_factor = 32 / b_q_type.size_bits();

  if (moe_block_size != 8) {
    TORCH_CHECK(moe_block_size % 16 == 0,
                "unsupported moe_block_size=", moe_block_size);
    TORCH_CHECK(moe_block_size >= 16 && moe_block_size <= 64,
                "unsupported moe_block_size=", moe_block_size);
  }

  // Verify A
  TORCH_CHECK(a.size(0) == size_m, "Shape mismatch: a.size(0) = ", a.size(0),
              ", size_m = ", size_m);
  TORCH_CHECK(a.size(1) == size_k, "Shape mismatch: a.size(1) = ", a.size(1),
              ", size_k = ", size_k);

  // Verify B
  TORCH_CHECK(
      size_k % MARLIN_NAMESPACE_NAME::tile_size == 0, "size_k = ", size_k,
      " is not divisible by tile_size = ", MARLIN_NAMESPACE_NAME::tile_size);
  TORCH_CHECK((size_k / MARLIN_NAMESPACE_NAME::tile_size) == b_q_weight.size(1),
              "Shape mismatch: b_q_weight.size(1) = ", b_q_weight.size(1),
              ", size_k = ", size_k,
              ", tile_size = ", MARLIN_NAMESPACE_NAME::tile_size);
  TORCH_CHECK(
      b_q_weight.size(2) % MARLIN_NAMESPACE_NAME::tile_size == 0,
      "b_q_weight.size(2) = ", b_q_weight.size(2),
      " is not divisible by tile_size = ", MARLIN_NAMESPACE_NAME::tile_size);
  int actual_size_n =
      (b_q_weight.size(2) / MARLIN_NAMESPACE_NAME::tile_size) * pack_factor;
  TORCH_CHECK(size_n == actual_size_n, "size_n = ", size_n,
              ", actual_size_n = ", actual_size_n);

  // Verify device and strides
  TORCH_CHECK(a.device().is_cuda(), "A is not on GPU");
  TORCH_CHECK(a.is_contiguous(), "A is not contiguous");

  TORCH_CHECK(b_q_weight.device().is_cuda(), "b_q_weight is not on GPU");
  TORCH_CHECK(b_q_weight.is_contiguous(), "b_q_weight is not contiguous");

  TORCH_CHECK(b_scales.device().is_cuda(), "b_scales is not on GPU");
  TORCH_CHECK(b_scales.is_contiguous(), "b_scales is not contiguous");

  // thread_k: `k` size of a thread_tile in `weights` (can usually be left as
  // auto -1)
  int thread_k = -1;
  // thread_n: `n` size of a thread_tile in `weights` (can usually be left as
  // auto -1)
  int thread_n = -1;
  // sms: number of SMs to use for the kernel
  int sms = -1;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, a.get_device());

  // Alloc buffers
  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
  auto options = torch::TensorOptions().dtype(a.dtype()).device(a.device());
  torch::Tensor c;
  if (c_or_none.has_value()) {
    c = c_or_none.value();
    TORCH_CHECK(c.device().is_cuda(), "c is not on GPU");
    TORCH_CHECK(c.is_contiguous(), "c is not contiguous");
    TORCH_CHECK(c.size(0) == size_m * top_k,
                "Shape mismatch: c.size(0) = ", c.size(0),
                ", size_m * topk = ", size_m * top_k);
    TORCH_CHECK(c.size(1) == size_n, "Shape mismatch: c.size(1) = ", c.size(1),
                ", size_n = ", size_n);
  } else {
    c = torch::empty({size_m * top_k, size_n}, options);
  }

  // Alloc C tmp buffer that is going to be used for the global reduce
  torch::Tensor c_tmp;
  auto options_fp32 =
      torch::TensorOptions().dtype(at::kFloat).device(a.device());
  if (use_fp32_reduce && !use_atomic_add) {
    // max num of threadblocks is sms * 4
    long max_c_tmp_size = min(
        (long)size_n * sorted_token_ids.size(0),
        (long)sms * 4 * moe_block_size * MARLIN_NAMESPACE_NAME::max_thread_n);
    if (moe_block_size == 8) max_c_tmp_size *= 2;
    c_tmp = torch::empty({max_c_tmp_size}, options_fp32);
  } else {
    c_tmp = torch::empty({0}, options_fp32);
  }

  // Detect groupsize and act_order
  int num_groups = -1;
  int group_size = -1;

  int rank = b_scales.sizes().size();
  TORCH_CHECK(rank == 3, "b_scales rank = ", rank, " is not 3");
  TORCH_CHECK(b_scales.size(2) == size_n, "b_scales dim 2 = ", b_scales.size(2),
              " is not size_n = ", size_n);
  num_groups = b_scales.size(1);

  torch::Tensor g_idx, perm, a_tmp;
  ;
  if (g_idx_or_none.has_value() && perm_or_none.has_value()) {
    g_idx = g_idx_or_none.value();
    perm = perm_or_none.value();

    TORCH_CHECK(g_idx.device().is_cuda(), "g_idx is not on GPU");
    TORCH_CHECK(g_idx.is_contiguous(), "g_idx is not contiguous");
    TORCH_CHECK(perm.device().is_cuda(), "perm is not on GPU");
    TORCH_CHECK(perm.is_contiguous(), "perm is not contiguous");

    // Verify g_idx and perm
    TORCH_CHECK((g_idx.size(-1) == 0 && perm.size(-1) == 0) ||
                    (g_idx.size(-1) == size_k && perm.size(-1) == size_k),
                "Unexpected g_idx.size(-1) = ", g_idx.size(-1),
                " and perm.size(-1) = ", perm.size(-1),
                ", where size_k = ", size_k);
  } else {
    g_idx = torch::empty({0}, options);
    perm = torch::empty({0}, options);
    a_tmp = torch::empty({0}, options);
  }
  bool has_act_order = g_idx.size(-1) > 0 && perm.size(-1) > 0;

  if (has_act_order) {
    a_tmp = torch::empty({size_m * top_k, size_k}, options);
    if (is_k_full) {
      TORCH_CHECK(num_groups > 1, "For act_order, num_groups must be > 1");
      TORCH_CHECK(size_k % num_groups == 0, "size_k = ", size_k,
                  ", is not divisible by num_groups = ", num_groups);
      group_size = size_k / num_groups;
    } else {
      group_size = 0;
    }

  } else {
    a_tmp = torch::empty({0}, options);
    if (num_groups > 1) {
      TORCH_CHECK(
          size_k % num_groups == 0, "size_k = ", size_k,
          ", is not divisible by b_scales.size(1) = ", b_scales.size(1));
      group_size = size_k / num_groups;
    } else {
      group_size = -1;
    }
  }

  torch::Tensor b_zeros;
  if (b_zeros_or_none.has_value()) {
    b_zeros = b_zeros_or_none.value();
    TORCH_CHECK(b_zeros.device().is_cuda(), "b_zeros is not on GPU");
    TORCH_CHECK(b_zeros.is_contiguous(), "b_zeros is not contiguous");
  } else {
    b_zeros = torch::empty({0}, options);
  }
  bool has_zp = b_zeros.size(-1) > 0;

  if (has_zp) {
    TORCH_CHECK(
        b_q_type == vllm::kU4,
        "b_q_type must be u4 when has_zp = True. Got = ", b_q_type.str());
  } else {
    TORCH_CHECK(
        b_q_type == vllm::kU4B8 || b_q_type == vllm::kU8B128,
        "b_q_type must be uint4b8 or uint8b128 when has_zp = False. Got = ",
        b_q_type.str());
  }

  if (has_zp && is_zp_float) {
    TORCH_CHECK(a.scalar_type() == at::ScalarType::Half,
                "Computation type must be float16 (half) when using float zero "
                "points.");
  }

  // Verify b_zeros
  if (has_zp) {
    int rank = b_zeros.sizes().size();
    TORCH_CHECK(rank == 3, "b_zeros rank = ", rank, " is not 3");
    if (is_zp_float) {
      TORCH_CHECK(b_zeros.size(2) == size_n,
                  "b_zeros dim 2 = ", b_zeros.size(2),
                  " is not size_n = ", size_n);
      TORCH_CHECK(num_groups == b_zeros.size(1),
                  "b_zeros dim 1 = ", b_zeros.size(1),
                  " is not num_groups = ", num_groups);
      TORCH_CHECK(num_groups != -1, "num_groups must be != -1");
    } else {
      TORCH_CHECK(b_zeros.size(1) == num_groups,
                  "b_zeros dim 1 = ", b_zeros.size(1),
                  " is not num_groups = ", num_groups);
      TORCH_CHECK(b_zeros.size(2) == size_n / pack_factor,
                  "b_zeros dim 2 = ", b_zeros.size(2),
                  " is not size_n / pack_factor = ", size_n / pack_factor);
    }
  }

  // Verify workspace size
  TORCH_CHECK(size_n % MARLIN_NAMESPACE_NAME::min_thread_n == 0,
              "size_n = ", size_n, ", is not divisible by min_thread_n = ",
              MARLIN_NAMESPACE_NAME::min_thread_n);

  int max_n_tiles = size_n / MARLIN_NAMESPACE_NAME::min_thread_n;
  int min_workspace_size = min(
      max_n_tiles * (int)(sorted_token_ids.size(0) / moe_block_size), sms * 4);
  TORCH_CHECK(workspace.numel() >= min_workspace_size,
              "workspace.numel = ", workspace.numel(),
              " is below min_workspace_size = ", min_workspace_size);

  int dev = a.get_device();
  if (a.scalar_type() == at::ScalarType::Half) {
    MARLIN_NAMESPACE_NAME::marlin_mm<half>(
        a.data_ptr<at::Half>(), b_q_weight.data_ptr(), c.data_ptr<at::Half>(),
        c_tmp.data_ptr<float>(), b_scales.data_ptr<at::Half>(),
        b_zeros.data_ptr(), g_idx.data_ptr(), perm.data_ptr(),
        a_tmp.data_ptr<at::Half>(), sorted_token_ids.data_ptr(),
        expert_ids.data_ptr(), num_tokens_past_padded.data_ptr(),
        topk_weights.data_ptr(), moe_block_size, top_k, mul_topk_weights, is_ep,
        size_m, size_n, size_k, workspace.data_ptr(), b_q_type, has_act_order,
        is_k_full, has_zp, num_groups, group_size, dev,
        at::cuda::getCurrentCUDAStream(dev), thread_k, thread_n, sms,
        use_atomic_add, use_fp32_reduce, is_zp_float);
  } else if (a.scalar_type() == at::ScalarType::BFloat16) {
    MARLIN_NAMESPACE_NAME::marlin_mm<nv_bfloat16>(
        a.data_ptr<at::BFloat16>(), b_q_weight.data_ptr(),
        c.data_ptr<at::BFloat16>(), c_tmp.data_ptr<float>(),
        b_scales.data_ptr<at::BFloat16>(), b_zeros.data_ptr(), g_idx.data_ptr(),
        perm.data_ptr(), a_tmp.data_ptr<at::BFloat16>(),
        sorted_token_ids.data_ptr(), expert_ids.data_ptr(),
        num_tokens_past_padded.data_ptr(), topk_weights.data_ptr(),
        moe_block_size, top_k, mul_topk_weights, is_ep, size_m, size_n, size_k,
        workspace.data_ptr(), b_q_type, has_act_order, is_k_full, has_zp,
        num_groups, group_size, dev, at::cuda::getCurrentCUDAStream(dev),
        thread_k, thread_n, sms, use_atomic_add, use_fp32_reduce, is_zp_float);
  } else {
    TORCH_CHECK(false,
                "moe_wna16_marlin_gemm only supports bfloat16 and float16");
  }

  return c;
}

#endif

TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, CUDA, m) {
  m.impl("moe_wna16_marlin_gemm", &moe_wna16_marlin_gemm);
}
