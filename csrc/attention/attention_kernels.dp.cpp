/*
 * Adapted from https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/kernels/decoder_masked_multihead_attention/decoder_masked_multihead_attention_template.hpp
 * Copyright (c) 2023, The vLLM team.
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <torch/extension.h>
#include <ATen/Aten.h>
#include <ipex.h>


#include "attention_dtypes.h"
#include "attention_utils.dp.hpp"

#include <algorithm>

#define WARP_SIZE 32
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define DIVIDE_ROUND_UP(a, b) (((a) + (b) - 1) / (b))

namespace at{

namespace cuda{

dpct::queue_ptr getCurrentCUDAStream(){

  auto device_type = c10::DeviceType::XPU;
  c10::impl::VirtualGuardImpl impl(device_type);
  c10::Stream c10_stream = impl.getStream(c10::Device(device_type));
  auto& queue = xpu::get_queue_from_stream(c10_stream);
  return &queue;

}

}

}

namespace vllm {

// Utility function for attention softmax.
template<int NUM_WARPS>
inline float block_sum(float* red_smem, float sum,
                       const sycl::nd_item<3> &item_ct1) {
  // Decompose the thread index into warp / lane.
  int warp = item_ct1.get_local_id(2) / WARP_SIZE;
  int lane = item_ct1.get_local_id(2) % WARP_SIZE;

  // Compute the sum per warp.
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
    /*
    DPCT1023:2: The SYCL sub-group does not support mask options for
    dpct::permute_sub_group_by_xor.
    */
    /*
    DPCT1096:42: The right-most dimension of the work-group used in the SYCL
    kernel that calls this function may be less than "32". The function
    "dpct::permute_sub_group_by_xor" may return an unexpected result on the CPU
    device. Modify the size of the work-group to ensure that the value of the
    right-most dimension is a multiple of "32".
    */
    sum += dpct::experimental::permute_sub_group_by_xor(0xffffffff, item_ct1.get_sub_group(), sum, mask);
  }

  // Warp leaders store the data to shared memory.
  if (lane == 0) {
    red_smem[warp] = sum;
  }

  // Make sure the data is in shared memory.
  /*
  DPCT1065:1: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();

  // The warps compute the final sums.
  if (lane < NUM_WARPS) {
    sum = red_smem[lane];
  }

  // Parallel reduction inside the warp.
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    /*
    DPCT1023:3: The SYCL sub-group does not support mask options for
    dpct::permute_sub_group_by_xor.
    */
    /*
    DPCT1096:43: The right-most dimension of the work-group used in the SYCL
    kernel that calls this function may be less than "32". The function
    "dpct::permute_sub_group_by_xor" may return an unexpected result on the CPU
    device. Modify the size of the work-group to ensure that the value of the
    right-most dimension is a multiple of "32".
    */
    sum += dpct::experimental::permute_sub_group_by_xor(0xffffffff, item_ct1.get_sub_group(), sum, mask);
  }

  // Broadcast to other threads.
  /*
  DPCT1023:4: The SYCL sub-group does not support mask options for
  dpct::select_from_sub_group.
  */
  /*
  DPCT1096:44: The right-most dimension of the work-group used in the SYCL
  kernel that calls this function may be less than "32". The function
  "dpct::select_from_sub_group" may return an unexpected result on the CPU
  device. Modify the size of the work-group to ensure that the value of the
  right-most dimension is a multiple of "32".
  */
  return dpct::experimental::select_from_sub_group(0xffffffff, item_ct1.get_sub_group(), sum, 0);
}

// TODO(woosuk): Merge the last two dimensions of the grid.
// Grid: (num_heads, num_seqs, max_num_partitions).
template<
  typename scalar_t,
  int HEAD_SIZE,
  int BLOCK_SIZE,
  int NUM_THREADS,
  int PARTITION_SIZE = 0> // Zero means no partitioning.
void paged_attention_kernel(
  float* __restrict__ exp_sums,           // [num_seqs, num_heads, max_num_partitions]
  float* __restrict__ max_logits,         // [num_seqs, num_heads, max_num_partitions]
  scalar_t* __restrict__ out,             // [num_seqs, num_heads, max_num_partitions, head_size]
  const scalar_t* __restrict__ q,         // [num_seqs, num_heads, head_size]
  const scalar_t* __restrict__ k_cache,   // [num_blocks, num_kv_heads, head_size/x, block_size, x]
  const scalar_t* __restrict__ v_cache,   // [num_blocks, num_kv_heads, head_size, block_size]
  const int* __restrict__ head_mapping,   // [num_heads]
  const float scale,
  const int* __restrict__ block_tables,   // [num_seqs, max_num_blocks_per_seq]
  const int* __restrict__ context_lens,   // [num_seqs]
  const int max_num_blocks_per_seq,
  const float* __restrict__ alibi_slopes, // [num_heads]
  const int q_stride,
  const int kv_block_stride,
  const int kv_head_stride,
  const sycl::nd_item<3> &item_ct1,
  uint8_t *dpct_local,
  sycl::local_accessor<Q_vec, 2> q_vecs,
  float *red_smem) {
  const int seq_idx = item_ct1.get_group(1);
  const int partition_idx = item_ct1.get_group(0);
  const int max_num_partitions = item_ct1.get_group_range(0);
  constexpr bool USE_PARTITIONING = PARTITION_SIZE > 0;
  const int context_len = context_lens[seq_idx];
  if (USE_PARTITIONING && partition_idx * PARTITION_SIZE >= context_len) {
    // No work to do. Terminate the thread block.
    return;
  }

  const int num_context_blocks = DIVIDE_ROUND_UP(context_len, BLOCK_SIZE);
  const int num_blocks_per_partition = USE_PARTITIONING ? PARTITION_SIZE / BLOCK_SIZE : num_context_blocks;

  // [start_block_idx, end_block_idx) is the range of blocks to process.
  const int start_block_idx = USE_PARTITIONING ? partition_idx * num_blocks_per_partition : 0;
  const int end_block_idx = MIN(start_block_idx + num_blocks_per_partition, num_context_blocks);
  const int num_blocks = end_block_idx - start_block_idx;

  // [start_token_idx, end_token_idx) is the range of tokens to process.
  const int start_token_idx = start_block_idx * BLOCK_SIZE;
  const int end_token_idx = MIN(start_token_idx + num_blocks * BLOCK_SIZE, context_len);
  const int num_tokens = end_token_idx - start_token_idx;

  constexpr int THREAD_GROUP_SIZE = MAX(WARP_SIZE / BLOCK_SIZE, 1);
  constexpr int NUM_THREAD_GROUPS = NUM_THREADS / THREAD_GROUP_SIZE; // Note: This assumes THREAD_GROUP_SIZE divides NUM_THREADS
  assert(NUM_THREADS % THREAD_GROUP_SIZE == 0);
  constexpr int NUM_TOKENS_PER_THREAD_GROUP = DIVIDE_ROUND_UP(BLOCK_SIZE, WARP_SIZE);
  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  const int thread_idx = item_ct1.get_local_id(2);
  const int warp_idx = thread_idx / WARP_SIZE;
  const int lane = thread_idx % WARP_SIZE;

  const int head_idx = item_ct1.get_group(2);
  const int num_heads = item_ct1.get_group_range(2);
  const int kv_head_idx = head_mapping[head_idx];
  const float alibi_slope = alibi_slopes == nullptr ? 0.f : alibi_slopes[head_idx];

  // A vector type to store a part of a key or a query.
  // The vector size is configured in such a way that the threads in a thread group
  // fetch or compute 16 bytes at a time.
  // For example, if the size of a thread group is 4 and the data type is half,
  // then the vector size is 16 / (4 * sizeof(half)) == 2.
  constexpr int VEC_SIZE = MAX(16 / (THREAD_GROUP_SIZE * sizeof(scalar_t)), 1);
  using K_vec = typename Vec<scalar_t, VEC_SIZE>::Type;
  using Q_vec = typename Vec<scalar_t, VEC_SIZE>::Type;

  constexpr int NUM_ELEMS_PER_THREAD = HEAD_SIZE / THREAD_GROUP_SIZE;
  constexpr int NUM_VECS_PER_THREAD = NUM_ELEMS_PER_THREAD / VEC_SIZE;

  const int thread_group_idx = thread_idx / THREAD_GROUP_SIZE;
  const int thread_group_offset = thread_idx % THREAD_GROUP_SIZE;

  // Load the query to registers.
  // Each thread in a thread group has a different part of the query.
  // For example, if the the thread group size is 4, then the first thread in the group
  // has 0, 4, 8, ... th vectors of the query, and the second thread has 1, 5, 9, ...
  // th vectors of the query, and so on.
  // NOTE(woosuk): Because q is split from a qkv tensor, it may not be contiguous.
  const scalar_t* q_ptr = q + seq_idx * q_stride + head_idx * HEAD_SIZE;

#pragma unroll
  for (int i = thread_group_idx; i < NUM_VECS_PER_THREAD; i += NUM_THREAD_GROUPS) {
    const int vec_idx = thread_group_offset + i * THREAD_GROUP_SIZE;
    q_vecs[thread_group_offset][i] = *reinterpret_cast<const Q_vec*>(q_ptr + vec_idx * VEC_SIZE);
  }
  /*
  DPCT1065:5: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier(); // TODO(naed90): possible speedup if this is replaced with
                      // a memory wall right before we use q_vecs

  // Memory planning.
  auto shared_mem = (char *)dpct_local;
  // NOTE(woosuk): We use FP32 for the softmax logits for better accuracy.
  float* logits = reinterpret_cast<float*>(shared_mem);
  // Workspace for reduction.

  // x == THREAD_GROUP_SIZE * VEC_SIZE
  // Each thread group fetches x elements from the key at a time.
  constexpr int x = 16 / sizeof(scalar_t);
  float qk_max = -FLT_MAX;

  // Iterate over the key blocks.
  // Each warp fetches a block of keys for each iteration.
  // Each thread group in a warp fetches a key from the block, and computes
  // dot product with the query.
  const int* block_table = block_tables + seq_idx * max_num_blocks_per_seq;
  for (int block_idx = start_block_idx + warp_idx; block_idx < end_block_idx; block_idx += NUM_WARPS) {
    // NOTE(woosuk): The block number is stored in int32. However, we cast it to int64
    // because int32 can lead to overflow when this variable is multiplied by large numbers
    // (e.g., kv_block_stride).
    const int64_t physical_block_number = static_cast<int64_t>(block_table[block_idx]);

    // Load a key to registers.
    // Each thread in a thread group has a different part of the key.
    // For example, if the the thread group size is 4, then the first thread in the group
    // has 0, 4, 8, ... th vectors of the key, and the second thread has 1, 5, 9, ... th
    // vectors of the key, and so on.
    for (int i = 0; i < NUM_TOKENS_PER_THREAD_GROUP; i++) {
      const int physical_block_offset = (thread_group_idx + i * WARP_SIZE) % BLOCK_SIZE;
      const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;
      K_vec k_vecs[NUM_VECS_PER_THREAD];

#pragma unroll
      for (int j = 0; j < NUM_VECS_PER_THREAD; j++) {
        const scalar_t* k_ptr = k_cache + physical_block_number * kv_block_stride
                                        + kv_head_idx * kv_head_stride
                                        + physical_block_offset * x;
        const int vec_idx = thread_group_offset + j * THREAD_GROUP_SIZE;
        const int offset1 = (vec_idx * VEC_SIZE) / x;
        const int offset2 = (vec_idx * VEC_SIZE) % x;
        k_vecs[j] = *reinterpret_cast<const K_vec*>(k_ptr + offset1 * BLOCK_SIZE * x + offset2);
      }

      // Compute dot product.
      // This includes a reduction across the threads in the same thread group.
      float qk = scale * Qk_dot<scalar_t, THREAD_GROUP_SIZE>::dot(q_vecs[thread_group_offset], k_vecs);
      // Add the ALiBi bias if slopes are given.
      qk += (alibi_slope != 0) ? alibi_slope * (token_idx - context_len + 1) : 0;

      if (thread_group_offset == 0) {
        // Store the partial reductions to shared memory.
        // NOTE(woosuk): It is required to zero out the masked logits.
        const bool mask = token_idx >= context_len;
        logits[token_idx - start_token_idx] = mask ? 0.f : qk;
        // Update the max value.
        qk_max = mask ? qk_max : sycl::fmax(qk_max, qk);
      }
    }
  }

  // Perform reduction across the threads in the same warp to get the
  // max qk value for each "warp" (not across the thread block yet).
  // The 0-th thread of each thread group already has its max qk value.
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= THREAD_GROUP_SIZE; mask /= 2) {
    /*
    DPCT1023:9: The SYCL sub-group does not support mask options for
    dpct::permute_sub_group_by_xor.
    */
    /*
    DPCT1096:38: The right-most dimension of the work-group used in the SYCL
    kernel that calls this function may be less than "32". The function
    "dpct::permute_sub_group_by_xor" may return an unexpected result on the CPU
    device. Modify the size of the work-group to ensure that the value of the
    right-most dimension is a multiple of "32".
    */
    qk_max = sycl::fmax(qk_max, dpct::experimental::permute_sub_group_by_xor(0xffffffff,
                                    item_ct1.get_sub_group(), qk_max, mask));
  }
  if (lane == 0) {
    red_smem[warp_idx] = qk_max;
  }
  /*
  DPCT1065:6: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();

  // TODO(woosuk): Refactor this part.
  // Get the max qk value for the sequence.
  qk_max = lane < NUM_WARPS ? red_smem[lane] : -FLT_MAX;
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    /*
    DPCT1023:10: The SYCL sub-group does not support mask options for
    dpct::permute_sub_group_by_xor.
    */
    /*
    DPCT1096:39: The right-most dimension of the work-group used in the SYCL
    kernel that calls this function may be less than "32". The function
    "dpct::permute_sub_group_by_xor" may return an unexpected result on the CPU
    device. Modify the size of the work-group to ensure that the value of the
    right-most dimension is a multiple of "32".
    */
    qk_max = sycl::fmax(qk_max, dpct::experimental::permute_sub_group_by_xor(0xffffffff,
                                    item_ct1.get_sub_group(), qk_max, mask));
  }
  // Broadcast the max qk value to all threads.
  /*
  DPCT1023:11: The SYCL sub-group does not support mask options for
  dpct::select_from_sub_group.
  */
  /*
  DPCT1096:40: The right-most dimension of the work-group used in the SYCL
  kernel that calls this function may be less than "32". The function
  "dpct::select_from_sub_group" may return an unexpected result on the CPU
  device. Modify the size of the work-group to ensure that the value of the
  right-most dimension is a multiple of "32".
  */
  qk_max = dpct::experimental::select_from_sub_group(0xffffffff, item_ct1.get_sub_group(), qk_max, 0);

  // Get the sum of the exp values.
  float exp_sum = 0.f;
  for (int i = thread_idx; i < num_tokens; i += NUM_THREADS) {
    float val = sycl::exp(logits[i] - qk_max);
    logits[i] = val;
    exp_sum += val;
  }
  exp_sum = block_sum<NUM_WARPS>(&red_smem[NUM_WARPS], exp_sum, item_ct1);

  // Compute softmax.
  const float inv_sum = 1.f / (exp_sum + 1e-6f);
  for (int i = thread_idx; i < num_tokens; i += NUM_THREADS) {
    logits[i] *= inv_sum;
  }
  /*
  DPCT1065:7: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();

  // If partitioning is enabled, store the max logit and exp_sum.
  if (USE_PARTITIONING && thread_idx == 0) {
    float* max_logits_ptr = max_logits + seq_idx * num_heads * max_num_partitions
                                       + head_idx * max_num_partitions
                                       + partition_idx;
    *max_logits_ptr = qk_max;
    float* exp_sums_ptr = exp_sums + seq_idx * num_heads * max_num_partitions
                                   + head_idx * max_num_partitions
                                   + partition_idx;
    *exp_sums_ptr = exp_sum;
  }

  // Each thread will fetch 16 bytes from the value cache at a time.
  constexpr int V_VEC_SIZE = MIN(16 / sizeof(scalar_t), BLOCK_SIZE);
  using V_vec = typename Vec<scalar_t, V_VEC_SIZE>::Type;
  using L_vec = typename Vec<scalar_t, V_VEC_SIZE>::Type;
  using Float_L_vec = typename FloatVec<L_vec>::Type;

  constexpr int NUM_V_VECS_PER_ROW = BLOCK_SIZE / V_VEC_SIZE;
  constexpr int NUM_ROWS_PER_ITER = WARP_SIZE / NUM_V_VECS_PER_ROW;
  constexpr int NUM_ROWS_PER_THREAD = DIVIDE_ROUND_UP(HEAD_SIZE, NUM_ROWS_PER_ITER);

  // NOTE(woosuk): We use FP32 for the accumulator for better accuracy.
  float accs[NUM_ROWS_PER_THREAD];
#pragma unroll
  for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
    accs[i] = 0.f;
  }

  scalar_t zero_value;
  zero(zero_value);
  for (int block_idx = start_block_idx + warp_idx; block_idx < end_block_idx; block_idx += NUM_WARPS) {
    // NOTE(woosuk): The block number is stored in int32. However, we cast it to int64
    // because int32 can lead to overflow when this variable is multiplied by large numbers
    // (e.g., kv_block_stride).
    const int64_t physical_block_number = static_cast<int64_t>(block_table[block_idx]);
    const int physical_block_offset = (lane % NUM_V_VECS_PER_ROW) * V_VEC_SIZE;
    const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;
    L_vec logits_vec;
    from_float(logits_vec, *reinterpret_cast<Float_L_vec*>(logits + token_idx - start_token_idx));

    const scalar_t* v_ptr = v_cache + physical_block_number * kv_block_stride
                                    + kv_head_idx * kv_head_stride;
#pragma unroll
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
      const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
      if (row_idx < HEAD_SIZE) {
        const int offset = row_idx * BLOCK_SIZE + physical_block_offset;
        V_vec v_vec = *reinterpret_cast<const V_vec*>(v_ptr + offset);
        if (block_idx == num_context_blocks - 1) {
          // NOTE(woosuk): When v_vec contains the tokens that are out of the context,
          // we should explicitly zero out the values since they may contain NaNs.
          // See https://github.com/vllm-project/vllm/issues/641#issuecomment-1682544472
          scalar_t* v_vec_ptr = reinterpret_cast<scalar_t*>(&v_vec);
#pragma unroll
          for (int j = 0; j < V_VEC_SIZE; j++) {
            v_vec_ptr[j] = token_idx + j < context_len ? v_vec_ptr[j] : zero_value;
          }
        }
        accs[i] += dot(logits_vec, v_vec);
      }
    }
  }

  // Perform reduction within each warp.
#pragma unroll
  for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
    float acc = accs[i];
#pragma unroll
    for (int mask = NUM_V_VECS_PER_ROW / 2; mask >= 1; mask /= 2) {
      /*
      DPCT1023:12: The SYCL sub-group does not support mask options for
      dpct::permute_sub_group_by_xor.
      */
      /*
      DPCT1096:41: The right-most dimension of the work-group used in the SYCL
      kernel that calls this function may be less than "32". The function
      "dpct::permute_sub_group_by_xor" may return an unexpected result on the
      CPU device. Modify the size of the work-group to ensure that the value of
      the right-most dimension is a multiple of "32".
      */
      acc +=
          dpct::experimental::permute_sub_group_by_xor(0xffffffff, item_ct1.get_sub_group(), acc, mask);
    }
    accs[i] = acc;
  }

  // NOTE(woosuk): A barrier is required because the shared memory space for logits
  // is reused for the output.
  /*
  DPCT1065:8: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();

  // Perform reduction across warps.
  float* out_smem = reinterpret_cast<float*>(shared_mem);
#pragma unroll
  for (int i = NUM_WARPS; i > 1; i /= 2) {
    int mid = i / 2;
    // Upper warps write to shared memory.
    if (warp_idx >= mid && warp_idx < i) {
      float* dst = &out_smem[(warp_idx - mid) * HEAD_SIZE];
#pragma unroll
      for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
        const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
        if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
          dst[row_idx] = accs[i];
        }
      }
    }
    /*
    DPCT1065:13: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    // Lower warps update the output.
    if (warp_idx < mid) {
      const float* src = &out_smem[warp_idx * HEAD_SIZE];
#pragma unroll
      for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
        const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
        if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
          accs[i] += src[row_idx];
        }
      }
    }
    /*
    DPCT1065:14: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
  }

  // Write the final output.
  if (warp_idx == 0) {
    scalar_t* out_ptr = out + seq_idx * num_heads * max_num_partitions * HEAD_SIZE
                            + head_idx * max_num_partitions * HEAD_SIZE
                            + partition_idx * HEAD_SIZE;
#pragma unroll
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
      const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
      if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
        from_float(*(out_ptr + row_idx), accs[i]);
      }
    }
  }
}

// Grid: (num_heads, num_seqs, 1).
template<
  typename scalar_t,
  int HEAD_SIZE,
  int BLOCK_SIZE,
  int NUM_THREADS>
void paged_attention_v1_kernel(
  scalar_t* __restrict__ out,             // [num_seqs, num_heads, head_size]
  const scalar_t* __restrict__ q,         // [num_seqs, num_heads, head_size]
  const scalar_t* __restrict__ k_cache,   // [num_blocks, num_kv_heads, head_size/x, block_size, x]
  const scalar_t* __restrict__ v_cache,   // [num_blocks, num_kv_heads, head_size, block_size]
  const int* __restrict__ head_mapping,   // [num_heads]
  const float scale,
  const int* __restrict__ block_tables,   // [num_seqs, max_num_blocks_per_seq]
  const int* __restrict__ context_lens,   // [num_seqs]
  const int max_num_blocks_per_seq,
  const float* __restrict__ alibi_slopes, // [num_heads]
  const int q_stride,
  const int kv_block_stride,
  const int kv_head_stride,
  const sycl::nd_item<3> &item_ct1,
  uint8_t *dpct_local,
  sycl::local_accessor<Q_vec, 2> q_vecs,
  float *red_smem) {
  paged_attention_kernel<scalar_t, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS>(
      /* exp_sums */ nullptr, /* max_logits */ nullptr, out, q, k_cache,
      v_cache, head_mapping, scale, block_tables, context_lens,
      max_num_blocks_per_seq, alibi_slopes, q_stride, kv_block_stride,
      kv_head_stride, item_ct1, dpct_local, q_vecs, red_smem);
}

// Grid: (num_heads, num_seqs, max_num_partitions).
template<
  typename scalar_t,
  int HEAD_SIZE,
  int BLOCK_SIZE,
  int NUM_THREADS,
  int PARTITION_SIZE>
void paged_attention_v2_kernel(
  float* __restrict__ exp_sums,           // [num_seqs, num_heads, max_num_partitions]
  float* __restrict__ max_logits,         // [num_seqs, num_heads, max_num_partitions]
  scalar_t* __restrict__ tmp_out,         // [num_seqs, num_heads, max_num_partitions, head_size]
  const scalar_t* __restrict__ q,         // [num_seqs, num_heads, head_size]
  const scalar_t* __restrict__ k_cache,   // [num_blocks, num_kv_heads, head_size/x, block_size, x]
  const scalar_t* __restrict__ v_cache,   // [num_blocks, num_kv_heads, head_size, block_size]
  const int* __restrict__ head_mapping,   // [num_heads]
  const float scale,
  const int* __restrict__ block_tables,   // [num_seqs, max_num_blocks_per_seq]
  const int* __restrict__ context_lens,   // [num_seqs]
  const int max_num_blocks_per_seq,
  const float* __restrict__ alibi_slopes, // [num_heads]
  const int q_stride,
  const int kv_block_stride,
  const int kv_head_stride,
  const sycl::nd_item<3> &item_ct1,
  uint8_t *dpct_local,
  sycl::local_accessor<Q_vec, 2> q_vecs,
  float *red_smem) {
  paged_attention_kernel<scalar_t, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS,
                         PARTITION_SIZE>(
      exp_sums, max_logits, tmp_out, q, k_cache, v_cache, head_mapping, scale,
      block_tables, context_lens, max_num_blocks_per_seq, alibi_slopes,
      q_stride, kv_block_stride, kv_head_stride, item_ct1, dpct_local, q_vecs,
      red_smem);
}

// Grid: (num_heads, num_seqs).
template<
  typename scalar_t,
  int HEAD_SIZE,
  int NUM_THREADS,
  int PARTITION_SIZE>
void paged_attention_v2_reduce_kernel(
  scalar_t* __restrict__ out,             // [num_seqs, num_heads, head_size]
  const float* __restrict__ exp_sums,     // [num_seqs, num_heads, max_num_partitions]
  const float* __restrict__ max_logits,   // [num_seqs, num_heads, max_num_partitions]
  const scalar_t* __restrict__ tmp_out,   // [num_seqs, num_heads, max_num_partitions, head_size]
  const int* __restrict__ context_lens,   // [num_seqs]
  const int max_num_partitions,
  const sycl::nd_item<3> &item_ct1,
  uint8_t *dpct_local,
  float *red_smem) {
  const int num_heads = item_ct1.get_group_range(2);
  const int head_idx = item_ct1.get_group(2);
  const int seq_idx = item_ct1.get_group(1);
  const int context_len = context_lens[seq_idx];
  const int num_partitions = DIVIDE_ROUND_UP(context_len, PARTITION_SIZE);
  if (num_partitions == 1) {
    // No need to reduce. Only copy tmp_out to out.
    scalar_t* out_ptr = out + seq_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE;
    const scalar_t* tmp_out_ptr = tmp_out + seq_idx * num_heads * max_num_partitions * HEAD_SIZE
                                          + head_idx * max_num_partitions * HEAD_SIZE;
    for (int i = item_ct1.get_local_id(2); i < HEAD_SIZE;
         i += item_ct1.get_local_range(2)) {
      out_ptr[i] = tmp_out_ptr[i];
    }
    // Terminate the thread block.
    return;
  }

  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  const int warp_idx = item_ct1.get_local_id(2) / WARP_SIZE;
  const int lane = item_ct1.get_local_id(2) % WARP_SIZE;

  // Size: 2 * num_partitions.
  auto shared_mem = (char *)dpct_local;
  // Workspace for reduction.

  // Load max logits to shared memory.
  float* shared_max_logits = reinterpret_cast<float*>(shared_mem);
  const float* max_logits_ptr = max_logits + seq_idx * num_heads * max_num_partitions
                                           + head_idx * max_num_partitions;
  float max_logit = -FLT_MAX;
  for (int i = item_ct1.get_local_id(2); i < num_partitions;
       i += item_ct1.get_local_range(2)) {
    const float l = max_logits_ptr[i];
    shared_max_logits[i] = l;
    max_logit = sycl::fmax(max_logit, (float)l);
  }
  /*
  DPCT1065:15: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();

  // Get the global max logit.
  // Reduce within the warp.
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
    /*
    DPCT1023:18: The SYCL sub-group does not support mask options for
    dpct::permute_sub_group_by_xor.
    */
    /*
    DPCT1096:45: The right-most dimension of the work-group used in the SYCL
    kernel that calls this function may be less than "32". The function
    "dpct::permute_sub_group_by_xor" may return an unexpected result on the CPU
    device. Modify the size of the work-group to ensure that the value of the
    right-most dimension is a multiple of "32".
    */
    max_logit =
        sycl::fmax(max_logit, dpct::experimental::permute_sub_group_by_xor(0xffffffff,
                                  item_ct1.get_sub_group(), max_logit, mask));
  }
  if (lane == 0) {
    red_smem[warp_idx] = max_logit;
  }
  /*
  DPCT1065:16: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();
  // Reduce across warps.
  max_logit = lane < NUM_WARPS ? red_smem[lane] : -FLT_MAX;
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    /*
    DPCT1023:19: The SYCL sub-group does not support mask options for
    dpct::permute_sub_group_by_xor.
    */
    /*
    DPCT1096:46: The right-most dimension of the work-group used in the SYCL
    kernel that calls this function may be less than "32". The function
    "dpct::permute_sub_group_by_xor" may return an unexpected result on the CPU
    device. Modify the size of the work-group to ensure that the value of the
    right-most dimension is a multiple of "32".
    */
    max_logit =
        sycl::fmax(max_logit, dpct::experimental::permute_sub_group_by_xor(0xffffffff,
                                  item_ct1.get_sub_group(), max_logit, mask));
  }
  // Broadcast the max value to all threads.
  /*
  DPCT1023:20: The SYCL sub-group does not support mask options for
  dpct::select_from_sub_group.
  */
  /*
  DPCT1096:47: The right-most dimension of the work-group used in the SYCL
  kernel that calls this function may be less than "32". The function
  "dpct::select_from_sub_group" may return an unexpected result on the CPU
  device. Modify the size of the work-group to ensure that the value of the
  right-most dimension is a multiple of "32".
  */
  max_logit =
      dpct::experimental::select_from_sub_group(0xffffffff, item_ct1.get_sub_group(), max_logit, 0);

  // Load rescaled exp sums to shared memory.
  float* shared_exp_sums = reinterpret_cast<float*>(shared_mem + sizeof(float) * num_partitions);
  const float* exp_sums_ptr = exp_sums + seq_idx * num_heads * max_num_partitions
                                       + head_idx * max_num_partitions;
  float global_exp_sum = 0.0f;
  for (int i = item_ct1.get_local_id(2); i < num_partitions;
       i += item_ct1.get_local_range(2)) {
    float l = shared_max_logits[i];
    float rescaled_exp_sum = exp_sums_ptr[i] * sycl::exp(l - max_logit);
    global_exp_sum += rescaled_exp_sum;
    shared_exp_sums[i] = rescaled_exp_sum;
  }
  /*
  DPCT1065:17: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();
  global_exp_sum =
      block_sum<NUM_WARPS>(&red_smem[NUM_WARPS], global_exp_sum, item_ct1);
  const float inv_global_exp_sum = 1.0f / (global_exp_sum + 1e-6f);

  // Aggregate tmp_out to out.
  const scalar_t* tmp_out_ptr = tmp_out + seq_idx * num_heads * max_num_partitions * HEAD_SIZE
                                        + head_idx * max_num_partitions * HEAD_SIZE;
  scalar_t* out_ptr = out + seq_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE;
#pragma unroll
  for (int i = item_ct1.get_local_id(2); i < HEAD_SIZE; i += NUM_THREADS) {
    float acc = 0.0f;
    for (int j = 0; j < num_partitions; ++j) {
      acc += to_float(tmp_out_ptr[j * HEAD_SIZE + i]) * shared_exp_sums[j] * inv_global_exp_sum;
    }
    from_float(out_ptr[i], acc);
  }
}

} // namespace vllm

/*
DPCT1049:21: The work-group size passed to the SYCL kernel may exceed the limit.
To get the device limit, query info::device::max_work_group_size. Adjust the
work-group size if needed.
*/
#define LAUNCH_PAGED_ATTENTION_V1(HEAD_SIZE)                                   \
   cudaFuncSetAttribute(                                                       \
       vllm::paged_attention_v1_kernel<T, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS>, \
       cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size);          \
   stream->submit([&](sycl::handler &cgh) {                                    \
      sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(                     \
          sycl::range<1>(shared_mem_size), cgh);                               \
      sycl::local_accessor<Q_vec, 2> q_vecs_acc_ct1(                           \
          sycl::range<2>(THREAD_GROUP_SIZE, NUM_VECS_PER_THREAD), cgh);        \
      sycl::local_accessor<float, 1> red_smem_acc_ct1(                         \
          sycl::range<1>(2 * NUM_WARPS), cgh);                                 \
                                                                               \
      auto out_ptr_ct0 = out_ptr;                                              \
      auto query_ptr_ct1 = query_ptr;                                          \
      auto key_cache_ptr_ct2 = key_cache_ptr;                                  \
      auto value_cache_ptr_ct3 = value_cache_ptr;                              \
      auto head_mapping_ptr_ct4 = head_mapping_ptr;                            \
      auto scale_ct5 = scale;                                                  \
      auto block_tables_ptr_ct6 = block_tables_ptr;                            \
      auto context_lens_ptr_ct7 = context_lens_ptr;                            \
      auto max_num_blocks_per_seq_ct8 = max_num_blocks_per_seq;                \
      auto alibi_slopes_ptr_ct9 = alibi_slopes_ptr;                            \
      auto q_stride_ct10 = q_stride;                                           \
      auto kv_block_stride_ct11 = kv_block_stride;                             \
      auto kv_head_stride_ct12 = kv_head_stride;                               \
                                                                               \
      cgh.parallel_for(                                                        \
          sycl::nd_range<3>(grid * block, block),                              \
          [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {  \
             vllm::paged_attention_v1_kernel<T, HEAD_SIZE, BLOCK_SIZE,         \
                                             NUM_THREADS>(                     \
                 out_ptr_ct0, query_ptr_ct1, key_cache_ptr_ct2,                \
                 value_cache_ptr_ct3, head_mapping_ptr_ct4, scale_ct5,         \
                 block_tables_ptr_ct6, context_lens_ptr_ct7,                   \
                 max_num_blocks_per_seq_ct8, alibi_slopes_ptr_ct9,             \
                 q_stride_ct10, kv_block_stride_ct11, kv_head_stride_ct12,     \
                 item_ct1, dpct_local_acc_ct1.get_pointer(), q_vecs_acc_ct1,   \
                 red_smem_acc_ct1.get_pointer());                              \
          });                                                                  \
   });

// TODO(woosuk): Tune NUM_THREADS.
template<
  typename T,
  int BLOCK_SIZE,
  int NUM_THREADS = 128>
void paged_attention_v1_launcher(
  torch::Tensor& out,
  torch::Tensor& query,
  torch::Tensor& key_cache,
  torch::Tensor& value_cache,
  torch::Tensor& head_mapping,
  float scale,
  torch::Tensor& block_tables,
  torch::Tensor& context_lens,
  int max_context_len,
  const c10::optional<torch::Tensor>& alibi_slopes) {
  int num_seqs = query.size(0);
  int num_heads = query.size(1);
  int head_size = query.size(2);
  int max_num_blocks_per_seq = block_tables.size(1);
  int q_stride = query.stride(0);
  int kv_block_stride = key_cache.stride(0);
  int kv_head_stride = key_cache.stride(1);

  int thread_group_size = MAX(WARP_SIZE / BLOCK_SIZE, 1);
  assert(head_size % thread_group_size == 0);

  // NOTE: alibi_slopes is optional.
  const float* alibi_slopes_ptr = alibi_slopes ?
    reinterpret_cast<const float*>(alibi_slopes.value().data_ptr())
    : nullptr;

  T* out_ptr = reinterpret_cast<T*>(out.data_ptr());
  T* query_ptr = reinterpret_cast<T*>(query.data_ptr());
  T* key_cache_ptr = reinterpret_cast<T*>(key_cache.data_ptr());
  T* value_cache_ptr = reinterpret_cast<T*>(value_cache.data_ptr());
  int* head_mapping_ptr = reinterpret_cast<int*>(head_mapping.data_ptr());
  int* block_tables_ptr = block_tables.data_ptr<int>();
  int* context_lens_ptr = context_lens.data_ptr<int>();

  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  int padded_max_context_len = DIVIDE_ROUND_UP(max_context_len, BLOCK_SIZE) * BLOCK_SIZE;
  /*
  DPCT1083:22: The size of local memory in the migrated code may be different
  from the original code. Check that the allocated memory size in the migrated
  code is correct.
  */
  int logits_size = padded_max_context_len * sizeof(float);
  int outputs_size = (NUM_WARPS / 2) * head_size * sizeof(float);
  // Python-side check in vllm.worker.worker._check_if_can_support_max_seq_len
  // Keep that in sync with the logic here!
  int shared_mem_size = std::max(logits_size, outputs_size);

  sycl::range<3> grid(1, num_seqs, num_heads);
  sycl::range<3> block(1, 1, NUM_THREADS);
  const dpct::queue_ptr stream = at::cuda::getCurrentCUDAStream();
  switch (head_size) {
    // NOTE(woosuk): To reduce the compilation time, we only compile for the
    // head sizes that we use in the model. However, we can easily extend this
    // to support any head size which is a multiple of 16.
    case 64:
      LAUNCH_PAGED_ATTENTION_V1(64);
      break;
    case 80:
      LAUNCH_PAGED_ATTENTION_V1(80);
      break;
    case 96:
      LAUNCH_PAGED_ATTENTION_V1(96);
      break;
    case 112:
      LAUNCH_PAGED_ATTENTION_V1(112);
      break;
    case 128:
      LAUNCH_PAGED_ATTENTION_V1(128);
      break;
    case 256:
      LAUNCH_PAGED_ATTENTION_V1(256);
      break;
    default:
      TORCH_CHECK(false, "Unsupported head size: ", head_size);
      break;
  }
}

#define CALL_V1_LAUNCHER(T, BLOCK_SIZE)                             \
  paged_attention_v1_launcher<T, BLOCK_SIZE>(                       \
    out,                                                            \
    query,                                                          \
    key_cache,                                                      \
    value_cache,                                                    \
    head_mapping,                                                   \
    scale,                                                          \
    block_tables,                                                   \
    context_lens,                                                   \
    max_context_len,                                                \
    alibi_slopes);

// NOTE(woosuk): To reduce the compilation time, we omitted block sizes
// 1, 2, 4, 64, 128, 256.
#define CALL_V1_LAUNCHER_BLOCK_SIZE(T)                              \
  switch (block_size) {                                             \
    case 8:                                                         \
      CALL_V1_LAUNCHER(T, 8);                                       \
      break;                                                        \
    case 16:                                                        \
      CALL_V1_LAUNCHER(T, 16);                                      \
      break;                                                        \
    case 32:                                                        \
      CALL_V1_LAUNCHER(T, 32);                                      \
      break;                                                        \
    default:                                                        \
      TORCH_CHECK(false, "Unsupported block size: ", block_size);   \
      break;                                                        \
  }

void paged_attention_v1(
  torch::Tensor& out,             // [num_seqs, num_heads, head_size]
  torch::Tensor& query,           // [num_seqs, num_heads, head_size]
  torch::Tensor& key_cache,       // [num_blocks, num_heads, head_size/x, block_size, x]
  torch::Tensor& value_cache,     // [num_blocks, num_heads, head_size, block_size]
  torch::Tensor& head_mapping,    // [num_heads]
  float scale,
  torch::Tensor& block_tables,    // [num_seqs, max_num_blocks_per_seq]
  torch::Tensor& context_lens,    // [num_seqs]
  int block_size,
  int max_context_len,
  const c10::optional<torch::Tensor>& alibi_slopes) {
  if (query.dtype() == at::ScalarType::Float) {
    CALL_V1_LAUNCHER_BLOCK_SIZE(float);
  } else if (query.dtype() == at::ScalarType::Half) {
    CALL_V1_LAUNCHER_BLOCK_SIZE(uint16_t);
  } else if (query.dtype() == at::ScalarType::BFloat16) {
    CALL_V1_LAUNCHER_BLOCK_SIZE(oneapi::mkl::bfloat16);
  } else {
    TORCH_CHECK(false, "Unsupported data type: ", query.dtype());
  }
}

/*
DPCT1049:23: The work-group size passed to the SYCL kernel may exceed the limit.
To get the device limit, query info::device::max_work_group_size. Adjust the
work-group size if needed.
*/
/*
DPCT1049:25: The work-group size passed to the SYCL kernel may exceed the limit.
To get the device limit, query info::device::max_work_group_size. Adjust the
work-group size if needed.
*/
#define LAUNCH_PAGED_ATTENTION_V2(HEAD_SIZE)                                   \
   stream->submit([&](sycl::handler &cgh) {                                    \
      sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(                     \
          sycl::range<1>(shared_mem_size), cgh);                               \
      sycl::local_accessor<Q_vec, 2> q_vecs_acc_ct1(                           \
          sycl::range<2>(THREAD_GROUP_SIZE, NUM_VECS_PER_THREAD), cgh);        \
      sycl::local_accessor<float, 1> red_smem_acc_ct1(                         \
          sycl::range<1>(2 * NUM_WARPS), cgh);                                 \
                                                                               \
      auto exp_sums_ptr_ct0 = exp_sums_ptr;                                    \
      auto max_logits_ptr_ct1 = max_logits_ptr;                                \
      auto tmp_out_ptr_ct2 = tmp_out_ptr;                                      \
      auto query_ptr_ct3 = query_ptr;                                          \
      auto key_cache_ptr_ct4 = key_cache_ptr;                                  \
      auto value_cache_ptr_ct5 = value_cache_ptr;                              \
      auto head_mapping_ptr_ct6 = head_mapping_ptr;                            \
      auto scale_ct7 = scale;                                                  \
      auto block_tables_ptr_ct8 = block_tables_ptr;                            \
      auto context_lens_ptr_ct9 = context_lens_ptr;                            \
      auto max_num_blocks_per_seq_ct10 = max_num_blocks_per_seq;               \
      auto alibi_slopes_ptr_ct11 = alibi_slopes_ptr;                           \
      auto q_stride_ct12 = q_stride;                                           \
      auto kv_block_stride_ct13 = kv_block_stride;                             \
      auto kv_head_stride_ct14 = kv_head_stride;                               \
                                                                               \
      cgh.parallel_for(                                                        \
          sycl::nd_range<3>(grid * block, block),                              \
          [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {  \
             vllm::paged_attention_v2_kernel<T, HEAD_SIZE, BLOCK_SIZE,         \
                                             NUM_THREADS, PARTITION_SIZE>(     \
                 exp_sums_ptr_ct0, max_logits_ptr_ct1, tmp_out_ptr_ct2,        \
                 query_ptr_ct3, key_cache_ptr_ct4, value_cache_ptr_ct5,        \
                 head_mapping_ptr_ct6, scale_ct7, block_tables_ptr_ct8,        \
                 context_lens_ptr_ct9, max_num_blocks_per_seq_ct10,            \
                 alibi_slopes_ptr_ct11, q_stride_ct12, kv_block_stride_ct13,   \
                 kv_head_stride_ct14, item_ct1,                                \
                 dpct_local_acc_ct1.get_pointer(), q_vecs_acc_ct1,             \
                 red_smem_acc_ct1.get_pointer());                              \
          });                                                                  \
   });                                                                         \
   stream->submit([&](sycl::handler &cgh) {                                    \
      sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(                     \
          sycl::range<1>(reduce_shared_mem_size), cgh);                        \
      sycl::local_accessor<float, 1> red_smem_acc_ct1(                         \
          sycl::range<1>(2 * NUM_WARPS), cgh);                                 \
                                                                               \
      auto out_ptr_ct0 = out_ptr;                                              \
      auto exp_sums_ptr_ct1 = exp_sums_ptr;                                    \
      auto max_logits_ptr_ct2 = max_logits_ptr;                                \
      auto tmp_out_ptr_ct3 = tmp_out_ptr;                                      \
      auto context_lens_ptr_ct4 = context_lens_ptr;                            \
      auto max_num_partitions_ct5 = max_num_partitions;                        \
                                                                               \
      cgh.parallel_for(                                                        \
          sycl::nd_range<3>(reduce_grid * block, block),                       \
          [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {  \
             vllm::paged_attention_v2_reduce_kernel<T, HEAD_SIZE, NUM_THREADS, \
                                                    PARTITION_SIZE>(           \
                 out_ptr_ct0, exp_sums_ptr_ct1, max_logits_ptr_ct2,            \
                 tmp_out_ptr_ct3, context_lens_ptr_ct4,                        \
                 max_num_partitions_ct5, item_ct1,                             \
                 dpct_local_acc_ct1.get_pointer(),                             \
                 red_smem_acc_ct1.get_pointer());                              \
          });                                                                  \
   });

template<
  typename T,
  int BLOCK_SIZE,
  int NUM_THREADS = 128,
  int PARTITION_SIZE = 512>
void paged_attention_v2_launcher(
  torch::Tensor& out,
  torch::Tensor& exp_sums,
  torch::Tensor& max_logits,
  torch::Tensor& tmp_out,
  torch::Tensor& query,
  torch::Tensor& key_cache,
  torch::Tensor& value_cache,
  torch::Tensor& head_mapping,
  float scale,
  torch::Tensor& block_tables,
  torch::Tensor& context_lens,
  int max_context_len,
  const c10::optional<torch::Tensor>& alibi_slopes) {
  int num_seqs = query.size(0);
  int num_heads = query.size(1);
  int head_size = query.size(2);
  int max_num_blocks_per_seq = block_tables.size(1);
  int q_stride = query.stride(0);
  int kv_block_stride = key_cache.stride(0);
  int kv_head_stride = key_cache.stride(1);

  int thread_group_size = MAX(WARP_SIZE / BLOCK_SIZE, 1);
  assert(head_size % thread_group_size == 0);

  // NOTE: alibi_slopes is optional.
  const float* alibi_slopes_ptr = alibi_slopes ?
    reinterpret_cast<const float*>(alibi_slopes.value().data_ptr())
    : nullptr;

  T* out_ptr = reinterpret_cast<T*>(out.data_ptr());
  float* exp_sums_ptr = reinterpret_cast<float*>(exp_sums.data_ptr());
  float* max_logits_ptr = reinterpret_cast<float*>(max_logits.data_ptr());
  T* tmp_out_ptr = reinterpret_cast<T*>(tmp_out.data_ptr());
  T* query_ptr = reinterpret_cast<T*>(query.data_ptr());
  T* key_cache_ptr = reinterpret_cast<T*>(key_cache.data_ptr());
  T* value_cache_ptr = reinterpret_cast<T*>(value_cache.data_ptr());
  int* head_mapping_ptr = reinterpret_cast<int*>(head_mapping.data_ptr());
  int* block_tables_ptr = block_tables.data_ptr<int>();
  int* context_lens_ptr = context_lens.data_ptr<int>();

  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  int max_num_partitions = DIVIDE_ROUND_UP(max_context_len, PARTITION_SIZE);
  /*
  DPCT1083:26: The size of local memory in the migrated code may be different
  from the original code. Check that the allocated memory size in the migrated
  code is correct.
  */
  int logits_size = PARTITION_SIZE * sizeof(float);
  int outputs_size = (NUM_WARPS / 2) * head_size * sizeof(float);

  // For paged attention v2 kernel.
  sycl::range<3> grid(max_num_partitions, num_seqs, num_heads);
  int shared_mem_size = std::max(logits_size, outputs_size);
  // For paged attention v2 reduce kernel.
  sycl::range<3> reduce_grid(1, num_seqs, num_heads);
  /*
  DPCT1083:24: The size of local memory in the migrated code may be different
  from the original code. Check that the allocated memory size in the migrated
  code is correct.
  */
  int reduce_shared_mem_size = 2 * max_num_partitions * sizeof(float);

  sycl::range<3> block(1, 1, NUM_THREADS);
  const dpct::queue_ptr stream = at::cuda::getCurrentCUDAStream();
  switch (head_size) {
    // NOTE(woosuk): To reduce the compilation time, we only compile for the
    // head sizes that we use in the model. However, we can easily extend this
    // to support any head size which is a multiple of 16.
    case 64:
      LAUNCH_PAGED_ATTENTION_V2(64);
      break;
    case 80:
      LAUNCH_PAGED_ATTENTION_V2(80);
      break;
    case 96:
      LAUNCH_PAGED_ATTENTION_V2(96);
      break;
    case 112:
      LAUNCH_PAGED_ATTENTION_V2(112);
      break;
    case 128:
      LAUNCH_PAGED_ATTENTION_V2(128);
      break;
    case 256:
      LAUNCH_PAGED_ATTENTION_V2(256);
      break;
    default:
      TORCH_CHECK(false, "Unsupported head size: ", head_size);
      break;
  }
}

#define CALL_V2_LAUNCHER(T, BLOCK_SIZE)                             \
  paged_attention_v2_launcher<T, BLOCK_SIZE>(                       \
    out,                                                            \
    exp_sums,                                                       \
    max_logits,                                                     \
    tmp_out,                                                        \
    query,                                                          \
    key_cache,                                                      \
    value_cache,                                                    \
    head_mapping,                                                   \
    scale,                                                          \
    block_tables,                                                   \
    context_lens,                                                   \
    max_context_len,                                                \
    alibi_slopes);

// NOTE(woosuk): To reduce the compilation time, we omitted block sizes
// 1, 2, 4, 64, 128, 256.
#define CALL_V2_LAUNCHER_BLOCK_SIZE(T)                              \
  switch (block_size) {                                             \
    case 8:                                                         \
      CALL_V2_LAUNCHER(T, 8);                                       \
      break;                                                        \
    case 16:                                                        \
      CALL_V2_LAUNCHER(T, 16);                                      \
      break;                                                        \
    case 32:                                                        \
      CALL_V2_LAUNCHER(T, 32);                                      \
      break;                                                        \
    default:                                                        \
      TORCH_CHECK(false, "Unsupported block size: ", block_size);   \
      break;                                                        \
  }

void paged_attention_v2(
  torch::Tensor& out,             // [num_seqs, num_heads, head_size]
  torch::Tensor& exp_sums,        // [num_seqs, num_heads, max_num_partitions]
  torch::Tensor& max_logits,      // [num_seqs, num_heads, max_num_partitions]
  torch::Tensor& tmp_out,         // [num_seqs, num_heads, max_num_partitions, head_size]
  torch::Tensor& query,           // [num_seqs, num_heads, head_size]
  torch::Tensor& key_cache,       // [num_blocks, num_heads, head_size/x, block_size, x]
  torch::Tensor& value_cache,     // [num_blocks, num_heads, head_size, block_size]
  torch::Tensor& head_mapping,    // [num_heads]
  float scale,
  torch::Tensor& block_tables,    // [num_seqs, max_num_blocks_per_seq]
  torch::Tensor& context_lens,    // [num_seqs]
  int block_size,
  int max_context_len,
  const c10::optional<torch::Tensor>& alibi_slopes) {
  if (query.dtype() == at::ScalarType::Float) {
    CALL_V2_LAUNCHER_BLOCK_SIZE(float);
  } else if (query.dtype() == at::ScalarType::Half) {
    CALL_V2_LAUNCHER_BLOCK_SIZE(uint16_t);
  } else if (query.dtype() == at::ScalarType::BFloat16) {
    CALL_V2_LAUNCHER_BLOCK_SIZE(oneapi::mkl::bfloat16);
  } else {
    TORCH_CHECK(false, "Unsupported data type: ", query.dtype());
  }
}

#undef WARP_SIZE
#undef MAX
#undef MIN
#undef DIVIDE_ROUND_UP
