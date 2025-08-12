#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "attention_utils.h"
#include "cuda_primitives.h"
#include "reduction_utils.h"

#include <algorithm>

#define WARP_SIZE 32
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

namespace cacheflow {

// Grid: (num_heads, num_seqs).
template<
  typename scalar_t,
  int HEAD_SIZE,
  int BLOCK_SIZE,
  int NUM_THREADS>
__global__ void single_query_cached_kv_attention_kernel(
  scalar_t* __restrict__ out,             // [num_seqs, num_heads, head_size]
  const scalar_t* __restrict__ q,         // [num_seqs, num_heads, head_size]
  const scalar_t* __restrict__ k_cache,   // [num_blocks, num_heads, head_size/x, block_size, x]
  const scalar_t* __restrict__ v_cache,   // [num_blocks, num_heads, head_size, block_size]
  const float scale,
  const int* __restrict__ block_tables,   // [num_seqs, max_num_blocks_per_seq]
  const int* __restrict__ context_lens,   // [num_seqs]
  const int max_num_blocks_per_seq,
  const int q_stride) {
  constexpr int THREAD_GROUP_SIZE = MAX(WARP_SIZE / BLOCK_SIZE, 1);
  constexpr int NUM_TOKENS_PER_THREAD_GROUP = (BLOCK_SIZE + WARP_SIZE - 1) / WARP_SIZE;
  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  const int thread_idx = threadIdx.x;
  const int warp_idx = thread_idx / WARP_SIZE;
  const int lane = thread_idx % WARP_SIZE;

  const int head_idx = blockIdx.x;
  const int num_heads = gridDim.x;
  const int seq_idx = blockIdx.y;

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
  Q_vec q_vecs[NUM_VECS_PER_THREAD];
#pragma unroll
  for (int i = 0; i < NUM_VECS_PER_THREAD; i++) {
    const int vec_idx = thread_group_offset + i * THREAD_GROUP_SIZE;
    q_vecs[i] = *reinterpret_cast<const Q_vec*>(q_ptr + vec_idx * VEC_SIZE);
  }

  // Memory planning.
  extern __shared__ char shared_mem[];
  // NOTE(woosuk): We use FP32 logits and accumulation.
  float *logits = reinterpret_cast<float*>(shared_mem);
  // Workspace for reduction.
  __shared__ float red_smem[2 * NUM_WARPS];

  // x == THREAD_GROUP_SIZE * VEC_SIZE
  // Each thread group fetches x elements from the key at a time.
  constexpr int x = 16 / sizeof(scalar_t);
  float qk_max = -FLT_MAX;

  const int* block_table = block_tables + seq_idx * max_num_blocks_per_seq;
  const int context_len = context_lens[seq_idx];
  const int num_blocks = (context_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

  // Iterate over the key blocks.
  // Each warp fetches a block of keys for each iteration.
  // Each thread group in a warp fetches a key from the block, and computes
  // dot product with the query.
  for (int block_idx = warp_idx; block_idx < num_blocks; block_idx += NUM_WARPS) {
    const int physical_block_number = block_table[block_idx];

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
        const scalar_t* k_ptr = k_cache + physical_block_number * num_heads * HEAD_SIZE * BLOCK_SIZE
                                        + head_idx * HEAD_SIZE * BLOCK_SIZE
                                        + physical_block_offset * x;
        const int vec_idx = thread_group_offset + j * THREAD_GROUP_SIZE;
        const int offset1 = (vec_idx * VEC_SIZE) / x;
        const int offset2 = (vec_idx * VEC_SIZE) % x;
        k_vecs[j] = *reinterpret_cast<const K_vec*>(k_ptr + offset1 * BLOCK_SIZE * x + offset2);
      }

      // Compute dot product.
      // This includes a reduction across the threads in the same thread group.
      const float qk = scale * Qk_dot<scalar_t, THREAD_GROUP_SIZE>::dot(q_vecs, k_vecs);
      const bool mask = token_idx >= context_len;
    
      if (thread_group_offset == 0) {
        // Store the partial reductions to shared memory.
        // NOTE(woosuk): It is required to zero out the masked logits.
        logits[token_idx] = mask ? 0.f : qk;
        // Update the max value.
        qk_max = mask ? qk_max : fmaxf(qk_max, qk);
      }
    }
  }

  // Perform reduction across the threads in the same warp to get the
  // max qk value for each "warp" (not across the thread block yet).
  // The 0-th thread of each thread group already has its max qk value.
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= THREAD_GROUP_SIZE; mask /= 2) {
    qk_max = fmaxf(qk_max, __shfl_xor_sync(uint32_t(-1), qk_max, mask));
  }
  if (lane == 0) {
    red_smem[warp_idx] = qk_max;
  }
  __syncthreads();

  // TODO(woosuk): Refactor this part.
  // Get the max qk value for the sequence.
  qk_max = lane < NUM_WARPS ? red_smem[lane] : -FLT_MAX;
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
      qk_max = fmaxf(qk_max, __shfl_xor_sync(uint32_t(-1), qk_max, mask));
  }
  // Broadcast the max qk value to all threads.
  qk_max = __shfl_sync(uint32_t(-1), qk_max, 0);

  // Get the sum of the exp values.
  float exp_sum = 0.f;
  for (int i = thread_idx; i < context_len; i += NUM_THREADS) {
    float val = __expf(logits[i] - qk_max);
    logits[i] = val;
    exp_sum += val;
  }
  exp_sum = block_sum<NUM_WARPS>(&red_smem[NUM_WARPS], exp_sum);

  // Compute softmax.
  const float inv_sum = __fdividef(1.f, exp_sum + 1e-6f);
  for (int i = thread_idx; i < context_len; i += NUM_THREADS) {
    logits[i] *= inv_sum;
  }
  __syncthreads();

  // Each thread will fetch 16 bytes from the value cache at a time.
  constexpr int V_VEC_SIZE = MIN(16 / sizeof(scalar_t), BLOCK_SIZE);
  using V_vec = typename Vec<scalar_t, V_VEC_SIZE>::Type;
  using L_vec = typename FloatVec<V_vec>::Type;

  constexpr int NUM_V_VECS_PER_ROW = BLOCK_SIZE / V_VEC_SIZE;
  constexpr int NUM_ROWS_PER_ITER = WARP_SIZE / NUM_V_VECS_PER_ROW;
  constexpr int NUM_ROWS_PER_THREAD = (HEAD_SIZE + NUM_ROWS_PER_ITER - 1) / NUM_ROWS_PER_ITER;

  float accs[NUM_ROWS_PER_THREAD];
#pragma unroll
  for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
    accs[i] = 0.f;
  }

  for (int block_idx = warp_idx; block_idx < num_blocks; block_idx += NUM_WARPS) {
    const int physical_block_number = block_table[block_idx];
    const int physical_block_offset = (lane % NUM_V_VECS_PER_ROW) * V_VEC_SIZE;
    const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;
    L_vec logits_vec = *reinterpret_cast<L_vec*>(logits + token_idx);

    const scalar_t* v_ptr = v_cache + physical_block_number * num_heads * HEAD_SIZE * BLOCK_SIZE
                                    + head_idx * HEAD_SIZE * BLOCK_SIZE;
#pragma unroll
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
      const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
      if (row_idx < HEAD_SIZE) {
        const int offset = row_idx * BLOCK_SIZE + physical_block_offset;
        V_vec v_vec = *reinterpret_cast<const V_vec*>(v_ptr + offset);
        accs[i] += dot(logits_vec, cast_to_float(v_vec));
      }
    }
  }

  // Perform reduction within each warp.
#pragma unroll
  for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
    float acc = accs[i];
#pragma unroll
    for (int mask = NUM_V_VECS_PER_ROW / 2; mask >= 1; mask /= 2) {
      acc += __shfl_xor_sync(uint32_t(-1), acc, mask);
    }
    accs[i] = acc;
  }

  // NOTE(woosuk): A barrier is required because the shared memory space for logits
  // is reused for the output.
  __syncthreads();

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
    __syncthreads();

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
    __syncthreads();
  }

  // Write the final output.
  if (warp_idx == 0) {
    scalar_t* out_ptr = out + seq_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE;
#pragma unroll
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
      const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
      if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
        convert_from_float(*(out_ptr + row_idx), accs[i]);
      }
    }
  }
}

} // namespace cacheflow

#define LAUNCH_ATTENTION_KERNEL(T, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS)                        \
  cacheflow::single_query_cached_kv_attention_kernel<T, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS>   \
  <<<grid, block, shared_mem_size, stream>>>(                                                 \
    out_ptr,                                                                                  \
    query_ptr,                                                                                \
    key_cache_ptr,                                                                            \
    value_cache_ptr,                                                                          \
    scale,                                                                                    \
    block_tables_ptr,                                                                         \
    context_lens_ptr,                                                                         \
    max_num_blocks_per_seq,                                                                   \
    query_stride);

// TODO(woosuk): Tune NUM_THREADS.
template<
  typename T,
  int BLOCK_SIZE,
  int NUM_THREADS = 128>
void single_query_cached_kv_attention_launcher(
  torch::Tensor& out,
  torch::Tensor& query,
  torch::Tensor& key_cache,
  torch::Tensor& value_cache,
  float scale,
  torch::Tensor& block_tables,
  torch::Tensor& context_lens,
  int max_context_len) {
  int num_seqs = query.size(0);
  int num_heads = query.size(1);
  int head_size = query.size(2);
  int max_num_blocks_per_seq = block_tables.size(1);
  int query_stride = query.stride(0);

  int thread_group_size = MAX(WARP_SIZE / BLOCK_SIZE, 1);
  assert(head_size % thread_group_size == 0);

  T* out_ptr = reinterpret_cast<T*>(out.data_ptr());
  T* query_ptr = reinterpret_cast<T*>(query.data_ptr());
  T* key_cache_ptr = reinterpret_cast<T*>(key_cache.data_ptr());
  T* value_cache_ptr = reinterpret_cast<T*>(value_cache.data_ptr());
  int* block_tables_ptr = block_tables.data_ptr<int>();
  int* context_lens_ptr = context_lens.data_ptr<int>();

  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  int padded_max_context_len = ((max_context_len + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
  int logits_size = padded_max_context_len * sizeof(float);
  int outputs_size = (NUM_WARPS / 2) * head_size * sizeof(float);
  int shared_mem_size = std::max(logits_size, outputs_size);

  dim3 grid(num_heads, num_seqs);
  dim3 block(NUM_THREADS);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  switch (head_size) {
    case 32:
      LAUNCH_ATTENTION_KERNEL(T, 32, BLOCK_SIZE, NUM_THREADS);
      break;
    case 64:
      LAUNCH_ATTENTION_KERNEL(T, 64, BLOCK_SIZE, NUM_THREADS);
      break;
    case 80:
      LAUNCH_ATTENTION_KERNEL(T, 80, BLOCK_SIZE, NUM_THREADS);
      break;
    case 96:
      LAUNCH_ATTENTION_KERNEL(T, 96, BLOCK_SIZE, NUM_THREADS);
      break;
    case 128:
      LAUNCH_ATTENTION_KERNEL(T, 128, BLOCK_SIZE, NUM_THREADS);
      break;
    case 160:
      LAUNCH_ATTENTION_KERNEL(T, 160, BLOCK_SIZE, NUM_THREADS);
      break;
    case 192:
      LAUNCH_ATTENTION_KERNEL(T, 192, BLOCK_SIZE, NUM_THREADS);
      break;
    case 256:
      LAUNCH_ATTENTION_KERNEL(T, 256, BLOCK_SIZE, NUM_THREADS);
      break;
    default:
      assert(false);
      break;
  }
}

#define CALL_KERNEL_LAUNCHER(T, BLOCK_SIZE)                         \
  single_query_cached_kv_attention_launcher<T, BLOCK_SIZE>(         \ 
        out,                                                        \
        query,                                                      \
        key_cache,                                                  \
        value_cache,                                                \
        scale,                                                      \
        block_tables,                                               \
        context_lens,                                               \
        max_context_len);

void single_query_cached_kv_attention(
  torch::Tensor& out,             // [num_seqs, num_heads, head_size]
  torch::Tensor& query,           // [num_seqs, num_heads, head_size]
  torch::Tensor& key_cache,       // [num_blocks, num_heads, head_size/x, block_size, x]
  torch::Tensor& value_cache,     // [num_blocks, num_heads, head_size, block_size]
  float scale,
  torch::Tensor& block_tables,    // [num_seqs, max_num_blocks_per_seq]
  torch::Tensor& context_lens,    // [num_seqs]
  int block_size,
  int max_context_len) {
  // TODO(woosuk): Support BF16.
  if (query.element_size() == 2) {
    // Half.
    if (block_size == 1) {
      CALL_KERNEL_LAUNCHER(uint16_t, 1);
    } else if (block_size == 2) {
      CALL_KERNEL_LAUNCHER(uint16_t, 2);
    } else if (block_size == 4) {
      CALL_KERNEL_LAUNCHER(uint16_t, 4);
    } else if (block_size == 8) {
      CALL_KERNEL_LAUNCHER(uint16_t, 8);
    } else if (block_size == 16) {
      CALL_KERNEL_LAUNCHER(uint16_t, 16);
    } else if (block_size == 32) {
      CALL_KERNEL_LAUNCHER(uint16_t, 32);
    } else if (block_size == 64) {
      CALL_KERNEL_LAUNCHER(uint16_t, 64);
    } else if (block_size == 128) {
      CALL_KERNEL_LAUNCHER(uint16_t, 128);
    } else if (block_size == 256) {
      CALL_KERNEL_LAUNCHER(uint16_t, 256);
    } else {
      assert(false);
    }
  } else {
    // Float.
    assert(false);
  }
}

// namespace cacheflow {

// // Grid: (num_heads, num_query_tokens).
// template<
//   typename scalar_t,
//   int HEAD_SIZE,
//   int BLOCK_SIZE,
//   int NUM_THREADS>
// __device__ void multi_query_cached_kv_attention_kernel_unoptimized_(
//   scalar_t* __restrict__ out,             // [num_seqs, num_heads, head_size]
//   const scalar_t* __restrict__ q,         // [num_seqs, num_heads, head_size]
//   const int seq_start_idx,
//   const int seq_len,
//   const scalar_t* __restrict__ k_cache,   // [num_blocks, num_heads, head_size/x, block_size, x]
//   const scalar_t* __restrict__ v_cache,   // [num_blocks, num_heads, head_size, block_size]
//   const float scale,
//   const int* __restrict__ block_table,   // [num_seqs, max_num_blocks_per_seq]
//   const int context_len,
//   const int max_num_blocks_per_seq,
//   const int q_stride) {
//   constexpr int THREAD_GROUP_SIZE = WARP_SIZE / BLOCK_SIZE;
//   constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
//   const int thread_idx = threadIdx.x;
//   const int warp_idx = thread_idx / WARP_SIZE;
//   const int lane = thread_idx % WARP_SIZE;

//   const int head_idx = blockIdx.x;
//   const int num_heads = gridDim.x;
//   const int seq_idx = blockIdx.y;

//   // A vector type to store a part of a key or a query.
//   // The vector size is configured in such a way that the threads in a thread group
//   // fetch or comput 16 bytes at a time.
//   // For example, if the size of a thread group is 4 and the data type is half,
//   // then the vector size is 16 / (4 * sizeof(half)) == 2.
//   constexpr int VEC_SIZE = 16 / (THREAD_GROUP_SIZE * sizeof(scalar_t));
//   using K_vec = typename Vec<scalar_t, VEC_SIZE>::Type;
//   using Q_vec = typename Vec<scalar_t, VEC_SIZE>::Type;

//   constexpr int NUM_ELEMS_PER_THREAD = HEAD_SIZE / THREAD_GROUP_SIZE;
//   constexpr int NUM_VECS_PER_THREAD = NUM_ELEMS_PER_THREAD / VEC_SIZE;

//   const int thread_group_idx = thread_idx / THREAD_GROUP_SIZE;
//   const int thread_group_offset = thread_idx % THREAD_GROUP_SIZE;

//   // Load the query to registers.
//   // Each thread in a thread group has a different part of the query.
//   // For example, if the the thread group size is 4, then the first thread in the group
//   // has 0, 4, 8, ... th vectors of the query, and the second thread has 1, 5, 9, ...
//   // th vectors of the query, and so on.
//   // NOTE(woosuk): Because q is split from a qkv tensor, it may not be contiguous.
//   const scalar_t* q_ptr = q + seq_idx * q_stride + head_idx * HEAD_SIZE;
//   Q_vec q_vecs[NUM_VECS_PER_THREAD];
// #pragma unroll
//   for (int i = 0; i < NUM_VECS_PER_THREAD; i++) {
//     const int vec_idx = thread_group_offset + i * THREAD_GROUP_SIZE;
//     q_vecs[i] = *reinterpret_cast<const Q_vec*>(q_ptr + vec_idx * VEC_SIZE);
//   }

//   // Memory planning.
//   extern __shared__ char shared_mem[];
//   // NOTE(woosuk): We use FP32 logits and accumulation.
//   float *logits = reinterpret_cast<float*>(shared_mem);
//   // Workspace for reduction.
//   __shared__ float red_smem[2 * NUM_WARPS];

//   // x == THREAD_GROUP_SIZE * VEC_SIZE
//   // Each thread group fetches x elements from the key at a time.
//   constexpr int x = 16 / sizeof(scalar_t);
//   float qk_max = -FLT_MAX;

//   const int num_blocks = (context_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
//   const int mask_boundary = context_len - seq_len + 1 + (seq_idx - seq_start_idx);

//   // Iterate over the key blocks.
//   // Each warp fetches a block of keys for each iteration.
//   // Each thread group in a warp fetches a key from the block, and computes
//   // dot product with the query.
//   for (int block_idx = warp_idx; block_idx < num_blocks; block_idx += NUM_WARPS) {
//     const int physical_block_number = block_table[block_idx];
//     const int physical_block_offset = thread_group_idx % BLOCK_SIZE;
//     const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;

//     // Load a key to registers.
//     // Each thread in a thread group has a different part of the key.
//     // For example, if the the thread group size is 4, then the first thread in the group
//     // has 0, 4, 8, ... th vectors of the key, and the second thread has 1, 5, 9, ... th
//     // vectors of the key, and so on.
//     K_vec k_vecs[NUM_VECS_PER_THREAD];
// #pragma unroll
//     for (int i = 0; i < NUM_VECS_PER_THREAD; i++) {
//       const scalar_t* k_ptr = k_cache + physical_block_number * num_heads * HEAD_SIZE * BLOCK_SIZE
//                                       + head_idx * HEAD_SIZE * BLOCK_SIZE
//                                       + physical_block_offset * x;
//       const int vec_idx = thread_group_offset + i * THREAD_GROUP_SIZE;
//       const int offset1 = (vec_idx * VEC_SIZE) / x;
//       const int offset2 = (vec_idx * VEC_SIZE) % x;
//       k_vecs[i] = *reinterpret_cast<const K_vec*>(k_ptr + offset1 * BLOCK_SIZE * x + offset2);
//     }

//     // Compute dot product.
//     // This includes a reduction across the threads in the same thread group.
//     const float qk = scale * Qk_dot<scalar_t, THREAD_GROUP_SIZE>::dot(q_vecs, k_vecs);
//     const bool mask = token_idx >= mask_boundary;

//     if (thread_group_offset == 0) {
//       // Store the partial reductions to shared memory.
//       // NOTE(woosuk): It is required to zero out the masked logits.
//       logits[token_idx] = mask ? 0.f : qk;
//       // Update the max value.
//       qk_max = mask ? qk_max : fmaxf(qk_max, qk);
//     }
//   }

//   // Perform reduction across the threads in the same warp to get the
//   // max qk value for each "warp" (not across the thread block yet).
//   // The 0-th thread of each thread group already has its max qk value.
// #pragma unroll
//   for (int mask = WARP_SIZE / 2; mask >= THREAD_GROUP_SIZE; mask /= 2) {
//     qk_max = fmaxf(qk_max, __shfl_xor_sync(uint32_t(-1), qk_max, mask));
//   }
//   if (lane == 0) {
//     red_smem[warp_idx] = qk_max;
//   }
//   __syncthreads();

//   // TODO(woosuk): Refactor this part.
//   // Get the max qk value for the sequence.
//   qk_max = lane < NUM_WARPS ? red_smem[lane] : -FLT_MAX;
// #pragma unroll
//   for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
//       qk_max = fmaxf(qk_max, __shfl_xor_sync(uint32_t(-1), qk_max, mask));
//   }
//   // Broadcast the max qk value to all threads.
//   qk_max = __shfl_sync(uint32_t(-1), qk_max, 0);

//   // Get the sum of the exp values.
//   float exp_sum = 0.f;
//   for (int i = thread_idx; i < mask_boundary; i += NUM_THREADS) {
//     float val = __expf(logits[i] - qk_max);
//     logits[i] = val;
//     exp_sum += val;
//   }
//   exp_sum = block_sum<NUM_WARPS>(&red_smem[NUM_WARPS], exp_sum);

//   // Compute softmax.
//   const float inv_sum = __fdividef(1.f, exp_sum + 1e-6f);
//   for (int i = thread_idx; i < context_len; i += NUM_THREADS) {
//     logits[i] *= inv_sum;
//   }
//   __syncthreads();

//   // Each thread will fetch 16 bytes from the value cache at a time.
//   constexpr int V_VEC_SIZE = 16 / sizeof(scalar_t);
//   using V_vec = typename Vec<scalar_t, V_VEC_SIZE>::Type;
//   using L_vec = typename FloatVec<V_vec>::Type;

//   constexpr int NUM_V_VECS_PER_ROW = BLOCK_SIZE / V_VEC_SIZE;
//   constexpr int NUM_ROWS_PER_ITER = WARP_SIZE / NUM_V_VECS_PER_ROW;
//   constexpr int NUM_ROWS_PER_THREAD = (HEAD_SIZE + NUM_ROWS_PER_ITER - 1) / NUM_ROWS_PER_ITER;

//   float accs[NUM_ROWS_PER_THREAD];
// #pragma unroll
//   for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
//     accs[i] = 0.f;
//   }

//   for (int block_idx = warp_idx; block_idx < num_blocks; block_idx += NUM_WARPS) {
//     const int physical_block_number = block_table[block_idx];
//     const int physical_block_offset = (lane % NUM_V_VECS_PER_ROW) * V_VEC_SIZE;
//     const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;
//     L_vec logits_vec = *reinterpret_cast<L_vec*>(logits + token_idx);

//     const scalar_t* v_ptr = v_cache + physical_block_number * num_heads * HEAD_SIZE * BLOCK_SIZE
//                                     + head_idx * HEAD_SIZE * BLOCK_SIZE;
// #pragma unroll
//     for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
//       const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
//       if (row_idx < HEAD_SIZE) {
//         const int offset = row_idx * BLOCK_SIZE + physical_block_offset;
//         V_vec v_vec = *reinterpret_cast<const V_vec*>(v_ptr + offset);
//         accs[i] += dot(logits_vec, cast_to_float(v_vec));
//       }
//     }
//   }

//   // Perform reduction within each warp.
// #pragma unroll
//   for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
//     float acc = accs[i];
// #pragma unroll
//     for (int mask = NUM_V_VECS_PER_ROW / 2; mask >= 1; mask /= 2) {
//       acc += __shfl_xor_sync(uint32_t(-1), acc, mask);
//     }
//     accs[i] = acc;
//   }

//   // NOTE(woosuk): A barrier is required because the shared memory space for logits
//   // is reused for the output.
//   __syncthreads();

//   // Perform reduction across warps.
//   float* out_smem = reinterpret_cast<float*>(shared_mem);
// #pragma unroll
//   for (int i = NUM_WARPS; i > 1; i /= 2) {
//     int mid = i / 2;
//     // Upper warps write to shared memory.
//     if (warp_idx >= mid && warp_idx < i) {
//       float* dst = &out_smem[(warp_idx - mid) * HEAD_SIZE];
// #pragma unroll
//       for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
//         const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
//         if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
//           dst[row_idx] = accs[i];
//         }
//       }
//     }
//     __syncthreads();

//     // Lower warps update the output.
//     if (warp_idx < mid) {
//       const float* src = &out_smem[warp_idx * HEAD_SIZE];
// #pragma unroll
//       for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
//         const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
//         if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
//           accs[i] += src[row_idx];
//         }
//       }
//     }
//     __syncthreads();
//   }

//   // Write the final output.
//   if (warp_idx == 0) {
//     scalar_t* out_ptr = out + seq_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE;
// #pragma unroll
//     for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
//       const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
//       if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
//         convert_from_float(*(out_ptr + row_idx), accs[i]);
//       }
//     }
//   }
// }


// // Grid: (num_heads, num_query_tokens).
// template<
//   typename scalar_t,
//   int HEAD_SIZE,
//   int BLOCK_SIZE,
//   int NUM_THREADS>
// __global__ void multi_query_cached_kv_attention_kernel(
//   const int* cu_query_lens,               // [num_prompts+1]
//   const int* seq_prompt_mapping,          // [num_seqs] mapping from seq_idx to prompt_idx
//   scalar_t* __restrict__ out,             // [num_seqs, num_heads, head_size]
//   const scalar_t* __restrict__ q,         // [num_seqs, num_heads, head_size]
//   const scalar_t* __restrict__ k_cache,   // [num_blocks, num_heads, head_size/x, block_size, x]
//   const scalar_t* __restrict__ v_cache,   // [num_blocks, num_heads, head_size, block_size]
//   const float scale,
//   const int* __restrict__ block_tables,   // [num_prompts, max_num_blocks_per_seq]
//   const int* __restrict__ context_lens,   // [num_prompts]
//   const int max_num_blocks_per_seq,
//   const int q_stride) {
//     const int seq_idx = blockIdx.y;
//     const int prompt_idx = seq_prompt_mapping[seq_idx];
//     const int seq_start_idx = cu_query_lens[prompt_idx];
//     const int seq_len = cu_query_lens[prompt_idx + 1] - seq_start_idx;
//     const int* block_table = block_tables + prompt_idx * max_num_blocks_per_seq;
//     const int context_len = context_lens[prompt_idx];
//     multi_query_cached_kv_attention_kernel_unoptimized_<
//         scalar_t, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS>(
//           out,
//           q,
//           seq_start_idx,
//           seq_len,
//           k_cache,
//           v_cache,
//           scale,
//           block_table,
//           context_len,
//           max_num_blocks_per_seq,
//           q_stride);
// }

// } // namespace cacheflow

// #define LAUNCH_MULTI_ATTENTION_KERNEL(T, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS)                  \
//   cacheflow::multi_query_cached_kv_attention_kernel<T, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS>    \
//   <<<grid, block, shared_mem_size, stream>>>(                                                 \
//     cu_query_lens_ptr,                                                                        \
//     seq_prompt_mapping_ptr,                                                                   \
//     out_ptr,                                                                                  \
//     query_ptr,                                                                                \
//     key_cache_ptr,                                                                            \
//     value_cache_ptr,                                                                          \
//     scale,                                                                                    \
//     block_tables_ptr,                                                                         \
//     context_lens_ptr,                                                                         \
//     max_num_blocks_per_seq,                                                                   \
//     query_stride);


// // TODO(woosuk): Tune NUM_THREADS.
// template<
//   typename T,
//   int BLOCK_SIZE,
//   int NUM_THREADS = 128>
// void multi_query_cached_kv_attention_launcher(
//   torch::Tensor& cu_query_lens,
//   torch::Tensor& seq_prompt_mapping,
//   torch::Tensor& out,
//   torch::Tensor& query,
//   torch::Tensor& key_cache,
//   torch::Tensor& value_cache,
//   float scale,
//   torch::Tensor& block_tables,
//   torch::Tensor& context_lens,
//   int max_context_len) {
//   int num_seqs = query.size(0);
//   int num_heads = query.size(1);
//   int head_size = query.size(2);
//   int max_num_blocks_per_seq = block_tables.size(1);
//   int query_stride = query.stride(0);

//   int* cu_query_lens_ptr = cu_query_lens.data_ptr<int>();
//   int* seq_prompt_mapping_ptr = seq_prompt_mapping.data_ptr<int>();
//   T* out_ptr = reinterpret_cast<T*>(out.data_ptr());
//   T* query_ptr = reinterpret_cast<T*>(query.data_ptr());
//   T* key_cache_ptr = reinterpret_cast<T*>(key_cache.data_ptr());
//   T* value_cache_ptr = reinterpret_cast<T*>(value_cache.data_ptr());
//   int* block_tables_ptr = block_tables.data_ptr<int>();
//   int* context_lens_ptr = context_lens.data_ptr<int>();

//   constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
//   int padded_max_context_len = ((max_context_len + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
//   int logits_size = padded_max_context_len * sizeof(float);
//   int outputs_size = (NUM_WARPS / 2) * head_size * sizeof(float);
//   int shared_mem_size = std::max(logits_size, outputs_size);

//   dim3 grid(num_heads, num_seqs);
//   dim3 block(NUM_THREADS);
//   const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
//   switch (head_size) {
//     case 32:
//       LAUNCH_MULTI_ATTENTION_KERNEL(T, 32, BLOCK_SIZE, NUM_THREADS);
//       break;
//     case 64:
//       LAUNCH_MULTI_ATTENTION_KERNEL(T, 64, BLOCK_SIZE, NUM_THREADS);
//       break;
//     case 80:
//       LAUNCH_MULTI_ATTENTION_KERNEL(T, 80, BLOCK_SIZE, NUM_THREADS);
//       break;
//     case 96:
//       LAUNCH_MULTI_ATTENTION_KERNEL(T, 96, BLOCK_SIZE, NUM_THREADS);
//       break;
//     case 128:
//       LAUNCH_MULTI_ATTENTION_KERNEL(T, 128, BLOCK_SIZE, NUM_THREADS);
//       break;
//     case 160:
//       LAUNCH_MULTI_ATTENTION_KERNEL(T, 160, BLOCK_SIZE, NUM_THREADS);
//       break;
//     case 192:
//       LAUNCH_MULTI_ATTENTION_KERNEL(T, 192, BLOCK_SIZE, NUM_THREADS);
//       break;
//     case 256:
//       LAUNCH_MULTI_ATTENTION_KERNEL(T, 256, BLOCK_SIZE, NUM_THREADS);
//       break;
//     default:
//       assert(false);
//       break;
//   }
// }

// void multi_query_cached_kv_attention(
//   torch::Tensor& cu_query_lens,
//   torch::Tensor& out,
//   torch::Tensor& query,
//   torch::Tensor& key_cache,
//   torch::Tensor& value_cache,
//   float scale,
//   torch::Tensor& block_tables,
//   torch::Tensor& context_lens,
//   int block_size,
//   int max_context_len) {

//   torch::Tensor query_lens = cu_query_lens.to(torch::kCPU);
  
//   int num_queries = query_lens.size(0) - 1;
//   const int* query_lens_ptr = query_lens.data_ptr<int>();
//   int num_seqs = query.size(0);

//   torch::Tensor cpu_tensor = torch::empty({num_seqs}, torch::dtype(torch::kInt32));
//   auto accessor = cpu_tensor.accessor<int32_t, 1>();
//   for (int i = 0, query_cursor = 0; i < num_seqs; ++i) {
//     if (i >= query_lens_ptr[query_cursor + 1]) {
//       ++query_cursor; 
//     }
//     accessor[i] = query_cursor;
//   }

//   // TODO(suquark): This can be slow, as it to(torch::kCPU) and to(torch::kCUDA)
//   // implicitly synchronizes the CPU and GPU. And we can avoid this issue by giving
//   // the mapping as an input parameter. Let's do this optimization in a later PR.
//   torch::Tensor seq_prompt_mapping = cpu_tensor.to(torch::kCUDA);

//   // TODO(woosuk): Support BF16.
//   if (query.element_size() == 2) {
//     // Half.
//     if (block_size == 8) {
//       multi_query_cached_kv_attention_launcher<uint16_t, 8>(
//         cu_query_lens,
//         seq_prompt_mapping,
//         out,
//         query,
//         key_cache,
//         value_cache,
//         scale,
//         block_tables,
//         context_lens,
//         max_context_len);
//     } else if (block_size == 16) {
//       multi_query_cached_kv_attention_launcher<uint16_t, 16>(
//         cu_query_lens,
//         seq_prompt_mapping,
//         out,
//         query,
//         key_cache,
//         value_cache,
//         scale,
//         block_tables,
//         context_lens,
//         max_context_len);
//     } else if (block_size == 32) {
//       multi_query_cached_kv_attention_launcher<uint16_t, 32>(
//         cu_query_lens,
//         seq_prompt_mapping,
//         out,
//         query,
//         key_cache,
//         value_cache,
//         scale,
//         block_tables,
//         context_lens,
//         max_context_len);
//     } else {
//       assert(false);
//     }
//   } else if (query.element_size() == 4) {
//     // Float.
//     if (block_size == 8) {
//       multi_query_cached_kv_attention_launcher<float, 8>(
//         cu_query_lens,
//         seq_prompt_mapping,
//         out,
//         query,
//         key_cache,
//         value_cache,
//         scale,
//         block_tables,
//         context_lens,
//         max_context_len);
//     } else if (block_size == 16) {
//       multi_query_cached_kv_attention_launcher<float, 16>(
//         cu_query_lens,
//         seq_prompt_mapping,
//         out,
//         query,
//         key_cache,
//         value_cache,
//         scale,
//         block_tables,
//         context_lens,
//         max_context_len);
//     } else if (block_size == 32) {
//       multi_query_cached_kv_attention_launcher<float, 32>(
//         cu_query_lens,
//         seq_prompt_mapping,
//         out,
//         query,
//         key_cache,
//         value_cache,
//         scale,
//         block_tables,
//         context_lens,
//         max_context_len);
//     } else {
//       assert(false);
//     }
//   } else {
//     assert(false);
//   }
// }

#undef WARP_SIZE
#undef MAX
#undef MIN
