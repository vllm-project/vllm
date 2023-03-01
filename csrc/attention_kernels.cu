#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "attention_utils.h"
#include "cuda_primitives.h"

#include <algorithm>

#define WARP_SIZE 32

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
  const scalar_t* __restrict__ v_cache,   // [num_blocks, num_heads, block_size, head_size]
  const float scale,
  const int* __restrict__ block_tables,   // [num_seqs, max_num_blocks_per_seq]
  const int* __restrict__ context_lens,   // [num_seqs]
  const int max_num_blocks_per_seq) {
  constexpr int THREAD_GROUP_SIZE = WARP_SIZE / BLOCK_SIZE;
  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  const int thread_idx = threadIdx.x;
  const int warp_idx = thread_idx / WARP_SIZE;
  const int lane = thread_idx % WARP_SIZE;

  const int head_idx = blockIdx.x;
  const int num_heads = gridDim.x;
  const int seq_idx = blockIdx.y;

  // A vector type to store a part of a key or a query.
  // The vector size is configured in such a way that the threads in a thread group
  // fetch or comput 16 bytes at a time.
  // For example, if the size of a thread group is 4 and the data type is half,
  // then the vector size is 16 / (4 * sizeof(half)) == 2.
  constexpr int VEC_SIZE = 16 / (THREAD_GROUP_SIZE * sizeof(scalar_t));
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
  const scalar_t* q_ptr = q + seq_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE;
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
    const int physical_block_offset = thread_group_idx % BLOCK_SIZE;
    const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;

    // Load a key to registers.
    // Each thread in a thread group has a different part of the key.
    // For example, if the the thread group size is 4, then the first thread in the group
    // has 0, 4, 8, ... th vectors of the key, and the second thread has 1, 5, 9, ... th
    // vectors of the key, and so on.
    K_vec k_vecs[NUM_VECS_PER_THREAD];
#pragma unroll
    for (int i = 0; i < NUM_VECS_PER_THREAD; i++) {
      const scalar_t* k_ptr = k_cache + physical_block_number * num_heads * HEAD_SIZE * BLOCK_SIZE
                                      + head_idx * HEAD_SIZE * BLOCK_SIZE
                                      + physical_block_offset * x;
      const int vec_idx = thread_group_offset + i * THREAD_GROUP_SIZE;
      const int offset1 = (vec_idx * VEC_SIZE) / x;
      const int offset2 = (vec_idx * VEC_SIZE) % x;
      k_vecs[i] = *reinterpret_cast<const K_vec*>(k_ptr + offset1 * BLOCK_SIZE * x + offset2);
    }

    // Compute dot product.
    // This includes a reduction across the threads in the same thread group.
    const float qk = scale * Qk_dot<scalar_t, THREAD_GROUP_SIZE>::dot(q_vecs, k_vecs);
    const bool mask = token_idx >= context_len;
  
    if (!mask && thread_group_offset == 0) {
      // Store the partial reductions to shared memory.
      logits[token_idx] = qk;
      // Update the max value.
      qk_max = fmaxf(qk_max, qk);
    }
  }

  // Perform reduction across the threads in the same warp to get the
  // max qk value for each "warp" (not across the thread block yet).
  // The 0-th thread of each thread group already has its max qk value.
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= THREAD_GROUP_SIZE; mask /= 2) {
    qk_max = fmaxf(qk_max, __shfl_xor_sync(uint32_t(-1), qk_max, mask));
  }
  // Perform reduction across the warps to get the max qk value for the sequence.
  if (lane == 0) {
    red_smem[warp_idx] = qk_max;
  }
  __syncthreads();
  qk_max = lane < NUM_WARPS ? red_smem[lane] : -FLT_MAX;
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
      qk_max = fmaxf(qk_max, __shfl_xor_sync(uint32_t(-1), qk_max, mask));
  }
  // Broadcast the max qk value to all threads.
  qk_max = __shfl_sync(uint32_t(-1), qk_max, 0);

  // Get the sum of the exp values.
  float sum = 0.0f;
  for (int i = thread_idx; i < context_len; i += NUM_THREADS) {
    float val = __expf(logits[i] - qk_max);
    logits[i] = val;
    sum += val;
  }
  sum = block_sum<NUM_WARPS>(&red_smem[NUM_WARPS], sum);

  // Compute softmax.
  const float inv_sum = __fdividef(1.f, sum + 1e-6f);
  for (int i = thread_idx; i < context_len; i += NUM_THREADS) {
    logits[i] *= inv_sum;
  }
  __syncthreads();

  // FIXME(woosuk)
  static_assert(HEAD_SIZE == 2 * WARP_SIZE || HEAD_SIZE == 4 * WARP_SIZE ||
                HEAD_SIZE == 8 * WARP_SIZE,
                "HEAD_SIZE must be one of 64, 128, and 256.");
  constexpr int V_VEC_SIZE = HEAD_SIZE / WARP_SIZE;
  using V_vec = typename Vec<scalar_t, V_VEC_SIZE>::Type;
  // The type of A_vec can be different from the type of K_vec.
  // 1. When the actual type of Q, K, V is half, the QKV vectors use uint types.
  //    However, A_vec always has a floating point type.
  // 2. Each element of A_vec is always a float, because we use FP32 accumulation.
  using A_vec = typename FloatVec<V_vec>::Type;
  A_vec out_vec;

  for (int block_idx = warp_idx; block_idx < num_blocks; block_idx += NUM_WARPS) {
    const int physical_block_number = block_table[block_idx];
#pragma unroll
    for (int i = 0; i < BLOCK_SIZE; i++) {
      const int physical_block_offset = i;
      const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;
      const bool mask = token_idx >= context_len;
      const float logit = mask ? 0.f : logits[token_idx];

      const scalar_t* v_ptr = v_cache + physical_block_number * num_heads * HEAD_SIZE * BLOCK_SIZE
                                      + head_idx * HEAD_SIZE * BLOCK_SIZE
                                      + physical_block_offset * HEAD_SIZE;
      V_vec v_vec = *reinterpret_cast<const V_vec*>(v_ptr + lane * V_VEC_SIZE);
      // Compute acc += logit * v.
      out_vec = fma(logit, cast_to_float(v_vec), out_vec);
    }
  }

  // NOTE(woosuk): A barrier is required because the shared memory space for logits
  // is reused for the output.
  __syncthreads();

  // Run final reduction.
  scalar_t* out_smem = reinterpret_cast<scalar_t*>(shared_mem);
#pragma unroll
  for (int i = NUM_WARPS; i > 1; i /= 2) {
    int mid = i / 2;
    // Upper warps write to shared memory.
    if (warp_idx >= mid && warp_idx < i) {
      scalar_t* dst = &out_smem[(warp_idx - mid) * HEAD_SIZE];
      convert_from_float(*reinterpret_cast<V_vec*>(dst + lane * V_VEC_SIZE), out_vec);
    }
    __syncthreads();

    // Lower thread groups update the output.
    if (warp_idx < mid) {
      scalar_t* src = &out_smem[warp_idx * HEAD_SIZE];
      out_vec = add(*reinterpret_cast<const V_vec*>(src + lane * V_VEC_SIZE), out_vec);
    }
    __syncthreads();
  }

  // Write the final output.
  if (warp_idx == 0) {
    scalar_t* out_ptr = out + seq_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE;
    convert_from_float(*reinterpret_cast<V_vec*>(out_ptr + lane * V_VEC_SIZE), out_vec);
  }
}

} // namespace cacheflow

#define LAUNCH_ATTENTION_KERNEL(T, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS)                        \
  cacheflow::single_query_cached_kv_attention_kernel<T, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS>   \
  <<<grid, block, shared_mem_size, stream>>>(                                                 \
    out,                                                                                      \
    query,                                                                                    \
    key_cache,                                                                                \
    value_cache,                                                                              \
    scale,                                                                                    \
    block_tables,                                                                             \
    context_lens,                                                                             \
    max_num_blocks_per_seq);


template<typename T>
void single_query_cached_kv_attention_launcher(
  T* out,
  T* query,
  T* key_cache,
  T* value_cache,
  float scale,
  int* block_tables,
  int* context_lens,
  int num_seqs,
  int num_heads,
  int head_size,
  int max_num_blocks_per_seq,
  int block_size,
  int max_context_len,
  cudaStream_t stream) {
  constexpr int NUM_THREADS = 128;
  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  int logits_size = max_context_len * sizeof(float);
  int outputs_size = NUM_WARPS / 2 * head_size * sizeof(T);
  int shared_mem_size = std::max(logits_size, outputs_size);

  dim3 grid(num_heads, num_seqs);
  dim3 block(NUM_THREADS);
  assert(block_size == 8);
  switch (head_size) {
    // case 32:
    //   LAUNCH_ATTENTION_KERNEL(T, 32, 8, NUM_THREADS);
    //   break;
    case 64:
      LAUNCH_ATTENTION_KERNEL(T, 64, 8, NUM_THREADS);
      break;
    case 128:
      LAUNCH_ATTENTION_KERNEL(T, 128, 8, NUM_THREADS);
      break;
    case 256:
      LAUNCH_ATTENTION_KERNEL(T, 256, 8, NUM_THREADS);
      break;
    default:
      assert(false);
      break;
  }
}

void single_query_cached_kv_attention(
  torch::Tensor& out,
  torch::Tensor& query,
  torch::Tensor& key_cache,
  torch::Tensor& value_cache,
  float scale,
  torch::Tensor& block_tables,
  torch::Tensor& context_lens,
  int block_size,
  int max_context_len) {
  int num_seqs = query.size(0);
  int num_heads = query.size(1);
  int head_size = query.size(2);
  int max_num_blocks_per_seq = block_tables.size(1);

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  // FIXME
  assert(query.element_size() == 2);
  single_query_cached_kv_attention_launcher<uint16_t>(
    reinterpret_cast<uint16_t*>(out.data_ptr()),
    reinterpret_cast<uint16_t*>(query.data_ptr()),
    reinterpret_cast<uint16_t*>(key_cache.data_ptr()),
    reinterpret_cast<uint16_t*>(value_cache.data_ptr()),
    scale,
    block_tables.data_ptr<int>(),
    context_lens.data_ptr<int>(),
    num_seqs,
    num_heads,
    head_size,
    max_num_blocks_per_seq,
    block_size,
    max_context_len,
    stream);
}

#undef WARP_SIZE
