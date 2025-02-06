#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "cuda_compat.h"
#include "dispatch_utils.h"

namespace vllm {

template <typename scalar_t, bool IS_NEOX>
inline __device__ void apply_token_rotary_embedding(
    scalar_t* __restrict__ arr, const scalar_t* __restrict__ cos_ptr,
    const scalar_t* __restrict__ sin_ptr, int rot_offset, int embed_dim) {
  int x_index, y_index;
  scalar_t cos, sin;
  if (IS_NEOX) {
    // GPT-NeoX style rotary embedding.
    x_index = rot_offset;
    y_index = embed_dim + rot_offset;
    cos = VLLM_LDG(cos_ptr + x_index);
    sin = VLLM_LDG(sin_ptr + x_index);
  } else {
    // GPT-J style rotary embedding.
    x_index = 2 * rot_offset;
    y_index = 2 * rot_offset + 1;
    cos = VLLM_LDG(cos_ptr + x_index / 2);
    sin = VLLM_LDG(sin_ptr + x_index / 2);
  }

  const scalar_t x = arr[x_index];
  const scalar_t y = arr[y_index];
  arr[x_index] = x * cos - y * sin;
  arr[y_index] = y * cos + x * sin;
}

template <typename scalar_t, bool IS_NEOX>
inline __device__ void apply_rotary_embedding(
    scalar_t* __restrict__ query,  // [batch_size, seq_len, num_heads,
                                   // head_size] or [num_tokens, num_heads,
                                   // head_size]
    scalar_t* __restrict__ key,    // [batch_size, seq_len, num_kv_heads,
                                   // head_size] or [num_tokens, num_kv_heads,
                                   // head_size]
    const scalar_t* cache_ptr, const int head_size, const int num_heads,
    const int num_kv_heads, const int rot_dim, const int token_idx,
    const int64_t query_stride, const int64_t key_stride) {
  const int embed_dim = rot_dim / 2;
  const scalar_t* cos_ptr = cache_ptr;
  const scalar_t* sin_ptr = cache_ptr + embed_dim;

  const int nq = num_heads * embed_dim;
  for (int i = threadIdx.x; i < nq; i += blockDim.x) {
    const int head_idx = i / embed_dim;
    const int64_t token_head = token_idx * query_stride + head_idx * head_size;
    const int rot_offset = i % embed_dim;
    apply_token_rotary_embedding<scalar_t, IS_NEOX>(
        query + token_head, cos_ptr, sin_ptr, rot_offset, embed_dim);
  }

  const int nk = num_kv_heads * embed_dim;
  for (int i = threadIdx.x; i < nk; i += blockDim.x) {
    const int head_idx = i / embed_dim;
    const int64_t token_head = token_idx * key_stride + head_idx * head_size;
    const int rot_offset = i % embed_dim;
    apply_token_rotary_embedding<scalar_t, IS_NEOX>(
        key + token_head, cos_ptr, sin_ptr, rot_offset, embed_dim);
  }
}

template <typename scalar_t, bool IS_NEOX>
__global__ void rotary_embedding_kernel(
    const int64_t* __restrict__ positions,  // [batch_size, seq_len] or
                                            // [num_tokens]
    scalar_t* __restrict__ query,           // [batch_size, seq_len, num_heads,
                                   // head_size] or [num_tokens, num_heads,
                                   // head_size]
    scalar_t* __restrict__ key,  // [batch_size, seq_len, num_kv_heads,
                                 // head_size] or [num_tokens, num_kv_heads,
                                 // head_size]
    const scalar_t* __restrict__ cos_sin_cache,  // [max_position, 2, rot_dim //
                                                 // 2]
    const int rot_dim, const int64_t query_stride, const int64_t key_stride,
    const int num_heads, const int num_kv_heads, const int head_size) {
  // Each thread block is responsible for one token.
  const int token_idx = blockIdx.x;
  int64_t pos = positions[token_idx];
  const scalar_t* cache_ptr = cos_sin_cache + pos * rot_dim;

  apply_rotary_embedding<scalar_t, IS_NEOX>(
      query, key, cache_ptr, head_size, num_heads, num_kv_heads, rot_dim,
      token_idx, query_stride, key_stride);
}

template <typename scalar_t, bool IS_NEOX>
__global__ void batched_rotary_embedding_kernel(
    const int64_t* __restrict__ positions,  // [batch_size, seq_len] or
                                            // [num_tokens]
    scalar_t* __restrict__ query,           // [batch_size, seq_len, num_heads,
                                   // head_size] or [num_tokens, num_heads,
                                   // head_size]
    scalar_t* __restrict__ key,  // [batch_size, seq_len, num_kv_heads,
                                 // head_size] or [num_tokens, num_kv_heads,
                                 // head_size]
    const scalar_t* __restrict__ cos_sin_cache,  // [max_position, 2, rot_dim //
                                                 // 2]
    const int64_t* __restrict__ cos_sin_cache_offsets,  // [batch_size, seq_len]
                                                        // or [num_tokens]
    const int rot_dim, const int64_t query_stride, const int64_t key_stride,
    const int num_heads, const int num_kv_heads, const int head_size) {
  // Each thread block is responsible for one token.
  const int token_idx = blockIdx.x;
  int64_t pos = positions[token_idx];
  int64_t cos_sin_cache_offset = cos_sin_cache_offsets[token_idx];
  const scalar_t* cache_ptr =
      cos_sin_cache + (cos_sin_cache_offset + pos) * rot_dim;

  apply_rotary_embedding<scalar_t, IS_NEOX>(
      query, key, cache_ptr, head_size, num_heads, num_kv_heads, rot_dim,
      token_idx, query_stride, key_stride);
}

}  // namespace vllm

void rotary_embedding(
    torch::Tensor& positions,  // [batch_size, seq_len] or [num_tokens]
    torch::Tensor& query,  // [batch_size, seq_len, num_heads * head_size] or
                           // [num_tokens, num_heads * head_size] or
                           // [batch_size, seq_len, num_heads, head_size] or
                           // [num_tokens, num_heads, head_size]
    torch::Tensor& key,    // [batch_size, seq_len, num_kv_heads * head_size] or
                           // [num_tokens, num_kv_heads * head_size] or
                           // [batch_size, seq_len, num_heads, head_size] or
                           // [num_tokens, num_heads, head_size]
    int64_t head_size,
    torch::Tensor& cos_sin_cache,  // [max_position, rot_dim]
    bool is_neox) {
  // num_tokens = batch_size * seq_len
  int64_t num_tokens = positions.numel();
  int positions_ndim = positions.dim();

  // Make sure num_tokens dim is consistent across positions, query, and key.
  TORCH_CHECK(
      positions_ndim == 1 || positions_ndim == 2,
      "positions must have shape [num_tokens] or [batch_size, seq_len]");
  if (positions_ndim == 1) {
    TORCH_CHECK(
        query.size(0) == positions.size(0) && key.size(0) == positions.size(0),
        "query, key and positions must have the same number of tokens");
  }
  if (positions_ndim == 2) {
    TORCH_CHECK(
        query.size(0) == positions.size(0) &&
            key.size(0) == positions.size(0) &&
            query.size(1) == positions.size(1) &&
            key.size(1) == positions.size(1),
        "query, key and positions must have the same batch_size and seq_len");
  }

  // Make sure head_size is valid for query and key
  // hidden_size = num_heads * head_size
  int query_hidden_size = query.numel() / num_tokens;
  int key_hidden_size = key.numel() / num_tokens;
  TORCH_CHECK(query_hidden_size % head_size == 0);
  TORCH_CHECK(key_hidden_size % head_size == 0);

  // Make sure query and key have consistent number of heads
  int num_heads = query_hidden_size / head_size;
  int num_kv_heads = key_hidden_size / head_size;
  TORCH_CHECK(num_heads % num_kv_heads == 0);

  int rot_dim = cos_sin_cache.size(1);
  int seq_dim_idx = positions_ndim - 1;
  int64_t query_stride = query.stride(seq_dim_idx);
  int64_t key_stride = key.stride(seq_dim_idx);

  dim3 grid(num_tokens);
  dim3 block(std::min<int64_t>(num_heads * rot_dim / 2, 512));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(query));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(query.scalar_type(), "rotary_embedding", [&] {
    if (is_neox) {
      vllm::rotary_embedding_kernel<scalar_t, true><<<grid, block, 0, stream>>>(
          positions.data_ptr<int64_t>(), query.data_ptr<scalar_t>(),
          key.data_ptr<scalar_t>(), cos_sin_cache.data_ptr<scalar_t>(), rot_dim,
          query_stride, key_stride, num_heads, num_kv_heads, head_size);
    } else {
      vllm::rotary_embedding_kernel<scalar_t, false>
          <<<grid, block, 0, stream>>>(
              positions.data_ptr<int64_t>(), query.data_ptr<scalar_t>(),
              key.data_ptr<scalar_t>(), cos_sin_cache.data_ptr<scalar_t>(),
              rot_dim, query_stride, key_stride, num_heads, num_kv_heads,
              head_size);
    }
  });
}

/*
Batched version of rotary embedding, pack multiple LoRAs together
and process in batched manner.
*/
void batched_rotary_embedding(
    torch::Tensor& positions,  // [batch_size, seq_len] or [num_tokens]
    torch::Tensor& query,  // [batch_size, seq_len, num_heads * head_size] or
                           // [num_tokens, num_heads * head_size] or
                           // [batch_size, seq_len, num_heads, head_size] or
                           // [num_tokens, num_heads, head_size]
    torch::Tensor& key,    // [batch_size, seq_len, num_kv_heads * head_size] or
                           // [num_tokens, num_kv_heads * head_size] or
                           // [batch_size, seq_len, num_heads, head_size] or
                           // [num_tokens, num_heads, head_size]
    int64_t head_size,
    torch::Tensor& cos_sin_cache,  // [max_position, rot_dim]
    bool is_neox, int64_t rot_dim,
    torch::Tensor& cos_sin_cache_offsets  // [num_tokens] or [batch_size]
) {
  // num_tokens = batch_size * seq_len
  int64_t num_tokens = cos_sin_cache_offsets.size(0);
  TORCH_CHECK(
      positions.size(0) == num_tokens || positions.numel() == num_tokens,
      "positions must have the same num_tokens or batch_size as "
      "cos_sin_cache_offsets");

  int positions_ndim = positions.dim();
  // Make sure num_tokens dim is consistent across positions, query, and key.
  TORCH_CHECK(
      positions_ndim == 1 || positions_ndim == 2,
      "positions must have shape [num_tokens] or [batch_size, seq_len]");
  if (positions_ndim == 1) {
    TORCH_CHECK(
        query.size(0) == positions.size(0) && key.size(0) == positions.size(0),
        "query, key and positions must have the same number of tokens");
  }
  if (positions_ndim == 2) {
    TORCH_CHECK(
        query.size(0) == positions.size(0) &&
            key.size(0) == positions.size(0) &&
            query.size(1) == positions.size(1) &&
            key.size(1) == positions.size(1),
        "query, key and positions must have the same batch_size and seq_len");
  }

  // Make sure head_size is valid for query and key
  int query_hidden_size = query.numel() / num_tokens;
  int key_hidden_size = key.numel() / num_tokens;
  TORCH_CHECK(query_hidden_size % head_size == 0);
  TORCH_CHECK(key_hidden_size % head_size == 0);

  // Make sure query and key have concistent number of heads
  int num_heads = query_hidden_size / head_size;
  int num_kv_heads = key_hidden_size / head_size;
  TORCH_CHECK(num_heads % num_kv_heads == 0);

  int seq_dim_idx = positions_ndim - 1;
  int64_t query_stride = query.stride(seq_dim_idx);
  int64_t key_stride = key.stride(seq_dim_idx);

  dim3 grid(num_tokens);
  dim3 block(std::min<int64_t>(num_heads * rot_dim / 2, 512));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(query));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(query.scalar_type(), "rotary_embedding", [&] {
    if (is_neox) {
      vllm::batched_rotary_embedding_kernel<scalar_t, true>
          <<<grid, block, 0, stream>>>(
              positions.data_ptr<int64_t>(), query.data_ptr<scalar_t>(),
              key.data_ptr<scalar_t>(), cos_sin_cache.data_ptr<scalar_t>(),
              cos_sin_cache_offsets.data_ptr<int64_t>(), rot_dim, query_stride,
              key_stride, num_heads, num_kv_heads, head_size);
    } else {
      vllm::batched_rotary_embedding_kernel<scalar_t, false>
          <<<grid, block, 0, stream>>>(
              positions.data_ptr<int64_t>(), query.data_ptr<scalar_t>(),
              key.data_ptr<scalar_t>(), cos_sin_cache.data_ptr<scalar_t>(),
              cos_sin_cache_offsets.data_ptr<int64_t>(), rot_dim, query_stride,
              key_stride, num_heads, num_kv_heads, head_size);
    }
  });
}
