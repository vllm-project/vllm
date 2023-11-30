#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "dispatch_utils.h"

namespace vllm {

template<typename scalar_t, bool IS_NEOX>
inline __device__ void apply_rotary_embedding(
  scalar_t* __restrict__ arr,
  float pos,
  float base,
  int rot_offset,
  int embed_dim,
  int rot_dim)
{
  const float inv_freq = pos / powf(base, rot_offset*2 / (float)rot_dim);
  scalar_t cos = static_cast<scalar_t>(cosf(inv_freq));
  scalar_t sin = static_cast<scalar_t>(sinf(inv_freq));
  if (IS_NEOX) {
    // GPT-NeoX style rotary embedding.
    // x_index = rot_offset;
    // y_index = embed_dim + rot_offset;
    // cos = __ldg(cos_ptr + x_index);
    // sin = __ldg(sin_ptr + x_index);
  } else {
    // GPT-J style rotary embedding.
    // x_index = 2 * rot_offset;
    // y_index = 2 * rot_offset + 1;
    // cos = __ldg(cos_ptr + x_index / 2);
    // sin = __ldg(sin_ptr + x_index / 2);
  }

  int x_index = rot_offset;
  int y_index = embed_dim + rot_offset;

  const scalar_t x = arr[x_index];
  const scalar_t y = arr[y_index];
  arr[x_index] = x * cos - y * sin;
  arr[y_index] = y * cos + x * sin;
}

inline __device__ float rotary_embedding_get_base(
  float true_seq_len,
  int seq_len,
  float rot_dim,
  float base) {
  if (true_seq_len <= seq_len) {
    return base;
  }
  float ntk_alpha = max(exp2f(ceilf(log2f(true_seq_len / seq_len) + 1.f)) - 1.f, 1.f);
  base *= powf(ntk_alpha, rot_dim / (rot_dim - 2.f));
  return base;
}

template<typename scalar_t, bool IS_NEOX>
__global__ void rotary_embedding_kernel(
  const int64_t* __restrict__ positions,        // [batch_size, seq_len] or [num_tokens]
  const int64_t* __restrict__ input_true_seq_len,
  scalar_t* __restrict__ query,                 // [batch_size, seq_len, num_heads, head_size] or [num_tokens, num_heads, head_size]
  scalar_t* __restrict__ key,                   // [batch_size, seq_len, num_kv_heads, head_size] or [num_tokens, num_kv_heads, head_size]
  const scalar_t* __restrict__ cos_sin_cache,   // [max_position, 2, rot_dim // 2]
  const int rot_dim,
  const int query_stride,
  const int key_stride,
  const int num_heads,
  const int num_kv_heads,
  const int head_size,
  const int seq_len) {
  // Each thread block is responsible for one token.
  const int token_idx = blockIdx.x;
  int64_t pos = positions[token_idx];
  const scalar_t* cache_ptr = cos_sin_cache + pos * rot_dim;

  float base = rotary_embedding_get_base(input_true_seq_len[token_idx/seq_len], 2048, rot_dim, 10000);

  const int embed_dim = rot_dim / 2;
  const scalar_t* cos_ptr = cache_ptr;
  const scalar_t* sin_ptr = cache_ptr + embed_dim;

  const int nq = num_heads * embed_dim;
  for (int i = threadIdx.x; i < nq; i += blockDim.x) {
    const int head_idx = i / embed_dim;
    const int token_head = token_idx * query_stride + head_idx * head_size;
    const int rot_offset = i % embed_dim;
    apply_rotary_embedding<scalar_t, IS_NEOX>(query + token_head, pos,
                                              base, rot_offset, embed_dim, rot_dim);
  }

  const int nk = num_kv_heads * embed_dim;
  for (int i = threadIdx.x; i < nk; i += blockDim.x) {
    const int head_idx = i / embed_dim;
    const int token_head = token_idx * key_stride + head_idx * head_size;
    const int rot_offset = i % embed_dim;
    apply_rotary_embedding<scalar_t, IS_NEOX>(key + token_head, pos,
                                              base, rot_offset, embed_dim, rot_dim);
  }
}

} // namespace vllm

void rotary_embedding(
  torch::Tensor& positions,         // [batch_size, seq_len] or [num_tokens]
  torch::Tensor& input_true_seq_len,
  torch::Tensor& query,             // [batch_size, seq_len, num_heads * head_size] or [num_tokens, num_heads * head_size]
  torch::Tensor& key,               // [batch_size, seq_len, num_kv_heads * head_size] or [num_tokens, num_kv_heads * head_size]
  int head_size,
  torch::Tensor& cos_sin_cache,     // [max_position, rot_dim]
  bool is_neox) {
  int num_tokens = query.numel() / query.size(-1);
  int rot_dim = cos_sin_cache.size(1);
  int num_heads = query.size(-1) / head_size;
  int num_kv_heads = key.size(-1) / head_size;
  int query_stride = query.stride(-2);
  int key_stride = key.stride(-2);

  int seq_len = query.size(1);

  assert(input_true_seq_len.size(0) == query.size(0));
  assert(query.size(1) * query.size(0) == num_tokens);

  dim3 grid(num_tokens);
  dim3 block(std::min(num_heads * rot_dim / 2, 512));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(
    query.scalar_type(),
    "rotary_embedding",
    [&] {
      if (is_neox) {
        vllm::rotary_embedding_kernel<scalar_t, true><<<grid, block, 0, stream>>>(
          positions.data_ptr<int64_t>(),
          input_true_seq_len.data_ptr<int64_t>(),
          query.data_ptr<scalar_t>(),
          key.data_ptr<scalar_t>(),
          cos_sin_cache.data_ptr<scalar_t>(),
          rot_dim,
          query_stride,
          key_stride,
          num_heads,
          num_kv_heads,
          head_size,
          seq_len);
      } else {
        vllm::rotary_embedding_kernel<scalar_t, false><<<grid, block, 0, stream>>>(
          positions.data_ptr<int64_t>(),
          input_true_seq_len.data_ptr<int64_t>(),
          query.data_ptr<scalar_t>(),
          key.data_ptr<scalar_t>(),
          cos_sin_cache.data_ptr<scalar_t>(),
          rot_dim,
          query_stride,
          key_stride,
          num_heads,
          num_kv_heads,
          head_size,
          seq_len);
      }
    });
}
