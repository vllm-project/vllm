#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

namespace cacheflow {

template<typename scalar_t>
__global__ void rotary_embedding_neox_kernel(
  scalar_t* __restrict__ out_query,             // [num_tokens, num_heads, head_size]
  scalar_t* __restrict__ out_key,               // [num_tokens, num_heads, head_size]
  const int64_t* __restrict__ positions,        // [num_tokens]
  const scalar_t* __restrict__ query,           // [num_tokens, num_heads, head_size]
  const scalar_t* __restrict__ key,             // [num_tokens, num_heads, head_size]
  const scalar_t* __restrict__ cos_sin_cache,   // [max_position, 2, head_size // 2]
  const int num_heads,
  const int head_size) {
  // Each thread block is responsible for one token.
  const int token_idx = blockIdx.x;
  int64_t pos = positions[token_idx];
  const scalar_t* cache_ptr = cos_sin_cache + pos * head_size;

  const int embed_dim = head_size / 2;
  const int n = num_heads * head_size;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const int idx = token_idx * n + i;

    const int head_idx = i / head_size;
    const int head_offset = i % head_size;
    const int token_head = token_idx * n + head_idx * head_size;

    const bool is_first_half = head_offset < embed_dim;
    const int rot_offset = head_offset % embed_dim;
    const int x_index = rot_offset;
    const int y_index = embed_dim + rot_offset;

    const scalar_t cos = __ldg(cache_ptr + x_index);
    const scalar_t sin = __ldg(cache_ptr + y_index);

    const scalar_t q_x = __ldg(query + token_head + x_index);
    const scalar_t q_y = __ldg(query + token_head + y_index);
    const scalar_t q_cos = is_first_half ? q_x : q_y;
    const scalar_t q_sin = is_first_half ? -q_y : q_x;
    out_query[idx] = q_cos * cos + q_sin * sin;

    const scalar_t k_x = __ldg(key + token_head + x_index);
    const scalar_t k_y = __ldg(key + token_head + y_index);
    const scalar_t k_cos = is_first_half ? k_x : k_y;
    const scalar_t k_sin = is_first_half ? -k_y : k_x;
    out_key[idx] = k_cos * cos + k_sin * sin;
  }
}

} // namespace cacheflow

void rotary_embedding_neox(
  torch::Tensor& out_query,         // [num_tokens, num_heads * head_size]
  torch::Tensor& out_key,           // [num_tokens, num_heads * head_size]
  torch::Tensor& positions,         // [num_tokens]
  torch::Tensor& query,             // [num_tokens, num_heads * head_size]
  torch::Tensor& key,               // [num_tokens, num_heads * head_size]
  torch::Tensor& cos_sin_cache)     // [max_position, head_size]
{
  int num_tokens = query.size(0);
  int head_size = cos_sin_cache.size(1);
  int num_heads = query.size(1) / head_size;

  dim3 grid(num_tokens);
  dim3 block(std::min(num_heads * head_size, 512));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    query.scalar_type(),
    "rotary_embedding_neox",
    [&] {
      cacheflow::rotary_embedding_neox_kernel<scalar_t><<<grid, block, 0, stream>>>(
        out_query.data_ptr<scalar_t>(),
        out_key.data_ptr<scalar_t>(),
        positions.data_ptr<int64_t>(),
        query.data_ptr<scalar_t>(),
        key.data_ptr<scalar_t>(),
        cos_sin_cache.data_ptr<scalar_t>(),
        num_heads,
        head_size);
    });
}
