/*

Adapted from the VLLM project:
https://github.com/vllm-project/vllm/blob/main/csrc/pos_encoding_kernels.cu

*/

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "pos_encoding.h"

template<typename scalar_t>
__global__ void rotary_embedding_neox_kernel(
  const int64_t* __restrict__ positions,        // [num_tokens]
  scalar_t* __restrict__ query,                 // [num_tokens, num_heads, head_size]
  scalar_t* __restrict__ key,                   // [num_tokens, num_heads, head_size]
  const scalar_t* __restrict__ cos_sin_cache,   // [max_position, 2, rot_dim // 2]
  const int rot_dim,
  const int stride,
  const int num_heads,
  const int head_size) {
  // Each thread block is responsible for one token.
  const int token_idx = blockIdx.x;
  int64_t pos = positions[token_idx];
  const scalar_t* cache_ptr = cos_sin_cache + pos * rot_dim;

  const int embed_dim = rot_dim / 2;
  const int n = num_heads * embed_dim;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const int head_idx = i / embed_dim;
    const int token_head = token_idx * stride + head_idx * head_size;

    const int rot_offset = i % embed_dim;
    const int x_index = rot_offset;
    const int y_index = embed_dim + rot_offset;

    const int out_x = token_idx * stride + head_idx * head_size + x_index;
    const int out_y = token_idx * stride + head_idx * head_size + y_index;

    const scalar_t cos = __ldg(cache_ptr + x_index);
    const scalar_t sin = __ldg(cache_ptr + y_index);

    const scalar_t q_x = query[token_head + x_index];
    const scalar_t q_y = query[token_head + y_index];
    query[out_x] = q_x * cos - q_y * sin;
    query[out_y] = q_y * cos + q_x * sin;

    const scalar_t k_x = key[token_head + x_index];
    const scalar_t k_y = key[token_head + y_index];
    key[out_x] = k_x * cos - k_y * sin;
    key[out_y] = k_y * cos + k_x * sin;
  }
}

void rotary_embedding_neox(
  torch::Tensor& positions,         // [b, num_tokens]
  torch::Tensor& query,             // [b, num_tokens, 1, num_heads, head_size]
  torch::Tensor& key,               // [b, num_tokens, 1, num_heads, head_size]
  int head_size,
  torch::Tensor& cos_sin_cache)     // [max_position, rot_dim]
{
  int num_tokens = query.size(0) * query.size(1);
  int rot_dim = cos_sin_cache.size(1);
  int num_heads = query.size(-2);
  int stride = num_heads * head_size;
  // TORCH_CHECK(stride == key.stride(0));

  dim3 grid(num_tokens);
  dim3 block(std::min(num_heads * rot_dim / 2, 512));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::ScalarType::Half,
    at::ScalarType::BFloat16,
    query.scalar_type(),
    "rotary_embedding_neox",
    [&] {
      rotary_embedding_neox_kernel<scalar_t><<<grid, block, 0, stream>>>(
        positions.data_ptr<int64_t>(),
        query.data_ptr<scalar_t>(),
        key.data_ptr<scalar_t>(),
        cos_sin_cache.data_ptr<scalar_t>(),
        rot_dim,
        stride,
        num_heads,
        head_size);
    });
}

