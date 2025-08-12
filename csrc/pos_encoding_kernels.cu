#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

namespace cacheflow {

template<typename scalar_t>
__global__ void rotary_embedding_neox_kernel(
  const int64_t* __restrict__ positions,        // [num_tokens]
  scalar_t* __restrict__ query,                 // [num_tokens, num_heads, head_size]
  scalar_t* __restrict__ key,                   // [num_tokens, num_heads, head_size]
  const scalar_t* __restrict__ cos_sin_cache,   // [max_position, 2, head_size // 2]
  const int stride,
  const int num_heads,
  const int head_size) {
  // Each thread block is responsible for one token.
  const int token_idx = blockIdx.x;
  int64_t pos = positions[token_idx];
  const scalar_t* cache_ptr = cos_sin_cache + pos * head_size;

  const int embed_dim = head_size / 2;
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

} // namespace cacheflow

void rotary_embedding_neox(
  torch::Tensor& positions,         // [num_tokens]
  torch::Tensor& query,             // [num_tokens, num_heads * head_size]
  torch::Tensor& key,               // [num_tokens, num_heads * head_size]
  torch::Tensor& cos_sin_cache)     // [max_position, head_size]
{
  int num_tokens = query.size(0);
  int head_size = cos_sin_cache.size(1);
  int num_heads = query.size(1) / head_size;
  int stride = query.stride(0);
  TORCH_CHECK(stride == key.stride(0));

  dim3 grid(num_tokens);
  dim3 block(std::min(num_heads * head_size / 2, 512));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    query.scalar_type(),
    "rotary_embedding_neox",
    [&] {
      cacheflow::rotary_embedding_neox_kernel<scalar_t><<<grid, block, 0, stream>>>(
        positions.data_ptr<int64_t>(),
        query.data_ptr<scalar_t>(),
        key.data_ptr<scalar_t>(),
        cos_sin_cache.data_ptr<scalar_t>(),
        stride,
        num_heads,
        head_size);
    });
}
