#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "dispatch_utils.h"

namespace vllm {

template <typename scalar_t, typename input_type, bool IS_NEOX,
            bool use_dequant>
  inline __device__ void apply_rotary_embedding(
      input_type * __restrict__ arr, const scalar_t *__restrict__ cos_ptr,
      const scalar_t *__restrict__ sin_ptr, int rot_offset, int embed_dim,
      scalar_t *__restrict__ arr_out = nullptr, const float scale = 1.0f) {
    int x_index, y_index;
    scalar_t cos, sin;
    if constexpr (IS_NEOX) {
      // GPT-NeoX style rotary embedding.
      x_index = rot_offset;
      y_index = embed_dim + rot_offset;
      cos = __ldg(cos_ptr + x_index);
      sin = __ldg(sin_ptr + x_index);
    } else {
      // GPT-J style rotary embedding.
      x_index = 2 * rot_offset;
      y_index = 2 * rot_offset + 1;
      cos = __ldg(cos_ptr + x_index / 2);
      sin = __ldg(sin_ptr + x_index / 2);
    }
    input_type x = arr[x_index];
    input_type y = arr[y_index];
    if constexpr (use_dequant) {
      x = (scalar_t)((float)(x)*scale);
      y = (scalar_t)((float)(y)*scale);
    }
    if constexpr (use_dequant) {
      arr_out[x_index] = x * cos - y * sin;
      arr_out[y_index] = y * cos + x * sin;
    } else {
      arr[x_index] = x * cos - y * sin;
      arr[y_index] = y * cos + x * sin;
    }
  }

template <typename scalar_t, typename input_type, bool IS_NEOX,
            bool use_dequant>
  __global__ void rotary_embedding_kernel(
      const int64_t *__restrict__ positions, 
      input_type *__restrict__ query, 
      input_type *__restrict__ key,
      const scalar_t
          *__restrict__ cos_sin_cache,
      const int rot_dim, const int query_stride, const int key_stride,
      const int num_heads, const int num_kv_heads, const int head_size,
      scalar_t *__restrict__ query_out =
          nullptr,
      scalar_t * __restrict__ key_out =
          nullptr,
      const int query_out_stride = 1, const int key_out_stride = 1,
      const float query_scale = 1.0f, const float key_scale = 1.0f) {
    // Each thread block is responsible for one token.
    const int token_idx = blockIdx.x;
    int64_t pos = positions[token_idx];
    const scalar_t *cache_ptr = cos_sin_cache + pos * rot_dim;

    const int embed_dim = rot_dim / 2;
    const scalar_t *cos_ptr = cache_ptr;
    const scalar_t *sin_ptr = cache_ptr + embed_dim;

    const int nq = num_heads * embed_dim;
    for (int i = threadIdx.x; i < nq; i += blockDim.x) {
      const int head_idx = i / embed_dim;
      const int token_head = token_idx * query_stride + head_idx * head_size;
      const int rot_offset = i % embed_dim;
      if constexpr (use_dequant) {
        const int token_out_head =
            token_idx * query_out_stride + head_idx * head_size;
        apply_rotary_embedding<scalar_t, input_type, IS_NEOX, use_dequant>(
            query + token_head, cos_ptr, sin_ptr, rot_offset, embed_dim,
            query_out + token_out_head, query_scale);
      } else {
        apply_rotary_embedding<scalar_t, input_type, IS_NEOX, use_dequant>(
            query + token_head, cos_ptr, sin_ptr, rot_offset, embed_dim);
      }
    }

    const int nk = num_kv_heads * embed_dim;
    for (int i = threadIdx.x; i < nk; i += blockDim.x) {
      const int head_idx = i / embed_dim;
      const int token_head = token_idx * key_stride + head_idx * head_size;
      const int rot_offset = i % embed_dim;
      if (use_dequant) {
        const int token_out_head =
            token_idx * key_out_stride + head_idx * head_size;
        apply_rotary_embedding<scalar_t, input_type, IS_NEOX, use_dequant>(
            key + token_head, cos_ptr, sin_ptr, rot_offset, embed_dim,
            key_out + token_out_head, key_scale);
      } else {
        apply_rotary_embedding<scalar_t, input_type, IS_NEOX, use_dequant>(
            key + token_head, cos_ptr, sin_ptr, rot_offset, embed_dim);
      }
    }
  }

} // namespace vllm

void rotary_embedding(
    torch::Tensor &positions, // [batch_size, seq_len] or [num_tokens]
    torch::Tensor &query,     // [batch_size, seq_len, num_heads * head_size] or [num_tokens, num_heads * head_size]
    torch::Tensor &key,       // [batch_size, seq_len, num_heads * head_size] or [num_tokens, num_heads * head_size]
    int head_size,
    torch::Tensor &cos_sin_cache, // [max_position, rot_dim]
    bool is_neox,
    torch::Tensor &query_out, // [batch_size, seq_len, num_heads * head_size] or [num_tokens, num_heads * head_size]
    torch::Tensor &key_out, // [batch_size, seq_len, num_heads * head_size] or [num_tokens, num_heads * head_size]
    bool use_dequant = false, const float query_scale = 1.0f,
    const float key_scale = 1.0f) {
  int num_tokens = query.size(0);
  int rot_dim = cos_sin_cache.size(1);
  int num_heads = query.size(-1) / head_size;
  int num_kv_heads = key.size(-1) / head_size;
  int query_stride = query.stride(-2);
  int key_stride = key.stride(-2);

  dim3 grid(num_tokens);
  dim3 block(std::min(num_heads * rot_dim / 2, 512));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(
      cos_sin_cache.scalar_type(), "rotary_embedding_kernel", [&] {
        if (use_dequant) {
          int query_out_stride = query_out.stride(-2);
          int key_out_stride = key_out.stride(-2);
          if (is_neox) {
            vllm::rotary_embedding_kernel<scalar_t, int32_t, true, true>
                <<<grid, block, 0, stream>>>(
                    positions.data_ptr<int64_t>(), query.data_ptr<int32_t>(),
                    key.data_ptr<int32_t>(), cos_sin_cache.data_ptr<scalar_t>(),
                    rot_dim, query_stride, key_stride, num_heads, num_kv_heads,
                    head_size, query_out.data_ptr<scalar_t>(),
                    key_out.data_ptr<scalar_t>(), query_out_stride,
                    key_out_stride, query_scale, key_scale);
          } else {
            vllm::rotary_embedding_kernel<scalar_t, int32_t, false, true>
                <<<grid, block, 0, stream>>>(
                    positions.data_ptr<int64_t>(), query.data_ptr<int32_t>(),
                    key.data_ptr<int32_t>(), cos_sin_cache.data_ptr<scalar_t>(),
                    rot_dim, query_stride, key_stride, num_heads, num_kv_heads,
                    head_size, query_out.data_ptr<scalar_t>(),
                    key_out.data_ptr<scalar_t>(), query_out_stride,
                    key_out_stride, query_scale, key_scale);
          }
        } else {
          if (is_neox) {
            vllm::rotary_embedding_kernel<scalar_t, scalar_t, true, false>
                <<<grid, block, 0, stream>>>(
                    positions.data_ptr<int64_t>(), query.data_ptr<scalar_t>(),
                    key.data_ptr<scalar_t>(),
                    cos_sin_cache.data_ptr<scalar_t>(), rot_dim, query_stride,
                    key_stride, num_heads, num_kv_heads, head_size);
          } else {
            vllm::rotary_embedding_kernel<scalar_t, scalar_t, false, true>
                <<<grid, block, 0, stream>>>(
                    positions.data_ptr<int64_t>(), query.data_ptr<scalar_t>(),
                    key.data_ptr<scalar_t>(),
                    cos_sin_cache.data_ptr<scalar_t>(), rot_dim, query_stride,
                    key_stride, num_heads, num_kv_heads, head_size);
          }
        }
      });
}
