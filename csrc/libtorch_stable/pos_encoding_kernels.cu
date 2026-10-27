#include "torch_utils.h"

#include "../cuda_compat.h"
#include "dispatch_utils.h"

namespace vllm {

template <typename scalar_t, typename cache_t, bool IS_NEOX>
inline __device__ void apply_token_rotary_embedding(
    scalar_t* __restrict__ arr, const cache_t* __restrict__ cos_ptr,
    const cache_t* __restrict__ sin_ptr, int rot_offset, int embed_dim,
    const bool inverse) {
  int x_index, y_index;
  float cos_f, sin_f;
  if (IS_NEOX) {
    x_index = rot_offset;
    y_index = embed_dim + rot_offset;
    cos_f = static_cast<float>(VLLM_LDG(cos_ptr + x_index));
    sin_f = static_cast<float>(VLLM_LDG(sin_ptr + x_index));
  } else {
    x_index = 2 * rot_offset;
    y_index = 2 * rot_offset + 1;
    cos_f = static_cast<float>(VLLM_LDG(cos_ptr + x_index / 2));
    sin_f = static_cast<float>(VLLM_LDG(sin_ptr + x_index / 2));
  }
  if (inverse) {
    sin_f = -sin_f;
  }
  const float x_f = static_cast<float>(arr[x_index]);
  const float y_f = static_cast<float>(arr[y_index]);
  arr[x_index] = static_cast<scalar_t>(x_f * cos_f - y_f * sin_f);
  arr[y_index] = static_cast<scalar_t>(y_f * cos_f + x_f * sin_f);
}

template <typename scalar_t, typename cache_t, bool IS_NEOX>
inline __device__ void apply_rotary_embedding(
    scalar_t* __restrict__ query,  // [batch_size, seq_len, num_heads,
                                   // head_size] or [num_tokens, num_heads,
                                   // head_size]
    scalar_t* __restrict__ key,    // nullptr or
                                   // [batch_size, seq_len, num_kv_heads,
                                   // head_size] or [num_tokens, num_kv_heads,
                                   // head_size]
    const cache_t* cache_ptr, const int head_size, const int num_heads,
    const int num_kv_heads, const int rot_dim, const int token_idx,
    const int64_t query_stride, const int64_t key_stride,
    const int64_t head_stride, const int64_t rope_dim_offset,
    const bool inverse) {
  const int embed_dim = rot_dim / 2;
  const cache_t* cos_ptr = cache_ptr;
  const cache_t* sin_ptr = cache_ptr + embed_dim;

  const int nq = num_heads * embed_dim;
  for (int i = threadIdx.x; i < nq; i += blockDim.x) {
    const int head_idx = i / embed_dim;
    const int64_t token_head =
        token_idx * query_stride + head_idx * head_stride + rope_dim_offset;
    const int rot_offset = i % embed_dim;
    apply_token_rotary_embedding<scalar_t, cache_t, IS_NEOX>(
        query + token_head, cos_ptr, sin_ptr, rot_offset, embed_dim, inverse);
  }

  if (key != nullptr) {
    const int nk = num_kv_heads * embed_dim;
    for (int i = threadIdx.x; i < nk; i += blockDim.x) {
      const int head_idx = i / embed_dim;
      const int64_t token_head =
          token_idx * key_stride + head_idx * head_stride + rope_dim_offset;
      const int rot_offset = i % embed_dim;
      apply_token_rotary_embedding<scalar_t, cache_t, IS_NEOX>(
          key + token_head, cos_ptr, sin_ptr, rot_offset, embed_dim, inverse);
    }
  }
}

template <typename scalar_t, typename cache_t, bool IS_NEOX>
__global__ void rotary_embedding_kernel(
    const int64_t* __restrict__ positions,  // [batch_size, seq_len] or
                                            // [num_tokens]
    scalar_t* __restrict__ query,           // [batch_size, seq_len, num_heads,
                                   // head_size] or [num_tokens, num_heads,
                                   // head_size]
    scalar_t* __restrict__ key,  // nullptr or
                                 // [batch_size, seq_len, num_kv_heads,
                                 // head_size] or [num_tokens, num_kv_heads,
                                 // head_size]
    const cache_t* __restrict__ cos_sin_cache,  // [max_position, rot_dim]
    const int rot_dim, const int64_t query_stride, const int64_t key_stride,
    const int64_t head_stride, const int num_heads, const int num_kv_heads,
    const int head_size, const int64_t rope_dim_offset, const bool inverse) {
  const int token_idx = blockIdx.x;
  int64_t pos = positions[token_idx];
  const cache_t* cache_ptr = cos_sin_cache + pos * rot_dim;

  apply_rotary_embedding<scalar_t, cache_t, IS_NEOX>(
      query, key, cache_ptr, head_size, num_heads, num_kv_heads, rot_dim,
      token_idx, query_stride, key_stride, head_stride, rope_dim_offset,
      inverse);
}

}  // namespace vllm

void rotary_embedding(
    torch::stable::Tensor& positions,  // [batch_size, seq_len] or [num_tokens]
    torch::stable::Tensor&
        query,  // [batch_size, seq_len, num_heads * head_size] or
                // [num_tokens, num_heads * head_size] or
                // [batch_size, seq_len, num_heads, head_size] or
                // [num_tokens, num_heads, head_size]
    std::optional<torch::stable::Tensor> key,
    // null or
    // [batch_size, seq_len, num_kv_heads * head_size] or
    // [num_tokens, num_kv_heads * head_size] or
    // [batch_size, seq_len, num_heads, head_size] or
    // [num_tokens, num_heads, head_size]
    int64_t head_size,
    torch::stable::Tensor& cos_sin_cache,  // [max_position, rot_dim]
    bool is_neox, int64_t rope_dim_offset, bool inverse) {
  // num_tokens = batch_size * seq_len
  int64_t num_tokens = positions.numel();
  int positions_ndim = positions.dim();

  // Make sure num_tokens dim is consistent across positions, query, and key
  STD_TORCH_CHECK(
      positions_ndim == 1 || positions_ndim == 2,
      "positions must have shape [num_tokens] or [batch_size, seq_len]");
  if (positions_ndim == 1) {
    STD_TORCH_CHECK(
        query.size(0) == positions.size(0) &&
            (!key.has_value() || key->size(0) == positions.size(0)),
        "query, key and positions must have the same number of tokens");
  }
  if (positions_ndim == 2) {
    STD_TORCH_CHECK(
        query.size(0) == positions.size(0) &&
            (!key.has_value() || key->size(0) == positions.size(0)) &&
            query.size(1) == positions.size(1) &&
            (!key.has_value() || key->size(1) == positions.size(1)),
        "query, key and positions must have the same batch_size and seq_len");
  }

  // Make sure head_size is valid for query and key
  // hidden_size = num_heads * head_size
  int query_hidden_size = query.numel() / num_tokens;
  int key_hidden_size = key.has_value() ? key->numel() / num_tokens : 0;
  STD_TORCH_CHECK(query_hidden_size % head_size == 0,
                  "query hidden size must be a multiple of head_size");
  STD_TORCH_CHECK(key_hidden_size % head_size == 0,
                  "key hidden size must be a multiple of head_size");

  // Make sure query and key have consistent number of heads
  int num_heads = query_hidden_size / head_size;
  int num_kv_heads = key.has_value() ? key_hidden_size / head_size : num_heads;
  STD_TORCH_CHECK(num_heads % num_kv_heads == 0);

  int rot_dim = cos_sin_cache.size(1);
  int seq_dim_idx = positions_ndim - 1;
  int64_t query_stride = query.stride(seq_dim_idx);
  int64_t key_stride = key.has_value() ? key->stride(seq_dim_idx) : 0;

  STD_TORCH_CHECK((rot_dim + rope_dim_offset) <= head_size);
  // Determine head stride: for [*, heads, head_size] use stride of last dim;
  // for flat [*, heads*head_size], heads blocks are contiguous of size
  // head_size
  int query_ndim = query.dim();
  int64_t head_stride =
      (query_ndim == positions_ndim + 2) ? query.stride(-2) : head_size;

  dim3 grid(num_tokens);
  dim3 block(std::min<int64_t>(num_heads * rot_dim / 2, 512));
  const torch::stable::accelerator::DeviceGuard device_guard(
      query.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream();
  VLLM_STABLE_DISPATCH_FLOATING_TYPES(
      query.scalar_type(), "rotary_embedding", [&] {
        using query_t = scalar_t;
        VLLM_STABLE_DISPATCH_FLOATING_TYPES(
            cos_sin_cache.scalar_type(), "rotary_embedding_cache", [&] {
              using cache_t = scalar_t;
              if (is_neox) {
                vllm::rotary_embedding_kernel<query_t, cache_t, true>
                    <<<grid, block, 0, stream>>>(
                        positions.const_data_ptr<int64_t>(),
                        query.mutable_data_ptr<query_t>(),
                        key.has_value() ? key->mutable_data_ptr<query_t>()
                                        : nullptr,
                        cos_sin_cache.const_data_ptr<cache_t>(), rot_dim,
                        query_stride, key_stride, head_stride, num_heads,
                        num_kv_heads, head_size, rope_dim_offset, inverse);
              } else {
                vllm::rotary_embedding_kernel<query_t, cache_t, false>
                    <<<grid, block, 0, stream>>>(
                        positions.const_data_ptr<int64_t>(),
                        query.mutable_data_ptr<query_t>(),
                        key.has_value() ? key->mutable_data_ptr<query_t>()
                                        : nullptr,
                        cos_sin_cache.const_data_ptr<cache_t>(), rot_dim,
                        query_stride, key_stride, head_stride, num_heads,
                        num_kv_heads, head_size, rope_dim_offset, inverse);
              }
            });
      });
}

namespace vllm {

// Out-of-place RoPE: reads heads from src, writes rotated heads to dst and
// copies the non-rotated tail (partial rotary) so dst is fully defined.
template <typename scalar_t, typename cache_t, bool IS_NEOX>
inline __device__ void rotate_heads_out(
    const scalar_t* src, scalar_t* dst, const int num_heads,
    const int64_t src_head_stride, const int64_t dst_head_stride,
    const cache_t* __restrict__ cos_ptr, const cache_t* __restrict__ sin_ptr,
    const int embed_dim, const int rot_dim, const int head_size) {
  for (int i = threadIdx.x; i < num_heads * embed_dim; i += blockDim.x) {
    const int head_idx = i / embed_dim;
    const int rot_offset = i % embed_dim;
    const scalar_t* s = src + head_idx * src_head_stride;
    scalar_t* d = dst + head_idx * dst_head_stride;
    const int x_index = IS_NEOX ? rot_offset : 2 * rot_offset;
    const int y_index = IS_NEOX ? embed_dim + rot_offset : 2 * rot_offset + 1;
    const float cos_f = static_cast<float>(VLLM_LDG(cos_ptr + rot_offset));
    const float sin_f = static_cast<float>(VLLM_LDG(sin_ptr + rot_offset));
    const float x_f = static_cast<float>(s[x_index]);
    const float y_f = static_cast<float>(s[y_index]);
    d[x_index] = static_cast<scalar_t>(x_f * cos_f - y_f * sin_f);
    d[y_index] = static_cast<scalar_t>(y_f * cos_f + x_f * sin_f);
  }
  const int pass_dim = head_size - rot_dim;
  for (int i = threadIdx.x; i < num_heads * pass_dim; i += blockDim.x) {
    const int head_idx = i / pass_dim;
    const int offset = rot_dim + i % pass_dim;
    dst[head_idx * dst_head_stride + offset] =
        src[head_idx * src_head_stride + offset];
  }
}

template <typename scalar_t, typename cache_t, bool IS_NEOX>
__global__ void rotary_embedding_out_kernel(
    const int64_t* __restrict__ positions,  // [num_tokens]
    const scalar_t* query, const scalar_t* key, scalar_t* query_out,
    scalar_t* key_out,
    const cache_t* __restrict__ cos_sin_cache,  // [max_position, rot_dim]
    const int rot_dim, const int64_t query_stride, const int64_t key_stride,
    const int64_t query_out_stride, const int64_t key_out_stride,
    const int64_t query_head_stride, const int64_t key_head_stride,
    const int64_t query_out_head_stride, const int64_t key_out_head_stride,
    const int num_heads, const int num_kv_heads, const int head_size) {
  const int token_idx = blockIdx.x;
  const int64_t pos = positions[token_idx];
  const int embed_dim = rot_dim / 2;
  const cache_t* cos_ptr = cos_sin_cache + pos * rot_dim;
  const cache_t* sin_ptr = cos_ptr + embed_dim;

  rotate_heads_out<scalar_t, cache_t, IS_NEOX>(
      query + token_idx * query_stride,
      query_out + token_idx * query_out_stride, num_heads, query_head_stride,
      query_out_head_stride, cos_ptr, sin_ptr, embed_dim, rot_dim, head_size);
  if (key != nullptr) {
    rotate_heads_out<scalar_t, cache_t, IS_NEOX>(
        key + token_idx * key_stride, key_out + token_idx * key_out_stride,
        num_kv_heads, key_head_stride, key_out_head_stride, cos_ptr, sin_ptr,
        embed_dim, rot_dim, head_size);
  }
}

}  // namespace vllm

void rotary_embedding_out(
    torch::stable::Tensor& positions,  // [num_tokens]
    torch::stable::Tensor& query,      // [num_tokens, num_heads * head_size] or
                                       // [num_tokens, num_heads, head_size]
    std::optional<torch::stable::Tensor> key,
    torch::stable::Tensor& query_out,  // same shape as query
    std::optional<torch::stable::Tensor> key_out, int64_t head_size,
    torch::stable::Tensor& cos_sin_cache,  // [max_position, rot_dim]
    bool is_neox) {
  STD_TORCH_CHECK(positions.dim() == 1, "positions must be [num_tokens]");
  STD_TORCH_CHECK(
      positions.scalar_type() == torch::headeronly::ScalarType::Long,
      "positions must be int64");
  STD_TORCH_CHECK(cos_sin_cache.is_contiguous(),
                  "cos_sin_cache must be contiguous");
  STD_TORCH_CHECK(key.has_value() == key_out.has_value(),
                  "key and key_out must both be given or both be None");
  STD_TORCH_CHECK(
      query_out.scalar_type() == query.scalar_type() &&
          (!key.has_value() || (key->scalar_type() == query.scalar_type() &&
                                key_out->scalar_type() == query.scalar_type())),
      "query, key and their outputs must share a dtype");
  const int64_t num_tokens = positions.size(0);
  STD_TORCH_CHECK(query_out.numel() == query.numel() &&
                      (!key.has_value() || key_out->numel() == key->numel()),
                  "outputs must have as many elements as their inputs");
  STD_TORCH_CHECK(
      query.size(0) == num_tokens && query_out.size(0) == num_tokens &&
          query.dim() == query_out.dim() &&
          (!key.has_value() ||
           (key->size(0) == num_tokens && key_out->size(0) == num_tokens &&
            key->dim() == query.dim() && key_out->dim() == query.dim())),
      "query, key, their outputs and positions must agree");
  STD_TORCH_CHECK(query.dim() == 2 || query.dim() == 3,
                  "query must be [num_tokens, hidden] or "
                  "[num_tokens, heads, head_size]");
  STD_TORCH_CHECK(query.stride(-1) == 1 && query_out.stride(-1) == 1 &&
                      (!key.has_value() ||
                       (key->stride(-1) == 1 && key_out->stride(-1) == 1)),
                  "the last dimension must be contiguous");

  const int query_hidden_size = query.numel() / num_tokens;
  const int key_hidden_size = key.has_value() ? key->numel() / num_tokens : 0;
  STD_TORCH_CHECK(query_hidden_size % head_size == 0);
  STD_TORCH_CHECK(key_hidden_size % head_size == 0);
  const int num_heads = query_hidden_size / head_size;
  const int num_kv_heads =
      key.has_value() ? key_hidden_size / head_size : num_heads;

  const int rot_dim = cos_sin_cache.size(1);
  STD_TORCH_CHECK(rot_dim <= head_size && rot_dim % 2 == 0,
                  "rotary dim must be even and at most head_size");

  // [num_tokens, heads, head_size] tensors may have any head stride (e.g. a
  // slice of wider heads); flat [num_tokens, hidden] heads are contiguous.
  const bool per_head = query.dim() == 3;
  auto head_stride = [&](const torch::stable::Tensor& t) -> int64_t {
    return per_head ? t.stride(1) : head_size;
  };
  STD_TORCH_CHECK(
      !per_head ||
          (query.size(2) == head_size && query_out.size(2) == head_size &&
           (!key.has_value() ||
            (key->size(2) == head_size && key_out->size(2) == head_size))),
      "3D tensors must be [num_tokens, heads, head_size]");
  const int64_t key_head_stride = key.has_value() ? head_stride(*key) : 0;
  const int64_t key_out_head_stride =
      key.has_value() ? head_stride(*key_out) : 0;

  dim3 grid(num_tokens);
  dim3 block(std::min<int64_t>(num_heads * head_size / 2, 512));
  const torch::stable::accelerator::DeviceGuard device_guard(
      query.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream();
  VLLM_STABLE_DISPATCH_FLOATING_TYPES(
      query.scalar_type(), "rotary_embedding_out", [&] {
        using query_t = scalar_t;
        VLLM_STABLE_DISPATCH_FLOATING_TYPES(
            cos_sin_cache.scalar_type(), "rotary_embedding_out_cache", [&] {
              using cache_t = scalar_t;
              const query_t* key_ptr =
                  key.has_value() ? key->const_data_ptr<query_t>() : nullptr;
              query_t* key_out_ptr = key.has_value()
                                         ? key_out->mutable_data_ptr<query_t>()
                                         : nullptr;
              const int64_t key_stride = key.has_value() ? key->stride(0) : 0;
              const int64_t key_out_stride =
                  key.has_value() ? key_out->stride(0) : 0;
              auto kernel =
                  is_neox ? vllm::rotary_embedding_out_kernel<query_t, cache_t,
                                                              true>
                          : vllm::rotary_embedding_out_kernel<query_t, cache_t,
                                                              false>;
              kernel<<<grid, block, 0, stream>>>(
                  positions.const_data_ptr<int64_t>(),
                  query.const_data_ptr<query_t>(), key_ptr,
                  query_out.mutable_data_ptr<query_t>(), key_out_ptr,
                  cos_sin_cache.const_data_ptr<cache_t>(), rot_dim,
                  query.stride(0), key_stride, query_out.stride(0),
                  key_out_stride, head_stride(query), key_head_stride,
                  head_stride(query_out), key_out_head_stride, num_heads,
                  num_kv_heads, head_size);
            });
      });
}
