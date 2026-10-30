#include "torch_utils.h"
#include "dispatch_utils.h"

#include <algorithm>

#include "core/cuda_compat.h"
#include "quantization/vectorization.cuh"

namespace vllm {

// Rotates the first rot_dim elements of each head from src into dst and copies
// the rest unchanged. The arithmetic matches the unfused rotary kernel.
template <typename T, bool IS_NEOX>
__device__ __forceinline__ void rotate_heads(
    const T* __restrict__ src, const int64_t src_stride_head,
    T* __restrict__ dst, const int64_t dst_stride_head,
    const T* __restrict__ cos_sin, const int num_heads, const int rot_dim,
    const int head_size) {
  const int embed_dim = rot_dim / 2;
  for (int i = threadIdx.x; i < num_heads * embed_dim; i += blockDim.x) {
    const int head = i / embed_dim;
    const int pair = i % embed_dim;
    const float cos_f = static_cast<float>(VLLM_LDG(cos_sin + pair));
    const float sin_f =
        static_cast<float>(VLLM_LDG(cos_sin + pair + embed_dim));
    const int x_idx = IS_NEOX ? pair : 2 * pair;
    const int y_idx = IS_NEOX ? embed_dim + pair : 2 * pair + 1;
    const T* head_src = src + head * src_stride_head;
    T* head_dst = dst + head * dst_stride_head;
    const float x = static_cast<float>(head_src[x_idx]);
    const float y = static_cast<float>(head_src[y_idx]);
    head_dst[x_idx] = static_cast<T>(x * cos_f - y * sin_f);
    head_dst[y_idx] = static_cast<T>(y * cos_f + x * sin_f);
  }
  const int pass_dim = head_size - rot_dim;
  for (int i = threadIdx.x; i < num_heads * pass_dim; i += blockDim.x) {
    const int head = i / pass_dim;
    const int offset = rot_dim + i % pass_dim;
    dst[head * dst_stride_head + offset] = src[head * src_stride_head + offset];
  }
}

// One thread block per token: rotates Q into a contiguous query_out, rotates K
// directly into the paged cache and copies V there. Tokens past the end of
// slot_mapping, or with a negative slot, only produce Q.
template <typename T, bool IS_NEOX, bool VECTORIZE_VALUE>
__global__ void fused_rope_and_reshape_cache_flash_q_out_kernel(
    const int64_t* __restrict__ positions,  // [num_tokens]
    const T* __restrict__ query,            // [num_tokens, num_q_heads, head]
    const T* __restrict__ key,              // [num_tokens, num_kv_heads, head]
    const T* __restrict__ value,            // [num_tokens, num_kv_heads, head]
    T* __restrict__ query_out,              // contiguous, shaped like query
    const T* __restrict__ cos_sin_cache,    // [max_position, rot_dim]
    T* __restrict__ key_cache,  // [num_blocks, block_size, num_kv_heads, head]
    T* __restrict__ value_cache,  // same logical shape as key_cache
    const int64_t* __restrict__ slot_mapping,  // [num_cache_tokens]
    const int64_t num_cache_tokens, const int rot_dim,
    const int64_t query_stride_token, const int64_t query_stride_head,
    const int64_t key_stride_token, const int64_t key_stride_head,
    const int64_t value_stride_token, const int64_t value_stride_head,
    const int64_t cos_sin_stride_token, const int64_t key_cache_stride_block,
    const int64_t key_cache_stride_token, const int64_t key_cache_stride_head,
    const int64_t value_cache_stride_block,
    const int64_t value_cache_stride_token,
    const int64_t value_cache_stride_head, const int num_q_heads,
    const int num_kv_heads, const int head_size, const int block_size) {
  const int64_t token = blockIdx.x;
  const T* cos_sin = cos_sin_cache + positions[token] * cos_sin_stride_token;
  rotate_heads<T, IS_NEOX>(query + token * query_stride_token,
                           query_stride_head,
                           query_out + token * num_q_heads * head_size,
                           head_size, cos_sin, num_q_heads, rot_dim, head_size);

  if (token >= num_cache_tokens) {
    return;
  }
  const int64_t slot = slot_mapping[token];
  if (slot < 0) {
    return;
  }
  const int64_t block_idx = slot / block_size;
  const int64_t page_offset = slot % block_size;
  rotate_heads<T, IS_NEOX>(key + token * key_stride_token, key_stride_head,
                           key_cache + block_idx * key_cache_stride_block +
                               page_offset * key_cache_stride_token,
                           key_cache_stride_head, cos_sin, num_kv_heads,
                           rot_dim, head_size);

  const T* v_src = value + token * value_stride_token;
  T* v_dst = value_cache + block_idx * value_cache_stride_block +
             page_offset * value_cache_stride_token;
  if constexpr (VECTORIZE_VALUE) {
    constexpr int vec_size = 8;
    using vec_t = vec_n_t<T, vec_size>;
    if (head_size % vec_size == 0 && value_stride_head % vec_size == 0 &&
        value_cache_stride_head % vec_size == 0 &&
        reinterpret_cast<uintptr_t>(v_src) % alignof(vec_t) == 0 &&
        reinterpret_cast<uintptr_t>(v_dst) % alignof(vec_t) == 0) {
      const int head_vecs = head_size / vec_size;
      for (int i = threadIdx.x; i < num_kv_heads * head_vecs; i += blockDim.x) {
        const int head = i / head_vecs;
        reinterpret_cast<vec_t*>(
            v_dst + head * value_cache_stride_head)[i % head_vecs] =
            reinterpret_cast<const vec_t*>(
                v_src + head * value_stride_head)[i % head_vecs];
      }
      return;
    }
  }
  for (int i = threadIdx.x; i < num_kv_heads * head_size; i += blockDim.x) {
    const int head = i / head_size;
    const int offset = i % head_size;
    v_dst[head * value_cache_stride_head + offset] =
        v_src[head * value_stride_head + offset];
  }
}

}  // namespace vllm

#define CALL_FUSED_ROPE_AND_RESHAPE_CACHE_FLASH_Q_OUT(IS_NEOX,                 \
                                                      VECTORIZE_VALUE)         \
  do {                                                                         \
    VLLM_STABLE_DISPATCH_HALF_TYPES(                                           \
        query.scalar_type(), "qk_scalar_type", [&] {                           \
          using qk_t = scalar_t;                                               \
          vllm::fused_rope_and_reshape_cache_flash_q_out_kernel<               \
              qk_t, IS_NEOX, VECTORIZE_VALUE><<<grid, block, 0, stream>>>(     \
              positions.const_data_ptr<int64_t>(),                             \
              query.const_data_ptr<qk_t>(), key.const_data_ptr<qk_t>(),        \
              value.const_data_ptr<qk_t>(),                                    \
              query_out.mutable_data_ptr<qk_t>(),                              \
              cos_sin_cache.const_data_ptr<qk_t>(),                            \
              key_cache.mutable_data_ptr<qk_t>(),                              \
              value_cache.mutable_data_ptr<qk_t>(),                            \
              slot_mapping.const_data_ptr<int64_t>(), num_cache_tokens,        \
              rot_dim, query_stride_token, query_stride_head,                  \
              key_stride_token, key_stride_head, value_stride_token,           \
              value_stride_head, cos_sin_stride_token, key_cache_stride_block, \
              key_cache_stride_token, key_cache_stride_head,                   \
              value_cache_stride_block, value_cache_stride_token,              \
              value_cache_stride_head, num_q_heads, num_kv_heads, head_size,   \
              block_size);                                                     \
        });                                                                    \
  } while (false)

// Manual-fusion operator with caller-owned Q storage. Writes rotated Q to the
// contiguous output, rotated K to the paged cache, and unmodified V to the
// cache without mutating inputs or materializing K.
void fused_rope_and_reshape_cache_flash_q_out(
    const torch::stable::Tensor& query, const torch::stable::Tensor& key,
    const torch::stable::Tensor& value, torch::stable::Tensor& query_out,
    const torch::stable::Tensor& positions,
    const torch::stable::Tensor& cos_sin_cache, bool is_neox,
    torch::stable::Tensor& key_cache, torch::stable::Tensor& value_cache,
    const torch::stable::Tensor& slot_mapping) {
  STD_TORCH_CHECK(query.dim() == 3 && key.dim() == 3 && value.dim() == 3 &&
                      query_out.dim() == 3,
                  "query, key, value, and query_out must be 3D tensors");
  STD_TORCH_CHECK(positions.dim() == 1 && slot_mapping.dim() == 1,
                  "positions and slot_mapping must be 1D tensors");
  STD_TORCH_CHECK(cos_sin_cache.dim() == 2,
                  "cos_sin_cache must be a 2D tensor");

  const int64_t num_rope_tokens = query.size(0);
  const int64_t num_cache_tokens = slot_mapping.size(0);
  const int num_q_heads = query.size(1);
  const int num_kv_heads = key.size(1);
  const int head_size = query.size(2);
  const int rot_dim = cos_sin_cache.size(1);
  STD_TORCH_CHECK(num_q_heads > 0 && num_kv_heads > 0);
  STD_TORCH_CHECK(num_cache_tokens <= num_rope_tokens);
  STD_TORCH_CHECK(key.size(0) == num_rope_tokens);
  STD_TORCH_CHECK(value.size(0) == num_rope_tokens);
  STD_TORCH_CHECK(key.size(2) == head_size);
  STD_TORCH_CHECK(value.size(2) == head_size);
  STD_TORCH_CHECK(value.size(1) == num_kv_heads);
  STD_TORCH_CHECK(query_out.size(0) == num_rope_tokens);
  STD_TORCH_CHECK(query_out.size(1) == num_q_heads);
  STD_TORCH_CHECK(query_out.size(2) == head_size);
  STD_TORCH_CHECK(key.scalar_type() == query.scalar_type());
  STD_TORCH_CHECK(value.scalar_type() == query.scalar_type());
  STD_TORCH_CHECK(query_out.scalar_type() == query.scalar_type());
  STD_TORCH_CHECK(cos_sin_cache.scalar_type() == query.scalar_type(),
                  "cos_sin_cache dtype must match query dtype");
  STD_TORCH_CHECK(query_out.is_contiguous());
  STD_TORCH_CHECK(rot_dim > 0 && rot_dim <= head_size && rot_dim % 2 == 0);

  const auto device = query.device();
  STD_TORCH_CHECK(
      key.device() == device && value.device() == device &&
          query_out.device() == device && positions.device() == device &&
          cos_sin_cache.device() == device && key_cache.device() == device &&
          value_cache.device() == device && slot_mapping.device() == device,
      "all inputs and outputs must be on the same device");

  STD_TORCH_CHECK(query.stride(2) == 1);
  STD_TORCH_CHECK(key.stride(2) == 1);
  STD_TORCH_CHECK(value.stride(2) == 1);
  STD_TORCH_CHECK(query.stride(0) > 0 && query.stride(1) > 0);
  STD_TORCH_CHECK(key.stride(0) > 0 && key.stride(1) > 0);
  STD_TORCH_CHECK(value.stride(0) > 0 && value.stride(1) > 0);

  STD_TORCH_CHECK(positions.scalar_type() ==
                  torch::headeronly::ScalarType::Long);
  STD_TORCH_CHECK(positions.size(0) == num_rope_tokens);
  STD_TORCH_CHECK(positions.stride(0) == 1);
  STD_TORCH_CHECK(slot_mapping.scalar_type() ==
                  torch::headeronly::ScalarType::Long);
  STD_TORCH_CHECK(slot_mapping.stride(0) == 1);
  STD_TORCH_CHECK(cos_sin_cache.stride(0) > 0 && cos_sin_cache.stride(1) == 1);

  STD_TORCH_CHECK(key_cache.dim() == 4);
  STD_TORCH_CHECK(value_cache.dim() == 4);
  STD_TORCH_CHECK(key_cache.size(3) == head_size);
  STD_TORCH_CHECK(key_cache.size(2) == num_kv_heads);
  STD_TORCH_CHECK(value_cache.size(3) == head_size);
  STD_TORCH_CHECK(value_cache.size(2) == num_kv_heads);
  STD_TORCH_CHECK(key_cache.size(0) == value_cache.size(0));
  STD_TORCH_CHECK(key_cache.size(1) == value_cache.size(1));
  STD_TORCH_CHECK(key_cache.scalar_type() == value_cache.scalar_type());
  STD_TORCH_CHECK(key_cache.stride(3) == 1);
  STD_TORCH_CHECK(value_cache.stride(3) == 1);
  STD_TORCH_CHECK(key_cache.size(0) > 0 && key_cache.size(1) > 0);
  STD_TORCH_CHECK(key_cache.stride(0) > 0 && key_cache.stride(1) > 0 &&
                  key_cache.stride(2) > 0);
  STD_TORCH_CHECK(value_cache.stride(0) > 0 && value_cache.stride(1) > 0 &&
                  value_cache.stride(2) > 0);

  STD_TORCH_CHECK(key_cache.scalar_type() == query.scalar_type(),
                  "cache dtype must match query dtype");

  if (num_rope_tokens == 0) {
    return;
  }

  const int block_size = key_cache.size(1);
  const int64_t query_stride_token = query.stride(0);
  const int64_t query_stride_head = query.stride(1);
  const int64_t key_stride_token = key.stride(0);
  const int64_t key_stride_head = key.stride(1);
  const int64_t value_stride_token = value.stride(0);
  const int64_t value_stride_head = value.stride(1);
  const int64_t cos_sin_stride_token = cos_sin_cache.stride(0);
  const int64_t key_cache_stride_block = key_cache.stride(0);
  const int64_t key_cache_stride_token = key_cache.stride(1);
  const int64_t key_cache_stride_head = key_cache.stride(2);
  const int64_t value_cache_stride_block = value_cache.stride(0);
  const int64_t value_cache_stride_token = value_cache.stride(1);
  const int64_t value_cache_stride_head = value_cache.stride(2);

  const int embed_dim = rot_dim / 2;
  const int query_work =
      std::max(num_q_heads * embed_dim, num_q_heads * (head_size - rot_dim));
  const int rope_work = std::max(query_work, num_kv_heads * embed_dim);
  const int cache_work = num_kv_heads * head_size;
  const int thread_block_size = std::min(std::max(rope_work, cache_work), 512);
  dim3 grid(num_rope_tokens, 1, 1);
  dim3 block(thread_block_size, 1, 1);

  const int32_t device_index = query.get_device_index();
  const torch::stable::accelerator::DeviceGuard device_guard(device_index);
  const cudaStream_t stream = get_current_cuda_stream(device_index);
  VLLM_STABLE_DISPATCH_BOOL(is_neox, neox, [&] {
    VLLM_STABLE_DISPATCH_BOOL(
        cache_work > thread_block_size, vectorize_value, [&] {
          CALL_FUSED_ROPE_AND_RESHAPE_CACHE_FLASH_Q_OUT(neox, vectorize_value);
        });
  });
}

#undef CALL_FUSED_ROPE_AND_RESHAPE_CACHE_FLASH_Q_OUT
