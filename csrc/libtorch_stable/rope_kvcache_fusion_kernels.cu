#include "torch_utils.h"
#include "dispatch_utils.h"

#include <algorithm>

#include "../cuda_compat.h"

namespace vllm {

// Functional Q-output variant for the model-visible manual fusion path.
//
// One thread block owns one token. Threads rotate Q into a contiguous output,
// rotate K directly into the paged cache, and copy V into the cache. K is never
// materialized outside the cache because ordinary non-DCP decoder attention
// consumes it from there. Keeping all work for a decode token in one CTA avoids
// the scheduling overhead of separate per-head tasks at small decode shapes.
template <typename qk_t, bool IS_NEOX>
__global__ void fused_rope_and_reshape_cache_flash_q_out_kernel(
    const int64_t* __restrict__ positions,  // [num_padded_tokens]
    const qk_t* __restrict__ query,         // [num_padded_tokens, num_q_heads,
                                            // head_size]
    const qk_t* __restrict__ key,           // [num_padded_tokens, num_kv_heads,
                                            // head_size]
    const qk_t* __restrict__ value,         // [num_padded_tokens, num_kv_heads,
                                            // head_size]
    qk_t* __restrict__ query_out,  // contiguous, same logical shape as query
    const qk_t* __restrict__ cos_sin_cache,  // [max_position, rot_dim]
    qk_t* __restrict__ key_cache,    // [num_blocks, block_size, num_kv_heads,
                                     // head_size]
    qk_t* __restrict__ value_cache,  // same logical shape
    const int64_t* __restrict__ slot_mapping,  // [num_cache_tokens]
    const int64_t num_rope_tokens, const int64_t num_cache_tokens,
    const int rot_dim, const int64_t query_stride_token,
    const int64_t query_stride_head, const int64_t key_stride_token,
    const int64_t key_stride_head, const int64_t value_stride_token,
    const int64_t value_stride_head, const int64_t cos_sin_stride_token,
    const int64_t key_cache_stride_block, const int64_t key_cache_stride_token,
    const int64_t key_cache_stride_head, const int64_t value_cache_stride_block,
    const int64_t value_cache_stride_token,
    const int64_t value_cache_stride_head, const int num_q_heads,
    const int num_kv_heads, const int head_size, const int block_size) {
  const int64_t token_idx = blockIdx.x;
  if (token_idx >= num_rope_tokens) {
    return;
  }
  const int64_t pos = positions[token_idx];
  const qk_t* cos_sin_ptr = cos_sin_cache + pos * cos_sin_stride_token;
  const int embed_dim = rot_dim / 2;

  const int nq = num_q_heads * embed_dim;
  for (int i = threadIdx.x; i < nq; i += blockDim.x) {
    const int head_idx = i / embed_dim;
    const int pair_idx = i % embed_dim;
    const float cos_f = static_cast<float>(VLLM_LDG(cos_sin_ptr + pair_idx));
    const float sin_f =
        static_cast<float>(VLLM_LDG(cos_sin_ptr + pair_idx + embed_dim));
    const int idx_x = IS_NEOX ? pair_idx : pair_idx * 2;
    const int idx_y = IS_NEOX ? embed_dim + pair_idx : pair_idx * 2 + 1;
    const qk_t* q_src =
        query + token_idx * query_stride_token + head_idx * query_stride_head;
    qk_t* q_dst = query_out + (token_idx * num_q_heads + head_idx) * head_size;
    const float x_f = static_cast<float>(q_src[idx_x]);
    const float y_f = static_cast<float>(q_src[idx_y]);
    q_dst[idx_x] = static_cast<qk_t>(x_f * cos_f - y_f * sin_f);
    q_dst[idx_y] = static_cast<qk_t>(y_f * cos_f + x_f * sin_f);
  }

  const int q_pass_dim = head_size - rot_dim;
  const int nq_pass = num_q_heads * q_pass_dim;
  for (int i = threadIdx.x; i < nq_pass; i += blockDim.x) {
    const int head_idx = i / q_pass_dim;
    const int head_offset = rot_dim + i % q_pass_dim;
    const qk_t* q_src =
        query + token_idx * query_stride_token + head_idx * query_stride_head;
    qk_t* q_dst = query_out + (token_idx * num_q_heads + head_idx) * head_size;
    q_dst[head_offset] = q_src[head_offset];
  }

  bool write_cache = token_idx < num_cache_tokens;
  int64_t block_idx = 0;
  int64_t page_offset = 0;
  if (write_cache) {
    const int64_t slot_idx = slot_mapping[token_idx];
    write_cache = slot_idx >= 0;
    if (write_cache) {
      block_idx = slot_idx / block_size;
      page_offset = slot_idx % block_size;
    }
  }
  if (!write_cache) {
    return;
  }

  const int nk = num_kv_heads * embed_dim;
  for (int i = threadIdx.x; i < nk; i += blockDim.x) {
    const int head_idx = i / embed_dim;
    const int pair_idx = i % embed_dim;
    const float cos_f = static_cast<float>(VLLM_LDG(cos_sin_ptr + pair_idx));
    const float sin_f =
        static_cast<float>(VLLM_LDG(cos_sin_ptr + pair_idx + embed_dim));
    const int idx_x = IS_NEOX ? pair_idx : pair_idx * 2;
    const int idx_y = IS_NEOX ? embed_dim + pair_idx : pair_idx * 2 + 1;
    const qk_t* k_src =
        key + token_idx * key_stride_token + head_idx * key_stride_head;
    qk_t* k_dst = key_cache + block_idx * key_cache_stride_block +
                  page_offset * key_cache_stride_token +
                  head_idx * key_cache_stride_head;
    const float x_f = static_cast<float>(k_src[idx_x]);
    const float y_f = static_cast<float>(k_src[idx_y]);
    const qk_t x_out = static_cast<qk_t>(x_f * cos_f - y_f * sin_f);
    const qk_t y_out = static_cast<qk_t>(y_f * cos_f + x_f * sin_f);
    k_dst[idx_x] = x_out;
    k_dst[idx_y] = y_out;
  }
  const int k_pass_dim = head_size - rot_dim;
  const int nk_pass = num_kv_heads * k_pass_dim;
  for (int i = threadIdx.x; i < nk_pass; i += blockDim.x) {
    const int head_idx = i / k_pass_dim;
    const int head_offset = rot_dim + i % k_pass_dim;
    const qk_t* k_src =
        key + token_idx * key_stride_token + head_idx * key_stride_head;
    qk_t* k_dst = key_cache + block_idx * key_cache_stride_block +
                  page_offset * key_cache_stride_token +
                  head_idx * key_cache_stride_head;
    k_dst[head_offset] = k_src[head_offset];
  }

  const int nv = num_kv_heads * head_size;
  for (int i = threadIdx.x; i < nv; i += blockDim.x) {
    const int head_idx = i / head_size;
    const int head_offset = i % head_size;
    const qk_t* v_src =
        value + token_idx * value_stride_token + head_idx * value_stride_head;
    qk_t* v_dst = value_cache + block_idx * value_cache_stride_block +
                  page_offset * value_cache_stride_token +
                  head_idx * value_cache_stride_head;
    v_dst[head_offset] = v_src[head_offset];
  }
}

}  // namespace vllm

#define CALL_FUSED_ROPE_AND_RESHAPE_CACHE_FLASH_Q_OUT()                        \
  do {                                                                         \
    VLLM_STABLE_DISPATCH_HALF_TYPES(                                           \
        query.scalar_type(), "qk_scalar_type", [&] {                           \
          using qk_t = scalar_t;                                               \
          if (is_neox) {                                                       \
            vllm::fused_rope_and_reshape_cache_flash_q_out_kernel<qk_t, true>  \
                <<<grid, block, 0, stream>>>(                                  \
                    positions.const_data_ptr<int64_t>(),                       \
                    query.const_data_ptr<qk_t>(), key.const_data_ptr<qk_t>(),  \
                    value.const_data_ptr<qk_t>(),                              \
                    query_out.mutable_data_ptr<qk_t>(),                        \
                    cos_sin_cache.const_data_ptr<qk_t>(),                      \
                    key_cache.mutable_data_ptr<qk_t>(),                        \
                    value_cache.mutable_data_ptr<qk_t>(),                      \
                    slot_mapping.const_data_ptr<int64_t>(), num_rope_tokens,   \
                    num_cache_tokens, rot_dim, query_stride_token,             \
                    query_stride_head, key_stride_token, key_stride_head,      \
                    value_stride_token, value_stride_head,                     \
                    cos_sin_stride_token, key_cache_stride_block,              \
                    key_cache_stride_token, key_cache_stride_head,             \
                    value_cache_stride_block, value_cache_stride_token,        \
                    value_cache_stride_head, num_q_heads, num_kv_heads,        \
                    head_size, block_size);                                    \
          } else {                                                             \
            vllm::fused_rope_and_reshape_cache_flash_q_out_kernel<qk_t, false> \
                <<<grid, block, 0, stream>>>(                                  \
                    positions.const_data_ptr<int64_t>(),                       \
                    query.const_data_ptr<qk_t>(), key.const_data_ptr<qk_t>(),  \
                    value.const_data_ptr<qk_t>(),                              \
                    query_out.mutable_data_ptr<qk_t>(),                        \
                    cos_sin_cache.const_data_ptr<qk_t>(),                      \
                    key_cache.mutable_data_ptr<qk_t>(),                        \
                    value_cache.mutable_data_ptr<qk_t>(),                      \
                    slot_mapping.const_data_ptr<int64_t>(), num_rope_tokens,   \
                    num_cache_tokens, rot_dim, query_stride_token,             \
                    query_stride_head, key_stride_token, key_stride_head,      \
                    value_stride_token, value_stride_head,                     \
                    cos_sin_stride_token, key_cache_stride_block,              \
                    key_cache_stride_token, key_cache_stride_head,             \
                    value_cache_stride_block, value_cache_stride_token,        \
                    value_cache_stride_head, num_q_heads, num_kv_heads,        \
                    head_size, block_size);                                    \
          }                                                                    \
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
  CALL_FUSED_ROPE_AND_RESHAPE_CACHE_FLASH_Q_OUT();
}

#undef CALL_FUSED_ROPE_AND_RESHAPE_CACHE_FLASH_Q_OUT
