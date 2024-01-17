
#include "cpu_types.hpp"

namespace {
template <typename scalar_t>
void rotary_embedding_impl(
    const int64_t
        *__restrict__ positions, // [batch_size, seq_len] or [num_tokens]
    scalar_t
        *__restrict__ query, /// [batch_size, seq_len, num_heads, head_size] or
                             /// [num_tokens, num_heads, head_size]
    scalar_t
        *__restrict__ key, // [batch_size, seq_len, num_kv_heads, head_size] or
                           // [num_tokens, num_kv_heads, head_size]
    const scalar_t
        *__restrict__ cos_sin_cache, // [max_position, 2, rot_dim // 2]
    const int rot_dim, const int64_t query_stride, const int64_t key_stride,
    const int num_heads, const int num_kv_heads, const int head_size,
    const int num_tokens) {
  using scalar_vec_t = vec_op::vec_t<scalar_t>;
  constexpr int VEC_ELEM_NUM = scalar_vec_t::get_elem_num();
  constexpr int ELEM_SIZE = sizeof(scalar_t);

  const int embed_dim = rot_dim / 2;
  TORCH_CHECK(embed_dim % VEC_ELEM_NUM == 0);

#pragma omp parallel for
  for (int token_idx = 0; token_idx < num_tokens; ++token_idx) {
    int64_t pos = positions[token_idx];
    const scalar_t *cache_ptr = cos_sin_cache + pos * rot_dim;

    for (int i = 0; i < num_heads; ++i) {
      const int head_idx = i;
      const int64_t token_head = token_idx * query_stride + head_idx * head_size;
      for (int j = 0; j < embed_dim; j += VEC_ELEM_NUM) {
        const int rot_offset = j;
        const int x_index = rot_offset;
        const int y_index = embed_dim + rot_offset;

        const int64_t out_x = token_head + x_index;
        const int64_t out_y = token_head + y_index;

        const scalar_vec_t cos(cache_ptr + x_index);
        const scalar_vec_t sin(cache_ptr + y_index);

        const scalar_vec_t q_x(query + out_x);
        const scalar_vec_t q_y(query + out_y);

        vec_op::FP32Vec8 fp32_cos(cos.reg);
        vec_op::FP32Vec8 fp32_sin(sin.reg);

        vec_op::FP32Vec8 fp32_q_x(q_x.reg);
        vec_op::FP32Vec8 fp32_q_y(q_y.reg);

        auto out1 = fp32_q_x * fp32_cos - fp32_q_y * fp32_sin;
        scalar_vec_t(out1.reg).save(query + out_x);

        auto out2 = fp32_q_y * fp32_cos + fp32_q_x * fp32_sin;
        scalar_vec_t(out2.reg).save(query + out_y);
      }
    }

    for (int i = 0; i < num_kv_heads; ++i) {
      const int head_idx = i;
      const int64_t token_head = token_idx * key_stride + head_idx * head_size;
      for (int j = 0; j < embed_dim; j += VEC_ELEM_NUM) {
        const int rot_offset = j;
        const int x_index = rot_offset;
        const int y_index = embed_dim + rot_offset;

        const int64_t out_x = token_head + x_index;
        const int64_t out_y = token_head + y_index;

        const scalar_vec_t cos(cache_ptr + x_index);
        const scalar_vec_t sin(cache_ptr + y_index);

        const scalar_vec_t k_x(key + out_x);
        const scalar_vec_t k_y(key + out_y);

        vec_op::FP32Vec8 fp32_cos(cos.reg);
        vec_op::FP32Vec8 fp32_sin(sin.reg);

        vec_op::FP32Vec8 fp32_k_x(k_x.reg);
        vec_op::FP32Vec8 fp32_k_y(k_y.reg);

        auto out1 = fp32_k_x * fp32_cos - fp32_k_y * fp32_sin;
        scalar_vec_t(out1.reg).save(key + out_x);
        auto out2 = fp32_k_y * fp32_cos + fp32_k_x * fp32_sin;
        scalar_vec_t(out2.reg).save(key + out_y);
      }
    }
  }
}
}; // namespace

void rotary_embedding_cpu(torch::Tensor &positions, torch::Tensor &query,
                          torch::Tensor &key, int head_size,
                          torch::Tensor &cos_sin_cache, bool is_neox) {
  TORCH_CHECK(is_neox);
  int num_tokens = query.numel() / query.size(-1);
  int rot_dim = cos_sin_cache.size(1);
  int num_heads = query.size(-1) / head_size;
  int num_kv_heads = key.size(-1) / head_size;
  int64_t key_stride = key.stride(-2);
  int64_t query_stride = query.stride(-2);

  VLLM_DISPATCH_FLOATING_TYPES(
      query.scalar_type(), "rotary_embedding_impl", [&] {
        CPU_KERNEL_GUARD_IN(rotary_embedding_impl)
        rotary_embedding_impl(
            positions.data_ptr<int64_t>(), query.data_ptr<scalar_t>(),
            key.data_ptr<scalar_t>(), cos_sin_cache.data_ptr<scalar_t>(),
            rot_dim, query_stride, key_stride, num_heads, num_kv_heads,
            head_size, num_tokens);
        CPU_KERNEL_GUARD_OUT(rotary_embedding_impl)
      });
}