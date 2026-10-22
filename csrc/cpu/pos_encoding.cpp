#include "cpu_types.hpp"

namespace {
template <typename scalar_t>
void rotary_embedding_impl(
    const int64_t* __restrict__ positions,  // [batch_size, seq_len] or
                                            // [num_tokens]
    scalar_t* __restrict__ query,           /// [batch_size, seq_len, num_heads,
                                   /// head_size] or [num_tokens, num_heads,
                                   /// head_size]
    scalar_t* __restrict__ key,  // nullptr (optional) or
                                 // [batch_size, seq_len, num_kv_heads,
                                 // head_size] or [num_tokens, num_kv_heads,
                                 // head_size]
    const scalar_t* __restrict__ cos_sin_cache,  // [max_position, 2, rot_dim //
                                                 // 2]
    const int rot_dim, const int64_t query_stride, const int64_t key_stride,
    const int num_heads, const int num_kv_heads, const int head_size,
    const int num_tokens) {
  using scalar_vec_t = vec_op::vec_t<scalar_t>;
  constexpr int VEC_ELEM_NUM = scalar_vec_t::get_elem_num();

  const int embed_dim = rot_dim / 2;
  bool flag = (embed_dim % VEC_ELEM_NUM == 0);
  const int loop_upper = flag ? embed_dim : embed_dim - VEC_ELEM_NUM;

  auto compute_loop = [&](const int64_t token_head, const scalar_t* cache_ptr,
                          scalar_t* qk) {
    int j = 0;
    for (; j < loop_upper; j += VEC_ELEM_NUM) {
      const int rot_offset = j;
      const int x_index = rot_offset;
      const int y_index = embed_dim + rot_offset;

      const int64_t out_x = token_head + x_index;
      const int64_t out_y = token_head + y_index;

      const scalar_vec_t cos(cache_ptr + x_index);
      const scalar_vec_t sin(cache_ptr + y_index);

      const scalar_vec_t q_x(qk + out_x);
      const scalar_vec_t q_y(qk + out_y);

      vec_op::FP32Vec8 fp32_cos(cos);
      vec_op::FP32Vec8 fp32_sin(sin);

      vec_op::FP32Vec8 fp32_q_x(q_x);
      vec_op::FP32Vec8 fp32_q_y(q_y);

      auto out1 = fp32_q_x * fp32_cos - fp32_q_y * fp32_sin;
      scalar_vec_t(out1).save(qk + out_x);

      auto out2 = fp32_q_y * fp32_cos + fp32_q_x * fp32_sin;
      scalar_vec_t(out2).save(qk + out_y);
    }
    if (!flag) {
      for (; j < embed_dim; ++j) {
        const int x_index = j;
        const int y_index = embed_dim + j;

        const int64_t out_x = token_head + x_index;
        const int64_t out_y = token_head + y_index;

        const float fp32_cos = cache_ptr[x_index];
        const float fp32_sin = cache_ptr[y_index];

        const float fp32_q_x = qk[out_x];
        const float fp32_q_y = qk[out_y];

        qk[out_x] = fp32_q_x * fp32_cos - fp32_q_y * fp32_sin;
        qk[out_y] = fp32_q_y * fp32_cos + fp32_q_x * fp32_sin;
      }
    }
  };

#pragma omp parallel for
  for (int token_idx = 0; token_idx < num_tokens; ++token_idx) {
    int64_t pos = positions[token_idx];
    const scalar_t* cache_ptr = cos_sin_cache + pos * rot_dim;

    for (int i = 0; i < num_heads; ++i) {
      const int head_idx = i;
      const int64_t token_head =
          token_idx * query_stride + head_idx * head_size;
      compute_loop(token_head, cache_ptr, query);
    }

    if (key != nullptr) {
      for (int i = 0; i < num_kv_heads; ++i) {
        const int head_idx = i;
        const int64_t token_head =
            token_idx * key_stride + head_idx * head_size;
        compute_loop(token_head, cache_ptr, key);
      }
    }
  }
}

template <>
void rotary_embedding_impl<c10::Half>(
    const int64_t* __restrict__ positions, c10::Half* __restrict__ query,
    c10::Half* __restrict__ key, const c10::Half* __restrict__ cos_sin_cache,
    const int rot_dim, const int64_t query_stride, const int64_t key_stride,
    const int num_heads, const int num_kv_heads, const int head_size,
    const int num_tokens) {
  using scalar_vec_t = vec_op::FP16Vec8;
  constexpr int VEC_ELEM_NUM = scalar_vec_t::get_elem_num();

  const int embed_dim = rot_dim / 2;
  bool flag = (embed_dim % VEC_ELEM_NUM == 0);
  const int loop_upper = flag ? embed_dim : embed_dim - VEC_ELEM_NUM;

  auto compute_loop = [&](const int64_t token_head, const c10::Half* cache_ptr,
                          c10::Half* qk) {
    int j = 0;
    for (; j < loop_upper; j += VEC_ELEM_NUM) {
      const int rot_offset = j;
      const int x_index = rot_offset;
      const int y_index = embed_dim + rot_offset;

      const int64_t out_x = token_head + x_index;
      const int64_t out_y = token_head + y_index;

      const vec_op::FP16Vec8 cos_fp16(cache_ptr + x_index);
      const vec_op::FP16Vec8 sin_fp16(cache_ptr + y_index);
      const vec_op::FP16Vec8 q_x_fp16(qk + out_x);
      const vec_op::FP16Vec8 q_y_fp16(qk + out_y);

      const vec_op::FP32Vec8 fp32_cos(cos_fp16);
      const vec_op::FP32Vec8 fp32_sin(sin_fp16);
      const vec_op::FP32Vec8 fp32_q_x(q_x_fp16);
      const vec_op::FP32Vec8 fp32_q_y(q_y_fp16);

      auto out1 = fp32_q_x * fp32_cos - fp32_q_y * fp32_sin;
      auto out2 = fp32_q_y * fp32_cos + fp32_q_x * fp32_sin;

      vec_op::FP16Vec8(out1).save(qk + out_x);
      vec_op::FP16Vec8(out2).save(qk + out_y);
    }
    if (!flag) {
      for (; j < embed_dim; ++j) {
        const int x_index = j;
        const int y_index = embed_dim + j;

        const int64_t out_x = token_head + x_index;
        const int64_t out_y = token_head + y_index;

        const float fp32_cos = static_cast<float>(cache_ptr[x_index]);
        const float fp32_sin = static_cast<float>(cache_ptr[y_index]);
        const float fp32_q_x = static_cast<float>(qk[out_x]);
        const float fp32_q_y = static_cast<float>(qk[out_y]);

        qk[out_x] =
            static_cast<c10::Half>(fp32_q_x * fp32_cos - fp32_q_y * fp32_sin);
        qk[out_y] =
            static_cast<c10::Half>(fp32_q_y * fp32_cos + fp32_q_x * fp32_sin);
      }
    }
  };

#pragma omp parallel for
  for (int token_idx = 0; token_idx < num_tokens; ++token_idx) {
    int64_t pos = positions[token_idx];
    const c10::Half* cache_ptr = cos_sin_cache + pos * rot_dim;

    for (int i = 0; i < num_heads; ++i) {
      const int head_idx = i;
      const int64_t token_head =
          token_idx * query_stride + head_idx * head_size;
      compute_loop(token_head, cache_ptr, query);
    }

    if (key != nullptr) {
      for (int i = 0; i < num_kv_heads; ++i) {
        const int head_idx = i;
        const int64_t token_head =
            token_idx * key_stride + head_idx * head_size;
        compute_loop(token_head, cache_ptr, key);
      }
    }
  }
}

template <typename scalar_t>
void rotary_embedding_gptj_impl(
    const int64_t* __restrict__ positions,  // [batch_size, seq_len] or
                                            // [num_tokens]
    scalar_t* __restrict__ query,           /// [batch_size, seq_len, num_heads,
                                   /// head_size] or [num_tokens, num_heads,
                                   /// head_size]
    scalar_t* __restrict__ key,  // nullptr (optional) or
                                 // [batch_size, seq_len, num_kv_heads,
                                 // head_size] or [num_tokens, num_kv_heads,
                                 // head_size]
    const scalar_t* __restrict__ cos_sin_cache,  // [max_position, 2, rot_dim //
                                                 // 2]
    const int rot_dim, const int64_t query_stride, const int64_t key_stride,
    const int num_heads, const int num_kv_heads, const int head_size,
    const int num_tokens) {
  const int embed_dim = rot_dim / 2;

#pragma omp parallel for collapse(2)
  for (int token_idx = 0; token_idx < num_tokens; ++token_idx) {
    for (int i = 0; i < num_heads; ++i) {
      int64_t pos = positions[token_idx];
      const scalar_t* cache_ptr = cos_sin_cache + pos * rot_dim;
      const scalar_t* cos_cache_ptr = cache_ptr;
      const scalar_t* sin_cache_ptr = cache_ptr + embed_dim;
      const int head_idx = i;
      const int64_t token_head =
          token_idx * query_stride + head_idx * head_size;
      scalar_t* head_query = token_head + query;
      for (int j = 0; j < embed_dim; j += 1) {
        const int rot_offset = j;
        const int x_index = 2 * rot_offset;
        const int y_index = 2 * rot_offset + 1;

        const float cos = cos_cache_ptr[rot_offset];
        const float sin = sin_cache_ptr[rot_offset];

        const float x = head_query[x_index];
        const float y = head_query[y_index];

        head_query[x_index] = x * cos - y * sin;
        head_query[y_index] = y * cos + x * sin;
      }
    }
  }

  if (key == nullptr) {
    return;
  }

#pragma omp parallel for collapse(2)
  for (int token_idx = 0; token_idx < num_tokens; ++token_idx) {
    for (int i = 0; i < num_kv_heads; ++i) {
      int64_t pos = positions[token_idx];
      const scalar_t* cache_ptr = cos_sin_cache + pos * rot_dim;
      const scalar_t* cos_cache_ptr = cache_ptr;
      const scalar_t* sin_cache_ptr = cache_ptr + embed_dim;
      const int head_idx = i;
      const int64_t token_head = token_idx * key_stride + head_idx * head_size;
      scalar_t* head_key = key + token_head;
      for (int j = 0; j < embed_dim; j += 1) {
        const int rot_offset = j;
        const int x_index = 2 * rot_offset;
        const int y_index = 2 * rot_offset + 1;

        const float cos = cos_cache_ptr[rot_offset];
        const float sin = sin_cache_ptr[rot_offset];

        const float x = head_key[x_index];
        const float y = head_key[y_index];

        head_key[x_index] = x * cos - y * sin;
        head_key[y_index] = y * cos + x * sin;
      }
    }
  }
}

template <>
void rotary_embedding_gptj_impl<c10::Half>(
    const int64_t* __restrict__ positions, c10::Half* __restrict__ query,
    c10::Half* __restrict__ key, const c10::Half* __restrict__ cos_sin_cache,
    const int rot_dim, const int64_t query_stride, const int64_t key_stride,
    const int num_heads, const int num_kv_heads, const int head_size,
    const int num_tokens) {
  const int embed_dim = rot_dim / 2;

#pragma omp parallel for collapse(2)
  for (int token_idx = 0; token_idx < num_tokens; ++token_idx) {
    for (int i = 0; i < num_heads; ++i) {
      int64_t pos = positions[token_idx];
      const c10::Half* cache_ptr = cos_sin_cache + pos * rot_dim;
      const c10::Half* cos_cache_ptr = cache_ptr;
      const c10::Half* sin_cache_ptr = cache_ptr + embed_dim;
      const int head_idx = i;
      const int64_t token_head =
          token_idx * query_stride + head_idx * head_size;
      c10::Half* head_query = token_head + query;
      for (int j = 0; j < embed_dim; j += 1) {
        const int rot_offset = j;
        const int x_index = 2 * rot_offset;
        const int y_index = 2 * rot_offset + 1;

        const float cos = static_cast<float>(cos_cache_ptr[rot_offset]);
        const float sin = static_cast<float>(sin_cache_ptr[rot_offset]);

        const float x = static_cast<float>(head_query[x_index]);
        const float y = static_cast<float>(head_query[y_index]);

        head_query[x_index] = static_cast<c10::Half>(x * cos - y * sin);
        head_query[y_index] = static_cast<c10::Half>(y * cos + x * sin);
      }
    }
  }

  if (key == nullptr) {
    return;
  }

#pragma omp parallel for collapse(2)
  for (int token_idx = 0; token_idx < num_tokens; ++token_idx) {
    for (int i = 0; i < num_kv_heads; ++i) {
      int64_t pos = positions[token_idx];
      const c10::Half* cache_ptr = cos_sin_cache + pos * rot_dim;
      const c10::Half* cos_cache_ptr = cache_ptr;
      const c10::Half* sin_cache_ptr = cache_ptr + embed_dim;
      const int head_idx = i;
      const int64_t token_head = token_idx * key_stride + head_idx * head_size;
      c10::Half* head_key = key + token_head;
      for (int j = 0; j < embed_dim; j += 1) {
        const int rot_offset = j;
        const int x_index = 2 * rot_offset;
        const int y_index = 2 * rot_offset + 1;

        const float cos = static_cast<float>(cos_cache_ptr[rot_offset]);
        const float sin = static_cast<float>(sin_cache_ptr[rot_offset]);

        const float x = static_cast<float>(head_key[x_index]);
        const float y = static_cast<float>(head_key[y_index]);

        head_key[x_index] = static_cast<c10::Half>(x * cos - y * sin);
        head_key[y_index] = static_cast<c10::Half>(y * cos + x * sin);
      }
    }
  }
}
};  // namespace

void rotary_embedding(torch::Tensor& positions, torch::Tensor& query,
                      std::optional<torch::Tensor> key, int64_t head_size,
                      torch::Tensor& cos_sin_cache, bool is_neox,
                      int64_t rope_dim_offset, bool inverse) {
  TORCH_CHECK(rope_dim_offset == 0,
              "rope_dim_offset != 0 is not supported on CPU");
  TORCH_CHECK(!inverse, "inverse rotary embedding is not supported on CPU");

  int num_tokens = positions.numel();
  int rot_dim = cos_sin_cache.size(1);
  int num_heads = query.size(-1) / head_size;
  int num_kv_heads = key.has_value() ? key->size(-1) / head_size : num_heads;
  int64_t key_stride = key.has_value() ? key->stride(-2) : 0;
  int64_t query_stride = query.stride(-2);

  VLLM_DISPATCH_FLOATING_TYPES(
      query.scalar_type(), "rotary_embedding_impl", [&] {
        CPU_KERNEL_GUARD_IN(rotary_embedding_impl)
        if (is_neox) {
          rotary_embedding_impl(
              positions.data_ptr<int64_t>(), query.data_ptr<scalar_t>(),
              key.has_value() ? key->data_ptr<scalar_t>() : nullptr,
              cos_sin_cache.data_ptr<scalar_t>(), rot_dim, query_stride,
              key_stride, num_heads, num_kv_heads, head_size, num_tokens);
        } else {
          rotary_embedding_gptj_impl(
              positions.data_ptr<int64_t>(), query.data_ptr<scalar_t>(),
              key.has_value() ? key->data_ptr<scalar_t>() : nullptr,
              cos_sin_cache.data_ptr<scalar_t>(), rot_dim, query_stride,
              key_stride, num_heads, num_kv_heads, head_size, num_tokens);
        }

        CPU_KERNEL_GUARD_OUT(rotary_embedding_impl)
      });
}
