#include "cpu_types.hpp"

namespace {

template <typename scalar_t, int HEAD_SIZE, int BLOCK_SIZE>
struct paged_attention_v1_impl {
  static void
  call(scalar_t *__restrict__ out,           // [num_seqs, num_heads, head_size]
       const scalar_t *__restrict__ q,       // [num_seqs, num_heads, head_size]
       const scalar_t *__restrict__ k_cache, // [num_blocks, num_kv_heads,
                                             // head_size/x, block_size, x]
       const scalar_t *__restrict__ v_cache, // [num_blocks, num_kv_heads,
                                             // head_size, block_size]
       const int num_kv_heads, const float scale,
       const int
           *__restrict__ block_tables, // [num_seqs, max_num_blocks_per_seq]
       const int *__restrict__ context_lens, // [num_seqs]
       const int max_num_blocks_per_seq,
       const float *__restrict__ alibi_slopes, // [num_heads]
       const int q_stride, const int kv_block_stride, const int kv_head_stride,
       const int num_seqs, const int num_heads) {
    TORCH_CHECK(HEAD_SIZE % 16 == 0);
    TORCH_CHECK(alibi_slopes == nullptr, "Unsupport alibi_slopes for CPU");
    constexpr int x = 16 / sizeof(scalar_t);
    const int num_queries_per_kv = num_heads / num_kv_heads;

    int max_context_len = max_num_blocks_per_seq * BLOCK_SIZE;
    int max_context_len_padded = (max_context_len + 15) & 0xFFFFFFF0;
    TORCH_CHECK((max_context_len_padded * sizeof(float)) % 64 == 0);

    size_t logits_bytes = num_heads * max_context_len_padded * sizeof(float);
    float *logits = (float *)std::aligned_alloc(
        64, logits_bytes); // Cacheline alignment for each context token.
                           // [head_num, max_context_len_padded]

    std::memset(out, 0, num_seqs * num_heads * HEAD_SIZE * sizeof(scalar_t));

    for (int seq_idx = 0; seq_idx < num_seqs; ++seq_idx) {
      int context_len = context_lens[seq_idx];
      const int *seq_block_table =
          block_tables + max_num_blocks_per_seq * seq_idx;
      const int block_num = (context_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
      std::memset(logits, 0, logits_bytes);

      // Compute attention logits
#pragma omp parallel for collapse(2)
      for (int block_idx = 0; block_idx < block_num; ++block_idx) {
        for (int head_idx = 0; head_idx < num_heads; ++head_idx) {
          const int64_t kv_head_idx = head_idx / num_queries_per_kv;
          const int64_t physical_block_idx = seq_block_table[block_idx];
          const scalar_t *__restrict__ q_vec_ptr =
              q + seq_idx * q_stride + head_idx * HEAD_SIZE;
          const scalar_t *__restrict__ k_block_cache_ptr =
              k_cache + physical_block_idx * kv_block_stride +
              kv_head_idx * kv_head_stride;
          float *__restrict__ head_block_logits =
              logits + head_idx * max_context_len_padded +
              block_idx * BLOCK_SIZE;

          for (int q_offset = 0; q_offset < HEAD_SIZE;
               q_offset += x, q_vec_ptr += x) {
            for (int token_idx = 0; token_idx < BLOCK_SIZE;
                 ++token_idx, k_block_cache_ptr += x) {
              for (int i = 0; i < x; ++i) {
                head_block_logits[token_idx] +=
                    q_vec_ptr[i] * k_block_cache_ptr[i] * scale;
              }
            }
          }
        }
      }

      // Compute softmax
#pragma omp parallel for
      for (int head_idx = 0; head_idx < num_heads; ++head_idx) {
        float *head_logit_ptr = logits + head_idx * max_context_len_padded;
        float max_logit = head_logit_ptr[0];
        for (int i = 1; i < context_len; ++i) {
          max_logit =
              max_logit >= head_logit_ptr[i] ? max_logit : head_logit_ptr[i];
        }

        float sum = 0;
        for (int i = 0; i < context_len; ++i) {
          head_logit_ptr[i] = std::exp(head_logit_ptr[i] - max_logit);
          sum += head_logit_ptr[i];
        }

        for (int i = 0; i < context_len; ++i) {
          head_logit_ptr[i] /= sum;
        }

        int remaining_seq_upper = block_num * BLOCK_SIZE;
        for (int i = context_len; i < remaining_seq_upper; ++i) {
          head_logit_ptr[i] = 0;
        }
      }

      // Compute value
      constexpr int head_partition_num = HEAD_SIZE / 16;
#pragma omp parallel for collapse(2)
      for (int head_idx = 0; head_idx < num_heads; ++head_idx) {
        for (int head_part_idx = 0; head_part_idx < head_partition_num;
             ++head_part_idx) {
          for (int block_idx = 0; block_idx < block_num; ++block_idx) {
            const int64_t kv_head_idx = head_idx / num_queries_per_kv;
            const int64_t physical_block_idx = seq_block_table[block_idx];
            const float *__restrict__ prob_vec_ptr =
                logits + head_idx * max_context_len_padded +
                block_idx * BLOCK_SIZE;
            const scalar_t *__restrict__ v_block_cache_ptr =
                v_cache + physical_block_idx * kv_block_stride +
                kv_head_idx * kv_head_stride + BLOCK_SIZE * head_part_idx * 16;
            scalar_t *__restrict__ out_ptr =
                out + seq_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE +
                head_part_idx * 16;

            for (int i = 0; i < 16; ++i, v_block_cache_ptr += BLOCK_SIZE) {
              for (int j = 0; j < BLOCK_SIZE; ++j) {
                out_ptr[i] += prob_vec_ptr[j] * v_block_cache_ptr[j];
              }
            }
          }
        }
      }
    }
    std::free(logits);
  }
};

template <int HEAD_SIZE, int BLOCK_SIZE>
struct paged_attention_v1_impl<c10::BFloat16, HEAD_SIZE, BLOCK_SIZE> {
  using scalar_t = c10::BFloat16;

  static void
  call(scalar_t *__restrict__ out,           // [num_seqs, num_heads, head_size]
       const scalar_t *__restrict__ q,       // [num_seqs, num_heads, head_size]
       const scalar_t *__restrict__ k_cache, // [num_blocks, num_kv_heads,
                                             // head_size/x, block_size, x]
       const scalar_t *__restrict__ v_cache, // [num_blocks, num_kv_heads,
                                             // head_size, block_size]
       const int num_kv_heads, const float scale,
       const int
           *__restrict__ block_tables, // [num_seqs, max_num_blocks_per_seq]
       const int *__restrict__ context_lens, // [num_seqs]
       const int max_num_blocks_per_seq,
       const float *__restrict__ alibi_slopes, // [num_heads]
       const int q_stride, const int kv_block_stride, const int kv_head_stride,
       const int num_seqs, const int num_heads) {
    TORCH_CHECK(alibi_slopes == nullptr, "Unsupport alibi_slopes for CPU");
    constexpr int x = 16 / sizeof(scalar_t);
    const int num_queries_per_kv = num_heads / num_kv_heads;

    using scalar_vec_t = vec_op::vec_t<scalar_t>;
    constexpr int VEC_ELEM_NUM = scalar_vec_t::get_elem_num();

    static_assert(x == VEC_ELEM_NUM);
    static_assert(BLOCK_SIZE == 16);
    static_assert(BLOCK_SIZE % VEC_ELEM_NUM == 0);

    int max_context_len = max_num_blocks_per_seq * BLOCK_SIZE;
    int max_context_len_padded = (max_context_len + 15) & 0xFFFFFFF0;
    TORCH_CHECK((max_context_len_padded * sizeof(float)) % 64 == 0);

    const int parallel_work_item_num = omp_get_max_threads();

    size_t logits_bytes =
        parallel_work_item_num * max_context_len_padded * sizeof(float);
    float *logits = (float *)std::aligned_alloc(
        64, logits_bytes); // Cacheline alignment for each context token.
                           // [parallel_work_item_num, max_context_len_padded]

#pragma omp parallel for schedule(dynamic) collapse(2)
    for (int seq_idx = 0; seq_idx < num_seqs; ++seq_idx) {
      for (int head_idx = 0; head_idx < num_heads; ++head_idx) {
        int context_len = context_lens[seq_idx];
        const int *seq_block_table =
            block_tables + max_num_blocks_per_seq * seq_idx;
        const int block_num = (context_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
        const int64_t kv_head_idx = head_idx / num_queries_per_kv;
        const scalar_t *__restrict__ q_vec_ptr =
            q + seq_idx * q_stride + head_idx * HEAD_SIZE;
        float *__restrict__ thread_block_logits =
            logits + omp_get_thread_num() * max_context_len_padded;

        // Compute logits
        for (int block_idx = 0; block_idx < block_num; ++block_idx) {
          const int64_t physical_block_idx = seq_block_table[block_idx];
          const scalar_t *__restrict__ k_block_cache_ptr =
              k_cache + physical_block_idx * kv_block_stride +
              kv_head_idx * kv_head_stride;
          float *__restrict__ head_block_logits =
              thread_block_logits + block_idx * BLOCK_SIZE;

          static_assert(vec_op::BF16Vec32::get_elem_num() % x == 0);
          constexpr int TOKEN_PER_GROUP = vec_op::BF16Vec32::get_elem_num() / x;
          static_assert(BLOCK_SIZE % TOKEN_PER_GROUP == 0);
          constexpr int TOKEN_GROUPS = BLOCK_SIZE / TOKEN_PER_GROUP;

          //   vec_op::FP32Vec8 accums[BLOCK_SIZE];
          vec_op::FP32Vec16 group_accums[TOKEN_GROUPS];

          for (int q_offset = 0; q_offset < HEAD_SIZE;
               q_offset += x, k_block_cache_ptr += x * BLOCK_SIZE) {
            scalar_vec_t q_vec(q_vec_ptr + q_offset);
            vec_op::BF16Vec32 q_group_vec(q_vec);

            vec_op::unroll_loop<int, TOKEN_GROUPS>(
                [k_block_cache_ptr, &q_group_vec,
                 &group_accums](int token_group_idx) {
                  vec_op::BF16Vec32 k_group_vec(k_block_cache_ptr +
                                                token_group_idx * x *
                                                    TOKEN_PER_GROUP);

                  group_accums[token_group_idx] = vec_op::fma(
                      q_group_vec, k_group_vec, group_accums[token_group_idx]);
                });
          }

          vec_op::unroll_loop<int, TOKEN_GROUPS>([&group_accums,
                                                  head_block_logits,
                                                  scale](int token_group_idx) {
            vec_op::unroll_loop<int, TOKEN_PER_GROUP>([&group_accums,
                                                       head_block_logits, scale,
                                                       token_group_idx](
                                                          int token_idx) {
              float dot_v =
                  group_accums[token_group_idx]
                      .template reduce_sub_sum<
                          vec_op::FP32Vec16::get_elem_num() / TOKEN_PER_GROUP>(
                          token_idx);
              head_block_logits[token_group_idx * TOKEN_PER_GROUP + token_idx] =
                  dot_v * scale;
            });
          });
        }

        // Compute softmax
        float max_logit = thread_block_logits[0];
        for (int i = 1; i < context_len; ++i) {
          max_logit = max_logit >= thread_block_logits[i]
                          ? max_logit
                          : thread_block_logits[i];
        }

        float sum = 0;
        for (int i = 0; i < context_len; ++i) {
          thread_block_logits[i] = std::exp(thread_block_logits[i] - max_logit);
          sum += thread_block_logits[i];
        }

        for (int i = 0; i < context_len; ++i) {
          thread_block_logits[i] /= sum;
        }

        int remaining_seq_upper = block_num * BLOCK_SIZE;
        for (int i = context_len; i < remaining_seq_upper; ++i) {
          thread_block_logits[i] = 0;
        }

        // Compute value
        constexpr int head_elem_num_per_partition = 16;
        constexpr int head_partition_num =
            HEAD_SIZE / head_elem_num_per_partition;
        for (int head_part_idx = 0; head_part_idx < head_partition_num;
             ++head_part_idx) {
          vec_op::FP32Vec16 accums[head_elem_num_per_partition];
          scalar_t *__restrict__ out_ptr =
              out + seq_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE +
              head_part_idx * head_elem_num_per_partition;
          for (int block_idx = 0; block_idx < block_num; ++block_idx) {
            const int64_t physical_block_idx = seq_block_table[block_idx];
            const float *__restrict__ prob_vec_ptr =
                thread_block_logits + block_idx * BLOCK_SIZE;
            const scalar_t *__restrict__ v_block_cache_ptr =
                v_cache + physical_block_idx * kv_block_stride +
                kv_head_idx * kv_head_stride +
                BLOCK_SIZE * head_part_idx * head_elem_num_per_partition;

            vec_op::FP32Vec16 prob_vec(prob_vec_ptr);

            vec_op::unroll_loop<int, head_elem_num_per_partition>(
                [&](int head_elem_idx) {
                  vec_op::BF16Vec16 v_vec(v_block_cache_ptr +
                                          BLOCK_SIZE * head_elem_idx);
                  vec_op::FP32Vec16 fp32_v_vec(v_vec.reg);
                  accums[head_elem_idx] =
                      accums[head_elem_idx] + prob_vec * fp32_v_vec;
                });
          }

          vec_op::unroll_loop<int, head_elem_num_per_partition>(
              [&](int head_elem_idx) {
                float value = accums[head_elem_idx].reduce_sum();
                vec_op::storeFP32ToT(value, out_ptr + head_elem_idx);
              });
        }
      }
    }
    std::free(logits);
  }
};

#define LAUNCH_ATTENTION_KERNEL(T, HEAD_SIZE, BLOCK_SIZE)                      \
  paged_attention_v1_impl<T, HEAD_SIZE, BLOCK_SIZE>::call(                     \
      out_ptr, query_ptr, key_cache_ptr, value_cache_ptr, num_kv_heads, scale, \
      block_tables_ptr, context_lens_ptr, max_num_blocks_per_seq,              \
      alibi_slopes_ptr, q_stride, kv_block_stride, kv_head_stride, num_seqs,   \
      num_heads);

template <typename T, int BLOCK_SIZE>
void paged_attention_v1_impl_launcher(
    torch::Tensor &out, torch::Tensor &query, torch::Tensor &key_cache,
    torch::Tensor &value_cache, int num_kv_heads, float scale,
    torch::Tensor &block_tables, torch::Tensor &context_lens,
    int max_context_len, const c10::optional<torch::Tensor> &alibi_slopes) {
  int num_seqs = query.size(0);
  int num_heads = query.size(1);
  int head_size = query.size(2);
  int max_num_blocks_per_seq = block_tables.size(1);
  int q_stride = query.stride(0);
  int kv_block_stride = key_cache.stride(0);
  int kv_head_stride = key_cache.stride(1);

  // NOTE: alibi_slopes is optional.
  const float *alibi_slopes_ptr =
      alibi_slopes
          ? reinterpret_cast<const float *>(alibi_slopes.value().data_ptr())
          : nullptr;

  T *out_ptr = reinterpret_cast<T *>(out.data_ptr());
  T *query_ptr = reinterpret_cast<T *>(query.data_ptr());
  T *key_cache_ptr = reinterpret_cast<T *>(key_cache.data_ptr());
  T *value_cache_ptr = reinterpret_cast<T *>(value_cache.data_ptr());
  int *block_tables_ptr = block_tables.data_ptr<int>();
  int *context_lens_ptr = context_lens.data_ptr<int>();

  switch (head_size) {
  case 64:
    LAUNCH_ATTENTION_KERNEL(T, 64, BLOCK_SIZE);
    break;
  case 80:
    LAUNCH_ATTENTION_KERNEL(T, 80, BLOCK_SIZE);
    break;
  case 96:
    LAUNCH_ATTENTION_KERNEL(T, 96, BLOCK_SIZE);
    break;
  case 112:
    LAUNCH_ATTENTION_KERNEL(T, 112, BLOCK_SIZE);
    break;
  case 128:
    LAUNCH_ATTENTION_KERNEL(T, 128, BLOCK_SIZE);
    break;
  case 256:
    LAUNCH_ATTENTION_KERNEL(T, 256, BLOCK_SIZE);
    break;
  default:
    TORCH_CHECK(false, "Unsupported head size: ", head_size);
    break;
  }
}

#define CALL_KERNEL_LAUNCHER(T, BLOCK_SIZE)                                    \
  paged_attention_v1_impl_launcher<T, BLOCK_SIZE>(                             \
      out, query, key_cache, value_cache, num_kv_heads, scale, block_tables,   \
      context_lens, max_context_len, alibi_slopes);

#define CALL_KERNEL_LAUNCHER_BLOCK_SIZE(T)                                     \
  switch (block_size) {                                                        \
  case 16:                                                                     \
    CALL_KERNEL_LAUNCHER(T, 16);                                               \
    break;                                                                     \
  default:                                                                     \
    TORCH_CHECK(false, "Unsupported block size: ", block_size);                \
    break;                                                                     \
  }
} // namespace

void paged_attention_v1_cpu(torch::Tensor &out, torch::Tensor &query,
                            torch::Tensor &key_cache,
                            torch::Tensor &value_cache, int num_kv_heads,
                            float scale, torch::Tensor &block_tables,
                            torch::Tensor &context_lens, int block_size,
                            int max_context_len,
                            const c10::optional<torch::Tensor> &alibi_slopes) {
  VLLM_DISPATCH_FLOATING_TYPES(query.scalar_type(), "paged_attention_v1_impl",
                               [&] {
                                 CPU_KERNEL_GUARD_IN(paged_attention_v1_impl)
                                 CALL_KERNEL_LAUNCHER_BLOCK_SIZE(scalar_t);
                                 CPU_KERNEL_GUARD_OUT(paged_attention_v1_impl)
                               });
}

void paged_attention_v2_cpu(torch::Tensor &out, torch::Tensor &exp_sums,
                            torch::Tensor &max_logits, torch::Tensor &tmp_out,
                            torch::Tensor &query, torch::Tensor &key_cache,
                            torch::Tensor &value_cache, int num_kv_heads,
                            float scale, torch::Tensor &block_tables,
                            torch::Tensor &context_lens, int block_size,
                            int max_context_len,
                            const c10::optional<torch::Tensor> &alibi_slopes) {
  TORCH_CHECK(false, "paged_attention_v2 is unsupported on CPU.")
}