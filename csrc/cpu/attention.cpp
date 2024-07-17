#include "cpu_types.hpp"

namespace {

template <typename scalar_t>
struct KernelVecType {
  using q_load_vec_type = void;
  using q_vec_type = void;
  using k_load_vec_type = void;
  using k_vec_type = void;
  using qk_acc_vec_type = void;
  using v_load_vec_type = void;
};

template <>
struct KernelVecType<float> {
  using q_load_vec_type = vec_op::FP32Vec4;
  using q_vec_type = vec_op::FP32Vec16;
  using k_load_vec_type = vec_op::FP32Vec16;
  using k_vec_type = vec_op::FP32Vec16;
  using qk_acc_vec_type = vec_op::FP32Vec16;
  using v_load_vec_type = vec_op::FP32Vec16;
};

#ifdef __AVX512BF16__
template <>
struct KernelVecType<c10::BFloat16> {
  using q_load_vec_type = vec_op::BF16Vec8;
  using q_vec_type = vec_op::BF16Vec32;
  using k_load_vec_type = vec_op::BF16Vec32;
  using k_vec_type = vec_op::BF16Vec32;
  using qk_acc_vec_type = vec_op::FP32Vec16;
  using v_load_vec_type = vec_op::BF16Vec16;
};
#else
template <>
struct KernelVecType<c10::BFloat16> {
  using q_load_vec_type = vec_op::BF16Vec8;
  using q_vec_type = vec_op::FP32Vec16;
  using k_load_vec_type = vec_op::BF16Vec16;
  using k_vec_type = vec_op::FP32Vec16;
  using qk_acc_vec_type = vec_op::FP32Vec16;
  using v_load_vec_type = vec_op::BF16Vec16;
};
#endif

template <typename T>
FORCE_INLINE std::pair<T, T> reduceSoftmax(T* data, const int size,
                                           const int capacity) {
  T max = data[0];
  for (int i = 1; i < size; ++i) {
    max = max >= data[i] ? max : data[i];
  }

  T sum = 0;
  for (int i = 0; i < size; ++i) {
    data[i] = std::exp(data[i] - max);
    sum += data[i];
  }

  int i = 0;
  for (; i < size; ++i) {
    data[i] /= sum;
  }

  for (; i < capacity; ++i) {
    data[i] = 0;
  }

  return {max, sum};
}

template <typename T>
FORCE_INLINE std::pair<T, T> reduceSoftmaxAlibi(T* data, const int size,
                                                const int capacity,
                                                const float alibi_slope,
                                                const int start_index,
                                                const int seq_len) {
  data[0] += alibi_slope * (start_index - seq_len + 1);
  T max = data[0];
  for (int i = 1; i < size; ++i) {
    T qk = data[i] + alibi_slope * (start_index + i - seq_len + 1);
    data[i] = qk;
    max = max >= qk ? max : qk;
  }

  T sum = 0;
  for (int i = 0; i < size; ++i) {
    data[i] = std::exp(data[i] - max);
    sum += data[i];
  }

  int i = 0;
  for (; i < size; ++i) {
    data[i] /= sum;
  }

  for (; i < capacity; ++i) {
    data[i] = 0;
  }

  return {max, sum};
}

template <typename T>
FORCE_INLINE void reducePartitonSoftmax(const T* max_data, T* sum_data,
                                        const int size) {
  T max = max_data[0];
  for (int i = 1; i < size; ++i) {
    max = max >= max_data[i] ? max : max_data[i];
  }

  T rescaled_sum = 0;
  for (int i = 0; i < size; ++i) {
    T rescale_factor = std::exp(max_data[i] - max);
    rescaled_sum += rescale_factor * sum_data[i];
    sum_data[i] *= rescale_factor;
  }
  for (int i = 0; i < size; ++i) {
    sum_data[i] /= rescaled_sum + 1e-8;
  }
}

template <typename scalar_t, int HEAD_SIZE, int BLOCK_SIZE, int x>
struct reduceQKBlockKernel {
  using q_load_vec_type = typename KernelVecType<scalar_t>::q_load_vec_type;
  using q_vec_type = typename KernelVecType<scalar_t>::q_vec_type;
  using k_load_vec_type = typename KernelVecType<scalar_t>::k_load_vec_type;
  using k_vec_type = typename KernelVecType<scalar_t>::k_vec_type;
  using qk_acc_vec_type = typename KernelVecType<scalar_t>::qk_acc_vec_type;

  constexpr static int TOKEN_PER_GROUP = k_load_vec_type::get_elem_num() / x;
  constexpr static int MAX_GROUP_NUM = 16 / TOKEN_PER_GROUP;
  constexpr static int UNROLL_GROUP_NUM = MAX_GROUP_NUM / 4;

  static_assert(MAX_GROUP_NUM == 8 || MAX_GROUP_NUM == 4);
  static_assert(k_load_vec_type::get_elem_num() % x == 0);
  static_assert(q_load_vec_type::get_elem_num() * sizeof(scalar_t) == 16);

  FORCE_INLINE static void call(const scalar_t* __restrict__ q,
                                const scalar_t* __restrict__ k_block,
                                float* __restrict__ logits, float scale,
                                const int token_num) {
    const int group_num = (token_num + TOKEN_PER_GROUP - 1) / TOKEN_PER_GROUP;

    qk_acc_vec_type group_accums[MAX_GROUP_NUM];
    if (token_num == BLOCK_SIZE) {
      for (int q_offset = 0; q_offset < HEAD_SIZE;
           q_offset += x, k_block += x * BLOCK_SIZE) {
        q_load_vec_type q_load_group_vec(q + q_offset);
        q_vec_type q_group_vec(q_load_group_vec);

        vec_op::unroll_loop<int, MAX_GROUP_NUM>(
            [k_block, &q_group_vec, &group_accums](int token_group_idx) {
              k_load_vec_type k_load_group_vec(k_block + token_group_idx * x *
                                                             TOKEN_PER_GROUP);
              k_vec_type k_group_vec(k_load_group_vec);
              vec_op::fma(group_accums[token_group_idx], q_group_vec,
                          k_group_vec);
              vec_op::prefetch(k_block + x * BLOCK_SIZE +
                               token_group_idx * x * TOKEN_PER_GROUP);
            });
      }
    } else {
      for (int q_offset = 0; q_offset < HEAD_SIZE;
           q_offset += x, k_block += x * BLOCK_SIZE) {
        q_load_vec_type q_load_group_vec(q + q_offset);
        q_vec_type q_group_vec(q_load_group_vec);
        for (int token_group_start = 0; token_group_start < group_num;
             token_group_start += UNROLL_GROUP_NUM) {
          vec_op::unroll_loop<int, UNROLL_GROUP_NUM>(
              [token_group_start, k_block, &q_group_vec,
               &group_accums](int token_group_idx) {
                token_group_idx += token_group_start;
                k_load_vec_type k_load_group_vec(k_block + token_group_idx * x *
                                                               TOKEN_PER_GROUP);
                k_vec_type k_group_vec(k_load_group_vec);
                vec_op::fma(group_accums[token_group_idx], q_group_vec,
                            k_group_vec);
                vec_op::prefetch(k_block + x * BLOCK_SIZE +
                                 token_group_idx * x * TOKEN_PER_GROUP);
              });
        }
      }
    }

    for (int token_group_idx = 0; token_group_idx < group_num;
         ++token_group_idx) {
      vec_op::unroll_loop<int, TOKEN_PER_GROUP>(
          [&group_accums, logits, scale, token_group_idx](int token_idx) {
            float dot_v =
                group_accums[token_group_idx]
                    .template reduce_sub_sum<qk_acc_vec_type::get_elem_num() /
                                             TOKEN_PER_GROUP>(token_idx);
            logits[token_group_idx * TOKEN_PER_GROUP + token_idx] =
                dot_v * scale;
          });
    }
  }
};

template <typename scalar_t, int HEAD_SIZE, int BLOCK_SIZE,
          int HEAD_PARTITION_SIZE, typename acc_t>
FORCE_INLINE void reduceValueBlock(const float* prob, const scalar_t* v_block,
                                   acc_t&& acc) {
  using v_load_vec_type = typename KernelVecType<scalar_t>::v_load_vec_type;
  constexpr int ELEM_NUM = v_load_vec_type::get_elem_num();
  static_assert(BLOCK_SIZE == ELEM_NUM);
  vec_op::FP32Vec16 prob_vec(prob);

  vec_op::unroll_loop<int, HEAD_PARTITION_SIZE>([&](int head_elem_idx) {
    v_load_vec_type v_vec(v_block + BLOCK_SIZE * head_elem_idx);
    vec_op::FP32Vec16 fp32_v_vec(v_vec);
    acc[head_elem_idx] = acc[head_elem_idx] + prob_vec * fp32_v_vec;
  });
}
};  // namespace

// Paged attention v1
namespace {
template <typename scalar_t, int HEAD_SIZE, int BLOCK_SIZE>
struct paged_attention_v1_impl {
  static void call(
      scalar_t* __restrict__ out,            // [num_seqs, num_heads, head_size]
      const scalar_t* __restrict__ q,        // [num_seqs, num_heads, head_size]
      const scalar_t* __restrict__ k_cache,  // [num_blocks, num_kv_heads,
                                             // head_size/x, block_size, x]
      const scalar_t* __restrict__ v_cache,  // [num_blocks, num_kv_heads,
                                             // head_size, block_size]
      const int num_kv_heads, const float scale,
      const int* __restrict__ block_tables,  // [num_seqs,
                                             // max_num_blocks_per_seq]
      const int* __restrict__ seq_lens,      // [num_seqs]
      const int max_num_blocks_per_seq,
      const float* __restrict__ alibi_slopes,  // [num_heads]
      const int q_stride, const int kv_block_stride, const int kv_head_stride,
      const int num_seqs, const int num_heads) {
    constexpr int x = 16 / sizeof(scalar_t);
    const int num_queries_per_kv = num_heads / num_kv_heads;

    static_assert(BLOCK_SIZE == 16);

    int max_seq_len = max_num_blocks_per_seq * BLOCK_SIZE;
    int max_seq_len_padded = (max_seq_len + 15) & 0xFFFFFFF0;
    TORCH_CHECK((max_seq_len_padded * sizeof(float)) % 64 == 0);

    const int parallel_work_item_num = omp_get_max_threads();

    size_t logits_bytes =
        parallel_work_item_num * max_seq_len_padded * sizeof(float);
    float* logits = (float*)std::aligned_alloc(
        64, logits_bytes);  // Cacheline alignment for each context token.
                            // [parallel_work_item_num, max_seq_len_padded]

#pragma omp parallel for collapse(2) schedule(dynamic, 1)
    for (int seq_idx = 0; seq_idx < num_seqs; ++seq_idx) {
      for (int head_idx = 0; head_idx < num_heads; ++head_idx) {
        int seq_len = seq_lens[seq_idx];
        const int* seq_block_table =
            block_tables + max_num_blocks_per_seq * seq_idx;
        const int block_num = (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
        const int64_t kv_head_idx = head_idx / num_queries_per_kv;
        const scalar_t* __restrict__ q_vec_ptr =
            q + seq_idx * q_stride + head_idx * HEAD_SIZE;
        const int last_block_token_num = seq_len - (block_num - 1) * BLOCK_SIZE;
        float* __restrict__ thread_block_logits =
            logits + omp_get_thread_num() * max_seq_len_padded;

        // Compute logits
        for (int block_idx = 0; block_idx < block_num; ++block_idx) {
          const int64_t physical_block_idx = seq_block_table[block_idx];
          const scalar_t* __restrict__ k_block_cache_ptr =
              k_cache + physical_block_idx * kv_block_stride +
              kv_head_idx * kv_head_stride;
          float* __restrict__ head_block_logits =
              thread_block_logits + block_idx * BLOCK_SIZE;

          reduceQKBlockKernel<scalar_t, HEAD_SIZE, BLOCK_SIZE, x>::call(
              q_vec_ptr, k_block_cache_ptr, head_block_logits, scale,
              block_idx == block_num - 1 ? last_block_token_num : BLOCK_SIZE);
        }

        // Compute softmax
        if (alibi_slopes) {
          reduceSoftmaxAlibi(thread_block_logits, seq_len,
                             block_num * BLOCK_SIZE, alibi_slopes[head_idx], 0,
                             seq_len);
        } else {
          reduceSoftmax(thread_block_logits, seq_len, block_num * BLOCK_SIZE);
        }

        // Compute value
        constexpr int head_elem_num_per_partition = 16;
        constexpr int head_partition_num =
            HEAD_SIZE / head_elem_num_per_partition;
        for (int head_part_idx = 0; head_part_idx < head_partition_num;
             ++head_part_idx) {
          vec_op::FP32Vec16 accums[head_elem_num_per_partition];
          scalar_t* __restrict__ out_ptr =
              out + seq_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE +
              head_part_idx * head_elem_num_per_partition;
          for (int block_idx = 0; block_idx < block_num; ++block_idx) {
            const int64_t physical_block_idx = seq_block_table[block_idx];
            const float* __restrict__ prob_vec_ptr =
                thread_block_logits + block_idx * BLOCK_SIZE;
            const scalar_t* __restrict__ v_block_cache_ptr =
                v_cache + physical_block_idx * kv_block_stride +
                kv_head_idx * kv_head_stride +
                BLOCK_SIZE * head_part_idx * head_elem_num_per_partition;
            reduceValueBlock<scalar_t, HEAD_SIZE, BLOCK_SIZE,
                             head_elem_num_per_partition>(
                prob_vec_ptr, v_block_cache_ptr, accums);

            if (block_idx != block_num - 1) {
              const int64_t next_physical_block_idx =
                  seq_block_table[block_idx + 1];
              const scalar_t* __restrict__ next_v_block_cache_ptr =
                  v_cache + next_physical_block_idx * kv_block_stride +
                  kv_head_idx * kv_head_stride +
                  BLOCK_SIZE * head_part_idx * head_elem_num_per_partition;
              vec_op::unroll_loop<int, head_elem_num_per_partition>(
                  [&](int head_elem_idx) {
                    if (head_elem_idx % 2 == 0) {
                      vec_op::prefetch(next_v_block_cache_ptr +
                                       BLOCK_SIZE * head_elem_idx);
                    }
                  });
            }
          }

          vec_op::unroll_loop<int, head_elem_num_per_partition>(
              [&](int head_elem_idx) {
                float value = accums[head_elem_idx].reduce_sum();
                vec_op::storeFP32(value, out_ptr + head_elem_idx);
              });
        }
      }
    }
    std::free(logits);
  }
};

#define LAUNCH_V1_ATTENTION_KERNEL(T, HEAD_SIZE, BLOCK_SIZE)                   \
  paged_attention_v1_impl<T, HEAD_SIZE, BLOCK_SIZE>::call(                     \
      out_ptr, query_ptr, key_cache_ptr, value_cache_ptr, num_kv_heads, scale, \
      block_tables_ptr, seq_lens_ptr, max_num_blocks_per_seq,                  \
      alibi_slopes_ptr, q_stride, kv_block_stride, kv_head_stride, num_seqs,   \
      num_heads);

template <typename T, int BLOCK_SIZE>
void paged_attention_v1_impl_launcher(
    torch::Tensor& out, torch::Tensor& query, torch::Tensor& key_cache,
    torch::Tensor& value_cache, int num_kv_heads, float scale,
    torch::Tensor& block_tables, torch::Tensor& seq_lens, int max_seq_len,
    const c10::optional<torch::Tensor>& alibi_slopes) {
  int num_seqs = query.size(0);
  int num_heads = query.size(1);
  int head_size = query.size(2);
  int max_num_blocks_per_seq = block_tables.size(1);
  int q_stride = query.stride(0);
  int kv_block_stride = key_cache.stride(0);
  int kv_head_stride = key_cache.stride(1);

  // NOTE: alibi_slopes is optional.
  const float* alibi_slopes_ptr =
      alibi_slopes
          ? reinterpret_cast<const float*>(alibi_slopes.value().data_ptr())
          : nullptr;

  T* out_ptr = reinterpret_cast<T*>(out.data_ptr());
  T* query_ptr = reinterpret_cast<T*>(query.data_ptr());
  T* key_cache_ptr = reinterpret_cast<T*>(key_cache.data_ptr());
  T* value_cache_ptr = reinterpret_cast<T*>(value_cache.data_ptr());
  int* block_tables_ptr = block_tables.data_ptr<int>();
  int* seq_lens_ptr = seq_lens.data_ptr<int>();

  switch (head_size) {
    case 64:
      LAUNCH_V1_ATTENTION_KERNEL(T, 64, BLOCK_SIZE);
      break;
    case 80:
      LAUNCH_V1_ATTENTION_KERNEL(T, 80, BLOCK_SIZE);
      break;
    case 96:
      LAUNCH_V1_ATTENTION_KERNEL(T, 96, BLOCK_SIZE);
      break;
    case 112:
      LAUNCH_V1_ATTENTION_KERNEL(T, 112, BLOCK_SIZE);
      break;
    case 128:
      LAUNCH_V1_ATTENTION_KERNEL(T, 128, BLOCK_SIZE);
      break;
    case 192:
      LAUNCH_V1_ATTENTION_KERNEL(T, 192, BLOCK_SIZE);
      break;
    case 256:
      LAUNCH_V1_ATTENTION_KERNEL(T, 256, BLOCK_SIZE);
      break;
    default:
      TORCH_CHECK(false, "Unsupported head size: ", head_size);
      break;
  }
}

#define CALL_V1_KERNEL_LAUNCHER(T, BLOCK_SIZE)                               \
  paged_attention_v1_impl_launcher<T, BLOCK_SIZE>(                           \
      out, query, key_cache, value_cache, num_kv_heads, scale, block_tables, \
      seq_lens, max_seq_len, alibi_slopes);

#define CALL_V1_KERNEL_LAUNCHER_BLOCK_SIZE(T)                     \
  switch (block_size) {                                           \
    case 16:                                                      \
      CALL_V1_KERNEL_LAUNCHER(T, 16);                             \
      break;                                                      \
    default:                                                      \
      TORCH_CHECK(false, "Unsupported block size: ", block_size); \
      break;                                                      \
  }
}  // namespace

void paged_attention_v1(
    torch::Tensor& out, torch::Tensor& query, torch::Tensor& key_cache,
    torch::Tensor& value_cache, int64_t num_kv_heads, double scale,
    torch::Tensor& block_tables, torch::Tensor& seq_lens, int64_t block_size,
    int64_t max_seq_len, const c10::optional<torch::Tensor>& alibi_slopes,
    const std::string& kv_cache_dtype, double k_scale, double v_scale,
    const int64_t tp_rank, const int64_t blocksparse_local_blocks,
    const int64_t blocksparse_vert_stride, const int64_t blocksparse_block_size,
    const int64_t blocksparse_head_sliding_step) {
  TORCH_CHECK(k_scale == 1.0f && v_scale == 1.0f);
  TORCH_CHECK(blocksparse_vert_stride <= 1,
              "CPU backend does not support blocksparse attention yet.");
  VLLM_DISPATCH_FLOATING_TYPES(query.scalar_type(), "paged_attention_v1_impl",
                               [&] {
                                 CPU_KERNEL_GUARD_IN(paged_attention_v1_impl)
                                 CALL_V1_KERNEL_LAUNCHER_BLOCK_SIZE(scalar_t);
                                 CPU_KERNEL_GUARD_OUT(paged_attention_v1_impl)
                               });
}

// Paged attention v2
namespace {
template <typename scalar_t, int HEAD_SIZE, int BLOCK_SIZE, int PARTITION_SIZE>
struct paged_attention_v2_impl {
  static void call(
      scalar_t* __restrict__ out,            // [num_seqs, num_heads, head_size]
      float* __restrict__ exp_sums,          // [num_seqs, num_heads,
                                             // max_num_partitions]
      float* __restrict__ max_logits,        // [num_seqs, num_heads,
                                             // max_num_partitions]
      scalar_t* __restrict__ tmp_out,        // [num_seqs, num_heads,
                                             // max_num_partitions, head_size]
      const scalar_t* __restrict__ q,        // [num_seqs, num_heads, head_size]
      const scalar_t* __restrict__ k_cache,  // [num_blocks, num_kv_heads,
                                             // head_size/x, block_size, x]
      const scalar_t* __restrict__ v_cache,  // [num_blocks, num_kv_heads,
                                             // head_size, block_size]
      const int num_kv_heads, const float scale,
      const int* __restrict__ block_tables,  // [num_seqs,
                                             // max_num_blocks_per_seq]
      const int* __restrict__ seq_lens,      // [num_seqs]
      const int max_num_blocks_per_seq,
      const float* __restrict__ alibi_slopes,  // [num_heads]
      const int q_stride, const int kv_block_stride, const int kv_head_stride,
      const int num_seqs, const int num_heads, const int max_num_partitions) {
    constexpr int x = 16 / sizeof(scalar_t);
    const int num_queries_per_kv = num_heads / num_kv_heads;

    static_assert(BLOCK_SIZE == 16);
    static_assert(PARTITION_SIZE * sizeof(float) % 64 == 0);
    static_assert(PARTITION_SIZE % BLOCK_SIZE == 0);

#pragma omp parallel for collapse(3) schedule(static, 1)
    for (int seq_idx = 0; seq_idx < num_seqs; ++seq_idx) {
      for (int partition_idx = 0; partition_idx < max_num_partitions;
           ++partition_idx) {
        for (int head_idx = 0; head_idx < num_heads; ++head_idx) {
          const int seq_len = seq_lens[seq_idx];
          const int start_token_idx = partition_idx * PARTITION_SIZE;

          if (start_token_idx >= seq_len) continue;

          const int partition_num =
              (seq_len + PARTITION_SIZE - 1) / PARTITION_SIZE;
          const bool no_reduce = (partition_num == 1);
          const int token_num =
              (std::min(seq_len, start_token_idx + PARTITION_SIZE) -
               start_token_idx);
          const int block_num = (token_num + BLOCK_SIZE - 1) / BLOCK_SIZE;
          const int last_block_token_num =
              token_num - (block_num - 1) * BLOCK_SIZE;
          const int* seq_block_table = block_tables +
                                       max_num_blocks_per_seq * seq_idx +
                                       start_token_idx / BLOCK_SIZE;
          const int64_t kv_head_idx = head_idx / num_queries_per_kv;
          const scalar_t* __restrict__ q_vec_ptr =
              q + seq_idx * q_stride + head_idx * HEAD_SIZE;

          float logits[PARTITION_SIZE] __attribute__((aligned(64))) = {0};

          // Compute logits
          for (int block_idx = 0; block_idx < block_num; ++block_idx) {
            const int64_t physical_block_idx = seq_block_table[block_idx];
            const scalar_t* __restrict__ k_block_cache_ptr =
                k_cache + physical_block_idx * kv_block_stride +
                kv_head_idx * kv_head_stride;
            float* __restrict__ head_block_logits =
                logits + block_idx * BLOCK_SIZE;

            reduceQKBlockKernel<scalar_t, HEAD_SIZE, BLOCK_SIZE, x>::call(
                q_vec_ptr, k_block_cache_ptr, head_block_logits, scale,
                block_idx == block_num - 1 ? last_block_token_num : BLOCK_SIZE);
          }

          std::pair<float, float> max_and_sum;
          if (alibi_slopes) {
            max_and_sum = reduceSoftmaxAlibi(
                logits, token_num, block_num * BLOCK_SIZE,
                alibi_slopes[head_idx], start_token_idx, seq_len);
          } else {
            max_and_sum =
                reduceSoftmax(logits, token_num, block_num * BLOCK_SIZE);
          }

          auto&& [max_logit, exp_sum] = max_and_sum;

          scalar_t* __restrict__ output_buffer = nullptr;
          if (!no_reduce) {
            auto idx = seq_idx * num_heads * max_num_partitions +
                       head_idx * max_num_partitions + partition_idx;
            max_logits[idx] = max_logit;
            exp_sums[idx] = exp_sum;
            output_buffer =
                tmp_out + seq_idx * num_heads * max_num_partitions * HEAD_SIZE +
                head_idx * max_num_partitions * HEAD_SIZE +
                partition_idx * HEAD_SIZE;
          } else {
            output_buffer =
                out + seq_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE;
          }

          // Compute value
          constexpr int head_elem_num_per_partition = 16;
          constexpr int head_partition_num =
              HEAD_SIZE / head_elem_num_per_partition;
          for (int head_part_idx = 0; head_part_idx < head_partition_num;
               ++head_part_idx) {
            vec_op::FP32Vec16 accums[head_elem_num_per_partition];
            scalar_t* __restrict__ out_ptr =
                output_buffer + head_part_idx * head_elem_num_per_partition;
            for (int block_idx = 0; block_idx < block_num; ++block_idx) {
              const int64_t physical_block_idx = seq_block_table[block_idx];
              const float* __restrict__ prob_vec_ptr =
                  logits + block_idx * BLOCK_SIZE;
              const scalar_t* __restrict__ v_block_cache_ptr =
                  v_cache + physical_block_idx * kv_block_stride +
                  kv_head_idx * kv_head_stride +
                  BLOCK_SIZE * head_part_idx * head_elem_num_per_partition;
              reduceValueBlock<scalar_t, HEAD_SIZE, BLOCK_SIZE,
                               head_elem_num_per_partition>(
                  prob_vec_ptr, v_block_cache_ptr, accums);

              if (block_idx != block_num - 1) {
                const int64_t next_physical_block_idx =
                    seq_block_table[block_idx + 1];
                const scalar_t* __restrict__ next_v_block_cache_ptr =
                    v_cache + next_physical_block_idx * kv_block_stride +
                    kv_head_idx * kv_head_stride +
                    BLOCK_SIZE * head_part_idx * head_elem_num_per_partition;
                vec_op::unroll_loop<int, head_elem_num_per_partition>(
                    [&](int head_elem_idx) {
                      if (head_elem_idx % 2 == 0) {
                        vec_op::prefetch(next_v_block_cache_ptr +
                                         BLOCK_SIZE * head_elem_idx);
                      }
                    });
              }
            }

            vec_op::unroll_loop<int, head_elem_num_per_partition>(
                [&](int head_elem_idx) {
                  float value = accums[head_elem_idx].reduce_sum();
                  vec_op::storeFP32(value, out_ptr + head_elem_idx);
                });
          }
        }
      }
    }

    // Rescale partition softmax and store the factors to exp_sums
#pragma omp parallel for collapse(2) schedule(static, 1)
    for (int seq_idx = 0; seq_idx < num_seqs; ++seq_idx) {
      for (int head_idx = 0; head_idx < num_heads; ++head_idx) {
        const int seq_len = seq_lens[seq_idx];
        const int partition_num =
            (seq_len + PARTITION_SIZE - 1) / PARTITION_SIZE;

        if (partition_num == 1) continue;

        reducePartitonSoftmax(
            max_logits + seq_idx * num_heads * max_num_partitions +
                head_idx * max_num_partitions,
            exp_sums + seq_idx * num_heads * max_num_partitions +
                head_idx * max_num_partitions,
            partition_num);
      }
    }

    // Reduce values
    using v_load_vec_type = typename KernelVecType<scalar_t>::v_load_vec_type;
    static_assert(v_load_vec_type::get_elem_num() == BLOCK_SIZE);
    constexpr int head_elem_num_per_group =
        16;  // Note: didn't align with the cacheline size, due to some
             // HEAD_SIZE didn't align with 64 bytes
    static_assert(HEAD_SIZE % head_elem_num_per_group == 0);
    constexpr int head_group_num = HEAD_SIZE / head_elem_num_per_group;
    const float* __restrict__ rescale_factors = exp_sums;
#pragma omp parallel for collapse(3) schedule(static, 1)
    for (int seq_idx = 0; seq_idx < num_seqs; ++seq_idx) {
      for (int head_idx = 0; head_idx < num_heads; ++head_idx) {
        for (int group_idx = 0; group_idx < head_group_num; ++group_idx) {
          const int seq_len = seq_lens[seq_idx];
          const int partition_num =
              (seq_len + PARTITION_SIZE - 1) / PARTITION_SIZE;

          if (partition_num == 1) continue;

          const float* __restrict__ seq_head_rescale_factors =
              rescale_factors + seq_idx * num_heads * max_num_partitions +
              head_idx * max_num_partitions;
          const scalar_t* __restrict__ seq_head_tmp_out =
              tmp_out + seq_idx * num_heads * max_num_partitions * HEAD_SIZE +
              head_idx * max_num_partitions * HEAD_SIZE +
              group_idx * head_elem_num_per_group;
          scalar_t* __restrict__ seq_head_output =
              out + seq_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE +
              group_idx * head_elem_num_per_group;

          vec_op::FP32Vec16 acc;
          for (int i = 0; i < partition_num; ++i) {
            vec_op::FP32Vec16 rescale_factor(seq_head_rescale_factors[i]);
            v_load_vec_type value(seq_head_tmp_out + i * HEAD_SIZE);
            vec_op::FP32Vec16 fp32_value(value);
            acc = acc + fp32_value * rescale_factor;
          }
          v_load_vec_type cast_acc(acc);
          cast_acc.save(seq_head_output);
        }
      }
    }
  }
};

#define LAUNCH_V2_ATTENTION_KERNEL(T, HEAD_SIZE, BLOCK_SIZE)                 \
  paged_attention_v2_impl<T, HEAD_SIZE, BLOCK_SIZE, PARTITION_SIZE>::call(   \
      out_ptr, exp_sums_ptr, max_logits_ptr, tmp_out_ptr, query_ptr,         \
      key_cache_ptr, value_cache_ptr, num_kv_heads, scale, block_tables_ptr, \
      seq_lens_ptr, max_num_blocks_per_seq, alibi_slopes_ptr, q_stride,      \
      kv_block_stride, kv_head_stride, num_seqs, num_heads,                  \
      max_num_partitions);

template <typename T, int BLOCK_SIZE, int PARTITION_SIZE = 512>
void paged_attention_v2_impl_launcher(
    torch::Tensor& out, torch::Tensor& exp_sums, torch::Tensor& max_logits,
    torch::Tensor& tmp_out, torch::Tensor& query, torch::Tensor& key_cache,
    torch::Tensor& value_cache, int num_kv_heads, float scale,
    torch::Tensor& block_tables, torch::Tensor& seq_lens, int block_size,
    int max_seq_len, const c10::optional<torch::Tensor>& alibi_slopes) {
  int num_seqs = query.size(0);
  int num_heads = query.size(1);
  int head_size = query.size(2);
  int max_num_blocks_per_seq = block_tables.size(1);
  int q_stride = query.stride(0);
  int kv_block_stride = key_cache.stride(0);
  int kv_head_stride = key_cache.stride(1);
  int max_num_partitions = exp_sums.size(-1);

  // NOTE: alibi_slopes is optional.
  const float* alibi_slopes_ptr =
      alibi_slopes
          ? reinterpret_cast<const float*>(alibi_slopes.value().data_ptr())
          : nullptr;

  T* out_ptr = reinterpret_cast<T*>(out.data_ptr());
  float* exp_sums_ptr = reinterpret_cast<float*>(exp_sums.data_ptr());
  float* max_logits_ptr = reinterpret_cast<float*>(max_logits.data_ptr());
  T* tmp_out_ptr = reinterpret_cast<T*>(tmp_out.data_ptr());
  T* query_ptr = reinterpret_cast<T*>(query.data_ptr());
  T* key_cache_ptr = reinterpret_cast<T*>(key_cache.data_ptr());
  T* value_cache_ptr = reinterpret_cast<T*>(value_cache.data_ptr());
  int* block_tables_ptr = block_tables.data_ptr<int>();
  int* seq_lens_ptr = seq_lens.data_ptr<int>();

  switch (head_size) {
    case 64:
      LAUNCH_V2_ATTENTION_KERNEL(T, 64, BLOCK_SIZE);
      break;
    case 80:
      LAUNCH_V2_ATTENTION_KERNEL(T, 80, BLOCK_SIZE);
      break;
    case 96:
      LAUNCH_V2_ATTENTION_KERNEL(T, 96, BLOCK_SIZE);
      break;
    case 112:
      LAUNCH_V2_ATTENTION_KERNEL(T, 112, BLOCK_SIZE);
      break;
    case 128:
      LAUNCH_V2_ATTENTION_KERNEL(T, 128, BLOCK_SIZE);
      break;
    case 192:
      LAUNCH_V2_ATTENTION_KERNEL(T, 192, BLOCK_SIZE);
      break;
    case 256:
      LAUNCH_V2_ATTENTION_KERNEL(T, 256, BLOCK_SIZE);
      break;
    default:
      TORCH_CHECK(false, "Unsupported head size: ", head_size);
      break;
  }
}

#define CALL_V2_KERNEL_LAUNCHER(T, BLOCK_SIZE)                              \
  paged_attention_v2_impl_launcher<T, BLOCK_SIZE>(                          \
      out, exp_sums, max_logits, tmp_out, query, key_cache, value_cache,    \
      num_kv_heads, scale, block_tables, seq_lens, block_size, max_seq_len, \
      alibi_slopes);

#define CALL_V2_KERNEL_LAUNCHER_BLOCK_SIZE(T)                     \
  switch (block_size) {                                           \
    case 16:                                                      \
      CALL_V2_KERNEL_LAUNCHER(T, 16);                             \
      break;                                                      \
    default:                                                      \
      TORCH_CHECK(false, "Unsupported block size: ", block_size); \
      break;                                                      \
  }
}  // namespace

void paged_attention_v2(
    torch::Tensor& out, torch::Tensor& exp_sums, torch::Tensor& max_logits,
    torch::Tensor& tmp_out, torch::Tensor& query, torch::Tensor& key_cache,
    torch::Tensor& value_cache, int64_t num_kv_heads, double scale,
    torch::Tensor& block_tables, torch::Tensor& seq_lens, int64_t block_size,
    int64_t max_seq_len, const c10::optional<torch::Tensor>& alibi_slopes,
    const std::string& kv_cache_dtype, double k_scale, double v_scale,
    const int64_t tp_rank, const int64_t blocksparse_local_blocks,
    const int64_t blocksparse_vert_stride, const int64_t blocksparse_block_size,
    const int64_t blocksparse_head_sliding_step) {
  TORCH_CHECK(k_scale == 1.0f && v_scale == 1.0f);
  TORCH_CHECK(blocksparse_vert_stride <= 1,
              "CPU backend does not support blocksparse attention yet.");
  VLLM_DISPATCH_FLOATING_TYPES(query.scalar_type(), "paged_attention_v2_impl",
                               [&] {
                                 CPU_KERNEL_GUARD_IN(paged_attention_v2_impl)
                                 CALL_V2_KERNEL_LAUNCHER_BLOCK_SIZE(scalar_t);
                                 CPU_KERNEL_GUARD_OUT(paged_attention_v2_impl)
                               });
}
