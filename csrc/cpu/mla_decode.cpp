#include "cpu_types.hpp"
#include <float.h>

namespace {
template <typename scalar_t>
struct KernelVecType {
  using qk_load_vec_type = void;
  using qk_vec_type = void;
  using v_load_vec_type = void;
};

template <>
struct KernelVecType<float> {
  using qk_load_vec_type = vec_op::FP32Vec16;
  using qk_vec_type = vec_op::FP32Vec16;
  using v_load_vec_type = vec_op::FP32Vec16;
};

template <>
struct KernelVecType<c10::Half> {
#if defined(__powerpc64__) || defined(__s390x__)
  // Power and s390x architecture-specific vector types
  using qk_load_vec_type = vec_op::FP32Vec16;
  using qk_vec_type = vec_op::FP32Vec16;
  using v_load_vec_type = vec_op::FP32Vec16;
#else
  // Fallback for other architectures, including x86
  using qk_load_vec_type = vec_op::FP16Vec16;
  using qk_vec_type = vec_op::FP32Vec16;
  using v_load_vec_type = vec_op::FP16Vec16;
#endif
};

#ifdef __AVX512BF16__
template <>
struct KernelVecType<c10::BFloat16> {
  using qk_load_vec_type = vec_op::BF16Vec32;
  using qk_vec_type = vec_op::BF16Vec32;
  using v_load_vec_type = vec_op::BF16Vec16;
};
#else
template <>
struct KernelVecType<c10::BFloat16> {
  using qk_load_vec_type = vec_op::BF16Vec16;
  using qk_vec_type = vec_op::FP32Vec16;
  using v_load_vec_type = vec_op::BF16Vec16;
};
#endif

template <int HEAD_DIM, int V_HEAD_DIM, int BLOCK_SIZE, int HEAD_UNROLL,
          typename qk_vec_type>
void mla_decode_block_head(
    const qk_vec_type* __restrict__ q_vecs,          // [HEAD_UNROLL, head_dim]
    const qk_vec_type* __restrict__ k_vecs,          // [block_size, head_dim]
    const vec_op::FP32Vec16* __restrict v_vecs_f32,  // [block_size, v_head_dim]
    float* __restrict__ acc_out,  // [HEAD_UNROLL, v_head_dim]
    float* __restrict__ acc_lse,  // [HEAD_UNROLL]
    const float scale, const int num_tokens) {
  using f32_vec_type = vec_op::FP32Vec16;
  constexpr int QK_NUM_ELEM = qk_vec_type::VEC_ELEM_NUM;
  constexpr int V_NUM_ELEM = f32_vec_type::VEC_ELEM_NUM;

  float logits[BLOCK_SIZE][HEAD_UNROLL] = {};  // initialize to zeros
  float max_val[HEAD_UNROLL];
  std::fill(max_val, max_val + HEAD_UNROLL, -FLT_MAX);

  f32_vec_type acc_vec[BLOCK_SIZE][HEAD_UNROLL];
  for (int i = 0; i < HEAD_DIM; i += QK_NUM_ELEM) {
    // load to registers
    qk_vec_type q_vec[HEAD_UNROLL];

#pragma unroll
    for (int unroll = 0; unroll < HEAD_UNROLL; ++unroll)
      q_vec[unroll] =
          qk_vec_type{q_vecs[(i + unroll * HEAD_DIM) / QK_NUM_ELEM]};

    for (int block_offset = 0; block_offset < num_tokens; ++block_offset) {
      qk_vec_type k_vec(k_vecs[(block_offset * HEAD_DIM + i) / QK_NUM_ELEM]);

#pragma unroll
      for (int unroll = 0; unroll < HEAD_UNROLL; ++unroll)
        vec_op::fma(acc_vec[block_offset][unroll], q_vec[unroll], k_vec);
    }
  }

  for (int block_offset = 0; block_offset < num_tokens; ++block_offset) {
#pragma unroll
    for (int unroll = 0; unroll < HEAD_UNROLL; ++unroll) {
      const float acc = acc_vec[block_offset][unroll].reduce_sum() * scale;
      logits[block_offset][unroll] = acc;
      max_val[unroll] = std::max(max_val[unroll], acc);
    }
  }

  float sum_exp[HEAD_UNROLL] = {};
  for (int block_offset = 0; block_offset < num_tokens; ++block_offset) {
#pragma unroll
    for (int unroll = 0; unroll < HEAD_UNROLL; ++unroll) {
      const float val =
          std::exp(logits[block_offset][unroll] - max_val[unroll]);
      logits[block_offset][unroll] = val;
      sum_exp[unroll] += val;
    }
  }

  f32_vec_type this_out[V_HEAD_DIM / V_NUM_ELEM][HEAD_UNROLL];

  for (int block_offset = 0; block_offset < num_tokens; ++block_offset) {
    // load to registers
    f32_vec_type scale_[HEAD_UNROLL];

#pragma unroll
    for (int unroll = 0; unroll < HEAD_UNROLL; ++unroll)
      scale_[unroll] =
          f32_vec_type{logits[block_offset][unroll] / sum_exp[unroll]};

    for (int i = 0; i < V_HEAD_DIM; i += V_NUM_ELEM) {
      f32_vec_type v_vec(
          v_vecs_f32[(block_offset * HEAD_DIM + i) / V_NUM_ELEM]);

#pragma unroll
      for (int unroll = 0; unroll < HEAD_UNROLL; ++unroll)
        vec_op::fma(this_out[i / V_NUM_ELEM][unroll], v_vec, scale_[unroll]);
    }
  }

  // merge attention state
  // section 2.2 in https://arxiv.org/pdf/2501.01005
  f32_vec_type prev_scale[HEAD_UNROLL];
  f32_vec_type curr_scale[HEAD_UNROLL];

#pragma unroll
  for (int unroll = 0; unroll < HEAD_UNROLL; ++unroll) {
    const float prev_lse = acc_lse[unroll];
    const float curr_lse = std::log(sum_exp[unroll]) +
                           max_val[unroll];  // add back max_val to get true lse
    // softmax trick
    const float max_lse = std::max(prev_lse, curr_lse);
    const float prev_sum_exp = std::exp(prev_lse - max_lse);
    const float curr_sum_exp = std::exp(curr_lse - max_lse);

    const float new_sum_exp = prev_sum_exp + curr_sum_exp;
    acc_lse[unroll] = std::log(new_sum_exp) + max_lse;

    prev_scale[unroll] = f32_vec_type{prev_sum_exp / new_sum_exp};
    curr_scale[unroll] = f32_vec_type{curr_sum_exp / new_sum_exp};
  }

  for (int i = 0; i < V_HEAD_DIM; i += V_NUM_ELEM) {
#pragma unroll
    for (int unroll = 0; unroll < HEAD_UNROLL; ++unroll) {
      f32_vec_type o_vec(acc_out + i + V_HEAD_DIM * unroll);
      o_vec = o_vec * prev_scale[unroll] +
              this_out[i / V_NUM_ELEM][unroll] * curr_scale[unroll];
      o_vec.save(acc_out + i + V_HEAD_DIM * unroll);
    }
  }

  q_vecs += HEAD_DIM / QK_NUM_ELEM * HEAD_UNROLL;
  acc_out += V_HEAD_DIM * HEAD_UNROLL;
}

template <typename scalar_t, int HEAD_DIM, int V_HEAD_DIM, int BLOCK_SIZE,
          typename qk_vec_type>
void mla_decode_block(
    const qk_vec_type* __restrict__ q_vecs,  // [num_heads, head_dim]
    const scalar_t* __restrict__ kv_cache,   // [block_size, head_dim]
    float* __restrict__ acc_out,             // [num_heads, v_head_dim]
    float* __restrict__ acc_lse,             // [num_heads]
    const int num_heads, const float scale, const int num_tokens) {
  using qk_load_vec_type = typename KernelVecType<scalar_t>::qk_load_vec_type;
  static_assert(
      std::is_same<qk_vec_type,
                   typename KernelVecType<scalar_t>::qk_vec_type>::value);
  using v_load_vec_type = typename KernelVecType<scalar_t>::v_load_vec_type;
  using f32_vec_type = vec_op::FP32Vec16;
  static_assert(qk_load_vec_type::VEC_ELEM_NUM == qk_vec_type::VEC_ELEM_NUM);
  static_assert(v_load_vec_type::VEC_ELEM_NUM == f32_vec_type::VEC_ELEM_NUM);
  constexpr int QK_NUM_ELEM = qk_vec_type::VEC_ELEM_NUM;
  constexpr int V_NUM_ELEM = v_load_vec_type::VEC_ELEM_NUM;

  const qk_vec_type* k_vecs;
  const f32_vec_type* v_vecs_f32;
  float* kv_cache_f32 = nullptr;

  if constexpr (!std::is_same<scalar_t, float>::value) {
    // convert KV cache block to FP32 to reuse it across query heads and
    // attn @ V computation, since FP16/BF16->FP32 is expensive.
    // TODO: move malloc outside of this fn to reuse across iterations.
    const int nbytes = BLOCK_SIZE * HEAD_DIM * sizeof(float);
    kv_cache_f32 = static_cast<float*>(std::aligned_alloc(64, nbytes));

    for (int block_offset = 0; block_offset < num_tokens; ++block_offset)
      for (int i = 0; i < HEAD_DIM; i += V_NUM_ELEM) {
        v_load_vec_type kv_load_vec(kv_cache + block_offset * HEAD_DIM + i);
        f32_vec_type kv_vec_f32(kv_load_vec);
        kv_vec_f32.save(kv_cache_f32 + block_offset * HEAD_DIM + i);
      }

    if constexpr (std::is_same<qk_load_vec_type, qk_vec_type>::value) {
      // for AVX512_BF16, Q @ K.T uses BF16 for K (no conversion)
      // NOTE: in this case, we only need to convert the V section to FP32.
      // But for simplicity, we will convert the whole KV block to FP32.
      k_vecs = reinterpret_cast<const qk_vec_type*>(kv_cache);
    } else {
      k_vecs = reinterpret_cast<const qk_vec_type*>(kv_cache_f32);
    }

    // attn @ V always use FP32 for V, since attn is FP32.
    v_vecs_f32 = reinterpret_cast<const f32_vec_type*>(kv_cache_f32);

  } else {
    // KV cache is FP32. don't need to do anything.
    k_vecs = reinterpret_cast<const qk_vec_type*>(kv_cache);
    v_vecs_f32 = reinterpret_cast<const f32_vec_type*>(kv_cache);
  }

  // compute 2 heads at the same time to improve ILP and
  // take advantage of register cache for K and V.
  constexpr int HEAD_UNROLL = 2;
  for (int iter = 0; iter < num_heads / HEAD_UNROLL; ++iter) {
    mla_decode_block_head<HEAD_DIM, V_HEAD_DIM, BLOCK_SIZE, HEAD_UNROLL>(
        q_vecs, k_vecs, v_vecs_f32, acc_out, acc_lse, scale, num_tokens);

    q_vecs += HEAD_UNROLL * HEAD_DIM / QK_NUM_ELEM;
    acc_out += HEAD_UNROLL * V_HEAD_DIM;
    acc_lse += HEAD_UNROLL;
  }

  // take care of the remaining heads
  for (int iter = 0; iter < num_heads % HEAD_UNROLL; ++iter) {
    mla_decode_block_head<HEAD_DIM, V_HEAD_DIM, BLOCK_SIZE, 1>(
        q_vecs, k_vecs, v_vecs_f32, acc_out, acc_lse, scale, num_tokens);

    q_vecs += HEAD_DIM / QK_NUM_ELEM;
    acc_out += V_HEAD_DIM;
    acc_lse += 1;
  }

  if (kv_cache_f32 != nullptr) {
    std::free(kv_cache_f32);
  }
}
}  // namespace

template <typename scalar_t, int HEAD_DIM, int V_HEAD_DIM, int BLOCK_SIZE>
void mla_decode_kvcache_cpu_impl(
    scalar_t* __restrict__ out,             // [num_seqs, num_heads, v_head_dim]
    const scalar_t* __restrict__ q,         // [num_seqs, num_heads, head_dim]
    const scalar_t* __restrict__ kv_cache,  // [num_blocks, block_size,
                                            // head_dim]
    const int num_heads, const float scale,
    const int* __restrict__ block_tables,  // [num_seqs, max_num_blocks_per_seq]
    const int* __restrict__ seq_lens,      // [num_seqs]
    const int max_num_blocks_per_seq, const int o_stride, const int q_stride,
    const int kv_stride, const int num_seqs) {
  using qk_load_vec_type = typename KernelVecType<scalar_t>::qk_load_vec_type;
  using qk_vec_type = typename KernelVecType<scalar_t>::qk_vec_type;
  constexpr int QK_NUM_ELEM = qk_vec_type::VEC_ELEM_NUM;

  // shared across threads
  const int max_threads = omp_get_max_threads();
  const int acc_out_nbytes =
      max_threads * num_heads * V_HEAD_DIM * sizeof(float);
  float* acc_out = static_cast<float*>(std::aligned_alloc(64, acc_out_nbytes));
  std::vector<float> acc_lse(max_threads * num_heads);

  // allocate memory to pre-convert query to FP32 later
  float* q_f32;
  constexpr bool PRE_CONVERT_QUERY =
      !std::is_same<scalar_t, float>::value &&
      std::is_same<qk_vec_type, vec_op::FP32Vec16>::value;
  if constexpr (PRE_CONVERT_QUERY) {
    const int q_f32_nbytes = num_heads * HEAD_DIM * sizeof(float);
    q_f32 = static_cast<float*>(std::aligned_alloc(64, q_f32_nbytes));
  }

#pragma omp parallel
  {
    const int num_threads = omp_get_num_threads();
    const int thread_id = omp_get_thread_num();
    float* __restrict__ acc_out_thread =
        acc_out + thread_id * num_heads * V_HEAD_DIM;
    float* __restrict__ acc_lse_thread = acc_lse.data() + thread_id * num_heads;

    for (int seq_idx = 0; seq_idx < num_seqs; ++seq_idx) {
      // reset accumulator
      std::fill(acc_out_thread, acc_out_thread + num_heads * V_HEAD_DIM, 0.0f);
      std::fill(acc_lse_thread, acc_lse_thread + num_heads, -FLT_MAX);

      const int seq_len = seq_lens[seq_idx];
      const int block_num = (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
      const int last_block_size = seq_len - (block_num - 1) * BLOCK_SIZE;

      const qk_vec_type* q_vecs;
      if constexpr (PRE_CONVERT_QUERY) {
// pre-convert query to FP32 since FP16/BF16->FP32 is slow.
#pragma omp for
        for (int i = 0; i < num_heads * HEAD_DIM; i += QK_NUM_ELEM) {
          qk_load_vec_type q_load_vec(q + seq_idx * q_stride + i);
          qk_vec_type q_vec(q_load_vec);
          q_vec.save(q_f32 + i);
        }
        q_vecs = reinterpret_cast<const qk_vec_type*>(q_f32);
      } else {
        q_vecs = reinterpret_cast<const qk_vec_type*>(q + seq_idx * q_stride);
      }

#pragma omp for
      for (int block_idx = 0; block_idx < block_num; ++block_idx) {
        const int physical_block_idx =
            block_tables[seq_idx * max_num_blocks_per_seq + block_idx];
        const int num_tokens =
            block_idx < block_num - 1 ? BLOCK_SIZE : last_block_size;

        mla_decode_block<scalar_t, HEAD_DIM, V_HEAD_DIM, BLOCK_SIZE>(
            q_vecs, kv_cache + physical_block_idx * kv_stride, acc_out_thread,
            acc_lse_thread, num_heads, scale, num_tokens);
      }

// merge attention states across threads
// section 2.2 in https://arxiv.org/pdf/2501.01005
// each thread is responsible for 1 head
#pragma omp for
      for (int head_idx = 0; head_idx < num_heads; ++head_idx) {
        float* acc_lse_head = acc_lse.data() + head_idx;
        float* acc_out_head = acc_out + head_idx * V_HEAD_DIM;

        float max_val = -FLT_MAX;
        for (int thread_id_ = 0; thread_id_ < num_threads; ++thread_id_) {
          max_val = std::max(max_val, acc_lse_head[thread_id_ * num_heads]);
        }

        float sum_exp = 0.0f;
        for (int thread_id_ = 0; thread_id_ < num_threads; ++thread_id_) {
          float val = std::exp(acc_lse_head[thread_id_ * num_heads] - max_val);
          acc_lse_head[thread_id_ * num_heads] = val;
          sum_exp += val;
        }

        float inv_sum = 1.0f / sum_exp;
        float out_head[V_HEAD_DIM] = {};
        for (int thread_id_ = 0; thread_id_ < num_threads; ++thread_id_) {
          float scale_ = acc_lse_head[thread_id_ * num_heads] * inv_sum;
          for (int i = 0; i < V_HEAD_DIM; ++i) {
            out_head[i] +=
                acc_out_head[thread_id_ * num_heads * V_HEAD_DIM + i] * scale_;
          }
        }

        for (int i = 0; i < V_HEAD_DIM; ++i) {
          vec_op::storeFP32(out_head[i], out + seq_idx * o_stride +
                                             head_idx * V_HEAD_DIM + i);
        }
      }
    }
  }
  if (PRE_CONVERT_QUERY) {
    std::free(q_f32);
  }
  std::free(acc_out);
}

void mla_decode_kvcache(torch::Tensor& out, torch::Tensor& query,
                        torch::Tensor& kv_cache, double scale,
                        torch::Tensor& block_tables, torch::Tensor& seq_lens) {
  const int num_seqs = query.size(0);
  const int num_heads = query.size(1);
  const int head_dim = query.size(2);
  const int block_size = kv_cache.size(1);
  const int v_head_dim = out.size(2);

  const int max_num_blocks_per_seq = block_tables.size(1);
  const int o_stride = out.stride(0);
  const int q_stride = query.stride(0);
  const int kv_stride = kv_cache.stride(0);

  VLLM_DISPATCH_FLOATING_TYPES(
      query.scalar_type(), "mla_decode_kvcache_cpu_impl", [&] {
        CPU_KERNEL_GUARD_IN(mla_decode_kvcache_cpu_impl)
        if (head_dim == 576 && v_head_dim == 512 && block_size == 16)
          mla_decode_kvcache_cpu_impl<scalar_t, 576, 512, 16>(
              out.data_ptr<scalar_t>(), query.data_ptr<scalar_t>(),
              kv_cache.data_ptr<scalar_t>(), num_heads, scale,
              block_tables.data_ptr<int>(), seq_lens.data_ptr<int>(),
              max_num_blocks_per_seq, o_stride, q_stride, kv_stride, num_seqs);
        else
          TORCH_CHECK(false, "Unsupported block size: ", block_size);
        CPU_KERNEL_GUARD_OUT(mla_decode_kvcache_cpu_impl)
      });
}