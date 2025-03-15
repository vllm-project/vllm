#include "cpu_types.hpp"
#include <float.h>
#ifdef __x86_64__
  #include <immintrin.h>
#endif

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
#elif defined(__aarch64__) && !defined(ARM_BF16_SUPPORT)
// pass
#else
template <>
struct KernelVecType<c10::BFloat16> {
  using qk_load_vec_type = vec_op::BF16Vec16;
  using qk_vec_type = vec_op::FP32Vec16;
  using v_load_vec_type = vec_op::BF16Vec16;
};
#endif

template <typename T1, typename T2>
void convert(T1& dst, const T2& src) {
  T1 tmp(src);
  dst = tmp;
}

#ifdef __x86_64__
  #ifdef __AVX512F__
  // pass
  #else  // AVX2
// custom BF16<->FP32 dtype conversion optimized for x86 AVX2
// this uses different layout compared to the ones defined in vec_op.
// this results in 10-20% speedup for the mla_decode_kv_cache() kernel.
// NOTE: might remove if AVX2 optimization is not important.

// 256-bit BF16 src: [a00, a01, ..., a15]
//               | 256-bit FP32 high dst | 256-bit FP32 low dst
// vec_op layout |  [a00, a01, ..., a07] |  [a08, a09, ..., a15]
// this layout   |  [a00, a02, ..., a14] |  [a01, a03, ..., a15]
template <>
void convert(vec_op::FP32Vec16& dst, const vec_op::BF16Vec16& src) {
  __m256i v_low = _mm256_setzero_si256();
  v_low = _mm256_blend_epi16(src.reg, v_low, 0xAA);  // select even elements
  v_low = _mm256_bslli_epi128(v_low, 2);
  dst.reg_low = _mm256_castsi256_ps(v_low);

  __m256i v_high = _mm256_setzero_si256();
  v_high = _mm256_blend_epi16(src.reg, v_high, 0x55);  // select odd elements
  dst.reg_high = _mm256_castsi256_ps(v_high);
}

// 2x 256-bit FP32 src: [a00, a01, ..., a07], [b00, b01, ..., b07]
//                |            256-bit BF16 dst
// vec_op layout  | [a00, a01, ..., a07, b00, b01, ..., b07]
// this layout    | [a00, b00, a01, b01, ..., ..., a07, b07]
//
// NOTE: similar to vec_op, there is no rounding handling here for FP32->BF16
template <>
void convert(vec_op::BF16Vec16& dst, const vec_op::FP32Vec16& src) {
  __m256i low = _mm256_setzero_si256();
  low = _mm256_blend_epi16(_mm256_castps_si256(src.reg_low), low,
                           0x55);  // take the higher 16-bit
  low = _mm256_bsrli_epi128(low, 2);

  __m256i high = _mm256_setzero_si256();
  high = _mm256_blend_epi16(_mm256_castps_si256(src.reg_high), high, 0x55);
  dst.reg = _mm256_or_si256(low, high);
}
  #endif
#endif

template <typename scalar_t, int HEAD_DIM, int V_HEAD_DIM, int BLOCK_SIZE>
void mla_decode_block(
    const scalar_t* __restrict__ q,         // [num_heads, head_dim]
    const scalar_t* __restrict__ kv_cache,  // [block_size, head_dim]
    float* __restrict__ acc_out,  // [num_heads, v_head_dim]. TODO: add
                                  // alignment hint?
    float* __restrict__ acc_lse,  // [num_heads]
    const int num_heads, const float scale, const int num_tokens) {
  using qk_load_vec_type = typename KernelVecType<scalar_t>::qk_load_vec_type;
  using qk_vec_type = typename KernelVecType<scalar_t>::qk_vec_type;
  using v_load_vec_type = typename KernelVecType<scalar_t>::v_load_vec_type;
  using f32_vec_type = vec_op::FP32Vec16;
  static_assert(qk_load_vec_type::VEC_ELEM_NUM == qk_vec_type::VEC_ELEM_NUM);
  static_assert(v_load_vec_type::VEC_ELEM_NUM == f32_vec_type::VEC_ELEM_NUM);
  constexpr int QK_NUM_ELEM = qk_vec_type::VEC_ELEM_NUM;
  constexpr int V_NUM_ELEM = v_load_vec_type::VEC_ELEM_NUM;

  const qk_vec_type* k_vecs;
  const f32_vec_type* v_vecs_f32;
  float* kv_cache_f32 = nullptr;

  // TODO: we should also pre-convert q to FP32

  if constexpr (!std::is_same<scalar_t, float>::value) {
    // convert KV cache block to FP32 to reuse it across query heads and
    // attn @ V computation, since FP16/BF16->FP32 is expensive. The FP32
    // KV cache should live in CPU cache.
    // TODO: move malloc outside of this fn to reuse across iterations.
    const int nbytes = BLOCK_SIZE * HEAD_DIM * sizeof(float);
    kv_cache_f32 = static_cast<float*>(std::aligned_alloc(64, nbytes));

    for (int block_offset = 0; block_offset < num_tokens; ++block_offset)
      for (int i = 0; i < HEAD_DIM; i += V_NUM_ELEM) {
        v_load_vec_type kv_load_vec(kv_cache + block_offset * HEAD_DIM + i);
        f32_vec_type kv_vec_f32;
        convert(kv_vec_f32, kv_load_vec);
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

  for (int head_idx = 0; head_idx < num_heads; ++head_idx) {
    float logits[BLOCK_SIZE] = {};  // initialize to zeros
    float max_val = -FLT_MAX;

    for (int block_offset = 0; block_offset < num_tokens; ++block_offset) {
      // dot product
      f32_vec_type acc_vec;
      for (int i = 0; i < HEAD_DIM; i += QK_NUM_ELEM) {
        qk_load_vec_type q_load_vec(q + head_idx * HEAD_DIM + i);
        qk_vec_type q_vec;
        convert(q_vec, q_load_vec);
        qk_vec_type k_vec(k_vecs[(block_offset * HEAD_DIM + i) / QK_NUM_ELEM]);
        vec_op::fma(acc_vec, q_vec, k_vec);
      }
      float acc = acc_vec.reduce_sum();

      acc *= scale;  // softmax scale
      logits[block_offset] = acc;
      max_val = std::max(max_val, acc);
    }

    float sum_exp = 0.0f;
    for (int block_offset = 0; block_offset < num_tokens; ++block_offset) {
      const float val = std::exp(logits[block_offset] - max_val);
      logits[block_offset] = val;
      sum_exp += val;
    }

    f32_vec_type this_out[V_HEAD_DIM / V_NUM_ELEM];
    float inv_sum = 1.0f / sum_exp;

    for (int block_offset = 0; block_offset < BLOCK_SIZE; ++block_offset) {
      f32_vec_type scale_(logits[block_offset] * inv_sum);

      for (int i = 0; i < V_HEAD_DIM; i += V_NUM_ELEM) {
        f32_vec_type v_vec(
            v_vecs_f32[(block_offset * HEAD_DIM + i) / V_NUM_ELEM]);
        vec_op::fma(this_out[i / V_NUM_ELEM], v_vec, scale_);
      }
    }

    // merge attention state
    // section 2.2 in https://arxiv.org/pdf/2501.01005
    const float prev_lse = acc_lse[head_idx];
    const float curr_lse =
        std::log(sum_exp) + max_val;  // add back max_val to get true lse
    // softmax trick
    max_val = std::max(prev_lse, curr_lse);
    const float prev_sum_exp = std::exp(prev_lse - max_val);
    const float curr_sum_exp = std::exp(curr_lse - max_val);

    const float new_sum_exp = prev_sum_exp + curr_sum_exp;
    f32_vec_type prev_scale(prev_sum_exp / new_sum_exp);
    f32_vec_type curr_scale(curr_sum_exp / new_sum_exp);

    acc_lse[head_idx] = std::log(new_sum_exp) + max_val;
    for (int i = 0; i < V_HEAD_DIM; i += V_NUM_ELEM) {
      f32_vec_type o_vec(acc_out + head_idx * V_HEAD_DIM + i);
      o_vec = o_vec * prev_scale + this_out[i / V_NUM_ELEM] * curr_scale;
      o_vec.save(acc_out + head_idx * V_HEAD_DIM + i);
    }
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
  // there is n query heads and 1 key-value head, where value overlaps witih key
  // to make good use of data reuse, we want to
  // -> for each kv token, compute required problem for all query heads
  // -> this means that we have to do softmax(QK.T)@V in 1 pass
  //   - if KV cache is separate, it's ok to do attention in 2 passes, since
  //     logits is not that big, and we need to read fresh V anyway.
  // -> let seq_len dim is the outer most loop. we chunk the seq_len, compute
  // sub-problem
  //   - chunk according to BLOCK_SIZE for convenience

#pragma omp parallel for schedule(static, 1)
  for (int seq_idx = 0; seq_idx < num_seqs; ++seq_idx) {
    const int seq_len = seq_lens[seq_idx];
    const int block_num = (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int last_block_size = seq_len - (block_num - 1) * BLOCK_SIZE;

    const int acc_out_nbytes = num_heads * V_HEAD_DIM * sizeof(float);
    float* acc_out =
        static_cast<float*>(std::aligned_alloc(64, acc_out_nbytes));
    std::fill(acc_out, acc_out + num_heads * V_HEAD_DIM, 0.0f);
    std::vector<float> acc_lse(num_heads, -FLT_MAX);

    for (int block_idx = 0; block_idx < block_num; ++block_idx) {
      const int physical_block_idx =
          block_tables[seq_idx * max_num_blocks_per_seq + block_idx];
      const int num_tokens =
          block_idx < block_num - 1 ? BLOCK_SIZE : last_block_size;

      mla_decode_block<scalar_t, HEAD_DIM, V_HEAD_DIM, BLOCK_SIZE>(
          q + seq_idx * q_stride, kv_cache + physical_block_idx * kv_stride,
          acc_out, acc_lse.data(), num_heads, scale, num_tokens);
    }

    // we can't use vec_op::storeFP32() here since acc_out may use custom layout
    // as described in convert() function above.
    using o_store_vec_type = typename KernelVecType<scalar_t>::v_load_vec_type;
    for (int i = 0; i < num_heads * V_HEAD_DIM;
         i += o_store_vec_type::VEC_ELEM_NUM) {
      vec_op::FP32Vec16 o_vec_f32(acc_out + i);
      o_store_vec_type o_vec;
      convert(o_vec, o_vec_f32);
      o_vec.save(out + seq_idx * o_stride + i);
    }

    std::free(acc_out);
  }
  return;
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