#ifndef CPU_ATTN_VEC_HPP
#define CPU_ATTN_VEC_HPP

#include "cpu_attn_fp8.hpp"
#include "cpu_attn_impl.hpp"

namespace cpu_attention {

namespace {
// 8-2-16 pattern, 8 regs for A, 2 regs for B, 16 regs for C, [8, K] @ [k, 32]
template <typename kv_cache_t>
class TileGemm82 {
 public:
  template <AttentionGemmPhase phase, int32_t k_size>
  FORCE_INLINE static void gemm(const int32_t m_size,
                                float* __restrict__ a_tile,
                                kv_cache_t* __restrict__ b_tile,
                                float* __restrict__ c_tile, const int64_t lda,
                                const int64_t ldb, const int64_t ldc,
                                const int32_t block_size,
                                const int32_t dynamic_k_size,
                                const bool accum_c) {
    switch (m_size) {
      case 1:
        gemm_micro<1>(a_tile, b_tile, c_tile, lda, ldb, ldc, block_size,
                      dynamic_k_size, accum_c);
        break;
      case 2:
        gemm_micro<2>(a_tile, b_tile, c_tile, lda, ldb, ldc, block_size,
                      dynamic_k_size, accum_c);
        break;
      case 3:
      case 4:
        gemm_micro<4>(a_tile, b_tile, c_tile, lda, ldb, ldc, block_size,
                      dynamic_k_size, accum_c);
        break;
      case 5:
      case 6:
        gemm_micro<6>(a_tile, b_tile, c_tile, lda, ldb, ldc, block_size,
                      dynamic_k_size, accum_c);
        break;
      case 7:
      case 8:
        gemm_micro<8>(a_tile, b_tile, c_tile, lda, ldb, ldc, block_size,
                      dynamic_k_size, accum_c);
        break;
    }
  }

  template <int32_t M>
  static void gemm_micro(float* __restrict__ a_tile,
                         kv_cache_t* __restrict__ b_tile,
                         float* __restrict__ c_tile, const int64_t lda,
                         const int64_t ldb, const int64_t ldc,
                         const int32_t block_size, const int32_t dynamic_k_size,
                         const bool accum_c) {
    static_assert(0 < M <= 8);
    using load_vec_t = typename VecTypeTrait<kv_cache_t>::vec_t;

    kv_cache_t* __restrict__ curr_b_0 = b_tile;
    kv_cache_t* __restrict__ curr_b_1 = b_tile + 16;
    float* __restrict__ curr_c_0 = c_tile;
    float* __restrict__ curr_c_1 = c_tile + 16;

    vec_op::FP32Vec16 c_regs[M * 2];
    if (accum_c) {
      float* __restrict__ curr_m_c_0 = curr_c_0;
      float* __restrict__ curr_m_c_1 = curr_c_1;
      vec_op::unroll_loop<int32_t, M>([&](int32_t i) {
        c_regs[i * 2] = vec_op::FP32Vec16(curr_m_c_0);
        c_regs[i * 2 + 1] = vec_op::FP32Vec16(curr_m_c_1);

        // update
        curr_m_c_0 += ldc;
        curr_m_c_1 += ldc;
      });
    }

    float* __restrict__ curr_a = a_tile;
    for (int32_t k = 0; k < dynamic_k_size; ++k) {
      load_vec_t b_0_reg(curr_b_0);
      vec_op::FP32Vec16 fp32_b_0_reg(b_0_reg);
      load_vec_t b_1_reg(curr_b_1);
      vec_op::FP32Vec16 fp32_b_1_reg(b_1_reg);

      float* __restrict__ curr_m_a = curr_a;
      vec_op::unroll_loop<int32_t, M>([&](int32_t i) {
        float v = *curr_m_a;
        vec_op::FP32Vec16 a_reg(v);
        c_regs[i * 2] = c_regs[i * 2] + a_reg * fp32_b_0_reg;
        c_regs[i * 2 + 1] = c_regs[i * 2 + 1] + a_reg * fp32_b_1_reg;

        // update
        curr_m_a += lda;
      });

      // update
      curr_a += 1;
      curr_b_0 += ldb;
      curr_b_1 += ldb;
    }

    vec_op::unroll_loop<int32_t, M>([&](int32_t i) {
      c_regs[i * 2].save(curr_c_0);
      c_regs[i * 2 + 1].save(curr_c_1);

      // update
      curr_c_0 += ldc;
      curr_c_1 += ldc;
    });
  }
};
// FP8 (E4M3/E5M2) variant of TileGemm82.  KV cache is stored as uint8_t.
// scale_2p8 (= k/v_scale × 2^8 for E4M3, = k/v_scale for E5M2) is folded
// into the Q buffer (QK phase) and P buffer (PV phase) by the caller, so
// gemm_micro uses a no-scale FP32Vec16 constructor, eliminating two
// _mm512_mul_ps calls per k-iteration.
class TileGemm82FP8 {
 public:
  using prob_t = float;

  static thread_local float s_k_scale;
  static thread_local float s_v_scale;
  static thread_local Fp8KVCacheDataType s_fp8_kv_dtype;
  static void set_scales(
      float k, float v,
      Fp8KVCacheDataType dtype = Fp8KVCacheDataType::kFp8E4M3) noexcept {
    s_k_scale = k;
    s_v_scale = v;
    s_fp8_kv_dtype = dtype;
  }

  // Scale the softmax probability buffer (shape [q_head_num, stride]) by
  // v_scale_2p8 in-place.  Called once per kv-tile after apply_softmax and
  // before the PV GEMM so that gemm_micro can use the no-scale constructor.
  static void scale_probs_buffer(float* __restrict__ probs, int32_t q_head_num,
                                 int32_t token_num, int64_t stride) noexcept {
    const float vscale = (s_fp8_kv_dtype == Fp8KVCacheDataType::kFp8E5M2)
                             ? s_v_scale
                             : s_v_scale * 0x1p8f;
    const vec_op::FP32Vec16 scale_vec(vscale);
    for (int32_t h = 0; h < q_head_num; ++h) {
      float* row = probs + h * stride;
      for (int32_t t = 0; t < token_num; t += 16)
        (vec_op::FP32Vec16(row + t) * scale_vec).save(row + t);
    }
  }

  template <AttentionGemmPhase phase, int32_t k_size>
  FORCE_INLINE static void gemm(const int32_t m_size,
                                float* __restrict__ a_tile,
                                uint8_t* __restrict__ b_tile,
                                float* __restrict__ c_tile, const int64_t lda,
                                const int64_t ldb, const int64_t ldc,
                                const int32_t block_size,
                                const int32_t dynamic_k_size,
                                const bool accum_c) {
    switch (m_size) {
      case 1:
        gemm_micro<1>(a_tile, b_tile, c_tile, lda, ldb, ldc, block_size,
                      dynamic_k_size, accum_c);
        break;
      case 2:
        gemm_micro<2>(a_tile, b_tile, c_tile, lda, ldb, ldc, block_size,
                      dynamic_k_size, accum_c);
        break;
      case 3:
      case 4:
        gemm_micro<4>(a_tile, b_tile, c_tile, lda, ldb, ldc, block_size,
                      dynamic_k_size, accum_c);
        break;
      case 5:
      case 6:
        gemm_micro<6>(a_tile, b_tile, c_tile, lda, ldb, ldc, block_size,
                      dynamic_k_size, accum_c);
        break;
      case 7:
      case 8:
        gemm_micro<8>(a_tile, b_tile, c_tile, lda, ldb, ldc, block_size,
                      dynamic_k_size, accum_c);
        break;
    }
  }

 private:
  template <int32_t M>
  static void gemm_micro(float* __restrict__ a_tile,
                         uint8_t* __restrict__ b_tile,
                         float* __restrict__ c_tile, const int64_t lda,
                         const int64_t ldb, const int64_t ldc,
                         const int32_t block_size, const int32_t dynamic_k_size,
                         const bool accum_c) {
    static_assert(0 < M && M <= 8);

    uint8_t* __restrict__ curr_b_0 = b_tile;
    float* __restrict__ curr_c_0 = c_tile;
    float* __restrict__ curr_c_1 = c_tile + 16;

    vec_op::FP32Vec16 c_regs[M * 2];
    if (accum_c) {
      float* __restrict__ curr_m_c_0 = curr_c_0;
      float* __restrict__ curr_m_c_1 = curr_c_1;
      vec_op::unroll_loop<int32_t, M>([&](int32_t i) {
        c_regs[i * 2] = vec_op::FP32Vec16(curr_m_c_0);
        c_regs[i * 2 + 1] = vec_op::FP32Vec16(curr_m_c_1);
        curr_m_c_0 += ldc;
        curr_m_c_1 += ldc;
      });
    }

    float* __restrict__ curr_a = a_tile;
    const bool is_e5m2 = (s_fp8_kv_dtype == Fp8KVCacheDataType::kFp8E5M2);
    for (int32_t k = 0; k < dynamic_k_size; ++k) {
      // Dequantize 32 FP8 bytes to pseudo-FP16 bits; convert to FP32 without
      // scale multiply — scale has been pre-folded into Q (QK) or P (PV).
      vec_op::BF16Vec32 fp16_b_reg =
          is_e5m2 ? vec_op::BF16Vec32(curr_b_0, vec_op::fp8_e5m2_tag{})
                  : vec_op::BF16Vec32(curr_b_0, 0.0f);
      vec_op::FP32Vec16 fp32_b_0_reg(fp16_b_reg, 0);
      vec_op::FP32Vec16 fp32_b_1_reg(fp16_b_reg, 1);

      float* __restrict__ curr_m_a = curr_a;
      vec_op::unroll_loop<int32_t, M>([&](int32_t i) {
        float v = *curr_m_a;
        vec_op::FP32Vec16 a_reg(v);
        c_regs[i * 2] = c_regs[i * 2] + a_reg * fp32_b_0_reg;
        c_regs[i * 2 + 1] = c_regs[i * 2 + 1] + a_reg * fp32_b_1_reg;
        curr_m_a += lda;
      });

      curr_a += 1;
      curr_b_0 += ldb;
    }

    vec_op::unroll_loop<int32_t, M>([&](int32_t i) {
      c_regs[i * 2].save(curr_c_0);
      c_regs[i * 2 + 1].save(curr_c_1);
      curr_c_0 += ldc;
      curr_c_1 += ldc;
    });
  }
};

thread_local float TileGemm82FP8::s_k_scale = 1.0f;
thread_local float TileGemm82FP8::s_v_scale = 1.0f;
thread_local Fp8KVCacheDataType TileGemm82FP8::s_fp8_kv_dtype =
    Fp8KVCacheDataType::kFp8E4M3;

}  // namespace

// This is a general but naive implementation based on vector instructions
template <typename scalar_t, int64_t head_dim>
class AttentionImpl<ISA::VEC, scalar_t, head_dim> {
 public:
  using query_t = scalar_t;
  using q_buffer_t = float;
  using kv_cache_t = scalar_t;
  using logits_buffer_t = float;
  using partial_output_buffer_t = float;
  using prob_buffer_t = float;

  constexpr static int64_t BlockSizeAlignment =
      32;  // KV token num unit of QK and PV phases
  constexpr static int64_t HeadDimAlignment =
      32;  // headdim num unit of PV phase
  constexpr static int64_t MaxQHeadNumPerIteration = 8;
  constexpr static int64_t HeadDim = head_dim;
  constexpr static ISA ISAType = ISA::VEC;
  constexpr static bool scale_on_logits = false;  // apply scale on q_buffer

 public:
  template <template <typename tile_gemm_t> typename attention>
  FORCE_INLINE void execute_attention(DEFINE_CPU_ATTENTION_PARAMS) {
    attention<TileGemm82<kv_cache_t>> attention_iteration;
    attention_iteration(CPU_ATTENTION_PARAMS);
  }

  // k_cache_token_group_stride: stride of K cache when move to next
  // BlockSizeAlignment tokens in a block
  constexpr static int64_t k_cache_token_group_stride(
      const int32_t block_size) {
    return BlockSizeAlignment;  // layout of k_cache block is [head_dim,
                                // block_size], row-major
  }

  // v_cache_token_group_stride: stride of V cache when move to next
  // BlockSizeAlignment tokens in a block
  constexpr static int64_t v_cache_token_group_stride(
      const int32_t block_size) {
    return head_dim * BlockSizeAlignment;  // layout of v_cache is [block_size,
                                           // head_dim], row-major
  }

  // v_cache_head_group_stride: stride of V cache when move to next
  // HeadDimAlignment head dims in a block
  constexpr static int64_t v_cache_head_group_stride(const int32_t block_size) {
    return HeadDimAlignment;  // layout of v_cache is [block_size, head_dim],
                              // row-major
  }

  // Copy q to q_buffer and cast it to fp32
  static void copy_q_heads_tile(
      scalar_t* __restrict__ src,  // [q_num, q_heads_per_kv, head_size]
      float* __restrict__ q_buffer, const int32_t q_num,
      const int32_t q_heads_per_kv, const int64_t q_num_stride,
      const int64_t q_head_stride, float scale) {
    static_assert(head_dim % 16 == 0);
    constexpr int32_t unroll_size = head_dim / 16;
    using load_vec_t = typename VecTypeTrait<scalar_t>::vec_t;

    vec_op::FP32Vec16 scale_vec(scale);
    for (int32_t q_num_idx = 0; q_num_idx < q_num; ++q_num_idx) {
      for (int32_t q_head_idx = 0; q_head_idx < q_heads_per_kv; ++q_head_idx) {
        scalar_t* __restrict__ curr_q =
            src + q_num_idx * q_num_stride + q_head_idx * q_head_stride;
        float* __restrict__ curr_q_buffer =
            q_buffer + q_num_idx * q_heads_per_kv * head_dim +
            q_head_idx * head_dim;

        vec_op::unroll_loop<int32_t, unroll_size>([&](int32_t i) {
          load_vec_t vec(curr_q);
          vec_op::FP32Vec16 fp32_vec(vec);
          fp32_vec = fp32_vec * scale_vec;
          fp32_vec.save(curr_q_buffer);

          curr_q += 16;
          curr_q_buffer += 16;
        });
      }
    }
  }

  // reshape K as column-major and V as row-major
  static void reshape_and_cache(
      const scalar_t* __restrict__ key, const scalar_t* __restrict__ value,
      scalar_t* __restrict__ key_cache, scalar_t* __restrict__ value_cache,
      const int64_t* __restrict__ slot_mapping, const int64_t token_num,
      const int64_t key_token_num_stride, const int64_t value_token_num_stride,
      const int64_t head_num, const int64_t key_head_num_stride,
      const int64_t value_head_num_stride, const int64_t num_blocks,
      const int64_t num_blocks_stride, const int64_t cache_head_num_stride,
      const int64_t block_size, const int64_t block_size_stride) {
#pragma omp parallel for collapse(2)
    for (int64_t token_idx = 0; token_idx < token_num; ++token_idx) {
      for (int64_t head_idx = 0; head_idx < head_num; ++head_idx) {
        const int64_t pos = slot_mapping[token_idx];
        if (pos < 0) {
          // skip
          continue;
        }

        const int64_t block_idx = pos / block_size;
        const int64_t block_offset = pos % block_size;
        {
          // Write Key as column-major
          const scalar_t* key_start_ptr = key +
                                          token_idx * key_token_num_stride +
                                          head_idx * key_head_num_stride;
          scalar_t* key_cache_start_ptr =
              key_cache + block_idx * num_blocks_stride +
              head_idx * cache_head_num_stride + block_offset;

#pragma GCC unroll 8
          for (int64_t i = 0, j = 0; i < head_dim; ++i, j += block_size) {
            key_cache_start_ptr[j] = key_start_ptr[i];
          }
        }
        {
          // Write Value as row-major
          const scalar_t* value_start_ptr = value +
                                            token_idx * value_token_num_stride +
                                            head_idx * value_head_num_stride;
          scalar_t* value_cache_start_ptr =
              value_cache + block_idx * num_blocks_stride +
              head_idx * cache_head_num_stride + block_offset * head_dim;
          std::memcpy(value_cache_start_ptr, value_start_ptr,
                      sizeof(scalar_t) * head_dim);
        }
      }
    }
  }
};
// FP8 KV cache specialisation for the VEC (AVX-512) path.
// Identical to AttentionImpl<ISA::VEC, scalar_t, head_dim> except that
// kv_cache_t is uint8_t and execute_attention uses TileGemm82FP8.
template <typename scalar_t, int64_t head_dim>
class AttentionImplFP8VEC : public AttentionImpl<ISA::VEC, scalar_t, head_dim> {
  using Base = AttentionImpl<ISA::VEC, scalar_t, head_dim>;

 public:
  using query_t = typename Base::query_t;
  using q_buffer_t = typename Base::q_buffer_t;
  using logits_buffer_t = typename Base::logits_buffer_t;
  using partial_output_buffer_t = typename Base::partial_output_buffer_t;
  using prob_buffer_t = typename Base::prob_buffer_t;
  // Override: KV cache is stored as uint8 (FP8 E4M3 or E5M2).
  using kv_cache_t = uint8_t;

  float k_scale = 1.0f;
  float v_scale = 1.0f;
  Fp8KVCacheDataType fp8_kv_dtype = Fp8KVCacheDataType::kFp8E4M3;

  void init_from_input(const AttentionInput* input) {
    k_scale = input->k_scale_fp8;
    v_scale = input->v_scale_fp8;
    fp8_kv_dtype = input->fp8_kv_dtype;
  }

  // Override: pre-multiply Q by k_scale_2p8 so that gemm_micro can skip the
  // per-element scale multiply (no-scale FP32Vec16 constructor is used).
  void copy_q_heads_tile(query_t* __restrict__ src,
                         q_buffer_t* __restrict__ q_buffer, const int32_t q_num,
                         const int32_t q_heads_per_kv,
                         const int64_t q_num_stride,
                         const int64_t q_head_stride, float scale) {
    const float k_scale_2p8 = (fp8_kv_dtype == Fp8KVCacheDataType::kFp8E5M2)
                                  ? k_scale
                                  : k_scale * 0x1p8f;
    Base::copy_q_heads_tile(src, q_buffer, q_num, q_heads_per_kv, q_num_stride,
                            q_head_stride, scale * k_scale_2p8);
  }

  template <template <typename tile_gemm_t> typename attention>
  FORCE_INLINE void execute_attention(DEFINE_CPU_ATTENTION_PARAMS) {
    TileGemm82FP8::set_scales(k_scale, v_scale, fp8_kv_dtype);
    attention<TileGemm82FP8> attention_iteration;
    attention_iteration(CPU_ATTENTION_PARAMS);
  }
};

}  // namespace cpu_attention

#endif
