#ifndef CPU_ATTN_VEC_HPP
#define CPU_ATTN_VEC_HPP

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
}  // namespace cpu_attention

#endif
