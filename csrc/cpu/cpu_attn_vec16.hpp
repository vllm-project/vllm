#ifndef CPU_ATTN_VEC16_HPP
#define CPU_ATTN_VEC16_HPP

#include "cpu_attn_vec.hpp"

namespace cpu_attention {

namespace {
// 16-1-16 pattern, 16 regs for A, 1 regs for B, 16 regs for C, [16, K] @ [k,
// 16]
template <typename kv_cache_t>
class TileGemm161 {
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
      case 9:
      case 10:
      case 11:
      case 12:
        gemm_micro<12>(a_tile, b_tile, c_tile, lda, ldb, ldc, block_size,
                       dynamic_k_size, accum_c);
        break;
      case 13:
      case 14:
      case 15:
      case 16:
        gemm_micro<16>(a_tile, b_tile, c_tile, lda, ldb, ldc, block_size,
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
    static_assert(0 < M <= 16);
    using load_vec_t = typename VecTypeTrait<kv_cache_t>::vec_t;

    kv_cache_t* __restrict__ curr_b_0 = b_tile;
    float* __restrict__ curr_c_0 = c_tile;

    vec_op::FP32Vec16 c_regs[M];
    if (accum_c) {
      float* __restrict__ curr_m_c_0 = curr_c_0;
      vec_op::unroll_loop<int32_t, M>([&](int32_t i) {
        c_regs[i] = vec_op::FP32Vec16(curr_m_c_0);

        // update
        curr_m_c_0 += ldc;
      });
    }

    float* __restrict__ curr_a = a_tile;
    for (int32_t k = 0; k < dynamic_k_size; ++k) {
      load_vec_t b_0_reg(curr_b_0);
      vec_op::FP32Vec16 fp32_b_0_reg(b_0_reg);

      float* __restrict__ curr_m_a = curr_a;
      vec_op::unroll_loop<int32_t, M>([&](int32_t i) {
        float v = *curr_m_a;
        vec_op::FP32Vec16 a_reg(v);
        c_regs[i] = c_regs[i] + a_reg * fp32_b_0_reg;

        // update
        curr_m_a += lda;
      });

      // update
      curr_a += 1;
      curr_b_0 += ldb;
    }

    vec_op::unroll_loop<int32_t, M>([&](int32_t i) {
      c_regs[i].save(curr_c_0);

      // update
      curr_c_0 += ldc;
    });
  }
};
}  // namespace

// This is a general but naive implementation based on vector instructions
template <typename scalar_t, int64_t head_dim>
class AttentionImpl<ISA::VEC16, scalar_t, head_dim>
    : public AttentionImpl<ISA::VEC, scalar_t, head_dim> {
 public:
  using query_t = scalar_t;
  using q_buffer_t = float;
  using kv_cache_t = scalar_t;
  using logits_buffer_t = float;
  using partial_output_buffer_t = float;
  using prob_buffer_t = float;

  constexpr static int64_t BlockSizeAlignment =
      16;  // KV token num unit of QK and PV phases
  constexpr static int64_t HeadDimAlignment =
      16;  // headdim num unit of PV phase
  constexpr static int64_t MaxQHeadNumPerIteration = 16;
  constexpr static int64_t HeadDim = head_dim;
  constexpr static ISA ISAType = ISA::VEC16;
  constexpr static bool scale_on_logits = false;  // apply scale on q_buffer

 public:
  template <template <typename tile_gemm_t> typename attention>
  FORCE_INLINE void execute_attention(DEFINE_CPU_ATTENTION_PARAMS) {
    attention<TileGemm161<kv_cache_t>> attention_iteration;
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
};
}  // namespace cpu_attention

#endif
