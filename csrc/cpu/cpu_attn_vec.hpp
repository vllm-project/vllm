#ifndef CPU_ATTN_VEC_HPP
#define CPU_ATTN_VEC_HPP

#include "cpu_attn_fp8.hpp"
#include "cpu_attn_impl.hpp"

namespace cpu_attention {

namespace {

// Load 32 kv_cache_t elements starting at ptr and return them as two FP32Vec16s
// covering the lower 16 and upper 16 positions.
// For FP8: both halves come from a single BF16Vec32 dequant of 32 bytes.
// For BF16/FP16/FP32: two separate vector loads at ptr and ptr+16.
template <typename kv_cache_t>
FORCE_INLINE std::pair<vec_op::FP32Vec16, vec_op::FP32Vec16> load_b_pair_vec(
    const kv_cache_t* ptr) {
  if constexpr (std::is_same_v<kv_cache_t, c10::Float8_e4m3fn>) {
    // BF16 container, but values are in the FP16 exponent range (bias 15 not
    // 127).
    vec_op::BF16Vec32 bf16_b_reg(reinterpret_cast<const uint8_t*>(ptr),
                                 vec_op::fp8_e4m3_tag{});
    return {vec_op::FP32Vec16(bf16_b_reg, 0), vec_op::FP32Vec16(bf16_b_reg, 1)};
  } else if constexpr (std::is_same_v<kv_cache_t, c10::Float8_e5m2>) {
    vec_op::BF16Vec32 bf16_b_reg(reinterpret_cast<const uint8_t*>(ptr),
                                 vec_op::fp8_e5m2_tag{});
    return {vec_op::FP32Vec16(bf16_b_reg, 0), vec_op::FP32Vec16(bf16_b_reg, 1)};
  } else {
    using load_vec_t = typename VecTypeTrait<kv_cache_t>::vec_t;
    return {vec_op::FP32Vec16(load_vec_t(ptr)),
            vec_op::FP32Vec16(load_vec_t(ptr + 16))};
  }
}

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
    static_assert(0 < M && M <= 8);

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
    kv_cache_t* __restrict__ curr_b = b_tile;

    for (int32_t k = 0; k < dynamic_k_size; ++k) {
      auto [fp32_b_0_reg, fp32_b_1_reg] = load_b_pair_vec(curr_b);

      float* __restrict__ curr_m_a = curr_a;
      vec_op::unroll_loop<int32_t, M>([&](int32_t i) {
        vec_op::FP32Vec16 a_reg(*curr_m_a);
        c_regs[i * 2] = c_regs[i * 2] + a_reg * fp32_b_0_reg;
        c_regs[i * 2 + 1] = c_regs[i * 2 + 1] + a_reg * fp32_b_1_reg;

        // update
        curr_m_a += lda;
      });

      // update
      curr_a += 1;
      curr_b += ldb;
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
template <typename scalar_t, int64_t head_dim, typename kv_cache_scalar_t>
class AttentionImpl<ISA::VEC, scalar_t, head_dim, kv_cache_scalar_t> {
  static constexpr bool fp8_kv =
      std::is_same_v<kv_cache_scalar_t, c10::Float8_e4m3fn> ||
      std::is_same_v<kv_cache_scalar_t, c10::Float8_e5m2>;

 public:
  using query_t = scalar_t;
  using q_buffer_t = float;
  using kv_cache_t = kv_cache_scalar_t;
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
  constexpr static bool scale_on_logits = fp8_kv;

  float k_scale = 1.0f;
  float v_scale = 1.0f;

 public:
  void init_from_input(const AttentionInput* input) {
    if constexpr (fp8_kv) {
      k_scale = input->k_scale_fp8;
      v_scale = input->v_scale_fp8;
    }
  }

  float get_output_v_scale() const noexcept {
    if constexpr (fp8_kv) {
      // VEC dequant unpacks FP8 into a pseudo-FP16 layout (exponent bias 15).
      // E4M3 (bias=7) needs correction 2^(15-7) = 2^8; E5M2 bias matches FP16
      // so no correction.
      if constexpr (std::is_same_v<kv_cache_t, c10::Float8_e5m2>) {
        return v_scale;
      } else {
        return v_scale * 0x1p8f;
      }
    }
    return 1.0f;
  }

  template <template <typename tile_gemm_t> typename attention>
  FORCE_INLINE void execute_attention(DEFINE_CPU_ATTENTION_PARAMS) {
    if constexpr (fp8_kv) {
      // Same bias correction as get_output_v_scale: VEC FP8→pseudo-FP16 dequant
      // uses bias 15; E4M3 (bias=7) needs ×2^8, E5M2 (bias=15) needs no
      // correction.
      if constexpr (std::is_same_v<kv_cache_t, c10::Float8_e5m2>) {
        scale *= k_scale;
      } else {
        scale *= k_scale * 0x1p8f;
      }
    }
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

  // Copy q to q_buffer and cast it to fp32.
  // FP8: QK scale is folded into execute_attention; copy Q unscaled here.
  void copy_q_heads_tile(scalar_t* __restrict__ src,
                         float* __restrict__ q_buffer, const int32_t q_num,
                         const int32_t q_heads_per_kv,
                         const int64_t q_num_stride,
                         const int64_t q_head_stride, float scale) {
    static_assert(head_dim % 16 == 0);
    constexpr int32_t unroll_size = head_dim / 16;
    using load_vec_t = typename VecTypeTrait<scalar_t>::vec_t;

    const float effective_scale = fp8_kv ? 1.0f : scale;
    vec_op::FP32Vec16 scale_vec(effective_scale);
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
      kv_cache_t* __restrict__ key_cache, kv_cache_t* __restrict__ value_cache,
      const int64_t* __restrict__ slot_mapping, const int64_t token_num,
      const int64_t key_token_num_stride, const int64_t value_token_num_stride,
      const int64_t head_num, const int64_t key_head_num_stride,
      const int64_t value_head_num_stride, const int64_t num_blocks,
      const int64_t num_blocks_stride, const int64_t cache_head_num_stride,
      const int64_t block_size, const int64_t block_size_stride,
      const float k_inv = 0.0f, const float v_inv = 0.0f) {
    if constexpr (fp8_kv) {
      constexpr auto qfn = select_fp8_quant_fn<kv_cache_t>();
      reshape_and_cache_fp8_vec_impl<scalar_t, qfn>(
          key, value, reinterpret_cast<uint8_t*>(key_cache),
          reinterpret_cast<uint8_t*>(value_cache), slot_mapping, token_num,
          head_num, head_dim, block_size, key_token_num_stride,
          key_head_num_stride, value_token_num_stride, value_head_num_stride,
          num_blocks_stride, cache_head_num_stride, num_blocks_stride,
          cache_head_num_stride, k_inv, v_inv);
      return;
    }

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
              reinterpret_cast<scalar_t*>(key_cache) +
              block_idx * num_blocks_stride + head_idx * cache_head_num_stride +
              block_offset;

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
              reinterpret_cast<scalar_t*>(value_cache) +
              block_idx * num_blocks_stride + head_idx * cache_head_num_stride +
              block_offset * head_dim;
          std::memcpy(value_cache_start_ptr, value_start_ptr,
                      sizeof(scalar_t) * head_dim);
        }
      }
    }
  }
};

}  // namespace cpu_attention

#endif
