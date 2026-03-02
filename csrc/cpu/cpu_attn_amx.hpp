#ifndef CPU_ATTN_AMX_HPP
#define CPU_ATTN_AMX_HPP

#include "cpu_attn_impl.hpp"

namespace cpu_attention {
namespace {
// AMX specific
constexpr static int64_t AMX_TILE_ROW_BYTES = 64;
constexpr static int64_t AMX_TILE_ROW_NUM = 16;
constexpr static int64_t AMX_TILE_BYTES = AMX_TILE_ROW_BYTES * AMX_TILE_ROW_NUM;

typedef struct __tile_config {
  uint8_t palette_id = 1;
  uint8_t start_row = 0;
  uint8_t reserved_0[14] = {0};
  uint16_t colsb[16] = {0};
  uint8_t rows[16] = {0};
} __tilecfg;

// 2-2-4 pattern, for 16 < m <= 32
// TILE 0, 1: load A matrix, row num should be 16, m - 16
// TILE 2, 3: load B matrix, row num should be 16
// TILE 4, 5, 6, 7: store results C matrix, row num should be 16, 16, m - 16, m
// - 16
template <typename kv_cache_t>
class TileGemm224 {
 public:
  template <AttentionGemmPhase phase, int32_t k_size>
  FORCE_INLINE static void gemm(const int32_t m_size, void* __restrict__ a_tile,
                                void* __restrict__ b_tile,
                                float* __restrict__ c_tile, const int64_t lda,
                                const int64_t ldb, const int64_t ldc,
                                const int32_t block_size,
                                const int32_t dynamic_k_size,
                                const bool accum_c) {
    TORCH_CHECK(false, "Unsupported kv cache type for TileGemm224");
  }

  FORCE_INLINE static void init_tile_config(int32_t m, __tilecfg& config) {
    TORCH_CHECK(false, "Unsupported kv cache type for TileGemm224");
  }
};

template <>
class TileGemm224<c10::BFloat16> {
 public:
  template <AttentionGemmPhase phase, int32_t k_size>
  FORCE_INLINE static void gemm(const int32_t m_size,
                                c10::BFloat16* __restrict__ a_tile,
                                c10::BFloat16* __restrict__ b_tile,
                                float* __restrict__ c_tile, const int64_t lda,
                                const int64_t ldb, const int64_t ldc,
                                const int32_t block_size,
                                const int32_t dynamic_k_size,
                                const bool accum_c) {
    const int32_t k_times =
        dynamic_k_size / (AMX_TILE_ROW_NUM * 4 / sizeof(c10::BFloat16));
    c10::BFloat16* __restrict__ a_tile_0 = a_tile;
    c10::BFloat16* __restrict__ a_tile_1 = a_tile + lda * AMX_TILE_ROW_NUM;
    const int64_t a_tile_stride = [&]() {
      if constexpr (phase == AttentionGemmPhase::QK) {
        // q_buffer is prepacked
        return AMX_TILE_ROW_BYTES;
      } else if constexpr (phase == AttentionGemmPhase::PV) {
        // logits_buffer is row-major
        return lda * sizeof(c10::BFloat16);
      } else {
        TORCH_CHECK(false, "Unreachable");
      }
    }();

    c10::BFloat16* __restrict__ b_tile_2 = b_tile;
    c10::BFloat16* __restrict__ b_tile_3 = [&]() {
      if constexpr (phase == AttentionGemmPhase::QK) {
        // k_cache is prepacked
        return b_tile + (k_size * AMX_TILE_ROW_BYTES / 4);
      } else if constexpr (phase == AttentionGemmPhase::PV) {
        // v_cache is prepacked
        return b_tile + (block_size * AMX_TILE_ROW_BYTES / 4);
      } else {
        TORCH_CHECK(false, "Unreachable");
      }
    }();
    // k_cache, v_cache are prepacked
    const int32_t b_tile_stride = AMX_TILE_ROW_BYTES;

    // logits_buffer, output_buffer are not prepacked
    float* __restrict__ c_tile_4 = c_tile;
    float* __restrict__ c_tile_5 =
        c_tile_4 + AMX_TILE_ROW_BYTES / sizeof(float);
    float* __restrict__ c_tile_6 = c_tile + AMX_TILE_ROW_NUM * ldc;
    float* __restrict__ c_tile_7 =
        c_tile_6 + AMX_TILE_ROW_BYTES / sizeof(float);
    const int32_t c_tile_stride = ldc * sizeof(float);

    if (accum_c) {
      _tile_loadd(4, c_tile_4, c_tile_stride);
      _tile_loadd(5, c_tile_5, c_tile_stride);
      _tile_loadd(6, c_tile_6, c_tile_stride);
      _tile_loadd(7, c_tile_7, c_tile_stride);
    } else {
      _tile_zero(4);
      _tile_zero(5);
      _tile_zero(6);
      _tile_zero(7);
    }

    for (int32_t k = 0; k < k_times; ++k) {
      _tile_loadd(0, a_tile_0, a_tile_stride);
      _tile_stream_loadd(2, b_tile_2, b_tile_stride);
      _tile_dpbf16ps(4, 0, 2);
      _tile_stream_loadd(3, b_tile_3, b_tile_stride);
      _tile_dpbf16ps(5, 0, 3);
      _tile_loadd(1, a_tile_1, a_tile_stride);
      _tile_dpbf16ps(6, 1, 2);
      _tile_dpbf16ps(7, 1, 3);

      // update ptrs
      if constexpr (phase == AttentionGemmPhase::QK) {
        // Q buffer is prepacked
        a_tile_0 += AMX_TILE_BYTES / sizeof(c10::BFloat16);
        a_tile_1 += AMX_TILE_BYTES / sizeof(c10::BFloat16);
      } else if constexpr (phase == AttentionGemmPhase::PV) {
        // P buffer is not prepacked
        a_tile_0 += AMX_TILE_ROW_BYTES / sizeof(c10::BFloat16);
        a_tile_1 += AMX_TILE_ROW_BYTES / sizeof(c10::BFloat16);
      } else {
        TORCH_CHECK(false, "Unreachable");
      }
      b_tile_2 += AMX_TILE_BYTES / sizeof(c10::BFloat16);
      b_tile_3 += AMX_TILE_BYTES / sizeof(c10::BFloat16);
    }

    _tile_stored(4, c_tile_4, c_tile_stride);
    _tile_stored(5, c_tile_5, c_tile_stride);
    _tile_stored(6, c_tile_6, c_tile_stride);
    _tile_stored(7, c_tile_7, c_tile_stride);
  }

  FORCE_INLINE static void init_tile_config(int32_t m, __tilecfg& config) {
    const int32_t m_0 = AMX_TILE_ROW_NUM;
    const int32_t m_1 = m - AMX_TILE_ROW_NUM;
    config.rows[0] = m_0;
    config.rows[1] = m_1;
    config.rows[2] = AMX_TILE_ROW_NUM;
    config.rows[3] = AMX_TILE_ROW_NUM;
    config.rows[4] = m_0;
    config.rows[5] = m_0;
    config.rows[6] = m_1;
    config.rows[7] = m_1;
    _tile_loadconfig(&config);
  }
};

// 1-2-2 pattern, for 0 < m <= 16
// TILE 0, (1): load A matrix, use extra 1 tile for prefetch, row num should be
// m, m
// TILE 2, 3, (4, 5): load B matrix, use extra 2 tiles for prefetch, row
// num should be 16
// TILE 6, 7, (6, 7): store results C matrix, row num should be
// m
template <typename kv_cache_t>
class TileGemm122 {
 public:
  template <AttentionGemmPhase phase, int32_t k_size>
  FORCE_INLINE static void gemm(const int32_t m_size, void* __restrict__ a_tile,
                                void* __restrict__ b_tile,
                                float* __restrict__ c_tile, const int64_t lda,
                                const int64_t ldb, const int64_t ldc,
                                const int32_t block_size,
                                const int32_t dynamic_k_size,
                                const bool accum_c) {
    TORCH_CHECK(false, "Unsupported kv cache type for TileGemm122");
  }

  FORCE_INLINE static void init_tile_config(int32_t m, __tilecfg& config) {
    TORCH_CHECK(false, "Unsupported kv cache type for TileGemm122");
  }
};

template <>
class TileGemm122<c10::BFloat16> {
 public:
  template <AttentionGemmPhase phase, int32_t k_size>
  FORCE_INLINE static void gemm(const int32_t m_size,
                                c10::BFloat16* __restrict__ a_tile,
                                c10::BFloat16* __restrict__ b_tile,
                                float* __restrict__ c_tile, const int64_t lda,
                                const int64_t ldb, const int64_t ldc,
                                const int32_t block_size,
                                const int32_t dynamic_k_size,
                                const bool accum_c) {
    c10::BFloat16* __restrict__ a_tile_0 = a_tile;
    c10::BFloat16* __restrict__ a_tile_1 = [&]() {
      if constexpr (phase == AttentionGemmPhase::QK) {
        // q_buffer is prepacked
        return a_tile + AMX_TILE_BYTES / sizeof(c10::BFloat16);
      } else if constexpr (phase == AttentionGemmPhase::PV) {
        // logits_buffer is row-major
        return a_tile + AMX_TILE_ROW_BYTES / sizeof(c10::BFloat16);
      } else {
        TORCH_CHECK(false, "Unreachable");
      }
    }();
    const int64_t a_tile_stride = [&]() {
      if constexpr (phase == AttentionGemmPhase::QK) {
        // q_buffer is prepacked
        return AMX_TILE_ROW_BYTES;
      } else if constexpr (phase == AttentionGemmPhase::PV) {
        // logits_buffer is row-major
        return lda * sizeof(c10::BFloat16);
      } else {
        TORCH_CHECK(false, "Unreachable");
      }
    }();

    c10::BFloat16* __restrict__ b_tile_2 = b_tile;
    c10::BFloat16* __restrict__ b_tile_3 = [&]() {
      if constexpr (phase == AttentionGemmPhase::QK) {
        // k_cache is prepacked
        return b_tile + (k_size * AMX_TILE_ROW_BYTES / 4);
      } else if constexpr (phase == AttentionGemmPhase::PV) {
        // v_cache is prepacked
        return b_tile + (block_size * AMX_TILE_ROW_BYTES / 4);
      } else {
        TORCH_CHECK(false, "Unreachable");
      }
    }();
    c10::BFloat16* __restrict__ b_tile_4 =
        b_tile_2 + AMX_TILE_BYTES / sizeof(c10::BFloat16);
    c10::BFloat16* __restrict__ b_tile_5 =
        b_tile_3 + AMX_TILE_BYTES / sizeof(c10::BFloat16);
    int64_t b_stride = AMX_TILE_ROW_BYTES;

    float* __restrict__ c_tile_6 = c_tile;
    float* __restrict__ c_tile_7 = c_tile + AMX_TILE_ROW_BYTES / sizeof(float);
    int64_t c_stride = ldc * sizeof(float);

    const int32_t k_times =
        dynamic_k_size / (AMX_TILE_ROW_NUM * 4 / sizeof(c10::BFloat16));
    const int32_t k_group_times = k_times / 2;
    const bool has_tail = (k_times % 2 == 1);

    if (accum_c) {
      _tile_loadd(6, c_tile_6, c_stride);
      _tile_loadd(7, c_tile_7, c_stride);
    } else {
      _tile_zero(6);
      _tile_zero(7);
    }

    for (int32_t k = 0; k < k_group_times; ++k) {
      _tile_loadd(0, a_tile_0, a_tile_stride);
      _tile_stream_loadd(2, b_tile_2, b_stride);
      _tile_dpbf16ps(6, 0, 2);
      _tile_stream_loadd(3, b_tile_3, b_stride);
      _tile_dpbf16ps(7, 0, 3);
      _tile_loadd(1, a_tile_1, a_tile_stride);
      _tile_stream_loadd(4, b_tile_4, b_stride);
      _tile_dpbf16ps(6, 1, 4);
      _tile_stream_loadd(5, b_tile_5, b_stride);
      _tile_dpbf16ps(7, 1, 5);

      // update ptrs
      if constexpr (phase == AttentionGemmPhase::QK) {
        // Q buffer is prepacked
        a_tile_0 += 2 * AMX_TILE_BYTES / sizeof(c10::BFloat16);
        a_tile_1 += 2 * AMX_TILE_BYTES / sizeof(c10::BFloat16);
      } else if constexpr (phase == AttentionGemmPhase::PV) {
        // P buffer is not prepacked
        a_tile_0 += 2 * AMX_TILE_ROW_BYTES / sizeof(c10::BFloat16);
        a_tile_1 += 2 * AMX_TILE_ROW_BYTES / sizeof(c10::BFloat16);
      }
      b_tile_2 += 2 * AMX_TILE_BYTES / sizeof(c10::BFloat16);
      b_tile_3 += 2 * AMX_TILE_BYTES / sizeof(c10::BFloat16);
      b_tile_4 += 2 * AMX_TILE_BYTES / sizeof(c10::BFloat16);
      b_tile_5 += 2 * AMX_TILE_BYTES / sizeof(c10::BFloat16);
    }

    if (has_tail) {
      _tile_loadd(0, a_tile_0, a_tile_stride);
      _tile_stream_loadd(2, b_tile_2, b_stride);
      _tile_dpbf16ps(6, 0, 2);
      _tile_stream_loadd(3, b_tile_3, b_stride);
      _tile_dpbf16ps(7, 0, 3);
    }

    _tile_stored(6, c_tile_6, c_stride);
    _tile_stored(7, c_tile_7, c_stride);
  }

  FORCE_INLINE static void init_tile_config(int32_t m, __tilecfg& config) {
    config.rows[0] = m;
    config.rows[1] = m;
    config.rows[2] = AMX_TILE_ROW_NUM;
    config.rows[3] = AMX_TILE_ROW_NUM;
    config.rows[4] = AMX_TILE_ROW_NUM;
    config.rows[5] = AMX_TILE_ROW_NUM;
    config.rows[6] = m;
    config.rows[7] = m;
    _tile_loadconfig(&config);
  }
};
}  // namespace

template <typename scalar_t, int64_t head_dim>
class AttentionImpl<ISA::AMX, scalar_t, head_dim> {
 public:
  using query_t = scalar_t;
  using q_buffer_t = scalar_t;
  using kv_cache_t = scalar_t;
  using logits_buffer_t = float;
  using partial_output_buffer_t = float;
  using prob_buffer_t = scalar_t;

  constexpr static int64_t BlockSizeAlignment =
      AMX_TILE_ROW_BYTES /
      sizeof(kv_cache_t);  // KV token num unit of QK and PV phases
  constexpr static int64_t HeadDimAlignment =
      2 * (AMX_TILE_ROW_BYTES / 4);  // headdim num unit of PV phase
  constexpr static int64_t MaxQHeadNumPerIteration = 32;
  constexpr static int64_t HeadDim = head_dim;
  constexpr static ISA ISAType = ISA::AMX;
  constexpr static bool scale_on_logits = true;

 public:
  AttentionImpl() : current_q_head_num_(0) {
    // Use all columns in AMX tiles
    vec_op::unroll_loop<int, 8>([&](int i) { amx_tile_config_.colsb[i] = 64; });
  }

  ~AttentionImpl() { _tile_release(); }

  template <template <typename tile_gemm_t> typename attention>
  FORCE_INLINE void execute_attention(DEFINE_CPU_ATTENTION_PARAMS) {
    if (q_head_num > AMX_TILE_ROW_NUM) {
      if (q_head_num != current_q_head_num_) {
        current_q_head_num_ = q_head_num;
        TileGemm224<kv_cache_t>::init_tile_config(q_head_num, amx_tile_config_);
      }
      attention<TileGemm224<kv_cache_t>> attention_iteration;
      attention_iteration(CPU_ATTENTION_PARAMS);
    } else {
      if (q_head_num != current_q_head_num_) {
        current_q_head_num_ = q_head_num;
        TileGemm122<kv_cache_t>::init_tile_config(q_head_num, amx_tile_config_);
      }
      attention<TileGemm122<kv_cache_t>> attention_iteration;
      attention_iteration(CPU_ATTENTION_PARAMS);
    }
  }

  // k_cache_token_group_stride: stride of K cache when move to next
  // BlockSizeAlignment tokens in a block
  constexpr static int64_t k_cache_token_group_stride(
      const int32_t block_size) {
    return BlockSizeAlignment * head_dim;
  }

  // v_cache_token_group_stride: stride of V cache when move to next
  // BlockSizeAlignment tokens in a block
  constexpr static int64_t v_cache_token_group_stride(
      const int32_t block_size) {
    return BlockSizeAlignment * (AMX_TILE_ROW_BYTES / 4);
  }

  // v_cache_head_group_stride: stride of V cache when move to next
  // HeadDimAlignment head dims in a block
  constexpr static int64_t v_cache_head_group_stride(const int32_t block_size) {
    return block_size * HeadDimAlignment;
  }

  static void copy_q_heads_tile(
      scalar_t* __restrict__ src,  // [q_num, q_heads_per_kv, head_size]
      scalar_t* __restrict__ q_buffer, const int32_t q_num,
      const int32_t q_heads_per_kv, const int64_t q_num_stride,
      const int64_t q_head_stride, const float scale) {
    constexpr int64_t bytes_per_head = head_dim * sizeof(scalar_t);
    static_assert(bytes_per_head % AMX_TILE_ROW_BYTES == 0);
    constexpr int64_t head_size_block_num = bytes_per_head / AMX_TILE_ROW_BYTES;
    constexpr int64_t head_elem_num_pre_block =
        AMX_TILE_ROW_BYTES / sizeof(scalar_t);

    int32_t idx = 0;
    int8_t* __restrict__ q_buffer_iter = reinterpret_cast<int8_t*>(q_buffer);
    for (int32_t q_num_idx = 0; q_num_idx < q_num;
         ++q_num_idx, src += q_num_stride) {
      scalar_t* __restrict__ src_iter = src;
      for (int32_t q_head_idx = 0; q_head_idx < q_heads_per_kv;
           ++q_head_idx, src_iter += q_head_stride) {
        vec_op::unroll_loop<int32_t, head_size_block_num>(
            [&](int32_t head_size_block_idx) {
              // Use INT8Vec64 for 64 bytes block
              vec_op::INT8Vec64 vec(src_iter + head_size_block_idx *
                                                   head_elem_num_pre_block);
              vec.save(q_buffer_iter + head_size_block_idx * AMX_TILE_BYTES);
            });

        ++idx;
        q_buffer_iter += AMX_TILE_ROW_BYTES;
        if ((idx & (AMX_TILE_ROW_NUM - 1)) == 0) {
          // head is in another amx tile
          q_buffer_iter -= AMX_TILE_ROW_NUM * AMX_TILE_ROW_BYTES;
          q_buffer_iter += head_size_block_num * AMX_TILE_BYTES;
        }
      }
    }
  }

  // reshape KV to AMX friendly layout
  static void reshape_and_cache(
      const scalar_t* __restrict__ key, const scalar_t* __restrict__ value,
      scalar_t* __restrict__ key_cache, scalar_t* __restrict__ value_cache,
      const int64_t* __restrict__ slot_mapping, const int64_t token_num,
      const int64_t key_token_num_stride, const int64_t value_token_num_stride,
      const int64_t head_num, const int64_t key_head_num_stride,
      const int64_t value_head_num_stride, const int64_t num_blocks,
      const int64_t num_blocks_stride, const int64_t cache_head_num_stride,
      const int64_t block_size, const int64_t block_size_stride) {
    // For AMX 2D tiles, size of each line is 64 bytes
    constexpr int64_t amx_tile_row_size = AMX_TILE_ROW_BYTES;
    // For AMX B martix, N always is 16
    constexpr int64_t amx_b_tile_n_size = AMX_TILE_ROW_BYTES / 4;
    constexpr int64_t amx_b_tile_k_size = amx_tile_row_size / sizeof(scalar_t);
    // For now suppose block_size is divisible by amx_tile_column_num
    TORCH_CHECK_EQ(block_size % amx_b_tile_k_size, 0);

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
          // Write Key
          // Head elements should be packed as quand-words and stored in token
          // groups with (quadword_stride/4) tokens
          constexpr int64_t token_num_per_group = amx_tile_row_size / 4;
          static_assert(head_dim % (4 / sizeof(scalar_t)) == 0);
          constexpr int64_t quadword_num = head_dim / (4 / sizeof(scalar_t));
          const int32_t* key_start_quadword_ptr =
              reinterpret_cast<const int32_t*>(
                  key + token_idx * key_token_num_stride +
                  head_idx * key_head_num_stride);
          const int64_t group_idx = block_offset / token_num_per_group;
          const int64_t group_offset = block_offset % token_num_per_group;
          constexpr int64_t quadword_num_per_group =
              token_num_per_group * quadword_num;
          int32_t* key_cache_start_ptr =
              reinterpret_cast<int32_t*>(key_cache +
                                         block_idx * num_blocks_stride +
                                         head_idx * cache_head_num_stride) +
              group_idx * quadword_num_per_group + group_offset;

#pragma GCC unroll 8
          for (int64_t i = 0, j = 0; j < quadword_num;
               i += token_num_per_group, ++j) {
            key_cache_start_ptr[i] = key_start_quadword_ptr[j];
          }
        }
        {
          // Write Value
          // Different from Key, block_size dimension is packed rather than
          // head_size dimension block_size dimension is packed as quand-words;
          constexpr int64_t token_num_per_sub_group = 4 / sizeof(scalar_t);
          const int64_t token_num_per_group = block_size;
          constexpr int64_t head_elems_per_group = amx_b_tile_n_size;
          const int64_t group_size = token_num_per_group * head_elems_per_group;
          // For now suppose head_dim is divisible by amx_b_tile_n_size
          static_assert(head_dim % head_elems_per_group == 0);
          constexpr int64_t group_num = head_dim / head_elems_per_group;
          const int64_t sub_group_idx = block_offset / token_num_per_sub_group;
          const int64_t sub_group_offset =
              block_offset % token_num_per_sub_group;

          const scalar_t* value_start_ptr = value +
                                            token_idx * value_token_num_stride +
                                            head_idx * value_head_num_stride;
          scalar_t* value_cache_start_ptr =
              value_cache + block_idx * num_blocks_stride +
              head_idx * cache_head_num_stride +
              sub_group_idx * token_num_per_sub_group * amx_b_tile_n_size +
              sub_group_offset;

          for (int64_t i = 0; i < group_num; ++i) {
#pragma GCC unroll head_elems_per_group
            for (int64_t j = 0, k = 0; j < head_elems_per_group;
                 ++j, k += token_num_per_sub_group) {
              value_cache_start_ptr[k] = value_start_ptr[j];
            }
            value_start_ptr += head_elems_per_group;
            value_cache_start_ptr += group_size;
          }
        }
      }
    }
  }

 private:
  alignas(64) __tilecfg amx_tile_config_;
  int32_t current_q_head_num_;
};
}  // namespace cpu_attention

#endif
