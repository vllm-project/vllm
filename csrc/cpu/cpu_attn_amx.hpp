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
// q_buffer_t: type of A (Q/P) tile; kv_cache_t: type of B (K/V cache) tile.
// When q_buffer_t == kv_cache_t: plain BF16/FP16 computation.
// When q_buffer_t != kv_cache_t: dequant KV first, then BF16/FP16 computation.
template <typename q_buffer_t, typename kv_cache_t>
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
class TileGemm224<c10::BFloat16, c10::BFloat16> {
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
// q_buffer_t: type of A (Q/P) tile; kv_cache_t: type of B (K/V cache) tile.
template <typename q_buffer_t, typename kv_cache_t>
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
class TileGemm122<c10::BFloat16, c10::BFloat16> {
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
// Shared FP8 base for TileGemm224 (2-2-4 pattern).
// E4M3 and E5M2 differ only in the BF16Vec32 dequant tag; everything else is
// identical, so both specialisations inherit from this base.
template <typename fp8_t>
class TileGemm224FP8Base {
  static_assert(std::is_same_v<fp8_t, c10::Float8_e4m3fn> ||
                    std::is_same_v<fp8_t, c10::Float8_e5m2>,
                "fp8_t must be Float8_e4m3fn or Float8_e5m2");

  FORCE_INLINE static void deq_tile(const uint8_t* src, c10::BFloat16* dst) {
    for (int r = 0; r < AMX_TILE_ROW_NUM; ++r) {
      if constexpr (std::is_same_v<fp8_t, c10::Float8_e4m3fn>) {
        vec_op::BF16Vec32(src + r * 32, vec_op::fp8_bf16_e4m3_tag{})
            .save(dst + r * 32);
      } else {
        vec_op::BF16Vec32(src + r * 32, vec_op::fp8_bf16_e5m2_tag{})
            .save(dst + r * 32);
      }
    }
  }

 public:
  template <AttentionGemmPhase phase, int32_t k_size>
  FORCE_INLINE static void gemm(
      const int32_t m_size, c10::BFloat16* __restrict__ a_tile,
      fp8_t* __restrict__ b_tile, float* __restrict__ c_tile, const int64_t lda,
      const int64_t ldb, const int64_t ldc, const int32_t block_size,
      const int32_t dynamic_k_size, const bool accum_c) {
    const int32_t k_times = dynamic_k_size / 32;
    constexpr int64_t fp8_tile_elems = AMX_TILE_BYTES / sizeof(c10::BFloat16);

    c10::BFloat16* __restrict__ a_tile_0 = a_tile;
    c10::BFloat16* __restrict__ a_tile_1 = a_tile + lda * AMX_TILE_ROW_NUM;
    const int64_t a_tile_stride = [&]() {
      if constexpr (phase == AttentionGemmPhase::QK) {
        return AMX_TILE_ROW_BYTES;
      } else if constexpr (phase == AttentionGemmPhase::PV) {
        return lda * static_cast<int64_t>(sizeof(c10::BFloat16));
      } else {
        TORCH_CHECK(false, "Unreachable");
      }
    }();

    const uint8_t* __restrict__ b_tile_2 =
        reinterpret_cast<const uint8_t*>(b_tile);
    const uint8_t* __restrict__ b_tile_3 = [&]() {
      if constexpr (phase == AttentionGemmPhase::QK) {
        return reinterpret_cast<const uint8_t*>(b_tile) +
               (k_size * AMX_TILE_ROW_BYTES / 4);
      } else if constexpr (phase == AttentionGemmPhase::PV) {
        return reinterpret_cast<const uint8_t*>(b_tile) +
               (block_size * AMX_TILE_ROW_BYTES / 4);
      } else {
        TORCH_CHECK(false, "Unreachable");
      }
    }();
    const int32_t b_tile_stride = AMX_TILE_ROW_BYTES;

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
      alignas(64) c10::BFloat16 scratch_2[fp8_tile_elems];
      alignas(64) c10::BFloat16 scratch_3[fp8_tile_elems];
      deq_tile(b_tile_2, scratch_2);
      deq_tile(b_tile_3, scratch_3);

      _tile_loadd(0, a_tile_0, a_tile_stride);
      _tile_stream_loadd(2, scratch_2, b_tile_stride);
      _tile_dpbf16ps(4, 0, 2);
      _tile_stream_loadd(3, scratch_3, b_tile_stride);
      _tile_dpbf16ps(5, 0, 3);
      _tile_loadd(1, a_tile_1, a_tile_stride);
      _tile_dpbf16ps(6, 1, 2);
      _tile_dpbf16ps(7, 1, 3);

      if constexpr (phase == AttentionGemmPhase::QK) {
        a_tile_0 += AMX_TILE_BYTES / sizeof(c10::BFloat16);
        a_tile_1 += AMX_TILE_BYTES / sizeof(c10::BFloat16);
      } else if constexpr (phase == AttentionGemmPhase::PV) {
        a_tile_0 += AMX_TILE_ROW_BYTES / sizeof(c10::BFloat16);
        a_tile_1 += AMX_TILE_ROW_BYTES / sizeof(c10::BFloat16);
      } else {
        TORCH_CHECK(false, "Unreachable");
      }
      b_tile_2 += fp8_tile_elems;
      b_tile_3 += fp8_tile_elems;
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

template <>
class TileGemm224<c10::BFloat16, c10::Float8_e4m3fn>
    : public TileGemm224FP8Base<c10::Float8_e4m3fn> {};

template <>
class TileGemm224<c10::BFloat16, c10::Float8_e5m2>
    : public TileGemm224FP8Base<c10::Float8_e5m2> {};

// Shared FP8 base for TileGemm122 (1-2-2 pattern).
// E4M3 and E5M2 differ only in the BF16Vec32 dequant tag.
template <typename fp8_t>
class TileGemm122FP8Base {
  static_assert(std::is_same_v<fp8_t, c10::Float8_e4m3fn> ||
                    std::is_same_v<fp8_t, c10::Float8_e5m2>,
                "fp8_t must be Float8_e4m3fn or Float8_e5m2");

  FORCE_INLINE static void deq_tile(const uint8_t* src, c10::BFloat16* dst) {
    for (int r = 0; r < AMX_TILE_ROW_NUM; ++r) {
      if constexpr (std::is_same_v<fp8_t, c10::Float8_e4m3fn>) {
        vec_op::BF16Vec32(src + r * 32, vec_op::fp8_bf16_e4m3_tag{})
            .save(dst + r * 32);
      } else {
        vec_op::BF16Vec32(src + r * 32, vec_op::fp8_bf16_e5m2_tag{})
            .save(dst + r * 32);
      }
    }
  }

 public:
  template <AttentionGemmPhase phase, int32_t k_size>
  FORCE_INLINE static void gemm(
      const int32_t m_size, c10::BFloat16* __restrict__ a_tile,
      fp8_t* __restrict__ b_tile, float* __restrict__ c_tile, const int64_t lda,
      const int64_t ldb, const int64_t ldc, const int32_t block_size,
      const int32_t dynamic_k_size, const bool accum_c) {
    constexpr int64_t fp8_tile_elems = AMX_TILE_BYTES / sizeof(c10::BFloat16);

    c10::BFloat16* __restrict__ a_tile_0 = a_tile;
    c10::BFloat16* __restrict__ a_tile_1 = [&]() {
      if constexpr (phase == AttentionGemmPhase::QK) {
        return a_tile + AMX_TILE_BYTES / sizeof(c10::BFloat16);
      } else if constexpr (phase == AttentionGemmPhase::PV) {
        return a_tile + AMX_TILE_ROW_BYTES / sizeof(c10::BFloat16);
      } else {
        TORCH_CHECK(false, "Unreachable");
      }
    }();
    const int64_t a_tile_stride = [&]() {
      if constexpr (phase == AttentionGemmPhase::QK) {
        return AMX_TILE_ROW_BYTES;
      } else if constexpr (phase == AttentionGemmPhase::PV) {
        return lda * static_cast<int64_t>(sizeof(c10::BFloat16));
      } else {
        TORCH_CHECK(false, "Unreachable");
      }
    }();

    const uint8_t* __restrict__ b_tile_2 =
        reinterpret_cast<const uint8_t*>(b_tile);
    const uint8_t* __restrict__ b_tile_3 = [&]() {
      if constexpr (phase == AttentionGemmPhase::QK) {
        return reinterpret_cast<const uint8_t*>(b_tile) +
               (k_size * AMX_TILE_ROW_BYTES / 4);
      } else if constexpr (phase == AttentionGemmPhase::PV) {
        return reinterpret_cast<const uint8_t*>(b_tile) +
               (block_size * AMX_TILE_ROW_BYTES / 4);
      } else {
        TORCH_CHECK(false, "Unreachable");
      }
    }();
    const uint8_t* __restrict__ b_tile_4 = b_tile_2 + fp8_tile_elems;
    const uint8_t* __restrict__ b_tile_5 = b_tile_3 + fp8_tile_elems;
    const int64_t b_stride = AMX_TILE_ROW_BYTES;

    float* __restrict__ c_tile_6 = c_tile;
    float* __restrict__ c_tile_7 = c_tile + AMX_TILE_ROW_BYTES / sizeof(float);
    const int64_t c_stride = ldc * sizeof(float);

    const int32_t k_times = dynamic_k_size / 32;
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
      alignas(64) c10::BFloat16 scratch_2[fp8_tile_elems];
      alignas(64) c10::BFloat16 scratch_3[fp8_tile_elems];
      alignas(64) c10::BFloat16 scratch_4[fp8_tile_elems];
      alignas(64) c10::BFloat16 scratch_5[fp8_tile_elems];
      deq_tile(b_tile_2, scratch_2);
      deq_tile(b_tile_3, scratch_3);
      deq_tile(b_tile_4, scratch_4);
      deq_tile(b_tile_5, scratch_5);

      _tile_loadd(0, a_tile_0, a_tile_stride);
      _tile_stream_loadd(2, scratch_2, b_stride);
      _tile_dpbf16ps(6, 0, 2);
      _tile_stream_loadd(3, scratch_3, b_stride);
      _tile_dpbf16ps(7, 0, 3);
      _tile_loadd(1, a_tile_1, a_tile_stride);
      _tile_stream_loadd(4, scratch_4, b_stride);
      _tile_dpbf16ps(6, 1, 4);
      _tile_stream_loadd(5, scratch_5, b_stride);
      _tile_dpbf16ps(7, 1, 5);

      if constexpr (phase == AttentionGemmPhase::QK) {
        a_tile_0 += 2 * AMX_TILE_BYTES / sizeof(c10::BFloat16);
        a_tile_1 += 2 * AMX_TILE_BYTES / sizeof(c10::BFloat16);
      } else if constexpr (phase == AttentionGemmPhase::PV) {
        a_tile_0 += 2 * AMX_TILE_ROW_BYTES / sizeof(c10::BFloat16);
        a_tile_1 += 2 * AMX_TILE_ROW_BYTES / sizeof(c10::BFloat16);
      }
      b_tile_2 += 2 * fp8_tile_elems;
      b_tile_3 += 2 * fp8_tile_elems;
      b_tile_4 += 2 * fp8_tile_elems;
      b_tile_5 += 2 * fp8_tile_elems;
    }

    if (has_tail) {
      alignas(64) c10::BFloat16 scratch_2[fp8_tile_elems];
      alignas(64) c10::BFloat16 scratch_3[fp8_tile_elems];
      deq_tile(b_tile_2, scratch_2);
      deq_tile(b_tile_3, scratch_3);

      _tile_loadd(0, a_tile_0, a_tile_stride);
      _tile_stream_loadd(2, scratch_2, b_stride);
      _tile_dpbf16ps(6, 0, 2);
      _tile_stream_loadd(3, scratch_3, b_stride);
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

template <>
class TileGemm122<c10::BFloat16, c10::Float8_e4m3fn>
    : public TileGemm122FP8Base<c10::Float8_e4m3fn> {};

template <>
class TileGemm122<c10::BFloat16, c10::Float8_e5m2>
    : public TileGemm122FP8Base<c10::Float8_e5m2> {};

}  // namespace

template <typename scalar_t, int64_t head_dim, typename kv_cache_t_>
class AttentionImpl<ISA::AMX, scalar_t, head_dim, kv_cache_t_> {
  // fp8_kv: true when KV cache is FP8 (E4M3 or E5M2)
  static constexpr bool fp8_kv =
      std::is_same_v<kv_cache_t_, c10::Float8_e4m3fn> ||
      std::is_same_v<kv_cache_t_, c10::Float8_e5m2>;

 public:
  using query_t = scalar_t;
  using q_buffer_t = scalar_t;
  using kv_cache_t = kv_cache_t_;
  using logits_buffer_t = float;
  using partial_output_buffer_t = float;
  using prob_buffer_t = scalar_t;

  // FP8 path: AMX_TILE_ROW_BYTES / sizeof(uint8_t) = 64 would reject
  // block_size=32, so cap at 32. BF16 path: 64/2 = 32 anyway — same value.
  constexpr static int64_t BlockSizeAlignment = 32;
  constexpr static int64_t HeadDimAlignment =
      2 * (AMX_TILE_ROW_BYTES / 4);  // headdim num unit of PV phase
  constexpr static int64_t MaxQHeadNumPerIteration = 32;
  constexpr static int64_t HeadDim = head_dim;
  constexpr static ISA ISAType = ISA::AMX;
  // AMX always applies scale on logits (after QK GEMM). For FP8 the scale is
  // k_scale*bias; for non-FP8 it is the plain attention scale.
  constexpr static bool scale_on_logits = true;

  // FP8 scales — only meaningful when fp8_kv=true.
  float k_scale = 1.0f;
  float v_scale = 1.0f;

 public:
  AttentionImpl() : current_q_head_num_(0) {
    // Use all columns in AMX tiles
    vec_op::unroll_loop<int, 8>([&](int i) { amx_tile_config_.colsb[i] = 64; });
  }

  ~AttentionImpl() { _tile_release(); }

  void init_from_input(const AttentionInput* input) {
    if constexpr (fp8_kv) {
      k_scale = input->k_scale_fp8;
      v_scale = input->v_scale_fp8;
    }
  }

  // Returns the v_scale that final_output applies after PV accumulation.
  // For FP8 folds the exponent-rebiasing correction; non-FP8 returns 1.0f.
  float get_output_v_scale() const noexcept {
    if constexpr (fp8_kv) {
      constexpr float bias =
          std::is_same_v<kv_cache_t, c10::Float8_e5m2> ? 0x1p112f : 0x1p120f;
      return v_scale * bias;
    }
    return 1.0f;
  }

  template <template <typename tile_gemm_t> typename attention>
  FORCE_INLINE void execute_attention(DEFINE_CPU_ATTENTION_PARAMS) {
    if constexpr (fp8_kv) {
      // Fold k_scale and exponent-rebiasing correction into the attention
      // scale. scale_on_logits=true applies it to QK logits.
      // The correction factor (2^120 for E4M3, 2^112 for E5M2) compensates
      // for the unscaled BF16 representation from the direct FP8→BF16 dequant.
      const float bias =
          std::is_same_v<kv_cache_t, c10::Float8_e5m2> ? 0x1p112f : 0x1p120f;
      scale *= k_scale * bias;
    }
    if (q_head_num > AMX_TILE_ROW_NUM) {
      if (q_head_num != current_q_head_num_) {
        current_q_head_num_ = q_head_num;
        TileGemm224<q_buffer_t, kv_cache_t>::init_tile_config(q_head_num,
                                                              amx_tile_config_);
      }
      attention<TileGemm224<q_buffer_t, kv_cache_t>> attention_iteration;
      attention_iteration(CPU_ATTENTION_PARAMS);
    } else {
      if (q_head_num != current_q_head_num_) {
        current_q_head_num_ = q_head_num;
        TileGemm122<q_buffer_t, kv_cache_t>::init_tile_config(q_head_num,
                                                              amx_tile_config_);
      }
      attention<TileGemm122<q_buffer_t, kv_cache_t>> attention_iteration;
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

  void copy_q_heads_tile(scalar_t* __restrict__ src,
                         scalar_t* __restrict__ q_buffer, const int32_t q_num,
                         const int32_t q_heads_per_kv,
                         const int64_t q_num_stride,
                         const int64_t q_head_stride, const float scale) {
    constexpr int64_t bytes_per_head = head_dim * sizeof(scalar_t);
    static_assert(bytes_per_head % AMX_TILE_ROW_BYTES == 0);
    constexpr int64_t head_size_block_num = bytes_per_head / AMX_TILE_ROW_BYTES;
    constexpr int64_t head_elem_num_per_block =
        AMX_TILE_ROW_BYTES / sizeof(scalar_t);

    int32_t idx = 0;
    int8_t* __restrict__ q_buffer_iter = reinterpret_cast<int8_t*>(q_buffer);
    for (int32_t q_num_idx = 0; q_num_idx < q_num;
         ++q_num_idx, src += q_num_stride) {
      scalar_t* __restrict__ src_iter = src;
      for (int32_t q_head_idx = 0; q_head_idx < q_heads_per_kv;
           ++q_head_idx, src_iter += q_head_stride) {
        if constexpr (fp8_kv) {
          // FP8 path: k_scale is folded into the attention scale in
          // execute_attention. Copy Q as plain BF16 without scaling.
          vec_op::unroll_loop<int32_t, head_size_block_num>([&](int32_t blk) {
            const scalar_t* src_blk = src_iter + blk * head_elem_num_per_block;
            c10::BFloat16* dst_blk = reinterpret_cast<c10::BFloat16*>(
                q_buffer_iter + blk * AMX_TILE_BYTES);
            vec_op::BF16Vec16(src_blk).save(dst_blk);
            vec_op::BF16Vec16(src_blk + 16).save(dst_blk + 16);
          });
        } else {
          vec_op::unroll_loop<int32_t, head_size_block_num>(
              [&](int32_t head_size_block_idx) {
                vec_op::INT8Vec64 vec(src_iter + head_size_block_idx *
                                                     head_elem_num_per_block);
                vec.save(q_buffer_iter + head_size_block_idx * AMX_TILE_BYTES);
              });
        }

        ++idx;
        q_buffer_iter += AMX_TILE_ROW_BYTES;
        if ((idx & (AMX_TILE_ROW_NUM - 1)) == 0) {
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
    // For AMX B matrix, N always is 16
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

 protected:
  alignas(64) __tilecfg amx_tile_config_;
  int32_t current_q_head_num_;
};

}  // namespace cpu_attention

#endif
