// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// AMX_FP8 attention implementation for Intel Diamond Rapids (DMR).
//
// Architecture:
//   QK phase: native FP8×FP8 MMA via _tile_dpfp8ps / _tile_dpbf8ps
//             Q is quantized from BF16 to FP8-E4M3 in copy_q_heads_tile().
//             K cache is stored in the native AMX-FP8 layout (1 byte/element).
//   PV phase: reuses the BF16 MMA path (_tile_dpbf16ps) with FP8 V dequanted
//             to BF16 on-the-fly (same as AMX_BF16+FP8KV in cpu_attn_amx.hpp).
//             V cache uses the same halfword-packed layout as commit 22524f7a92.
//
// Scale bookkeeping for QK:
//   true_score = Q_fp32 · K_fp32
//              = (Q_fp8 * q_scale) · (K_fp8 * k_scale)
//              = dot(Q_fp8, K_fp8) * q_scale * k_scale
//   execute_attention() applies: scale *= q_dynamic_scale * k_scale
//   where q_dynamic_scale is computed per-tile in copy_q_heads_tile().
//
// Scale bookkeeping for PV (same as AMX_BF16+FP8KV):
//   V_bf16 = V_fp8 * v_scale * bias   (bias = 2^120 for E4M3, 2^112 for E5M2)
//   output corrected via output_v_scale in get_output_v_scale().

#ifndef CPU_ATTN_AMX_FP8_HPP
#define CPU_ATTN_AMX_FP8_HPP

#ifdef CPU_CAPABILITY_AMXFP8

#include "cpu_attn_amx.hpp"   // pulls in TileGemm224/TileGemm122 BF16 specialisations
#include "cpu_attn_fp8.hpp"
#include "cpu_attn_impl.hpp"
#include "cpu_types.hpp"

#include <amxfp8intrin.h>  // requires GCC 14+ / Clang 17+, -mamx-fp8

namespace cpu_attention {

// ---------------------------------------------------------------------------
// Native-FP8 K cache layout reshape helper.
//
// Layout per 16-token group (token_num_per_group = AMX_TILE_ROW_NUM = 16):
//   k-slice s  (s = 0 .. head_dim/64 - 1):
//     row t (t = 0..15): FP8 key[token_group_base + t, s*64 .. (s+1)*64 - 1]
//   One k-slice  = 16 rows × 64 bytes = 1024 bytes = 1 AMX FP8 tile.
//   Full group   = (head_dim / 64) slices × 1024 bytes.
//
// For head_dim=128: group = 2 × 1024 = 2048 bytes.
// This matches k_cache_token_group_stride = BlockSizeAlignment × head_dim
//            = 32 × 128 = 4096 uint8_t (for 32-token aligned groups).
// ---------------------------------------------------------------------------
template <typename scalar_t, uint8_t (*quant_fn)(float, float)>
inline void reshape_and_cache_k_amx_fp8_impl(
    const scalar_t* key_ptr, uint8_t* key_cache_ptr,
    const int64_t* slot_ptr, int64_t token_num, int64_t head_num,
    int64_t head_dim, int64_t block_size, int64_t k_stride0, int64_t k_stride1,
    int64_t kc_stride0, int64_t kc_stride1, float k_inv) {
  constexpr int64_t token_num_per_group = AMX_TILE_ROW_NUM;  // 16
  constexpr int64_t fp8_per_row = AMX_TILE_ROW_BYTES;        // 64 FP8/row
  const int64_t slice_num = head_dim / fp8_per_row;
  // Bytes per 16-token group = slice_num * (16 * 64) = head_dim * 16
  const int64_t group_bytes = token_num_per_group * head_dim;

#pragma omp parallel for collapse(2) schedule(static)
  for (int64_t tok = 0; tok < token_num; ++tok) {
    for (int64_t h = 0; h < head_num; ++h) {
      const int64_t slot = slot_ptr[tok];
      if (slot < 0) continue;
      const int64_t block_idx = slot / block_size;
      const int64_t block_offset = slot % block_size;
      const int64_t group_idx = block_offset / token_num_per_group;
      const int64_t group_offset = block_offset % token_num_per_group;

      const scalar_t* ksrc = key_ptr + tok * k_stride0 + h * k_stride1;
      uint8_t* group_base = key_cache_ptr + block_idx * kc_stride0
                            + h * kc_stride1 + group_idx * group_bytes;

      // Each k-slice: 16 rows × 64 bytes.  Token group_offset occupies row
      // group_offset within each slice.
      for (int64_t s = 0; s < slice_num; ++s) {
        uint8_t* row_base = group_base + s * token_num_per_group * fp8_per_row
                            + group_offset * fp8_per_row;
        for (int64_t e = 0; e < fp8_per_row; ++e) {
          row_base[e] = quant_fn(static_cast<float>(ksrc[s * fp8_per_row + e]),
                                 k_inv);
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// TileGemm224<uint8_t, kv_cache_t>:  2-2-4 pattern for AMX_FP8.
//
// QK phase: native FP8 × FP8 MMA (_tile_dpfp8ps / _tile_dpbf8ps).
// PV phase: delegate to the BF16×FP8(dequant) path in TileGemm224<BF16,FP8>.
// ---------------------------------------------------------------------------
template <typename kv_cache_t>
class TileGemm224<uint8_t, kv_cache_t> {
  static_assert(std::is_same_v<kv_cache_t, c10::Float8_e4m3fn> ||
                    std::is_same_v<kv_cache_t, c10::Float8_e5m2>,
                "kv_cache_t must be Float8_e4m3fn or Float8_e5m2");

  // For PV we reuse the BF16 dequant path (V cache is halfword-packed).
  using pv_gemm_t = TileGemm224<c10::BFloat16, kv_cache_t>;

  // FP8 tile size in uint8_t elements (= AMX_TILE_BYTES, one per byte).
  static constexpr int64_t fp8_tile_elems = AMX_TILE_BYTES;  // 1024

 public:
  // ------------------------------------------------------------------
  // QK: k_times = head_dim / 64 (each FP8 tile covers 64 k-elements).
  // PV: delegate to BF16 specialisation (a_tile = BF16 probs).
  // ------------------------------------------------------------------
  template <AttentionGemmPhase phase, int32_t k_size>
  FORCE_INLINE static void gemm(const int32_t m_size, void* __restrict__ a_tile,
                                void* __restrict__ b_tile,
                                float* __restrict__ c_tile, const int64_t lda,
                                const int64_t ldb, const int64_t ldc,
                                const int32_t block_size,
                                const int32_t dynamic_k_size,
                                const bool accum_c) {
    if constexpr (phase == AttentionGemmPhase::PV) {
      // PV uses BF16 probs × FP8 V (dequant to BF16 on the fly).
      pv_gemm_t::template gemm<phase, k_size>(
          m_size, reinterpret_cast<c10::BFloat16*>(a_tile),
          reinterpret_cast<kv_cache_t*>(b_tile), c_tile, lda, ldb, ldc,
          block_size,
          dynamic_k_size, accum_c);
      return;
    }

    // -------- QK phase: native FP8 MMA --------
    // k_times: each FP8 AMX tile covers (AMX_TILE_ROW_NUM * 4) FP8 = 64
    //          inner-product elements.
    const int32_t k_times =
        dynamic_k_size / (AMX_TILE_ROW_NUM * 4 / sizeof(uint8_t));

    uint8_t* __restrict__ a_tile_0 = static_cast<uint8_t*>(a_tile);
    // Q buffer prepacked: tile_1 immediately follows tile_0 in memory.
    uint8_t* __restrict__ a_tile_1 = a_tile_0 + fp8_tile_elems;

    uint8_t* __restrict__ b_tile_2 = static_cast<uint8_t*>(b_tile);
    // K native layout: tile_3 is the second 16-token group,
    // at offset k_size * AMX_TILE_ROW_BYTES / 4 (uint8_t pointer units).
    uint8_t* __restrict__ b_tile_3 =
        b_tile_2 + (k_size * AMX_TILE_ROW_BYTES / 4);

    const int32_t b_tile_stride = AMX_TILE_ROW_BYTES;  // 64 bytes

    const int32_t m_0 =
      std::min(m_size, static_cast<int32_t>(AMX_TILE_ROW_NUM));
    const int32_t m_1 = m_size - m_0;
    float* __restrict__ c_tile_4 = c_tile;
    float* __restrict__ c_tile_5 = c_tile_4 + AMX_TILE_ROW_BYTES / sizeof(float);
    float* __restrict__ c_tile_6 = c_tile + AMX_TILE_ROW_NUM * ldc;
    float* __restrict__ c_tile_7 = c_tile_6 + AMX_TILE_ROW_BYTES / sizeof(float);
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
      _tile_loadd(0, a_tile_0, AMX_TILE_ROW_BYTES);
      _tile_stream_loadd(2, b_tile_2, b_tile_stride);
      if constexpr (std::is_same_v<kv_cache_t, c10::Float8_e4m3fn>) {
        _tile_dphf8ps(4, 0, 2);
      } else {
        _tile_dpbf8ps(4, 0, 2);
      }
      _tile_stream_loadd(3, b_tile_3, b_tile_stride);
      if constexpr (std::is_same_v<kv_cache_t, c10::Float8_e4m3fn>) {
        _tile_dphf8ps(5, 0, 3);
      } else {
        _tile_dpbf8ps(5, 0, 3);
      }
      _tile_loadd(1, a_tile_1, AMX_TILE_ROW_BYTES);
      if constexpr (std::is_same_v<kv_cache_t, c10::Float8_e4m3fn>) {
        _tile_dphf8ps(6, 1, 2);
        _tile_dphf8ps(7, 1, 3);
      } else {
        _tile_dpbf8ps(6, 1, 2);
        _tile_dpbf8ps(7, 1, 3);
      }

      // Advance Q buffer (prepacked): 2 FP8 tiles per k-step.
      a_tile_0 += 2 * fp8_tile_elems;
      a_tile_1 += 2 * fp8_tile_elems;
      // Advance K cache (native layout): 1 FP8 tile per k-step.
      b_tile_2 += fp8_tile_elems;
      b_tile_3 += fp8_tile_elems;
    }

    _tile_stored(4, c_tile_4, c_tile_stride);
    _tile_stored(5, c_tile_5, c_tile_stride);
    _tile_stored(6, c_tile_6, c_tile_stride);
    _tile_stored(7, c_tile_7, c_tile_stride);
  }

  FORCE_INLINE static void init_tile_config(int32_t m, __tilecfg& config) {
    // Tile rows/cols for FP8 tiles: same row count (16), 64 bytes/row.
    // For result tiles (C): same as BF16 (16 rows, 64 bytes of FP32).
    // We set up for both QK (FP8 A/B) and PV (BF16 A, FP8 B dequanted):
    // the PV path re-uses _tile_dpbf16ps so it needs BF16-sized tiles.
    // Configure for the union of both layouts (BF16 columns = 64 bytes/row).
    const int32_t m_0 =
      std::min(m, static_cast<int32_t>(AMX_TILE_ROW_NUM));
    const int32_t m_1 = m - m_0;
    // Use all 64 bytes per row for every tile (same as BF16 AMX).
    // This works for both FP8 (64 FP8/row) and BF16 (32 BF16/row = 64B).
    for (int i = 0; i < 8; ++i) config.colsb[i] = AMX_TILE_ROW_BYTES;
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

// ---------------------------------------------------------------------------
// TileGemm122<uint8_t, kv_cache_t>:  1-2-2 pattern for AMX_FP8.
// ---------------------------------------------------------------------------
template <typename kv_cache_t>
class TileGemm122<uint8_t, kv_cache_t> {
  static_assert(std::is_same_v<kv_cache_t, c10::Float8_e4m3fn> ||
                    std::is_same_v<kv_cache_t, c10::Float8_e5m2>,
                "kv_cache_t must be Float8_e4m3fn or Float8_e5m2");

  using pv_gemm_t = TileGemm122<c10::BFloat16, kv_cache_t>;
  static constexpr int64_t fp8_tile_elems = AMX_TILE_BYTES;

 public:
  template <AttentionGemmPhase phase, int32_t k_size>
  FORCE_INLINE static void gemm(const int32_t m_size, void* __restrict__ a_tile,
                                void* __restrict__ b_tile,
                                float* __restrict__ c_tile, const int64_t lda,
                                const int64_t ldb, const int64_t ldc,
                                const int32_t block_size,
                                const int32_t dynamic_k_size,
                                const bool accum_c) {
    if constexpr (phase == AttentionGemmPhase::PV) {
      pv_gemm_t::template gemm<phase, k_size>(
          m_size, reinterpret_cast<c10::BFloat16*>(a_tile),
          reinterpret_cast<kv_cache_t*>(b_tile), c_tile, lda, ldb, ldc,
          block_size,
          dynamic_k_size, accum_c);
      return;
    }

    // -------- QK phase: native FP8 MMA (1-2-2 pattern) --------
    const int32_t k_times =
        dynamic_k_size / (AMX_TILE_ROW_NUM * 4 / sizeof(uint8_t));
    const int32_t k_group_times = k_times / 2;
    const bool has_tail = (k_times % 2 == 1);

    uint8_t* __restrict__ a_tile_0 = static_cast<uint8_t*>(a_tile);
    // Prepacked Q: second group immediately follows first in memory.
    uint8_t* __restrict__ a_tile_1 = a_tile_0 + fp8_tile_elems;

    uint8_t* __restrict__ b_tile_2 = static_cast<uint8_t*>(b_tile);
    uint8_t* __restrict__ b_tile_3 =
        b_tile_2 + (k_size * AMX_TILE_ROW_BYTES / 4);
    uint8_t* __restrict__ b_tile_4 = b_tile_2 + fp8_tile_elems;
    uint8_t* __restrict__ b_tile_5 = b_tile_3 + fp8_tile_elems;

    const int32_t b_stride = AMX_TILE_ROW_BYTES;

    float* __restrict__ c_tile_6 = c_tile;
    float* __restrict__ c_tile_7 = c_tile + AMX_TILE_ROW_BYTES / sizeof(float);
    const int32_t c_stride = ldc * sizeof(float);

    if (accum_c) {
      _tile_loadd(6, c_tile_6, c_stride);
      _tile_loadd(7, c_tile_7, c_stride);
    } else {
      _tile_zero(6);
      _tile_zero(7);
    }

    for (int32_t k = 0; k < k_group_times; ++k) {
      _tile_loadd(0, a_tile_0, AMX_TILE_ROW_BYTES);
      _tile_stream_loadd(2, b_tile_2, b_stride);
      if constexpr (std::is_same_v<kv_cache_t, c10::Float8_e4m3fn>) {
        _tile_dphf8ps(6, 0, 2);
      } else {
        _tile_dpbf8ps(6, 0, 2);
      }
      _tile_stream_loadd(3, b_tile_3, b_stride);
      if constexpr (std::is_same_v<kv_cache_t, c10::Float8_e4m3fn>) {
        _tile_dphf8ps(7, 0, 3);
      } else {
        _tile_dpbf8ps(7, 0, 3);
      }
      _tile_loadd(1, a_tile_1, AMX_TILE_ROW_BYTES);
      _tile_stream_loadd(4, b_tile_4, b_stride);
      if constexpr (std::is_same_v<kv_cache_t, c10::Float8_e4m3fn>) {
        _tile_dphf8ps(6, 1, 4);
      } else {
        _tile_dpbf8ps(6, 1, 4);
      }
      _tile_stream_loadd(5, b_tile_5, b_stride);
      if constexpr (std::is_same_v<kv_cache_t, c10::Float8_e4m3fn>) {
        _tile_dphf8ps(7, 1, 5);
      } else {
        _tile_dpbf8ps(7, 1, 5);
      }

      a_tile_0 += 2 * fp8_tile_elems;
      a_tile_1 += 2 * fp8_tile_elems;
      b_tile_2 += 2 * fp8_tile_elems;
      b_tile_3 += 2 * fp8_tile_elems;
      b_tile_4 += 2 * fp8_tile_elems;
      b_tile_5 += 2 * fp8_tile_elems;
    }

    if (has_tail) {
      _tile_loadd(0, a_tile_0, AMX_TILE_ROW_BYTES);
      _tile_stream_loadd(2, b_tile_2, b_stride);
      if constexpr (std::is_same_v<kv_cache_t, c10::Float8_e4m3fn>) {
        _tile_dphf8ps(6, 0, 2);
      } else {
        _tile_dpbf8ps(6, 0, 2);
      }
      _tile_stream_loadd(3, b_tile_3, b_stride);
      if constexpr (std::is_same_v<kv_cache_t, c10::Float8_e4m3fn>) {
        _tile_dphf8ps(7, 0, 3);
      } else {
        _tile_dpbf8ps(7, 0, 3);
      }
    }

    _tile_stored(6, c_tile_6, c_stride);
    _tile_stored(7, c_tile_7, c_stride);
  }

  FORCE_INLINE static void init_tile_config(int32_t m, __tilecfg& config) {
    for (int i = 0; i < 8; ++i) config.colsb[i] = AMX_TILE_ROW_BYTES;
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

// ---------------------------------------------------------------------------
// AttentionImpl<ISA::AMX_FP8, scalar_t, head_dim, kv_cache_scalar_t>
//
// scalar_t     = BF16 (input query/output type)
// kv_cache_scalar_t = Float8_e4m3fn or Float8_e5m2
// ---------------------------------------------------------------------------
template <typename scalar_t, int64_t head_dim, typename kv_cache_scalar_t>
  requires(!std::is_same_v<scalar_t, c10::BFloat16> ||
           !(std::is_same_v<kv_cache_scalar_t, c10::Float8_e4m3fn> ||
             std::is_same_v<kv_cache_scalar_t, c10::Float8_e5m2>) ||
           (head_dim % 64 != 0))
class AttentionImpl<ISA::AMX_FP8, scalar_t, head_dim, kv_cache_scalar_t>
    : public AttentionImpl<ISA::AMX, scalar_t, head_dim, kv_cache_scalar_t> {
};

template <typename scalar_t, int64_t head_dim, typename kv_cache_scalar_t>
  requires(std::is_same_v<scalar_t, c10::BFloat16> &&
           (std::is_same_v<kv_cache_scalar_t, c10::Float8_e4m3fn> ||
            std::is_same_v<kv_cache_scalar_t, c10::Float8_e5m2>) &&
           (head_dim % 64 == 0))
class AttentionImpl<ISA::AMX_FP8, scalar_t, head_dim, kv_cache_scalar_t> {
 public:
  // Q buffer holds FP8-quantized query.
  using query_t = scalar_t;         // BF16 input
  using q_buffer_t = uint8_t;       // FP8 quantized, stored in Q buffer
  using kv_cache_t = kv_cache_scalar_t;
  using logits_buffer_t = float;
  using partial_output_buffer_t = float;
  using prob_buffer_t = scalar_t;   // BF16 softmax probs (PV phase)

  // Same block alignment as AMX_BF16: 32 tokens per group (2 AMX tiles).
  constexpr static int64_t BlockSizeAlignment = 32;
  // HeadDimAlignment for PV: same as AMX_BF16 (BF16 PV path).
  constexpr static int64_t HeadDimAlignment = 2 * (AMX_TILE_ROW_BYTES / 4);
  constexpr static int64_t MaxQHeadNumPerIteration = 32;
  constexpr static int64_t HeadDim = head_dim;
  constexpr static ISA ISAType = ISA::AMX_FP8;
  constexpr static bool scale_on_logits = true;

  float k_scale = 1.0f;
  float v_scale = 1.0f;
  // Dynamically computed per copy_q_heads_tile() call.
  float q_dynamic_scale = 1.0f;

 public:
  AttentionImpl() : current_q_head_num_(0) {
    vec_op::unroll_loop<int, 8>([&](int i) { amx_tile_config_.colsb[i] = 64; });
  }

  ~AttentionImpl() { _tile_release(); }

  void init_from_input(const AttentionInput* input) {
    k_scale = input->k_scale_fp8;
    v_scale = input->v_scale_fp8;
    // q_scale_fp8 from input is used as a hint; dynamic scale is recomputed
    // per-tile in copy_q_heads_tile().
  }

  // Correction factor for the PV output:
  // AMX FP8→BF16 dequant shifts exponent bias by (127 - FP8_bias).
  // E4M3 bias=7  → 2^120; E5M2 bias=15 → 2^112.
  float get_output_v_scale() const noexcept {
    constexpr float bias =
        std::is_same_v<kv_cache_t, c10::Float8_e5m2> ? 0x1p112f : 0x1p120f;
    return v_scale * bias;
  }

  template <template <typename tile_gemm_t> typename attention>
  FORCE_INLINE void execute_attention(DEFINE_CPU_ATTENTION_PARAMS) {
    // Apply QK scale correction: score_fp32 = dot(Q_fp8, K_fp8) * q_dyn * k
    scale *= q_dynamic_scale * k_scale;

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

  // K cache stride: same element-count formula as AMX_BF16.
  // (BlockSizeAlignment * head_dim) uint8_t elements = same byte count as
  // before since kv_cache_t is 1-byte FP8.
  constexpr static int64_t k_cache_token_group_stride(
      const int32_t /*block_size*/) {
    return BlockSizeAlignment * head_dim;
  }

  // V cache strides: same as AMX_BF16 with FP8 KV (halfword-packed V layout).
  constexpr static int64_t v_cache_token_group_stride(
      const int32_t /*block_size*/) {
    return BlockSizeAlignment * (AMX_TILE_ROW_BYTES / 4);
  }

  constexpr static int64_t v_cache_head_group_stride(
      const int32_t block_size) {
    return block_size * HeadDimAlignment;
  }

  // Copy Q heads tile: quantize BF16 → FP8-E4M3 and pack into q_buffer.
  //
  // Q buffer layout (FP8):
  //   Same tile-prepacking as AMX_BF16 copy_q_heads_tile but with uint8_t
  //   (FP8) elements.  Each FP8 tile row = 64 bytes = 64 FP8 elements.
  //   head_size_block_num = head_dim / 64 (vs head_dim/32 for BF16).
  //
  // Dynamic scale: max(abs(Q)) / 448.0f is computed over all elements,
  // stored in q_dynamic_scale for use in execute_attention().
  void copy_q_heads_tile(
      scalar_t* __restrict__ src,  // [q_num, q_heads_per_kv, head_size] BF16
      q_buffer_t* __restrict__ q_buffer,
      const int32_t q_num, const int32_t q_heads_per_kv,
      const int64_t q_num_stride, const int64_t q_head_stride,
      const float /*scale*/) {
    // ---- Step 1: compute dynamic per-tile Q scale ----
    constexpr float fp8_e4m3_max = 448.0f;
    float max_abs = 0.0f;
    {
      scalar_t* src_iter = src;
      for (int32_t q = 0; q < q_num; ++q, src_iter += q_num_stride) {
        scalar_t* head_iter = src_iter;
        for (int32_t h = 0; h < q_heads_per_kv;
             ++h, head_iter += q_head_stride) {
          for (int64_t e = 0; e < head_dim; ++e) {
            float v = static_cast<float>(head_iter[e]);
            max_abs = std::max(max_abs, std::abs(v));
          }
        }
      }
    }
    if (max_abs == 0.0f) max_abs = 1.0f;  // avoid div-by-zero
    const float inv_scale = fp8_e4m3_max / max_abs;
    // Store true scale for score correction in execute_attention().
    q_dynamic_scale = max_abs / fp8_e4m3_max;

    // ---- Step 2: quantize BF16 Q → FP8, pack into q_buffer ----
    // Layout: same tile-prepacking as AMX_BF16, but with FP8 (1 byte/elem).
    //   head_size_block_num = head_dim * sizeof(uint8_t) / AMX_TILE_ROW_BYTES
    //                       = head_dim / 64
    constexpr int64_t bytes_per_head = head_dim * sizeof(uint8_t);
    static_assert(bytes_per_head % AMX_TILE_ROW_BYTES == 0,
                  "head_dim must be divisible by 64 for AMX_FP8");
    constexpr int64_t head_size_block_num = bytes_per_head / AMX_TILE_ROW_BYTES;
    // 64 FP8 elements per 64-byte block.
    constexpr int64_t head_elem_num_per_block = AMX_TILE_ROW_BYTES;

    int32_t idx = 0;
    int8_t* __restrict__ q_buf_iter = reinterpret_cast<int8_t*>(q_buffer);
    for (int32_t q_idx = 0; q_idx < q_num; ++q_idx, src += q_num_stride) {
      scalar_t* __restrict__ src_iter = src;
      for (int32_t h_idx = 0; h_idx < q_heads_per_kv;
           ++h_idx, src_iter += q_head_stride) {
        // Quantize head_size_block_num blocks of 64 BF16 → 64 FP8.
        vec_op::unroll_loop<int32_t, head_size_block_num>(
            [&](int32_t blk) {
#if defined(__AVX512F__)
              vec_op::quant_bf16x32_to_fp8e4m3_avx512(
                  src_iter + blk * head_elem_num_per_block,
                  reinterpret_cast<uint8_t*>(
                      q_buf_iter + blk * AMX_TILE_BYTES),
                  inv_scale);
              // Second half of the 64-element block.
              vec_op::quant_bf16x32_to_fp8e4m3_avx512(
                  src_iter + blk * head_elem_num_per_block + 32,
                  reinterpret_cast<uint8_t*>(
                      q_buf_iter + blk * AMX_TILE_BYTES + 32),
                  inv_scale);
#else
              // Scalar fallback
              for (int32_t e = 0; e < head_elem_num_per_block; ++e) {
                float v = static_cast<float>(
                    src_iter[blk * head_elem_num_per_block + e]);
                *(reinterpret_cast<uint8_t*>(
                    q_buf_iter + blk * AMX_TILE_BYTES) + e) =
                    float_to_fp8e4m3_scalar(v, inv_scale);
              }
#endif
            });

        ++idx;
        q_buf_iter += AMX_TILE_ROW_BYTES;
        if ((idx & (AMX_TILE_ROW_NUM - 1)) == 0) {
          q_buf_iter -= AMX_TILE_ROW_NUM * AMX_TILE_ROW_BYTES;
          q_buf_iter += head_size_block_num * AMX_TILE_BYTES;
        }
      }
    }
  }

  // reshape KV to AMX-FP8 friendly layout.
  //   K: native FP8 layout (1 byte/element, 16-token groups × k-slices).
  //   V: halfword-packed FP8 layout (same as commit 22524f7a92, compatible
  //      with BF16 PV path via prepare_b_tile dequant).
  static void reshape_and_cache(
      const scalar_t* __restrict__ key, const scalar_t* __restrict__ value,
      kv_cache_t* __restrict__ key_cache, kv_cache_t* __restrict__ value_cache,
      const int64_t* __restrict__ slot_mapping, const int64_t token_num,
      const int64_t key_token_num_stride, const int64_t value_token_num_stride,
      const int64_t head_num, const int64_t key_head_num_stride,
      const int64_t value_head_num_stride, const int64_t num_blocks,
      const int64_t num_blocks_stride, const int64_t cache_head_num_stride,
      const int64_t block_size, const int64_t /*block_size_stride*/,
      const float k_inv = 0.0f, const float v_inv = 0.0f) {
    constexpr auto qfn = select_fp8_quant_fn<kv_cache_t>();

    // K: native AMX-FP8 layout
    reshape_and_cache_k_amx_fp8_impl<scalar_t, qfn>(
        key, reinterpret_cast<uint8_t*>(key_cache), slot_mapping, token_num,
        head_num, head_dim, block_size, key_token_num_stride,
        key_head_num_stride, num_blocks_stride, cache_head_num_stride, k_inv);

    // V: reuse halfword-packed FP8 layout from reshape_and_cache_fp8_amx_impl
    // (only V, key pointers are dummies that we skip by passing nullptr for
    //  the key cache — use the vec impl for V only).
    //
    // reshape_and_cache_fp8_amx_impl writes both K and V; we only want V here.
    // Call it with a temporary key buffer on the stack to avoid corrupting the
    // real K cache that we just wrote in native layout.
    //
    // A cleaner approach: replicate the V-only portion inline.
    {
      constexpr int64_t token_num_per_sub_group = 2;  // = 4/sizeof(BF16)
      constexpr int64_t head_elems_per_group = 16;    // AMX_TILE_ROW_BYTES/4
      const int64_t group_num = head_dim / head_elems_per_group;
      const int64_t group_size = block_size * head_elems_per_group;
      uint8_t* vc = reinterpret_cast<uint8_t*>(value_cache);

#pragma omp parallel for collapse(2) schedule(static)
      for (int64_t tok = 0; tok < token_num; ++tok) {
        for (int64_t h = 0; h < head_num; ++h) {
          const int64_t slot = slot_mapping[tok];
          if (slot < 0) continue;
          const int64_t block_idx = slot / block_size;
          const int64_t block_offset = slot % block_size;
          const int64_t sub_group_idx = block_offset / token_num_per_sub_group;
          const int64_t sub_group_offset =
              block_offset % token_num_per_sub_group;

          const scalar_t* vsrc =
              value + tok * value_token_num_stride + h * value_head_num_stride;
          uint8_t* vdst =
              vc + block_idx * num_blocks_stride + h * cache_head_num_stride +
              sub_group_idx * token_num_per_sub_group * head_elems_per_group +
              sub_group_offset;

          for (int64_t i = 0; i < group_num; ++i) {
            for (int64_t j = 0; j < head_elems_per_group; ++j)
              vdst[j * token_num_per_sub_group] =
                  qfn(static_cast<float>(vsrc[j]), v_inv);
            vsrc += head_elems_per_group;
            vdst += group_size;
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

#endif  // CPU_CAPABILITY_AMXFP8
#endif  // CPU_ATTN_AMX_FP8_HPP
