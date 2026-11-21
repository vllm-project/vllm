// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#ifndef CPU_ATTN_SVE_BFMMLA_HPP
#define CPU_ATTN_SVE_BFMMLA_HPP

// SVE256 BF16 attention (ISA::SVE). The kernel needs two 128-bit segments per
// vector; other builds alias AttentionImplSVEBFMMLA to the NEON BFMMLA backend.

#include "cpu_attn_impl.hpp"

#if CPU_CAPABILITY_SVE256

  #include <arm_sve.h>

  #include <algorithm>
  #include <cstdint>

namespace cpu_attention {

namespace sve_bfmmla {

// svbfmmla: C[2x2] += A[2x4] * B[4x2], once per 128-bit segment.
constexpr int32_t kChunkK = 4;
constexpr int32_t kColGroup = 8;
constexpr int32_t kGroupElems = kColGroup * kChunkK;
constexpr int32_t kRowPairChunkElems = 2 * kChunkK;
constexpr int32_t kMaxPanelRows = 8;
constexpr int32_t kKCacheTokenGroup = 8;
constexpr int32_t kVCacheTokenGroup = 4;
constexpr int32_t kPvPackTokens = 256;

// a: row pairs, each [k/4][r0 x4 | r1 x4]; b: [N/8][k/4][32].
struct GemmOperands {
  const bfloat16_t* a;
  const bfloat16_t* b;
  float* c;
  int32_t m;
  int32_t k;
  int64_t a_pair_stride;
  int64_t b_col_group_stride;
  int64_t ldc;
  bool accumulate;

  FORCE_INLINE GemmOperands from_column(int32_t col) const {
    GemmOperands shifted = *this;
    shifted.b += (col / kColGroup) * b_col_group_stride;
    shifted.c += col;
    return shifted;
  }
};

FORCE_INLINE constexpr int32_t rows_in_pair(int32_t m, int32_t pair) {
  const int32_t left = m - 2 * pair;
  return left >= 2 ? 2 : (left > 0 ? left : 0);
}

FORCE_INLINE svuint64_t scale_8_bf16(const uint16_t* src,
                                     const svfloat32_t scale) {
  const svbool_t all32 = svptrue_b32();
  const svuint32_t bits = svld1uh_u32(all32, src);
  const svfloat32_t scaled = svmul_f32_x(
      all32, svreinterpret_f32_u32(svlsl_n_u32_x(all32, bits, 16)), scale);
  // svcvt fills the even 16-bit lanes; uzp1 compacts them.
  const svuint16_t rounded =
      svreinterpret_u16_bf16(svcvt_bf16_f32_x(all32, scaled));
  return svreinterpret_u64_u16(svuzp1_u16(rounded, rounded));
}

FORCE_INLINE void pack_row_pair(const uint16_t* __restrict__ row0,
                                const uint16_t* __restrict__ row1,
                                uint16_t* __restrict__ dst, const int32_t len) {
  const svbool_t all16 = svptrue_b16();
  int32_t k = 0;
  for (; k + 16 <= len; k += 16) {
    const svuint64_t r0 = svreinterpret_u64_u16(svld1_u16(all16, row0 + k));
    const svuint64_t r1 = svreinterpret_u64_u16(svld1_u16(all16, row1 + k));
    svst1_u16(all16, dst + 2 * k, svreinterpret_u16_u64(svzip1_u64(r0, r1)));
    svst1_u16(all16, dst + 2 * k + 16,
              svreinterpret_u16_u64(svzip2_u64(r0, r1)));
  }
  const int32_t full_chunks = (len - k) / kChunkK;
  if (full_chunks > 0) {
    const svbool_t in = svwhilelt_b16(0, full_chunks * kChunkK);
    const svuint64_t r0 = svreinterpret_u64_u16(svld1_u16(in, row0 + k));
    const svuint64_t r1 = svreinterpret_u64_u16(svld1_u16(in, row1 + k));
    svst1_u16(svwhilelt_b16(0, 2 * full_chunks * kChunkK), dst + 2 * k,
              svreinterpret_u16_u64(svzip1_u64(r0, r1)));
    if (full_chunks > 2) {
      svst1_u16(svwhilelt_b16(0, (2 * full_chunks - 4) * kChunkK),
                dst + 2 * k + 16, svreinterpret_u16_u64(svzip2_u64(r0, r1)));
    }
    k += full_chunks * kChunkK;
  }
  if (k < len) {
    for (int32_t t = 0; t < kChunkK; ++t) {
      const bool valid = k + t < len;
      dst[2 * k + t] = valid ? row0[k + t] : uint16_t(0);
      dst[2 * k + kChunkK + t] = valid ? row1[k + t] : uint16_t(0);
    }
  }
}

  // SVE types cannot form arrays, so macros expand one named accumulator per
  // (row pair, column group) slot; unused slots compile away.
  // clang-format off
  #define SVB_FOR_0_TO_3(X) X(0) X(1) X(2) X(3)
  #define SVB_FOR_SLOTS(X)                                            \
    X(0, 0) X(0, 1) X(0, 2) X(0, 3) X(1, 0) X(1, 1) X(1, 2) X(1, 3) \
    X(2, 0) X(2, 1) X(2, 2) X(2, 3) X(3, 0) X(3, 1) X(3, 2) X(3, 3)
// clang-format on

// One ld1rq A load feeds both segments, so each bfmmla computes two 2x2 tiles
// (acc*_lo: cols 0..3, acc*_hi: cols 4..7 of a group). kLastRowUnpaired: the
// last row is a single row-major row, read with a 4-lane ld1rq.
template <int32_t RP, int32_t CG, bool kLastRowUnpaired>
FORCE_INLINE void micro_tile(const GemmOperands& op) {
  static_assert(RP >= 1 && RP <= 4 && CG >= 1 && CG <= 4);
  static_assert(RP * CG <= 8, "accumulators must fit in 16 registers");

  const svbool_t all16 = svptrue_b16();
  const svbool_t all32 = svptrue_b32();
  [[maybe_unused]] const svbool_t first4 = svwhilelt_b16(0, kChunkK);
  float* __restrict__ c = op.c;
  const int64_t ldc = op.ldc;

  #define SVB_DECLARE_PAIR(p)                                              \
    [[maybe_unused]] const int32_t rows##p = rows_in_pair(op.m, p);        \
    [[maybe_unused]] const bfloat16_t* a##p = op.a + p * op.a_pair_stride; \
    [[maybe_unused]] constexpr bool kUnpaired##p =                         \
        kLastRowUnpaired && p == RP - 1;
  SVB_FOR_0_TO_3(SVB_DECLARE_PAIR)
  #undef SVB_DECLARE_PAIR

  #define SVB_DECLARE_GROUP(g) \
    [[maybe_unused]] const bfloat16_t* b##g = op.b + g * op.b_col_group_stride;
  SVB_FOR_0_TO_3(SVB_DECLARE_GROUP)
  #undef SVB_DECLARE_GROUP

  #define SVB_INIT(p, g)                                                     \
    [[maybe_unused]] svfloat32_t acc##p##g##_lo, acc##p##g##_hi;             \
    if constexpr (RP > p && CG > g) {                                        \
      if (op.accumulate) {                                                   \
        const float* cp = c + p * 2 * ldc + g * kColGroup;                   \
        const svfloat64_t r0 = svreinterpret_f64_f32(svld1_f32(all32, cp));  \
        const svfloat64_t r1 =                                               \
            rows##p == 2 ? svreinterpret_f64_f32(svld1_f32(all32, cp + ldc)) \
                         : svdup_n_f64(0.0);                                 \
        acc##p##g##_lo = svreinterpret_f32_f64(svzip1_f64(r0, r1));          \
        acc##p##g##_hi = svreinterpret_f32_f64(svzip2_f64(r0, r1));          \
      } else {                                                               \
        acc##p##g##_lo = svdup_n_f32(0.0f);                                  \
        acc##p##g##_hi = svdup_n_f32(0.0f);                                  \
      }                                                                      \
    }
  SVB_FOR_SLOTS(SVB_INIT)
  #undef SVB_INIT

  #pragma GCC unroll 4
  for (int32_t kk = 0; kk < op.k; kk += kChunkK) {
  #define SVB_LOAD_A(p)                                          \
    [[maybe_unused]] svbfloat16_t va##p;                         \
    if constexpr (RP > p) {                                      \
      va##p = svld1rq_bf16(kUnpaired##p ? first4 : all16, a##p); \
    }
    SVB_FOR_0_TO_3(SVB_LOAD_A)
  #undef SVB_LOAD_A

  #define SVB_LOAD_B(g)                                       \
    [[maybe_unused]] svbfloat16_t vb##g##_lo, vb##g##_hi;     \
    if constexpr (CG > g) {                                   \
      vb##g##_lo = svld1_bf16(all16, b##g);                   \
      vb##g##_hi = svld1_bf16(all16, b##g + kGroupElems / 2); \
    }
    SVB_FOR_0_TO_3(SVB_LOAD_B)
  #undef SVB_LOAD_B

  #define SVB_MMLA(p, g)                                                \
    if constexpr (RP > p && CG > g) {                                   \
      acc##p##g##_lo = svbfmmla_f32(acc##p##g##_lo, va##p, vb##g##_lo); \
      acc##p##g##_hi = svbfmmla_f32(acc##p##g##_hi, va##p, vb##g##_hi); \
    }
    SVB_FOR_SLOTS(SVB_MMLA)
  #undef SVB_MMLA

  #define SVB_ADVANCE(i)                                                       \
    if constexpr (RP > i) a##i += kUnpaired##i ? kChunkK : kRowPairChunkElems; \
    if constexpr (CG > i) b##i += kGroupElems;
    SVB_FOR_0_TO_3(SVB_ADVANCE)
  #undef SVB_ADVANCE
  }

  #define SVB_STORE(p, g)                                                \
    if constexpr (RP > p && CG > g) {                                    \
      if (rows##p > 0) {                                                 \
        float* cp = c + p * 2 * ldc + g * kColGroup;                     \
        const svfloat64_t lo = svreinterpret_f64_f32(acc##p##g##_lo);    \
        const svfloat64_t hi = svreinterpret_f64_f32(acc##p##g##_hi);    \
        svst1_f32(all32, cp, svreinterpret_f32_f64(svuzp1_f64(lo, hi))); \
        if (rows##p == 2) {                                              \
          svst1_f32(all32, cp + ldc,                                     \
                    svreinterpret_f32_f64(svuzp2_f64(lo, hi)));          \
        }                                                                \
      }                                                                  \
    }
  SVB_FOR_SLOTS(SVB_STORE)
  #undef SVB_STORE
}

  #undef SVB_FOR_SLOTS
  #undef SVB_FOR_0_TO_3

// Fewer row pairs leave registers for wider tiles.
template <int32_t RP, int32_t N, bool kLastRowUnpaired>
FORCE_INLINE void row_panel(const GemmOperands& op) {
  static_assert(N % kColGroup == 0);
  constexpr int32_t kGroups = (RP <= 2) ? 4 : 2;
  constexpr int32_t kTileCols = kGroups * kColGroup;
  constexpr int32_t kFullTiles = N / kTileCols;
  constexpr int32_t kLeftoverGroups = (N % kTileCols) / kColGroup;

  for (int32_t t = 0; t < kFullTiles; ++t) {
    micro_tile<RP, kGroups, kLastRowUnpaired>(op.from_column(t * kTileCols));
  }
  if constexpr (kLeftoverGroups >= 2) {
    micro_tile<RP, 2, kLastRowUnpaired>(op.from_column(kFullTiles * kTileCols));
  }
  if constexpr (kLeftoverGroups % 2 == 1) {
    micro_tile<RP, 1, kLastRowUnpaired>(op.from_column(N - kColGroup));
  }
}

template <int32_t N, bool kLastRowUnpaired = false>
FORCE_INLINE void panel_gemm(const GemmOperands& op) {
  switch ((op.m + 1) >> 1) {
    case 1:
      row_panel<1, N, kLastRowUnpaired>(op);
      break;
    case 2:
      row_panel<2, N, kLastRowUnpaired>(op);
      break;
    case 3:
      row_panel<3, N, kLastRowUnpaired>(op);
      break;
    default:
      row_panel<4, N, kLastRowUnpaired>(op);
      break;
  }
}

}  // namespace sve_bfmmla

// QK: A is q_buffer, already row-pair packed by copy_q_heads_tile.
// PV: A is the row-major bf16 probability buffer, packed here per block.
template <typename kv_cache_t, int32_t BlockTokens, int32_t HeadDim>
class TileGemmSVEBFMMLA {
 public:
  template <AttentionGemmPhase phase, int32_t head_dim_ct>
  FORCE_INLINE static void gemm(const int32_t m_size, void* __restrict__ a_tile,
                                kv_cache_t* __restrict__ b_tile,
                                float* __restrict__ c_tile, const int64_t lda,
                                [[maybe_unused]] const int64_t ldb,
                                const int64_t ldc, const int32_t block_size,
                                const int32_t dynamic_k_size,
                                const bool accum_c) {
    using namespace sve_bfmmla;
    static_assert(BlockTokens % kColGroup == 0);
    static_assert(HeadDim % kColGroup == 0);
    if constexpr (head_dim_ct >= 0) {
      static_assert(head_dim_ct == HeadDim);
    }

    const auto* kv = reinterpret_cast<const bfloat16_t*>(b_tile);

    if constexpr (phase == AttentionGemmPhase::QK) {
      const auto* q = reinterpret_cast<const bfloat16_t*>(a_tile);
      // Only the last panel of an odd chunk ends with an unpaired row.
      for (int32_t row = 0; row < m_size; row += kMaxPanelRows) {
        const GemmOperands op{
            .a = q + row * HeadDim,
            .b = kv,
            .c = c_tile + row * ldc,
            .m = std::min(kMaxPanelRows, m_size - row),
            .k = HeadDim,
            .a_pair_stride = 2 * HeadDim,
            .b_col_group_stride = (HeadDim / kChunkK) * kGroupElems,
            .ldc = ldc,
            .accumulate = accum_c};
        if (op.m & 1) {
          panel_gemm<BlockTokens, true>(op);
        } else {
          panel_gemm<BlockTokens, false>(op);
        }
      }
    } else {
      constexpr int64_t kPackedPairStride = 2 * kPvPackTokens;
      alignas(64) uint16_t packed_p[(kMaxPanelRows / 2) * kPackedPairStride];
      const auto* p = reinterpret_cast<const uint16_t*>(a_tile);

      for (int32_t row = 0; row < m_size; row += kMaxPanelRows) {
        const int32_t panel_rows = std::min(kMaxPanelRows, m_size - row);
        for (int32_t k0 = 0; k0 < dynamic_k_size; k0 += kPvPackTokens) {
          const int32_t len = std::min(kPvPackTokens, dynamic_k_size - k0);
          for (int32_t pair = 0; pair < (panel_rows + 1) / 2; ++pair) {
            const int32_t r = row + 2 * pair;
            const uint16_t* row0 = p + r * lda + k0;
            // An odd panel's missing row duplicates row 0; never stored.
            const uint16_t* row1 = (r + 1 < m_size) ? row0 + lda : row0;
            pack_row_pair(row0, row1, packed_p + pair * kPackedPairStride, len);
          }
          panel_gemm<HeadDim>(
              GemmOperands{.a = reinterpret_cast<const bfloat16_t*>(packed_p),
                           .b = kv + (k0 / kChunkK) * kGroupElems,
                           .c = c_tile + row * ldc,
                           .m = panel_rows,
                           .k = len,
                           .a_pair_stride = kPackedPairStride,
                           .b_col_group_stride =
                               (block_size / kVCacheTokenGroup) * kGroupElems,
                           .ldc = ldc,
                           .accumulate = accum_c || k0 > 0});
        }
      }
    }
  }
};

template <int64_t block_size_alignment, ISA isa_type, int64_t head_dim>
class AttentionImplSVEBFMMLA {
 public:
  using query_t = c10::BFloat16;
  using q_buffer_t = c10::BFloat16;
  using kv_cache_t = c10::BFloat16;
  using logits_buffer_t = float;
  using partial_output_buffer_t = float;
  using prob_buffer_t = c10::BFloat16;

  static constexpr int64_t BlockSizeAlignment = block_size_alignment;
  static constexpr int64_t HeadDimAlignment = head_dim;
  static constexpr int64_t MaxQHeadNumPerIteration = 16;
  static constexpr int64_t HeadDim = head_dim;
  static constexpr ISA ISAType = isa_type;
  static constexpr bool scale_on_logits = false;
  static constexpr int64_t VCacheNGroup = sve_bfmmla::kColGroup;
  static constexpr int64_t VCacheKGroupStride = sve_bfmmla::kGroupElems;

  static_assert(HeadDim % 16 == 0, "K cache scatter handles 16 dims at once");
  static_assert(BlockSizeAlignment % sve_bfmmla::kKCacheTokenGroup == 0);

  template <template <typename tile_gemm_t> typename attention>
  FORCE_INLINE void execute_attention(DEFINE_CPU_ATTENTION_PARAMS) {
    attention<
        TileGemmSVEBFMMLA<kv_cache_t, static_cast<int32_t>(BlockSizeAlignment),
                          static_cast<int32_t>(HeadDim)>>
        attention_iteration;
    attention_iteration(CPU_ATTENTION_PARAMS);
  }

  // K cache: [block_size/8][head_dim/4][32]
  static constexpr int64_t k_cache_token_group_stride(
      [[maybe_unused]] const int32_t block_size) {
    return (BlockSizeAlignment / sve_bfmmla::kKCacheTokenGroup) *
           (head_dim / sve_bfmmla::kChunkK) * sve_bfmmla::kGroupElems;
  }

  // V cache: [head_dim/8][block_size/4][32]
  static constexpr int64_t v_cache_token_group_stride(
      [[maybe_unused]] const int32_t block_size) {
    return (BlockSizeAlignment / sve_bfmmla::kVCacheTokenGroup) *
           sve_bfmmla::kGroupElems;
  }

  static constexpr int64_t v_cache_head_group_stride(
      [[maybe_unused]] const int32_t block_size) {
    return head_dim * block_size;
  }

  // Rows are paired within each q-iteration chunk, which has an odd row count
  // when q_heads_per_kv is odd; an odd chunk's last row stays row-major.
  static void copy_q_heads_tile(c10::BFloat16* __restrict__ src,
                                c10::BFloat16* __restrict__ q_buffer,
                                const int32_t q_num,
                                const int32_t q_heads_per_kv,
                                const int64_t q_num_stride,
                                const int64_t q_head_stride, float scale) {
    using sve_bfmmla::kColGroup;
    const svfloat32_t scale_vec = svdup_n_f32(scale);
    const svbool_t all16 = svptrue_b16();
    const svbool_t first8 = svwhilelt_b16(0, kColGroup);
    const int32_t total_rows = q_num * q_heads_per_kv;
    const int32_t chunk_rows =
        static_cast<int32_t>(MaxQHeadNumPerIteration / q_heads_per_kv) *
        q_heads_per_kv;

    auto src_row = [&](int32_t r) {
      return reinterpret_cast<const uint16_t*>(
          src + (r / q_heads_per_kv) * q_num_stride +
          (r % q_heads_per_kv) * q_head_stride);
    };
    auto dst_row = [&](int32_t r) {
      return reinterpret_cast<uint16_t*>(q_buffer + r * head_dim);
    };

    for (int32_t chunk = 0; chunk < total_rows; chunk += chunk_rows) {
      const int32_t chunk_end = std::min(chunk + chunk_rows, total_rows);
      int32_t r = chunk;
      for (; r + 1 < chunk_end; r += 2) {
        const uint16_t* row0 = src_row(r);
        const uint16_t* row1 = src_row(r + 1);
        uint16_t* dst = dst_row(r);
        for (int32_t d = 0; d < head_dim; d += kColGroup) {
          const svuint64_t s0 = sve_bfmmla::scale_8_bf16(row0 + d, scale_vec);
          const svuint64_t s1 = sve_bfmmla::scale_8_bf16(row1 + d, scale_vec);
          svst1_u16(all16, dst + 2 * d,
                    svreinterpret_u16_u64(svzip1_u64(s0, s1)));
        }
      }
      if (r < chunk_end) {
        const uint16_t* row0 = src_row(r);
        uint16_t* dst = dst_row(r);
        for (int32_t d = 0; d < head_dim; d += kColGroup) {
          svst1_u16(first8, dst + d,
                    svreinterpret_u16_u64(
                        sve_bfmmla::scale_8_bf16(row0 + d, scale_vec)));
        }
      }
    }
  }

  static void reshape_and_cache(
      const c10::BFloat16* __restrict__ key,
      const c10::BFloat16* __restrict__ value,
      c10::BFloat16* __restrict__ key_cache,
      c10::BFloat16* __restrict__ value_cache,
      const int64_t* __restrict__ slot_mapping, const int64_t token_num,
      const int64_t key_token_num_stride, const int64_t value_token_num_stride,
      const int64_t head_num, const int64_t key_head_num_stride,
      const int64_t value_head_num_stride,
      [[maybe_unused]] const int64_t num_blocks,
      const int64_t num_blocks_stride, const int64_t cache_head_num_stride,
      const int64_t block_size,
      [[maybe_unused]] const int64_t block_size_stride,
      const float /*k_inv*/ = 0.0f, const float /*v_inv*/ = 0.0f) {
    using namespace sve_bfmmla;
    const int64_t k_token_group_stride = (head_dim / kChunkK) * kGroupElems;
    const int64_t v_col_group_stride =
        (block_size / kVCacheTokenGroup) * kGroupElems;

  #pragma omp parallel for collapse(2)
    for (int64_t token = 0; token < token_num; ++token) {
      for (int64_t head = 0; head < head_num; ++head) {
        const int64_t slot = slot_mapping[token];
        if (slot < 0) continue;
        const int64_t block_offset = slot % block_size;
        const int64_t cache_offset = (slot / block_size) * num_blocks_stride +
                                     head * cache_head_num_stride;

        // K: one 64-bit scatter writes 16 dims (4-dim chunks, 32 apart).
        {
          const auto* src = reinterpret_cast<const uint16_t*>(
              key + token * key_token_num_stride + head * key_head_num_stride);
          const int64_t token_pair = (block_offset % kKCacheTokenGroup) / 2;
          c10::BFloat16* dst =
              key_cache + cache_offset +
              (block_offset / kKCacheTokenGroup) * k_token_group_stride +
              token_pair * kRowPairChunkElems + (block_offset & 1) * kChunkK;
          const svuint64_t chunk_index = svindex_u64(0, kGroupElems / 4);
          for (int64_t d = 0; d < head_dim; d += 16) {
            const svuint64_t chunks =
                svreinterpret_u64_u16(svld1_u16(svptrue_b16(), src + d));
            svst1_scatter_u64index_u64(
                svptrue_b64(),
                reinterpret_cast<uint64_t*>(dst + (d / kChunkK) * kGroupElems),
                chunk_index, chunks);
          }
        }

        // V: dim j of 8-dim group lands at group[4 * j + token % 4].
        {
          const auto* src = reinterpret_cast<const uint16_t*>(
              value + token * value_token_num_stride +
              head * value_head_num_stride);
          auto* dst = reinterpret_cast<uint16_t*>(
              value_cache + cache_offset +
              (block_offset / kVCacheTokenGroup) * kGroupElems +
              block_offset % kVCacheTokenGroup);
          const svuint32_t dim_index = svindex_u32(0, kVCacheTokenGroup);
          for (int64_t d = 0; d < head_dim; d += kColGroup) {
            svst1h_scatter_u32index_u32(
                svptrue_b32(), dst + (d / kColGroup) * v_col_group_stride,
                dim_index, svld1uh_u32(svptrue_b32(), src + d));
          }
        }
      }
    }
  }
};

}  // namespace cpu_attention

#else

  #include "cpu_attn_neon_bfmmla.hpp"

namespace cpu_attention {

template <int64_t block_size_alignment, ISA isa_type, int64_t head_dim>
class AttentionImplSVEBFMMLA
    : public AttentionImplNEONBFMMLA<block_size_alignment, isa_type, head_dim> {
};

}  // namespace cpu_attention

#endif  // CPU_CAPABILITY_SVE256

#endif  // CPU_ATTN_SVE_BFMMLA_HPP