// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Adapted from DeepGEMM (https://github.com/deepseek-ai/DeepGEMM) at
// 88965b078186ee7510ab9fc4f1d5ebc19adfa8d1, the last release before its SM100
// MQA-logits kernels went MX-only. FP8 MQA-logits building blocks: the
// shared-storage layout with fp32 per-key scales and fp32 weights, the
// grid-stride contiguous-KV scheduler and the 2D/3D TMA load. Trimmed to what
// the fused kernel instantiates (FP8 only, no metadata schedule, no multicast,
// no paging).
#pragma once

#include <cute/arch/copy_sm90_tma.hpp>
#include <cutlass/arch/barrier.h>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/packing.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/common/utils.cuh>

namespace fused_mqa_topk::dg {
namespace math = deep_gemm::math;

// FP8 Q/K (one byte per element), fp32 per-key scales, fp32 weights: the
// DeepGEMM layout without its MX / FP4 variants.
template <uint32_t kNumHeads, uint32_t kHeadDim, uint32_t BLOCK_Q,
          uint32_t SPLIT_KV, uint32_t kNumQStages, uint32_t kNumKVStages,
          uint32_t kNumTmemStages, typename qk_dtype_t,
          typename reduce_dtype_t = float>
struct MQALogitsSharedStorage {
  using Barrier = cutlass::arch::ClusterTransactionBarrier;
  DG_STATIC_ASSERT(sizeof(qk_dtype_t) == 1, "FP8 Q/K only");

  static constexpr uint32_t BLOCK_QH = BLOCK_Q * kNumHeads;
  static constexpr uint32_t UMMA_N = math::constexpr_align(BLOCK_QH, 8u);
  // Align to one 8-row Q/K swizzle tile (head_dim bytes per FP8 token).
  static constexpr uint32_t kSwizzleAlignment = 8 * kHeadDim;
  static constexpr uint32_t kNumQElementsPerStage = UMMA_N * kHeadDim;
  static constexpr uint32_t kNumKVElementsPerStage = SPLIT_KV * kHeadDim;
  // TMA destinations in shared memory must be 128-byte aligned.
  static constexpr uint32_t kTmaAlignment = 128;
  static constexpr uint32_t kNumWeightBytesPerRow = math::constexpr_align(
      kNumHeads * static_cast<uint32_t>(sizeof(reduce_dtype_t)), 16u);
  static constexpr uint32_t kNumWeightBytesPerStage =
      math::constexpr_align(BLOCK_Q * kNumWeightBytesPerRow, kTmaAlignment);
  static constexpr uint32_t kNumWeightElementsPerStage =
      kNumWeightBytesPerStage / static_cast<uint32_t>(sizeof(reduce_dtype_t));

  DG_STATIC_ASSERT(kNumQElementsPerStage % kSwizzleAlignment == 0,
                   "Unaligned TMA swizzling");
  DG_STATIC_ASSERT(kNumKVElementsPerStage % kSwizzleAlignment == 0,
                   "Unaligned TMA swizzling");
  DG_STATIC_ASSERT(kSwizzleAlignment % 128 == 0,
                   "TMA destination must be 128-byte aligned");

  alignas(kSwizzleAlignment)
      qk_dtype_t smem_q[kNumQStages][kNumQElementsPerStage];
  alignas(kSwizzleAlignment)
      qk_dtype_t smem_kv[kNumKVStages][kNumKVElementsPerStage];
  alignas(kTmaAlignment) float smem_sf_kv[kNumKVStages][SPLIT_KV];
  alignas(kTmaAlignment)
      reduce_dtype_t smem_weights[kNumQStages][kNumWeightElementsPerStage];
  // Barriers require 8-byte alignment, already guaranteed by the preceding
  // TMA-aligned arrays.
  Barrier full_q_barriers[kNumQStages];
  Barrier empty_q_barriers[kNumQStages];
  Barrier full_kv_barriers[kNumKVStages];
  Barrier empty_kv_barriers[kNumKVStages];
  Barrier full_tmem_barriers[kNumTmemStages];
  Barrier empty_tmem_barriers[kNumTmemStages];
  uint32_t tmem_ptr_in_smem;
};

// Grid-stride contiguous-KV scheduler: SM `s` owns Q blocks `s`, `s + kNumSMs`,
// ..., each visiting the KV range [min k_start, max k_end) of its rows.
template <uint32_t BLOCK_Q, uint32_t SPLIT_KV, uint32_t kNumSMs>
struct SM100MQALogitsScheduler {
  uint32_t current_q_block_idx;
  uint32_t num_q_blocks;
  uint32_t num_q_tokens;
  uint32_t num_kv_tokens;
  const uint32_t* cu_seq_len_k_start;
  const uint32_t* cu_seq_len_k_end;

  CUTLASS_DEVICE SM100MQALogitsScheduler(const uint32_t& sm_idx,
                                         const uint32_t& num_q_tokens,
                                         const uint32_t& num_kv_tokens,
                                         const uint32_t* cu_seq_len_k_start,
                                         const uint32_t* cu_seq_len_k_end)
      : current_q_block_idx(sm_idx),
        num_q_blocks(math::ceil_div(num_q_tokens, BLOCK_Q)),
        num_q_tokens(num_q_tokens),
        num_kv_tokens(num_kv_tokens),
        cu_seq_len_k_start(cu_seq_len_k_start),
        cu_seq_len_k_end(cu_seq_len_k_end) {
    DG_STATIC_ASSERT(kNumSMs > 0, "Invalid SM count");
  }

  // NOTE: Out-of-range rows are clamped to the last valid row
  CUTLASS_DEVICE bool next_q_block(uint32_t& q_block_idx,
                                   uint32_t& kv_token_base,
                                   uint32_t& num_kv_splits) {
    if (current_q_block_idx >= num_q_blocks) return false;
    q_block_idx = current_q_block_idx;
    current_q_block_idx += kNumSMs;
    uint32_t start = cute::numeric_limits<uint32_t>::max();
    uint32_t end = cute::numeric_limits<uint32_t>::min();
#pragma unroll 8
    for (uint32_t token_idx = 0; token_idx < BLOCK_Q; ++token_idx) {
      const auto row_idx =
          cute::min(q_block_idx * BLOCK_Q + token_idx, num_q_tokens - 1);
      start = cute::min(start,
                        cute::min(cu_seq_len_k_start[row_idx], num_kv_tokens));
      end = cute::max(end, cute::min(cu_seq_len_k_end[row_idx], num_kv_tokens));
    }
    kv_token_base = start / 4 * 4;
    num_kv_splits = math::ceil_div(end - kv_token_base, SPLIT_KV);
    return true;
  }

  CUTLASS_DEVICE uint32_t
  get_q_tma_token_base(const uint32_t& q_block_idx) const {
    return q_block_idx * BLOCK_Q;
  }

  CUTLASS_DEVICE static uint32_t get_kv_tma_offset(
      const uint32_t& kv_token_base, const uint32_t& kv_split_idx) {
    return kv_token_base + kv_split_idx * SPLIT_KV;
  }

  CUTLASS_DEVICE static uint32_t get_logits_row(const uint32_t& q_block_idx,
                                                const uint32_t& token_idx) {
    return q_block_idx * BLOCK_Q + token_idx;
  }

  CUTLASS_DEVICE static uint32_t get_logits_col(
      const uint32_t& kv_token_base, const uint32_t& kv_split_idx,
      const uint32_t& math_thread_idx) {
    return kv_token_base + kv_split_idx * SPLIT_KV + math_thread_idx;
  }
};

// Single-CTA TMA load of a BLOCK_OUTER x BLOCK_INNER box (3D adds a batch
// coordinate).
template <uint32_t BLOCK_INNER, uint32_t BLOCK_OUTER, uint32_t kSwizzleMode,
          typename dtype_t, bool kIs3DTMA = false>
CUTLASS_DEVICE void tma_copy(
    void const* desc_ptr, cutlass::arch::ClusterTransactionBarrier* barrier_ptr,
    dtype_t* smem_ptr, const uint32_t& inner_idx, const uint32_t& outer_idx,
    const uint32_t& batch_idx = 0) {
  constexpr auto kCacheHint =
      static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL);
  constexpr uint32_t kPackFactor = deep_gemm::get_smem_pack_factor<dtype_t>();
  DG_STATIC_ASSERT(kPackFactor == 1 or sizeof(dtype_t) == 1,
                   "Packing expects a 1-byte storage type");
  constexpr uint32_t BLOCK_INNER_ATOM_STORAGE =
      kSwizzleMode == 0 ? BLOCK_INNER / kPackFactor
                        : kSwizzleMode / sizeof(dtype_t);
  constexpr uint32_t BLOCK_INNER_ATOM_LOGICAL =
      BLOCK_INNER_ATOM_STORAGE * kPackFactor;
  DG_STATIC_ASSERT(BLOCK_INNER % BLOCK_INNER_ATOM_LOGICAL == 0,
                   "TMA inner block must contain whole atoms");

#pragma unroll
  for (uint32_t i = 0; i < BLOCK_INNER / BLOCK_INNER_ATOM_LOGICAL; ++i) {
    if constexpr (kIs3DTMA) {
      cute::SM90_TMA_LOAD_3D::copy(
          desc_ptr, reinterpret_cast<uint64_t*>(barrier_ptr), kCacheHint,
          smem_ptr + i * BLOCK_OUTER * BLOCK_INNER_ATOM_STORAGE,
          inner_idx + i * BLOCK_INNER_ATOM_LOGICAL, outer_idx, batch_idx);
    } else {
      cute::SM90_TMA_LOAD_2D::copy(
          desc_ptr, reinterpret_cast<uint64_t*>(barrier_ptr), kCacheHint,
          smem_ptr + i * BLOCK_OUTER * BLOCK_INNER_ATOM_STORAGE,
          inner_idx + i * BLOCK_INNER_ATOM_LOGICAL, outer_idx);
    }
  }
}

}  // namespace fused_mqa_topk::dg
