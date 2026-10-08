// SPDX-License-Identifier: MIT
// Experimental derivative of DeepGEMM 2.8 sm100_mqa_logits.cuh.
// Persistent DeepGEMM FP8 MQA producer with fixed-threshold candidate emission.
#pragma once
#ifndef MQA_TMEM_ROWS
  #define MQA_TMEM_ROWS 2
#endif

#include <cutlass/arch/reg_reconfig.h>
#include "fallback_shared.cuh"
#ifndef MQA_HANDOFF_INTERVAL
  #define MQA_HANDOFF_INTERVAL 64
#endif
static_assert(MQA_HANDOFF_INTERVAL == 32 || MQA_HANDOFF_INTERVAL == 64 ||
              MQA_HANDOFF_INTERVAL == 128);

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>

#include <deep_gemm/common/cute_tie.cuh>
#include <deep_gemm/common/packing.cuh>
#include <deep_gemm/common/ring_pipeline.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/epilogue/clean_logits.cuh>
#include <deep_gemm/layout/mqa_logits.cuh>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <deep_gemm/scheduler/sm100_mqa_logits.cuh>
#include <deep_gemm/scheduler/sm100_paged_mqa_logits.cuh>
#include <cuda_fp16.h>
#include <cutlass/arch/barrier.h>
#include <deep_gemm/common/math.cuh>

// Shared SM100 MQA logits core plus contiguous-KV and paged entries
// Both entries use the same q / sf_q / kv / sf_kv / weights TMA signature

namespace seed_fallback {
using namespace deep_gemm;

// Convert runtime valid-token count to `cute::Int` so token loops stay
// compile-time constant
template <uint32_t kBlockQ, uint32_t kCandidate = kBlockQ, typename Fn>
CUTLASS_DEVICE void dispatch_num_block_tokens(const uint32_t& num_block_tokens,
                                              Fn&& fn) {
  if constexpr (kCandidate <= 1) {
    fn(cute::Int<1>{});
  } else if (num_block_tokens >= kCandidate) {
    fn(cute::Int<kCandidate>{});
  } else {
    dispatch_num_block_tokens<kBlockQ, kCandidate - 1>(num_block_tokens,
                                                       static_cast<Fn&&>(fn));
  }
}

// Load heads using power-of-two TMEM chunks.
template <uint32_t kNumHeads, uint32_t kNumLoaded = 0>
CUTLASS_DEVICE void load_tmem_heads_decomposed(const uint32_t& tmem_col,
                                               float* accum) {
  constexpr uint32_t kNumRemaining = kNumHeads - kNumLoaded;
  if constexpr (kNumRemaining > 0) {
    constexpr uint32_t kChunk = kNumRemaining >= 64   ? 64
                                : kNumRemaining >= 32 ? 32
                                : kNumRemaining >= 16 ? 16
                                : kNumRemaining >= 8  ? 8
                                                      : 4;
    ptx::tmem_load_32dp32b<kChunk>(
        tmem_col + kNumLoaded, reinterpret_cast<uint32_t*>(accum + kNumLoaded));
    load_tmem_heads_decomposed<kNumHeads, kNumLoaded + kChunk>(tmem_col, accum);
  }
}

// Shared device core parameterized by dtype and scheduler geometry/addressing
template <
    uint32_t kNumHeads, uint32_t kHeadDim, bool kIsMXSF,
    bool kIsCompressedLogits, bool kCleanLogits, uint32_t BLOCK_Q,
    uint32_t SPLIT_KV, uint32_t kNumQStages, uint32_t kNumKVStages,
    uint32_t kNumSMs, uint32_t kNumSpecializedThreads, uint32_t kNumMathThreads,
    typename qk_dtype_t, typename logits_dtype_t, typename reduce_dtype_t,
    typename MakeScheduler, uint32_t kNumMathWarpGroups = kNumMathThreads / 128>
CUTLASS_DEVICE void sm100_mqa_logits_core_impl(
    const uint32_t num_q_tokens, FallbackBuffers buffers,
    const cute::TmaDescriptor& tensor_map_q,
    const cute::TmaDescriptor& tensor_map_sf_q,
    const cute::TmaDescriptor& tensor_map_kv,
    const cute::TmaDescriptor& tensor_map_sf_kv,
    const cute::TmaDescriptor& tensor_map_weights,
    const MakeScheduler& make_scheduler) {
  constexpr bool kIsFP4 = cute::is_same_v<qk_dtype_t, cutlass::float_e2m1_t>;

  const auto sm_idx = blockIdx.x;
  const auto warp_idx = cutlass::canonical_warp_idx_sync();
  const auto lane_idx = ptx::get_lane_idx();
  constexpr uint32_t kSpecWarpStart = kNumMathWarpGroups * 4;

  const auto kNegInf = -cute::numeric_limits<logits_dtype_t>::infinity();

  if (warp_idx == kSpecWarpStart) {
    cute::prefetch_tma_descriptor(&tensor_map_q);
    cute::prefetch_tma_descriptor(&tensor_map_sf_q);
    cute::prefetch_tma_descriptor(&tensor_map_weights);
    cute::prefetch_tma_descriptor(&tensor_map_kv);
    cute::prefetch_tma_descriptor(&tensor_map_sf_kv);
  }

  static constexpr uint32_t kNumTmemStages = 2;
  static_assert(BLOCK_Q == 8 && kNumMathThreads == 256 && !kIsMXSF);
  static constexpr uint32_t kNumUTCCPAlignedElems = 128;
  static constexpr uint32_t UMMA_M = 128;
  static constexpr uint32_t BLOCK_QH = BLOCK_Q * kNumHeads;
  static constexpr uint32_t UMMA_N = math::constexpr_align(BLOCK_QH, 8u);
  static constexpr uint32_t UMMA_K = kIsFP4 ? 64 : 32;
  static constexpr uint32_t kNumSFQ =
      kIsMXSF ? math::constexpr_align(UMMA_N, kNumUTCCPAlignedElems) : 0;
  static constexpr uint32_t kNumSFKV =
      kIsMXSF ? math::constexpr_align(SPLIT_KV, kNumUTCCPAlignedElems) : 0;
  static constexpr uint32_t kNumQKBytesPerToken =
      kIsFP4 ? (kHeadDim / 2) : kHeadDim;
  static constexpr uint32_t SMEM_Q_SIZE_PER_STAGE =
      BLOCK_QH * kNumQKBytesPerToken;
  static constexpr uint32_t SMEM_KV_SIZE_PER_STAGE =
      SPLIT_KV * kNumQKBytesPerToken;
  static constexpr uint32_t SMEM_SF_Q_SIZE_PER_STAGE =
      kIsMXSF ? (BLOCK_QH * sizeof(int)) : 0;
  static constexpr uint32_t SMEM_SF_KV_SIZE_PER_STAGE =
      kIsMXSF ? (kNumSFKV * sizeof(int)) : (SPLIT_KV * sizeof(float));
  static constexpr uint32_t kNumWeightBytesPerRow = math::constexpr_align(
      kNumHeads * static_cast<uint32_t>(sizeof(reduce_dtype_t)), 16u);
  static constexpr uint32_t kNumWeightElementsPerRow =
      kNumWeightBytesPerRow / static_cast<uint32_t>(sizeof(reduce_dtype_t));
  static constexpr uint32_t SMEM_WEIGHT_SIZE_PER_STAGE =
      BLOCK_Q * kNumWeightBytesPerRow;

  DG_STATIC_ASSERT(kNumSpecializedThreads == 128 and kNumMathThreads % 128 == 0,
                   "Invalid threads");
  DG_STATIC_ASSERT(SPLIT_KV == kNumMathWarpGroups * UMMA_M and
                       SPLIT_KV % kNumUTCCPAlignedElems == 0,
                   "Invalid `SPLIT_KV`");
  DG_STATIC_ASSERT(not(kIsCompressedLogits and kCleanLogits),
                   "Compressed logits cannot be cleaned in-kernel");

  using SharedStorage = layout::MQALogitsSharedStorage<
      kNumHeads, kHeadDim, kIsMXSF, BLOCK_Q, SPLIT_KV, kNumQStages,
      kNumKVStages, kNumTmemStages, qk_dtype_t, reduce_dtype_t>;
  extern __shared__ __align__(SharedStorage::kSwizzleAlignment)
      uint8_t smem_buffer[];
  auto& smem = *reinterpret_cast<SharedStorage*>(smem_buffer);
  constexpr uint32_t kRingSlots = 8;
  constexpr uint32_t kRecordBlockMask = 255;
  auto& stream =
      *reinterpret_cast<FallbackShared*>(smem_buffer + sizeof(SharedStorage));
  stream.init(threadIdx.x);

  constexpr uint32_t kNumAccumTmemCols = UMMA_N * kNumTmemStages;
  constexpr uint32_t kNumTmemCols =
      utils::get_num_aligned_tmem_cols<kNumAccumTmemCols + kNumSFQ / 32 +
                                       kNumSFKV / 32>();
  constexpr uint32_t kTmemStartColOfSFQ = kNumAccumTmemCols;
  constexpr uint32_t kTmemStartColOfSFKV = kNumAccumTmemCols + kNumSFQ / 32;
  DG_STATIC_ASSERT(kNumTmemCols <= 512, "Too many tensor memory");

  if (warp_idx == kSpecWarpStart + 1 and cute::elect_one_sync()) {
#pragma unroll
    for (uint32_t i = 0; i < kNumQStages; ++i) {
      smem.full_q_barriers[i].init(1);
      smem.empty_q_barriers[i].init(kNumMathThreads + 32);
    }
#pragma unroll
    for (uint32_t i = 0; i < kNumKVStages; ++i) {
      smem.full_kv_barriers[i].init(1);
      smem.empty_kv_barriers[i].init(kIsMXSF ? 1 : kNumMathThreads);
    }
#pragma unroll
    for (uint32_t i = 0; i < kNumTmemStages; ++i) {
      smem.full_tmem_barriers[i].init(1);
      smem.empty_tmem_barriers[i].init(256);
    }
    cutlass::arch::fence_barrier_init();
  }
  __syncwarp();

  if (warp_idx == kSpecWarpStart + 2)
    cute::TMEM::Allocator1Sm().allocate(kNumTmemCols, &smem.tmem_ptr_in_smem);
  __syncthreads();

  RingPipeline<kNumQStages> q_pipeline;
  RingPipeline<kNumKVStages> kv_pipeline;
  RingPipeline<kNumTmemStages> tmem_pipeline;

  constexpr uint32_t kNumSpecializedRegisters = 56;
  constexpr uint32_t kNumMathRegisters = 224;

  cudaGridDependencySynchronize();

  if (warp_idx == kSpecWarpStart) {
    cutlass::arch::warpgroup_reg_dealloc<56>();
    if (cute::elect_one_sync()) {
      auto scheduler = make_scheduler(sm_idx);
      // NOTES: split index for paged scheduler, token offset for contiguous-KV
      // scheduler.
      uint32_t q_block_idx, kv_base, num_kv_splits;
      while (scheduler.next_q_block(q_block_idx, kv_base, num_kv_splits)) {
        CUTE_TIE_DECL(q_pipeline.advance(), q_stage_idx, q_phase);
        smem.empty_q_barriers[q_stage_idx].wait(q_phase ^ 1);

        const uint32_t q_token_base =
            scheduler.get_q_tma_token_base(q_block_idx);
        tma::copy<kHeadDim, BLOCK_Q * kNumHeads, 0>(
            &tensor_map_q, &smem.full_q_barriers[q_stage_idx],
            smem.smem_q[q_stage_idx], 0, q_token_base * kNumHeads);
        if constexpr (kIsMXSF)
          tma::copy<BLOCK_Q * kNumHeads, 1, 0>(
              &tensor_map_sf_q, &smem.full_q_barriers[q_stage_idx],
              smem.smem_sf_q[q_stage_idx], 0, q_token_base);
        tma::copy<kNumWeightElementsPerRow, BLOCK_Q, 0>(
            &tensor_map_weights, &smem.full_q_barriers[q_stage_idx],
            smem.smem_weights[q_stage_idx], 0, q_token_base);
        smem.full_q_barriers[q_stage_idx].arrive_and_expect_tx(
            SMEM_Q_SIZE_PER_STAGE + SMEM_SF_Q_SIZE_PER_STAGE +
            SMEM_WEIGHT_SIZE_PER_STAGE);
      }
    }
    __syncwarp();
  } else if (warp_idx == kSpecWarpStart + 1) {
    cutlass::arch::warpgroup_reg_dealloc<56>();

    auto scheduler = make_scheduler(sm_idx);
    uint32_t cached_kv_page_base = 0;
    uint32_t cached_kv_page_coord = 0;
    // NOTES: split index for paged scheduler, token offset for contiguous-KV
    // scheduler.
    uint32_t q_block_idx, kv_base, num_kv_splits;
    while (scheduler.next_q_block(q_block_idx, kv_base, num_kv_splits)) {
      cached_kv_page_base = cute::numeric_limits<uint32_t>::max();
#pragma unroll 1
      for (uint32_t kv_split_idx = 0; kv_split_idx < num_kv_splits;
           ++kv_split_idx) {
        if constexpr (decltype(scheduler)::kIsPaged) {
          constexpr uint32_t kPageKV = decltype(scheduler)::kPageKV;
          constexpr uint32_t kNumPagesPerSplit =
              decltype(scheduler)::kNumPagesPerSplit;
          DG_STATIC_ASSERT(kNumPagesPerSplit <= 32,
                           "Split spans more pages than a warp can cache");

          const uint32_t kv_page_base =
              (kv_base + kv_split_idx) * kNumPagesPerSplit;
          if (kv_page_base < cached_kv_page_base or
              kv_page_base + kNumPagesPerSplit > cached_kv_page_base + 32) {
            cached_kv_page_base = (kv_page_base / 32) * 32;
            cached_kv_page_coord = scheduler.get_kv_page_coord_by_page_offset(
                cached_kv_page_base + lane_idx);
          }

          CUTE_TIE_DECL(kv_pipeline.advance(), kv_stage_idx, kv_phase);
          if (cute::elect_one_sync())
            smem.empty_kv_barriers[kv_stage_idx].wait(kv_phase ^ 1);
          __syncwarp();

          int page_coords[kNumPagesPerSplit];
#pragma unroll
          for (uint32_t page_idx = 0; page_idx < kNumPagesPerSplit;
               ++page_idx) {
            const auto src_lane =
                static_cast<int>(kv_page_base - cached_kv_page_base + page_idx);
            page_coords[page_idx] =
                __shfl_sync(0xffffffff, cached_kv_page_coord, src_lane);
          }

          if (cute::elect_one_sync()) {
#pragma unroll
            for (uint32_t page_idx = 0; page_idx < kNumPagesPerSplit;
                 ++page_idx) {
              tma::copy<kHeadDim, kPageKV, 0, qk_dtype_t, true>(
                  &tensor_map_kv, &smem.full_kv_barriers[kv_stage_idx],
                  smem.smem_kv[kv_stage_idx] +
                      page_idx * kPageKV * kNumQKBytesPerToken,
                  0, 0, 1, page_coords[page_idx]);
              tma::copy<kPageKV, 1, 0>(
                  &tensor_map_sf_kv, &smem.full_kv_barriers[kv_stage_idx],
                  smem.smem_sf_kv[kv_stage_idx] + page_idx * kPageKV, 0,
                  page_coords[page_idx]);
            }
            smem.full_kv_barriers[kv_stage_idx].arrive_and_expect_tx(
                SMEM_KV_SIZE_PER_STAGE + SMEM_SF_KV_SIZE_PER_STAGE);
          }
          __syncwarp();
        } else if (cute::elect_one_sync()) {
          CUTE_TIE_DECL(kv_pipeline.advance(), kv_stage_idx, kv_phase);
          smem.empty_kv_barriers[kv_stage_idx].wait(kv_phase ^ 1);

          const uint32_t kv_tma_offset =
              scheduler.get_kv_tma_offset(kv_base, kv_split_idx);
          tma::copy<kHeadDim, SPLIT_KV, 0>(
              &tensor_map_kv, &smem.full_kv_barriers[kv_stage_idx],
              smem.smem_kv[kv_stage_idx], 0, kv_tma_offset);
          tma::copy<SPLIT_KV, 1, 0>(
              &tensor_map_sf_kv, &smem.full_kv_barriers[kv_stage_idx],
              smem.smem_sf_kv[kv_stage_idx], kv_tma_offset, 0);
          smem.full_kv_barriers[kv_stage_idx].arrive_and_expect_tx(
              SMEM_KV_SIZE_PER_STAGE + SMEM_SF_KV_SIZE_PER_STAGE);
        }
        __syncwarp();
      }
    }
  } else if (warp_idx == kSpecWarpStart + 2) {
    cutlass::arch::warpgroup_reg_dealloc<56>();
    DG_TRAP_ONLY_DEVICE_ASSERT(ptx::ld_shared(&smem.tmem_ptr_in_smem) == 0);

    auto utccp_required_smem_warp_transpose = [&](const uint32_t* smem_ptr) {
      DG_STATIC_ASSERT(kNumUTCCPAlignedElems == 128,
                       "Invalid aligned elements");
      uint32_t values[4];
#pragma unroll
      for (uint32_t i = 0; i < 4; ++i)
        values[i] = ptx::ld_shared(smem_ptr + i * 32 + lane_idx);
      __syncwarp();
      ptx::st_shared(smem_ptr + lane_idx * 4, values[0], values[1], values[2],
                     values[3]);
    };

    auto sf_desc = mma::sm100::make_sf_desc(nullptr);

    auto scheduler = make_scheduler(sm_idx);
    // NOTES: split index for paged scheduler, token offset for contiguous-KV
    // scheduler.
    uint32_t q_block_idx, kv_base, num_kv_splits;
    while (scheduler.next_q_block(q_block_idx, kv_base, num_kv_splits)) {
      CUTE_TIE_DECL(q_pipeline.advance(), q_stage_idx, q_phase);
      smem.full_q_barriers[q_stage_idx].wait(q_phase);

      if constexpr (kIsMXSF) {
#pragma unroll
        for (uint32_t i = 0; i < kNumSFQ / kNumUTCCPAlignedElems; ++i) {
          auto smem_ptr =
              smem.smem_sf_q[q_stage_idx] + i * kNumUTCCPAlignedElems;
          utccp_required_smem_warp_transpose(smem_ptr);
        }
        cutlass::arch::fence_view_async_shared();
#pragma unroll
        for (uint32_t i = 0; i < kNumSFQ / kNumUTCCPAlignedElems; ++i) {
          auto smem_ptr =
              smem.smem_sf_q[q_stage_idx] + i * kNumUTCCPAlignedElems;
          mma::sm100::replace_smem_desc_addr(sf_desc, smem_ptr);
          if (cute::elect_one_sync())
            cute::SM100_UTCCP_4x32dp128bit_1cta::copy(
                sf_desc, kTmemStartColOfSFQ + i * 4);
          __syncwarp();
        }
      }

      for (uint32_t kv_split_idx = 0; kv_split_idx < num_kv_splits;
           ++kv_split_idx) {
        CUTE_TIE_DECL(kv_pipeline.advance(), kv_stage_idx, kv_phase);
        smem.full_kv_barriers[kv_stage_idx].wait(kv_phase);

        if constexpr (kIsMXSF) {
#pragma unroll
          for (uint32_t i = 0; i < kNumSFKV / kNumUTCCPAlignedElems; ++i) {
            auto smem_ptr =
                smem.smem_sf_kv[kv_stage_idx] + i * kNumUTCCPAlignedElems;
            utccp_required_smem_warp_transpose(smem_ptr);
          }
          cutlass::arch::fence_view_async_shared();
        }

        if (cute::elect_one_sync()) {
          if constexpr (kIsMXSF) {
#pragma unroll
            for (uint32_t i = 0; i < kNumSFKV / kNumUTCCPAlignedElems; ++i) {
              auto smem_ptr =
                  smem.smem_sf_kv[kv_stage_idx] + i * kNumUTCCPAlignedElems;
              mma::sm100::replace_smem_desc_addr(sf_desc, smem_ptr);
              cute::SM100_UTCCP_4x32dp128bit_1cta::copy(
                  sf_desc, kTmemStartColOfSFKV + i * 4);
            }
          }
#pragma unroll
          for (uint32_t i = 0; i < kNumMathWarpGroups; ++i) {
            CUTE_TIE_DECL(tmem_pipeline.advance(), tmem_stage_idx, tmem_phase);
            uint32_t tmem_addr = tmem_stage_idx * UMMA_N;

            smem.empty_tmem_barriers[tmem_stage_idx].wait(tmem_phase ^ 1);
            ptx::tcgen05_after_thread_sync();

            if constexpr (kIsMXSF) {
              DG_STATIC_ASSERT((not kIsFP4 and kHeadDim == 32) or
                                   kHeadDim == 64 or kHeadDim == 128,
                               "Invalid head dim");

              constexpr uint32_t kPackFactor =
                  get_smem_pack_factor<qk_dtype_t>();
              constexpr uint32_t kQKSwizzleMode = kHeadDim / kPackFactor;

              using mma_op_t =
                  cute::conditional_t<kIsFP4, ptx::SM100_MMA_MXF4_SS,
                                      ptx::SM100_MMA_MXF8F6F4_SS>;
              auto instr_desc = cute::UMMA::make_instr_desc_block_scaled<
                  qk_dtype_t, qk_dtype_t, float, cutlass::float_ue8m0_t, UMMA_M,
                  UMMA_N, cute::UMMA::Major::K, cute::UMMA::Major::K>();
#pragma unroll
              for (uint32_t k = 0; k < kHeadDim / UMMA_K; ++k) {
                auto runtime_instr_desc =
                    mma::sm100::make_runtime_instr_desc_with_sf_id(
                        instr_desc, k * kPackFactor, k * kPackFactor);
                auto a_desc =
                    mma::sm100::make_umma_desc<cute::UMMA::Major::K, 0,
                                               kHeadDim, kQKSwizzleMode>(
                        smem.smem_kv[kv_stage_idx], i * UMMA_M, k * UMMA_K);
                auto b_desc =
                    mma::sm100::make_umma_desc<cute::UMMA::Major::K, 0,
                                               kHeadDim, kQKSwizzleMode>(
                        smem.smem_q[q_stage_idx], 0, k * UMMA_K);
                mma_op_t::fma(a_desc, b_desc, tmem_addr, k, runtime_instr_desc,
                              kTmemStartColOfSFKV + i * 4, kTmemStartColOfSFQ);
              }
            } else {
              auto instr_desc = cute::UMMA::make_instr_desc<
                  cutlass::float_e4m3_t, cutlass::float_e4m3_t, float, UMMA_M,
                  UMMA_N, cute::UMMA::Major::K, cute::UMMA::Major::K>();
              auto runtime_instr_desc =
                  cute::UMMA::make_runtime_instr_desc(instr_desc);
#pragma unroll
              for (uint32_t k = 0; k < kHeadDim / UMMA_K; ++k) {
                auto a_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K,
                                                         0, kHeadDim, kHeadDim>(
                    smem.smem_kv[kv_stage_idx], i * UMMA_M, k * UMMA_K);
                auto b_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K,
                                                         0, kHeadDim, kHeadDim>(
                    smem.smem_q[q_stage_idx], 0, k * UMMA_K);
                ptx::SM100_MMA_F8F6F4_SS::fma(a_desc, b_desc, tmem_addr, k,
                                              runtime_instr_desc);
              }
            }

            ptx::umma_arrive_no_elect(smem.full_tmem_barriers[tmem_stage_idx]);
          }
        }
        __syncwarp();
        if constexpr (kIsMXSF)
          cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(
              &smem.empty_kv_barriers[kv_stage_idx]));
      }
      smem.empty_q_barriers[q_stage_idx].arrive();
    }
  } else if (warp_idx == kSpecWarpStart + 3) {
    cutlass::arch::warpgroup_reg_dealloc<56>();
  } else if (warp_idx < kSpecWarpStart) {
    cutlass::arch::warpgroup_reg_alloc<224>();

    constexpr bool kLoadSeqBounds = true;
    uint32_t seq_k_start[4];
    uint32_t seq_k_end[4];
    const auto math_warpgroup_idx = warp_idx / 4;
    const auto math_thread_idx = warp_idx * 32 + lane_idx;
    DG_STATIC_ASSERT(kNumMathWarpGroups <= kNumTmemStages,
                     "Math warp groups exceed TMEM stages");
    const unsigned query_base = math_warpgroup_idx * 4;

    constexpr bool kIsReduceBF16 = not cute::is_same_v<reduce_dtype_t, float>;
    DG_STATIC_ASSERT(kNumHeads % 4 == 0, "Head count must be a multiple of 4");
    DG_STATIC_ASSERT(8 <= UMMA_N and UMMA_N <= 256, "Invalid UMMA_N for MMA");
    using weights_reg_dtype_t =
        cute::conditional_t<kIsReduceBF16, nv_bfloat162, float>;
    constexpr uint32_t kNumWeightsRegPerToken =
        kIsReduceBF16 ? (kNumHeads / 2) : kNumHeads;
    weights_reg_dtype_t weights[4][32];
    float accum[kNumHeads];

    unsigned produce_epoch = 0;
    auto scheduler = make_scheduler(sm_idx);
    // NOTES: split index for paged scheduler, token offset for contiguous-KV
    // scheduler.
    uint32_t q_block_idx, kv_base, num_kv_splits;
    while (scheduler.next_q_block(q_block_idx, kv_base, num_kv_splits)) {
      CUTE_TIE_DECL(q_pipeline.advance(), q_stage_idx, q_phase);
      smem.full_q_barriers[q_stage_idx].wait(q_phase);

      const auto process_q_block = [&](auto num_valid_tokens_t) {
        constexpr uint32_t kNumValidTokens = 4;
        uint32_t lane_counts[4] = {};
        uint32_t flush_base = 0;
        float gate_score[4];

#pragma unroll
        for (uint32_t i = 0; i < kNumValidTokens; ++i) {
          const uint32_t row =
              scheduler.get_logits_row(q_block_idx, query_base + i);
          // Use the raw-score threshold estimated from the distributed sample.
          // This threshold remains fixed for the complete scan.
          gate_score[i] = -CUDART_INF_F;
          seq_k_start[i] = cute::min(scheduler.cu_seq_len_k_start[row],
                                     scheduler.num_kv_tokens);
          seq_k_end[i] = cute::min(scheduler.cu_seq_len_k_end[row],
                                   scheduler.num_kv_tokens);
          const auto smem_weights_row =
              smem.smem_weights[q_stage_idx] +
              (query_base + i) * kNumWeightElementsPerRow;
          if constexpr (kIsReduceBF16) {
            // Load two bf16 weights at a time as one packed shared u32
            const auto packed_row =
                reinterpret_cast<const uint32_t*>(smem_weights_row);
#pragma unroll
            for (uint32_t j = 0; j < kNumHeads / 2; ++j) {
              const auto packed = ptx::ld_shared(packed_row + j);
              weights[i][j] = ptx::exchange(
                  *reinterpret_cast<const nv_bfloat162*>(&packed), 0);
            }
          } else {
#pragma unroll
            for (unsigned j = 0; j < 32; ++j)
              weights[i][j] = 0.5f * ptx::ld_shared(smem_weights_row + j);
          }
        }

        for (uint32_t kv_split_idx = 0; kv_split_idx < num_kv_splits;
             ++kv_split_idx) {
          if ((kv_split_idx % (FallbackSegment / 256)) == 0) {
            __syncwarp();
            for (unsigned i = 0; i < 4; ++i)
              gate_score[i] =
                  __uint_as_float(atomicAdd(stream.gates + query_base + i, 0u));
          }
          CUTE_TIE_DECL(kv_pipeline.advance(), kv_stage_idx, kv_phase);
          smem.full_kv_barriers[kv_stage_idx].wait(kv_phase);
#pragma unroll
          for (unsigned kv_half = 0; kv_half < 2; ++kv_half) {
            const unsigned kv_lane = (math_thread_idx % 128) + kv_half * 128;
            const unsigned kv_offset =
                scheduler.get_logits_col(kv_base, kv_split_idx, kv_lane);
            const float scale_kv =
                ptx::ld_shared(smem.smem_sf_kv[kv_stage_idx] + kv_lane);
            CUTE_TIE_DECL(tmem_pipeline.advance(), tmem_stage_idx, tmem_phase);
            smem.full_tmem_barriers[tmem_stage_idx].wait(tmem_phase);
            ptx::tcgen05_after_thread_sync();
            // Both KV half-MMAs have committed before the shared stage is
            // freed.
            if (kv_half == 1) {
              cutlass::arch::fence_view_async_shared();
              smem.empty_kv_barriers[kv_stage_idx].arrive();
            }

            static_assert(!kIsReduceBF16, "FP32-only pair reduction");
            float accum_pair[MQA_TMEM_ROWS][kNumHeads];
#pragma unroll
            for (uint32_t pair = 0; pair < kNumValidTokens;
                 pair += MQA_TMEM_ROWS) {
              // Load only MQA_TMEM_ROWS query rows at once to limit
              // live accumulator registers during score reduction.
              const uint32_t tmem_addr =
                  tmem_stage_idx * UMMA_N + (query_base + pair) * kNumHeads;
              ptx::tmem_load_32dp32b<MQA_TMEM_ROWS * kNumHeads>(
                  tmem_addr, reinterpret_cast<uint32_t*>(accum_pair));
              cutlass::arch::fence_view_async_tmem_load();
              if (pair + MQA_TMEM_ROWS >= kNumValidTokens) {
                ptx::tcgen05_before_thread_sync();
                smem.empty_tmem_barriers[tmem_stage_idx].arrive();
              }

              float2 sum_0[MQA_TMEM_ROWS] = {};
              float2 sum_1[MQA_TMEM_ROWS] = {};
#pragma unroll
              for (uint32_t j = 0; j < kNumHeads; j += 4) {
#pragma unroll
                for (uint32_t r = 0; r < MQA_TMEM_ROWS; ++r) {
                  if (pair + r >= kNumValidTokens) continue;
                  const uint32_t i = pair + r;
                  auto* a = accum_pair[r];
                  auto transform = [&](uint32_t at, float2 sum) {
                    const auto x = make_float2(a[at], a[at + 1]);
                    const auto abs_x =
                        make_float2(fabsf(a[at]), fabsf(a[at + 1]));
                    const auto w =
                        make_float2(weights[i][at], weights[i][at + 1]);
                    return __ffma2_rn(__fadd2_rn(x, abs_x), w, sum);
                  };
                  sum_0[r] = transform(j, sum_0[r]);
                  sum_1[r] = transform(j + 2, sum_1[r]);
                }
              }
#pragma unroll
              for (uint32_t r = 0; r < MQA_TMEM_ROWS; ++r) {
                if (pair + r >= kNumValidTokens) continue;
                const uint32_t i = pair + r;
                const auto sum = __fadd2_rn(sum_0[r], sum_1[r]);
                const float reduced = sum.x + sum.y;
                const float bq = reduced * static_cast<float>(scale_kv);
                const uint32_t row =
                    scheduler.get_logits_row(q_block_idx, query_base + i);
                bool valid = kv_offset >= seq_k_start[i] &&
                             kv_offset < seq_k_end[i] && isfinite(bq) &&
                             bq > gate_score[i];
                unsigned position =
                    (kv_split_idx % (FallbackSegment / 256)) * 256 + kv_lane;
                stream.ring[produce_epoch & 1][query_base + i][position] =
                    valid ? bq : -CUDART_INF_F;
              }
            }
          }  // KV halves; query ownership is independent of the KV half.
          if (((kv_split_idx + 1) % (FallbackSegment / 256)) == 0 ||
              kv_split_idx + 1 == num_kv_splits) {
            __threadfence_block();

            fb_sync();
            unsigned tid = math_thread_idx;
            unsigned row_base = scheduler.get_logits_row(q_block_idx, 0);
            for (unsigned local = 0; local < 8; ++local)
              if (stream.kept[local] > FallbackRetain - FallbackSegment)
                fb_compact(stream, buffers, local, row_base + local, tid);
            fb_sync();
            unsigned local = tid / 32;
            fb_select(stream, buffers, local, row_base + local, produce_epoch,
                      kv_base + (kv_split_idx / (FallbackSegment / 256)) *
                                    FallbackSegment,
                      (kv_split_idx % (FallbackSegment / 256) + 1) * 256,
                      tid % 32, false);
            fb_sync();
            if (kv_split_idx + 1 == num_kv_splits)
              for (unsigned local = 0; local < 8; ++local)
                if (stream.kept[local] > 2048)
                  fb_compact(stream, buffers, local, row_base + local, tid,
                             true);
            fb_sync();
            ++produce_epoch;
          }
        }
      };

      if constexpr (decltype(scheduler)::kHasPartialBlock)
        dispatch_num_block_tokens<BLOCK_Q>(
            scheduler.get_num_block_tokens(q_block_idx), process_q_block);
      else
        process_q_block(cute::Int<BLOCK_Q>{});

      unsigned local = warp_idx,
               row = scheduler.get_logits_row(q_block_idx, local),
               count = stream.kept[local];
      if (row < scheduler.num_q_tokens)
        for (unsigned j = lane_idx; j < 2048; j += 32)
          buffers.output[(uint64_t)row * 2048 + j] =
              j < count ? buffers.indices[(uint64_t)row * FallbackRetain + j]
                        : -1;
      cutlass::arch::fence_view_async_shared();
      smem.empty_q_barriers[q_stage_idx].arrive();
    }

    cutlass::arch::NamedBarrier(kNumMathThreads, 0).sync();
    if (warp_idx == 0) cute::TMEM::Allocator1Sm().free(0, kNumTmemCols);
  }
}

}  // namespace seed_fallback
