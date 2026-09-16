#pragma once

// OPTIMIZED decode score kernel — based on
//   deep_gemm/impls/sm90_fp8_paged_mqa_logits.cuh  (vLLM 0.27.1)
//
// Unchanged: TMA warp-specialization, Q/KV pipelining, WGMMA, scheduler,
// ReLU-weight reduce. Same Hopper paged MQA datapath and smem pipes.
//
// [OPT] epilogue only:
//   Write compact (score, physical_idx) into
//     pack[row, kv_split, 0:SPLIT_KV]
//   instead of logits[row, logical]. Invalid tokens get -inf / -1.
//   Top-K is a separate kernel so it cannot steal WGMMA registers.

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>

#include <deep_gemm/common/cute_tie.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/mma/sm90.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <deep_gemm/ptx/wgmma.cuh>
#include <deep_gemm/scheduler/sm90_paged_mqa_logits.cuh>

namespace deep_gemm {

template <uint32_t kNextN, uint32_t kNumHeads,
          uint32_t kHeadDim, uint32_t BLOCK_KV,
          bool kIsContextLens2D, bool kIsVarlen,
          uint32_t kNumQStages, uint32_t kNumKVStages,
          uint32_t SPLIT_KV,
          uint32_t kNumTMAThreads, uint32_t kNumMathThreads,
          bool kEmitPhysical>
CUTLASS_GLOBAL __launch_bounds__(kNumTMAThreads + kNumMathThreads, 1)
void sm90_fp8_paged_mqa_logits_fused(
        const uint32_t batch_size,
        const uint32_t block_table_stride,
        const uint32_t* context_lens,
        const uint32_t* block_table,
        const uint32_t* indices,
        const uint32_t* schedule_meta,
        float* pack_scores,          // [num_rows, max_parts, SPLIT_KV]
        int32_t* pack_indices,       // [num_rows, max_parts, SPLIT_KV]
        const uint32_t max_parts,
        const uint32_t num_rows,
        const __grid_constant__ cute::TmaDescriptor tensor_map_q,
        const __grid_constant__ cute::TmaDescriptor tensor_map_kv,
        const __grid_constant__ cute::TmaDescriptor tensor_map_kv_scales,
        const __grid_constant__ cute::TmaDescriptor tensor_map_weights) {
    DG_STATIC_ASSERT(not kIsVarlen, "Varlen is not supported for SM90 paged MQA logits");
    DG_STATIC_ASSERT(BLOCK_KV == 64, "Invalid block size");

    using WGMMA = typename mma::sm90::FP8MMASelector<kNextN * kNumHeads>::type;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    const auto warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const auto warpgroup_idx = warp_idx / 4;
    const auto lane_idx = ptx::get_lane_idx();

    static constexpr uint32_t kNumMathWarpGroups = kNumMathThreads / 128;
    static constexpr uint32_t kTileTokens = BLOCK_KV * kNumMathWarpGroups;
    DG_STATIC_ASSERT(kNumTMAThreads == 128 and kNumMathThreads % 128 == 0, "Invalid threads");
    DG_STATIC_ASSERT(SPLIT_KV == kTileTokens, "Invalid `SPLIT_KV`");
    if (warp_idx == kNumMathThreads / 32 and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_q);
        cute::prefetch_tma_descriptor(&tensor_map_kv);
        cute::prefetch_tma_descriptor(&tensor_map_kv_scales);
        cute::prefetch_tma_descriptor(&tensor_map_weights);
    }
    __syncwarp();

    static constexpr uint32_t kSwizzleAlignment = kHeadDim * 8;
    static constexpr uint32_t SMEM_Q_SIZE_PER_STAGE = kNextN * kNumHeads * kHeadDim * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_WEIGHT_SIZE_PER_STAGE = kNextN * kNumHeads * sizeof(float);
    static constexpr uint32_t ALIGNED_SMEM_WEIGHT_SIZE_PER_STAGE = math::constexpr_align(SMEM_WEIGHT_SIZE_PER_STAGE, kSwizzleAlignment);
    static constexpr uint32_t SMEM_Q_PIPE_SIZE = kNumQStages * (SMEM_Q_SIZE_PER_STAGE + ALIGNED_SMEM_WEIGHT_SIZE_PER_STAGE) +
                                                 math::constexpr_align(kNumQStages * 8 * 2, kSwizzleAlignment);

    static constexpr uint32_t SMEM_KV_SIZE_PER_STAGE = BLOCK_KV * kHeadDim * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_KV_SCALE_SIZE_PER_STAGE = BLOCK_KV * sizeof(float);
    static constexpr uint32_t ALIGNED_SMEM_KV_SCALE_SIZE_PER_STAGE = math::constexpr_align(SMEM_KV_SCALE_SIZE_PER_STAGE, kSwizzleAlignment);
    static constexpr uint32_t SMEM_KV_PIPE_SIZE = kNumKVStages * (SMEM_KV_SIZE_PER_STAGE + ALIGNED_SMEM_KV_SCALE_SIZE_PER_STAGE) +
                                                  math::constexpr_align(kNumKVStages * 8 * 2, kSwizzleAlignment);

    extern __shared__ __align__(kSwizzleAlignment) uint8_t smem_buffer[];

    auto smem_q = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + SMEM_Q_SIZE_PER_STAGE * i);
    });
    auto smem_weights = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<float*>(smem_buffer + SMEM_Q_SIZE_PER_STAGE * kNumQStages + ALIGNED_SMEM_WEIGHT_SIZE_PER_STAGE * i);
    });
    auto q_barrier_ptr = reinterpret_cast<Barrier*>(smem_weights[kNumQStages]);
    auto full_q_barriers  = utils::PatternVisitor([&](const uint32_t& i) { return q_barrier_ptr + i; });
    auto empty_q_barriers = utils::PatternVisitor([&](const uint32_t& i) { return q_barrier_ptr + (kNumQStages + i); });

    const auto kv_group_idx = __shfl_sync(0xffffffff, threadIdx.x >= kNumMathThreads ? (threadIdx.x - kNumMathThreads) / 32 : warpgroup_idx, 0);

    const auto smem_offset = SMEM_Q_PIPE_SIZE + SMEM_KV_PIPE_SIZE * kv_group_idx;
    auto smem_kv = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + smem_offset + SMEM_KV_SIZE_PER_STAGE * i);
    });
    auto smem_kv_scales = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<float*>(smem_buffer + smem_offset + SMEM_KV_SIZE_PER_STAGE * kNumKVStages + ALIGNED_SMEM_KV_SCALE_SIZE_PER_STAGE * i);
    });
    auto kv_barrier_ptr = reinterpret_cast<Barrier*>(smem_kv_scales[kNumKVStages]);
    auto full_kv_barriers  = utils::PatternVisitor([&](const uint32_t& i) { return kv_barrier_ptr + i; });
    auto empty_kv_barriers = utils::PatternVisitor([&](const uint32_t& i) { return kv_barrier_ptr + kNumKVStages + i; });

    if (warp_idx >= kNumMathThreads / 32 and cute::elect_one_sync()) {
        if (kv_group_idx == 0) {
            #pragma unroll
            for (uint32_t i = 0; i < kNumQStages; ++ i) {
                full_q_barriers[i]->init(1);
                empty_q_barriers[i]->init(kNumMathThreads);
            }
        }
        if (kv_group_idx < kNumMathWarpGroups) {
            #pragma unroll
            for (uint32_t i = 0; i < kNumKVStages; ++ i) {
                full_kv_barriers[i]->init(1);
                empty_kv_barriers[i]->init(128);
            }
        }
        cutlass::arch::fence_barrier_init();
    }
    __syncthreads();

    constexpr uint32_t kNumTMARegisters = 64;
    constexpr uint32_t kNumMathRegisters = 104;

    cudaGridDependencySynchronize();

    auto scheduler = sched::SM90PagedMQALogitsScheduler<kNextN, kIsContextLens2D, kIsVarlen, BLOCK_KV, kNumMathWarpGroups, 1>(
        blockIdx.x, batch_size, context_lens, schedule_meta, indices);

    const auto get_q_pipeline = [=](const uint32_t& q_iter_idx) -> cute::tuple<uint32_t, uint32_t> {
        return {q_iter_idx % kNumQStages, (q_iter_idx / kNumQStages) & 1};
    };
    const auto get_kv_pipeline = [=](const uint32_t& kv_iter_idx) -> cute::tuple<uint32_t, uint32_t> {
        return {kv_iter_idx % kNumKVStages, (kv_iter_idx / kNumKVStages) & 1};
    };
    uint32_t q_iter_idx = 0, kv_iter_idx = 0;

    if (warp_idx >= kNumMathThreads / 32) {
        cutlass::arch::warpgroup_reg_dealloc<kNumTMARegisters>();
        if (kv_group_idx >= kNumMathWarpGroups)
            return;

        const auto issue_tma_q = [&](const uint32_t& stage_idx, const uint32_t& q_idx) {
            if (kv_group_idx == 0 and cute::elect_one_sync()) {
                tma::copy<kHeadDim, kNextN * kNumHeads, kHeadDim>(&tensor_map_q, full_q_barriers[stage_idx], smem_q[stage_idx], 0, q_idx * kNextN * kNumHeads);
                tma::copy<kNextN * kNumHeads, 1, 0>(&tensor_map_weights, full_q_barriers[stage_idx], smem_weights[stage_idx], 0, q_idx * kNextN);
                full_q_barriers[stage_idx]->arrive_and_expect_tx(SMEM_Q_SIZE_PER_STAGE + SMEM_WEIGHT_SIZE_PER_STAGE);
            }
        };

        uint32_t q_idx = batch_size, kv_idx, num_kv;
        uint32_t next_q_idx, next_kv_idx, next_num_kv;
        bool fetched_next_task;

        if ((fetched_next_task = scheduler.fetch_next_task(next_q_idx, next_kv_idx, next_num_kv)))
            issue_tma_q(0, next_q_idx), q_iter_idx = 1;

        int kv_block_idx_ptr = 32;
        uint32_t kv_block_idx_storage;

        while (fetched_next_task) {
            bool prefetch_q = (q_idx != next_q_idx and scheduler.exist_q_atom_idx(next_q_idx + 1));
            q_idx = next_q_idx;
            kv_idx = next_kv_idx;
            num_kv = next_num_kv;

            if (prefetch_q) {
                CUTE_TIE_DECL(get_q_pipeline(q_iter_idx ++), q_stage_idx, q_phase);
                empty_q_barriers[q_stage_idx]->wait(q_phase ^ 1);
                issue_tma_q(q_stage_idx, q_idx + 1);
            }

            if (kv_idx == 0 or kv_block_idx_ptr == 32) {
                kv_block_idx_ptr = 0;
                kv_block_idx_storage = (kv_idx + kv_group_idx + lane_idx * kNumMathWarpGroups < num_kv ?
                    block_table[q_idx * static_cast<uint64_t>(block_table_stride) + (kv_idx + kv_group_idx + lane_idx * kNumMathWarpGroups)] : 0);
            }
            const auto kv_block_idx = __shfl_sync(0xffffffff, kv_block_idx_storage, kv_block_idx_ptr ++);

            CUTE_TIE_DECL(get_kv_pipeline(kv_iter_idx ++), kv_stage_idx, kv_phase);
            empty_kv_barriers[kv_stage_idx]->wait(kv_phase ^ 1);

            if (cute::elect_one_sync()) {
                tma::copy<kHeadDim, BLOCK_KV, 0, __nv_fp8_e4m3, true>(&tensor_map_kv, full_kv_barriers[kv_stage_idx],
                                                                      smem_kv[kv_stage_idx], 0, 0, 1, kv_block_idx);
                tma::copy<BLOCK_KV, 1, 0>(&tensor_map_kv_scales, full_kv_barriers[kv_stage_idx],
                                          smem_kv_scales[kv_stage_idx], 0, kv_block_idx);
                full_kv_barriers[kv_stage_idx]->arrive_and_expect_tx(SMEM_KV_SIZE_PER_STAGE + SMEM_KV_SCALE_SIZE_PER_STAGE);
            }

            fetched_next_task = scheduler.fetch_next_task(next_q_idx, next_kv_idx, next_num_kv);
        }
        cudaTriggerProgrammaticLaunchCompletion();
    } else {
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

        float accum[WGMMA::kNumAccum], weights[kNextN][kNumHeads / 4];
        const auto sub_warp_offset = (warp_idx % 4) * 16;
        const auto v_0_offset = lane_idx / 4 + 0;
        const auto v_1_offset = lane_idx / 4 + 8;

        uint32_t q_idx = batch_size, kv_idx;
        uint32_t next_q_idx, next_kv_idx, next_num_kv;
        uint32_t q_stage_idx, q_phase;

        while (scheduler.fetch_next_task(next_q_idx, next_kv_idx, next_num_kv)) {
            if (q_idx != next_q_idx) {
                if (q_iter_idx > 0)
                    empty_q_barriers[(q_iter_idx - 1) % kNumQStages]->arrive();

                CUTE_TIE(get_q_pipeline(q_iter_idx ++), q_stage_idx, q_phase);
                full_q_barriers[q_stage_idx]->wait(q_phase);

                #pragma unroll
                for (uint32_t i = 0; i < kNextN; ++ i) {
                    #pragma unroll
                    for (uint32_t j = 0; j < kNumHeads / 4; ++ j)
                        weights[i][j] = ptx::ld_shared(smem_weights[q_stage_idx] + i * kNumHeads + (j / 2) * 8 + (j & 1) + (lane_idx % 4) * 2);
                }
            }

            q_idx = next_q_idx;
            kv_idx = next_kv_idx;

            CUTE_TIE_DECL(get_kv_pipeline(kv_iter_idx ++), kv_stage_idx, kv_phase);
            full_kv_barriers[kv_stage_idx]->wait(kv_phase);

            #pragma unroll
            for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                ptx::warpgroup_fence_operand(accum[i]);
            ptx::warpgroup_arrive();
            #pragma unroll
            for (uint32_t k = 0; k < kHeadDim / WGMMA::K; ++ k) {
                auto desc_a = mma::sm90::make_smem_desc(
                    smem_kv[kv_stage_idx] + k * WGMMA::K,
                    mma::sm90::to_swizzle_cute_type<kHeadDim>(), 0, kHeadDim * 8);
                auto desc_b = mma::sm90::make_smem_desc(
                    smem_q[q_stage_idx] + k * WGMMA::K,
                    mma::sm90::to_swizzle_cute_type<kHeadDim>(), 0, kHeadDim * 8);
                WGMMA::wgmma(desc_a, desc_b, accum, k);
            }
            ptx::warpgroup_commit_batch();
            #pragma unroll
            for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                ptx::warpgroup_fence_operand(accum[i]);

            float scale_kv_0 = ptx::ld_shared(smem_kv_scales[kv_stage_idx] + sub_warp_offset + v_0_offset);
            float scale_kv_1 = ptx::ld_shared(smem_kv_scales[kv_stage_idx] + sub_warp_offset + v_1_offset);

            ptx::warpgroup_wait<0>();
            empty_kv_barriers[kv_stage_idx]->arrive();

            static constexpr uint32_t kNumAccumPerReduce = kNumHeads / 2;
            const uint32_t bt_row = sched::SM90PagedMQALogitsScheduler<
                kNextN, kIsContextLens2D, kIsVarlen, BLOCK_KV, kNumMathWarpGroups, 1
            >::atom_to_block_table_row(q_idx);
            const uint32_t token_row = sched::SM90PagedMQALogitsScheduler<
                kNextN, kIsContextLens2D, kIsVarlen, BLOCK_KV, kNumMathWarpGroups, 1
            >::atom_to_token_idx(q_idx);
            const uint32_t logical_blk = kv_idx + kv_group_idx;
            const uint32_t part = kv_idx / kNumMathWarpGroups;
            const uint32_t page = (logical_blk < next_num_kv)
                ? block_table[bt_row * static_cast<uint64_t>(block_table_stride) + logical_blk]
                : 0u;

            #pragma unroll
            for (uint32_t i = 0; i < kNextN; ++ i) {
                auto shifted_accum = accum + i * kNumAccumPerReduce;
                const auto transform = [&](const uint32_t& j) {
                    return fmaxf(shifted_accum[j], 0) * weights[i][(j / 4) * 2 + (j & 1)];
                };

                float sum[4] = {transform(0), transform(1), transform(2), transform(3)};
                #pragma unroll
                for (uint32_t j = 1; j < kNumHeads / 8; ++ j) {
                    #pragma unroll
                    for (uint32_t k = 0; k < 4; k ++)
                        sum[k] += transform(j * 4 + k);
                }
                float v_0 = (sum[0] + sum[1]) * scale_kv_0;
                float v_1 = (sum[2] + sum[3]) * scale_kv_1;

                #pragma unroll
                for (uint32_t j = 0; j < 2; ++ j) {
                    const auto offset = static_cast<int>(1u << j);
                    v_0 += __shfl_xor_sync(0xffffffffu, v_0, offset);
                    v_1 += __shfl_xor_sync(0xffffffffu, v_1, offset);
                }

                const uint32_t lens_idx = kIsContextLens2D ? bt_row * kNextN + i : bt_row;
                const uint32_t ctx = context_lens[lens_idx];
                const uint32_t in0 = sub_warp_offset + v_0_offset;
                const uint32_t in1 = sub_warp_offset + v_1_offset;
                const uint32_t logical0 = logical_blk * BLOCK_KV + in0;
                const uint32_t logical1 = logical_blk * BLOCK_KV + in1;
                const uint32_t row = token_row + i;
                const bool in_range = (part < max_parts && row < num_rows);
                const long long base = in_range
                    ? (static_cast<long long>(row) * max_parts + part) * kTileTokens
                    : 0;

                const bool ok0 = logical0 < ctx;
                const bool ok1 = logical1 < ctx;
                const int32_t id0 = ok0
                    ? (kEmitPhysical ? static_cast<int32_t>(page * BLOCK_KV + in0)
                                     : static_cast<int32_t>(logical0))
                    : -1;
                const int32_t id1 = ok1
                    ? (kEmitPhysical ? static_cast<int32_t>(page * BLOCK_KV + in1)
                                     : static_cast<int32_t>(logical1))
                    : -1;

                if (in_range) {
                    pack_scores[base + kv_group_idx * BLOCK_KV + in0] = ok0 ? v_0 : -1.0e30f;
                    pack_indices[base + kv_group_idx * BLOCK_KV + in0] = id0;
                    pack_scores[base + kv_group_idx * BLOCK_KV + in1] = ok1 ? v_1 : -1.0e30f;
                    pack_indices[base + kv_group_idx * BLOCK_KV + in1] = id1;
                }
            }
        }
        cudaTriggerProgrammaticLaunchCompletion();
    }
}

}  // namespace deep_gemm
