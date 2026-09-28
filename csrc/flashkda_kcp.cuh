// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// KCP on FlashKDA kernels. Context-parallel KDA prefill runs FlashKDA's
// preparation (K1) once on a rank's local segments, then two recurrences
// (K2) over the same workspace:
//
//  * Summary: each segment's affine summary, from which ranks merge every
//    segment's initial state. The recurrence is linear and independent per
//    value column, so the summary is one recurrence over 2D value columns:
//    D real value columns from a zero state (S) and D zero-input columns from
//    the identity (the transition M). Each segment writes its final
//    value-first [2D, D] BF16 state, i.e. [S^T; M^T], to a caller-chosen row.
//  * Scan: FlashKDA's recurrence from the merged FP32 initial states over a
//    listed subset of the segments, writing the attention output.
//
// K1 and the scan are FlashKDA's own launches. The summary kernel is
// FlashKDA's K2 (vllm-project/FlashKDA csrc/smxx/fwd_kernel2.cuh, Apache-2.0)
// without the q/output path, with a zero or identity initial state per value
// slice and an indexed final state; like K2, it keeps the state in FP32
// between tiles.
#pragma once

#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/tensor_inl.h>

#include "fwd.h"
#include "smxx/fwd_kernel1.cuh"
#include "smxx/fwd_kernel2.cuh"

namespace kcp_flash {

// Shared memory of the summary recurrence: the resident state and K2's
// inputs other than q_decayed and Mqk, which only feed the output.
template <class Layouts, int InputStages>
struct SummaryStorage {
  using BF16 = cutlass::bfloat16_t;
  alignas(128) cute::ArrayEngine<
      BF16, cute::cosize_v<typename Layouts::StateSmemLayout>> state_acc;
  struct InputStorage {
    alignas(128)
        cute::ArrayEngine<BF16, cute::cosize_v<typename Layouts::VOLayout>> v;
    alignas(128) cute::ArrayEngine<
        BF16, cute::cosize_v<typename Layouts::BetaSmemLayout>> beta;
    alignas(128) cute::ArrayEngine<
        BF16, cute::cosize_v<typename Layouts::MMALayout>> k_decayed;
    alignas(128) cute::ArrayEngine<
        BF16, cute::cosize_v<typename Layouts::MMALayout>> k_restored;
    alignas(128) cute::ArrayEngine<
        float, cute::cosize_v<typename Layouts::GTotalLayout>> g_total;
    alignas(128)
        cute::ArrayEngine<BF16, cute::cosize_v<typename Layouts::LMLayout>> INV;
  };
  InputStorage input[InputStages];
  typename cutlass::PipelineTmaAsync<InputStages>::SharedStorage load_pipeline;
};

template <class TmaLoadV, class TmaLoadBeta, class TmaStoreState, int CHUNK,
          int D, int InputStages, int NumThreads, int MinBlocks,
          typename SeqlenT, int VD>
__global__ void __launch_bounds__(NumThreads, MinBlocks)
    summary_recurrence(CUTE_GRID_CONSTANT TmaLoadV const tma_load_v,
                       CUTE_GRID_CONSTANT TmaLoadBeta const tma_load_beta,
                       CUTE_GRID_CONSTANT TmaStoreState const tma_store_summary,
                       int T_total, int H, int num_rows,
                       SeqlenT const* cu_seqlens, int64_t const* out_rows,
                       int total_tiles, cutlass::bfloat16_t const* ws_kd,
                       cutlass::bfloat16_t const* ws_kr, float const* ws_gt,
                       cutlass::bfloat16_t const* ws_inv) {
  using BF16 = cutlass::bfloat16_t;
  using Layouts = K2Layouts<D, CHUNK, VD>;
  using MMALayout = typename Layouts::MMALayout;
  using TransposedMMALayout = typename Layouts::TransposedMMALayout;
  using VOLayout = typename Layouts::VOLayout;
  using BetaSmemLayout = typename Layouts::BetaSmemLayout;
  using StateSmemLayout = typename Layouts::StateSmemLayout;
  using TransposedStateSmemLayout = typename Layouts::TransposedStateSmemLayout;
  using GTotalLayout = typename Layouts::GTotalLayout;
  using LMLayout = typename Layouts::LMLayout;
  using TMAVOLayout = typename Layouts::TMAVOLayout;
  using TMABetaSmemLayout = typename Layouts::TMABetaSmemLayout;
  using TMAStateSmemLayout = typename Layouts::TMAStateSmemLayout;
  using SharedStorageT = SummaryStorage<Layouts, InputStages>;

  extern __shared__ __align__(128) unsigned char shared_mem[];
  SharedStorageT& shared_storage =
      *reinterpret_cast<SharedStorageT*>(shared_mem);

  constexpr int kWarpSize = 32;
  constexpr int kComputeThreads = 128;
  // Value slices: D / VD real columns, then D / VD identity columns.
  constexpr int kRealSlices = D / VD;
  constexpr int kVSlices = 2 * kRealSlices;

  // Identity slices have zero input and skip the v tile.
  constexpr uint32_t kVBytes =
      uint32_t(cute::cosize_v<VOLayout>) * uint32_t(sizeof(BF16));
  constexpr uint32_t kTmaTransactionBytes =
      uint32_t(32) * uint32_t(sizeof(BF16)) +
      uint32_t(cute::cosize_v<MMALayout>) * uint32_t(sizeof(BF16)) * 2 +
      uint32_t(cute::cosize_v<GTotalLayout>) * uint32_t(sizeof(float)) +
      uint32_t(cute::cosize_v<LMLayout>) * uint32_t(sizeof(BF16));

  int warp_id = cutlass::canonical_warp_idx_sync();
  bool lane_predicate = cute::elect_one_sync();
  WarpRole warp_role = WarpRole::NonParticipant;
  if (warp_id < kComputeThreads / kWarpSize) {
    warp_role = WarpRole::MMA;
  } else if (warp_id < kComputeThreads / kWarpSize + 1) {
    warp_role = WarpRole::LOAD_QKG;
  }

  int v_idx = int(blockIdx.x) % kVSlices;
  int head_idx = int(blockIdx.x) / kVSlices;
  bool identity = v_idx >= kRealSlices;

  using LoadPipelineState = cutlass::PipelineState<InputStages>;
  using LoadPipeline = cutlass::PipelineTmaAsync<InputStages>;
  LoadPipeline load_pipeline = make_load_pipeline<InputStages>(
      shared_storage.load_pipeline,
      kTmaTransactionBytes + (identity ? 0u : kVBytes), warp_role, 1,
      kComputeThreads);
  cutlass::pipeline_init_wait(1);
  // Issue segments in reverse order, as FlashKDA does.
  int seq_idx = int(gridDim.y) - 1 - int(blockIdx.y);
  int64_t bos = cu_seqlens[seq_idx];
  int64_t eos = cu_seqlens[seq_idx + 1];
  int tile_base = 0;
  for (int i = 0; i < seq_idx; i++) {
    tile_base += (int(cu_seqlens[i + 1] - cu_seqlens[i]) + CHUNK - 1) / CHUNK;
  }
  int seq_len = int(eos - bos);
  int t_tiles = (seq_len + CHUNK - 1) / CHUNK;

  // Initial state: zero for S, the identity for M's columns.
  {
    Tensor s_acc = make_tensor(make_smem_ptr(shared_storage.state_acc.begin()),
                               StateSmemLayout{});
    int column_base = (v_idx - kRealSlices) * VD;
    for (int i = threadIdx.x; i < VD * D; i += NumThreads) {
      int row = i / D;
      int col = i % D;
      s_acc(row, col) =
          BF16(identity && col == column_base + row ? 1.0f : 0.0f);
    }
    cutlass::arch::fence_view_async_shared();
    __syncthreads();
  }

  if (warp_role == WarpRole::LOAD_QKG && lane_predicate) {
    Tensor g_v = tma_load_v.get_tma_tensor(make_shape(H, T_total, D));
    Tensor g_beta = tma_load_beta.get_tma_tensor(make_shape(H * T_total));
    LoadPipelineState load_write =
        cutlass::make_producer_start_state<LoadPipeline>();
    auto cta_tma_load_v = tma_load_v.get_slice(Int<0>{});
    auto cta_tma_load_beta = tma_load_beta.get_slice(Int<0>{});
    for (int t = 0; t < t_tiles; ++t) {
      load_pipeline.producer_acquire(load_write);
      using LoadBarrierType = typename LoadPipeline::ProducerBarrierType;
      LoadBarrierType* tma_barrier =
          load_pipeline.producer_get_barrier(load_write);
      int stage = load_write.index();
      int ws_idx = head_idx * total_tiles + tile_base + t;

      if (!identity) {
        auto v_off = g_v.layout()(head_idx, int(bos) + t * CHUNK, v_idx * VD);
        Tensor g_v_tile = make_tensor(
            g_v.data() + v_off,
            make_layout(make_shape(Int<1>{}, Int<CHUNK>{}, Int<VD>{}),
                        stride(g_v.layout())));
        Tensor s_v_tile =
            make_tensor(make_smem_ptr(shared_storage.input[stage].v.begin()),
                        TMAVOLayout{});
        cute::copy(tma_load_v.with(*tma_barrier),
                   cta_tma_load_v.partition_S(g_v_tile),
                   cta_tma_load_v.partition_D(s_v_tile));
      }

      int beta_linear = head_idx * T_total + (int(bos) + t * CHUNK);
      int beta_aligned = beta_linear & ~7;
      auto beta_off = g_beta.layout()(beta_aligned);
      Tensor g_beta_tile =
          make_tensor(g_beta.data() + beta_off, BetaSmemLayout{});
      Tensor s_beta_tile =
          make_tensor(make_smem_ptr(shared_storage.input[stage].beta.begin()),
                      TMABetaSmemLayout{});
      cute::copy(tma_load_beta.with(*tma_barrier),
                 cta_tma_load_beta.partition_S(g_beta_tile),
                 cta_tma_load_beta.partition_D(s_beta_tile));

      cute::SM90_BULK_COPY_G2S::copy(
          ws_kd + int64_t(ws_idx) * (CHUNK * D),
          reinterpret_cast<uint64_t*>(tma_barrier),
          shared_storage.input[stage].k_decayed.begin(),
          int32_t(CHUNK * D * sizeof(BF16)));
      cute::SM90_BULK_COPY_G2S::copy(
          ws_kr + int64_t(ws_idx) * (CHUNK * D),
          reinterpret_cast<uint64_t*>(tma_barrier),
          shared_storage.input[stage].k_restored.begin(),
          int32_t(CHUNK * D * sizeof(BF16)));
      cute::SM90_BULK_COPY_G2S::copy(
          ws_gt + int64_t(ws_idx) * D, reinterpret_cast<uint64_t*>(tma_barrier),
          shared_storage.input[stage].g_total.begin(),
          int32_t(D * sizeof(float)));
      cute::SM90_BULK_COPY_G2S::copy(ws_inv + int64_t(ws_idx) * (CHUNK * CHUNK),
                                     reinterpret_cast<uint64_t*>(tma_barrier),
                                     shared_storage.input[stage].INV.begin(),
                                     int32_t(CHUNK * CHUNK * sizeof(BF16)));
      ++load_write;
    }
    load_pipeline.producer_tail(load_write);
  }

  if (warp_role == WarpRole::MMA) {
    LoadPipelineState load_read;
    int compute_tid = threadIdx.x;
    constexpr int kValueBlocksPerWarp =
        VD / ((kComputeThreads / kWarpSize) * 16);
    static_assert(kValueBlocksPerWarp == 1 || kValueBlocksPerWarp == 2);

    Tensor resident_s_acc_T =
        make_tensor(make_smem_ptr(shared_storage.state_acc.begin()),
                    TransposedStateSmemLayout{});
    auto resident_mma =
        make_tiled_mma(MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>{},
                       Layout<Shape<_1, _1>>{}, Tile<_16, _16, _16>{});
    const int resident_warp_id = compute_tid / kWarpSize;
    const int resident_lane_id = compute_tid % kWarpSize;
    auto resident_thr_mma = resident_mma.get_slice(resident_lane_id);
    auto resident_load_c =
        make_tiled_copy_C(Copy_Atom<SM75_U16x8_LDSM_T, BF16>{}, resident_mma);
    auto resident_thr_load_c = resident_load_c.get_slice(resident_lane_id);
    Tensor resident_state_ref =
        local_tile(resident_s_acc_T, make_shape(Int<16>{}, Int<16>{}),
                   make_coord(0, resident_warp_id * kValueBlocksPerWarp));
    auto resident_c_ref = resident_thr_mma.partition_C(resident_state_ref);
    using ResidentStateFragment = decltype(make_fragment_like<BF16>(
        resident_thr_mma.make_fragment_C(resident_c_ref)));
    // The state stays in FP32 between tiles, as in FlashKDA's K2.
    using ResidentStateAcc =
        decltype(resident_thr_mma.make_fragment_C(resident_c_ref));
    constexpr int kResidentStateRowBlocks = D / 16;
    ResidentStateAcc resident_state[kValueBlocksPerWarp]
                                   [kResidentStateRowBlocks];

#pragma unroll
    for (int m = 0; m < kResidentStateRowBlocks; ++m) {
#pragma unroll
      for (int bi = 0; bi < kValueBlocksPerWarp; ++bi) {
        Tensor state_block = local_tile(
            resident_s_acc_T, make_shape(Int<16>{}, Int<16>{}),
            make_coord(m, resident_warp_id * kValueBlocksPerWarp + bi));
        ResidentStateFragment state_bf16;
        copy(resident_load_c, resident_thr_load_c.partition_S(state_block),
             resident_thr_load_c.retile_D(state_bf16));
        cute::transform(state_bf16, resident_state[bi][m], ToF32{});
      }
    }

    for (int t = 0; t < t_tiles; ++t) {
      load_pipeline.consumer_wait(load_read);
      int load_stage = load_read.index();

      Tensor v_tile =
          make_tensor(make_smem_ptr(shared_storage.input[load_stage].v.begin()),
                      VOLayout{});
      Tensor beta_tile = make_tensor(
          make_smem_ptr(shared_storage.input[load_stage].beta.begin()),
          BetaSmemLayout{});
      int beta_smem_offset = (head_idx * T_total + int(bos) + t * CHUNK) & 7;
      Tensor k_decayed = make_tensor(
          make_smem_ptr(shared_storage.input[load_stage].k_decayed.begin()),
          MMALayout{});
      Tensor g_total = make_tensor(
          make_smem_ptr(shared_storage.input[load_stage].g_total.begin()),
          GTotalLayout{});
      Tensor INV = make_tensor(
          make_smem_ptr(shared_storage.input[load_stage].INV.begin()),
          LMLayout{});
      Tensor s_acc = make_tensor(
          make_smem_ptr(shared_storage.state_acc.begin()), StateSmemLayout{});
      Tensor s_acc_T =
          make_tensor(make_smem_ptr(shared_storage.state_acc.begin()),
                      TransposedStateSmemLayout{});
      {
        Tensor k_restored_t = make_tensor(
            make_smem_ptr(shared_storage.input[load_stage].k_restored.begin()),
            TransposedMMALayout{});
        auto mma =
            make_tiled_mma(MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>{},
                           Layout<Shape<_1, _1>>{}, Tile<_16, _16, _16>{});
        const int warp = compute_tid / 32;
        const int lane_id = compute_tid % 32;
        const int group_id = (lane_id / 4) % 8;
        ThrMMA thr_mma = mma.get_slice(lane_id);
        auto smem_tiled_copy_A =
            make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, BF16>{}, mma);
        auto smem_thr_copy_A = smem_tiled_copy_A.get_thread_slice(lane_id);
        auto smem_tiled_copy_A_T =
            make_tiled_copy_A(Copy_Atom<SM75_U16x8_LDSM_T, BF16>{}, mma);
        auto smem_thr_copy_A_T = smem_tiled_copy_A_T.get_thread_slice(lane_id);
        auto smem_tiled_load_C =
            make_tiled_copy_C(Copy_Atom<SM75_U32x4_LDSM_N, BF16>{}, mma);
        auto smem_thr_load_C = smem_tiled_load_C.get_slice(lane_id);
        auto smem_tiled_store_C_T =
            make_tiled_copy_C(Copy_Atom<SM90_U16x8_STSM_T, BF16>{}, mma);
        auto smem_thr_store_C_T = smem_tiled_store_C_T.get_slice(lane_id);

        Tensor A_ref = local_tile(k_decayed, make_shape(Int<16>{}, Int<16>{}),
                                  make_coord(0, 0));
        Tensor B_ref = local_tile(s_acc, make_shape(Int<16>{}, Int<16>{}),
                                  make_coord(0, 0));
        Tensor C_ref = local_tile(v_tile, make_shape(Int<16>{}, Int<16>{}),
                                  make_coord(0, 0));
        Tensor tCrAi_k =
            make_fragment_like<BF16>(thr_mma.partition_fragment_A(A_ref));
        auto tCrAi_k_view = smem_thr_copy_A.retile_D(tCrAi_k);
        auto tCrA_k = thr_mma.partition_fragment_A(A_ref);
        auto tCrB = thr_mma.partition_fragment_B(B_ref);
        auto tCrC_ref = thr_mma.partition_C(C_ref);
        using AccFragT = decltype(thr_mma.make_fragment_C(tCrC_ref));
        using SFragT = decltype(make_fragment_like<BF16>(
            thr_mma.make_fragment_C(tCrC_ref)));
        using AFragT = decltype(thr_mma.partition_fragment_A(A_ref));
        using BFragT_u = decltype(thr_mma.partition_fragment_B(B_ref));

        AccFragT u_acc[kValueBlocksPerWarp];
#pragma unroll
        for (int i = 0; i < kValueBlocksPerWarp; ++i) {
          u_acc[i] = thr_mma.make_fragment_C(tCrC_ref);
          clear(u_acc[i]);
        }

        // Phase 1: u = k_decayed @ s.
        constexpr int K_BLOCKS = decltype(cute::size<1>(k_decayed))::value / 16;
        copy(
            smem_tiled_copy_A,
            smem_thr_copy_A.partition_S(local_tile(
                k_decayed, make_shape(Int<16>{}, Int<16>{}), make_coord(0, 0))),
            tCrAi_k_view);
#pragma unroll
        for (int k = 0; k < K_BLOCKS; ++k) {
          cute::transform(tCrAi_k, tCrA_k, cute::identity{});
          if (k + 1 < K_BLOCKS) {
            copy(smem_tiled_copy_A,
                 smem_thr_copy_A.partition_S(
                     local_tile(k_decayed, make_shape(Int<16>{}, Int<16>{}),
                                make_coord(0, k + 1))),
                 tCrAi_k_view);
          }
#pragma unroll
          for (int bi = 0; bi < kValueBlocksPerWarp; ++bi) {
            movm_transpose_c_to_b_16x16(
                narrow_state<ResidentStateFragment>(resident_state[bi][k]),
                tCrB);
            gemm(thr_mma, tCrA_k(_, _, Int<0>{}), tCrB(_, _, Int<0>{}),
                 u_acc[bi]);
          }
        }

        // Phase 2: v (zero for identity columns), INV and beta.
        SFragT v_bf16[kValueBlocksPerWarp];
#pragma unroll
        for (int i = 0; i < kValueBlocksPerWarp; ++i) {
          if (identity) {
            cute::fill(v_bf16[i], BF16(0.0f));
          } else {
            Tensor v_block =
                local_tile(v_tile, make_shape(Int<16>{}, Int<16>{}),
                           make_coord(0, warp * kValueBlocksPerWarp + i));
            copy(smem_tiled_load_C, smem_thr_load_C.partition_S(v_block),
                 smem_thr_load_C.retile_D(v_bf16[i]));
          }
        }
        copy(smem_tiled_copy_A, smem_thr_copy_A.partition_S(INV), tCrAi_k_view);
        cute::transform(tCrAi_k, tCrA_k, cute::identity{});
        BF16 beta0 = BF16(sigmoid_tanh_approx_f32(
            float(beta_tile(beta_smem_offset + group_id))));
        BF16 beta1 = BF16(sigmoid_tanh_approx_f32(
            float(beta_tile(beta_smem_offset + group_id + 8))));

        // Phase 3: U = INV @ ((v - u) * beta), kept as MMA-B operands.
        BFragT_u tCrB_u_arr[kValueBlocksPerWarp];
        uint32_t u_b_regs[4];
#pragma unroll
        for (int i = 0; i < kValueBlocksPerWarp; ++i) {
          SFragT u_bf16;
          cute::transform(u_acc[i], u_bf16,
                          [] __device__(float x) { return BF16(x); });
#pragma unroll
          for (int a = 0; a < 2; ++a) {
#pragma unroll
            for (int d = 0; d < 2; ++d) {
              auto c0 = make_coord(make_coord(a, 0), 0, d);
              auto c1 = make_coord(make_coord(a, 1), 0, d);
              u_bf16(c0) = (v_bf16[i](c0) - u_bf16(c0)) * beta0;
              u_bf16(c1) = (v_bf16[i](c1) - u_bf16(c1)) * beta1;
            }
          }
          uint32_t* u_c = reinterpret_cast<uint32_t*>(&u_bf16(0));
#pragma unroll
          for (int r = 0; r < 4; ++r)
            SM75_U32x1_MOVM_T::copy(u_c[r], u_b_regs[r]);
          auto tCrB_u_tmp = thr_mma.partition_fragment_B(B_ref);
          uint32_t* b_dst = reinterpret_cast<uint32_t*>(&tCrB_u_tmp(0));
#pragma unroll
          for (int r = 0; r < 4; ++r) b_dst[r] = u_b_regs[r];
          clear(u_acc[i]);
          gemm(thr_mma, tCrA_k(_, _, Int<0>{}), tCrB_u_tmp(_, _, Int<0>{}),
               u_acc[i]);
          cute::transform(u_acc[i], u_bf16,
                          [] __device__(float x) { return BF16(x); });
#pragma unroll
          for (int r = 0; r < 4; ++r)
            SM75_U32x1_MOVM_T::copy(u_c[r], u_b_regs[r]);
          tCrB_u_arr[i] = thr_mma.partition_fragment_B(B_ref);
          b_dst = reinterpret_cast<uint32_t*>(&tCrB_u_arr[i](0));
#pragma unroll
          for (int r = 0; r < 4; ++r) b_dst[r] = u_b_regs[r];
        }

        // Phase 4: s = s * g_total + k_restored^T @ U.
        constexpr int PREFETCH = 1;
        constexpr int S_M_BLOCKS =
            decltype(cute::size<0>(k_restored_t))::value / 16;
        Tensor tCrAi_kr =
            make_fragment_like<BF16>(thr_mma.partition_fragment_A(A_ref));
        auto tCrAi_kr_view = smem_thr_copy_A_T.retile_D(tCrAi_kr);
        AFragT ring_A_kr[PREFETCH];
        float ring_g0[PREFETCH], ring_g1[PREFETCH];
#pragma unroll
        for (int i = 0; i < PREFETCH; ++i) {
          Tensor kr_block = local_tile(
              k_restored_t, make_shape(Int<16>{}, Int<16>{}), make_coord(i, 0));
          copy(smem_tiled_copy_A_T, smem_thr_copy_A_T.partition_S(kr_block),
               tCrAi_kr_view);
          cute::transform(tCrAi_kr, ring_A_kr[i], cute::identity{});
          ring_g0[i] = g_total(i * 16 + group_id);
          ring_g1[i] = g_total(i * 16 + group_id + 8);
        }
#pragma unroll
        for (int m = 0; m < S_M_BLOCKS; ++m) {
          const int slot = m % PREFETCH;
          float g0 = ring_g0[slot];
          float g1 = ring_g1[slot];
#pragma unroll
          for (int bi = 0; bi < kValueBlocksPerWarp; ++bi) {
            clear(u_acc[bi]);
            gemm(thr_mma, ring_A_kr[slot](_, _, Int<0>{}),
                 tCrB_u_arr[bi](_, _, Int<0>{}), u_acc[bi]);
          }
          if (m + PREFETCH < S_M_BLOCKS) {
            Tensor kr_next =
                local_tile(k_restored_t, make_shape(Int<16>{}, Int<16>{}),
                           make_coord(m + PREFETCH, 0));
            copy(smem_tiled_copy_A_T, smem_thr_copy_A_T.partition_S(kr_next),
                 tCrAi_kr_view);
            cute::transform(tCrAi_kr, ring_A_kr[slot], cute::identity{});
            ring_g0[slot] = g_total((m + PREFETCH) * 16 + group_id);
            ring_g1[slot] = g_total((m + PREFETCH) * 16 + group_id + 8);
          }
#pragma unroll
          for (int bi = 0; bi < kValueBlocksPerWarp; ++bi) {
            auto& state_fragment = resident_state[bi][m];
#pragma unroll
            for (int a = 0; a < 2; ++a) {
#pragma unroll
              for (int d = 0; d < 2; ++d) {
                auto c0 = make_coord(make_coord(a, 0), 0, d);
                auto c1 = make_coord(make_coord(a, 1), 0, d);
                state_fragment(c0) =
                    fmaf(state_fragment(c0), g0, u_acc[bi](c0));
                state_fragment(c1) =
                    fmaf(state_fragment(c1), g1, u_acc[bi](c1));
              }
            }
            if (t + 1 == t_tiles) {
              Tensor s_block =
                  local_tile(s_acc_T, make_shape(Int<16>{}, Int<16>{}),
                             make_coord(m, warp * kValueBlocksPerWarp + bi));
              auto state_bf16 =
                  narrow_state<ResidentStateFragment>(state_fragment);
              copy(smem_tiled_store_C_T,
                   smem_thr_store_C_T.retile_S(state_bf16),
                   smem_thr_store_C_T.partition_D(s_block));
            }
          }
        }
      }
      load_pipeline.consumer_release(load_read);
      ++load_read;
    }
  }

  // Final state: the MMA warps' last shared-memory writes, then one TMA store.
  cutlass::arch::fence_view_async_shared();
  __syncthreads();
  if (warp_role == WarpRole::LOAD_QKG && lane_predicate) {
    Tensor g_summary =
        tma_store_summary.get_tma_tensor(make_shape(num_rows * H, 2 * D, D));
    int64_t row = out_rows[seq_idx];
    auto off = g_summary.layout()(int(row) * H + head_idx, v_idx * VD, 0);
    Tensor g_tile =
        make_tensor(g_summary.data() + off,
                    make_layout(make_shape(Int<1>{}, Int<VD>{}, Int<D>{}),
                                stride(g_summary.layout())));
    Tensor s_state = make_tensor(
        make_smem_ptr(shared_storage.state_acc.begin()), TMAStateSmemLayout{});
    auto cta = tma_store_summary.get_slice(Int<0>{});
    cute::copy(tma_store_summary, cta.partition_S(s_state),
               cta.partition_D(g_tile));
    tma_store_arrive();
    tma_store_wait<0>();
  }
}

// Shared host-side geometry of one rank's local segments.
struct Geometry {
  int T_total;
  int H;
  int N;
  int total_tiles;
  cudaStream_t stream;
};

inline Geometry geometry(const torch::stable::Tensor& v,
                         const torch::stable::Tensor& cu_seqlens) {
  constexpr int CHUNK = 16;
  STD_TORCH_CHECK(v.dim() == 4 && v.size(0) == 1 && v.size(3) == 128,
                  "KCP FlashKDA tensors must be [1, T, H, 128]");
  STD_TORCH_CHECK(
      cu_seqlens.dim() == 1 && cu_seqlens.numel() >= 2 &&
          cu_seqlens.is_contiguous() &&
          cu_seqlens.scalar_type() == torch::headeronly::ScalarType::Int,
      "cu_seqlens must be contiguous int32 with a segment");
  Geometry g;
  g.T_total = int(v.size(1));
  g.H = int(v.size(2));
  g.N = int(cu_seqlens.numel() - 1);
  // FlashKDA's varlen upper bound; K1 and K2 must agree on it.
  g.total_tiles = (g.T_total + CHUNK - 1) / CHUNK + g.N;
  void* stream_ptr = nullptr;
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_get_current_cuda_stream(v.get_device_index(), &stream_ptr));
  g.stream = static_cast<cudaStream_t>(stream_ptr);
  return g;
}

struct WorkspacePtrs {
  cutlass::bfloat16_t* kd;
  cutlass::bfloat16_t* qd;
  cutlass::bfloat16_t* kr;
  float* gt;
  cutlass::bfloat16_t* inv;
  cutlass::bfloat16_t* mqk;
};

inline WorkspacePtrs workspace_ptrs(const torch::stable::Tensor& workspace,
                                    const Geometry& g) {
  using BF16 = cutlass::bfloat16_t;
  using WS = WorkspaceSizes<16, 128>;
  int64_t n_ht = int64_t(g.H) * g.total_tiles;
  STD_TORCH_CHECK(
      workspace.is_contiguous() &&
          workspace.numel() * workspace.element_size() >= n_ht * WS::kPerTile,
      "workspace is too small");
  char* ws = reinterpret_cast<char*>(workspace.mutable_data_ptr());
  WorkspacePtrs p;
  p.kd = reinterpret_cast<BF16*>(ws);
  p.qd = reinterpret_cast<BF16*>(ws + n_ht * WS::kKDecayed);
  p.kr = reinterpret_cast<BF16*>(ws + n_ht * (WS::kKDecayed + WS::kQDecayed));
  p.gt = reinterpret_cast<float*>(
      ws + n_ht * (WS::kKDecayed + WS::kQDecayed + WS::kKRestored));
  p.inv = reinterpret_cast<BF16*>(ws + n_ht * (WS::kKDecayed + WS::kQDecayed +
                                               WS::kKRestored + WS::kGTotal));
  p.mqk = reinterpret_cast<BF16*>(ws + n_ht * (WS::kKDecayed + WS::kQDecayed +
                                               WS::kKRestored + WS::kGTotal +
                                               WS::kINV));
  return p;
}

inline void check_launch() {
  cudaError_t status = cudaGetLastError();
  STD_TORCH_CHECK(status == cudaSuccess, cudaGetErrorString(status));
}

inline cutlass::bfloat16_t const* bf16(const torch::stable::Tensor& t) {
  return reinterpret_cast<cutlass::bfloat16_t const*>(t.const_data_ptr());
}

inline int num_sms(const torch::stable::Tensor& t) {
  int count = 0;
  cudaError_t status = cudaDeviceGetAttribute(
      &count, cudaDevAttrMultiProcessorCount, t.get_device_index());
  STD_TORCH_CHECK(status == cudaSuccess, cudaGetErrorString(status));
  return count;
}

}  // namespace kcp_flash

using KcpTensor = torch::stable::Tensor;

// K1: FlashKDA preparation of local segments into ``workspace``. q, k and the
// raw gate are [1, T, H, 128] BF16; ``beta_t`` is the raw beta as [H, T].
void kcp_flash_prepare(const KcpTensor& q, const KcpTensor& k,
                       const KcpTensor& raw_g, const KcpTensor& beta_t,
                       const KcpTensor& A_log, const KcpTensor& dt_bias,
                       const KcpTensor& cu_seqlens, const KcpTensor& workspace,
                       double scale, double lower_bound) {
  using namespace kcp_flash;
  constexpr int D = 128;
  const torch::stable::accelerator::DeviceGuard guard(q.get_device_index());
  Geometry g = geometry(q, cu_seqlens);
  if (g.T_total == 0) return;
  for (const KcpTensor* t : {&q, &k, &raw_g, &beta_t})
    STD_TORCH_CHECK(
        t->is_contiguous() &&
            t->scalar_type() == torch::headeronly::ScalarType::BFloat16,
        "q, k, raw_g and beta_t must be contiguous BF16");
  STD_TORCH_CHECK(
      beta_t.dim() == 2 && beta_t.size(0) == g.H && beta_t.size(1) == g.T_total,
      "beta_t must be [H, T]");
  for (const KcpTensor* t : {&A_log, &dt_bias})
    STD_TORCH_CHECK(
        t->is_contiguous() &&
            t->scalar_type() == torch::headeronly::ScalarType::Float,
        "A_log and dt_bias must be contiguous FP32");
  STD_TORCH_CHECK(A_log.numel() == g.H && dt_bias.numel() == int64_t(g.H) * D,
                  "A_log must have H elements and dt_bias H * 128");
  workspace_ptrs(workspace, g);  // Checks the workspace size.
  launch_fwd<D, false, false, false, false, true, int32_t>(
      bf16(q), bf16(k), nullptr, bf16(raw_g), bf16(beta_t), nullptr,
      float(scale), nullptr, nullptr, nullptr, nullptr,
      workspace.mutable_data_ptr(), g.total_tiles, g.T_total, g.H, g.N,
      reinterpret_cast<int32_t const*>(cu_seqlens.const_data_ptr()),
      reinterpret_cast<float const*>(A_log.const_data_ptr()),
      reinterpret_cast<float const*>(dt_bias.const_data_ptr()),
      float(lower_bound * 1.4426950408889634), num_sms(q), g.stream, nullptr,
      -1, /*run_prepare=*/true, /*run_recurrence=*/false);
  check_launch();
}

// K2 summary: writes each segment's value-first [2 * 128, 128] BF16 final
// state ([S^T; M^T]) to ``summaries[out_rows[segment]]`` ([R, H, 256, 128]).
void kcp_flash_summary(const KcpTensor& v, const KcpTensor& beta_t,
                       const KcpTensor& workspace, const KcpTensor& cu_seqlens,
                       const KcpTensor& out_rows, const KcpTensor& summaries) {
  using namespace kcp_flash;
  using BF16 = cutlass::bfloat16_t;
  constexpr int CHUNK = 16;
  constexpr int D = 128;
  const torch::stable::accelerator::DeviceGuard guard(v.get_device_index());
  Geometry g = geometry(v, cu_seqlens);
  for (const KcpTensor* t : {&v, &beta_t})
    STD_TORCH_CHECK(
        t->is_contiguous() &&
            t->scalar_type() == torch::headeronly::ScalarType::BFloat16,
        "v and beta_t must be contiguous BF16");
  STD_TORCH_CHECK(
      summaries.is_contiguous() && summaries.dim() == 4 &&
          summaries.size(1) == g.H && summaries.size(2) == 2 * D &&
          summaries.size(3) == D &&
          summaries.scalar_type() == torch::headeronly::ScalarType::BFloat16,
      "summaries must be contiguous BF16 [R, H, 256, 128]");
  STD_TORCH_CHECK(
      out_rows.is_contiguous() && out_rows.numel() == g.N &&
          out_rows.scalar_type() == torch::headeronly::ScalarType::Long,
      "out_rows must be contiguous int64 with a row per segment");
  if (g.T_total == 0) return;
  WorkspacePtrs ws = workspace_ptrs(workspace, g);
  int num_rows = int(summaries.size(0));
  auto launch = [&](auto vd, auto stages, auto min_blocks) {
    constexpr int VD = decltype(vd)::value;
    constexpr int kStages = decltype(stages)::value;
    constexpr int kMinBlocks = decltype(min_blocks)::value;
    using L = K2Layouts<D, CHUNK, VD>;
    auto gmem_layout =
        make_layout(make_shape(g.H, g.T_total, D), make_stride(D, D * g.H, 1));
    Tensor m_v = make_tensor(
        make_gmem_ptr(reinterpret_cast<BF16 const*>(v.const_data_ptr())),
        gmem_layout);
    Tensor m_beta = make_tensor(
        make_gmem_ptr(reinterpret_cast<BF16 const*>(beta_t.const_data_ptr())),
        make_layout(make_shape(g.H * g.T_total)));
    Tensor m_summary = make_tensor(
        make_gmem_ptr(reinterpret_cast<BF16*>(summaries.mutable_data_ptr())),
        make_layout(make_shape(num_rows * g.H, 2 * D, D), LayoutRight{}));
    auto tma_v = make_tma_copy(SM90_TMA_LOAD{}, m_v, typename L::TMAVOLayout{});
    auto tma_beta =
        make_tma_copy(SM90_TMA_LOAD{}, m_beta, typename L::TMABetaSmemLayout{});
    auto tma_summary = make_tma_copy(SM90_TMA_STORE{}, m_summary,
                                     typename L::TMAStateSmemLayout{});
    constexpr int kThreads = 128 + 32;
    auto kernel = summary_recurrence<decltype(tma_v), decltype(tma_beta),
                                     decltype(tma_summary), CHUNK, D, kStages,
                                     kThreads, kMinBlocks, int32_t, VD>;
    constexpr int smem = sizeof(SummaryStorage<L, kStages>);
    static bool configured = [&] {
      cudaError_t err = cudaFuncSetAttribute(
          kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
      STD_TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err));
      return true;
    }();
    (void)configured;
    kernel<<<dim3(g.H * 2 * (D / VD), g.N), dim3(kThreads), smem, g.stream>>>(
        tma_v, tma_beta, tma_summary, g.T_total, g.H, num_rows,
        reinterpret_cast<int32_t const*>(cu_seqlens.const_data_ptr()),
        reinterpret_cast<int64_t const*>(out_rows.const_data_ptr()),
        g.total_tiles, ws.kd, ws.kr, ws.gt, ws.inv);
  };
  // With the state in FP32, 64-column slices fit the registers of 3 CTAs/SM
  // without spilling; full width spills there and is slower at 2 CTAs/SM.
  launch(Int<64>{}, Int<3>{}, Int<3>{});
  check_launch();
}

// K2 scan: FlashKDA's recurrence from FP32 value-first initial states
// ([N, H, 128, 128]), writing every segment's token outputs.
void kcp_flash_scan(const KcpTensor& v, const KcpTensor& beta_t,
                    const KcpTensor& workspace, const KcpTensor& cu_seqlens,
                    const KcpTensor& initial_state, const KcpTensor& out) {
  using namespace kcp_flash;
  using BF16 = cutlass::bfloat16_t;
  constexpr int D = 128;
  const torch::stable::accelerator::DeviceGuard guard(v.get_device_index());
  Geometry g = geometry(v, cu_seqlens);
  for (const KcpTensor* t : {&v, &beta_t})
    STD_TORCH_CHECK(
        t->is_contiguous() &&
            t->scalar_type() == torch::headeronly::ScalarType::BFloat16,
        "v and beta_t must be contiguous BF16");
  STD_TORCH_CHECK(
      out.is_contiguous() && out.dim() == 4 && out.size(1) == g.T_total &&
          out.size(2) == g.H &&
          out.scalar_type() == torch::headeronly::ScalarType::BFloat16,
      "out must be contiguous BF16 [1, T, H, 128]");
  STD_TORCH_CHECK(
      initial_state.is_contiguous() && initial_state.dim() == 4 &&
          initial_state.size(0) == g.N && initial_state.size(1) == g.H &&
          initial_state.scalar_type() == torch::headeronly::ScalarType::Float,
      "initial_state must be contiguous FP32 [N, H, 128, 128]");
  if (g.T_total == 0) return;
  workspace_ptrs(workspace, g);  // Checks the workspace size.
  launch_fwd<D, true, false, true, false, true, int32_t>(
      nullptr, nullptr, bf16(v), nullptr, bf16(beta_t),
      initial_state.const_data_ptr(), 0.0f, nullptr, nullptr, nullptr,
      reinterpret_cast<BF16*>(out.mutable_data_ptr()),
      workspace.mutable_data_ptr(), g.total_tiles, g.T_total, g.H, g.N,
      reinterpret_cast<int32_t const*>(cu_seqlens.const_data_ptr()), nullptr,
      nullptr, 0.0f, num_sms(v), g.stream, nullptr, -1,
      /*run_prepare=*/false, /*run_recurrence=*/true);
  check_launch();
}
