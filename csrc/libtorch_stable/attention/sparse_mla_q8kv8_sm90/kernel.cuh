/* Copyright 2025 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// SM90 FP8 native sparse MLA prefill kernel.

// Algorithm inspired by DeepSeek FlashMLA
// (https://github.com/deepseek-ai/FlashMLA); the kernel itself is a
// clean-room re-implementation targeting the Q8KV8 sparse prefill path.

// Design: Native fp8 GMMA path
//   QK GEMM: fp8 SS (E4M3 x E4M3 -> F32, k=32, 2x throughput vs bf16)
//   PV GEMM: fp8 RS/SS (E4M3 x E4M3 -> F32, V physically transposed in smem)
//   Producer: loads fp8 KV from gmem via cp.async.cg direct to smem, then
//   transposes V Q: consumer WG0 loads fp8 Q from gmem directly to fp8 smem

#pragma once

#include "config.h"
#include "helpers.h"
#include <cuda_fp8.h>

// using namespace cute must be at global scope BEFORE including dense_fp8
// headers (they use bare Tensor, make_tensor etc. from cute namespace). That
// mid-file include is also why this subtree stays outside `namespace sglang`
// while the rest of csrc/ moved into it - only entry.cuh, the host wrapper, is
// wrapped.
using namespace cute;

// Include the fp8 transpose utility
#include "dense_fp8_transpose_v.h"
// Include the dense_fp8 utils for permute_Cregs_fp8, convert_layout_acc_Aregs,
// convert_type_out
#include "dense_fp8_utils.h"

namespace sm90 {
namespace fwd {

template <typename Kernel, typename TMAParamsT>
__global__ void sparse_mla_q8kv8_prefill_kernel(
    __grid_constant__ const SparseMlaQ8Kv8PrefillParams params,
    __grid_constant__ const TMAParamsT tma_params);

template <int D_QK, bool HAVE_TOPK_LENGTH, bool HAVE_ATTN_SINK>
struct SparseMlaQ8Kv8PrefillKernel {
  static constexpr int D_Q = D_QK;
  static constexpr int D_K = D_QK;
  static constexpr int D_V = 512;

  static constexpr int B_H = 64;
  static constexpr int B_TOPK = 64;
  static constexpr int NUM_THREADS = 128 * 3;
  static constexpr float MAX_INIT_VAL = -1e30f;

  using fp8_t = cutlass::float_e4m3_t;

  enum NamedBarriers : uint32_t {
    wg0_bunch_0_ready = 0,  // WG0 publishes max logits and local P buffer.
    wg1_bunch_0_ready = 1,  // WG1 publishes max logits and local P buffer.
    vt0_left_ready = 2,     // V[0] left half done (producer + WG0 arrivals).
    vt0_right_ready = 3,    // V[0] right half done (producer + WG1 arrivals).
    sL_ready = 4,           // post-loop only
    warpgroup0_sync = 5,    // post-loop only; reused in-loop as vt1_for_wg0
    warpgroup1_sync = 6,    // post-loop only; reused in-loop as vt1_for_wg1
    epilogue_sync = 7,      // never used as call; alias for q_load_done
                            // SM90: max 8 user NamedBarrier IDs (PTX 8-15).
                            // All 8 IDs used: 0-3 in-loop only, 4-7 temporally
                            // reused between in-loop and post-loop.
  };
  // Barrier ID aliases -- temporally disjoint reuse:
  static constexpr uint32_t q_load_done =
      epilogue_sync;  // pre-loop (256 arrivals)
  static constexpr uint32_t vt1_for_wg0 =
      warpgroup0_sync;  // in-loop (256 = prod+WG0)
  static constexpr uint32_t vt1_for_wg1 =
      warpgroup1_sync;  // in-loop (256 = prod+WG1)
  static constexpr uint32_t s_consumed_ready =
      sL_ready;  // in-loop (256 = WG0+WG1)

  // ========================================================================
  // FP8 Smem Layouts -- native fp8 in smem
  // ========================================================================
  // Q: fp8, K-major for QK SS GMMA A-operand
  // SW64 because D_QK=576, 576/64=9 (int), 576/128=4.5 (not int)
  template <int NUM_TILES>
  using SmemLayoutQTiles_FP8 = decltype(coalesce(
      tile_to_shape(GMMA::Layout_K_SW64_Atom<fp8_t>{},
                    Shape<Int<B_H>, Int<64 * NUM_TILES>>{}, Step<_1, _2>{}),
      Shape<_1, _1>{}));

  // K: fp8, K-major for QK SS GMMA B-operand
  template <int NUM_TILES>
  using SmemLayoutKTiles_FP8 = decltype(coalesce(
      tile_to_shape(GMMA::Layout_K_SW64_Atom<fp8_t>{},
                    Shape<Int<B_TOPK>, Int<64 * NUM_TILES>>{}, Step<_1, _2>{}),
      Shape<_1, _1>{}));

  // Vt (transposed V): fp8, K-major for PV GMMA B-operand
  // Shape: (D_V, B_TOPK) = (512, 64)
  template <int NUM_TILES>
  using SmemLayoutVtTiles_FP8 = decltype(coalesce(
      tile_to_shape(GMMA::Layout_K_SW64_Atom<fp8_t>{},
                    Shape<Int<64 * NUM_TILES>, Int<B_TOPK>>{}, Step<_1, _2>{}),
      Shape<_1, _1>{}));

  // O: bf16 output (unchanged from q16)
  template <int NUM_TILES>
  using SmemLayoutOTiles = decltype(coalesce(
      tile_to_shape(GMMA::Layout_K_SW128_Atom<bf16>{},
                    Shape<Int<B_H>, Int<64 * NUM_TILES>>{}, Step<_1, _2>{}),
      Shape<_1, _1>{}));

  using SmemLayoutQ = SmemLayoutQTiles_FP8<D_Q / 64>;
  using SmemLayoutK = SmemLayoutKTiles_FP8<D_Q / 64>;
  using SmemLayoutVt = SmemLayoutVtTiles_FP8<D_V / 64>;  // (512, 64) fp8
  using SmemLayoutHalfVt =
      SmemLayoutVtTiles_FP8<D_V / 64 / 2>;  // (256, 64) fp8
  using SmemLayoutO = SmemLayoutOTiles<D_V / 64>;

  using SmemTransposeV = SmemTransposeFp8_64x64<B_TOPK, D_V>;

  // ========================================================================
  // FP8 GMMA atoms -- native E4M3, k=32
  // ========================================================================
  // QK: SS, both K-major, 64x64x32 -> 2x throughput vs bf16 k=16
  using TiledMMA_QK = decltype(make_tiled_mma(
      GMMA::MMA_64x64x32_F32E4M3E4M3_SS_TN<>{}, Layout<Shape<_1, _1, _1>>{}));

  // PV local: RS, fp8 P in regs x fp8 Vt in smem (K-major)
  using TiledMMA_PV_LocalP = decltype(make_tiled_mma(
      GMMA::MMA_64x256x32_F32E4M3E4M3_RS_TN<>{}, Layout<Shape<_1, _1, _1>>{}));

  // PV remote: SS, fp8 P from sS x fp8 Vt in smem (K-major)
  using TiledMMA_PV_RemoteP = decltype(make_tiled_mma(
      GMMA::MMA_64x256x32_F32E4M3E4M3_SS_TN<>{}, Layout<Shape<_1, _1, _1>>{}));

  // ========================================================================
  // Shared Memory Plan -- native fp8
  // ========================================================================
  struct SharedMemoryPlan {
    union {
      array_aligned<fp8_t, cosize_v<SmemLayoutQ>> q;  // B_H * D_Q fp8
      array_aligned<bf16, cosize_v<SmemLayoutO>> o;   // B_H * D_V/2 bf16
    } q_o;
    array_aligned<fp8_t, cosize_v<SmemLayoutK>>
        k[2];  // 2x K double-buffer, fp8
    // Vt is split into four half-buffers (256x64 each) so each half can be
    // handed to its consumer warpgroup as soon as it is transposed:
    //   vt_loc[i]: i=0 block0-left (WG0 local), i=1 block1-right (WG1 local)
    //   vt_rem[i]: i=0 block1-left (WG0 remote), i=1 block0-right (WG1 remote)
    array_aligned<fp8_t, cosize_v<SmemLayoutHalfVt>> vt_loc[2];
    array_aligned<fp8_t, cosize_v<SmemLayoutHalfVt>> vt_rem[2];
    array_aligned<fp8_t, 128 * 36> s[2];  // 2x S buffer, padded to a 36B row
                                          // stride to avoid bank conflicts.

    // double-buffer the VALIDITY DATA by iteration-pair
    // parity, matching bar_is_kv_valid_ready[2].  The consumer arrives
    // bar_k_free BEFORE mask_rP reads these bits (perf: releases the producer's
    // next K-load early), so a single-buffered array let the producer overwrite
    // pair N+1's bits while a laggard consumer was still masking pair N ->
    // wrong -INF pattern (nondeterministic corruption growing with CTA
    // count).  Dims: [k-buf/warpgroup][pair
    // parity][slot]; producer ahead-ness is bounded to one pair by the
    // bar_k_free protocol, so depth 2 suffices (same argument as the barrier).
    bool is_kv_valid[2][2][B_TOPK];
    // double-buffer the WG0<->WG1 running-max exchange
    // by iteration-pair parity (same recipe as is_kv_valid): any +-1-iteration
    // overrun of the producer side lands in the OTHER slot instead of
    // overwriting a value the peer is still reading.  Empirical signature of
    // the race: single-row, ~half-heads rescale blow-up (peer_scale computed
    // from the wrong iteration's max), vanishing on same-tensor retry.
    float2 sM[2][32];
    float2 sL[64];
    float final_max_logits[64], final_lse[64];
    transac_bar_t bar_q, bar_k0_ready[2], bar_k1_ready[2],
        bar_is_kv_valid_ready[2];  // double-buffer is_kv_valid_ready
    transac_bar_t bar_k0_free, bar_k1_free;
    // Consumers arrive after PV drains; the producer waits before reusing the
    // Vt buffer. These barriers are separate from K-free so K buffers can be
    // released earlier.
    transac_bar_t bar_vt_free[2];  // bar_vt_free[0] protects Vt[0],
                                   // bar_vt_free[1] protects Vt[1]
  };

  struct TmaParams_t {
    CUtensorMap tensor_map_O;
  };

  // ========================================================================
  // devfunc -- main kernel logic, native fp8 GMMA
  // ========================================================================
  template <typename TMAParamType>
  static __device__ __forceinline__ void devfunc(
      const SparseMlaQ8Kv8PrefillParams& params,
      const TMAParamType& tma_params) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 900)) || \
    (defined(__CLION_IDE__) || defined(__VSCODE_IDE__))
    const int q_h_idx = blockIdx.x % (params.h_q / B_H);
    const int s_q_idx = blockIdx.x / (params.h_q / B_H);
    const int warpgroup_idx = cutlass::canonical_warp_group_idx();
    const int warp_idx = cutlass::canonical_warp_idx_sync();
    const int idx_in_warpgroup = threadIdx.x % 128;

    extern __shared__ char wksp_buf[];
    SharedMemoryPlan& plan = *reinterpret_cast<SharedMemoryPlan*>(wksp_buf);

    const float q_scale = __ldg(params.q_scale_ptr);
    const float kv_scale = __ldg(params.kv_scale_ptr);
    const float qk_combined_scale_div_log2 =
        q_scale * kv_scale * params.sm_scale_div_log2;

    if (warp_idx == 0 && elect_one_sync()) {
      cute::prefetch_tma_descriptor(&tma_params.tensor_map_O);

      plan.bar_q.init(1);
      plan.bar_k0_free.init(128);
      plan.bar_k1_free.init(128);
      CUTE_UNROLL
      for (int i = 0; i < 2; ++i) {
        plan.bar_k0_ready[i].init(128);
        plan.bar_k1_ready[i].init(128);
      }
      CUTE_UNROLL
      for (int i = 0; i < 2; ++i) {
        plan.bar_is_kv_valid_ready[i].init(16);  // double-buffer
      }
      CUTE_UNROLL
      for (int i = 0; i < 2; ++i) {
        // Transaction barriers for Vt buffer safety: 128 arrivals from each
        // consumer WG.
        plan.bar_vt_free[i].init(256);
      }
      fence_barrier_init();
    }

    __syncthreads();
    const int topk_length =
        HAVE_TOPK_LENGTH ? __ldg(params.topk_length + s_q_idx) : params.topk;
    const int num_topk_blocks =
        HAVE_TOPK_LENGTH
            ? ku::ceil_div(topk_length, (int)B_TOPK)
            : (int)((unsigned int)params.topk / (unsigned int)B_TOPK);

    // ================================================================
    // Consumer WG0/WG1
    // ================================================================
    if (warpgroup_idx == 0 || warpgroup_idx == 1) {
      cutlass::arch::warpgroup_reg_alloc<216>();

      // --------------------------------------------------------
      // Load Q from global fp8 -> fp8 smem (thread-based writes)
      // Only WG0 loads Q, then sync with WG1 via NamedBarrier
      // --------------------------------------------------------
      if (warpgroup_idx == 0) {
        const fp8_t* gQ = reinterpret_cast<const fp8_t*>(params.q) +
                          s_q_idx * (int64_t)params.stride_q_s_q +
                          q_h_idx * B_H * (int64_t)params.stride_q_h_q;

        // Vectorized Q loading via cp.async.cg (16 bytes per op).
        // Group size 4 (not 8): with 8-row groups two warps' cp.async stores
        // could overlap the same Q smem rows (WAW hazard); 4 keeps each row
        // owned by exactly one group.
        constexpr int Q_GROUP_SIZE = 4;
        constexpr int Q_NUM_GROUPS = 128 / Q_GROUP_SIZE;
        constexpr int Q_ROWS_PER_GROUP = B_H / Q_NUM_GROUPS;
        int q_ig = idx_in_warpgroup % Q_GROUP_SIZE;
        int q_gg = idx_in_warpgroup / Q_GROUP_SIZE;
        fp8_t* sQ_base =
            &(make_tensor(make_smem_ptr(plan.q_o.q.data()),
                          SmemLayoutQTiles_FP8<1>{})(q_gg, q_ig * 16));
        constexpr int NUM_Q_TILES = D_Q / 64;
        int64_t q_cache_policy = createpolicy_evict_first();
        CUTE_UNROLL
        for (int lr = 0; lr < Q_ROWS_PER_GROUP; ++lr) {
          CUTE_UNROLL
          for (int ti = 0; ti < NUM_Q_TILES; ++ti) {
            // Guard against OOB: last tile may be partial when D_Q%64!=0
            bool q_pred = (ti * 64 + q_ig * 16 + 16) <= D_Q;
            cp_async_cacheglobal_l2_prefetch_256B(
                gQ + (q_gg + lr * Q_NUM_GROUPS) * (int64_t)params.stride_q_h_q +
                    ti * 64 + q_ig * 16,
                sQ_base + ti * (B_H * 64) + lr * Q_NUM_GROUPS * 64, q_pred,
                q_cache_policy);
          }
        }
        asm volatile("cp.async.commit_group;\n" ::);
        asm volatile("cp.async.wait_group 0;\n" ::);
      }
      fence_view_async_shared();
      NamedBarrier::arrive_and_wait(256, q_load_done);

      // --------------------------------------------------------
      // Register fragments
      // --------------------------------------------------------
      float rM[2] = {MAX_INIT_VAL, MAX_INIT_VAL};
      float rL[2] = {0.0f, 0.0f};
      // WG1 consumes WG0's peer P (s[0]) which is in WG0's LOCAL max frame
      // (max(prev,block0)), not the combined frame. peer_scale = exp2(WG0_local
      // - combined) brings it into the combined frame (<=1, ==1 at low
      // magnitude => no-op). Clamped away from 0 so the 1/peer_scale pre-scale
      // cannot overflow. WG0's peer (s[1]) is already combined-frame, so
      // peer_scale stays 1.0 for WG0.
      float peer_scale[2] = {1.0f, 1.0f};
      Tensor rO = partition_fragment_C(TiledMMA_PV_LocalP{},
                                       Shape<Int<B_H>, Int<D_V / 2>>{});
      Tensor rP =
          partition_fragment_C(TiledMMA_QK{}, Shape<Int<B_H>, Int<B_TOPK>>{});
      cute::fill(rO, 0.0f);

      // fp8 P register for local PV RS GMMA
      // Use the same layout that convert_layout_acc_Aregs will produce
      using rP_fp8_layout_t =
          decltype(flash::convert_layout_acc_Aregs<TiledMMA_PV_LocalP>(
              partition_fragment_C(TiledMMA_QK{},
                                   Shape<Int<B_H>, Int<B_TOPK>>{})
                  .layout()));
      Tensor rP_fp8_local = make_tensor<fp8_t>(rP_fp8_layout_t{});

      bool cur_bar_wait_phase = 0;
      bool kv_valid_phase[2] = {
          false, false};  // per-instance phase for bar_is_kv_valid_ready[2]
      struct Warpgroup0 {};
      struct Warpgroup1 {};

      // fp8 QK GEMM: 64-wide tiles, k=32, so 64/32=2 k-steps per tile
      auto qkt_gemm_one_tile = [&](auto wg_tag, int tile_idx,
                                   bool clear_accum) {
        constexpr bool IS_WG1 = std::is_same_v<decltype(wg_tag), Warpgroup1>;
        TiledMMA_QK tiled_mma_QK;
        Tensor sQ_tile =
            make_tensor(make_smem_ptr(plan.q_o.q.data() + tile_idx * B_H * 64),
                        SmemLayoutQTiles_FP8<1>{});
        Tensor sK_tile = make_tensor(
            make_smem_ptr(plan.k[(int)IS_WG1].data() + tile_idx * B_TOPK * 64),
            SmemLayoutKTiles_FP8<1>{});
        gemm_ss(clear_accum, tiled_mma_QK, sQ_tile, sK_tile, rP,
                idx_in_warpgroup);
      };

      auto mask_rP = [&](auto wg_tag, int block_idx) {
        constexpr bool IS_WG1 = std::is_same_v<decltype(wg_tag), Warpgroup1>;
        // bar_is_kv_valid_ready was single-buffered but arrived EARLY by the
        // producer and waited LATE here (after the prefetched QK gemm). The
        // producer runs 1 loop-iter ahead, so it can arrive the single-bit
        // barrier TWICE before this wait consumes the first flip -> the parity
        // aliases back and the laggard consumer waits for a flip that never
        // comes (producer wedged at bar_vt_free) -> deadlock under load. Fix:
        // double-buffer by iteration parity (producer is at most 1 loop-iter
        // ahead => always the OTHER instance) with an independent per-instance
        // phase.
        int _kv_buf = (block_idx >> 1) & 1;
        plan.bar_is_kv_valid_ready[_kv_buf].wait(kv_valid_phase[_kv_buf]);
        kv_valid_phase[_kv_buf] ^= 1;
        CUTE_UNROLL
        for (int row_idx = 0; row_idx < 2; ++row_idx) {
          CUTE_UNROLL
          for (int i = row_idx * 2; i < size(rP); i += 4) {
            int col = 8 * (i / 4) + (idx_in_warpgroup % 4) * 2;
            if (!plan.is_kv_valid[IS_WG1][_kv_buf][col])
              rP(i) = -INFINITY;  // parity-buffered
            if (!plan.is_kv_valid[IS_WG1][_kv_buf][col + 1])
              rP(i + 1) = -INFINITY;
          }
        }
      };

      // online_softmax: compute softmax on rP (f32), then convert to fp8
      // _par = iteration-pair parity ((block_idx>>1)&1) selecting
      // the sM slot for this iteration's WG0<->WG1 max exchange.
      auto online_softmax_and_rescale_o = [&](auto wg_tag, int _par) {
        // mask_rP already waits for the validity mask.
        constexpr bool IS_WG1 = std::is_same_v<decltype(wg_tag), Warpgroup1>;
        const float scale = qk_combined_scale_div_log2;
        float r_sM[2];
        if constexpr (IS_WG1) {
          *(float2*)r_sM = plan.sM[_par][idx_in_warpgroup / 4];
        }
        float new_maxs[2];
        CUTE_UNROLL
        for (int row_idx = 0; row_idx < 2; ++row_idx) {
          float cur_max = -INFINITY;
          CUTE_UNROLL
          for (int i = row_idx * 2; i < size(rP); i += 4) {
            cur_max = max(cur_max, max(rP(i), rP(i + 1)));
          }
          cur_max = max(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 1));
          cur_max = max(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 2));
          cur_max *= scale;
          new_maxs[row_idx] =
              max(IS_WG1 ? r_sM[row_idx] : rM[row_idx], cur_max);
          // peer P (WG0's s[0]) is in WG0's local frame r_sM; bring it to the
          // combined frame new_maxs. <=1, ==1 at low magnitude. Floor at 2^-30
          // so the 1/peer_scale pre-scale stays finite (peer weight there is
          // negligible anyway).
          if constexpr (IS_WG1) {
            peer_scale[row_idx] =
                fmaxf(exp2f(r_sM[row_idx] - new_maxs[row_idx]), exp2f(-30.0f));
          }
          float scale_for_o = exp2f(rM[row_idx] - new_maxs[row_idx]);
          CUTE_UNROLL
          for (int i = row_idx * 2; i < size(rO); i += 4) {
            rO(i) *= scale_for_o;
            rO(i + 1) *= scale_for_o;
          }
          float cur_sum = 0;
          CUTE_UNROLL
          for (int i = row_idx * 2; i < size(rP); i += 4) {
            float p0 = exp2f(rP(i) * scale - new_maxs[row_idx]);
            float p1 = exp2f(rP(i + 1) * scale - new_maxs[row_idx]);
            rP(i) = p0;
            rP(i + 1) = p1;
            cur_sum += p0 + p1;
          }
          rL[row_idx] = rL[row_idx] * scale_for_o + cur_sum;
        }
        __syncwarp();
        if (idx_in_warpgroup % 4 == 0) {
          plan.sM[_par][idx_in_warpgroup / 4] = *(float2*)new_maxs;
        }
        rM[0] = new_maxs[0];
        rM[1] = new_maxs[1];

        // Convert rP f32 (GMMA C layout) -> fp8 (RS A-operand layout)
        // permute_Cregs_fp8 reorders C regs for fp8 A-operand layout
        flash::permute_Cregs_fp8(rP);
        // Reinterpret layout: C -> A-operand
        Tensor rP_acc = make_tensor(
            rP.data(),
            flash::convert_layout_acc_Aregs<TiledMMA_PV_LocalP>(rP.layout()));
        // f32 -> fp8
        flash::convert_type_out(rP_acc, rP_fp8_local);
      };

      auto reduce_L = [&]() {
        rL[0] += __shfl_xor_sync(0xffffffff, rL[0], 1);
        rL[0] += __shfl_xor_sync(0xffffffff, rL[0], 2);
        rL[1] += __shfl_xor_sync(0xffffffff, rL[1], 1);
        rL[1] += __shfl_xor_sync(0xffffffff, rL[1], 2);
        if (idx_in_warpgroup % 4 == 0)
          plan.sL[threadIdx.x / 4] = *(float2*)(rL);
        NamedBarrier::arrive_and_wait(256, NamedBarriers::sL_ready);
        float2 peer_L = plan.sL[(threadIdx.x / 4) ^ 32];
        rL[0] += peer_L.x;
        rL[1] += peer_L.y;
      };

      auto store_O = [&]() {
        float scale_factors[2];
        CUTE_UNROLL
        for (int i = 0; i < 2; ++i) {
          if constexpr (HAVE_ATTN_SINK) {
            int attn_sink_idx =
                q_h_idx * B_H + get_AorC_row_idx(i, idx_in_warpgroup);
            float attn_sink =
                __ldg(params.attn_sink + attn_sink_idx) * CUDART_L2E_F;
            scale_factors[i] = kv_scale / (rL[i] + exp2f(attn_sink - rM[i]));
          } else {
            scale_factors[i] = kv_scale / rL[i];
          }
          if (rL[i] == 0.0f) scale_factors[i] = 0.0f;
        }

        Tensor sO_tile = make_tensor(
            make_smem_ptr(plan.q_o.o.data() + warpgroup_idx * B_H * (D_V / 2)),
            SmemLayoutOTiles<4>{});
        bf16* stsm_addrs[4];
        int stsm_row = (idx_in_warpgroup / 32) * 16 + (idx_in_warpgroup % 16);
        CUTE_UNROLL
        for (int i = 0; i < 64 / 16; ++i) {
          stsm_addrs[i] =
              &sO_tile(stsm_row, (idx_in_warpgroup % 32 / 16 * 8) + 16 * i);
        }
        bool s2g_pred = idx_in_warpgroup == 0;

        warpgroup_wait<0>();
        warpgroup_fence_operand(rO);
        CUTE_UNROLL
        for (int tile_idx = 0; tile_idx < (D_V / 2) / 64; tile_idx += 1) {
          constexpr int NUM_ELEMS_EACH_TILE = B_H * 64 / 128;
          bf16 cur_rOb[NUM_ELEMS_EACH_TILE];
          CUTE_UNROLL
          for (int i = 0; i < NUM_ELEMS_EACH_TILE; ++i) {
            float out_value = rO(tile_idx * NUM_ELEMS_EACH_TILE + i) *
                              scale_factors[i % 4 >= 2];
            cur_rOb[i] = (bf16)out_value;
          }
          CUTE_UNROLL
          for (int i = 0; i < 64 / 16; ++i) {
            SM90_U32x4_STSM_N::copy(
                *reinterpret_cast<uint32_t*>(cur_rOb + i * 8 + 0),
                *reinterpret_cast<uint32_t*>(cur_rOb + i * 8 + 2),
                *reinterpret_cast<uint32_t*>(cur_rOb + i * 8 + 4),
                *reinterpret_cast<uint32_t*>(cur_rOb + i * 8 + 6),
                *reinterpret_cast<uint128_t*>(stsm_addrs[i] +
                                              tile_idx * (B_H * 64)));
          }
          // Make the STSM writes visible to the subsequent TMA store proxy.
          cute::tma_store_fence();
          NamedBarrier::arrive_and_wait(
              128, warpgroup_idx ? NamedBarriers::warpgroup1_sync
                                 : NamedBarriers::warpgroup0_sync);
          if (s2g_pred) {
            int g_tile_idx = warpgroup_idx * 4 + tile_idx;
            SM90_TMA_STORE_3D::copy(&tma_params.tensor_map_O,
                                    plan.q_o.o.data() + g_tile_idx * (B_H * 64),
                                    g_tile_idx * 64, q_h_idx * B_H, s_q_idx);
          }
        }
        cute::tma_store_arrive();
      };

      // Save/load P regs to/from smem using a flat thread-indexed layout.
      // Each thread writes/reads its 32 fp8 values at a unique offset.
      // This preserves the RS A-reg ordering exactly, so the reader can
      // load back and use RS GMMA directly (no SS GMMA layout issues).
      constexpr int kP_per_thread = 32;  // ((4,2,2),1,2) = 32 fp8 per thread
      // Pad stride from 32 to 36 bytes to avoid shared-memory bank conflicts.
      // Stride 32B = only 4 banks (8-way conflict). Stride 36B = 9 banks
      // (gcd(9,32)=1 -> zero conflicts: every warp thread hits a unique bank).
      constexpr int kP_stride = 36;

      auto save_rP_fp8_to_sS = [&](fp8_t* sS_data) {
        uint32_t* dst =
            reinterpret_cast<uint32_t*>(sS_data + idx_in_warpgroup * kP_stride);
        uint32_t* src = reinterpret_cast<uint32_t*>(&rP_fp8_local(0));
        CUTE_UNROLL
        for (int i = 0; i < kP_per_thread / 4; i++) {
          dst[i] = src[i];
        }
      };

      auto load_sS_to_rP = [&](fp8_t* sS_data) {
        uint32_t* src =
            reinterpret_cast<uint32_t*>(sS_data + idx_in_warpgroup * kP_stride);
        uint32_t* dst = reinterpret_cast<uint32_t*>(&rP_fp8_local(0));
        CUTE_UNROLL
        for (int i = 0; i < kP_per_thread / 4; i++) {
          dst[i] = src[i];
        }
      };

      // No output un-permutation is needed here: the PV output columns are
      // produced by the Vt B-operand, not by the permute_Cregs_fp8-rearranged P
      // A-operand (the permute only reorders the P contraction/K dimension,
      // never the output N).

      // ============================================================
      // WG0 Pipeline -- native fp8
      // ============================================================
      if (warpgroup_idx == 0) {
        auto pipelined_wait_and_qkt_gemm_l =
            [&]() __attribute__((always_inline)) {
              plan.bar_k0_ready[0].wait(cur_bar_wait_phase);
              qkt_gemm_one_tile(Warpgroup0{}, 0, true);
              qkt_gemm_one_tile(Warpgroup0{}, 1, false);
              qkt_gemm_one_tile(Warpgroup0{}, 2, false);
              qkt_gemm_one_tile(Warpgroup0{}, 3, false);
              warpgroup_commit_batch();
            };

        auto pipelined_wait_and_qkt_gemm_r =
            [&]() __attribute__((always_inline)) {
              plan.bar_k0_ready[1].wait(cur_bar_wait_phase);
              qkt_gemm_one_tile(Warpgroup0{}, 4, false);
              qkt_gemm_one_tile(Warpgroup0{}, 5, false);
              qkt_gemm_one_tile(Warpgroup0{}, 6, false);
              qkt_gemm_one_tile(Warpgroup0{}, 7, false);
              if constexpr (D_QK == 576) {
                qkt_gemm_one_tile(Warpgroup0{}, 8, false);
              }
              warpgroup_commit_batch();
            };

        auto rescale_rO = [&](float scales[2]) {
          CUTE_UNROLL
          for (int row = 0; row < 2; ++row) {
            CUTE_UNROLL
            for (int i = row * 2; i < size(rO); i += 4) {
              rO(i) *= scales[row];
              rO(i + 1) *= scales[row];
            }
            rL[row] *= scales[row];
          }
        };

        CUTE_NO_UNROLL
        for (int block_idx = 0; block_idx < num_topk_blocks; block_idx += 2) {
          // Vt[0] left half: (256, 64) fp8 -- only half we transpose & use
          Tensor sVt0l = make_tensor(make_smem_ptr(plan.vt_loc[0].data()),
                                     SmemLayoutHalfVt{});

          if (block_idx == 0) {
            pipelined_wait_and_qkt_gemm_l();
            pipelined_wait_and_qkt_gemm_r();
            warpgroup_wait<0>();
            warpgroup_fence_operand(rP);
            plan.bar_k0_free.arrive();
          }

          mask_rP(Warpgroup0{}, block_idx);  // pass block_idx for kv_buf parity
          online_softmax_and_rescale_o(Warpgroup0{}, (block_idx >> 1) & 1);

          save_rP_fp8_to_sS(plan.s[0].data());
          // was arrive-only: a +1-iteration overrun by WG0 could
          // alias the named-barrier count against WG1's pending wait.  Full
          // rendezvous kills the aliasing; WG1's path here (mask_rP -> its
          // bar_is_kv_valid wait) does not depend on anything WG0 does after
          // this point, so no deadlock surface is added.
          NamedBarrier::arrive_and_wait(256, NamedBarriers::wg0_bunch_0_ready);

          // Wait for Vt[0] left half only (producer + WG0 arrivals).
          // V[0]-RIGHT may still be transposing; WG0 doesn't need it.
          NamedBarrier::arrive_and_wait(256, vt0_left_ready);

          // Local PV: rP_fp8 x Vt0 left half -> RS fp8 GMMA
          gemm_rs(false, TiledMMA_PV_LocalP{}, rP_fp8_local, sVt0l, rO,
                  idx_in_warpgroup);
          warpgroup_commit_batch();

          // Overlap PV-local GMMA drain with barrier waits, sM read, and peer P
          // load.
          NamedBarrier::arrive_and_wait(256, NamedBarriers::wg1_bunch_0_ready);
          float new_rM[2], scale_factors_arr[2];
          *(float2*)new_rM =
              plan.sM[(block_idx >> 1) & 1][idx_in_warpgroup / 4];
          CUTE_UNROLL
          for (int i = 0; i < 2; ++i) {
            scale_factors_arr[i] = exp2f(rM[i] - new_rM[i]);
            rM[i] = new_rM[i];
          }

          warpgroup_wait<0>();
          warpgroup_fence_operand(rO);
          warpgroup_fence_operand(rP_fp8_local);
          plan.bar_vt_free[0].arrive();

          load_sS_to_rP(plan.s[1].data());
          NamedBarrier::arrive_and_wait(256, s_consumed_ready);

          // Wait for Vt[1] transpose (prod+WG0 barrier)
          NamedBarrier::arrive_and_wait(256, vt1_for_wg0);

          // Rescale rO: must be after wait<0> since rO is PV-local accumulator
          rescale_rO(scale_factors_arr);

          Tensor sVt1l = make_tensor(make_smem_ptr(plan.vt_rem[0].data()),
                                     SmemLayoutHalfVt{});
          gemm_rs(false, TiledMMA_PV_LocalP{}, rP_fp8_local, sVt1l, rO,
                  idx_in_warpgroup);
          warpgroup_commit_batch();

          cur_bar_wait_phase ^= 1;

          if (block_idx + 2 < num_topk_blocks) {
            pipelined_wait_and_qkt_gemm_l();
            warpgroup_wait<1>();
            warpgroup_fence_operand(rO);
            warpgroup_fence_operand(rP_fp8_local);
            plan.bar_vt_free[1].arrive();
            pipelined_wait_and_qkt_gemm_r();
            warpgroup_wait<0>();
            warpgroup_fence_operand(rP);
            plan.bar_k0_free.arrive();
          } else {
            warpgroup_wait<0>();
            warpgroup_fence_operand(rO);
            plan.bar_vt_free[1].arrive();
          }
        }

        reduce_L();
        store_O();

      } else {
        // ============================================================
        // WG1 Pipeline -- native fp8
        // ============================================================
        // Split QK into R/L halves for loop-end overlap (mirrors WG0 pattern)
        auto pipelined_wait_and_qkt_gemm_r_wg1 =
            [&]() __attribute__((always_inline)) {
              // Right half first: K[1]-right arrives earlier from producer
              plan.bar_k1_ready[1].wait(cur_bar_wait_phase);
              qkt_gemm_one_tile(Warpgroup1{}, 4, true);
              qkt_gemm_one_tile(Warpgroup1{}, 5, false);
              qkt_gemm_one_tile(Warpgroup1{}, 6, false);
              qkt_gemm_one_tile(Warpgroup1{}, 7, false);
              if constexpr (D_QK == 576) {
                qkt_gemm_one_tile(Warpgroup1{}, 8, false);
              }
              warpgroup_commit_batch();
            };

        auto pipelined_wait_and_qkt_gemm_l_wg1 =
            [&]() __attribute__((always_inline)) {
              plan.bar_k1_ready[0].wait(cur_bar_wait_phase);
              qkt_gemm_one_tile(Warpgroup1{}, 0, false);
              qkt_gemm_one_tile(Warpgroup1{}, 1, false);
              qkt_gemm_one_tile(Warpgroup1{}, 2, false);
              qkt_gemm_one_tile(Warpgroup1{}, 3, false);
              warpgroup_commit_batch();
            };

        CUTE_NO_UNROLL
        for (int block_idx = 0; block_idx < num_topk_blocks; block_idx += 2) {
          // Vt[1] right half: (256, 64) fp8 -- only half we transpose & use
          Tensor sVt1r = make_tensor(make_smem_ptr(plan.vt_loc[1].data()),
                                     SmemLayoutHalfVt{});

          if (block_idx == 0) {
            pipelined_wait_and_qkt_gemm_r_wg1();
            pipelined_wait_and_qkt_gemm_l_wg1();
            warpgroup_wait<0>();
            warpgroup_fence_operand(rP);
            plan.bar_k1_free.arrive();
          }

          mask_rP(Warpgroup1{}, block_idx);  // pass block_idx for kv_buf parity

          NamedBarrier::arrive_and_wait(256, NamedBarriers::wg0_bunch_0_ready);
          online_softmax_and_rescale_o(Warpgroup1{}, (block_idx >> 1) & 1);

          save_rP_fp8_to_sS(plan.s[1].data());
          // was arrive-only — the mirror of the wg0_bunch case: if WG1 runs a
          // full iteration ahead, its second arrive aliases the named-barrier
          // count against WG0's pending wait at this rendezvous, so WG0
          // releases early and reads the combined max (sM) BEFORE WG1 wrote it
          // -> wrong rescale factors on one CTA's 64-head half.  No deadlock
          // surface: WG0 reaches its wait via wg0_bunch (which WG1 arrives
          // earlier) plus producer vt0_left; neither depends on WG1's progress
          // past here.
          NamedBarrier::arrive_and_wait(256, NamedBarriers::wg1_bunch_0_ready);

          // Wait for Vt[1] transpose (prod+WG1 barrier)
          NamedBarrier::arrive_and_wait(256, vt1_for_wg1);

          // Local PV: rP_fp8 x Vt1 right half -> RS
          gemm_rs(false, TiledMMA_PV_LocalP{}, rP_fp8_local, sVt1r, rO,
                  idx_in_warpgroup);
          warpgroup_commit_batch();

          warpgroup_wait<0>();
          warpgroup_fence_operand(rO);
          warpgroup_fence_operand(rP_fp8_local);
          plan.bar_vt_free[1].arrive();
          // pre-scale rO (= local block1, combined frame) by 1/peer_scale so
          // the peer PV (WG0's block0 P in WG0's local frame) lands correct
          // after the post-scale: rO/peer_scale; rO += peer_PV; rO *=
          // peer_scale  =>  local + peer_scale*peer_PV.
          CUTE_UNROLL
          for (int row = 0; row < 2; ++row) {
            float inv = 1.0f / peer_scale[row];
            CUTE_UNROLL
            for (int i = row * 2; i < size(rO); i += 4) {
              rO(i) *= inv;
              rO(i + 1) *= inv;
            }
          }
          load_sS_to_rP(plan.s[0].data());
          NamedBarrier::arrive_and_wait(256, s_consumed_ready);

          // Wait for Vt[0] right half only (producer + WG1 arrivals).
          // V[0]-LEFT was signaled earlier; WG1 doesn't need it.
          NamedBarrier::arrive_and_wait(256, vt0_right_ready);

          Tensor sVt0r = make_tensor(make_smem_ptr(plan.vt_rem[1].data()),
                                     SmemLayoutHalfVt{});
          gemm_rs(false, TiledMMA_PV_LocalP{}, rP_fp8_local, sVt0r, rO,
                  idx_in_warpgroup);
          warpgroup_commit_batch();

          if (block_idx + 2 < num_topk_blocks) {
            cur_bar_wait_phase ^= 1;
            // Overlap: start next-iteration QK-right while PV drains
            pipelined_wait_and_qkt_gemm_r_wg1();
            warpgroup_wait<1>();  // drains the peer PV (committed before the
                                  // QK-right batch)
            warpgroup_fence_operand(rO);
            // post-scale: rO = local + peer_scale * peer_PV (combined frame).
            CUTE_UNROLL
            for (int row = 0; row < 2; ++row) {
              CUTE_UNROLL
              for (int i = row * 2; i < size(rO); i += 4) {
                rO(i) *= peer_scale[row];
                rO(i + 1) *= peer_scale[row];
              }
            }
            warpgroup_fence_operand(rP_fp8_local);
            plan.bar_vt_free[0].arrive();
            pipelined_wait_and_qkt_gemm_l_wg1();
            warpgroup_wait<0>();
            warpgroup_fence_operand(rP);
            plan.bar_k1_free.arrive();
          } else {
            warpgroup_wait<0>();  // drains the peer PV
            warpgroup_fence_operand(rO);
            // post-scale (final iteration).
            CUTE_UNROLL
            for (int row = 0; row < 2; ++row) {
              CUTE_UNROLL
              for (int i = row * 2; i < size(rO); i += 4) {
                rO(i) *= peer_scale[row];
                rO(i + 1) *= peer_scale[row];
              }
            }
            plan.bar_vt_free[0].arrive();
          }
        }

        reduce_L();
        store_O();

        if (idx_in_warpgroup % 4 == 0) {
          for (int row = 0; row < 2; ++row) {
            int real_row = get_AorC_row_idx(row, idx_in_warpgroup);
            bool is_no_valid_tokens = rL[row] == 0.0f;
            plan.final_max_logits[real_row] =
                is_no_valid_tokens ? -INFINITY : rM[row] * CUDART_LN2_F;
            plan.final_lse[real_row] =
                is_no_valid_tokens ? +INFINITY
                                   : logf(rL[row]) + rM[row] * CUDART_LN2_F;
          }
          // Regular stores are not async-proxy operations; the barrier provides
          // ordering.
          asm volatile("" ::: "memory");
        }

        NamedBarrier::arrive_and_wait(128, NamedBarriers::warpgroup1_sync);
        if (idx_in_warpgroup == 0) {
          int g_offset = s_q_idx * params.h_q + q_h_idx * B_H;
          SM90_BULK_COPY_S2G::copy(plan.final_max_logits,
                                   params.max_logits + g_offset,
                                   B_H * sizeof(float));
          SM90_BULK_COPY_S2G::copy(plan.final_lse, params.lse + g_offset,
                                   B_H * sizeof(float));
          cute::tma_store_arrive();
        }
      }

    } else {
      // ================================================================
      // Producer WG2: load fp8 KV via cp.async, then transpose V in smem
      // ================================================================
      cutlass::arch::warpgroup_reg_dealloc<72>();

      constexpr int GROUP_SIZE = 8, NUM_GROUPS = 128 / GROUP_SIZE;
      constexpr int NUM_ROWS_PER_GROUP = B_TOPK / NUM_GROUPS;
      int idx_in_group = idx_in_warpgroup % GROUP_SIZE;
      int group_idx = idx_in_warpgroup / GROUP_SIZE;
      const int* gIndices =
          params.indices + s_q_idx * params.stride_indices_s_q;

      int tile_shift = idx_in_group / 4;
      int col_in_tile = (idx_in_group % 4) * 16;
      fp8_t* my_sK_base =
          &(make_tensor(make_smem_ptr(plan.k[0].data()),
                        SmemLayoutKTiles_FP8<1>{})(group_idx, col_in_tile)) +
          tile_shift * (B_TOPK * 64);
      const fp8_t* my_gKV_base =
          reinterpret_cast<const fp8_t*>(params.kv) + idx_in_group * 16;

      int64_t token_indices[2][NUM_ROWS_PER_GROUP];
      bool is_token_valid[2][NUM_ROWS_PER_GROUP];

      // Hoisted invariant: base of the `topk` trailing zero pad
      // rows in kv.  -1 sentinels map to (pad_base + slot) = distinct zero
      // rows.
      const int pad_base = params.s_kv - params.topk;
      auto load_token_indices = [&](int block_idx) {
        CUTE_UNROLL
        for (int buf_idx = 0; buf_idx < 2; ++buf_idx) {
          CUTE_UNROLL
          for (int local_row = 0; local_row < NUM_ROWS_PER_GROUP; ++local_row) {
            int offs = (block_idx + buf_idx) * B_TOPK + local_row * NUM_GROUPS +
                       group_idx;
            int t = __ldg(gIndices + offs);
            bool is_cur_token_valid;
            if constexpr (!HAVE_TOPK_LENGTH) {
              // Map -1 sentinels to DISTINCT zero pad rows (slot `offs` ->
              // pad_base+offs, distinct within each query) -> avoids the
              // duplicate-index kernel slowdown while keeping uniform full-topk
              // loads (data-independent, no DP hang).  Replaces the per-layer
              // torch.where in the integration (eliminates an elementwise
              // launch). After clamping, t in [0, s_kv) by construction (offs <
              // topk), so the LOAD is always safe (real or zero-pad row).

              // but the pad slots must be MASKED in the
              // softmax (is_valid=false -> -INF in mask_rP), NOT scored: a zero
              // KV row contributes exp(0 - max) to the denominator, which for
              // few-valid rows (ctx << topk, e.g. the first tokens of a prompt)
              // crushes the output by up to ~2048x.  chunk<=16384 never exposed
              // this because rows with ctx<=topk take the DENSE prefill path
              // (per-rank chunk 2048 = topk); chunk32768 packs them into the
              // sparse call, exposing the bug.  Control
              // flow is UNCHANGED (uniform full-topk loads, data-independent;
              // only the validity bit differs -> no-hang properties preserved).
              const bool t_is_pad = (t < 0);
              t = t_is_pad ? (pad_base + offs) : t;
              is_cur_token_valid = !t_is_pad;
            } else {
              is_cur_token_valid =
                  (t >= 0 && t < params.s_kv) && (offs < topk_length);
            }
            token_indices[buf_idx][local_row] =
                (int64_t)t * (int64_t)params.stride_kv_s_kv;
            is_token_valid[buf_idx][local_row] = is_cur_token_valid;
          }
        }
      };

      int64_t cache_policy = createpolicy_evict_last();

      auto copy_tiles = [&](int buf_idx, int smem_buf, int tile_start,
                            int tile_end) {
        CUTE_UNROLL
        for (int local_row = 0; local_row < NUM_ROWS_PER_GROUP; ++local_row) {
          int64_t token_index = token_indices[buf_idx][local_row];
          CUTE_UNROLL
          for (int tile_idx = tile_start; tile_idx < tile_end; tile_idx += 2) {
            int phys_tile = tile_idx + tile_shift;
            if constexpr ((D_K % 128) != 0) {
              if (phys_tile >= (D_K / 64)) continue;
            }
            bool kv_pred =
                is_token_valid[buf_idx][local_row] && phys_tile < (D_K / 64);
            cp_async_cacheglobal_l2_prefetch_256B(
                my_gKV_base + token_index + tile_idx * 64,
                my_sK_base +
                    (smem_buf * cosize_v<SmemLayoutK> +
                     tile_idx * (B_TOPK * 64) + local_row * NUM_GROUPS * 64),
                kv_pred, cache_policy);
          }
        }
      };

      auto commit_to_mbar = [&](transac_bar_t& bar) {
        cutlass::arch::cpasync_barrier_arrive_noinc((uint64_t*)(&bar));
      };

      // V transpose helper instance
      SmemTransposeV smem_transpose_v;
      using SmemLayoutTransposeV_t =
          typename SmemTransposeV::SmemLayoutTransposeV;
      using SmemLayoutTransposeVt_t =
          typename SmemTransposeV::SmemLayoutTransposeVt;
      // Half (256x64) Vt dst layout for the split vt_loc/vt_rem half-buffers.
      // Source stays the FULL [64,512] K layout (tile index j handles the
      // swizzle); only the DST is a half-buffer, tiles remapped tile_start..end
      // -> 0..3. transpose_pair is tile-agnostic (operates on 64x64 tiles) so
      // smem_transpose_v is reused.
      using SmemTransposeV_half_t = SmemTransposeFp8_64x64<B_TOPK, D_V / 2>;
      using SmemLayoutTransposeVt_half_t =
          typename SmemTransposeV_half_t::SmemLayoutTransposeVt;

      // Use the FA3-style STSM thread layout for the fp8 V transpose.
      auto transpose_v_to_half = [&](int smem_k_buf, fp8_t* vt_dst,
                                     int tile_start, int tile_end) {
        Tensor sV_src = as_position_independent_swizzle_tensor(
            make_tensor(make_smem_ptr(plan.k[smem_k_buf].data()),
                        SmemLayoutTransposeV_t{}));
        Tensor sVt_dst = as_position_independent_swizzle_tensor(
            make_tensor(make_smem_ptr(vt_dst), SmemLayoutTransposeVt_half_t{}));

        static_assert((D_V / 64 / 2) % 2 == 0,
                      "half tile count must be even for pair transpose");
        CUTE_UNROLL
        for (int j = tile_start; j < tile_end; j += 2) {
          smem_transpose_v.transpose_pair(
              flatten(sV_src(_, 0, j)), flatten(sVt_dst(_, 0, j - tile_start)),
              flatten(sV_src(_, 0, j + 1)),
              flatten(sVt_dst(_, 0, j + 1 - tile_start)));
        }
        asm volatile("" ::: "memory");
      };

      int cur_bar_wait_phase_prod = 1;

      // Prologue: prefetch the first iteration's indices before the loop.
      // Subsequent iterations' indices are prefetched during V transpose
      // of the prior iteration, hiding __ldg latency behind compute.
      load_token_indices(0);

      CUTE_NO_UNROLL
      for (int block_idx = 0; block_idx < num_topk_blocks; block_idx += 2) {
        // Indices are already loaded by the prologue or the previous
        // iteration's prefetch.
        plan.bar_k0_free.wait(cur_bar_wait_phase_prod);
        plan.bar_k1_free.wait(cur_bar_wait_phase_prod);

        // is_kv_valid write: AFTER bar_k_free waits to avoid race condition.
        // Consumers may still be reading prior iteration's is_kv_valid during
        // mask_rP until they signal k_free. Writing before waits could
        // overwrite values consumers are still reading.
        if (idx_in_group == 0) {
          CUTE_UNROLL
          for (int buf_idx = 0; buf_idx < 2; ++buf_idx) {
            CUTE_UNROLL
            for (int local_row = 0; local_row < NUM_ROWS_PER_GROUP;
                 ++local_row) {
              plan.is_kv_valid[buf_idx][(block_idx >> 1) & 1]
                              [local_row * NUM_GROUPS + group_idx] =
                  is_token_valid[buf_idx][local_row];  // parity-buffered
            }
          }
          plan.bar_is_kv_valid_ready[(block_idx >> 1) & 1]
              .arrive();  // index by iter parity
        }

        copy_tiles(0, 0, 0, 4);
        commit_to_mbar(plan.bar_k0_ready[0]);
        asm volatile("cp.async.commit_group;\n" ::);

        constexpr int kv_tile_end = D_K / 64;

        copy_tiles(1, 1, 4, kv_tile_end);
        commit_to_mbar(plan.bar_k1_ready[1]);

        copy_tiles(0, 0, 4, kv_tile_end);
        commit_to_mbar(plan.bar_k0_ready[1]);

        copy_tiles(1, 1, 0, 4);
        commit_to_mbar(plan.bar_k1_ready[0]);
        asm volatile("cp.async.commit_group;\n" ::);

        // Wait for K[0]-left. NOTE: keep cp.async.wait_group at 1, not 0 -
        // wait_group 0 re-times the producer into a latent cross-iteration
        // race; the consumer must overlap the in-flight K[1] load here.
        asm volatile("cp.async.wait_group 1;\n" ::);
        // fence.proxy.async: make cp.async data visible through generic proxy
        // (required for LDSM reads in V transpose; cp.async uses async proxy)
        fence_view_async_shared();
        asm volatile("bar.sync 7, 128;\n" ::: "memory");

        if (block_idx > 0) {
          plan.bar_vt_free[0].wait(cur_bar_wait_phase_prod);
        }

        // Prefetch next iteration's indices before V[0]-LEFT transpose so the
        // __ldg latency is hidden behind the full V transpose window.
        if (block_idx + 2 < num_topk_blocks) {
          load_token_indices(block_idx + 2);
        }

        transpose_v_to_half(0, plan.vt_loc[0].data(), 0,
                            4);  // block0-left  -> WG0 local
        // The arrive is placed below the wait_group-0 + fence point: the
        // fence_view_async_shared there also orders these STSM (generic proxy)
        // writes for the consumer's WGMMA (async proxy) reads.  Without the
        // fence the STSM->WGMMA proxy crossing is unordered and corrupts
        // values.

        // Transpose V[1] left before V[0] right to match the consumer handoff
        // order. WG0 is on the critical path (feeds WG1 via sM/wg0_bunch). WG0
        // waits for vt1_for_wg0 (V[1]-LEFT) for PV-remote. Moving V[1]-LEFT
        // earlier (2nd instead of 4th) reduces WG0 critical-path stall.

        // NOTE: K[1]-left tiles 0-3 are in cp.async group-1,
        // NOT group-0. wait_group 1 only waits for group-0. Under high
        // CTA counts (512+), memory bandwidth saturation delays group-1
        // completion past the V[0]-LEFT transpose timing margin, causing
        // the V[1]-LEFT transpose to read stale/partial smem data.
        // Fix: wait_group 0 before V[1]-LEFT ensures group-1 has completed.
        // V[0]-LEFT transpose still overlaps with group-1 async copies.

        // Wait for all groups before V[1]-LEFT transpose
        asm volatile("cp.async.wait_group 0;\n" ::);
        // fence.proxy.async: make cp.async group-1 data visible through
        // generic proxy for LDSM reads in V transpose
        fence_view_async_shared();
        asm volatile("bar.sync 7, 128;\n" ::: "memory");
        NamedBarrier::arrive(256,
                             vt0_left_ready);  // covered by the fence above

        // V[1]-LEFT: tiles 0-3 from K[1] -- WG0 needs this for PV-remote
        if (block_idx > 0) {
          plan.bar_vt_free[1].wait(cur_bar_wait_phase_prod);
        }
        transpose_v_to_half(1, plan.vt_rem[0].data(), 0,
                            4);  // block1-left  -> WG0 remote
        // The fence is cheap here: cp.async queue already drained by
        // wait_group 0 above; orders V[1]L STSM writes before the arrive.
        fence_view_async_shared();
        NamedBarrier::arrive(256, vt1_for_wg0);

        // V[0]-RIGHT: tiles 4-7 from K[0]
        transpose_v_to_half(0, plan.vt_rem[1].data(), 4,
                            8);  // block0-right -> WG1 remote
        // V[1]-RIGHT: tiles 4-7 from K[1]
        transpose_v_to_half(1, plan.vt_loc[1].data(), 4,
                            8);  // block1-right -> WG1 local
        // One fence covers both right-half transposes, then arrive both.
        fence_view_async_shared();
        NamedBarrier::arrive(256, vt0_right_ready);
        NamedBarrier::arrive(256, vt1_for_wg1);

        asm volatile("bar.sync 7, 128;\n" ::: "memory");

        cur_bar_wait_phase_prod ^= 1;
      }
    }

    cute::tma_store_wait<0>();
#else
    if (cute::thread0()) {
      CUTE_INVALID_CONTROL_PATH("This kernel only supports sm90");
    }
#endif
  }

  // ========================================================================
  // run() -- host-side launch
  // ========================================================================
  static void run(const SparseMlaQ8Kv8PrefillParams& params) {
    KU_ASSERT(params.h_kv == 1);
    KU_ASSERT(params.topk % (2 * B_TOPK) == 0);
    KU_ASSERT(params.topk > 0);
    KU_ASSERT(params.h_q % B_H == 0);

    CUtensorMap tensor_map_O;
    {
      uint64_t size[3] = {(uint64_t)D_V, (uint64_t)params.h_q,
                          (uint64_t)params.s_q};
      uint64_t stride[2] = {D_V * sizeof(bf16),
                            D_V * params.h_q * sizeof(bf16)};
      uint32_t box_size[3] = {64, B_H, 1};
      uint32_t elem_stride[3] = {1, 1, 1};
      CUresult res = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
          &tensor_map_O, CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
          3, params.out, size, stride, box_size, elem_stride,
          CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
          CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
          CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
          CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      KU_ASSERT(res == CUresult::CUDA_SUCCESS);
    }

    TmaParams_t tma_p = {tensor_map_O};

    auto kernel = &sparse_mla_q8kv8_prefill_kernel<
        SparseMlaQ8Kv8PrefillKernel<D_QK, HAVE_TOPK_LENGTH, HAVE_ATTN_SINK>,
        TmaParams_t>;

    constexpr size_t smem_size = sizeof(SharedMemoryPlan);
    KU_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    cutlass::ClusterLaunchParams launch_params = {
        dim3((params.h_q / B_H) * params.s_q, 1, 1), dim3(NUM_THREADS, 1, 1),
        dim3(1, 1, 1), smem_size, params.stream};
    cutlass::launch_kernel_on_cluster(launch_params, (void*)kernel, params,
                                      tma_p);
    KU_CHECK_KERNEL_LAUNCH();
  }
};

// ============================================================================
// Global kernel entry point
// ============================================================================
template <typename Kernel, typename TMAParamsT>
__global__ void __launch_bounds__(Kernel::NUM_THREADS, 1, 1)
    sparse_mla_q8kv8_prefill_kernel(
        __grid_constant__ const SparseMlaQ8Kv8PrefillParams params,
        __grid_constant__ const TMAParamsT tma_params) {
  Kernel::devfunc(params, tma_params);
}

// ============================================================================
// External dispatch function
// ============================================================================
template <int D_QK, bool HAVE_TOPK_LENGTH, bool HAVE_ATTN_SINK>
void run_sparse_mla_q8kv8_prefill_kernel(
    const SparseMlaQ8Kv8PrefillParams& params) {
  SparseMlaQ8Kv8PrefillKernel<D_QK, HAVE_TOPK_LENGTH, HAVE_ATTN_SINK>::run(
      params);
}

}  // namespace fwd
}  // namespace sm90
