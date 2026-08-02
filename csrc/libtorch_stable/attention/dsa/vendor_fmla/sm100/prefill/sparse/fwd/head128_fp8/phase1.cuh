#pragma once
#include "phase1.h"

#include <math_constants.h>
#include <cuda_fp8.h>
#include <cute/tensor.hpp>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/arch/arch.h>
#include <cutlass/cuda_host_adapter.hpp>

#include "params.h"
#include "utils.h"
#include "sm100/helpers.h"

#include "config.h"

namespace sm100::fwd::head128_fp8 {

using namespace cute;

CUTE_DEVICE int32x8_t ldg_256_indices(void* src_ptr) {
  int32x8_t val;
  const int4 lo = __ldg((const int4*)src_ptr);
  const int4 hi = __ldg((const int4*)src_ptr + 1);
  val.a0 = lo.x;
  val.a1 = lo.y;
  val.a2 = lo.z;
  val.a3 = lo.w;
  val.a4 = hi.x;
  val.a5 = hi.y;
  val.a6 = hi.z;
  val.a7 = hi.w;
  return val;
}

// Pipeline identical to the bf16 kernel (see ../head128/phase1.cuh); Q now
// lives entirely in smem so the S->T (UTCCP) stage is gone.

template <int D_QK>
template <typename TmaParams>
__device__ void KernelTemplate<D_QK>::sparse_attn_fwd_kernel_devfunc(
    const SparseAttnFwdParams& params, const TmaParams& tma_params) {
#if (defined(__CUDA_ARCH__) &&                           \
     (__CUDA_ARCH__ >= 1000 && __CUDA_ARCH__ < 1200)) || \
    (defined(__CLION_IDE__) || defined(__VSCODE_IDE__))
  const int cta_idx = blockIdx.x % 2;
  const int s_q_idx = blockIdx.x / 2;
  const int warp_idx = cutlass::canonical_warp_idx_sync();
  const int lane_idx = threadIdx.x % 32;
  int topk_length = params.topk_length != nullptr
                        ? __ldg(params.topk_length + s_q_idx)
                        : params.topk;
  if (params.topk_length_per_q != nullptr) {
    // group's block loop runs to the LAST query's bound (= union total)
    const int G_ = params.h_q / params.h_per_q;
    topk_length =
        __ldg(params.topk_length_per_q + (long)s_q_idx * G_ + (G_ - 1));
  }
  const int num_k_blocks = max(cute::ceil_div(topk_length, (int)B_TOPK), 1);
  const int warpgroup_idx = __shfl_sync(0xffffffff, threadIdx.x / 128, 0);
  const int idx_in_warpgroup = threadIdx.x % 128;

  if (threadIdx.x == 0) {
    cute::prefetch_tma_descriptor(tma_params.tma_Q.get_tma_descriptor());
    cute::prefetch_tma_descriptor(tma_params.tma_O.get_tma_descriptor());
    cute::prefetch_tma_descriptor(&(tma_params.tensor_map_kv));
    cute::prefetch_tma_descriptor(&(tma_params.tensor_map_kv_v));
  }

  extern __shared__ char wksp_buf[];
  SharedMemoryPlan& plan = *reinterpret_cast<SharedMemoryPlan*>(wksp_buf);
  Tensor sQ_full = make_tensor(make_smem_ptr(plan.u.s.q_full.data()),
                               SmemLayoutQTiles<D_Q / 64>{});

  int* gIndices =
      params.indices + s_q_idx * params.stride_indices_s_q;  // [topk]

  TiledMMA tiled_mma_P = TiledMMA_P{};
  TiledMMA tiled_mma_O = TiledMMA_O{};
  Tensor tP =
      partition_fragment_C(tiled_mma_P, Shape<Int<B_H / 2>, Int<B_TOPK>>{});
  Tensor tO =
      partition_fragment_C(tiled_mma_O, Shape<Int<B_H / 2>, Int<D_V>>{});
  tP.data().get() = tmem_cols::p;
  tO.data().get() = tmem_cols::o;

  if (warp_idx == 0) {
    if (elect_one_sync()) {
      plan.bar_prologue_q.init(1);
      CUTE_UNROLL
      for (int i = 0; i < NUM_BUFS; ++i) {
        plan.bar_qk_part_done[i].init(1);
        plan.bar_qk_done[i].init(1);
        plan.bar_sv_part_done[i].init(1);
        plan.bar_sv_done[i].init(1);

        plan.bar_v_part0_ready[i].init(1);
        plan.bar_v_part1_ready[i].init(1);
        plan.bar_k_part0_ready[i].init(1);
        plan.bar_k_part1_ready[i].init(1);
        plan.bar_p_free[i].init(128 * 2);
        plan.bar_so_ready[i].init(128 * 2);
        plan.bar_k_valid_ready[i].init(16);
        plan.bar_k_valid_free[i].init(128);
      }
      fence_barrier_init();
    }
  }

  cute::cluster_sync();

  if (warp_idx == 0) {
    if (elect_one_sync()) {
      Tensor gQ =
          flat_divide(tma_params.tma_Q.get_tma_tensor(tma_params.shape_Q)(
                          _, _, s_q_idx / params.q_group_div),
                      Tile<Int<B_H / 2>>{})(_, cta_idx, _);
      ku::launch_tma_copy(tma_params.tma_Q, gQ, sQ_full, plan.bar_prologue_q,
                          TMA::CacheHintSm90::EVICT_FIRST);
    }

    cute::TMEM::Allocator2Sm().allocate(512, plan.tmem_start_addr.data());
    TRAP_ONLY_DEVICE_ASSERT(plan.tmem_start_addr.data()[0] == 0);
    cute::TMEM::Allocator2Sm().release_allocation_lock();
  }

  __syncthreads();  // Wait for TMEM allocation

  if (warpgroup_idx == 0) {
    cutlass::arch::warpgroup_reg_alloc<144>();
    // Scale & Exp warps

    float mi = MAX_INIT_VAL;
    float li = 0.0f;
    float real_mi = -CUDART_INF_F;

    const float2 scale =
        float2{params.sm_scale_div_log2, params.sm_scale_div_log2};
    // e4m3 P store: thread t (and t+64) writes row t%64; its 64 values
    // form 4 u128 (16 e4m3 each) at column-block stride 64 u128s.
    // Per-query causal prefix bound (qlen mode): loaded once per row.
    int row_bound = INT_MAX;
    if (params.topk_length_per_q != nullptr) {
      const int row = cta_idx * (B_H / 2) + idx_in_warpgroup % 64;
      const int G_ = params.h_q / params.h_per_q;
      row_bound = __ldg(params.topk_length_per_q + (long)s_q_idx * G_ +
                        row / params.h_per_q);
    }
    // Query-major membership (exact tier, no warp13 transpose): this
    // row's two mask words per block, software-pipelined one block
    // ahead like the producers' index loads.
    const uint32_t* qm_row = nullptr;
    uint2 qm_cur, qm_nx;
    if (params.membership_qm != nullptr) {
      const int row = cta_idx * (B_H / 2) + idx_in_warpgroup % 64;
      const int G_ = params.h_q / params.h_per_q;
      const int capw = params.topk / 32;
      qm_row = params.membership_qm +
               ((long)s_q_idx * G_ + row / params.h_per_q) * capw +
               (idx_in_warpgroup >= 64 ? 2 : 0);
      qm_cur = __ldg((const uint2*)qm_row);
    }
    uint128_t* sS_base0 = (uint128_t*)plan.s[0].data() + idx_in_warpgroup % 64 +
                          64 * ((idx_in_warpgroup / 64) * 4);
    uint128_t* sS_base1 = (uint128_t*)plan.s[1].data() + idx_in_warpgroup % 64 +
                          64 * ((idx_in_warpgroup / 64) * 4);

    CUTE_NO_UNROLL
    for (int k = 0; k < num_k_blocks; ++k) {
      if (qm_row != nullptr && k + 1 < num_k_blocks)
        qm_nx = __ldg((const uint2*)(qm_row + (size_t)(k + 1) * (B_TOPK / 32)));
      plan.bar_qk_done[k % NUM_BUFS].wait((k / NUM_BUFS) & 1);
      ku::tcgen05_after_thread_sync();

      float2 p[(B_TOPK / 2) / 2];
      ku::tmem_ld_32dp32bNx<B_TOPK / 2>(tmem_cols::p, p);
      cutlass::arch::fence_view_async_tmem_load();
      ku::tcgen05_before_thread_sync();
      plan.bar_p_free[k % NUM_BUFS].arrive(0u);

      // Mask
      plan.bar_k_valid_ready[k % NUM_BUFS].wait((k / NUM_BUFS) & 1);
      const char* kv_valid_base = plan.is_k_valid[k % NUM_BUFS];
      if (params.membership != nullptr) {
        const int row = cta_idx * (B_H / 2) + idx_in_warpgroup % 64;
        kv_valid_base = plan.is_kq_valid[k % NUM_BUFS][row / params.h_per_q];
      }
      uint32_t is_k_valid_lo =
          *(uint32_t*)(kv_valid_base +
                       (idx_in_warpgroup >= 64 ? B_TOPK / 8 / 2 : 0));
      uint32_t is_k_valid_hi =
          *(uint32_t*)(kv_valid_base +
                       (idx_in_warpgroup >= 64 ? B_TOPK / 8 / 2 : 0) + 4);
      if (params.topk_length_per_q != nullptr) {
        // causal prefix folded into the validity bits: two ops
        // instead of a 64-iteration compare loop
        const int abs0 = k * B_TOPK + (idx_in_warpgroup >= 64 ? B_TOPK / 2 : 0);
        const int n = min(max(row_bound - abs0, 0), (int)(B_TOPK / 2));
        const uint64_t pm = n >= 64 ? ~0ull : ((1ull << n) - 1ull);
        is_k_valid_lo &= (uint32_t)pm;
        is_k_valid_hi &= (uint32_t)(pm >> 32);
      }
      if (qm_row != nullptr) {
        is_k_valid_lo &= qm_cur.x;
        is_k_valid_hi &= qm_cur.y;
        qm_cur = qm_nx;
      }
      float* p_float = (float*)p;
      CUTE_UNROLL
      for (int i = 0; i < (B_TOPK / 2) / 2; i += 1) {
        if (!(is_k_valid_lo >> i & 1)) p_float[i] = -CUDART_INF_F;
      }
      CUTE_UNROLL
      for (int i = 0; i < (B_TOPK / 2) / 2; i += 1) {
        if (!(is_k_valid_hi >> i & 1))
          p_float[i + (B_TOPK / 2) / 2] = -CUDART_INF_F;
      }

      float cur_pi_max = -CUDART_INF_F;
      CUTE_UNROLL
      for (int i = 0; i < (B_TOPK / 2); i += 1) {
        cur_pi_max = max(cur_pi_max, p_float[i]);
      }
      cur_pi_max *= params.sm_scale_div_log2;

      plan.bar_k_valid_free[k % NUM_BUFS].arrive();

      // parity-buffered exchange: one barrier per block instead of two
      {
        float* mbuf = plan.rowwise_max_buf[k & 1];
        mbuf[idx_in_warpgroup] = cur_pi_max;
        NamedBarrier::arrive_and_wait(128, 0);
        cur_pi_max = max(cur_pi_max, mbuf[idx_in_warpgroup ^ 64]);
      }
      real_mi = max(real_mi, cur_pi_max);
      bool should_scale_o = __any_sync(0xffffffff, cur_pi_max - mi > 6.0f);

      float new_max, scale_for_old;
      if (!should_scale_o) {
        scale_for_old = 1.0f;
        new_max = mi;
      } else {
        new_max = max(cur_pi_max, mi);
        scale_for_old = exp2f(mi - new_max);
      }
      mi = new_max;
      li *= scale_for_old;

      // Calculate S = exp2(p*scale - new_max + log2(7)) as e4m3.
      // The pre-scale cancels between O and li at the epilogue; 7
      // leaves 2^6 headroom for the hysteresis overshoot (values can
      // reach exp2(6)*7 = 448 = e4m3 max, never clipped).
      // ex2 on f16x2 halves the SFU work; e4m3's 2^-3 rounding
      // dominates half's 2^-11, so P precision is unchanged.
      __nv_fp8x2_storage_t s8[(B_TOPK / 2) / 2];
      float2 neg_new_max =
          float2{-new_max + P_SCALE_LOG2, -new_max + P_SCALE_LOG2};
      CUTE_UNROLL
      for (int i = 0; i < (B_TOPK / 2) / 2; i += 1) {
        float2 d = ku::float2_fma(p[i], scale, neg_new_max);
        d.x = exp2f(d.x);
        d.y = exp2f(d.y);
        li += d.x + d.y;
        s8[i] = __nv_cvt_float2_to_fp8x2(d, __NV_SATFINITE, __NV_E4M3);
      }

      // S double-buffered: wait only for the SV gemm that last
      // consumed THIS buffer (block k-2).
      if (k > 1) {
        plan.bar_sv_done[(k - 2) % NUM_BUFS].wait(((k - 2) / NUM_BUFS) & 1);
      }
      uint128_t* sS_base = (k & 1) ? sS_base1 : sS_base0;
      CUTE_UNROLL
      for (int i = 0; i < (B_TOPK / 2) / 16; i += 1) {
        sS_base[64 * i] = *(uint128_t*)(s8 + i * 8);
      }

      // Scale O (needs SV k-1 complete; only on rescale blocks)
      if (k > 0 && should_scale_o) {
        float2 scale_for_old_float2 = float2{scale_for_old, scale_for_old};
        plan.bar_sv_done[(k - 1) % NUM_BUFS].wait(((k - 1) / NUM_BUFS) & 1);
        ku::tcgen05_after_thread_sync();

        static constexpr int CHUNK_SIZE = 32;
        float2 o[CHUNK_SIZE / 2];
        CUTE_UNROLL
        for (int chunk_idx = 0; chunk_idx < (D_V / 2) / CHUNK_SIZE;
             ++chunk_idx) {
          ku::tmem_ld_32dp32bNx<CHUNK_SIZE>(
              tmem_cols::o + chunk_idx * CHUNK_SIZE, o);
          cutlass::arch::fence_view_async_tmem_load();
          for (int i = 0; i < CHUNK_SIZE / 2; ++i) {
            o[i] = ku::float2_mul(o[i], scale_for_old_float2);
          }
          ku::tmem_st_32dp32bNx<CHUNK_SIZE>(
              tmem_cols::o + chunk_idx * CHUNK_SIZE, o);
          cutlass::arch::fence_view_async_tmem_store();
        }
        ku::tcgen05_before_thread_sync();
      }

      fence_view_async_shared();
      plan.bar_so_ready[k % NUM_BUFS].arrive(0u);
    }

    // Epilogue

    if (real_mi == -CUDART_INF_F) {
      li = 0.0f;
      mi = -CUDART_INF_F;
    }

    plan.rowwise_li_buf[idx_in_warpgroup] = li;
    NamedBarrier::arrive_and_wait(128, 0);
    li += plan.rowwise_li_buf[idx_in_warpgroup ^ 64];

    if (idx_in_warpgroup < 64) {
      int global_index =
          s_q_idx * params.h_q + cta_idx * (B_H / 2) + idx_in_warpgroup;
      // li carries the P pre-scale; remove it from the lse
      float cur_lse = logf(li) - P_SCALE_LN + mi * CUDART_LN2_F;
      cur_lse = cur_lse == -CUDART_INF_F ? +CUDART_INF_F : cur_lse;
      params.max_logits[global_index] = real_mi * CUDART_LN2_F;
      params.lse[global_index] = cur_lse;
    }

    plan.bar_sv_done[(num_k_blocks - 1) % NUM_BUFS].wait(
        ((num_k_blocks - 1) / NUM_BUFS) & 1);
    ku::tcgen05_after_thread_sync();

    // Store O. out_scale carries the V dequant factor (k_scale); the P
    // pre-scale cancels between the O accumulator and li.
    float attn_sink = params.attn_sink == nullptr
                          ? -CUDART_INF_F
                          : __ldg(params.attn_sink + cta_idx * B_H / 2 +
                                  (idx_in_warpgroup % 64)) *
                                CUDART_L2E_F;
    float output_scale =
        __fdividef(params.out_scale, li + exp2f(attn_sink - mi + P_SCALE_LOG2));
    Tensor sO = make_tensor(make_smem_ptr(plan.u.o.data()), SmemLayoutO{});
    constexpr int B_EPI = 64;
    Tensor tma_gO = flat_divide(
        tma_params.tma_O.get_tma_tensor(tma_params.shape_O)(_, _, s_q_idx),
        Shape<Int<B_H / 2>, Int<B_EPI>>{})(_, _, cta_idx, _);
    Tensor sO_divided =
        flat_divide(sO, Shape<Int<B_H / 2>, Int<B_EPI>>{})(_, _, _0{}, _);
    auto thr_tma = tma_params.tma_O.get_slice(_0{});

    float2 o[B_EPI / 2];
    bool have_valid_indices = __any_sync(0xffffffff, li != 0);
    if (!have_valid_indices) {
      CUTE_UNROLL
      for (int i = 0; i < B_EPI / 2; ++i) o[i].x = o[i].y = 0.0f;
      output_scale = 1.0f;
    }

    float2 output_scale_float2 = make_float2(output_scale, output_scale);

    CUTE_UNROLL
    for (int k = 0; k < (D_V / 2) / B_EPI; ++k) {
      if (have_valid_indices) {
        ku::tmem_ld_32dp32bNx<B_EPI>(tmem_cols::o + k * B_EPI, o);
        cutlass::arch::fence_view_async_tmem_load();
      }

      CUTE_UNROLL
      for (int i = 0; i < B_EPI / 8; ++i) {
        __nv_bfloat162 o_bf16[4];
        CUTE_UNROLL
        for (int j = 0; j < 4; ++j) {
          float2 d = ku::float2_mul(o[i * 4 + j], output_scale_float2);
          o_bf16[j] = __float22bfloat162_rn(d);
        }
        int smem_row = idx_in_warpgroup % 64;
        int smem_col = (idx_in_warpgroup / 64) * (D_V / 2) + k * B_EPI + i * 8;
        *(uint128_t*)(&sO(smem_row, smem_col)) = *(uint128_t*)(o_bf16);
      }

      fence_view_async_shared();
      NamedBarrier::arrive_and_wait(128, 0);

      if (warp_idx == 0 && elect_one_sync()) {
        cute::copy(tma_params.tma_O, thr_tma.partition_S(sO_divided(_, _, k)),
                   thr_tma.partition_D(tma_gO(_, _, k)));
      }
      if (warp_idx == 1 && elect_one_sync()) {
        int k2 = k + (D_V / B_EPI / 2);
        cute::copy(tma_params.tma_O, thr_tma.partition_S(sO_divided(_, _, k2)),
                   thr_tma.partition_D(tma_gO(_, _, k2)));
      }
    }

    if (warp_idx == 0) {
      cute::TMEM::Allocator2Sm().free(0, 512);
    }
  } else if (warpgroup_idx == 1) {
    // Producer warps for K (fp8: 64-elem cols via the SW64 map)
    cutlass::arch::warpgroup_reg_dealloc<96>();
    int warp_idx = cutlass::canonical_warp_idx_sync() - 4;
    constexpr int NUM_WARPS = 4,
                  NUM_LOCAL_ROWS_PER_WARP = (B_TOPK / 2) / 4 / NUM_WARPS;
    if (elect_one_sync()) {
      // index loads software-pipelined one block ahead (8.7%+4.3% of
      // stall samples were the raw __ldg latency here)
      int4 indices[NUM_LOCAL_ROWS_PER_WARP],
          indices_nx[NUM_LOCAL_ROWS_PER_WARP];
      CUTE_UNROLL
      for (int r = 0; r < NUM_LOCAL_ROWS_PER_WARP; ++r)
        indices[r] = __ldg((int4*)(gIndices + cta_idx * (B_TOPK / 2)) +
                           r * NUM_WARPS + warp_idx);
      CUTE_NO_UNROLL
      for (int k = 0; k < num_k_blocks; ++k) {
        fp8_e4m3* sK_base = plan.u.s.k[k & 1].data() + warp_idx * 4 * 64;
        if (k + 1 < num_k_blocks) {
          CUTE_UNROLL
          for (int r = 0; r < NUM_LOCAL_ROWS_PER_WARP; ++r)
            indices_nx[r] = __ldg(
                (int4*)(gIndices + (k + 1) * B_TOPK + cta_idx * (B_TOPK / 2)) +
                r * NUM_WARPS + warp_idx);
        }
        int max_indices = -1, min_indices = params.s_kv;
        CUTE_UNROLL
        for (int local_row = 0; local_row < NUM_LOCAL_ROWS_PER_WARP;
             ++local_row) {
          max_indices = max(max_indices, int4_max(indices[local_row]));
          min_indices = min(min_indices, int4_min(indices[local_row]));
        }
        bool is_all_rows_invalid =
            min_indices == params.s_kv || max_indices == -1;
        bool should_skip_tma = is_all_rows_invalid && k >= NUM_BUFS;

        // 128-elem SW128 boxes for dims [0,512): smem element base of
        // (token t, 128-chunk c) = c*(B_TOPK/2)*128 + t*128; the SW64
        // rope tail [512,576) lives after the SW128 region.
        fp8_e4m3* sK_tail = plan.u.s.k[k & 1].data() + (B_TOPK / 2) * D_SW128 +
                            warp_idx * 4 * 64;
        auto load_k128 = [&](transac_bar_t& bar, int c0, int c1) {
          CUTE_UNROLL
          for (int local_row = 0; local_row < NUM_LOCAL_ROWS_PER_WARP;
               ++local_row) {
            CUTE_UNROLL
            for (int c = c0; c < c1; ++c)
              ku::tma_gather4_cta_group_2<true>(
                  &(tma_params.tensor_map_kv_v), bar,
                  plan.u.s.k[k & 1].data() + c * ((B_TOPK / 2) * 128) +
                      (warp_idx * 4 + local_row * (4 * NUM_WARPS)) * 128,
                  c * 128, indices[local_row],
                  (int64_t)TMA::CacheHintSm90::EVICT_LAST);
          }
        };
        auto load_ktail = [&](transac_bar_t& bar) {
          CUTE_UNROLL
          for (int local_row = 0; local_row < NUM_LOCAL_ROWS_PER_WARP;
               ++local_row)
            ku::tma_gather4_cta_group_2<true>(
                &(tma_params.tensor_map_kv), bar,
                sK_tail + local_row * (4 * NUM_WARPS) * 64, D_SW128,
                indices[local_row], (int64_t)TMA::CacheHintSm90::EVICT_LAST);
        };

        int cur_buf = k % NUM_BUFS;
        // K double-buffered: only wait for QK of block k-2 (this
        // buffer's previous consumer) -> gathers run 2 blocks ahead.
        if (k > 1) {
          plan.bar_qk_done[(k - 2) % NUM_BUFS].wait(((k - 2) / NUM_BUFS) & 1);
        }
        if (!should_skip_tma) {
          load_k128(plan.bar_k_part0_ready[cur_buf], 0, D_PART0 / 128);
        } else {
          plan.bar_k_part0_ready[cur_buf].complete_transaction(
              0u, NUM_LOCAL_ROWS_PER_WARP * 4 * D_PART0 * sizeof(fp8_e4m3), 1u);
        }

        if (!should_skip_tma) {
          load_k128(plan.bar_k_part1_ready[cur_buf], D_PART0 / 128,
                    D_SW128 / 128);
          load_ktail(plan.bar_k_part1_ready[cur_buf]);
        } else {
          plan.bar_k_part1_ready[cur_buf].complete_transaction(
              0u, NUM_LOCAL_ROWS_PER_WARP * 4 * D_PART1 * sizeof(fp8_e4m3), 1u);
        }
        CUTE_UNROLL
        for (int r = 0; r < NUM_LOCAL_ROWS_PER_WARP; ++r)
          indices[r] = indices_nx[r];
      }
    }
  } else if (warpgroup_idx == 2) {
    // Producer warps for V (fp8: 128-elem cols via the SW128 map; no
    // UTCCP wait — V no longer aliases Q)
    cutlass::arch::warpgroup_reg_dealloc<96>();
    int warp_idx = cutlass::canonical_warp_idx_sync() - 8;
    constexpr int NUM_WARPS = 4;

    if (elect_one_sync()) {
      constexpr int NROWS_V = B_TOPK / 4 / NUM_WARPS;
      int4 vrows[NROWS_V], vrows_nx[NROWS_V];
      CUTE_UNROLL
      for (int r = 0; r < NROWS_V; ++r)
        vrows[r] = __ldg((int4*)(gIndices) + r * NUM_WARPS + warp_idx);
      CUTE_NO_UNROLL
      for (int k = 0; k < num_k_blocks; ++k) {
        fp8_e4m3* sV_base = plan.u.s.v[k & 1].data() + warp_idx * 4 * 128;
        if (k + 1 < num_k_blocks) {
          CUTE_UNROLL
          for (int r = 0; r < NROWS_V; ++r)
            vrows_nx[r] = __ldg((int4*)(gIndices + (k + 1) * B_TOPK) +
                                r * NUM_WARPS + warp_idx);
        }
        auto load_part_vi = [&](transac_bar_t& bar, int local_row_start,
                                int local_row_end) {
          CUTE_UNROLL
          for (int local_row = local_row_start; local_row < local_row_end;
               ++local_row) {
            int4 token_idxs = vrows[local_row];
            CUTE_UNROLL
            for (int local_col = 0; local_col < (D_V / 2) / 128; ++local_col)
              ku::tma_gather4_cta_group_2<true>(
                  &(tma_params.tensor_map_kv_v), bar,
                  sV_base + local_row * (4 * NUM_WARPS) * 128 +
                      local_col * (B_TOPK * 128),
                  local_col * 128 + (cta_idx ? 256 : 0), token_idxs,
                  (int64_t)TMA::CacheHintSm90::EVICT_LAST);
          }
        };

        int cur_buf = k % NUM_BUFS;
        // V double-buffered: wait only for SV of block k-2.
        if (k > 1) {
          plan.bar_sv_done[(k - 2) % NUM_BUFS].wait(((k - 2) / NUM_BUFS) & 1);
        }
        load_part_vi(plan.bar_v_part0_ready[cur_buf], 0,
                     (B_TOPK / 2) / 4 / NUM_WARPS);
        load_part_vi(plan.bar_v_part1_ready[cur_buf],
                     (B_TOPK / 2) / 4 / NUM_WARPS, B_TOPK / 4 / NUM_WARPS);
        CUTE_UNROLL
        for (int r = 0; r < NROWS_V; ++r) vrows[r] = vrows_nx[r];
      }
    }
  } else {
    cutlass::arch::warpgroup_reg_alloc<168>();

    // MMA warp
    if (cta_idx == 0 && warp_idx == 12 && elect_one_sync()) {
      // Wait for Q (all-smem; no S->T copy)
      plan.bar_prologue_q.arrive_and_expect_tx(B_H * D_K * sizeof(fp8_e4m3));
      plan.bar_prologue_q.wait(0);

      CUTE_NO_UNROLL
      for (int k = 0; k < num_k_blocks + 1; ++k) {
        if (k < num_k_blocks) {
          // Pi = QKi^T
          int cur_buf = k % NUM_BUFS;
          Tensor sQ0 = make_tensor(make_smem_ptr(plan.u.s.q_full.data()),
                                   SmemLayoutQTiles<NUM_P0_TILES>{});
          Tensor sQ1 = make_tensor(
              make_smem_ptr(plan.u.s.q_full.data() + (B_H / 2) * D_PART0),
              SmemLayoutQTiles<(D_SW128 - D_PART0) / 64>{});
          Tensor sQ2 = make_tensor(
              make_smem_ptr(plan.u.s.q_full.data() + (B_H / 2) * D_SW128),
              SmemLayoutQTiles<1>{});
          Tensor sK0 = make_tensor(make_smem_ptr(plan.u.s.k[k & 1].data()),
                                   SmemLayoutK128<D_PART0 / 128>{});
          Tensor sK1 = make_tensor(
              make_smem_ptr(plan.u.s.k[k & 1].data() + (B_TOPK / 2) * D_PART0),
              SmemLayoutK128<(D_SW128 - D_PART0) / 128>{});
          Tensor sK2 = make_tensor(
              make_smem_ptr(plan.u.s.k[k & 1].data() + (B_TOPK / 2) * D_SW128),
              SmemLayoutKTail{});

          plan.bar_k_part0_ready[cur_buf].arrive_and_expect_tx(
              B_TOPK * D_PART0 * sizeof(fp8_e4m3));
          plan.bar_k_part0_ready[cur_buf].wait((k / NUM_BUFS) & 1);
          if (k > 0) {
            plan.bar_p_free[(k - 1) % NUM_BUFS].wait(((k - 1) / NUM_BUFS) & 1);
          }
          ku::tcgen05_after_thread_sync();

          ku::utcmma_ss(tiled_mma_P, sQ0, sK0, tP, true);
          ku::umma_arrive_multicast_2x1SM_noelect(
              plan.bar_qk_part_done[cur_buf], 1 | 2);

          plan.bar_k_part1_ready[cur_buf].arrive_and_expect_tx(
              B_TOPK * D_PART1 * sizeof(fp8_e4m3));
          plan.bar_k_part1_ready[cur_buf].wait((k / NUM_BUFS) & 1);
          ku::tcgen05_after_thread_sync();

          ku::utcmma_ss(tiled_mma_P, sQ1, sK1, tP, false);
          ku::utcmma_ss(tiled_mma_P, sQ2, sK2, tP, false);
          ku::umma_arrive_multicast_2x1SM_noelect(plan.bar_qk_done[cur_buf],
                                                  1 | 2);
        }
        if (k > 0) {
          // O += S(i-1)V(i-1)
          int cur_buf = (k - 1) % NUM_BUFS;

          Tensor sS = make_tensor(make_smem_ptr(plan.s[(k - 1) & 1].data()),
                                  SmemLayoutS{});
          Tensor sV = make_tensor(make_smem_ptr(plan.u.s.v[(k - 1) & 1].data()),
                                  SmemLayoutV{});
          Tensor sS_divided = flat_divide(sS, Tile<Int<B_H / 2>, _64>{})(
              _, _, _0{}, _);  // (B_H/2, 64, 2)
          Tensor sV_divided = flat_divide(sV, Tile<Int<D_V / 2>, _64>{})(
              _, _, _0{}, _);  // (D_V/2, 64, 2)

          plan.bar_so_ready[cur_buf].wait(((k - 1) / NUM_BUFS) & 1);

          plan.bar_v_part0_ready[cur_buf].arrive_and_expect_tx(
              (B_TOPK / 2) * D_V * sizeof(fp8_e4m3));
          plan.bar_v_part0_ready[cur_buf].wait(((k - 1) / NUM_BUFS) & 1);
          ku::tcgen05_after_thread_sync();

          ku::utcmma_ss(tiled_mma_O, sS_divided(_, _, _0{}),
                        sV_divided(_, _, _0{}), tO, k == 1);
          ku::umma_arrive_multicast_2x1SM_noelect(
              plan.bar_sv_part_done[cur_buf], 1 | 2);

          plan.bar_v_part1_ready[cur_buf].arrive_and_expect_tx(
              (B_TOPK / 2) * D_V * sizeof(fp8_e4m3));
          plan.bar_v_part1_ready[cur_buf].wait(((k - 1) / NUM_BUFS) & 1);
          ku::tcgen05_after_thread_sync();
          ku::utcmma_ss(tiled_mma_O, sS_divided(_, _, _1{}),
                        sV_divided(_, _, _1{}), tO, false);
          ku::umma_arrive_multicast_2x1SM_noelect(plan.bar_sv_done[cur_buf],
                                                  1 | 2);
        }
      }
    } else if (warp_idx == 13) {
      // KV valid loading warp (+ per-query membership masks)
      static_assert(B_TOPK == 128);
      if (lane_idx < 16) {
        int32x8_t vind = ldg_256_indices(gIndices + lane_idx * 8);
        int32x8_t vind_nx;
        CUTE_NO_UNROLL
        for (int k = 0; k < num_k_blocks; ++k) {
          int cur_buf = k % NUM_BUFS;
          if (k + 1 < num_k_blocks)
            vind_nx =
                ldg_256_indices(gIndices + (k + 1) * B_TOPK + lane_idx * 8);
          const int32x8_t indices = vind;
          auto is_valid = [&](int rel_pos_in_lane, int index) -> char {
            int abs_pos = k * B_TOPK + lane_idx * 8 + rel_pos_in_lane;
            return index >= 0 && index < params.s_kv && abs_pos < topk_length;
          };
          char is_ks_valid_mask =
              is_valid(7, indices.a7) << 7 | is_valid(6, indices.a6) << 6 |
              is_valid(5, indices.a5) << 5 | is_valid(4, indices.a4) << 4 |
              is_valid(3, indices.a3) << 3 | is_valid(2, indices.a2) << 2 |
              is_valid(1, indices.a1) << 1 | is_valid(0, indices.a0) << 0;

          plan.bar_k_valid_free[cur_buf].wait((k / NUM_BUFS) & 1 ^ 1);
          plan.is_k_valid[cur_buf][lane_idx] = is_ks_valid_mask;
          if (params.membership != nullptr) {
            const uint16_t* mrow = params.membership +
                                   (long)s_q_idx * params.topk + k * B_TOPK +
                                   lane_idx * 8;
            uint4 mv = *(const uint4*)mrow;
            const uint16_t* mw = (const uint16_t*)&mv;
            const int G = params.h_q / params.h_per_q;
            CUTE_UNROLL
            for (int q = 0; q < 16; ++q) {
              if (q >= G) break;
              char mq = 0;
              CUTE_UNROLL
              for (int j = 0; j < 8; ++j) mq |= ((mw[j] >> q) & 1) << j;
              plan.is_kq_valid[cur_buf][q][lane_idx] = mq & is_ks_valid_mask;
            }
          }
          plan.bar_k_valid_ready[cur_buf].arrive();
          vind = vind_nx;
        }
      }
    }
  }

#else
  if (cute::thread0()) {
    CUTE_INVALID_CONTROL_PATH("This kernel only supports sm100");
  }
#endif
}

template <typename Kernel, typename TmaParams>
__global__ void __launch_bounds__(Kernel::NUM_THREADS, 1, 2)
    sparse_attn_fwd_kernel(__grid_constant__ const SparseAttnFwdParams params,
                           __grid_constant__ const TmaParams tma_params) {
  Kernel::sparse_attn_fwd_kernel_devfunc(params, tma_params);
}

template <int D_QK>
void run_fwd_phase1_kernel(const SparseAttnFwdParams& params) {
  static_assert(D_QK == 576);
  using Kernel = KernelTemplate<D_QK>;
  using fp8_e4m3 = cutlass::float_e4m3_t;

  KU_ASSERT(params.h_kv == 1);
  KU_ASSERT(params.topk % Kernel::B_TOPK == 0);
  KU_ASSERT(params.h_q == Kernel::B_H);
  KU_ASSERT(params.d_qk == D_QK);

  auto shape_Q =
      make_shape(params.h_q, params.d_qk, params.s_q / params.q_group_div);
  auto tma_Q = cute::make_tma_copy(
      SM100_TMA_2SM_LOAD_NOSPLIT{},
      make_tensor(make_gmem_ptr((fp8_e4m3*)params.q),
                  make_layout(shape_Q, make_stride(params.stride_q_h_q, _1{},
                                                   params.stride_q_s_q))),
      (typename Kernel::template SmemLayoutQTiles<D_QK / 64>){});

  auto shape_O = make_shape(params.h_q, params.d_v, params.s_q);
  auto tma_O = cute::make_tma_copy(
      SM90_TMA_STORE{},
      make_tensor(make_gmem_ptr((bf16*)params.out),
                  make_layout(shape_O, make_stride(params.d_v, _1{},
                                                   params.h_q * params.d_v))),
      (typename Kernel::template SmemLayoutOTiles<1>){});

  // K gathers: 64-element boxes, 64B swizzle
  CUtensorMap tensor_map_kv;
  {
    uint64_t size[2] = {D_QK, (unsigned long)params.s_kv};
    uint64_t stride[1] = {params.stride_kv_s_kv * sizeof(fp8_e4m3)};
    uint32_t box_size[2] = {64, 1};
    uint32_t elem_stride[2] = {1, 1};
    CUresult res = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
        &tensor_map_kv, CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8, 2,
        params.kv, size, stride, box_size, elem_stride,
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_64B,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    KU_ASSERT(res == CUresult::CUDA_SUCCESS);
  }

  // V gathers: 128-element boxes, 128B swizzle
  CUtensorMap tensor_map_kv_v;
  {
    uint64_t size[2] = {D_QK, (unsigned long)params.s_kv};
    uint64_t stride[1] = {params.stride_kv_s_kv * sizeof(fp8_e4m3)};
    uint32_t box_size[2] = {128, 1};
    uint32_t elem_stride[2] = {1, 1};
    CUresult res = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
        &tensor_map_kv_v, CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8, 2,
        params.kv, size, stride, box_size, elem_stride,
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    KU_ASSERT(res == CUresult::CUDA_SUCCESS);
  }

  TmaParams<decltype(shape_Q), decltype(tma_Q), decltype(shape_O),
            decltype(tma_O)>
      tma_params = {shape_Q, tma_Q,         shape_O,
                    tma_O,   tensor_map_kv, tensor_map_kv_v};
  auto kernel = &sparse_attn_fwd_kernel<Kernel, decltype(tma_params)>;

  constexpr size_t smem_size = sizeof(typename Kernel::SharedMemoryPlan);
  KU_CUDA_CHECK(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

  cutlass::ClusterLaunchParams launch_params = {
      dim3(2 * params.s_q, 1, 1), dim3(Kernel::NUM_THREADS, 1, 1),
      dim3(2, 1, 1), smem_size, params.stream};
  KU_CUTLASS_CHECK(cutlass::launch_kernel_on_cluster(
      launch_params, (void*)kernel, params, tma_params));
}

}  // namespace sm100::fwd::head128_fp8
