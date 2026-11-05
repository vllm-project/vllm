// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// C128 prefix-sharing adaptation contributed by wangshuo.
// Adapted from FlashMLA e2afa5fd8c220fbd05fe03fc24c9d5e64c24641b.
// Preserve its two-warpgroup arithmetic while sharing the C128 prefix.

/*
MIT License

Copyright (c) 2025 DeepSeek

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#pragma once

#include <cuda_runtime.h>
#include "params.h"
#include "cuda_kernels/legacy/sm90/prefill/sparse/config.h"
#include "cuda_kernels/legacy/sm90/helpers.h"

namespace vllm::c128 {
using namespace cute;
using namespace legacy::sm90;

struct Params : SparseAttnFwdParams {
  float *raw_o, *raw_m, *raw_l;
  const int *seq_lens, *gather_lens;
  int compressed_width, query_tokens;
};

template <bool IS_TAIL>
struct Kernel : legacy::sm90::prefill::sparse_fwd::KernelTemplate<512, true> {
  static constexpr int D_QK = 512;
  template <typename TMAParams>
  static __device__ void devfunc(const Params& params,
                                 const TMAParams& tma_params);
  static void run(const Params& params);
};

template <bool IS_TAIL>
template <typename TMAParams>
__device__ void Kernel<IS_TAIL>::devfunc(const Params& params,
                                         const TMAParams& tma_params) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 900)) || \
    (defined(__CLION_IDE__) || defined(__VSCODE_IDE__))
  const int q_h_idx = blockIdx.x % (params.h_q / B_H);
  const int s_q_idx = blockIdx.x / (params.h_q / B_H);
  const int warpgroup_idx = cutlass::canonical_warp_group_idx();
  const int warp_idx = cutlass::canonical_warp_idx_sync();
  const int idx_in_warpgroup = threadIdx.x % 128;

  // Define shared tensors
  extern __shared__ char wksp_buf[];
  SharedMemoryPlan& plan = *reinterpret_cast<SharedMemoryPlan*>(wksp_buf);
  Tensor sQ = make_tensor(make_smem_ptr(plan.q_o.q.data()), SmemLayoutQ{});
  Tensor sS0 = make_tensor(make_smem_ptr(plan.s[1].data()), SmemLayoutS{});
  Tensor sS1 = make_tensor(make_smem_ptr(plan.s[0].data()), SmemLayoutS{});

  if (warp_idx == 0 && elect_one_sync()) {
    // Prefetch TMA descriptors
    cute::prefetch_tma_descriptor(tma_params.tma_Q.get_tma_descriptor());

    // Initialize barriers
    plan.bar_q.init(1);
    CUTE_UNROLL
    for (int i = 0; i < 2; ++i) {
      plan.bar_k0_free[i].init(128);
      plan.bar_k0_ready[i].init(128);
      plan.bar_k1_free[i].init(128);
      plan.bar_k1_ready[i].init(128);
    }
    plan.bar_is_kv_valid_ready.init(16);
    fence_barrier_init();
  }

  __syncthreads();

  const int common_length =
      __ldg(params.topk_length + (IS_TAIL ? s_q_idx / 8 : s_q_idx));
  const int seq_len = __ldg(params.seq_lens);
  const int position = seq_len - params.query_tokens + s_q_idx;
  const int compressed = (position + 1) / 128;
  const int swa = min(position + 1, 128);
  const int topk_length = IS_TAIL ? compressed + swa : common_length;
  const int begin_topk_block = IS_TAIL ? common_length / 64 : 0;
  const int num_topk_blocks = (topk_length + 63) / 64;

  if (warpgroup_idx == 0 || warpgroup_idx == 1) {
    cutlass::arch::warpgroup_reg_alloc<216>();

    if (warp_idx == 0 && elect_one_sync()) {
      // Load Q
      Tensor gQ = flat_divide(
          tma_params.tma_Q.get_tma_tensor(tma_params.shape_Q)(_, _, s_q_idx),
          Tile<Int<B_H>, Int<D_Q>>{})(_, _, q_h_idx, _0{});
      launch_tma_copy(tma_params.tma_Q, gQ, sQ, plan.bar_q,
                      TMA::CacheHintSm90::EVICT_FIRST);
      plan.bar_q.arrive_and_expect_tx(B_H * D_Q * sizeof(bf16));
    }

    float rM[2] = {
        MAX_INIT_VAL,
        MAX_INIT_VAL};  // Meaning: the `max_logits` used for O / rL calculation
    float rL[2] = {0.0f, 0.0f};
    Tensor rO = partition_fragment_C(TiledMMA_PV_LocalP{},
                                     Shape<Int<B_H>, Int<D_V / 2>>{});
    Tensor rP =
        partition_fragment_C(TiledMMA_QK{}, Shape<Int<B_H>, Int<B_TOPK>>{});
    Tensor rS = make_tensor<bf16>(partition_shape_A(
        TiledMMA_PV_LocalP{}, Shape<Int<B_H>, Int<B_TOPK>>{}));
    cute::fill(rO, 0.0f);
    if constexpr (IS_TAIL) {
      CUTE_UNROLL
      for (int i = 0; i < 2; ++i) {
        int head = get_AorC_row_idx(i, idx_in_warpgroup);
        if (head < 8) {
          int row = s_q_idx * 8 + head;
          rM[i] = params.raw_m[row];
          rL[i] =
              params
                  .raw_l[(row * 2 + warpgroup_idx) * 4 + idx_in_warpgroup % 4];
        }
      }
      CUTE_UNROLL
      for (int i = 0; i < size(rO); i += 2) {
        int head = get_AorC_row_idx(i % 4 >= 2, idx_in_warpgroup);
        if (head < 8) {
          int col = get_AorC_col_idx(i, idx_in_warpgroup);
          float2 value = *reinterpret_cast<const float2*>(
              params.raw_o + (s_q_idx * 8 + head) * 512 + warpgroup_idx * 256 +
              col);
          rO(i) = value.x;
          rO(i + 1) = value.y;
        }
      }
    }

    // Wait for Q
    plan.bar_q.wait(0);

    bool cur_bar_wait_phase = 0;

    struct Warpgroup0 {};
    struct Warpgroup1 {};

    auto qkt_gemm_one_tile = [&](auto warpgroup_idx, int tile_idx,
                                 bool clear_accum) {
      constexpr bool IS_WG1 =
          std::is_same_v<decltype(warpgroup_idx), Warpgroup1>;
      TiledMMA tiled_mma_QK = TiledMMA_QK{};
      Tensor sQ_tile =
          flat_divide(sQ, Tile<Int<B_H>, Int<64>>{})(_, _, _0{}, tile_idx);
      Tensor sK_tile = make_tensor(
          make_smem_ptr(plan.k[(int)IS_WG1].data() + tile_idx * B_TOPK * 64),
          SmemLayoutKTiles<1>{});
      gemm_ss(clear_accum, tiled_mma_QK, sQ_tile, sK_tile, rP,
              idx_in_warpgroup);
    };

    auto mask_rP = [&](auto warpgroup_idx) {
      constexpr bool IS_WG1 =
          std::is_same_v<decltype(warpgroup_idx), Warpgroup1>;
      plan.bar_is_kv_valid_ready.wait(cur_bar_wait_phase);
      CUTE_UNROLL
      for (int row_idx = 0; row_idx < 2; ++row_idx) {
        CUTE_UNROLL
        for (int i = row_idx * 2; i < size(rP); i += 4) {
          int col = 8 * (i / 4) + (idx_in_warpgroup % 4) * 2;
          if (!plan.is_kv_valid[IS_WG1][col]) rP(i) = -INFINITY;
          if (!plan.is_kv_valid[IS_WG1][col + 1]) rP(i + 1) = -INFINITY;
        }
      }
    };

    auto online_softmax_and_rescale_o = [&](auto warpgroup_idx) {
      plan.bar_is_kv_valid_ready.wait(cur_bar_wait_phase);
      constexpr bool IS_WG1 =
          std::is_same_v<decltype(warpgroup_idx), Warpgroup1>;
      const float scale = params.sm_scale_div_log2;
      float r_sM[2];
      if constexpr (IS_WG1) {
        *(float2*)r_sM = plan.sM[idx_in_warpgroup / 4];
      }
      float new_maxs[2];
      CUTE_UNROLL
      for (int row_idx = 0; row_idx < 2; ++row_idx) {
        // Get rowwise max
        float cur_max = -INFINITY;
        CUTE_UNROLL
        for (int i = row_idx * 2; i < size(rP); i += 4) {
          cur_max = max(cur_max, max(rP(i), rP(i + 1)));
        }
        cur_max = max(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 1));
        cur_max = max(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 2));
        cur_max *= scale;

        // Get new max and scale
        // For WG1, old_max comes from sM (written by WG0); for WG0, old_max
        // comes from rM (read by WG0 from sM in the last round)
        new_maxs[row_idx] = max(IS_WG1 ? r_sM[row_idx] : rM[row_idx], cur_max);

        // Scale O
        float scale_for_o = exp2f(rM[row_idx] - new_maxs[row_idx]);
        CUTE_UNROLL
        for (int i = row_idx * 2; i < size(rO); i += 4) {
          rO(i) *= scale_for_o;
          rO(i + 1) *= scale_for_o;
        }

        // Get rS
        float cur_sum = 0;
        CUTE_UNROLL
        for (int i = row_idx * 2; i < size(rP); i += 4) {
          rP(i) = exp2f(rP(i) * scale - new_maxs[row_idx]);
          rP(i + 1) = exp2f(rP(i + 1) * scale - new_maxs[row_idx]);
          rS(i) = (bf16)rP(i);
          rS(i + 1) = (bf16)rP(i + 1);
          cur_sum += rP(i) + rP(i + 1);
        }
        rL[row_idx] = rL[row_idx] * scale_for_o + cur_sum;
      }
      __syncwarp();
      if (idx_in_warpgroup % 4 == 0) {
        plan.sM[idx_in_warpgroup / 4] = *(float2*)new_maxs;
      }
      rM[0] = new_maxs[0];
      rM[1] = new_maxs[1];
    };

    auto reduce_L = [&]() {
      // Reduce L
      // For example, thread 0 reduces with thread 1, 2, and 3, as well as
      // thread 128, 129, 130, and 131
      rL[0] += __shfl_xor_sync(0xffffffff, rL[0], 1);
      rL[0] += __shfl_xor_sync(0xffffffff, rL[0], 2);
      rL[1] += __shfl_xor_sync(0xffffffff, rL[1], 1);
      rL[1] += __shfl_xor_sync(0xffffffff, rL[1], 2);
      if (idx_in_warpgroup % 4 == 0) plan.sL[threadIdx.x / 4] = *(float2*)(rL);
      NamedBarrier::arrive_and_wait(256, NamedBarriers::sL_ready);
      float2 peer_L = plan.sL[(threadIdx.x / 4) ^ 32];
      rL[0] += peer_L.x;
      rL[1] += peer_L.y;
    };

    auto store_O = [&]() {
      warpgroup_wait<0>();
      if constexpr (!IS_TAIL) {
        CUTE_UNROLL
        for (int row_idx = 0; row_idx < 2; ++row_idx) {
          int row = get_AorC_row_idx(row_idx, idx_in_warpgroup);
          int global_row = s_q_idx * 64 + row;
          params.raw_l[(global_row * 2 + warpgroup_idx) * 4 +
                       idx_in_warpgroup % 4] = rL[row_idx];
          if (warpgroup_idx == 1 && idx_in_warpgroup % 4 == 0)
            params.raw_m[global_row] = rM[row_idx];
        }
        CUTE_UNROLL
        for (int i = 0; i < size(rO); i += 2) {
          int row = get_AorC_row_idx(i % 4 >= 2, idx_in_warpgroup);
          int col = get_AorC_col_idx(i, idx_in_warpgroup);
          float* dst = params.raw_o + (s_q_idx * 64 + row) * 512 +
                       warpgroup_idx * 256 + col;
          *reinterpret_cast<float2*>(dst) = make_float2(rO(i), rO(i + 1));
        }
      } else {
        float factors[2];
        CUTE_UNROLL
        for (int i = 0; i < 2; ++i) {
          int head = get_AorC_row_idx(i, idx_in_warpgroup);
          float sink = head < 8
                           ? __fmul_rn(params.attn_sink[head], CUDART_L2E_F)
                           : -INFINITY;
          factors[i] = __fdiv_rn(1.0f, rL[i] + exp2f(sink - rM[i]));
          if (rL[i] == 0.0f) factors[i] = 0.0f;
        }
        CUTE_UNROLL
        for (int i = 0; i < size(rO); i += 2) {
          int row = get_AorC_row_idx(i % 4 >= 2, idx_in_warpgroup);
          if (row < 8) {
            int col = get_AorC_col_idx(i, idx_in_warpgroup);
            bf16* dst = params.out + s_q_idx * params.stride_out_s_q +
                        row * 512 + warpgroup_idx * 256 + col;
            bf16 a = bf16(rO(i) * factors[i % 4 >= 2]);
            bf16 b = bf16(rO(i + 1) * factors[i % 4 >= 2]);
            uint32_t bits = uint32_t(a.raw()) | (uint32_t(b.raw()) << 16);
            *reinterpret_cast<uint32_t*>(dst) = bits;
          }
        }
      }
    };

    if (warpgroup_idx == 0) {
      // Warpgroup 0

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

      auto scale_rS = [&](float scales[2]) {
        CUTE_UNROLL
        for (int row = 0; row < 2; ++row) {
          CUTE_UNROLL
          for (int i = row * 2; i < size(rP); i += 4) {
            rS(i) = (bf16)(rP(i) * scales[row]);
            rS(i + 1) = (bf16)(rP(i + 1) * scales[row]);
          }
        }
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
      for (int block_idx = begin_topk_block; block_idx < num_topk_blocks;
           block_idx += 2) {
        Tensor sV0l = make_tensor(make_smem_ptr(plan.k[0].data()),
                                  SmemLayoutKTilesTransposed<4>{});
        Tensor sV1l = make_tensor(make_smem_ptr(plan.k[1].data()),
                                  SmemLayoutKTilesTransposed<4>{});

        if (block_idx == begin_topk_block) {
          // NOTE: We put this code here to avoid register spilling
          pipelined_wait_and_qkt_gemm_l();
          pipelined_wait_and_qkt_gemm_r();
          warpgroup_wait<0>();
        }

        // Online softmax, inform WG1
        mask_rP(Warpgroup0{});

        online_softmax_and_rescale_o(Warpgroup0{});
        NamedBarrier::arrive(256, NamedBarriers::wg0_bunch_0_ready);

        // Issue rO0 += rS0 @ sV0l
        gemm_rs(false, TiledMMA_PV_LocalP{}, rS, sV0l, rO, idx_in_warpgroup);
        warpgroup_commit_batch();

        // Mark V0L as free
        warpgroup_wait<0>();
        plan.bar_k0_free[0].arrive();

        // Wait for new sM, scale rS, save, inform WG1
        NamedBarrier::arrive_and_wait(256, NamedBarriers::wg1_bunch_0_ready);
        float new_rM[2], scale_factors[2];
        *(float2*)new_rM = plan.sM[idx_in_warpgroup / 4];
        CUTE_UNROLL
        for (int i = 0; i < 2; ++i) {
          scale_factors[i] = exp2f(rM[i] - new_rM[i]);
          rM[i] = new_rM[i];
        }
        scale_rS(scale_factors);
        save_rS_to_sS(rS, sS0, idx_in_warpgroup);
        fence_view_async_shared();
        NamedBarrier::arrive(256, NamedBarriers::wg0_s0_ready);

        // Wait for sS1
        NamedBarrier::arrive_and_wait(256, NamedBarriers::wg1_s1_ready);

        // Rescale rO0, Issue rO0 += sS1 @ sV1L
        rescale_rO(scale_factors);
        gemm_ss(false, TiledMMA_PV_RemoteP{}, sS1, sV1l, rO, idx_in_warpgroup);
        warpgroup_commit_batch();

        cur_bar_wait_phase ^= 1;

        if (block_idx + 2 < num_topk_blocks) {
          // Launch the next QK^T GEMM
          pipelined_wait_and_qkt_gemm_l();

          // Mark V1L as free
          warpgroup_wait<1>();
          plan.bar_k1_free[0].arrive();
          pipelined_wait_and_qkt_gemm_r();

          // Wait for rP0 = sQ @ sK0
          warpgroup_wait<0>();
        } else {
          // Mark V1L as free
          warpgroup_wait<0>();
          plan.bar_k1_free[0].arrive();
        }
      }

      if constexpr (IS_TAIL) reduce_L();
      store_O();
    } else {
      // Warpgroup 1

      auto pipelined_wait_and_qkt_gemm = [&]() __attribute__((always_inline)) {
        plan.bar_k1_ready[1].wait(cur_bar_wait_phase);
        qkt_gemm_one_tile(Warpgroup1{}, 4, true);
        qkt_gemm_one_tile(Warpgroup1{}, 5, false);
        qkt_gemm_one_tile(Warpgroup1{}, 6, false);
        qkt_gemm_one_tile(Warpgroup1{}, 7, false);
        if constexpr (D_QK == 576) {
          qkt_gemm_one_tile(Warpgroup1{}, 8, false);
        }
        plan.bar_k1_ready[0].wait(cur_bar_wait_phase);
        qkt_gemm_one_tile(Warpgroup1{}, 0, false);
        qkt_gemm_one_tile(Warpgroup1{}, 1, false);
        qkt_gemm_one_tile(Warpgroup1{}, 2, false);
        qkt_gemm_one_tile(Warpgroup1{}, 3, false);
        warpgroup_commit_batch();
      };

      CUTE_NO_UNROLL
      for (int block_idx = begin_topk_block; block_idx < num_topk_blocks;
           block_idx += 2) {
        Tensor sV0r = make_tensor(make_smem_ptr(plan.k[0].data() + 64 * 256),
                                  SmemLayoutKTilesTransposed<4>{});
        Tensor sV1r = make_tensor(make_smem_ptr(plan.k[1].data() + 64 * 256),
                                  SmemLayoutKTilesTransposed<4>{});

        // Issue rP1 = sQ @ sK1, and wait
        pipelined_wait_and_qkt_gemm();
        warpgroup_wait<0>();

        mask_rP(Warpgroup1{});

        // Wait for WG0 (for sM), online softmax, Notify WG0 (sM ready)
        NamedBarrier::arrive_and_wait(256, NamedBarriers::wg0_bunch_0_ready);
        online_softmax_and_rescale_o(Warpgroup1{});
        NamedBarrier::arrive(256, NamedBarriers::wg1_bunch_0_ready);

        // Issue rO1 += rS1 @ sV1R
        gemm_rs(false, TiledMMA_PV_LocalP{}, rS, sV1r, rO, idx_in_warpgroup);
        warpgroup_commit_batch();

        // Wait for WG0 (for sS0), Issue rO1 += rS0 @ sV0R
        save_rS_to_sS(rS, sS1, idx_in_warpgroup);  // Put it here is faster
        NamedBarrier::arrive_and_wait(256, NamedBarriers::wg0_s0_ready);
        gemm_ss(false, TiledMMA_PV_RemoteP{}, sS0, sV0r, rO, idx_in_warpgroup);
        warpgroup_commit_batch();

        // Save rS1, inform WG0
        fence_view_async_shared();
        NamedBarrier::arrive(256, NamedBarriers::wg1_s1_ready);

        // Wait for GEMM, and inform that sV1R is free
        warpgroup_wait<1>();
        plan.bar_k1_free[1].arrive();

        // Wait for GEMM, and inform that sV0R is free
        warpgroup_wait<0>();
        plan.bar_k0_free[1].arrive();

        cur_bar_wait_phase ^= 1;
      }

      if constexpr (IS_TAIL) reduce_L();
      store_O();
    }
  } else {
    // Producer warpgroup
    cutlass::arch::warpgroup_reg_dealloc<72>();

    constexpr int GROUP_SIZE = 8, NUM_GROUPS = 128 / GROUP_SIZE;
    constexpr int NUM_ROWS_PER_GROUP = B_TOPK / NUM_GROUPS;
    int idx_in_group = idx_in_warpgroup % GROUP_SIZE;
    int group_idx = idx_in_warpgroup / GROUP_SIZE;

    bf16* my_sKV_base =
        &(make_tensor(make_smem_ptr(plan.k[0].data()), SmemLayoutKTiles<1>{})(
            group_idx, idx_in_group * 8));
    bf16* my_gKV_base = params.kv + idx_in_group * 8;

    int64_t token_indices[2][NUM_ROWS_PER_GROUP];
    bool is_token_valid[2][NUM_ROWS_PER_GROUP];
    auto load_token_indices = [&](int block_idx) {
      CUTE_UNROLL
      for (int buf_idx = 0; buf_idx < 2; ++buf_idx) {
        CUTE_UNROLL
        for (int local_row = 0; local_row < NUM_ROWS_PER_GROUP; ++local_row) {
          int offs = (block_idx + buf_idx) * B_TOPK + local_row * NUM_GROUPS +
                     group_idx;
          int t = offs;
          if constexpr (IS_TAIL) {
            int gather_start = seq_len - __ldg(params.gather_lens);
            t = offs < compressed ? offs
                                  : params.compressed_width + position - swa +
                                        1 + offs - compressed - gather_start;
          }
          token_indices[buf_idx][local_row] =
              t * (int64_t)params
                      .stride_kv_s_kv;  // We mult it with params.stride_kv_s_kv
                                        // here since it's faster
          bool is_cur_token_valid = t >= 0 && t < params.s_kv;
          is_cur_token_valid &= offs < topk_length;
          is_token_valid[buf_idx][local_row] = is_cur_token_valid;
        }
      }
    };

    int64_t cache_policy = createpolicy_evict_last();
    auto copy_tiles = [&](int block_idx, int buf_idx, int tile_start,
                          int tile_end) {
      // Copy some K/V tiles from global memory to shared memory
      // A tile has a shape of 64 (B_TOPK) x 64
      // `buf_idx` is the index of the shared memory buffer, 0 or 1
      // `tile_idx` is the index of the tile to load, from 0 to D_K/64-1 = 8
      CUTE_UNROLL
      for (int local_row = 0; local_row < NUM_ROWS_PER_GROUP; ++local_row) {
        int64_t token_index = token_indices[buf_idx][local_row];
        CUTE_UNROLL
        for (int tile_idx = tile_start; tile_idx < tile_end; ++tile_idx) {
          cp_async_cacheglobal_l2_prefetch_256B(
              my_gKV_base + token_index + tile_idx * 64,
              my_sKV_base + (buf_idx * B_TOPK * D_K + tile_idx * (B_TOPK * 64) +
                             local_row * NUM_GROUPS * 64),
              is_token_valid[buf_idx][local_row], cache_policy);
        }
      }
    };

    auto commit_to_mbar = [&](transac_bar_t& bar) {
      cutlass::arch::cpasync_barrier_arrive_noinc((uint64_t*)(&bar));
    };

    int cur_bar_wait_phase = 1;

    CUTE_NO_UNROLL
    for (int block_idx = begin_topk_block; block_idx < num_topk_blocks;
         block_idx += 2) {
      load_token_indices(block_idx);

      // V0L
      plan.bar_k0_free[0].wait(cur_bar_wait_phase);
      copy_tiles(block_idx + 0, 0, 0, 4);
      commit_to_mbar(plan.bar_k0_ready[0]);

      // V1R
      plan.bar_k1_free[1].wait(cur_bar_wait_phase);
      copy_tiles(block_idx + 1, 1, 4, D_K / 64);
      commit_to_mbar(plan.bar_k1_ready[1]);

      // V0R
      plan.bar_k0_free[1].wait(cur_bar_wait_phase);
      copy_tiles(block_idx + 0, 0, 4, D_K / 64);
      commit_to_mbar(plan.bar_k0_ready[1]);

      // V1L
      plan.bar_k1_free[0].wait(cur_bar_wait_phase);
      copy_tiles(block_idx + 1, 1, 0, 4);
      commit_to_mbar(plan.bar_k1_ready[0]);

      // Valid mask
      // NOTE: V1R's finish implies maskings of the last round have finished
      if (idx_in_group == 0) {
        CUTE_UNROLL
        for (int buf_idx = 0; buf_idx < 2; ++buf_idx) {
          CUTE_UNROLL
          for (int local_row = 0; local_row < NUM_ROWS_PER_GROUP; ++local_row)
            plan.is_kv_valid[buf_idx][local_row * NUM_GROUPS + group_idx] =
                is_token_valid[buf_idx][local_row];
        }
        plan.bar_is_kv_valid_ready.arrive();
      }

      cur_bar_wait_phase ^= 1;
    }
  }

#else
  if (cute::thread0()) {
    CUTE_INVALID_CONTROL_PATH("This kernel only supports sm90");
  }
#endif
}

template <typename Kernel, typename TMAParams>
__global__ void __launch_bounds__(Kernel::NUM_THREADS, 1, 1)
    sparse_attn_fwd_kernel(__grid_constant__ const Params params,
                           __grid_constant__ const TMAParams tma_params) {
  Kernel::devfunc(params, tma_params);
}

template <bool IS_TAIL>
void Kernel<IS_TAIL>::run(const Params& params) {
  KU_ASSERT(params.h_kv == 1);
  KU_ASSERT(params.topk % (2 * B_TOPK) == 0);  // Avoid boundary checks.
  KU_ASSERT(params.topk > 0);
  KU_ASSERT(params.h_q % B_H == 0);

  auto shape_Q = make_shape(params.h_q, params.d_qk, params.s_q);
  auto tma_Q = cute::make_tma_copy(
      SM90_TMA_LOAD{},
      make_tensor(make_gmem_ptr((bf16*)params.q),
                  make_layout(shape_Q, make_stride(params.stride_q_h_q, _1{},
                                                   params.stride_q_s_q))),
      SmemLayoutQ{});

  CUtensorMap tensor_map_O{};
  TmaParams<decltype(shape_Q), decltype(tma_Q)> tma_params = {shape_Q, tma_Q,
                                                              tensor_map_O};
  auto kernel = &sparse_attn_fwd_kernel<Kernel<IS_TAIL>, decltype(tma_params)>;

  constexpr size_t smem_size = sizeof(SharedMemoryPlan);
  KU_CUDA_CHECK(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

  cutlass::ClusterLaunchParams launch_params = {
      dim3((params.h_q / B_H) * params.s_q, 1,
           1),  // NOTE: We put s_q on the first dim since it can be larger than
                // 65536 (the maximum size of griddim.y and griddim.z)
      dim3(NUM_THREADS, 1, 1), dim3(1, 1, 1), smem_size, params.stream};
  cutlass::launch_kernel_on_cluster(launch_params, (void*)kernel, params,
                                    tma_params);
  KU_CHECK_KERNEL_LAUNCH();
}

}  // namespace vllm::c128
