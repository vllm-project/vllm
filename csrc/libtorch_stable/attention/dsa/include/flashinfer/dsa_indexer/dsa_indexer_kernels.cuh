// LiteTopK DSA scoring kernel V3 = "hybrid":
//   * scoring loop of DeepGEMM 2.5 (commit 891d57b, the vLLM-pinned version):
//     per-q-block weights held in REGISTERS, per-row 32-element TMEM loads
//     with early UMMA release, tight tcgen05 fencing -- the generation that
//     makes the official kernel fast at large Q;
//   * scheduling of our V1: NON-persistent KV-split. blockIdx.x = q-block,
//     blockIdx.y = KV split window. This is what keeps all 148 SMs busy on the
//     tiny-Q chunks vLLM actually produces at long context (its 512MB logits
//     budget shrinks Q to ~128 at S=1M, where a persistent grid would idle
//     116/148 SMs);
//   * LiteTopK sparse epilogue (batched-vote emit, strided gate reload,
//     warp-local candidate queues) and the spare-warp threshold-refresh daemon
//     (V1 semantics: one q-block per CTA, fixed rows).
//
// Ragged Q (vLLM chunks) handled by forcing an empty KV range on padded rows.

#pragma once

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>

#include <deep_gemm/common/cute_tie.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace dsa_litetopk {

using namespace deep_gemm;

#define DSA_WARP_QUEUE_CAP 64
#define DSA_REFRESH_STRIDE 16
#define DSA_GATE_STRIDE 16

#define DSA_ST_CAND_VAL(dst, v) __stcs(&(dst), (v))
#define DSA_ST_CAND_IDX(dst, v) __stcs(&(dst), (v))

template <uint32_t kNumHeads, uint32_t kHeadDim, uint32_t BLOCK_Q,
          uint32_t BLOCK_KV, uint32_t kNumQStages, uint32_t kNumKVStages,
          uint32_t kNumSMs, uint32_t kNumSpecializedThreads,
          uint32_t kNumMathThreads,
          uint32_t kNumMathWarpGroups = kNumMathThreads / 128>
CUTLASS_GLOBAL __launch_bounds__(
    kNumSpecializedThreads + kNumMathThreads,
    1) void sm100_dsa_litetopk(const uint32_t seq_len,
                               const uint32_t seq_len_kv,
                               uint32_t* cu_seq_len_k_start,
                               uint32_t* cu_seq_len_k_end,
                               const float* __restrict__ origin,  // [seq_len]
                               const float* __restrict__ inv_delta,  // [seq_len]
                               int32_t* __restrict__ th_bucket,  // [seq_len]
                               int32_t* __restrict__ bcount,     // [seq_len,
                                                                 // num_buckets]
                               const uint32_t num_buckets, const uint32_t topk,
                               const uint32_t refresh_every,
                               const uint32_t num_kv_splits,
                               const uint32_t
                                   probe_group,  // compacted-space group size
                                                 // (pstp-1)*64; 0 = no probe
                                                 // compaction (identity map)
                               const uint64_t
                                   probe_magic,  // ceil(2^42/probe_group):
                                                 // exact div via mul-shift
                               const uint32_t
                                   probe_add_max,  // npage*64 cap for the map
                               float* __restrict__ cand_val,    // [seq_len,
                                                                // cand_cap]
                               int32_t* __restrict__ cand_idx,  // [seq_len,
                                                                // cand_cap]
                               int32_t* __restrict__ cand_cnt,  // [seq_len]
                               const uint32_t cand_cap,
                               const __grid_constant__ cute::TmaDescriptor
                                   tensor_map_q,
                               const __grid_constant__ cute::TmaDescriptor
                                   tensor_map_kv,
                               const __grid_constant__ cute::TmaDescriptor
                                   tensor_map_kv_scales,
                               const __grid_constant__ cute::TmaDescriptor
                                   tensor_map_weights) {
  const auto num_q_blocks = math::ceil_div(seq_len, BLOCK_Q);

  using Barrier = cutlass::arch::ClusterTransactionBarrier;

  const auto warp_idx = cutlass::canonical_warp_idx_sync();
  const auto warpgroup_idx = warp_idx / 4;
  const auto lane_idx = ptx::get_lane_idx();
  constexpr uint32_t kSpecWarpStart = kNumMathWarpGroups * 4;
  constexpr uint32_t kNumMathWarps = kNumMathThreads / 32;

  DG_STATIC_ASSERT(kNumSpecializedThreads == 128 and kNumMathThreads % 128 == 0,
                   "Invalid threads");

  if (warp_idx == kSpecWarpStart) {
    cute::prefetch_tma_descriptor(&tensor_map_q);
    cute::prefetch_tma_descriptor(&tensor_map_kv);
    cute::prefetch_tma_descriptor(&tensor_map_kv_scales);
    cute::prefetch_tma_descriptor(&tensor_map_weights);
  }

  static constexpr uint32_t SMEM_Q_SIZE_PER_STAGE =
      BLOCK_Q * kNumHeads * kHeadDim * sizeof(__nv_fp8_e4m3);
  static constexpr uint32_t SMEM_WEIGHT_SIZE_PER_STAGE =
      BLOCK_Q * kNumHeads * sizeof(float);
  static constexpr uint32_t SMEM_KV_SIZE_PER_STAGE =
      BLOCK_KV * kHeadDim * sizeof(__nv_fp8_e4m3);
  static constexpr uint32_t SMEM_KV_SCALE_SIZE_PER_STAGE =
      BLOCK_KV * sizeof(float);
  static constexpr uint32_t ALIGNED_SMEM_KV_SCALE_SIZE_PER_STAGE =
      math::constexpr_align(SMEM_KV_SCALE_SIZE_PER_STAGE, 512u);

  extern __shared__ __align__(512) uint8_t smem_buffer[];
  DG_STATIC_ASSERT(SMEM_Q_SIZE_PER_STAGE % 512 == 0, "Unaligned TMA swizzling");
  DG_STATIC_ASSERT(SMEM_WEIGHT_SIZE_PER_STAGE % 512 == 0,
                   "Unaligned TMA swizzling");
  DG_STATIC_ASSERT(SMEM_KV_SIZE_PER_STAGE % 512 == 0,
                   "Unaligned TMA swizzling");

  constexpr uint32_t kNumTmemCols = BLOCK_Q * kNumHeads * kNumMathWarpGroups;
  DG_STATIC_ASSERT(kNumTmemCols <= 512, "Too many tensor memory");

  auto smem_q = utils::PatternVisitor([&](const uint32_t& i) {
    return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer +
                                            SMEM_Q_SIZE_PER_STAGE * i);
  });
  auto smem_weights = utils::PatternVisitor([&](const uint32_t& i) {
    return reinterpret_cast<float*>(smem_buffer +
                                    SMEM_Q_SIZE_PER_STAGE * kNumQStages +
                                    SMEM_WEIGHT_SIZE_PER_STAGE * i);
  });
  auto smem_kv = utils::PatternVisitor([&](const uint32_t& i) {
    return reinterpret_cast<__nv_fp8_e4m3*>(
        smem_buffer + (SMEM_Q_SIZE_PER_STAGE * kNumQStages +
                       SMEM_WEIGHT_SIZE_PER_STAGE * kNumQStages +
                       SMEM_KV_SIZE_PER_STAGE * i));
  });
  auto smem_kv_scales = utils::PatternVisitor([&](const uint32_t& i) {
    return reinterpret_cast<float*>(smem_buffer +
                                    SMEM_Q_SIZE_PER_STAGE * kNumQStages +
                                    SMEM_WEIGHT_SIZE_PER_STAGE * kNumQStages +
                                    SMEM_KV_SIZE_PER_STAGE * kNumKVStages +
                                    ALIGNED_SMEM_KV_SCALE_SIZE_PER_STAGE * i);
  });

  auto barrier_ptr = reinterpret_cast<Barrier*>(smem_kv_scales[kNumKVStages]);
  auto full_q_barriers =
      utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + i; });
  auto empty_q_barriers = utils::PatternVisitor(
      [&](const uint32_t& i) { return barrier_ptr + (kNumQStages + i); });
  auto full_kv_barriers = utils::PatternVisitor(
      [&](const uint32_t& i) { return barrier_ptr + (kNumQStages * 2 + i); });
  auto empty_kv_barriers = utils::PatternVisitor([&](const uint32_t& i) {
    return barrier_ptr + (kNumQStages * 2 + kNumKVStages + i);
  });
  auto full_umma_barriers = utils::PatternVisitor([&](const uint32_t& i) {
    return barrier_ptr + (kNumQStages * 2 + kNumKVStages * 2 + i);
  });
  auto empty_umma_barriers = utils::PatternVisitor([&](const uint32_t& i) {
    return barrier_ptr +
           (kNumQStages * 2 + kNumKVStages * 2 + kNumMathWarpGroups + i);
  });

  auto tmem_ptr_in_smem =
      reinterpret_cast<uint32_t*>(barrier_ptr + kNumQStages * 2 +
                                  kNumKVStages * 2 + kNumMathWarpGroups * 2);
  auto scan_done_flag = reinterpret_cast<volatile int*>(tmem_ptr_in_smem + 1);
  auto kv_progress_ptr = reinterpret_cast<volatile int*>(tmem_ptr_in_smem + 2);
  auto warpq_count = reinterpret_cast<int32_t*>(tmem_ptr_in_smem + 4);
  auto warpq_val =
      reinterpret_cast<float*>(warpq_count + kNumMathWarps * BLOCK_Q);
  auto warpq_idx = reinterpret_cast<int32_t*>(
      warpq_val + kNumMathWarps * BLOCK_Q * DSA_WARP_QUEUE_CAP);
  // Per-CTA refresh histogram (BLOCK_Q x num_buckets). When this CTA is the
  // ONLY scanner of its rows (num_kv_splits == 1, i.e. all large-Q shapes),
  // the per-candidate histogram feed goes to smem instead of RED.GLOBAL:
  // cheaper atomic, no 64-bit address math, no L2 pressure. The daemon then
  // reads global bcount (seed counts) + this smem part. Counts and totals
  // are identical to the global path, so thresholds and recall are
  // unchanged; a racing read can only UNDERcount -> looser gate -> safe.
  auto smem_hist = reinterpret_cast<int32_t*>(
      warpq_idx + kNumMathWarps * BLOCK_Q * DSA_WARP_QUEUE_CAP);

  DG_STATIC_ASSERT(
      kNumSpecializedThreads % 128 == 0 and kNumSpecializedThreads >= 64,
      "Invalid threads");
  if (warp_idx == kSpecWarpStart and cute::elect_one_sync()) {
#pragma unroll
    for (uint32_t i = 0; i < kNumQStages; ++i) {
      full_q_barriers[i]->init(1);
      empty_q_barriers[i]->init(kNumMathThreads + 32);
    }
#pragma unroll
    for (uint32_t i = 0; i < kNumKVStages; ++i) {
      full_kv_barriers[i]->init(1);
      empty_kv_barriers[i]->init(kNumMathThreads);
    }
    *scan_done_flag = 0;
    *kv_progress_ptr = 0;
    cutlass::arch::fence_barrier_init();
  }
  if (warp_idx == kSpecWarpStart + 1) {
    if (cute::elect_one_sync()) {
#pragma unroll
      for (uint32_t i = 0; i < kNumMathWarpGroups; ++i) {
        full_umma_barriers[i]->init(1);
        empty_umma_barriers[i]->init(128);
      }
      cutlass::arch::fence_barrier_init();
    }
    cute::TMEM::Allocator1Sm().allocate(kNumTmemCols, tmem_ptr_in_smem);
  }
  const bool hist_in_smem = (num_kv_splits == 1) && (refresh_every > 0) &&
                            (refresh_every != 0x7fffffff);
  if (hist_in_smem) {
    for (uint32_t idx = threadIdx.x; idx < BLOCK_Q * num_buckets;
         idx += blockDim.x)
      smem_hist[idx] = 0;
  }
  __syncthreads();

  constexpr uint32_t kNumSpecializedRegisters = 40;
  constexpr uint32_t kNumMathRegisters = 232;

  // V1 KV-split scheduling: blockIdx.x = q-block (one per CTA), blockIdx.y =
  // contiguous KV sub-window. Split boundaries are BLOCK_KV-aligned.
  const uint32_t block_q_idx = blockIdx.x;
  const uint32_t kv_split = blockIdx.y;
  uint32_t seq_k_start[BLOCK_Q], seq_k_end[BLOCK_Q];
  const auto load_schedule =
      [&](const uint32_t block_q_idx) -> cute::tuple<uint32_t, uint32_t> {
    uint32_t start = cute::numeric_limits<uint32_t>::max();
    uint32_t end = cute::numeric_limits<uint32_t>::min();

#pragma unroll
    for (uint32_t i = 0; i < BLOCK_Q; ++i) {
      const auto q_idx = min(block_q_idx * BLOCK_Q + i, seq_len - 1);
      seq_k_start[i] = cu_seq_len_k_start[q_idx];
      seq_k_end[i] = cu_seq_len_k_end[q_idx];
      if (block_q_idx * BLOCK_Q + i >= seq_len) {
        // Padded row of a ragged final q-block: empty, aggregation-neutral.
        seq_k_start[i] = seq_len_kv;
        seq_k_end[i] = 0;
      }
      start = min(start, min(seq_k_start[i], seq_len_kv));
      end = max(end, min(seq_k_end[i], seq_len_kv));
    }
    const uint32_t total_blocks = math::ceil_div(seq_len_kv, BLOCK_KV);
    const uint32_t blocks_per_split =
        math::ceil_div(total_blocks, num_kv_splits);
    const uint32_t split_lo = kv_split * blocks_per_split * BLOCK_KV;
    const uint32_t split_hi =
        min((kv_split + 1) * blocks_per_split * BLOCK_KV, seq_len_kv);
    start = start / 4 * 4;  // TMA alignment for SF KV
    if (start < split_lo) start = split_lo;
    if (end > split_hi) end = split_hi;
    const uint32_t nkv =
        (end > start) ? math::ceil_div(end - start, BLOCK_KV) : 0;
    return {start, nkv};
  };

  const auto get_kv_pipeline =
      [&](const uint32_t& kv_block_idx) -> cute::tuple<uint32_t, uint32_t> {
    return {kv_block_idx % kNumKVStages, (kv_block_idx / kNumKVStages) & 1};
  };

  constexpr uint32_t UMMA_M = 128;
  constexpr uint32_t UMMA_K = 32 / sizeof(cutlass::float_e4m3_t);
  constexpr uint32_t UMMA_N = BLOCK_Q * kNumHeads;

  if (warp_idx == kSpecWarpStart) {
    cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();

    if (cute::elect_one_sync()) {
      if (block_q_idx < num_q_blocks) {
        // Q + weights once for this q-block.
        tma::copy<kHeadDim, BLOCK_Q * kNumHeads, kHeadDim>(
            &tensor_map_q, full_q_barriers[0], smem_q[0], 0,
            block_q_idx * BLOCK_Q * kNumHeads);
        tma::copy<kNumHeads, BLOCK_Q, 0>(&tensor_map_weights,
                                         full_q_barriers[0], smem_weights[0], 0,
                                         block_q_idx * BLOCK_Q);
        full_q_barriers[0]->arrive_and_expect_tx(SMEM_Q_SIZE_PER_STAGE +
                                                 SMEM_WEIGHT_SIZE_PER_STAGE);

        CUTE_TIE_DECL(load_schedule(block_q_idx), kv_start, num_kv_blocks);
        for (uint32_t kv_block_idx = 0; kv_block_idx < num_kv_blocks;
             ++kv_block_idx) {
          CUTE_TIE_DECL(get_kv_pipeline(kv_block_idx), kv_stage_idx, kv_phase);
          empty_kv_barriers[kv_stage_idx]->wait(kv_phase ^ 1);

          tma::copy<kHeadDim, BLOCK_KV, kHeadDim>(
              &tensor_map_kv, full_kv_barriers[kv_stage_idx],
              smem_kv[kv_stage_idx], 0, kv_start + kv_block_idx * BLOCK_KV);
          tma::copy<BLOCK_KV, 1, 0>(&tensor_map_kv_scales,
                                    full_kv_barriers[kv_stage_idx],
                                    smem_kv_scales[kv_stage_idx],
                                    kv_start + kv_block_idx * BLOCK_KV, 0);
          full_kv_barriers[kv_stage_idx]->arrive_and_expect_tx(
              SMEM_KV_SIZE_PER_STAGE + SMEM_KV_SCALE_SIZE_PER_STAGE);
        }
      }
    }
  } else if (warp_idx == kSpecWarpStart + 1) {
    cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();

    DG_TRAP_ONLY_DEVICE_ASSERT(ptx::ld_shared(tmem_ptr_in_smem) == 0);

    auto instr_desc = cute::UMMA::make_instr_desc<
        cutlass::float_e4m3_t, cutlass::float_e4m3_t, float, UMMA_M, UMMA_N,
        cute::UMMA::Major::K, cute::UMMA::Major::K>();
    auto runtime_instr_desc = cute::UMMA::make_runtime_instr_desc(instr_desc);

    if (block_q_idx < num_q_blocks) {
      CUTE_TIE_DECL(load_schedule(block_q_idx), kv_start, num_kv_blocks);
      full_q_barriers[0]->wait(0);

      for (uint32_t kv_block_idx = 0; kv_block_idx < num_kv_blocks;
           ++kv_block_idx) {
        const uint32_t kvg = kv_block_idx;
        CUTE_TIE_DECL(get_kv_pipeline(kvg), kv_stage_idx, kv_phase);
        full_kv_barriers[kv_stage_idx]->wait(kv_phase);

        DG_STATIC_ASSERT(BLOCK_KV == kNumMathThreads, "Invalid block size");
        DG_STATIC_ASSERT(kHeadDim % UMMA_K == 0, "Invalid head dim");
#pragma unroll
        for (uint32_t i = 0; i < kNumMathWarpGroups; ++i) {
          empty_umma_barriers[i]->wait((kvg & 1) ^ 1);
          ptx::tcgen05_after_thread_sync();
#pragma unroll
          for (uint32_t k = 0; k < kHeadDim / UMMA_K; ++k) {
            auto a_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, 0,
                                                     kHeadDim, kHeadDim>(
                smem_kv[kv_stage_idx], i * UMMA_M, k * UMMA_K);
            auto b_desc =
                mma::sm100::make_umma_desc<cute::UMMA::Major::K, 0, kHeadDim,
                                           kHeadDim>(smem_q[0], 0, k * UMMA_K);
            cute::SM100_MMA_F8F6F4_SS::fma(a_desc, b_desc, i * UMMA_N, k,
                                           runtime_instr_desc);
          }
          cutlass::arch::umma_arrive(
              reinterpret_cast<uint64_t*>(full_umma_barriers[i]));
        }
      }
      empty_q_barriers[0]->arrive();
    }
  } else if (warp_idx == kSpecWarpStart + 2 or warp_idx == kSpecWarpStart + 3) {
    // Spare-warp threshold-refresh daemon (V1 semantics: fixed rows).
    // NOTE: moving this refresh into the math warps' gate-reload point
    // (tidal-style C2) measured 12-13% SLOWER at 256K/512K here: unlike
    // tidal, this kernel has no register spill (setmaxnreg 232/40) and
    // sleeping was only 0.28 cyc/issue — the daemon overlaps well, while
    // inline refresh puts bcount global-read latency on the math warps'
    // critical path (a warpgroup hiccup every GATE_STRIDE blocks).
    cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();

    const bool in_scan_refresh =
        (refresh_every > 0 && refresh_every != 0x7fffffff);
    if (in_scan_refresh && block_q_idx < num_q_blocks) {
      const uint32_t spare_id = warp_idx - (kSpecWarpStart + 2);  // 0 or 1
      const auto refresh_row = [&](const uint32_t row) {
        if (row >= seq_len) return;
        const int32_t* brow = bcount + static_cast<uint64_t>(row) * num_buckets;
        const int32_t* srow =
            smem_hist + (row - block_q_idx * BLOCK_Q) * num_buckets;
        int carry = 0;
        int found = static_cast<int>(num_buckets) - 1;
        bool done = false;
        for (uint32_t base = 0; base < num_buckets && !done; base += 32) {
          uint32_t b = base + lane_idx;
          int v = (b < num_buckets) ? brow[b] : 0;
          if (hist_in_smem && b < num_buckets) v += srow[b];
          int prefix = v;
#pragma unroll
          for (int off = 1; off < 32; off <<= 1) {
            int nsh = __shfl_up_sync(0xffffffffu, prefix, off);
            if (static_cast<int>(lane_idx) >= off) prefix += nsh;
          }
          int incl = carry + prefix;
          bool hit = (b < num_buckets) && (incl >= static_cast<int>(topk)) &&
                     (incl - v < static_cast<int>(topk));
          unsigned hm = __ballot_sync(0xffffffffu, hit);
          if (hm) {
            found = static_cast<int>(base) + (__ffs(hm) - 1);
            done = true;
          } else {
            carry += __shfl_sync(0xffffffffu, prefix, 31);
          }
        }
        if (lane_idx == 0 && found < th_bucket[row]) th_bucket[row] = found;
      };
      int last_prog = 0;
      while (true) {
        const int done = *scan_done_flag;
        const int prog = *kv_progress_ptr;
        if (prog > last_prog) {
          for (uint32_t r = spare_id, li = 0; r < BLOCK_Q; r += 2, ++li) {
            refresh_row(block_q_idx * BLOCK_Q + r);
          }
          last_prog = prog;
        } else if (done) {
          for (uint32_t r = spare_id, li = 0; r < BLOCK_Q; r += 2, ++li) {
            refresh_row(block_q_idx * BLOCK_Q + r);
          }
          break;
        } else {
          __nanosleep(256);
        }
      }
    }
  } else if (warp_idx < kSpecWarpStart) {
    cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

    const auto tmem_start = warpgroup_idx * UMMA_N;
    const auto math_thread_idx = warp_idx * 32 + lane_idx;

    auto tmem_load = [](auto num_elems_c, const uint32_t& tmem_addr,
                        float* accum) {
      constexpr int N = decltype(num_elems_c)::value;
      DG_STATIC_ASSERT(N == 32 or N == 64, "Unsupported TMEM load size");
      using Loader =
          cute::conditional_t<N == 32, cute::SM100_TMEM_LOAD_32dp32b32x,
                              cute::SM100_TMEM_LOAD_32dp32b64x>;
      [&]<size_t... Is>(cute::index_sequence<Is...>) {
        Loader::copy(tmem_addr, reinterpret_cast<uint32_t*>(accum)[Is]...);
      }(cute::make_index_sequence<N>{});
      cutlass::arch::fence_view_async_tmem_load();
    };

    // NOTE (DSA_BIT_GATE, tried + REJECTED 2026-07-11): ALU bit-pattern
    // buckets (flipped-float compare + shift bucketing) break recall
    // (0.5-96%): the aminmax span crosses octaves/zero, so bit space is
    // wildly nonuniform (radix-judgment redux) and (found+1)<<k overflows
    // u32 at large k. Column-frequency +2 INT ops also pre-decided the
    // speed. Do not revisit without a threshold-anchored, overflow-safe
    // bucket space AND a hit-frequency-only costing.
    float weights[BLOCK_Q][kNumHeads];
    float o_reg[BLOCK_Q], inv_reg[BLOCK_Q], vth_reg[BLOCK_Q];
    uint32_t kstart_reg[BLOCK_Q],
        kspan_reg[BLOCK_Q];  // unsigned range-check trick
    int gate_reg[BLOCK_Q];
    const unsigned FULL = 0xffffffffu;

    if (block_q_idx < num_q_blocks) {
      CUTE_TIE_DECL(load_schedule(block_q_idx), kv_start, num_kv_blocks);
      full_q_barriers[0]->wait(0);

// Weights into registers (once per CTA -- the 2.5-generation win).
#pragma unroll
      for (uint32_t i = 0; i < BLOCK_Q; ++i) {
#pragma unroll
        for (uint32_t j = 0; j < kNumHeads; ++j)
          weights[i][j] = ptx::ld_shared(smem_weights[0] + i * kNumHeads + j);
      }
      // Queue fill counts are warp-uniform: every lane tracks them
      // redundantly in registers (qn_reg), so the hot emit path needs no
      // smem bookkeeping and no shfl broadcast.
      // NOTE (adaptive dual-mode emit, rejected 2026-07-09): switching
      // rows to direct ballot-group scatter when the window hit count
      // is low measured 4-13% SLOWER at every real shape — the extra 5
      // registers + branch alone spill at the 168-reg cap. The queue +
      // register-count path below is the measured optimum for emit.
      int qn_reg[BLOCK_Q];
#pragma unroll
      for (uint32_t i = 0; i < BLOCK_Q; ++i) {
        const uint32_t rq = min(block_q_idx * BLOCK_Q + i, seq_len - 1);
        o_reg[i] = origin[rq];
        inv_reg[i] = inv_delta[rq];
        // GATE4 (user's final form): bucket-space FLOAT end to end.
        // bq = fmaf(scale_kv, sum', c0) -- form-identical to the sign
        // gate; gate = INT compare of bq's BITS vs edge float(g+1)
        // bits (edge >= 1 > 0, so ALL negative bq bit-patterns
        // compare below it and pass: no sign flip needed). cand_val
        // stores bq itself (affine preserves order; select runs in
        // bucket space, indices are the only output -- the exact
        // score is never reconstructed). vth_reg = c0; o_reg is
        // repurposed at consume time to hold the edge float.
        vth_reg[i] = -o_reg[i] * inv_reg[i];
        o_reg[i] = 0.0f;  // gate closed until the first consume
        gate_reg[i] = cute::numeric_limits<int32_t>::max();
        qn_reg[i] = 0;
        kstart_reg[i] = seq_k_start[i];
        kspan_reg[i] =
            seq_k_end[i] > seq_k_start[i] ? seq_k_end[i] - seq_k_start[i] : 0;
      }
// Fold -inv into the register weights: the whole ReLU-weighted
// chain then accumulates directly in bucket units. 128 FMULs
// once per qb, amortized over thousands of kv blocks.
#pragma unroll
      for (uint32_t i = 0; i < BLOCK_Q; ++i) {
#pragma unroll
        for (uint32_t j = 0; j < kNumHeads; ++j) weights[i][j] *= -inv_reg[i];
      }
      // Interior-block bounds (warp-uniform): a kv block fully inside
      // every row's [ks, ke) needs no per-element range checks.
      uint32_t rs_max = 0, re_min = 0xffffffffu;
#pragma unroll
      for (uint32_t i = 0; i < BLOCK_Q; ++i) {
        rs_max = max(rs_max, kstart_reg[i]);
        re_min = min(re_min, kstart_reg[i] + kspan_reg[i]);
      }

      // Gate PREFETCH: th_bucket lives in global and is tightened
      // concurrently by the refresh; loading it at the consume point
      // stalls the warp on the LDG->ISETP->BRA chain (NCU: ~27% of all
      // stall samples, long_scoreboard on the reload branches). Instead
      // consume the value fetched one window earlier and issue the next
      // window's load right after -- GATE_STRIDE blocks of latency
      // cover. A one-window-stale gate is recall-safe: refresh only
      // TIGHTENS th, so staleness admits extra candidates, never drops.
      int th_pf[BLOCK_Q];
#pragma unroll
      for (uint32_t i = 0; i < BLOCK_Q; ++i)
        th_pf[i] =
            __ldcg(th_bucket + min(block_q_idx * BLOCK_Q + i, seq_len - 1));

      // Drain a (warp,row) queue segment to the global candidate
      // buffer. The probe index mapping lives HERE, not on the insert
      // path: it used to run warp-wide per ballot group (~4 predicated
      // instructions dragged by a single hit); at drain time 32 lanes
      // retire DSA_WARP_QUEUE_CAP entries in parallel, so it costs
      // ~1/30th in issue slots and is semantically identical.
      const auto drain_queue = [&](const uint32_t i, const uint32_t row_q,
                                   const uint32_t queue_base, const int qn,
                                   const int base) {
        const uint64_t out_base = static_cast<uint64_t>(row_q) * cand_cap;
        for (int t = static_cast<int>(lane_idx); t < qn; t += 32) {
          const float x = warpq_val[queue_base + t];
          uint32_t kvo = static_cast<uint32_t>(warpq_idx[queue_base + t]);
          if (probe_group != 0) {
            // compacted -> original position (probe pages were
            // excluded from the workspace; the probe itself
            // seeds them); exact c/probe_group via magic mul-shift
            const uint32_t sup =
                (uint32_t)(((uint64_t)kvo * probe_magic) >> 42);
            kvo += min((sup + 1) * 64u, probe_add_max);
          }
          const int w = base + t;
          if (w < static_cast<int>(cand_cap)) {
            DSA_ST_CAND_VAL(cand_val[out_base + w], x);
            DSA_ST_CAND_IDX(cand_idx[out_base + w], static_cast<int32_t>(kvo));
          }
        }
      };

      for (uint32_t kv_block_idx = 0; kv_block_idx < num_kv_blocks;
           ++kv_block_idx) {
        const uint32_t kvg = kv_block_idx;
        CUTE_TIE_DECL(get_kv_pipeline(kvg), kv_stage_idx, kv_phase);
        full_kv_barriers[kv_stage_idx]->wait(kv_phase);

        if ((kv_block_idx % DSA_GATE_STRIDE) == 0) {
#pragma unroll
          for (uint32_t i = 0; i < BLOCK_Q; ++i) {
            const int g = th_pf[i];
            if (g != gate_reg[i]) {
              gate_reg[i] = g;
              // edge = float(g+1): exact for small ints, no
              // division. The gate compares BITS against it.
              o_reg[i] = static_cast<float>(g + 1);
            }
          }
#pragma unroll
          for (uint32_t i = 0; i < BLOCK_Q; ++i)
            th_pf[i] =
                __ldcg(th_bucket + min(block_q_idx * BLOCK_Q + i, seq_len - 1));
        }

        float scale_kv =
            ptx::ld_shared(smem_kv_scales[kv_stage_idx] + math_thread_idx);

        full_umma_barriers[warpgroup_idx]->wait(kvg & 1);
        ptx::tcgen05_after_thread_sync();

        empty_kv_barriers[kv_stage_idx]->arrive();

        const auto kv_offset =
            kv_start + kv_block_idx * BLOCK_KV + math_thread_idx;
        DG_STATIC_ASSERT(kNumHeads % 8 == 0, "Invalid head");

        uint32_t pass_bits = 0;
        float v_row[BLOCK_Q];

        // P1: row-PAIR TMEM loads (32dp32b64x): half the tcgen05.ld
        // instructions and half the fences on the governor loop; the
        // UMMA release also moves one row earlier.
        // Interior-block gate elision: one warp-uniform branch per
        // block picks a loop body WITHOUT the per-element range
        // checks (SASS: saves 4x VIADD + 4x ISETP per column) for the
        // >99% of blocks fully inside every row's [ks, ke).
        DG_STATIC_ASSERT(BLOCK_Q % 2 == 0, "row-pair loads need even BLOCK_Q");
// GATE4: column cost IDENTICAL to the sign gate (FFMA whose
// addend is c0 instead of th_x, ISETP on bits instead of the
// sign). NaN bq maps to a large positive pattern -> DROPPED
// (old FSETP semantics; recall check is the arbiter).
#define DSA_SCORE_GATE(i, RC)                                               \
  const float bq = fmaf(scale_kv, sum.x + sum.y, vth_reg[i]);               \
  v_row[i] = bq;                                                            \
  bool g = __float_as_int(bq) < __float_as_int(o_reg[i]);                   \
  if constexpr (RC) g = g and ((kv_offset - kstart_reg[i]) < kspan_reg[i]); \
  pass_bits |= g ? (1u << i) : 0u;
        const uint32_t kv_base = kv_start + kv_block_idx * BLOCK_KV;
        const bool interior =
            (kv_base >= rs_max) && (kv_base + BLOCK_KV <= re_min);
#define DSA_SCORE_ROWS(RANGE_CHECK)                                        \
  _Pragma("unroll") for (uint32_t pr = 0; pr < BLOCK_Q / 2; ++pr) {        \
    float accum2[kNumHeads * 2];                                           \
    tmem_load(cute::Int<kNumHeads * 2>{}, tmem_start + pr * 2 * kNumHeads, \
              accum2);                                                     \
    if (pr == BLOCK_Q / 2 - 1) {                                           \
      ptx::tcgen05_before_thread_sync();                                   \
      empty_umma_barriers[warpgroup_idx]->arrive();                        \
    }                                                                      \
    _Pragma("unroll") for (uint32_t k = 0; k < 2; ++k) {                   \
      const uint32_t i = pr * 2 + k;                                       \
      const float* accum = accum2 + k * kNumHeads;                         \
      auto sum_0 = make_float2(0, 0);                                      \
      auto sum_1 = make_float2(0, 0);                                      \
      const auto transform = [&](const uint32_t& j, const float2& sum) {   \
        auto a = make_float2(fmaxf(accum[j], 0), fmaxf(accum[j + 1], 0));  \
        auto b = make_float2(weights[i][j], weights[i][j + 1]);            \
        return __ffma2_rn(a, b, sum);                                      \
      };                                                                   \
      _Pragma("unroll") for (uint32_t j = 0; j < kNumHeads; j += 4) {      \
        sum_0 = transform(j, sum_0);                                       \
        sum_1 = transform(j + 2, sum_1);                                   \
      }                                                                    \
      auto sum = __fadd2_rn(sum_0, sum_1);                                 \
      DSA_SCORE_GATE(i, RANGE_CHECK)                                       \
    }                                                                      \
  }
        if (interior) {
          DSA_SCORE_ROWS(false)
        } else {
          DSA_SCORE_ROWS(true)
        }
#undef DSA_SCORE_ROWS
#undef DSA_SCORE_GATE

        // redux pruning: inside an active block, one redux.sync.or
        // gives the warp-wide union of hit rows, so the ballot (and
        // its queue bookkeeping) runs only for rows that actually
        // have hits (~1.3 of BLOCK_Q=4 at production density). The
        // cheap VOTE.ANY stays as the outer gate: redux costs more
        // than a vote and must not run on the ~40% inactive blocks.
        // Branches are warp-uniform (no divergence around collectives).
        if (__any_sync(FULL, pass_bits)) {
          const uint32_t rows_union = __reduce_or_sync(FULL, pass_bits);
#pragma unroll
          for (uint32_t i = 0; i < BLOCK_Q; ++i) {
            if (rows_union & (1u << i)) {
              const bool g = (pass_bits >> i) & 1u;
              const unsigned m = __ballot_sync(FULL, g);
              // Emit via the smem warp queue, but with the fill count
              // in registers (warp-uniform, every lane tracks it):
              // the hot path has NO smem bookkeeping, NO shfl
              // broadcast, NO syncwarp. The returning atomicAdd (L2
              // round-trip the whole warp must wait on through the
              // shfl) happens only on drain, ~once per 32 candidates.
              // (Direct per-group scatter measured 10-15% SLOWER
              // end-to-end: the round-trip per ballot group is the
              // expensive part, not the stores. NCU @256K: queue
              // bookkeeping showed up as wait/short_scoreboard/
              // branch stalls starving the UMMA pipe.)
              const uint32_t row_q = block_q_idx * BLOCK_Q + i;
              const int cnt = __popc(m);
              const uint32_t queue_base =
                  (warp_idx * BLOCK_Q + i) * DSA_WARP_QUEUE_CAP;
              int qn = qn_reg[i];
              if (qn + cnt > static_cast<int>(DSA_WARP_QUEUE_CAP)) {
                int base = 0;
                if (lane_idx == 0) base = atomicAdd(cand_cnt + row_q, qn);
                base = __shfl_sync(FULL, base, 0);
                drain_queue(i, row_q, queue_base, qn, base);
                qn = 0;
                __syncwarp(FULL);  // queue slots reusable
              }
              if (g) {
                // Lean insert: position + two STS; the probe
                // index mapping is deferred to drain_queue
                // (semantically free). The histogram feed is
                // NOT deferred: a queue-depth of count lag
                // slows the daemon's tightening and the
                // looser gate costs more than the feed saves
                // (measured +0.2..0.9ms @q8192 production).
                const float x = v_row[i];  // bucket-space value IS the payload
                const unsigned below = (1u << lane_idx) - 1u;
                const int pos = qn + __popc(m & below);
                warpq_val[queue_base + pos] = x;
                warpq_idx[queue_base + pos] = static_cast<int32_t>(kv_offset);
                if (refresh_every > 0) {
                  // one F2I off the stored bucket float --
                  // bucket-identical to the gate.
                  int braw = static_cast<int>(v_row[i]);
                  int b = braw < 0 ? 0
                                   : (braw > static_cast<int>(num_buckets) - 1
                                          ? static_cast<int>(num_buckets) - 1
                                          : braw);
                  if (hist_in_smem) {
                    atomicAdd(smem_hist + i * num_buckets + b, 1);
                    // (DEFER: daemon batch-feeds from the
                    // cand buffer; b's chain dead-codes.)
                  } else {
                    atomicAdd(
                        &bcount[static_cast<uint64_t>(row_q) * num_buckets + b],
                        1);
                  }
                }
              }
              qn_reg[i] = qn + cnt;
            }
          }
        }

        if (threadIdx.x == 0 &&
            ((kv_block_idx + 1) % DSA_REFRESH_STRIDE) == 0) {
          __threadfence_block();
          *kv_progress_ptr = static_cast<int>(kvg + 1);
        }
      }

// Flush this CTA's warp queues (counts live in qn_reg).
#pragma unroll
      for (uint32_t i = 0; i < BLOCK_Q; ++i) {
        const uint32_t row_q = block_q_idx * BLOCK_Q + i;
        const int qn = qn_reg[i];
        if (row_q < seq_len && qn > 0) {
          const uint32_t queue_base =
              (warp_idx * BLOCK_Q + i) * DSA_WARP_QUEUE_CAP;
          int base = 0;
          if (lane_idx == 0) base = atomicAdd(cand_cnt + row_q, qn);
          base = __shfl_sync(FULL, base, 0);
          drain_queue(i, row_q, queue_base, qn, base);
        }
      }

      empty_q_barriers[0]->arrive();
    }

    // Signal the refresh daemon, then free tensor memory.
    cutlass::arch::NamedBarrier(kNumMathThreads, 0).sync();
    if (threadIdx.x == 0) {
      __threadfence_block();
      *scan_done_flag = 1;
    }
    if (warp_idx == 0) cute::TMEM::Allocator1Sm().free(0, kNumTmemCols);
  }
}

}  // namespace dsa_litetopk
