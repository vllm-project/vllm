// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Body of one persistent epoch (included twice by persistent_kernel.cuh: the
// first epoch is straight-line code so the hot loops are not nested in an outer
// loop).
if (warp_idx == 9 && cute::elect_one_sync()) {
  if (epoch)
    for (uint32_t i = 0; i < BarIdx::total; ++i) mbar_inval(bar_words + i);
  for (uint32_t i = 0; i < BarIdx::total; ++i)
    reinterpret_cast<volatile uint64_t*>(bar_words)[i] = 0;
  full_q.init(1);
  empty_q.init(kMathThreads + 32);  // math threads + the UMMA warp
  for (uint32_t i = 0; i < kKVStages; ++i) {
    full_kv[i].init(kFullKvArrivals);
    empty_kv[i].init(kMathThreads);
  }
  for (uint32_t i = 0; i < BarIdx::kTmemSlots; ++i) {
    full_tmem[i].init(1);
    empty_tmem[i].init(kMathWgThreads);
  }
  full_lin->init(32);
  // Every reader lane arrives on claim_empty.
  for (uint32_t i = 0; i < 2; ++i) {
    claim_full[i].init(32);
    claim_empty[i].init(kClaimConsumers * 32);
  }
  sel_done.init(kMathWarps);  // one lane per math warp
  for (uint32_t i = 0; i < 2 * kMathWarps; ++i) {
    h_full[i].init(32);
    h_empty[i].init(32);
  }
  cutlass::arch::fence_barrier_init();
  unsafe_rows = 0;
}
__syncwarp();
if (warp_idx == 10) cute::TMEM::Allocator1Sm().allocate(512, &tmem_ptr);
__syncthreads();

if (warp_idx == 8) {
  // -------- claimer + Q/weights producer + drain of WG1 rows --------
  cutlass::arch::warpgroup_reg_dealloc<kSpecRegs>();
  // n_blk: blocks this CTA processed (all-short blocks excluded).
  uint32_t n_blk = 0, drain_interval = 0;
  for (uint32_t k = 0;; ++k) {
    if (lane == 0) {
      empty_q.wait((n_blk & 1) ^ 1);
    }
    __syncwarp();
    // Every lane waits: lanes 0-7 write the entry.
    claim_empty[k & 1].wait(((k >> 1) & 1) ^ 1);
    uint32_t b = 0;
    if (lane == 0)
      b = atomicOr(&pause_req, 0u) ? kPause : atomicAdd(counter, 1u);
    b = __shfl_sync(kFullMask, b, 0);
    ClaimEntry& e = entries[k & 1];
    if (b >= nblocks) {
      if (lane == 0) {
        const uint32_t kind = b == kPause ? kPause : kTerm;
        e.vblock = kind;
        epoch_kind = kind;
      }
      __syncwarp();
      claim_full[k & 1].arrive();  // every writer lane arrives
      break;
    }
    // Blocks are claimed heaviest (last rows, longest context) first.
    const uint32_t vblock = nblocks - 1 - b, row_base = vblock * kBlockRows;
    uint32_t st = 0, en = 0;
    if (lane < kBlockRows) {
      const uint32_t r = row_base + lane;
      if (r < rows) {
        st = min(starts_g[r], keys);
        en = max(min(ends_g[r], keys), st);
      }
      e.starts[lane] = st;
      e.ends[lane] = en;
    }
    const bool all_short =
        __ballot_sync(kFullMask, lane < kBlockRows && en - st > topk) == 0;
    const bool gated =
        __ballot_sync(kFullMask,
                      lane < kBlockRows && en - st > sample_target) != 0;
    // KV range: rows past the end are clamped to the last valid row (DeepGEMM
    // scheduler).
    const uint32_t src = min(row_base + (lane & 7), rows - 1) - row_base;
    uint32_t lo = __shfl_sync(kFullMask, st, src),
             hi = __shfl_sync(kFullMask, en, src);
#pragma unroll
    for (int d = 4; d >= 1; d >>= 1) {
      lo = min(lo, __shfl_xor_sync(kFullMask, lo, d));
      hi = max(hi, __shfl_xor_sync(kFullMask, hi, d));
    }
    const uint32_t kv_base = lo / 4 * 4,
                   splits = (hi - kv_base + kTileKeys - 1) / kTileKeys;
    // CTA-private slot pair: block n of this CTA uses slot 2 * cta + (n & 1);
    // at most two blocks (one in selection, one claimed ahead) are in flight,
    // so no claim atomics.
    const uint32_t slot = kSlotsPerCta * blockIdx.x + (n_blk & 1);
    if (!all_short && lane < kBlockRows) row_counts[row_base + lane] = 0;
    if (lane == 0) {
      e.vblock = vblock;
      e.all_short = all_short;
      e.kv_base = kv_base;
      e.splits = splits;
      e.samples = gated ? sample_tiles : 0u;
      e.slot = slot;
    }
    __syncwarp();
    claim_full[k & 1].arrive();  // every writer lane arrives
    if (all_short) continue;
    if (lane == 0) {
      dg::tma_copy<kHeadDim, kBlockRows * kNumHeads, 0>(&tq, &full_q, sm.q, 0,
                                                        row_base * kNumHeads);
      dg::tma_copy<kNumHeads, kBlockRows, 0>(&tw, &full_q, sm.weights, 0,
                                             row_base);
      full_q.arrive_and_expect_tx(kQBytes + kWBytes);
    }
    __syncwarp();
    ++n_blk;
    const int64_t row_delta = slot_row_delta(slot, row_base);
    for (uint32_t base = 0; base < splits; base += Interval, ++drain_interval) {
      for (uint32_t p = kMathWarps / 2; p < kMathWarps; ++p)
        drain_handoff(hdata, h_full, h_empty, p, drain_interval, row_base + 4,
                      row_counts, base + Interval >= splits,
                      [&](uint32_t row, uint32_t slot2, uint32_t record,
                          uint32_t batch_base, uint32_t source) {
                        store_ring_record(vals, indices, row, row_delta, slot2,
                                          record, kv_base, batch_base,
                                          source % kMathWgThreads);
                      });
    }
    cutlass::arch::NamedBarrier(kDrainSyncThreads, kBarDrain).arrive();
  }
} else if (warp_idx == 9) {
  // -------- KV producer: sample tiles then scan tiles, one ring --------
  cutlass::arch::warpgroup_reg_dealloc<kSpecRegs>();
  RingPipeline<kKVStages> kv_pipe;
  uint32_t n_blk = 0;
  for (uint32_t k = 0;; ++k) {
    ClaimHeader claim;
    if (!consume_claim(k, claim_full, claim_empty, entries, claim)) break;
    if (claim.all_short) continue;
    const uint32_t kv_base = claim.kv_base, splits = claim.splits,
                   samples = claim.samples;
    {
      // The whole warp stages the strided sample scales into sf_kv[stage]
      // (prefetched one tile ahead in registers); lane 0 issues the KV TMA.
      if (lane == 0 && n_blk > 0) sel_done.wait((n_blk - 1) & 1);
      __syncwarp();
      const uint32_t center = sample_stride / 2;
      // Per-key scales of one sample tile, kScalesPerLane per lane.
      constexpr uint32_t kScalesPerLane = kTileKeys / 32;
      float pf[kScalesPerLane];
#pragma unroll
      for (uint32_t j = 0; j < kScalesPerLane; ++j) {
        const uint32_t g = j * 32 + lane;
        pf[j] = g < sample_groups ? __ldg(scales + g * sample_stride + center)
                                  : 0.0f;
      }
#pragma unroll 1
      for (uint32_t t = 0; t < samples; ++t) {
        // Only lane 0's ring state is maintained across blocks (the scan loop
        // runs on lane 0).
        CUTE_TIE_DECL(kv_pipe.advance(), s0, ph);
        const uint32_t s = __shfl_sync(kFullMask, s0, 0);
        // Every writing lane acquires and releases the stage itself.
        empty_kv[s].wait(__shfl_sync(kFullMask, ph, 0) ^ 1);
#pragma unroll
        for (uint32_t j = 0; j < kScalesPerLane; ++j)
          sm.sf_kv[s][j * 32 + lane] = pf[j];
        // Generic writes before later TMA writes of this stage.
        cutlass::arch::fence_view_async_shared();
        if (lane == 0) {
          full_kv[s].expect_transaction(kKVBytes);
          dg::tma_copy<kHeadDim, kTileKeys, 0, cutlass::float_e4m3_t, true>(
              &tks, &full_kv[s], sm.kv[s], 0, t * kTileKeys, center);
        }
        full_kv[s].arrive();
#pragma unroll
        for (uint32_t j = 0; j < kScalesPerLane; ++j) {
          const uint32_t g = (t + 1) * kTileKeys + j * 32 + lane;
          pf[j] = g < sample_groups ? __ldg(scales + g * sample_stride + center)
                                    : 0.0f;
        }
      }
    }
    if (lane == 0) {
#pragma unroll 1
      for (uint32_t t = 0; t < splits; ++t) {
        CUTE_TIE_DECL(kv_pipe.advance(), s, ph);
        empty_kv[s].wait(ph ^ 1);
        const uint32_t off = kv_base + t * kTileKeys;
        dg::tma_copy<kHeadDim, kTileKeys, 0>(&tk, &full_kv[s], sm.kv[s], 0,
                                             off);
        dg::tma_copy<kTileKeys, 1, 0>(&ts, &full_kv[s], sm.sf_kv[s], off, 0);
        mbar_arrive_expect_tx(&full_kv[s], kKVBytes + kScaleBytes,
                              kFullKvArrivals);
      }
    }
    __syncwarp();
    ++n_blk;
  }
} else if (warp_idx == 10) {
  // -------- UMMA issuer --------
  cutlass::arch::warpgroup_reg_dealloc<kSpecRegs>();
  RingPipeline<1> q_pipe;
  RingPipeline<kKVStages> kv_pipe;
  RingPipeline<BarIdx::kTmemSlots> tmem_pipe;
  auto instr_desc =
      cute::UMMA::make_instr_desc<cutlass::float_e4m3_t, cutlass::float_e4m3_t,
                                  float, 128, kUmmaN, cute::UMMA::Major::K,
                                  cute::UMMA::Major::K>();
  auto runtime_instr_desc = cute::UMMA::make_runtime_instr_desc(instr_desc);
  for (uint32_t k = 0;; ++k) {
    ClaimHeader claim;
    if (!consume_claim(k, claim_full, claim_empty, entries, claim)) break;
    if (claim.all_short) continue;
    const uint32_t splits = claim.splits, samples = claim.samples;
    CUTE_TIE_DECL(q_pipe.advance(), q_stage, q_phase);
    (void)q_stage;
    full_q.wait(q_phase);
    full_lin->wait(q_phase);  // this block's linear-term rows (warp 11)
    ptx::tcgen05_after_thread_sync();
    const uint32_t total = samples + splits;
#pragma unroll 1
    for (uint32_t t = 0; t < total; ++t) {
      CUTE_TIE_DECL(kv_pipe.advance(), s, ph);
      full_kv[s].wait(ph);
      if (cute::elect_one_sync()) {
        // Per KV half, one N=144 MMA per math warpgroup into its own TMEM slot
        // (ring of 3; WG0 takes even, WG1 odd positions of the issue sequence).
#pragma unroll
        for (uint32_t i = 0; i < 2; ++i) {
#pragma unroll
          for (uint32_t g = 0; g < 2; ++g) {
            CUTE_TIE_DECL(tmem_pipe.advance(), ts_idx, ts_ph);
            const uint32_t tmem_addr = ts_idx * kSlotCols;
            empty_tmem[ts_idx].wait(ts_ph ^ 1);
            ptx::tcgen05_after_thread_sync();
            cutlass::float_e4m3_t* b_base =
                g == 0 ? reinterpret_cast<cutlass::float_e4m3_t*>(sm.lin_lo)
                       : sm.q + 4 * kNumHeads * kHeadDim;  // WG1's rows
#pragma unroll
            for (uint32_t kk = 0; kk < 4; ++kk) {
              auto a_desc =
                  mma::sm100::make_umma_desc<cute::UMMA::Major::K, 0, 128, 128>(
                      sm.kv[s], i * 128, kk * 32);
              auto b_desc =
                  mma::sm100::make_umma_desc<cute::UMMA::Major::K, 0, 128, 128>(
                      b_base, 0, kk * 32);
              ptx::SM100_MMA_F8F6F4_SS::fma(a_desc, b_desc, tmem_addr, kk,
                                            runtime_instr_desc);
            }
            ptx::umma_arrive_no_elect(full_tmem[ts_idx]);
          }
        }
      }
      __syncwarp();
    }
    empty_q.arrive();
  }
} else if (warp_idx == 11) {
  // -------- linear-term rows + drain of WG0 rows --------
  cutlass::arch::warpgroup_reg_dealloc<kSpecRegs>();
  uint32_t drain_interval = 0, n_lin = 0;
  for (uint32_t k = 0;; ++k) {
    ClaimHeader claim;
    if (!consume_claim(k, claim_full, claim_empty, entries, claim)) break;
    if (claim.all_short) continue;
    const uint32_t vblock = claim.vblock, kv_base = claim.kv_base,
                   splits = claim.splits, slot = claim.slot;
    // Q of block k lands while the math warps still select block k-1, so this
    // is off the critical path except for a CTA's first block.
    full_q.wait(n_lin & 1);
    ++n_lin;
    compute_lin_rows(sm, lane);
    // Generic smem writes -> UMMA (async proxy).
    cutlass::arch::fence_view_async_shared();
    full_lin->arrive();  // every lane arrives (count 32)
    const uint32_t row_base = vblock * kBlockRows;
    const int64_t row_delta = slot_row_delta(slot, row_base);
    for (uint32_t base = 0; base < splits; base += Interval, ++drain_interval) {
      for (uint32_t p = 0; p < kMathWarps / 2; ++p)
        drain_handoff(hdata, h_full, h_empty, p, drain_interval, row_base,
                      row_counts, base + Interval >= splits,
                      [&](uint32_t row, uint32_t slot2, uint32_t record,
                          uint32_t batch_base, uint32_t source) {
                        store_ring_record(vals, indices, row, row_delta, slot2,
                                          record, kv_base, batch_base,
                                          source % kMathWgThreads);
                      });
    }
    cutlass::arch::NamedBarrier(kDrainSyncThreads, kBarDrain).arrive();
  }
} else {
  // -------- math warps 0-7 --------
  cutlass::arch::warpgroup_reg_alloc<kMathRegs>();
  const uint32_t math_tid = threadIdx.x;
  const uint32_t query_base = (warp_idx / 4) * kWgRows;  // 4 warps per WG
  RingPipeline<1> q_pipe;
  RingPipeline<kKVStages> kv_pipe;
  RingPipeline<BarIdx::kTmemSlots> tmem_pipe;
  // Each warpgroup consumes every second TMEM slot position (WG1 starts at 1).
  constexpr uint32_t kTmemStep = 2;
  if (warp_idx >= 4) tmem_pipe.advance(1);
  const uint32_t heads_col = warp_idx < 4 ? 16u : 0u;
  const uint32_t lin_col = warp_idx < 4 ? 0u : 128u;
  uint32_t produce_interval = 0;
  for (uint32_t k = 0;; ++k) {
    claim_full[k & 1].wait((k >> 1) & 1);
    const ClaimEntry& e = entries[k & 1];
    const uint32_t vblock = e.vblock;
    if (vblock >= kPause) break;
    const uint32_t row_base = vblock * kBlockRows;
    if (e.all_short) {
      for (uint32_t r = 0; r < kBlockRows && row_base + r < rows; ++r)
        write_all_keys(out + (uint64_t)(row_base + r) * topk, topk,
                       e.ends[r] - e.starts[r], math_tid, kMathThreads);
      __syncwarp();
      claim_empty[k & 1].arrive();
      continue;
    }
    // ---- block setup: Q, weights, sample histogram ----
    {
      CUTE_TIE_DECL(q_pipe.advance(), q_stage, q_phase);
      (void)q_stage;
      full_q.wait(q_phase);
      full_lin->wait(q_phase);
    }
    static_assert(sizeof(SampleShared) % 16 == 0);
    for (uint32_t j = math_tid; j < sizeof(SampleShared) / 16;
         j += kMathThreads)
      reinterpret_cast<uint4*>(sample_smem.hist)[j] = make_uint4(0, 0, 0, 0);
    float weights[kWgRows][kNumHeads];
    float row_s[kWgRows];  // pow2 row scale of the linear term
    uint32_t seq_k_start[kWgRows], seq_k_end[kWgRows];
#pragma unroll
    for (uint32_t i = 0; i < kWgRows; ++i) {
      seq_k_start[i] = e.starts[query_base + i];
      seq_k_end[i] = e.ends[query_base + i];
      // Weights in units of 1 / s_r (exact: s_r is a power of two).
      row_s[i] = sm.lin_scale[query_base + i];
      const float inv_s = 1.0f / row_s[i];
#pragma unroll
      for (uint32_t j = 0; j < kNumHeads; j += 4) {
        const float4 w4 = *reinterpret_cast<const float4*>(
            sm.weights + (query_base + i) * kNumHeads + j);
        weights[i][j] = w4.x * 0.5f * inv_s;
        weights[i][j + 1] = w4.y * 0.5f * inv_s;
        weights[i][j + 2] = w4.z * 0.5f * inv_s;
        weights[i][j + 3] = w4.w * 0.5f * inv_s;
      }
    }
    {
      // Weights that are subnormal, or become subnormal or infinite once
      // scaled, take the exact recovery path.
      const float w = sm.weights[warp_idx * kNumHeads + lane];  // lane = head
      // 2^-125: the smallest magnitude whose half is still a normal float.
      constexpr float kMinScalable = 0x1p-125f;
      bool bad_w = w != 0.0f && fabsf(w) < kMinScalable;
      const float wr = w * 0.5f * (1.0f / sm.lin_scale[warp_idx]);
      bad_w |= (wr != 0.0f && fabsf(wr) < kMinScalable) || isinf(wr);
      const uint32_t bad = __ballot_sync(kFullMask, bad_w);
      if (lane == 0 && bad) atomicOr(&unsafe_rows, 1u << warp_idx);
    }
    math_wg_sync();
    // ---- sample tiles ----
#pragma unroll 1
    for (uint32_t t = 0; t < e.samples; ++t) {
      CUTE_TIE_DECL(kv_pipe.advance(), kv_stage, kv_phase);
      full_kv[kv_stage].wait(kv_phase);
#pragma unroll
      for (uint32_t kv_half = 0; kv_half < 2; ++kv_half) {
        const uint32_t kv_lane =
            (math_tid % kMathWgThreads) + kv_half * kMathWgThreads;
        const uint32_t group = t * kTileKeys + kv_lane;
        const uint32_t kv_offset = group * sample_stride + sample_stride / 2;
        const bool in_view = group < sample_groups;
        // 0 outside the view (staged by the KV producer).
        const float scale_kv = ptx::ld_shared(sm.sf_kv[kv_stage] + kv_lane);
        CUTE_TIE_DECL(tmem_pipe.advance(kTmemStep), tmem_stage, tmem_phase);
        full_tmem[tmem_stage].wait(tmem_phase);
        ptx::tcgen05_after_thread_sync();
        if (kv_half == 1) {
          cutlass::arch::fence_view_async_shared();
          empty_kv[kv_stage].arrive();
        }
        // One query row (32 TMEM columns) at a time keeps the sample loop's
        // live registers low enough to carry the scan weights through it.
#pragma unroll
        for (uint32_t i = 0; i < kWgRows; ++i) {
          float acc[kNumHeads];
          float lin[4];
          ptx::tmem_load_32dp32b<32>(
              tmem_stage * kSlotCols + heads_col + i * kNumHeads,
              reinterpret_cast<uint32_t*>(acc));
          ptx::tmem_load_32dp32b<4>(tmem_stage * kSlotCols + lin_col + i * 4,
                                    reinterpret_cast<uint32_t*>(lin));
          cutlass::arch::fence_view_async_tmem_load();
          if (i == kWgRows - 1) {
            ptx::tcgen05_before_thread_sync();
            empty_tmem[tmem_stage].arrive();
          }
          const float bq = reduce_heads_with_lin(acc, weights[i], lin) *
                           (row_s[i] * scale_kv);
          const bool valid = in_view && kv_offset >= seq_k_start[i] &&
                             kv_offset < seq_k_end[i] && isfinite(bq);
          if (valid) {
            const uint32_t bin = half_bin<5>(bq);
            atomicAdd(sample_smem.hist[query_base + i] + padded_bin(bin), 1u);
          }
        }
      }
    }
    math_wg_sync();
    finish_sample_histogram(sample_smem.hist[warp_idx], gate_s + warp_idx,
                            e.ends[warp_idx] - e.starts[warp_idx],
                            sample_target, lane);
    math_wg_sync();
    // The block index is re-read from the claim entry (volatile) instead of
    // kept live across the sample loop.
    const uint32_t scan_row_base = volatile_load(e.vblock) * kBlockRows;

    // ---- scan tiles ----
    const uint32_t kv_base = e.kv_base, splits = e.splits;
    const int64_t cand_row_delta = slot_row_delta(e.slot, scan_row_base);
    float gate_score[kWgRows];
#pragma unroll
    for (uint32_t i = 0; i < kWgRows; ++i) {
      gate_score[i] = gate_s[query_base + i];
      if ((volatile_load(unsafe_rows) >> (query_base + i)) & 1u) {
        // Unsafe weights: no key passes (+inf gate), and a count above the
        // candidate cap sends the block to the exact rescan.
        gate_score[i] = CUDART_INF_F;
        if (math_tid % kMathWgThreads == 0)
          atomicAdd(row_counts + scan_row_base + query_base + i,
                    static_cast<int>(kCandidatesPerRow) + 1);
      }
      // Compare S * scale_kv against gate / row_s (row_s is a power of two ->
      // exact).
      gate_score[i] *= 1.0f / row_s[i];
    }
    uint32_t lane_counts[kWgRows] = {};
    uint32_t flush_base = 0;
#pragma unroll 1
    for (uint32_t kv_split_idx = 0; kv_split_idx < splits; ++kv_split_idx) {
      if ((kv_split_idx & (Interval - 1)) == 0) {
        h_empty[(produce_interval & 1) * kMathWarps + warp_idx].wait(
            ((produce_interval >> 1) & 1) ^ 1);
        __syncwarp();
      }
      CUTE_TIE_DECL(kv_pipe.advance(), kv_stage, kv_phase);
      full_kv[kv_stage].wait(kv_phase);
#pragma unroll
      for (uint32_t kv_half = 0; kv_half < 2; ++kv_half) {
        const uint32_t kv_lane =
            (math_tid % kMathWgThreads) + kv_half * kMathWgThreads;
        const uint32_t kv_offset = kv_base + kv_split_idx * kTileKeys + kv_lane;
        const float scale_kv = ptx::ld_shared(sm.sf_kv[kv_stage] + kv_lane);
        CUTE_TIE_DECL(tmem_pipe.advance(kTmemStep), tmem_stage, tmem_phase);
        full_tmem[tmem_stage].wait(tmem_phase);
        ptx::tcgen05_after_thread_sync();
        if (kv_half == 1) {
          cutlass::arch::fence_view_async_shared();
          empty_kv[kv_stage].arrive();
        }
        float accum_pair[2][kNumHeads];
        float lin_pair[2][4];
        float
            bp4[kWgRows];  // row scores of the slot, in units of the row scale
#pragma unroll
        for (uint32_t pair = 0; pair < kWgRows; pair += 2) {
          const uint32_t slot_base = tmem_stage * kSlotCols;
          ptx::tmem_load_32dp32b<2 * kNumHeads>(
              slot_base + heads_col + pair * kNumHeads,
              reinterpret_cast<uint32_t*>(accum_pair));
          ptx::tmem_load_32dp32b<8>(slot_base + lin_col + pair * 4,
                                    reinterpret_cast<uint32_t*>(lin_pair));
          cutlass::arch::fence_view_async_tmem_load();
          if (pair + 2 >= kWgRows) {
            ptx::tcgen05_before_thread_sync();
            empty_tmem[tmem_stage].arrive();
          }
          // The linear term (sum_t term_t . k, in units of the row scale; terms
          // 2, 3 in units of s * 2^-8) seeds the sums.
#pragma unroll
          for (uint32_t r = 0; r < 2; ++r)
            bp4[pair + r] = reduce_heads_with_lin(
                                accum_pair[r], weights[pair + r], lin_pair[r]) *
                            scale_kv;
        }
        // One warp vote per slot: most keys fail every gate, so the common path
        // is 4 compares + 1 vote + 1 uniform branch.
        const bool any_pass =
            (bp4[0] > gate_score[0]) | (bp4[1] > gate_score[1]) |
            (bp4[2] > gate_score[2]) | (bp4[3] > gate_score[3]);
        if (__any_sync(kFullMask, any_pass)) [[unlikely]] {
#pragma unroll
          for (uint32_t i = 0; i < kWgRows; ++i) {
            const float bp = bp4[i];
            // Gate first; range and finiteness only for passing keys.
            if (bp > gate_score[i]) {
              const float bq = bp * row_s[i];
              const uint32_t row = scan_row_base + query_base + i;
              if (kv_offset >= seq_k_start[i] && kv_offset < seq_k_end[i] &&
                  isfinite(bq)) {
                const uint32_t count = lane_counts[i];
                if (count < kRingSlots) {
                  const uint32_t code = candidate_fp24_code(bq);
                  hdata
                      .records[produce_interval & 1][warp_idx][i][count][lane] =
                      (code << 8) | ((kv_split_idx - flush_base) * 2 + kv_half);
                  lane_counts[i] = count + 1;
                } else {
                  const int slot = atomicAdd(row_counts + row, 1);
                  if (slot < static_cast<int>(kCandidatesPerRow)) {
                    const uint64_t off =
                        candidate_offset(row, cand_row_delta, slot);
                    store_candidate(vals + off, indices + off, bq, kv_offset);
                  }
                }
              }
            }
          }
        }
      }
      if (((kv_split_idx + 1) & (Interval - 1)) == 0 &&
          kv_split_idx + 1 != splits) {
#pragma unroll
        for (uint32_t i = 0; i < kWgRows; ++i) {
          hdata.num_records[produce_interval & 1][warp_idx][i][lane] =
              lane_counts[i];
          lane_counts[i] = 0;
        }
        if (lane == 0) hdata.bases[produce_interval & 1][warp_idx] = flush_base;
        __syncwarp();
        h_full[(produce_interval & 1) * kMathWarps + warp_idx].arrive();
        ++produce_interval;
        flush_base = kv_split_idx + 1;
      }
    }
    {
      // Final interval: every math warp stores its own ring records (8 warps in
      // parallel instead of 2 drain warps after the scan); the drain warps only
      // cycle the barriers.
      const uint32_t bank = produce_interval & 1,
                     rbase = scan_row_base + query_base;
      static_assert(sizeof(hdata.records[0][0]) ==
                        kWgRows * kRingSlots * 32 * sizeof(uint32_t),
                    "ring record layout");
      self_drain_last(&hdata.records[bank][warp_idx][0][0][0], lane_counts[0],
                      lane_counts[1], lane_counts[2], lane_counts[3],
                      row_counts, rbase, kv_base + (math_tid % kMathWgThreads),
                      flush_base, cand_row_delta, vals, indices);
#pragma unroll
      for (uint32_t row = 0; row < kWgRows; ++row) lane_counts[row] = 0;
      __syncwarp();
      h_full[bank * kMathWarps + warp_idx].arrive();
      ++produce_interval;
    }
    cutlass::arch::fence_view_async_shared();
    empty_q.arrive();
    // Drains of this block done.
    cutlass::arch::NamedBarrier(kDrainSyncThreads, kBarDrain).sync();
    // ---- selection ----
    // Re-read as well (not kept live across the scan).
    const uint32_t sel_vblock = volatile_load(e.vblock),
                   sel_row_base = sel_vblock * kBlockRows;
    __threadfence_block();
    cutlass::arch::NamedBarrier(kMathThreads, kBarSelect).sync();
    const ClaimEntry& sel_entry = entries[k & 1];
    if (math_tid == 0) {
      uint32_t recover = 0, short_rows = 0;
      row_offset[0] = 0;
      for (uint32_t r = 0; r < kBlockRows; ++r) {
        const bool is_short = sel_entry.ends[r] - sel_entry.starts[r] <= topk;
        short_rows |= uint32_t(is_short) << r;
        const int c = __ldcg(row_counts + sel_row_base + r);
        recover |= !is_short && (c < int(topk) || c > int(kCandidatesPerRow));
        // Rows start 16-byte aligned for select_staged_row's vector loads.
        row_offset[r + 1] =
            row_offset[r] + (is_short ? 0 : (uint32_t(max(c, 0)) + 3u) & ~3u);
      }
      // 32-bit byte arithmetic (sizeof is a 64-bit size_t).
      constexpr uint32_t kWordBytes = sizeof(uint32_t);
      // Stage all 8 rows' candidates at once (1), in two rounds of 4 (2), or,
      // when they exceed the scratch, rescan exactly.
      const uint32_t half = max(row_offset[kWgRows],
                                row_offset[kBlockRows] - row_offset[kWgRows]);
      const uint32_t staged =
          recover ? 0u
                  : ((kBlockRows * selection::kBins + row_offset[kBlockRows]) *
                                 kWordBytes <=
                             phase_bytes
                         ? 1u
                         : ((kWgRows * selection::kBins + half) * kWordBytes <=
                                    phase_bytes
                                ? 2u
                                : 0u));
      if (staged == 0) recover = 1;
      if (recover) {
        defer_next[sel_vblock] = int(defer_head);
        defer_head = sel_vblock;
        atomicExch(&pause_req, 1u);
      }
      sel_recover = recover;
      sel_staged = staged;
      sel_short = short_rows;
      unsafe_rows = 0;
    }
    cutlass::arch::NamedBarrier(kMathThreads, kBarSelect).sync();
    const uint32_t recover = sel_recover, short_rows = sel_short;
    const auto write_short = [&](uint32_t r, uint32_t t, uint32_t stride) {
      if (r >= rows) return;
      write_all_keys(
          out + (uint64_t)r * topk, topk,
          sel_entry.ends[r - sel_row_base] - sel_entry.starts[r - sel_row_base],
          t, stride);
    };
    if (!recover) {
      const int64_t row_delta = slot_row_delta(sel_entry.slot, sel_row_base);
      const uint32_t per_round = sel_staged == 1 ? kBlockRows : kWgRows;
      for (uint32_t round = 0; round < kBlockRows / per_round; ++round) {
        if (warp_idx < per_round) {
          const uint32_t local = round * per_round + warp_idx,
                         first = round * per_round;
          uint32_t* hist = reinterpret_cast<uint32_t*>(smem_buffer + kSelBase) +
                           warp_idx * selection::kBins;
          uint32_t* codes =
              reinterpret_cast<uint32_t*>(smem_buffer + kSelBase) +
              per_round * selection::kBins +
              (row_offset[local] - row_offset[first]);
          const uint32_t r = sel_row_base + local;
          const uint64_t base = candidate_offset(r, row_delta, 0);
          if ((short_rows >> local) & 1u)
            write_short(r, lane, 32);
          else {
            const uint32_t cnt = __ldcg(row_counts + r);
            int* o = out + (uint64_t)r * topk;
            if (!selection::select_staged_row(codes, cnt, vals + base,
                                              indices + base, o, topk, hist))
              selection::select_row(codes, cnt, vals + base, indices + base, o,
                                    topk, hist);
          }
        }
        cutlass::arch::NamedBarrier(kMathThreads, kBarSelect).sync();
      }
    }
    // Every row is final here: the last selection round ends with a barrier,
    // and a recovered block skipped the rounds right after the barrier above.
    // Short rows were written relative already.
    if (!recover)
      make_relative(out, topk, rows, sel_row_base, starts_g, math_tid,
                    kMathThreads, [&](uint32_t local) {
                      return sel_entry.ends[local] - sel_entry.starts[local] <=
                             topk;
                    });
    __syncwarp();
    if (lane == 0) sel_done.arrive();
    claim_empty[k & 1].arrive();
  }
}

// -------- exact rescan of the blocks queued by the selection --------
__syncthreads();
if (warp_idx == 0) cute::TMEM::Allocator1Sm().free(0, 512);
if (defer_head != kTerm) {
  // The rescan expects the 224 / 56 register split of the math warps.
  __syncthreads();
  for (uint32_t cur = defer_head; cur != kTerm;) {
    const uint32_t vblock = cur, row_base = vblock * kBlockRows;
    if (threadIdx.x < kBlockRows) {
      const uint32_t r = row_base + threadIdx.x;
      row_bounds[0][threadIdx.x] = r < rows ? min(starts_g[r], keys) : 0u;
      row_bounds[1][threadIdx.x] =
          r < rows ? max(min(ends_g[r], keys), row_bounds[0][threadIdx.x]) : 0u;
    }
    __syncthreads();
    const uint32_t* const starts = row_bounds[0] - row_base;
    const uint32_t* const ends = row_bounds[1] - row_base;
    // No block is in flight during the rescan: use the CTA's first slot.
    const int64_t row_delta =
        slot_row_delta(kSlotsPerCta * blockIdx.x, row_base);
    const RescanBuffers block_buffers{
        rescan.retained + row_delta * kRetainedPerRow,
        rescan.retained_indices + row_delta * kRetainedPerRow};
    rescan::rescan_block(dense::Scheduler(vblock, rows, keys, starts, ends),
                         block_buffers, out, topk, tq, tk, ts, tw);
    __syncthreads();
    if (threadIdx.x == 0)
      dense::invalidate_barriers(
          *reinterpret_cast<dense::SharedStorage*>(smem_buffer));
    __syncthreads();
    // The rescan writes absolute indices for every row of the block.
    make_relative(out, topk, rows, row_base, starts_g, threadIdx.x, kNumThreads,
                  [](uint32_t) { return false; });
    cur = uint32_t(__ldcg(defer_next + vblock));
    __syncthreads();
  }
  // The loop ran at least once and ended with a barrier.
  if (threadIdx.x == 0) {
    defer_head = kTerm;
    pause_req = 0;
  }
}  // deferred blocks
more = epoch_kind != kTerm;
__syncthreads();
