// Body of one persistent epoch (included twice by persistent_kernel.cuh: the
// first epoch is straight-line code so the hot loops are not nested in an outer
// loop).
if (warp_idx == 9 && cute::elect_one_sync()) {
  if (epoch)
    for (unsigned i = 0; i < B::total; ++i) mbar_inval(bar_words + i);
  for (unsigned i = 0; i < B::total; ++i)
    reinterpret_cast<volatile uint64_t*>(bar_words)[i] = 0;
  full_q.init(1);
  empty_q.init(256 + 32);
  for (unsigned i = 0; i < KVS; ++i) {
    full_kv[i].init(kFullKvArrivals);
    empty_kv[i].init(kRel ? 1 : 256);
  }
  if constexpr (kRel)
    for (unsigned i = 0; i < 2 * KVS; ++i) sf_full[i].init(1);
  for (unsigned i = 0; i < B::NT; ++i) {
    full_tmem[i].init(1);
    empty_tmem[i].init(kLin ? 128 : 256);
  }
  if constexpr (kLin) full_lin->init(32);
  for (unsigned i = 0; i < 2; ++i) {
    claim_full[i].init(32);
    claim_empty[i].init(kClaimConsumers * 32);
  }  // every reader lane arrives
  sel_done.init(8);
  for (unsigned i = 0; i < 16; ++i) {
    h_full[i].init(32);
    h_empty[i].init(32);
  }
  cutlass::arch::fence_barrier_init();
  unsafe_s = 0;
}
__syncwarp();
if (warp_idx == 10) cute::TMEM::Allocator1Sm().allocate(512, &tmem_ptr);
__syncthreads();
if (epoch == 0) cudaGridDependencySynchronize();

if (warp_idx == 8) {
  // ---------------- claimer + Q/weights producer (+ drain of WG1 rows)
  // ----------------
  cutlass::arch::warpgroup_reg_dealloc<kSpecRegs>();
  unsigned n_q = 0, n_blk = 0, drain_epoch = 0;
  for (unsigned k = 0;; ++k) {
    if (lane == 0) {
      empty_q.wait((n_q & 1) ^ 1);
    }
    __syncwarp();
    claim_empty[k & 1].wait(((k >> 1) & 1) ^
                            1);  // every lane: lanes 0-7 write the entry
    unsigned b = 0;
    if (lane == 0)
      b = atomicOr(&pause_req, 0u) ? kPause : atomicAdd(counter, 1u);
    b = __shfl_sync(0xffffffffu, b, 0);
    ClaimEntry& e = entries[k & 1];
    if (b >= nblocks) {
      if (lane == 0) {
        const unsigned kind = b == kPause ? kPause : kTerm;
        e.vblock = kind;
        epoch_kind = kind;
      }
      __syncwarp();
      claim_full[k & 1].arrive();  // every writer lane arrives
      break;
    }
    const unsigned vblock = nblocks - 1 - b, row_base = vblock * 8;
    unsigned st = 0, en = 0;
    if (lane < 8) {
      const unsigned r = row_base + lane;
      if (r < rows) {
        st = min(starts_g[r], keys);
        en = max(min(ends_g[r], keys), st);
      }
      e.starts[lane] = st;
      e.ends[lane] = en;
    }
    const bool all_short =
        __ballot_sync(0xffffffffu, lane < 8 && en - st > topk) == 0;
    const bool gated =
        __ballot_sync(0xffffffffu, lane < 8 && en - st > sample_target) != 0;
    // KV range: rows past the end are clamped to the last valid row (DeepGEMM
    // scheduler).
    const unsigned src = min(row_base + (lane & 7), rows - 1) - row_base;
    unsigned lo = __shfl_sync(0xffffffffu, st, src),
             hi = __shfl_sync(0xffffffffu, en, src);
#pragma unroll
    for (int d = 4; d >= 1; d >>= 1) {
      lo = min(lo, __shfl_xor_sync(0xffffffffu, lo, d));
      hi = max(hi, __shfl_xor_sync(0xffffffffu, hi, d));
    }
    const unsigned kv_base = lo / 4 * 4;
    if (!all_short && lane < 8) counts[row_base + lane] = 0;
    if (lane == 0) {
      // CTA-private slot pair: block n of this CTA uses slot 2 * cta + (n & 1);
      // at most two blocks (one in selection, one claimed ahead) are in flight,
      // so no claim atomics.
      const unsigned slot = 2 * blockIdx.x + (n_blk & 1);
      e.vblock = vblock;
      e.all_short = all_short;
      e.kv_base = kv_base;
      e.splits = (hi - kv_base + 255) / 256;
      e.samples = gated ? sample_tiles : 0u;
      e.slot = slot;
    }
    __syncwarp();
    claim_full[k & 1].arrive();  // every writer lane arrives
    if (all_short) continue;
    if (lane == 0) {
      fused_mqa_dg::tma_copy<128, 256, 0>(&tq, &full_q, sm.q, 0, row_base * 32);
      fused_mqa_dg::tma_copy<32, 8, 0>(&tw, &full_q, sm.weights, 0, row_base);
      full_q.arrive_and_expect_tx(kQBytes + kWBytes);
    }
    __syncwarp();
    ++n_q;
    ++n_blk;
    if constexpr (!kSmem) {
      const unsigned splits = (hi - kv_base + 255) / 256;
      const unsigned slot = __shfl_sync(0xffffffffu, e.slot, 0);
      const int64_t row_delta = int64_t(slot) * 8 - int64_t(row_base);
      for (unsigned base = 0; base < splits; base += Interval, ++drain_epoch) {
        for (unsigned p = 4; p < 8; ++p)
          ho_drain(hdata, h_full, h_empty, p, drain_epoch, row_base + 4, counts,
                   kCandGlobalCap, 1 && base + Interval >= splits,
                   [&](unsigned row, unsigned slot2, unsigned record,
                       unsigned batch_base, unsigned source) {
                     const uint32_t physical =
                         kv_base + (batch_base * 2 + (record & 255u)) * 128 +
                         (source % 128);
                     const uint64_t off =
                         static_cast<uint64_t>(int64_t(row) + row_delta) *
                             kCandGlobalCap +
                         slot2;
                     fused_candidates::store_candidate_record(
                         vals + off, indices + off, record >> 8, physical);
                   });
      }
      cutlass::arch::NamedBarrier(320, 4).arrive();
    }
  }
} else if (warp_idx == 9) {
  // ---------------- KV producer: sample tiles then scan tiles, one ring
  // ----------------
  cutlass::arch::warpgroup_reg_dealloc<kSpecRegs>();
  RingPipeline<KVS> kv_pipe;
  [[maybe_unused]] RingPipeline<2 * KVS> sf_pipe;
  unsigned n_blk = 0;
  for (unsigned k = 0;; ++k) {
    claim_full[k & 1].wait((k >> 1) & 1);
    const ClaimEntry& e = entries[k & 1];
    const unsigned vblock = e.vblock, all_short = e.all_short,
                   kv_base = e.kv_base, splits = e.splits, samples = e.samples;
    __syncwarp();
    if (vblock >= kPause) break;
    claim_empty[k & 1].arrive();
    if (all_short) continue;
    if constexpr (KVS == 3) {
      // > 16K: the whole warp stages the strided sample scales into
      // sf_kv[stage] (prefetched one tile ahead in registers), lane 0 issues
      // the KV TMA.
      if (lane == 0 && n_blk > 0) sel_done.wait((n_blk - 1) & 1);
      __syncwarp();
      const unsigned center = sample_stride / 2;
      float pf[8];
#pragma unroll
      for (unsigned j = 0; j < 8; ++j) {
        const unsigned g = j * 32 + lane;
        pf[j] = g < sample_groups ? __ldg(scales + g * sample_stride + center)
                                  : 0.0f;
      }
#pragma unroll 1
      for (unsigned t = 0; t < samples; ++t) {
        // Only lane 0's ring state is maintained across blocks (the scan loop
        // runs on lane 0).
        CUTE_TIE_DECL(kv_pipe.advance(), s0, ph);
        const unsigned s = __shfl_sync(0xffffffffu, s0, 0);
        // Every writing lane acquires and releases the stage itself.
        empty_kv[s].wait(__shfl_sync(0xffffffffu, ph, 0) ^ 1);
#pragma unroll
        for (unsigned j = 0; j < 8; ++j) sm.sf_kv[s][j * 32 + lane] = pf[j];
        cutlass::arch::fence_view_async_shared();  // generic writes before
                                                   // later TMA writes of this
                                                   // stage
        if (lane == 0) {
          full_kv[s].expect_transaction(kKVBytes);
          fused_mqa_dg::tma_copy<128, 256, 0, cutlass::float_e4m3_t, true>(
              &tks, &full_kv[s], sm.kv[s], 0, t * 256, 1, center);
        }
        full_kv[s].arrive();
#pragma unroll
        for (unsigned j = 0; j < 8; ++j) {
          const unsigned g = (t + 1) * 256 + j * 32 + lane;
          pf[j] = g < sample_groups ? __ldg(scales + g * sample_stride + center)
                                    : 0.0f;
        }
      }
    }
    if (lane == 0) {
      if (KVS != 3 && n_blk > 0)
        sel_done.wait((n_blk - 1) &
                      1);  // selection scratch overlaps the KV stages
      const unsigned center = sample_stride / 2;
#pragma unroll 1
      for (unsigned t = 0; t < (KVS == 3 ? 0u : samples); ++t) {
        CUTE_TIE_DECL(kv_pipe.advance(), s, ph);
        empty_kv[s].wait(ph ^ 1);
        fused_mqa_dg::tma_copy<128, 256, 0, cutlass::float_e4m3_t, true>(
            &tks, &full_kv[s], sm.kv[s], 0, t * 256, 1, center);
        mbar_arrive_expect_tx(&full_kv[s], kKVBytes, kFullKvArrivals);
      }
#pragma unroll 1
      for (unsigned t = 0; t < splits; ++t) {
        CUTE_TIE_DECL(kv_pipe.advance(), s, ph);
        empty_kv[s].wait(ph ^ 1);
        const uint32_t off = kv_base + t * 256;
        fused_mqa_dg::tma_copy<128, 256, 0>(&tk, &full_kv[s], sm.kv[s], 0, off);
        if constexpr (kRel) {
          mbar_arrive_expect_tx(&full_kv[s], kKVBytes, kFullKvArrivals);
          CUTE_TIE_DECL(sf_pipe.advance(), sfs, sfph);
          (void)sfph;
          fused_mqa_dg::tma_copy<256, 1, 0>(&ts, &sf_full[sfs], sm.sf_kv[sfs],
                                            off, 0);
          sf_full[sfs].arrive_and_expect_tx(1024);
        } else {
          fused_mqa_dg::tma_copy<256, 1, 0>(&ts, &full_kv[s], sm.sf_kv[s], off,
                                            0);
          mbar_arrive_expect_tx(&full_kv[s], kKVBytes + 1024, kFullKvArrivals);
        }
      }
    }
    __syncwarp();
    ++n_blk;
  }
} else if (warp_idx == 10) {
  // ---------------- UMMA issuer ----------------
  cutlass::arch::warpgroup_reg_dealloc<kSpecRegs>();
  RingPipeline<1> q_pipe;
  RingPipeline<KVS> kv_pipe;
  RingPipeline<B::NT> tmem_pipe;
  auto instr_desc =
      cute::UMMA::make_instr_desc<cutlass::float_e4m3_t, cutlass::float_e4m3_t,
                                  float, 128, kLin ? 144 : 256,
                                  cute::UMMA::Major::K, cute::UMMA::Major::K>();
  auto runtime_instr_desc = cute::UMMA::make_runtime_instr_desc(instr_desc);
  for (unsigned k = 0;; ++k) {
    claim_full[k & 1].wait((k >> 1) & 1);
    const ClaimEntry& e = entries[k & 1];
    const unsigned vblock = e.vblock, all_short = e.all_short,
                   splits = e.splits, samples = e.samples;
    __syncwarp();
    if (vblock >= kPause) break;
    claim_empty[k & 1].arrive();
    if (all_short) continue;
    CUTE_TIE_DECL(q_pipe.advance(), q_stage, q_phase);
    (void)q_stage;
    full_q.wait(q_phase);
    if constexpr (kLin) {
      full_lin->wait(q_phase);  // this block's linear-term rows (warp 11)
      ptx::tcgen05_after_thread_sync();
    }
    const unsigned total = samples + splits;
#pragma unroll 1
    for (unsigned t = 0; t < total; ++t) {
      CUTE_TIE_DECL(kv_pipe.advance(), s, ph);
      full_kv[s].wait(ph);
      if (cute::elect_one_sync()) {
        if constexpr (kLin) {
          // Per KV half, one N=144 MMA per math warpgroup into its own TMEM
          // slot (ring of 3; WG0 takes even, WG1 odd positions of the issue
          // sequence).
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
                         : sm.q + 128 * 128;
#pragma unroll
              for (uint32_t kk = 0; kk < 4; ++kk) {
                auto a_desc =
                    mma::sm100::make_umma_desc<cute::UMMA::Major::K, 0, 128,
                                               128>(sm.kv[s], i * 128, kk * 32);
                auto b_desc =
                    mma::sm100::make_umma_desc<cute::UMMA::Major::K, 0, 128,
                                               128>(b_base, 0, kk * 32);
                ptx::SM100_MMA_F8F6F4_SS::fma(a_desc, b_desc, tmem_addr, kk,
                                              runtime_instr_desc);
              }
              ptx::umma_arrive_no_elect(full_tmem[ts_idx]);
            }
          }
        } else {
#pragma unroll
          for (uint32_t i = 0; i < 2; ++i) {
            CUTE_TIE_DECL(tmem_pipe.advance(), ts_idx, ts_ph);
            const uint32_t tmem_addr = ts_idx * 256;
            empty_tmem[ts_idx].wait(ts_ph ^ 1);
            ptx::tcgen05_after_thread_sync();
#pragma unroll
            for (uint32_t kk = 0; kk < 4; ++kk) {
              auto a_desc =
                  mma::sm100::make_umma_desc<cute::UMMA::Major::K, 0, 128, 128>(
                      sm.kv[s], i * 128, kk * 32);
              auto b_desc =
                  mma::sm100::make_umma_desc<cute::UMMA::Major::K, 0, 128, 128>(
                      sm.q, 0, kk * 32);
              ptx::SM100_MMA_F8F6F4_SS::fma(a_desc, b_desc, tmem_addr, kk,
                                            runtime_instr_desc);
            }
            ptx::umma_arrive_no_elect(full_tmem[ts_idx]);
          }
          if constexpr (kRel) ptx::umma_arrive_no_elect(empty_kv[s]);
        }
      }
      __syncwarp();
    }
    empty_q.arrive();
  }
} else if (warp_idx == 11) {
  // ---------------- drain of WG0 rows (> 16K) ----------------
  cutlass::arch::warpgroup_reg_dealloc<kSpecRegs>();
  if constexpr (!kSmem) {
    unsigned drain_epoch = 0, n_lin = 0;
    for (unsigned k = 0;; ++k) {
      claim_full[k & 1].wait((k >> 1) & 1);
      const ClaimEntry& e = entries[k & 1];
      const unsigned vblock = e.vblock, all_short = e.all_short,
                     kv_base = e.kv_base, splits = e.splits, slot = e.slot;
      __syncwarp();
      if (vblock >= kPause) break;
      claim_empty[k & 1].arrive();
      if (all_short) continue;
      if constexpr (kLin) {
        // Q of block k lands while the math warps still select block k-1, so
        // this is off the critical path except for a CTA's first block.
        full_q.wait(n_lin & 1);
        ++n_lin;
        compute_lin_rows(sm, lane);
        cutlass::arch::fence_view_async_shared();  // generic smem writes ->
                                                   // UMMA (async proxy)
        full_lin->arrive();  // every lane arrives (count 32)
      }
      const unsigned row_base = vblock * 8;
      const int64_t row_delta = int64_t(slot) * 8 - int64_t(row_base);
      for (unsigned base = 0; base < splits; base += Interval, ++drain_epoch) {
        for (unsigned p = 0; p < 4; ++p)
          ho_drain(hdata, h_full, h_empty, p, drain_epoch, row_base, counts,
                   kCandGlobalCap, 1 && base + Interval >= splits,
                   [&](unsigned row, unsigned slot2, unsigned record,
                       unsigned batch_base, unsigned source) {
                     const uint32_t physical =
                         kv_base + (batch_base * 2 + (record & 255u)) * 128 +
                         (source % 128);
                     const uint64_t off =
                         static_cast<uint64_t>(int64_t(row) + row_delta) *
                             kCandGlobalCap +
                         slot2;
                     fused_candidates::store_candidate_record(
                         vals + off, indices + off, record >> 8, physical);
                   });
      }
      cutlass::arch::NamedBarrier(320, 4).arrive();
    }
  }
} else {
  // ---------------- math warps 0-7 ----------------
  cutlass::arch::warpgroup_reg_alloc<kMathRegs>();
  const unsigned math_tid = threadIdx.x;
  const unsigned query_base = (warp_idx / 4) * 4;
  RingPipeline<1> q_pipe;
  RingPipeline<KVS> kv_pipe;
  [[maybe_unused]] RingPipeline<2 * KVS> sf_pipe;
  RingPipeline<B::NT> tmem_pipe;
  // > 16K: each warpgroup consumes every second TMEM slot position (WG1 starts
  // at 1).
  constexpr unsigned kTmemStep = kLin ? 2 : 1;
  if (kLin && warp_idx >= 4) tmem_pipe.advance(1);
  [[maybe_unused]] const uint32_t heads_col = warp_idx < 4 ? 16u : 0u;
  [[maybe_unused]] const uint32_t lin_col = warp_idx < 4 ? 0u : 128u;
  unsigned produce_epoch = 0;
  constexpr uint32_t kRingSlots = 8;
  for (unsigned k = 0;; ++k) {
    claim_full[k & 1].wait((k >> 1) & 1);
    const ClaimEntry& e = entries[k & 1];
    const unsigned vblock = e.vblock;
    if (vblock >= kPause) break;
    const unsigned row_base = vblock * 8;
    if (e.all_short) {
      for (unsigned r = 0; r < 8 && row_base + r < rows; ++r) {
        const unsigned s0 = e.starts[r], vis = e.ends[r] - s0;
        for (unsigned j = math_tid; j < topk; j += 256)
          fallback.output[(uint64_t)(row_base + r) * topk + j] =
              j < vis ? int((relative ? 0u : s0) + j) : -1;
      }
      __syncwarp();
      claim_empty[k & 1].arrive();
      continue;
    }
    // ---- block setup: Q, weights, sample histogram ----
    {
      CUTE_TIE_DECL(q_pipe.advance(), q_stage, q_phase);
      (void)q_stage;
      full_q.wait(q_phase);
      if constexpr (kLin) full_lin->wait(q_phase);
    }
    static_assert(SampleQ * 2112 % 4 == 0);
    for (unsigned j = math_tid; j < SampleQ * 2112 / 4; j += 256)
      reinterpret_cast<uint4*>(stream.hist)[j] = make_uint4(0, 0, 0, 0);
    float weights[4][32];
    [[maybe_unused]] float
        row_s[4];  // > 16K: pow2 row scale of the linear term
    uint32_t seq_k_start[4], seq_k_end[4];
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i) {
      seq_k_start[i] = e.starts[query_base + i];
      seq_k_end[i] = e.ends[query_base + i];
      if constexpr (kLin) {
        // Weights in units of 1 / s_r (exact: s_r is a power of two).
        row_s[i] = sm.lin_scale[query_base + i];
        const float inv_s = 1.0f / row_s[i];
#pragma unroll
        for (uint32_t j = 0; j < 32; j += 4) {
          const float4 w4 = *reinterpret_cast<const float4*>(
              sm.weights + (query_base + i) * 32 + j);
          weights[i][j] = w4.x * 0.5f * inv_s;
          weights[i][j + 1] = w4.y * 0.5f * inv_s;
          weights[i][j + 2] = w4.z * 0.5f * inv_s;
          weights[i][j + 3] = w4.w * 0.5f * inv_s;
        }
      } else {
#pragma unroll
        for (uint32_t j = 0; j < 32; j += 4) {
          const float4 w4 = *reinterpret_cast<const float4*>(
              sm.weights + (query_base + i) * 32 + j);
          weights[i][j] = w4.x * 0.5f;
          weights[i][j + 1] = w4.y * 0.5f;
          weights[i][j + 2] = w4.z * 0.5f;
          weights[i][j + 3] = w4.w * 0.5f;
        }
      }
    }
    {
      const float w = sm.weights[warp_idx * 32 + lane];
      bool bad_w = w != 0.0f && fabsf(w) < 0x1p-125f;
      if constexpr (kLin) {
        // Scaled weights that become subnormal or infinite take the exact
        // recovery path.
        const float wr = w * 0.5f * (1.0f / sm.lin_scale[warp_idx]);
        bad_w |= (wr != 0.0f && fabsf(wr) < 0x1p-125f) || isinf(wr);
      }
      const unsigned bad = __ballot_sync(0xffffffffu, bad_w);
      if (lane == 0 && bad) atomicOr(&unsafe_s, 1u << warp_idx);
    }
    sample_sync();
    // ---- sample tiles ----
#pragma unroll 1
    for (unsigned t = 0; t < e.samples; ++t) {
      CUTE_TIE_DECL(kv_pipe.advance(), kv_stage, kv_phase);
      if constexpr (!kRel) full_kv[kv_stage].wait(kv_phase);
#pragma unroll
      for (unsigned kv_half = 0; kv_half < 2; ++kv_half) {
        const unsigned kv_lane = (math_tid % 128) + kv_half * 128;
        const unsigned group = t * 256 + kv_lane;
        const unsigned kv_offset = group * sample_stride + sample_stride / 2;
        const bool in_view = group < sample_groups;
        float scale_kv;
        if constexpr (kLin)
          scale_kv = ptx::ld_shared(sm.sf_kv[kv_stage] +
                                    kv_lane);  // 0 outside the view
        else
          scale_kv = in_view ? __ldg(scales + kv_offset) : 0.0f;
        CUTE_TIE_DECL(tmem_pipe.advance(kTmemStep), tmem_stage, tmem_phase);
        full_tmem[tmem_stage].wait(tmem_phase);
        ptx::tcgen05_after_thread_sync();
        if (!kRel && kv_half == 1) {
          cutlass::arch::fence_view_async_shared();
          empty_kv[kv_stage].arrive();
        }
        // One query row (32 TMEM columns) at a time keeps the sample loop's
        // live registers low enough to carry the scan weights through it.
#pragma unroll
        for (uint32_t i = 0; i < 4; ++i) {
          float acc[32];
          [[maybe_unused]] float lin[4];
          if constexpr (kLin) {
            ptx::tmem_load_32dp32b<32>(
                tmem_stage * kSlotCols + heads_col + i * 32,
                reinterpret_cast<uint32_t*>(acc));
            ptx::tmem_load_32dp32b<4>(tmem_stage * kSlotCols + lin_col + i * 4,
                                      reinterpret_cast<uint32_t*>(lin));
          } else {
            ptx::tmem_load_32dp32b<32>(tmem_stage * 256 + (query_base + i) * 32,
                                       reinterpret_cast<uint32_t*>(acc));
          }
          cutlass::arch::fence_view_async_tmem_load();
          if (i == 3) {
            ptx::tcgen05_before_thread_sync();
            empty_tmem[tmem_stage].arrive();
          }
          float2 sum_0 = {}, sum_1 = {};
          if constexpr (kLin) {  // terms 2, 3 are in units of s * 2^-8
            sum_0 = make_float2(lin[0], lin[1]);
            sum_1 = __fmul2_rn(make_float2(lin[2], lin[3]),
                               make_float2(0x1p-8f, 0x1p-8f));
          }
#pragma unroll
          for (uint32_t j = 0; j < 32; j += 4) {
            auto transform = [&](uint32_t at, float2 sum) {
              const auto abs_x =
                  make_float2(fabsf(acc[at]), fabsf(acc[at + 1]));
              const auto w = make_float2(weights[i][at], weights[i][at + 1]);
              if constexpr (kLin) {
                return __ffma2_rn(
                    abs_x, w,
                    sum);  // relu = (x + |x|) / 2, linear part on the TC
              } else {
                const auto x = make_float2(acc[at], acc[at + 1]);
                return __ffma2_rn(__fadd2_rn(x, abs_x), w, sum);
              }
            };
            sum_0 = transform(j, sum_0);
            sum_1 = transform(j + 2, sum_1);
          }
          const auto sum = __fadd2_rn(sum_0, sum_1);
          float bq;
          if constexpr (kLin)
            bq = (sum.x + sum.y) * (row_s[i] * scale_kv);
          else
            bq = (sum.x + sum.y) * scale_kv;
          const bool valid = in_view && kv_offset >= seq_k_start[i] &&
                             kv_offset < seq_k_end[i] && isfinite(bq);
          if (valid) {
            const unsigned bin = sample_bin(bq);
            atomicAdd(stream.hist[query_base + i] + bin + bin / 32, 1u);
          }
        }
      }
    }
    sample_sync();
    finish_sample_histogram(stream.hist[warp_idx], gate_s + warp_idx,
                            e.ends[warp_idx] - e.starts[warp_idx],
                            sample_target, lane);
    sample_sync();
    const unsigned row_base2 =
        *reinterpret_cast<const volatile unsigned*>(&e.vblock) * 8;

    // ---- scan tiles ----
    if constexpr (kLin) {
#pragma unroll
      for (uint32_t i = 0; i < 4; ++i) {
        const float inv_s = 1.0f / row_s[i];
#pragma unroll
        for (uint32_t j = 0; j < 32; j += 4) {
          const float4 w4 = *reinterpret_cast<const float4*>(
              sm.weights + (query_base + i) * 32 + j);
          weights[i][j] = w4.x * 0.5f * inv_s;
          weights[i][j + 1] = w4.y * 0.5f * inv_s;
          weights[i][j + 2] = w4.z * 0.5f * inv_s;
          weights[i][j + 3] = w4.w * 0.5f * inv_s;
        }
      }
    }
    const unsigned kv_base = e.kv_base, splits = e.splits;
    const int64_t cand_row_delta = int64_t(e.slot) * 8 - int64_t(row_base2);
    float gate_score[4];
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i) {
      gate_score[i] = gate_s[query_base + i];
      if (*reinterpret_cast<const volatile unsigned*>(&unsafe_s) >>
              (query_base + i) &
          1u) {
        gate_score[i] = __int_as_float(0x7f800000);
        if (math_tid % 128 == 0)
          atomicAdd(counts + row_base2 + query_base + i,
                    static_cast<int>(kCandGlobalCap) + 1);
      }
      // > 16K: compare S * scale_kv against gate / row_s (row_s is a power of
      // two -> exact).
      if constexpr (kLin) gate_score[i] *= 1.0f / row_s[i];
    }
    uint32_t lane_counts[4] = {};
    uint32_t flush_base = 0;
    const uint32_t logits_stride = kCandCap;
    [[maybe_unused]] const uint32_t pool_size = logits_stride / 6,
                                    seg_size = (logits_stride - pool_size) / 4;
    [[maybe_unused]] const uint32_t seg_warp = warp_idx % 4;
    [[maybe_unused]] const uint32_t cand_ext = 8 * logits_stride + 256;
    [[maybe_unused]] uint32_t seg_cnt[4] = {};
    if constexpr (kSmem) {
      if (math_tid < 8) {
        row_count[math_tid] = 0;
        pool_cnt[math_tid] = 0;
      }
      cutlass::arch::NamedBarrier(256, 3).sync();
    }
#pragma unroll 1
    for (uint32_t kv_split_idx = 0; kv_split_idx < splits; ++kv_split_idx) {
      if constexpr (!kSmem)
        if ((kv_split_idx & (Interval - 1)) == 0) {
          h_empty[(produce_epoch & 1) * 8 + warp_idx].wait(
              ((produce_epoch >> 1) & 1) ^ 1);
          __syncwarp();
        }
      CUTE_TIE_DECL(kv_pipe.advance(), kv_stage, kv_phase);
      unsigned sf_stage = kv_stage;
      if constexpr (kRel) {
        CUTE_TIE_DECL(sf_pipe.advance(), rs, rph);
        sf_full[rs].wait(rph);
        sf_stage = rs;
      } else {
        full_kv[kv_stage].wait(kv_phase);
      }
#pragma unroll
      for (unsigned kv_half = 0; kv_half < 2; ++kv_half) {
        const unsigned kv_lane = (math_tid % 128) + kv_half * 128;
        const unsigned kv_offset = kv_base + kv_split_idx * 256 + kv_lane;
        const float scale_kv = ptx::ld_shared(sm.sf_kv[sf_stage] + kv_lane);
        CUTE_TIE_DECL(tmem_pipe.advance(kTmemStep), tmem_stage, tmem_phase);
        full_tmem[tmem_stage].wait(tmem_phase);
        ptx::tcgen05_after_thread_sync();
        if (!kRel && kv_half == 1) {
          cutlass::arch::fence_view_async_shared();
          empty_kv[kv_stage].arrive();
        }
        float accum_pair[2][32];
        [[maybe_unused]] float lin_pair[2][4];
        [[maybe_unused]] float bp4[4];  // > 16K: row scores of the slot
#pragma unroll
        for (uint32_t pair = 0; pair < 4; pair += 2) {
          if constexpr (kLin) {
            const uint32_t slot_base = tmem_stage * kSlotCols;
            ptx::tmem_load_32dp32b<64>(slot_base + heads_col + pair * 32,
                                       reinterpret_cast<uint32_t*>(accum_pair));
            ptx::tmem_load_32dp32b<8>(slot_base + lin_col + pair * 4,
                                      reinterpret_cast<uint32_t*>(lin_pair));
          } else {
            const uint32_t tmem_addr =
                tmem_stage * 256 + (query_base + pair) * 32;
            ptx::tmem_load_32dp32b<64>(tmem_addr,
                                       reinterpret_cast<uint32_t*>(accum_pair));
          }
          cutlass::arch::fence_view_async_tmem_load();
          if (pair + 2 >= 4) {
            ptx::tcgen05_before_thread_sync();
            empty_tmem[tmem_stage].arrive();
          }
          float2 sum_0[2] = {}, sum_1[2] = {};
          if constexpr (kLin) {
            // The linear term (sum_t term_t . k, in units of the row scale)
            // seeds the sums.
#pragma unroll
            for (uint32_t r = 0; r < 2; ++r) {
              sum_0[r] = make_float2(lin_pair[r][0], lin_pair[r][1]);
              sum_1[r] =
                  __fmul2_rn(make_float2(lin_pair[r][2], lin_pair[r][3]),
                             make_float2(0x1p-8f, 0x1p-8f));  // units s * 2^-8
            }
          }
#pragma unroll
          for (uint32_t j = 0; j < 32; j += 4) {
#pragma unroll
            for (uint32_t r = 0; r < 2; ++r) {
              const uint32_t i = pair + r;
              auto* a = accum_pair[r];
              auto transform = [&](uint32_t at, float2 sum) {
                const auto abs_x = make_float2(fabsf(a[at]), fabsf(a[at + 1]));
                const auto w = make_float2(weights[i][at], weights[i][at + 1]);
                if constexpr (kLin) {
                  return __ffma2_rn(abs_x, w, sum);
                } else {
                  const auto x = make_float2(a[at], a[at + 1]);
                  return __ffma2_rn(__fadd2_rn(x, abs_x), w, sum);
                }
              };
              sum_0[r] = transform(j, sum_0[r]);
              sum_1[r] = transform(j + 2, sum_1[r]);
            }
          }
          if constexpr (kLin) {
#pragma unroll
            for (uint32_t r = 0; r < 2; ++r) {
              const auto sum = __fadd2_rn(sum_0[r], sum_1[r]);
              bp4[pair + r] = (sum.x + sum.y) * scale_kv;
            }
          } else
#pragma unroll
            for (uint32_t r = 0; r < 2; ++r) {
              const uint32_t i = pair + r;
              const auto sum = __fadd2_rn(sum_0[r], sum_1[r]);
              float bq;
              if constexpr (kLin)
                bq = (sum.x + sum.y) * (row_s[i] * scale_kv);
              else
                bq = (sum.x + sum.y) * scale_kv;
              const uint32_t row = row_base2 + query_base + i;
              const bool pass =
                  (kSmem ? kv_offset - seq_k_start[i] <
                               seq_k_end[i] - seq_k_start[i]
                         : kv_offset >= seq_k_start[i] &&
                               kv_offset < seq_k_end[i] && isfinite(bq)) &&
                  bq > gate_score[i];
              if constexpr (kSmem) {
                const unsigned pass_mask = __ballot_sync(0xffffffffu, pass);
                const unsigned slot =
                    seg_cnt[i] + __popc(pass_mask & ((1u << lane) - 1u));
                seg_cnt[i] += __popc(pass_mask);
                const uint32_t code = fused_candidates::candidate_fp24_code(bq);
                const uint32_t seg_cap = seg_size;
                const uint32_t at =
                    pass && slot < seg_cap
                        ? (query_base + i) * logits_stride +
                              seg_warp * seg_size + slot
                        : 8 * logits_stride + warp_idx * 32 + lane;
                auto* cbase = region;
                reinterpret_cast<uint16_t*>(cbase)[at] =
                    static_cast<uint16_t>(code >> 8);
                cbase[cand_ext * 2 + at] = static_cast<uint8_t>(code);
                reinterpret_cast<uint16_t*>(cbase + cand_ext * 3)[at] =
                    static_cast<uint16_t>(kv_offset);
                if (seg_cnt[i] > seg_cap) {
                  const bool over = pass && slot >= seg_cap;
                  const unsigned over_mask = __ballot_sync(0xffffffffu, over);
                  if (over_mask) {
                    unsigned pool_base = 0;
                    if (lane == 0)
                      pool_base = atomicAdd(&pool_cnt[query_base + i],
                                            __popc(over_mask));
                    pool_base = __shfl_sync(0xffffffffu, pool_base, 0);
                    const unsigned p =
                        pool_base + __popc(over_mask & ((1u << lane) - 1u));
                    if (over && p < pool_size) {
                      const uint32_t pat =
                          (query_base + i) * logits_stride + 4 * seg_size + p;
                      reinterpret_cast<uint16_t*>(cbase)[pat] =
                          static_cast<uint16_t>(code >> 8);
                      cbase[cand_ext * 2 + pat] = static_cast<uint8_t>(code);
                      reinterpret_cast<uint16_t*>(cbase + cand_ext * 3)[pat] =
                          static_cast<uint16_t>(kv_offset);
                    } else if (over && p - pool_size < kCandGlobalCap - 1) {
                      const uint64_t off =
                          static_cast<uint64_t>(int64_t(row) + cand_row_delta) *
                              kCandGlobalCap +
                          (p - pool_size);
                      __stcg(vals + off, static_cast<uint16_t>(code >> 8));
                      __stcg(indices + off,
                             static_cast<int32_t>((code & 0xffu) << 16 |
                                                  kv_offset));
                    }
                  }
                }
              } else if (pass) {
                const uint32_t count = lane_counts[i];
                if (count < kRingSlots) {
                  const uint32_t code =
                      fused_candidates::candidate_fp24_code(bq);
                  hdata.records[produce_epoch & 1][warp_idx][i][count][lane] =
                      (code << 8) | ((kv_split_idx - flush_base) * 2 + kv_half);
                  lane_counts[i] = count + 1;
                } else {
                  const int slot = atomicAdd(counts + row, 1);
                  if (slot < static_cast<int>(kCandGlobalCap)) {
                    const uint64_t off =
                        static_cast<uint64_t>(int64_t(row) + cand_row_delta) *
                            kCandGlobalCap +
                        slot;
                    fused_candidates::store_candidate(vals + off, indices + off,
                                                      bq, kv_offset);
                  }
                }
              }
            }
        }
        if constexpr (kLin) {
          // One warp vote per slot: most keys fail every gate, so the common
          // path is 4 compares + 1 vote + 1 uniform branch.
          const bool any_pass = (bp4[0] > gate_score[0]) | (bp4[1] > gate_score[1]) |
                                (bp4[2] > gate_score[2]) | (bp4[3] > gate_score[3]);
          if (__any_sync(0xffffffffu, any_pass)) [[unlikely]] {
#pragma unroll
            for (uint32_t i = 0; i < 4; ++i) {
              const float bp = bp4[i];
              if (bp > gate_score[i]) {  // gate first; range / finiteness only
                                         // for passing keys
                const float bq = bp * row_s[i];
                const uint32_t row = row_base2 + query_base + i;
                if (kv_offset >= seq_k_start[i] && kv_offset < seq_k_end[i] &&
                    isfinite(bq)) {
                  const uint32_t count = lane_counts[i];
                  if (count < kRingSlots) {
                    const uint32_t code =
                        fused_candidates::candidate_fp24_code(bq);
                    hdata.records[produce_epoch & 1][warp_idx][i][count][lane] =
                        (code << 8) |
                        ((kv_split_idx - flush_base) * 2 + kv_half);
                    lane_counts[i] = count + 1;
                  } else {
                    const int slot = atomicAdd(counts + row, 1);
                    if (slot < static_cast<int>(kCandGlobalCap)) {
                      const uint64_t off =
                          static_cast<uint64_t>(int64_t(row) + cand_row_delta) *
                              kCandGlobalCap +
                          slot;
                      fused_candidates::store_candidate(
                          vals + off, indices + off, bq, kv_offset);
                    }
                  }
                }
              }
            }
          }
        }
      }
      if (!kSmem &&
          (((kv_split_idx + 1) & (Interval - 1)) == 0 ||
           kv_split_idx + 1 == splits) &&
          kv_split_idx + 1 != splits) {
#pragma unroll
        for (unsigned i = 0; i < 4; ++i) {
          hdata.counts[produce_epoch & 1][warp_idx][i][lane] = lane_counts[i];
          lane_counts[i] = 0;
        }
        if (lane == 0) hdata.bases[produce_epoch & 1][warp_idx] = flush_base;
        __syncwarp();
        h_full[(produce_epoch & 1) * 8 + warp_idx].arrive();
        ++produce_epoch;
        flush_base = kv_split_idx + 1;
      }
    }
    if constexpr (!kSmem) {
      // Final interval: every math warp stores its own ring records (8 warps in
      // parallel instead of 2 drain warps after the scan); the drain warps only
      // cycle the barriers.
      const unsigned bank = produce_epoch & 1, rbase = row_base2 + query_base;
      static_assert(sizeof(hdata.records[0][0]) == 4 * 8 * 32 * 4,
                    "ring record layout");
      self_drain_last(&hdata.records[bank][warp_idx][0][0][0], lane_counts[0],
                      lane_counts[1], lane_counts[2], lane_counts[3], counts,
                      rbase, kv_base + (math_tid % 128), flush_base,
                      cand_row_delta, vals, indices);
#pragma unroll
      for (unsigned row = 0; row < 4; ++row) lane_counts[row] = 0;
      __syncwarp();
      h_full[bank * 8 + warp_idx].arrive();
      ++produce_epoch;
    }
    if constexpr (kSmem) {
#pragma unroll
      for (uint32_t i = 0; i < 4; ++i) {
        auto* base = region;
        for (uint32_t slot = min(seg_cnt[i], seg_size) + lane; slot < seg_size;
             slot += 32) {
          const uint32_t at =
              (query_base + i) * logits_stride + seg_warp * seg_size + slot;
          reinterpret_cast<uint16_t*>(base)[at] = 0;
          base[cand_ext * 2 + at] = 0;
          reinterpret_cast<uint16_t*>(base + cand_ext * 3)[at] = 0;
        }
        if (lane == 0) atomicAdd(&row_count[query_base + i], seg_cnt[i]);
      }
      cutlass::arch::NamedBarrier(256, 3).sync();
      for (uint32_t q = 0; q < 8; ++q) {
        auto* base = region;
        for (uint32_t p = pool_cnt[q] + math_tid; p < pool_size; p += 256) {
          const uint32_t at = q * logits_stride + 4 * seg_size + p;
          reinterpret_cast<uint16_t*>(base)[at] = 0;
          base[cand_ext * 2 + at] = 0;
          reinterpret_cast<uint16_t*>(base + cand_ext * 3)[at] = 0;
        }
      }
      if (math_tid < 8) {
        const uint32_t row = row_base2 + math_tid;
        const uint32_t spilled =
            pool_cnt[math_tid] > pool_size
                ? min(pool_cnt[math_tid] - pool_size, kCandGlobalCap - 1)
                : 0u;
        __stcg(indices +
                   static_cast<uint64_t>(int64_t(row) + cand_row_delta) *
                       kCandGlobalCap +
                   kCandGlobalCap - 1,
               static_cast<int32_t>(spilled));
        if (spilled) atomicOr(&row_count[math_tid], 1u << 21);
      }
      cutlass::arch::NamedBarrier(256, 3).sync();
      if (math_tid < 8)
        atomicAdd(counts + row_base2 + math_tid,
                  static_cast<int>(row_count[math_tid]));
    }
    cutlass::arch::fence_view_async_shared();
    empty_q.arrive();
    if constexpr (!kSmem)
      cutlass::arch::NamedBarrier(320, 4).sync();  // drains of this block done
    // ---- selection ----
    const unsigned vblock3 =
                       *reinterpret_cast<const volatile unsigned*>(&e.vblock),
                   row_base3 = vblock3 * 8;
    __threadfence_block();
    cutlass::arch::NamedBarrier(256, 5).sync();
    const ClaimEntry& ce = entries[k & 1];
    if (math_tid == 0) {
      unsigned recover = 0, short_rows = 0;
      row_offset[0] = 0;
      for (unsigned r = 0; r < 8; ++r) {
        const bool is_short = ce.ends[r] - ce.starts[r] <= topk;
        short_rows |= unsigned(is_short) << r;
        const int c =
            kSmem ? int(row_count[r]) : __ldcg(counts + row_base3 + r);
        const int real = kSmem ? (c & ((1 << 21) - 1)) : c;
        recover |= !is_short && (real < int(topk) || real > 16384);
        // > 16K rows start 16-byte aligned for select_staged_row's vector
        // loads.
        row_offset[r + 1] =
            row_offset[r] + (is_short
                                 ? 0
                                 : (!kSmem ? (unsigned(max(c, 0)) + 3u) & ~3u
                                           : unsigned(max(c, 0))));
      }
      unsigned staged = 1;
      if (!kSmem) {
        const unsigned half = max(row_offset[4], row_offset[8] - row_offset[4]);
        staged = recover ? 0u
                         : (8u * fast_select::kBins * 4u + row_offset[8] * 4u <=
                                    phase_bytes
                                ? 1u
                                : (4u * fast_select::kBins * 4u + half * 4u <=
                                           phase_bytes
                                       ? 2u
                                       : 0u));
        if (staged == 0)
          recover = 1;  // candidates exceed the scratch: exact rescan
      }
      if (recover) {
        defer_next[vblock3] = int(defer_head);
        defer_head = vblock3;
        atomicExch(&pause_req, 1u);
      }
      sel_recover = recover;
      sel_staged = staged;
      sel_short = short_rows;
      unsafe_s = 0;
    }
    cutlass::arch::NamedBarrier(256, 5).sync();
    const unsigned recover = sel_recover, short_rows = sel_short;
    const auto write_short = [&](unsigned r, unsigned t, unsigned stride) {
      if (r >= rows) return;
      const unsigned s0 = ce.starts[r - row_base3],
                     visible = ce.ends[r - row_base3] - s0;
      int* o = fallback.output + (uint64_t)r * topk;
      for (unsigned j = t; j < topk; j += stride)
        o[j] = j < visible ? int((relative ? 0u : s0) + j) : -1;
    };
    if (!recover) {
      const int64_t row_delta = int64_t(ce.slot) * 8 - int64_t(row_base3);
      if constexpr (kSmem) {
        const unsigned warp = warp_idx;
        if (row_base3 + warp < rows) {
          const unsigned r = row_base3 + warp;
          int* o = fallback.output + (uint64_t)r * topk;
          if (short_rows >> warp & 1u)
            write_short(r, lane, 32);
          else {
            const uint16_t* hi =
                reinterpret_cast<const uint16_t*>(region) + warp * kCandCap;
            const uint8_t* lo = region + kCandExt * 2 + warp * kCandCap;
            const uint16_t* idx =
                reinterpret_cast<const uint16_t*>(region + kCandExt * 3) +
                warp * kCandCap;
            const auto key_of = [idx](unsigned e2) { return int32_t(idx[e2]); };
            unsigned char* scratch = smem_buffer + kSelB + warp * kScratchS;
            const int c = int(row_count[warp]);
            if (c >> 21 & 1) {
              const uint64_t gbase = uint64_t(int64_t(r) + row_delta) * 16384;
              const uint16_t* gv = vals + gbase;
              const int* gi = indices + gbase;
              const unsigned total = kCandCap + unsigned(__ldcg(gi + 16383));
              fast_select::select_hist_generic(
                  [&](unsigned e2) {
                    return e2 < kCandCap
                               ? (unsigned(hi[e2]) << 8 | lo[e2])
                               : (unsigned(__ldcg(gv + e2 - kCandCap)) << 8 |
                                  (unsigned(__ldcg(gi + e2 - kCandCap)) >> 16));
                  },
                  total, o, topk, reinterpret_cast<unsigned*>(scratch),
                  [&](unsigned e2) {
                    return e2 < kCandCap
                               ? int32_t(idx[e2])
                               : int32_t(__ldcg(gi + e2 - kCandCap) & 0xffff);
                  });
            } else if (!row_select::select_shared_row(
                           hi, lo, idx, kCandCap, unsigned(c & ((1 << 21) - 1)),
                           o, topk, scratch)) {
              fast_select::select_resident(hi, lo, kCandCap, 0u, o, topk,
                                           reinterpret_cast<unsigned*>(scratch),
                                           key_of);
            }
          }
        }
      } else {
        const unsigned per_round = sel_staged == 1 ? 8u : 4u;
        for (unsigned round = 0; round < 8 / per_round; ++round) {
          if (warp_idx < per_round) {
            const unsigned local = round * per_round + warp_idx,
                           first = round * per_round;
            unsigned* hist = reinterpret_cast<unsigned*>(smem_buffer + kSelB) +
                             warp_idx * fast_select::kBins;
            unsigned* codes = reinterpret_cast<unsigned*>(smem_buffer + kSelB) +
                              per_round * fast_select::kBins +
                              (row_offset[local] - row_offset[first]);
            const unsigned r = row_base3 + local;
            const uint64_t base = uint64_t(int64_t(r) + row_delta) * 16384;
            if (short_rows >> local & 1u)
              write_short(r, lane, 32);
            else {
              const unsigned cnt = __ldcg(counts + r);
              int* o = fallback.output + (uint64_t)r * topk;
              if (!row_select::select_staged_row(codes, cnt, vals + base,
                                                 indices + base, o, topk, hist))
                fast_select::select_row(codes, cnt, vals + base, indices + base,
                                        o, topk, hist);
            }
          }
          cutlass::arch::NamedBarrier(256, 5).sync();
        }
      }
    }
    cutlass::arch::NamedBarrier(256, 5).sync();
    if (relative && !recover) {
      const unsigned quads = topk / 4;
      for (unsigned i = math_tid; i < 8 * quads; i += 256) {
        const unsigned r = row_base3 + i / quads;
        if (r >= rows || ce.ends[i / quads] - ce.starts[i / quads] <= topk)
          continue;
        const int s = int(starts_g[r]);
        if (s == 0) continue;
        int4* p =
            reinterpret_cast<int4*>(fallback.output + (uint64_t)r * topk) +
            i % quads;
        int4 v = *p;
        v.x = v.x < 0 ? -1 : v.x - s;
        v.y = v.y < 0 ? -1 : v.y - s;
        v.z = v.z < 0 ? -1 : v.z - s;
        v.w = v.w < 0 ? -1 : v.w - s;
        *p = v;
      }
    }
    __syncwarp();
    if (lane == 0) sel_done.arrive();
    claim_empty[k & 1].arrive();
  }
}

// ---------------- exact rescan of the blocks queued by the selection
// ----------------
__syncthreads();
if (warp_idx == 0) cute::TMEM::Allocator1Sm().free(0, 512);
if (defer_head != kTerm) {
  if constexpr (kMathRegs != 224) {
    // The fallback core expects the 224 / 56 split.
    if (warp_idx < 8) cutlass::arch::warpgroup_reg_dealloc<224>();
    __syncthreads();
    if (warp_idx >= 8) cutlass::arch::warpgroup_reg_alloc<56>();
  }
  __syncthreads();
  for (unsigned cur = defer_head; cur != kTerm;) {
    const unsigned vblock = cur, row_base = vblock * 8;
    if (threadIdx.x < 8) {
      const unsigned r = row_base + threadIdx.x;
      row_bounds[0][threadIdx.x] = r < rows ? min(starts_g[r], keys) : 0u;
      row_bounds[1][threadIdx.x] =
          r < rows ? max(min(ends_g[r], keys), row_bounds[0][threadIdx.x]) : 0u;
    }
    if (threadIdx.x == 0)
      fslot = 2 * blockIdx.x;  // no block is in flight during the rescan
    __syncthreads();
    const unsigned* const starts = row_bounds[0] - row_base;
    const unsigned* const ends = row_bounds[1] - row_base;
    const int64_t row_delta = int64_t(fslot) * 8 - int64_t(row_base);
    FallbackBuffers fb = fallback;
    fb.retained += row_delta * 4096;
    fb.indices += row_delta * 4096;
    fused_mqa_fallback::rescan_block(
        fused_mqa_dense::Scheduler(vblock, rows, keys, starts, ends), fb, topk,
        tq, tk, ts, tw);
    __syncthreads();
    if (threadIdx.x == 0) {
      auto& fst =
          *reinterpret_cast<fused_mqa_dense::SharedStorage*>(smem_buffer);
      for (auto& bb : fst.full_q_barriers) mbar_inval(&bb);
      for (auto& bb : fst.empty_q_barriers) mbar_inval(&bb);
      for (auto& bb : fst.full_kv_barriers) mbar_inval(&bb);
      for (auto& bb : fst.empty_kv_barriers) mbar_inval(&bb);
      for (auto& bb : fst.full_tmem_barriers) mbar_inval(&bb);
      for (auto& bb : fst.empty_tmem_barriers) mbar_inval(&bb);
    }
    __syncthreads();
    if (relative) {
      const unsigned quads = topk / 4;
      for (unsigned i = threadIdx.x; i < 8 * quads; i += 384) {
        const unsigned r = row_base + i / quads;
        if (r >= rows) continue;
        const int s = int(starts_g[r]);
        if (s == 0) continue;
        int4* p =
            reinterpret_cast<int4*>(fallback.output + (uint64_t)r * topk) +
            i % quads;
        int4 v = *p;
        v.x = v.x < 0 ? -1 : v.x - s;
        v.y = v.y < 0 ? -1 : v.y - s;
        v.z = v.z < 0 ? -1 : v.z - s;
        v.w = v.w < 0 ? -1 : v.w - s;
        *p = v;
      }
    }
    cur = unsigned(__ldcg(defer_next + vblock));
    __syncthreads();
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    defer_head = kTerm;
    pause_req = 0;
  }
}  // deferred blocks
more = epoch_kind != kTerm;
__syncthreads();
