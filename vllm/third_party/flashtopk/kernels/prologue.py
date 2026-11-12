# Copyright (c) 2026 the FlashTopK authors. Licensed under the Apache License, Version 2.0.
"""Small per-row kernels that would otherwise be a dozen torch operations a call, in CuTe DSL.

  * SampleWindows: each row's shared-sample window [lo, hi) from its KV window, and the call's health zeroed;
  * Decide: adaptive pruning's choice from the planner's estimate (pooled over the estimate's row groups, the share
    of a group's in-window sampled keys the bound keeps for some row of the group, against the threshold) as a device
    flag, and the windows the scan and the pruning kernels then take: the row's own for the path chosen, empty for
    the other.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass import Float32, Int32

from .common import select_i32
from .simt import THREADS, WARPS
from .selection import HEALTH_WORDS


@cute.jit
def samples_below(x, seq_len_kv, samples) -> Int32:
    """Shared-sample keys p_j = floor((2j + 1) L / (2M)) below x: p_j < x iff (2j + 1) L <= 2Mx - 1."""
    n = Int32(0)
    if x > 0:
        odd_max = (cutlass.Int64(2) * cutlass.Int64(samples) * cutlass.Int64(x) - 1) // cutlass.Int64(seq_len_kv)
        n = Int32(min((odd_max + 1) // 2, cutlass.Int64(samples)))
    return n


class SampleWindows:
    @cute.jit
    def __call__(self, ks: cute.Tensor, ke: cute.Tensor, lo: cute.Tensor, hi: cute.Tensor, health: cute.Tensor,
                 num_q: Int32, seq_len_kv: Int32, samples: Int32, stream: cuda.CUstream):
        self.kernel(ks, ke, lo, hi, health, num_q, seq_len_kv, samples).launch(
            grid=((num_q + THREADS - 1) // THREADS, 1, 1), block=[THREADS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, mKs, mKe, mLo, mHi, mHealth, num_q: Int32, seq_len_kv: Int32, samples: Int32):
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        if (bid == 0) & (tid < HEALTH_WORDS):
            mHealth[tid] = cutlass.Int64(0)
        r = bid * THREADS + tid
        if r < num_q:
            mLo[r] = samples_below(mKs[r], seq_len_kv, samples)
            mHi[r] = samples_below(mKe[r], seq_len_kv, samples)


EST_GROUP = 8  # the refine's row group (refine_sm90 / refine_sm100 BLOCK_Q)
EST_WORDS = 16  # an estimate row's in-window mask over the 256 sample keys (8 words), then its past-the-gate mask


class Decide:
    @cute.jit
    def __call__(self, est: cute.Tensor, ks: cute.Tensor, ke: cute.Tensor, scan_ks: cute.Tensor,
                 prune_ks: cute.Tensor, flag: cute.Tensor, estimate: cute.Tensor, num_q: Int32, threshold: Float32,
                 stream: cuda.CUstream):
        self.kernel(est, ks, ke, scan_ks, prune_ks, flag, estimate, num_q, threshold).launch(
            grid=((num_q + THREADS - 1) // THREADS, 1, 1), block=[THREADS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, mEst, mKs, mKe, mScanKs, mPruneKs, mFlag, mEstimate, num_q: Int32, threshold: Float32):
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        smem = utils.SmemAllocator()
        sWin = smem.allocate_tensor(Float32, cute.make_layout((WARPS,)), 16)
        sPass = smem.allocate_tensor(Float32, cute.make_layout((WARPS,)), 16)
        groups = mEst.layout.shape[0] // EST_GROUP
        half = EST_WORDS // 2
        # Each group's union, its rows' masks or-ed, counted over its in-window samples: the refine's work is a
        # group's union over its window, so the groups pool (counts are exact in FP32 below 2^24).
        n_win = Float32(0.0)
        n_pass = Float32(0.0)
        for g in cutlass.range(tid, groups, THREADS, unroll=1):
            for w in cutlass.range_constexpr(half):
                win = Int32(0)
                kept = Int32(0)
                for r in cutlass.range_constexpr(EST_GROUP):
                    win = win | mEst[(g * EST_GROUP + r, w)]
                    kept = kept | mEst[(g * EST_GROUP + r, half + w)]
                n_win = n_win + Float32(cute.arch.popc(win))
                n_pass = n_pass + Float32(cute.arch.popc(kept))
        n_win = cute.arch.warp_reduction_sum(n_win)
        n_pass = cute.arch.warp_reduction_sum(n_pass)
        if tid % 32 == 0:
            sWin[tid // 32] = n_win
            sPass[tid // 32] = n_pass
        cute.arch.sync_threads()
        n_win = Float32(0.0)
        n_pass = Float32(0.0)
        for w in cutlass.range_constexpr(WARPS):
            n_win = n_win + sWin[w]
            n_pass = n_pass + sPass[w]
        # Without in-window samples the call scans.
        share = Float32(1.0)
        if n_win > Float32(0.0):
            share = n_pass / n_win
        prune = share <= threshold
        if (bid == 0) & (tid == 0):
            mFlag[0] = select_i32(prune, Int32(1), Int32(0))
            mEstimate[0] = share
        r = bid * THREADS + tid
        if r < num_q:
            s = mKs[r]
            e = mKe[r]
            mScanKs[r] = select_i32(prune, e, s)
            mPruneKs[r] = select_i32(prune, s, e)
