# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright (c) 2026 the FlashTopK authors.
# Licensed under the Apache License, Version 2.0.
#
# The radix selection follows LiteTopK's compact top-k kernel
# (https://github.com/Heisenberg-Yin/LiteTopK, Apache 2.0).
"""Selection: each row's exact top-k over its candidate buffer, with the call's health,
in CuTe DSL.

One CTA per row (256 threads). The candidates are (-score, kv index) pairs, the scan's
or the refine's:

  * the k smallest values win: four radix passes over the values' ordered bits find the
    k-th smallest (the pivot) and how many lie below it; when more candidates share the
    pivot than slots remain, four more passes over their indices find the cutoff that
    keeps the smallest indices, so the selected set is unique and deterministic;
  * winners are compacted into unordered output slots (padded with -1 when fewer than k
    keys are visible); scores are written only for multi-split dense recovery stages;
  * health: a row whose count passed the buffer (overflow) or that selected fewer than
    min(k, window) keys (underfill) is bad; the first RECOVERY_ROWS bad rows are
    recorded for recovery.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass import Int32

from .common import select_i32
from .simt import (
    THREADS,
    WARPS,
    ascending_key,
    atomic_add_global_u64,
    atomic_add_shared_u32,
    bits_f32,
    block_inclusive_scan_i32,
    block_sum_i32,
    finite,
    red_add_shared_u32,
)

RADIX = 256
RECOVERY_ROWS = 64
HEALTH_WORDS = 4
H_OVERFLOW, H_UNDERFILL, H_BAD = 0, 1, 2
_NEG_INF_BITS = 0xFF800000
_UNROLL = 4
CANDIDATES, DENSE, STAGED = 0, 1, 2
MIN_BLOCKS = 6


@cute.jit
def load_value(mVal, row, col, source: cutlass.Constexpr):
    """The entry as -score, the order the selection ranks in."""
    v = mVal[(row, col)]
    if cutlass.const_expr(source != CANDIDATES):
        v = -v
    return v


@cute.jit
def load_index(mIdx, row, col, source: cutlass.Constexpr) -> Int32:
    index = col
    if cutlass.const_expr(source != DENSE):
        index = mIdx[(row, col)]
    return index


@cute.jit
def radix_bin(sHist, sScan, sState, tid, kfind):
    """The bin holding the kfind-th smallest key (acc < kfind <= acc + hist[bin]): its
    owner thread writes (bin, acc, count) to sState[0..2]."""
    h = sHist[tid]
    incl, _ = block_inclusive_scan_i32(h, sScan, tid)
    acc = incl - h
    if (acc < kfind) & (kfind <= incl):
        sState[0] = tid
        sState[1] = acc
        sState[2] = h


class Select:
    """source CANDIDATES: one CTA per row of the call, with the health. DENSE: `splits`
    CTAs per slot, each the top-k of its slice of the slot's window of dense scores,
    written to output row map[slot] when splits is 1 and to its own row of the stage
    buffers (with write_scores=True) otherwise. STAGED: one CTA per slot, over the
    slot's stage rows. A slot whose map entry is negative is skipped."""

    def __init__(self, topk: int, source: int = CANDIDATES, write_scores: bool = False):
        self.topk = topk
        self.source = source
        self.write_scores = write_scores

    @cute.jit
    def __call__(
        self,
        values: cute.Tensor,
        idx: cute.Tensor,
        cand_cnt: cute.Tensor,
        ks: cute.Tensor,
        ke: cute.Tensor,
        row_map: cute.Tensor,
        scores: cute.Tensor,
        indices: cute.Tensor,
        health: cute.Tensor,
        bad_rows: cute.Tensor,
        num_ctas: Int32,
        splits: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            values,
            idx,
            cand_cnt,
            ks,
            ke,
            row_map,
            scores,
            indices,
            health,
            bad_rows,
            splits,
        ).launch(
            grid=(num_ctas, 1, 1),
            block=[THREADS, 1, 1],
            min_blocks_per_mp=MIN_BLOCKS,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mVal,
        mIdx,
        mCnt,
        mKs,
        mKe,
        mMap,
        mScores,
        mIndices,
        mHealth,
        mBad,
        splits: Int32,
    ):
        K = self.topk
        src = self.source
        write_scores = self.write_scores
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        smem = utils.SmemAllocator()
        sHist = smem.allocate_tensor(Int32, cute.make_layout((RADIX,)), 16)
        sScan = smem.allocate_tensor(Int32, cute.make_layout((WARPS,)), 16)
        sState = smem.allocate_tensor(Int32, cute.make_layout((4,)), 16)
        hist = sHist.iterator.toint()
        state = sState.iterator.toint()

        cap = mVal.layout.shape[1]
        row = bid
        out_row = bid
        lo = Int32(0)
        n = cap
        raw_n = Int32(0)
        target = Int32(0)
        active = True
        if cutlass.const_expr(src == CANDIDATES):
            raw_n = mCnt[row]
            n = min(max(raw_n, Int32(0)), cap)
        elif cutlass.const_expr(src == DENSE):
            row = bid // splits
            part = bid - row * splits
            target = mMap[row]
            active = target >= 0
            out_row = select_i32(splits > 1, bid, target)
            start = max(Int32(0), min(cap, mKs[row]))
            end = max(start, min(cap, mKe[row]))
            chunk = (end - start + splits - 1) // splits
            lo = min(end, start + part * chunk)
            n = min(end, lo + chunk) - lo
        else:
            target = mMap[row]
            active = target >= 0
            out_row = target

        if active:
            sHist[tid] = Int32(0)
            cute.arch.sync_threads()
            fin = Int32(0)
            for base in cutlass.range(0, n, THREADS * _UNROLL, unroll=1):
                vs = [
                    load_value(
                        mVal, row, lo + min(base + u * THREADS + tid, n - 1), src
                    )
                    for u in range(_UNROLL)
                ]
                for u in cutlass.range_constexpr(_UNROLL):
                    if (base + u * THREADS + tid < n) & finite(vs[u]):
                        fin = fin + 1
                        red_add_shared_u32(
                            hist
                            + Int32(ascending_key(vs[u]) >> cutlass.Uint32(24)) * 4,
                            Int32(1),
                        )
            total = block_sum_i32(fin, sScan, tid)
            target = min(total, Int32(K))

            pivot = cutlass.Uint32(0xFFFFFFFF)
            tie_cut = cutlass.Uint32(0xFFFFFFFF)
            if total > K:
                desired = cutlass.Uint32(0)
                mask = cutlass.Uint32(0)
                kfind = target
                below = Int32(0)
                equal = Int32(0)
                for p in cutlass.range_constexpr(4):
                    shift = 24 - 8 * p
                    if p > 0:
                        sHist[tid] = Int32(0)
                        cute.arch.sync_threads()
                        for base in cutlass.range(0, n, THREADS * _UNROLL, unroll=1):
                            vs = [
                                load_value(
                                    mVal,
                                    row,
                                    lo + min(base + u * THREADS + tid, n - 1),
                                    src,
                                )
                                for u in range(_UNROLL)
                            ]
                            for u in cutlass.range_constexpr(_UNROLL):
                                e = ascending_key(vs[u])
                                if (
                                    (base + u * THREADS + tid < n)
                                    & finite(vs[u])
                                    & ((e & mask) == desired)
                                ):
                                    b = Int32(
                                        (e >> cutlass.Uint32(shift))
                                        & cutlass.Uint32(0xFF)
                                    )
                                    red_add_shared_u32(hist + b * 4, Int32(1))
                        cute.arch.sync_threads()
                    radix_bin(sHist, sScan, sState, tid, kfind)
                    cute.arch.sync_threads()
                    desired = desired | (
                        cutlass.Uint32(sState[0]) << cutlass.Uint32(shift)
                    )
                    kfind = kfind - sState[1]
                    below = below + sState[1]
                    equal = sState[2]
                    mask = mask | (cutlass.Uint32(0xFF) << cutlass.Uint32(shift))
                    cute.arch.sync_threads()
                pivot = desired
                eq_take = target - below
                if equal > eq_take:
                    idesired = cutlass.Uint32(0)
                    imask = cutlass.Uint32(0)
                    ikfind = eq_take
                    for p in cutlass.range_constexpr(4):
                        shift = 24 - 8 * p
                        sHist[tid] = Int32(0)
                        cute.arch.sync_threads()
                        for j in cutlass.range(tid, n, THREADS, unroll=1):
                            v = load_value(mVal, row, lo + j, src)
                            if finite(v):  # noqa: SIM102
                                if ascending_key(v) == pivot:
                                    ik = cutlass.Uint32(
                                        load_index(mIdx, row, lo + j, src)
                                    )
                                    if (ik & imask) == idesired:
                                        b = Int32(
                                            (ik >> cutlass.Uint32(shift))
                                            & cutlass.Uint32(0xFF)
                                        )
                                        red_add_shared_u32(hist + b * 4, Int32(1))
                        cute.arch.sync_threads()
                        radix_bin(sHist, sScan, sState, tid, ikfind)
                        cute.arch.sync_threads()
                        idesired = idesired | (
                            cutlass.Uint32(sState[0]) << cutlass.Uint32(shift)
                        )
                        ikfind = ikfind - sState[1]
                        imask = imask | (cutlass.Uint32(0xFF) << cutlass.Uint32(shift))
                        cute.arch.sync_threads()
                    tie_cut = idesired

            if tid == 0:
                sState[3] = Int32(0)
            cute.arch.sync_threads()
            valid = Int32(0)
            lane = tid % 32
            lt_mask = (cutlass.Uint32(1) << cutlass.Uint32(lane)) - cutlass.Uint32(1)
            for base in cutlass.range(0, n, THREADS * _UNROLL, unroll=1):
                vs = [
                    load_value(
                        mVal, row, lo + min(base + u * THREADS + tid, n - 1), src
                    )
                    for u in range(_UNROLL)
                ]
                xs = [
                    load_index(
                        mIdx, row, lo + min(base + u * THREADS + tid, n - 1), src
                    )
                    for u in range(_UNROLL)
                ]
                takes = []
                ballots = []
                reserve = Int32(0)
                for u in cutlass.range_constexpr(_UNROLL):
                    e = ascending_key(vs[u])
                    take = (
                        (base + u * THREADS + tid < n)
                        & finite(vs[u])
                        & (
                            (e < pivot)
                            | ((e == pivot) & (cutlass.Uint32(xs[u]) <= tie_cut))
                        )
                    )
                    ballot = cutlass.Uint32(cute.arch.vote_ballot_sync(take))
                    takes.append(take)
                    ballots.append(ballot)
                    reserve = reserve + Int32(cute.arch.popc(ballot))
                first = Int32(0)
                if (lane == 0) & (reserve > 0):
                    first = atomic_add_shared_u32(state + 3 * 4, reserve)
                first = cute.arch.shuffle_sync(first, 0)
                for u in cutlass.range_constexpr(_UNROLL):
                    slot = first + Int32(cute.arch.popc(ballots[u] & lt_mask))
                    if takes[u] & (slot < K):
                        mIndices[(out_row, slot)] = xs[u]
                        if cutlass.const_expr(write_scores):
                            mScores[(out_row, slot)] = -vs[u]
                        if xs[u] >= 0:
                            valid = valid + 1
                    first = first + Int32(cute.arch.popc(ballots[u]))
            valid = block_sum_i32(valid, sScan, tid)
            written = min(sState[3], Int32(K))
            for i in cutlass.range(written + tid, K, THREADS, unroll=1):
                if cutlass.const_expr(write_scores):
                    mScores[(out_row, i)] = bits_f32(cutlass.Uint32(_NEG_INF_BITS))
                mIndices[(out_row, i)] = Int32(-1)

            if cutlass.const_expr(src == CANDIDATES):  # noqa: SIM102
                if tid == 0:
                    expected = min(max(mKe[row] - mKs[row], Int32(0)), Int32(K))
                    health = mHealth.iterator.toint()
                    overflow = raw_n > cap
                    underfill = valid < expected
                    if overflow:
                        atomic_add_global_u64(health + H_OVERFLOW * 8, cutlass.Int64(1))
                    if underfill:
                        atomic_add_global_u64(
                            health + H_UNDERFILL * 8, cutlass.Int64(1)
                        )
                    if overflow | underfill:
                        bad_slot = atomic_add_global_u64(
                            health + H_BAD * 8, cutlass.Int64(1)
                        )
                        if bad_slot < RECOVERY_ROWS:
                            mBad[Int32(bad_slot)] = row
