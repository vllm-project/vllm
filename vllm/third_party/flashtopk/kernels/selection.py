# Copyright (c) 2026 the FlashTopK authors. Licensed under the Apache License, Version 2.0.
#
# The radix selection follows LiteTopK's compact top-k kernel (https://github.com/Heisenberg-Yin/LiteTopK, Apache 2.0).
"""Selection: each row's exact top-k over its candidate buffer, in canonical order, with the call's health, in CuTe DSL.

One CTA per row (256 threads). The candidates are (-score, kv index) pairs, the scan's or the refine's:

  * the k smallest values win: four radix passes over the values' ordered bits find the k-th smallest (the pivot)
    and how many lie below it; when more candidates share the pivot than slots remain, four more passes over their
    indices find the cutoff that keeps the smallest indices, so the selected set is unique and reproducible;
  * canonical order (score descending, index ascending; -0.0 ties +0.0; padding (-inf, -1) last), by a counting sort:
    the selected keys go to buckets spaced logarithmically above the worst selected score, a few keys each whatever
    the scores' scale or sign (a linear split of their range let one outlier, or scores either side of zero, put
    nearly every key in one bucket), and each key's place is its bucket's offset plus its bucket's keys below it. A
    row the buckets cannot split (equal scores) takes a sample sort instead. The output is the same bits on every
    run;
  * health: a row whose count passed the buffer (overflow) or that selected fewer than min(k, window) keys
    (underfill) is bad; the first RECOVERY_ROWS bad rows are recorded for recovery.

Health words (int64): 0 overflow rows, 1 underfill rows, 3 the largest candidate count (past the buffer, or always
with stats), 4 the candidate total (with stats), 7 bad rows.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass import Float32, Int32

from .common import add_rn, mul_rn, select_i32, sub_rn
from .simt import THREADS, WARPS, ascending_key, atomic_add_global_u64, atomic_add_shared_u32, \
    atomic_max_global_u64, atomic_max_shared_u32, atomic_min_shared_u32, bit_length_u32, bits_f32, \
    block_inclusive_scan_i32, block_max_i32, block_sum_i32, f32_bits, finite, red_add_shared_u32

RADIX = 256
LOG_BUCKETS = 2048
SORT_BUCKETS = LOG_BUCKETS  # counters, scanned LOG_BUCKETS // THREADS a thread (the sample sort uses THREADS + 1)
# A row whose fullest log bucket holds more keys falls back to the sample sort (each key's place costs a pass over
# its bucket).
BUCKET_LIMIT = 48
RECOVERY_ROWS = 64
HEALTH_WORDS = 8
H_OVERFLOW, H_UNDERFILL, H_CAND_MAX, H_CAND_SUM, H_BAD = 0, 1, 3, 4, 7
_NEG_INF_BITS = 0xFF800000
_UNROLL = 4
# What a CTA selects from: a row's candidate buffer (the scan's -score, kv index pairs), a slice of a row of dense
# scores (recovery; the index is the column), or the per-slice selections of a row (recovery's second stage).
CANDIDATES, DENSE, STAGED = 0, 1, 2
# Six CTAs an SM: the register budget that keeps the sweeps' loads in flight without spilling the sort (measured on
# SM100 against four and eight).
MIN_BLOCKS = 6


@cute.jit
def order_key(v, index) -> cutlass.Uint64:
    """Ascending order of this key is canonical order of (-score v, index >= 0): v ascending with -0.0 tying +0.0,
    then the index ascending. The lowest bit keeps a zero's sign, so the key alone gives back v's bits."""
    bits = f32_bits(v)
    e = ascending_key(v)
    negative_zero = cutlass.Uint32(0)
    if (bits & cutlass.Uint32(0x7FFFFFFF)) == cutlass.Uint32(0):
        e = cutlass.Uint32(0x80000000)
        negative_zero = bits >> cutlass.Uint32(31)
    low = (cutlass.Uint32(index) << cutlass.Uint32(1)) | negative_zero
    return (cutlass.Uint64(e) << cutlass.Uint64(32)) | cutlass.Uint64(low)


@cute.jit
def order_index(key) -> Int32:
    return Int32(cutlass.Uint32(key & cutlass.Uint64(0xFFFFFFFF)) >> cutlass.Uint32(1))


@cute.jit
def order_score_bits(key) -> cutlass.Uint32:
    """The score (-v) whose order_key this is."""
    e = cutlass.Uint32(key >> cutlass.Uint64(32))
    v = e ^ cutlass.Uint32(0xFFFFFFFF)
    if (e & cutlass.Uint32(0x80000000)) != cutlass.Uint32(0):
        v = e ^ cutlass.Uint32(0x80000000)
    if (cutlass.Uint32(key & cutlass.Uint64(1))) != cutlass.Uint32(0):
        v = cutlass.Uint32(0x80000000)
    return v ^ cutlass.Uint32(0x80000000)


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
def order_high_score(high) -> Float32:
    """The score whose order_key has this high word (a zero comes back as -0.0)."""
    v = high ^ cutlass.Uint32(0xFFFFFFFF)
    if (high & cutlass.Uint32(0x80000000)) != cutlass.Uint32(0):
        v = high ^ cutlass.Uint32(0x80000000)
    return bits_f32(v ^ cutlass.Uint32(0x80000000))


@cute.jit
def bitonic_sort_asc(sA, tid):
    """Sort sA[0:THREADS] ascending, one element a thread; every thread must reach this."""
    for size_log in cutlass.range_constexpr(1, THREADS.bit_length()):
        size = 1 << size_log
        for j in cutlass.range_constexpr(size_log):
            stride = 1 << (size_log - 1 - j)
            partner = tid ^ stride
            if partner > tid:
                a = sA[tid]
                b = sA[partner]
                if (a > b) == ((tid & size) == 0):
                    sA[tid] = b
                    sA[partner] = a
            cute.arch.sync_threads()


@cute.jit
def radix_bin(sHist, sScan, sState, tid, kfind):
    """The bin holding the kfind-th smallest key (acc < kfind <= acc + hist[bin]): its owner thread writes (bin, acc,
    count) to sState[0..2]. Every thread must reach this; the caller syncs before reading sState."""
    h = sHist[tid]
    incl, _ = block_inclusive_scan_i32(h, sScan, tid)
    acc = incl - h
    if (acc < kfind) & (kfind <= incl):
        sState[0] = tid
        sState[1] = acc
        sState[2] = h


class Select:
    """source CANDIDATES: one CTA per row of the call, with the health. DENSE: `splits` CTAs per slot, each the top-k
    of its slice of the slot's window of dense scores, written to output row map[slot] when splits is 1 and to its
    own row of the stage buffers otherwise. STAGED: one CTA per slot, over the slot's stage rows. A slot whose map
    entry is negative is skipped."""

    def __init__(self, topk: int, source: int = CANDIDATES):
        self.topk = topk
        self.source = source

    @cute.jit
    def __call__(
        self,
        values: cute.Tensor,  # [rows, cap] f32: -score (CANDIDATES) or scores
        idx: cute.Tensor,  # [rows, cap] i32 (unused for DENSE)
        cand_cnt: cute.Tensor,  # [rows] i32: candidates appended, may exceed cap (CANDIDATES)
        ks: cute.Tensor,  # [rows] window starts
        ke: cute.Tensor,
        row_map: cute.Tensor,  # [slots] i32: output row, or -1 (DENSE, STAGED)
        scores: cute.Tensor,  # [out rows, topk] f32 out
        indices: cute.Tensor,  # [out rows, topk] i32 out
        health: cute.Tensor,  # [HEALTH_WORDS] i64, accumulated (CANDIDATES)
        bad_rows: cute.Tensor,  # [RECOVERY_ROWS] i32 out (CANDIDATES)
        num_ctas: Int32,
        splits: Int32,
        stats: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(values, idx, cand_cnt, ks, ke, row_map, scores, indices, health, bad_rows, splits, stats).launch(
            grid=(num_ctas, 1, 1), block=[THREADS, 1, 1], min_blocks_per_mp=MIN_BLOCKS, stream=stream
        )

    @cute.kernel
    def kernel(self, mVal, mIdx, mCnt, mKs, mKe, mMap, mScores, mIndices, mHealth, mBad, splits: Int32, stats: Int32):
        K = self.topk
        src = self.source
        items = (K + THREADS - 1) // THREADS
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        smem = utils.SmemAllocator()
        sHist = smem.allocate_tensor(Int32, cute.make_layout((RADIX,)), 16)
        sScan = smem.allocate_tensor(Int32, cute.make_layout((WARPS,)), 16)
        # 0..2 radix_bin, 3 the write counter, 4 / 5 the selected keys' smallest / largest high word
        sState = smem.allocate_tensor(Int32, cute.make_layout((8,)), 16)
        sKey = smem.allocate_tensor(cutlass.Uint64, cute.make_layout((K,)), 16)
        sCount = smem.allocate_tensor(Int32, cute.make_layout((SORT_BUCKETS + 1,)), 16)
        sSplit = smem.allocate_tensor(cutlass.Uint64, cute.make_layout((THREADS,)), 16)
        sPlace = smem.allocate_tensor(Int32, cute.make_layout((K,)), 16)  # each slot's (bucket << 16) | rank
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
            # pass 0 of the value radix, with the finite count. Every sweep issues _UNROLL loads before it uses one: the
            # candidates come from L2, and the shared-memory atomics would otherwise keep them one at a time.
            sHist[tid] = Int32(0)
            cute.arch.sync_threads()
            fin = Int32(0)
            for base in cutlass.range(0, n, THREADS * _UNROLL, unroll=1):
                vs = [load_value(mVal, row, lo + min(base + u * THREADS + tid, n - 1), src) for u in range(_UNROLL)]
                for u in cutlass.range_constexpr(_UNROLL):
                    if (base + u * THREADS + tid < n) & finite(vs[u]):
                        fin = fin + 1
                        red_add_shared_u32(hist + Int32(ascending_key(vs[u]) >> cutlass.Uint32(24)) * 4, Int32(1))
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
                            vs = [load_value(mVal, row, lo + min(base + u * THREADS + tid, n - 1), src)
                                  for u in range(_UNROLL)]
                            for u in cutlass.range_constexpr(_UNROLL):
                                e = ascending_key(vs[u])
                                if (base + u * THREADS + tid < n) & finite(vs[u]) & ((e & mask) == desired):
                                    b = Int32((e >> cutlass.Uint32(shift)) & cutlass.Uint32(0xFF))
                                    red_add_shared_u32(hist + b * 4, Int32(1))
                        cute.arch.sync_threads()
                    radix_bin(sHist, sScan, sState, tid, kfind)
                    cute.arch.sync_threads()
                    desired = desired | (cutlass.Uint32(sState[0]) << cutlass.Uint32(shift))
                    kfind = kfind - sState[1]
                    below = below + sState[1]
                    equal = sState[2]
                    mask = mask | (cutlass.Uint32(0xFF) << cutlass.Uint32(shift))
                    cute.arch.sync_threads()
                pivot = desired
                eq_take = target - below
                if equal > eq_take:
                    # More candidates share the pivot than slots remain: keep the smallest indices.
                    idesired = cutlass.Uint32(0)
                    imask = cutlass.Uint32(0)
                    ikfind = eq_take
                    for p in cutlass.range_constexpr(4):
                        shift = 24 - 8 * p
                        sHist[tid] = Int32(0)
                        cute.arch.sync_threads()
                        for j in cutlass.range(tid, n, THREADS, unroll=1):
                            v = load_value(mVal, row, lo + j, src)
                            if finite(v):
                                if ascending_key(v) == pivot:
                                    ik = cutlass.Uint32(load_index(mIdx, row, lo + j, src))
                                    if (ik & imask) == idesired:
                                        b = Int32((ik >> cutlass.Uint32(shift)) & cutlass.Uint32(0xFF))
                                        red_add_shared_u32(hist + b * 4, Int32(1))
                        cute.arch.sync_threads()
                        radix_bin(sHist, sScan, sState, tid, ikfind)
                        cute.arch.sync_threads()
                        idesired = idesired | (cutlass.Uint32(sState[0]) << cutlass.Uint32(shift))
                        ikfind = ikfind - sState[1]
                        imask = imask | (cutlass.Uint32(0xFF) << cutlass.Uint32(shift))
                        cute.arch.sync_threads()
                    tie_cut = idesired

            # the selected pairs into shared memory, with the best and worst of their scores
            if tid == 0:
                sState[3] = Int32(0)
                sState[4] = Int32(-1)
                sState[5] = Int32(0)
            cute.arch.sync_threads()
            key_lo = cutlass.Uint32(0xFFFFFFFF)
            key_hi = cutlass.Uint32(0)
            valid = Int32(0)
            lane = tid % 32
            lt_mask = (cutlass.Uint32(1) << cutlass.Uint32(lane)) - cutlass.Uint32(1)
            # The trip count is the block's, so whole warps reserve their slots with one atomic.
            for base in cutlass.range(0, n, THREADS * _UNROLL, unroll=1):
                vs = [load_value(mVal, row, lo + min(base + u * THREADS + tid, n - 1), src) for u in range(_UNROLL)]
                xs = [load_index(mIdx, row, lo + min(base + u * THREADS + tid, n - 1), src) for u in range(_UNROLL)]
                takes = []
                ballots = []
                reserve = Int32(0)
                for u in cutlass.range_constexpr(_UNROLL):
                    e = ascending_key(vs[u])
                    take = (base + u * THREADS + tid < n) & finite(vs[u]) & (
                        (e < pivot) | ((e == pivot) & (cutlass.Uint32(xs[u]) <= tie_cut)))
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
                        key = order_key(vs[u], xs[u])
                        sKey[slot] = key
                        high = cutlass.Uint32(key >> cutlass.Uint64(32))
                        key_lo = min(key_lo, high)
                        key_hi = max(key_hi, high)
                        if xs[u] >= 0:
                            valid = valid + 1
                    first = first + Int32(cute.arch.popc(ballots[u]))
            atomic_min_shared_u32(state + 4 * 4, key_lo)
            atomic_max_shared_u32(state + 5 * 4, key_hi)
            for b in cutlass.range(tid, SORT_BUCKETS + 1, THREADS, unroll=1):
                sCount[b] = Int32(0)
            valid = block_sum_i32(valid, sScan, tid)
            written = min(sState[3], Int32(K))

            # Buckets spaced logarithmically above the worst selected score: the bits of t + eps, t = s - worst and
            # eps = span * 2^-13, are monotone in s and close to log2(t + eps), so the buckets follow the scores'
            # spread whatever its scale or sign, a few keys each. Every bucket's keys sort after the next better
            # bucket's.
            best = order_high_score(cutlass.Uint32(sState[4]))
            worst = order_high_score(cutlass.Uint32(sState[5]))
            span = sub_rn(best, worst)
            eps = mul_rn(span, Float32(2.0**-13))
            base = f32_bits(eps)
            shift = max(bit_length_u32(f32_bits(add_rn(span, eps)) - base) - 11, Int32(0))
            spread = finite(span) & (span > Float32(0.0))
            for i in cutlass.range_constexpr(items):
                slot = tid + i * THREADS
                if slot < written:
                    s_key = order_high_score(cutlass.Uint32(sKey[slot] >> cutlass.Uint64(32)))
                    up = Int32((f32_bits(add_rn(sub_rn(s_key, worst), eps)) - base) >> cutlass.Uint32(shift))
                    d = Int32(LOG_BUCKETS - 1) - min(max(up, Int32(0)), Int32(LOG_BUCKETS - 1))
                    d = select_i32(spread, d, Int32(0))
                    rank = atomic_add_shared_u32(sCount.iterator.toint() + d * 4, Int32(1))
                    sPlace[slot] = (d << 16) | rank
            cute.arch.sync_threads()
            fullest = Int32(0)
            for u in cutlass.range_constexpr(LOG_BUCKETS // THREADS):
                fullest = max(fullest, sCount[tid * (LOG_BUCKETS // THREADS) + u])
            fullest = block_max_i32(fullest, sScan, tid)
            if fullest > BUCKET_LIMIT:
                # Scores the log buckets cannot split (equal ones, mostly): a sample sort instead. THREADS splitters
                # taken evenly over the arrival order and sorted (keys are distinct, so equal scores spread by index),
                # then each key's bucket is how many splitters lie below it.
                for b in cutlass.range(tid, SORT_BUCKETS + 1, THREADS, unroll=1):
                    sCount[b] = Int32(0)
                splitter = cutlass.Uint64(0)
                if written > 0:
                    splitter = sKey[min((tid * written) // THREADS, Int32(K - 1))]
                sSplit[tid] = splitter
                cute.arch.sync_threads()
                bitonic_sort_asc(sSplit, tid)
                for i in cutlass.range_constexpr(items):
                    slot = tid + i * THREADS
                    if slot < written:
                        key = sKey[slot]
                        pos = Int32(0)
                        for j in cutlass.range_constexpr(THREADS.bit_length() - 1):
                            half = THREADS >> (j + 1)
                            if sSplit[pos + half - 1] < key:
                                pos = pos + half
                        if sSplit[pos] < key:
                            pos = pos + 1
                        rank = atomic_add_shared_u32(sCount.iterator.toint() + pos * 4, Int32(1))
                        sPlace[slot] = (pos << 16) | rank
                cute.arch.sync_threads()
            keys = []
            digits = []
            ranks = []
            for i in cutlass.range_constexpr(items):
                slot = tid + i * THREADS
                at = min(slot, Int32(K - 1))
                packed = sPlace[at]
                keys.append(sKey[at])
                digits.append(select_i32(slot < written, packed >> 16, Int32(SORT_BUCKETS)))
                ranks.append(packed & 0xFFFF)
            # bucket offsets: an exclusive scan of the counts, SORT_BUCKETS // THREADS buckets a thread
            per = SORT_BUCKETS // THREADS
            counts = [sCount[tid * per + u] for u in range(per)]
            mine = Int32(0)
            for u in cutlass.range_constexpr(per):
                mine = mine + counts[u]
            incl, _ = block_inclusive_scan_i32(mine, sScan, tid)
            run = incl - mine
            for u in cutlass.range_constexpr(per):
                sCount[tid * per + u] = run
                run = run + counts[u]
            if tid == THREADS - 1:
                sCount[SORT_BUCKETS] = incl
            cute.arch.sync_threads()
            # the keys grouped by bucket (every key was read above)
            for i in cutlass.range_constexpr(items):
                if digits[i] < SORT_BUCKETS:
                    sKey[sCount[digits[i]] + ranks[i]] = keys[i]
            cute.arch.sync_threads()
            # each key's place: its bucket's offset plus the bucket's keys below it
            for i in cutlass.range_constexpr(items):
                if digits[i] < SORT_BUCKETS:
                    first = sCount[digits[i]]
                    last = sCount[digits[i] + 1]
                    place = first
                    for q in cutlass.range(first, last, unroll=1):
                        if sKey[q] < keys[i]:
                            place = place + 1
                    mScores[(out_row, place)] = bits_f32(order_score_bits(keys[i]))
                    mIndices[(out_row, place)] = order_index(keys[i])
            for i in cutlass.range(written + tid, K, THREADS, unroll=1):
                mScores[(out_row, i)] = bits_f32(cutlass.Uint32(_NEG_INF_BITS))
                mIndices[(out_row, i)] = Int32(-1)

            if cutlass.const_expr(src == CANDIDATES):
                if tid == 0:
                    expected = min(max(mKe[row] - mKs[row], Int32(0)), Int32(K))
                    health = mHealth.iterator.toint()
                    overflow = raw_n > cap
                    underfill = valid < expected
                    if overflow:
                        atomic_add_global_u64(health + H_OVERFLOW * 8, cutlass.Int64(1))
                    if underfill:
                        atomic_add_global_u64(health + H_UNDERFILL * 8, cutlass.Int64(1))
                    if overflow | underfill:
                        bad_slot = atomic_add_global_u64(health + H_BAD * 8, cutlass.Int64(1))
                        if bad_slot < RECOVERY_ROWS:
                            mBad[Int32(bad_slot)] = row
                    candidates = cutlass.Int64(max(raw_n, Int32(0)))
                    if (stats != 0) | overflow:
                        atomic_max_global_u64(health + H_CAND_MAX * 8, candidates)
                    if stats != 0:
                        atomic_add_global_u64(health + H_CAND_SUM * 8, candidates)
