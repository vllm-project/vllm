# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The QSA indexer's top-k block selection of one decode row, by one CTA:
the ``top`` largest of the row's ``visible`` scores (f32 >= 0), exact on the
f32 values, ties at the boundary to the lower block, as an ascending list of
block indices (stock's ``top_k_per_row_decode`` picks the same set but for
ties at the boundary, where its pick is not deterministic either). A row with
at most ``top`` visible blocks selects them all (``identity``), as stock does.

``select`` is a radix select on the keys, the scores' bits as int32 (monotone
for scores >= 0; -0.0 reads as 0): up to three levels of an LDS histogram
(key bits 30..19, then 18..7 and 6..0 of the keys still tied with the
boundary so far, counted with LDS atomics), each followed by a CTA-wide scan
of the bins from the top that finds the bin holding the ``top``-th key; a
level whose bin holds exactly the keys still wanted ends the search (the
bin is taken whole). A scan of each thread's strictly-greater and tied
counts (on the key bits the search resolved) then gives every chosen block
its slot. A thread owns a contiguous run of the row's keys; the keys are staged
in LDS (as many as the launch's LDS allows, ``lds_keys``) or, past that,
re-read from global memory on every pass (correct at any context, slower).
"""

from __future__ import annotations

import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr

from vllm.models.qwen4_exp.amd.specialized_decode.wide.common.ops import (
    CM_DEV,
    ld,
    lds_ld,
    lds_st,
    st,
    traced,
)

i32 = fx.Int32

THREADS = 256
WAVES = THREADS // 64
MAX_KEYS = 33792  # keys a row stages in LDS at most: 132K tokens at ratio 4
BINS = 4096  # a level's histogram (12 key bits)
PER_THREAD = BINS // THREADS  # bins a thread scans
# (shift, bits) of each level's digit: key bits 30..19, 18..7, 6..0
LEVELS = ((19, 12), (7, 12), (0, 7))
STEP = 4  # keys a thread reads at a time
UNROLLED = 8  # STEP-key reads of a thread's run in flight at once
AUX = 2 * WAVES + 4  # the scans' wave totals, then the found (bin, above, count)


def lds_keys(free_bytes: int) -> int:
    """Keys a row stages in LDS when ``free_bytes`` of LDS are left for
    ``select``: whole 1024-key steps up to ``MAX_KEYS``."""
    keys = (free_bytes // 4 - BINS - AUX) // 1024 * 1024
    return max(0, min(MAX_KEYS, keys))


def lds_words(keys: int) -> int:
    """i32 words of LDS ``select`` needs with ``keys`` staged keys: the keys,
    the histogram, ``AUX``."""
    return keys + BINS + AUX


@traced
def identity(sel, row, stride, visible, top: int, tid):
    """A row within the budget: blocks ``0 .. visible`` in order, -1 after."""
    for r in range_constexpr(-(-top // THREADS)):
        i = tid + i32(r * THREADS)
        if i < top:
            st(sel, row * stride + i, (i < visible).select(i, i32(-1)))


def _key(word):
    """The ranking key of a stored f32 score: its bits without the sign."""
    return fx.Int32(word) & i32(0x7FFFFFFF)


@traced
def _block_scan2(tid, aux, a, b):
    """Exclusive prefix sums of ``a`` and ``b`` over the CTA's threads in
    order, and ``a``'s total."""
    lane, wave = tid % 64, tid // 64
    xa, xb = a, b
    for step in range_constexpr(6):
        d = 1 << step
        xa = xa + (lane >= d).select(fx.shuffle_up(xa, i32(d), i32(64)), i32(0))
        xb = xb + (lane >= d).select(fx.shuffle_up(xb, i32(d), i32(64)), i32(0))
    if lane == 63:
        lds_st(aux + wave, xa)
        lds_st(aux + WAVES + wave, xb)
    fx.barrier()
    before_a, before_b, total_a = i32(0), i32(0), i32(0)
    for w in range_constexpr(WAVES):
        ta = lds_ld(aux + w)
        tb = lds_ld(aux + WAVES + w)
        before_a = before_a + (i32(w) < wave).select(ta, i32(0))
        before_b = before_b + (i32(w) < wave).select(tb, i32(0))
        total_a = total_a + ta
    fx.barrier()
    return before_a + xa - a, before_b + xb - b, total_a


@traced
def _count(hist, load4, lo, hi, prefix, shift: int, bits: int):
    """The thread's keys whose bits above ``shift + bits`` are ``prefix``,
    counted into ``hist`` by their next ``bits``."""
    mask = (1 << bits) - 1
    for j in range(lo, hi, i32(STEP)):
        k4 = load4(j)
        for e in range_constexpr(STEP):
            key = k4[e]
            hit = (j + i32(e) < hi) & ((key >> i32(shift + bits)) == prefix)
            if hit:
                digit = (key >> i32(shift)) & i32(mask)
                fx.llvm.atomic_add(hist + digit, i32(1), syncscope="workgroup")


@traced
def _find(tid, hist, aux, bins: int, want, clear: bool):
    """The bin holding the ``want``-th key from the top and the keys in the
    bins above it, (bin, above), from every thread. Thread ``t`` scans bins
    ``bins - (t + 1) * per .. bins - t * per`` (the top bins first);
    ``clear`` (``bins >= THREADS * STEP``): it zeroes them once read, for the
    next level."""
    per = max(1, bins // THREADS)
    first = i32(bins) - (tid + i32(1)) * i32(per)
    mine = []
    for b in range_constexpr(per):
        at = first + i32(b)
        mine.append((at >= i32(0)).select(lds_ld(hist + fx.max(at, i32(0))), i32(0)))
    if const_expr(clear):
        for b in range_constexpr(0, per, STEP):
            lds_st(
                hist + first + i32(b),
                fx.Vector.from_elements([i32(0)] * STEP, fx.Int32),
            )
    total = mine[0]
    for b in range_constexpr(1, per):
        total = total + mine[b]
    above, _, _ = _block_scan2(tid, aux, total, i32(0))
    if (above < want) & (above + total >= want):
        acc = above
        found, below, count = i32(-1), i32(0), i32(0)
        for b in range_constexpr(per):
            i = per - 1 - b
            hit = (found < i32(0)) & (acc + mine[i] >= want)
            found = hit.select(first + i32(i), found)
            below = hit.select(acc, below)
            count = hit.select(mine[i], count)
            acc = acc + mine[i]
        lds_st(aux + 2 * WAVES, found)
        lds_st(aux + 2 * WAVES + 1, below)
        lds_st(aux + 2 * WAVES + 2, count)
    fx.barrier()
    return (
        lds_ld(aux + 2 * WAVES),
        lds_ld(aux + 2 * WAVES + 1),
        lds_ld(aux + 2 * WAVES + 2),
    )


@traced
def _zero(tid, hist, bins: int):
    """The thread's share of zeroing ``bins`` histogram bins (no barrier)."""
    for r in range_constexpr(-(-bins // (THREADS * STEP))):
        at = (tid + i32(r * THREADS)) * i32(STEP)
        if at < i32(bins):
            lds_st(hist + at, fx.Vector.from_elements([i32(0)] * STEP, fx.Int32))


@traced
def _boundary(tid, hist, aux, load4, lo, hi, top: int):
    """(prefix, shift): the ``top``-th largest key's bits above ``shift``; the
    keys whose bits above ``shift`` exceed ``prefix`` and those equal to it
    taken in block order up to ``top`` are the top ``top``. ``hist``: a
    zeroed histogram; each level's scan zeroes it for the next (``_find``)."""
    prefix, want, shift_at, done = i32(0), i32(top), i32(31), i32(0)
    for level, (shift, bits) in enumerate(LEVELS):
        if done == i32(0):
            _count(hist, load4, lo, hi, prefix, shift, bits)
            fx.barrier()
            found, below, count = _find(
                tid, hist, aux, 1 << bits, want, level + 1 < len(LEVELS)
            )
            prefix = (prefix << i32(bits)) | found
            want = want - below
            shift_at = i32(shift)
            done = (count == want).select(i32(1), i32(0))
    return prefix, shift_at


@traced
def _emit(tid, aux, load4, lo, hi, prefix, shift, sel, row, stride, top: int):
    """Every key whose bits above ``shift`` exceed ``prefix`` and the first of
    those equal to it up to ``top`` in all, as ascending block indices at
    ``sel[row]``."""
    strict, ties = i32(0), i32(0)
    for j in range(lo, hi, i32(STEP)):
        k4 = [k >> shift for k in load4(j)]
        for e in range_constexpr(STEP):
            ok = j + i32(e) < hi
            strict = strict + (ok & (k4[e] > prefix)).select(i32(1), i32(0))
            ties = ties + (ok & (k4[e] == prefix)).select(i32(1), i32(0))
    strict_before, ties_before, strict_total = _block_scan2(tid, aux, strict, ties)
    wanted = i32(top) - strict_total
    take = fx.max(fx.min(wanted - ties_before, ties), i32(0))
    slot = strict_before + fx.min(ties_before, wanted)
    seen = i32(0)
    for j in range(lo, hi, i32(STEP)):
        k4 = [k >> shift for k in load4(j)]
        for e in range_constexpr(STEP):
            ok = j + i32(e) < hi
            tie = ok & (k4[e] == prefix)
            chosen = (ok & (k4[e] > prefix)) | (tie & (seen < take))
            if chosen:
                st(sel, row * stride + slot, j + i32(e))
            slot = slot + chosen.select(i32(1), i32(0))
            seen = seen + tie.select(i32(1), i32(0))


@traced
def _stage(keys, from_global, lo, hi):
    """The thread's keys into LDS, ``UNROLLED`` reads in flight at a time (no
    barrier)."""
    for j in range(lo, hi, i32(UNROLLED * STEP)):
        loads = [from_global(j + i32(u * STEP)) for u in range(UNROLLED)]
        for u in range_constexpr(UNROLLED):
            at = j + i32(u * STEP)
            if at < hi:
                lds_st(keys + at, fx.Vector.from_elements(loads[u], fx.Int32))


@traced
def select(
    tid, keys, staged: int, logits, row, logits_stride, visible, sel, stride, top: int
):
    """``sel[row]`` = the ``top`` best of ``logits[row, :visible]`` (``visible >
    top``), ascending. ``keys``: ``lds_words(staged)`` i32 words of LDS, the
    first ``staged`` for the row's keys. Reads the scores device-coherent
    (``CM_DEV``): other CTAs of the launch wrote them."""
    hist = keys + staged
    aux = hist + BINS
    # the thread's run: whole STEP-key reads, every thread's run contiguous
    per = (visible + i32(THREADS * STEP - 1)) // i32(THREADS * STEP) * i32(STEP)
    lo = tid * per
    hi = fx.min(lo + per, visible)
    base = row * logits_stride

    def from_global(j):
        # past the row's scores (a run's last read) the keys are never used
        w = ld(
            logits,
            base + fx.min(j, i32(logits_stride - STEP)),
            fx.Int32,
            STEP,
            cm=CM_DEV,
        )
        return [_key(w[e]) for e in range(STEP)]

    def from_lds(j):
        w = lds_ld(keys + j, STEP)
        return [w[e] for e in range(STEP)]

    _zero(tid, hist, BINS)
    if visible <= i32(staged):
        _stage(keys, from_global, lo, hi)
        fx.barrier()
        p, sh = _boundary(tid, hist, aux, from_lds, lo, hi, top)
        _emit(tid, aux, from_lds, lo, hi, p, sh, sel, row, stride, top)
    else:
        fx.barrier()
        p, sh = _boundary(tid, hist, aux, from_global, lo, hi, top)
        _emit(tid, aux, from_global, lo, hi, p, sh, sel, row, stride, top)
