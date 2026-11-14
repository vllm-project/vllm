# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact two-query KV unions, preserving duplicate-index multiplicities.

Query-union prior art: vllm-project/vllm#55372.
"""

from vllm.triton_utils import tl, triton


@triton.jit
def _max(a, b):
    return tl.maximum(a, b)


@triton.jit
def _segment(ka, ca, da, kb, cb, db):
    same = ka == kb
    return kb, tl.where(same, ca + cb, cb), tl.where(same, da + db, db)


@triton.jit(do_not_specialize=["T"])
def _pairs(req, valid, rows, count, T, B: tl.constexpr):
    i = tl.arange(0, B)
    r = tl.load(req + i, i < T, other=-1)
    v = tl.load(valid + i, i < T, other=False) & (r >= 0)
    prev_r = tl.load(req + i - 1, (i > 0) & (i < T), other=-1)
    prev_v = tl.load(valid + i - 1, (i > 0) & (i < T), other=False)
    run = v & (~prev_v | (prev_r != r))
    first = tl.associative_scan(tl.where(run, i, 0), 0, _max)
    start = v & ((i - first) % 2 == 0)
    group = tl.cumsum(start.to(tl.int32)) - 1
    next_r = tl.load(req + i + 1, i + 1 < T, other=-1)
    next_v = tl.load(valid + i + 1, i + 1 < T, other=False)
    second = tl.where(next_v & (next_r == r), i + 1, -1)
    tl.store(rows + group * 2, i, start)
    tl.store(rows + group * 2 + 1, second, start)
    tl.store(count, tl.sum(start.to(tl.int32)))


@triton.jit
def _union(
    rows,
    count,
    ix,
    lengths,
    out_ix,
    out_counts,
    out_lens,
    STRIDE: tl.constexpr,
    WIDTH: tl.constexpr,
    HALF: tl.constexpr,
    BOUNDED: tl.constexpr = False,
):
    g = tl.program_id(0)
    groups = tl.load(count)
    if g >= groups:
        tl.store(out_lens + g, 0)
        return
    second = tl.load(rows + 2 * g + 1)
    if second < 0:
        first = tl.load(rows + 2 * g)
        size = tl.minimum(tl.load(lengths + first), WIDTH)
        col_single = tl.arange(0, HALF)
        keys_single = tl.load(
            ix + first.to(tl.int64) * STRIDE + col_single,
            col_single < size,
            other=-1,
        )
        offset_single = g.to(tl.int64) * (2 * HALF) + col_single
        tl.store(out_ix + offset_single, keys_single, col_single < size)
        tl.store(
            out_counts + offset_single,
            (keys_single >= 0).to(tl.int32),
            col_single < size,
        )
        tl.store(out_lens + g, size)
        return
    lane = tl.arange(0, 2 * HALF)
    which = lane // HALF
    col = lane % HALF
    row = tl.load(rows + g * 2 + which)
    length = tl.load(lengths + row, row >= 0, other=0)
    key = tl.load(
        ix + row.to(tl.int64) * STRIDE + col,
        (row >= 0) & (col < length) & (col < WIDTH),
        other=-1,
    )
    sentinel: tl.constexpr = 0xFFFFFFFF if BOUNDED else 0x7FFFFFFFFFFFFFFF
    if BOUNDED:
        # Caller proves allocation has fewer than 2**30 slots, without GPU sync.
        code = tl.where(key >= 0, key.to(tl.uint32) * 2 + which.to(tl.uint32), sentinel)
    else:
        code = tl.where(key >= 0, key.to(tl.int64) * 2 + which, sentinel)
    code = tl.sort(code, descending=False)
    k = code >> 1
    live = code != sentinel
    c0 = (live & ((code & 1) == 0)).to(tl.int32)
    c1 = (live & ((code & 1) == 1)).to(tl.int32)
    _, c0, c1 = tl.associative_scan((k, c0, c1), 0, _segment)
    prev = tl.gather(k, tl.maximum(lane - 1, 0), 0)
    nxt = tl.gather(k, tl.minimum(lane + 1, 2 * HALF - 1), 0)
    begin = live & ((lane == 0) | (prev != k))
    end = live & ((lane == 2 * HALF - 1) | (nxt != k))
    dest = tl.cumsum(begin.to(tl.int32)) - 1
    offset = g.to(tl.int64) * (2 * HALF) + dest
    tl.store(out_ix + offset, k.to(tl.int32), end)
    tl.store(out_counts + offset, c0 | (c1 << 16), end)
    tl.store(out_lens + g, tl.sum(begin.to(tl.int32)))


@triton.jit
def _both(
    rows,
    count,
    ix0,
    len0,
    out0,
    counts0,
    lens0,
    ix1,
    len1,
    out1,
    counts1,
    lens1,
    S0: tl.constexpr,
    W0: tl.constexpr,
    H0: tl.constexpr,
    S1: tl.constexpr,
    W1: tl.constexpr,
    H1: tl.constexpr,
):
    if tl.program_id(1) == 0:
        _union(rows, count, ix0, len0, out0, counts0, lens0, S0, W0, H0, True)
    else:
        _union(rows, count, ix1, len1, out1, counts1, lens1, S1, W1, H1, True)
