# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from ROCm/ATOM (https://github.com/ROCm/ATOM) at a526f0d, MIT License,
# Copyright (C) 2026, Advanced Micro Devices, Inc.:
# atom/mono/device/ops.py
# atom/mono/device/ranks.py
# atom/mono/plan/execution.py
# atom/mono/plan/layout.py
"""The execution model the mono kernels are written for, their mailbox
regions' layout, and the device helpers that depend on it.

One CTA per CU of an MI355X, all of them co-resident -- a spin wait on another
CTA can only make progress if that CTA is running, so the runner refuses a GPU
with another CU count -- and THREADS threads each.
"""

import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr

from vllm.models.qwen3_5.amd.specialized_decode.abi import health_words
from vllm.models.qwen3_5.amd.specialized_decode.ops import (
    bf2_f32,
    ld_dev,
    ld_ptr64,
    traced,
    wave_incl_scan,
)

BLOCKS = 256
THREADS = 512
WAVES = THREADS // 64

PAIR_BYTES = 8  # a (value, tag) pair
# A rank's health words (``abi``), for this grid.
HEALTH_WORDS = health_words(BLOCKS)


def pair_layout(items, align: int = PAIR_BYTES, start: int = 0) -> dict:
    """`items` (name, pairs) laid out in order: name -> (byte offset, bytes),
    each region from an `align`-byte boundary at or past `start`."""
    out, off = {}, start
    for name, pairs in items:
        off = (off + align - 1) // align * align
        out[name] = (off, pairs * PAIR_BYTES)
        off += pairs * PAIR_BYTES
    return out


# --------------------------------------------------------------------------
# the TP group
# --------------------------------------------------------------------------


def peer_bases(peers, tp):
    """Every rank's peer buffer base from the `peers` table (an i64 a rank),
    loaded once: a put's address must not wait on a load of its own."""
    return [ld_ptr64(peers, p) for p in range(tp)]


def sum_partials(poll, own, pair_of, tp):
    """The `tp` ranks' bf16 pair partials at `pair_of(src)` of the mailbox
    region `own`, summed in fp32 in rank order 0 .. tp - 1: one poll batch ->
    (lo sum, hi sum). Every rank sums in the same order, so they agree."""
    ws = poll([(own, pair_of(src), 1) for src in range(tp)])
    acc0, acc1 = bf2_f32(ws[0][0])
    for src in range_constexpr(1, tp):
        lo, hi = bf2_f32(ws[src][0])
        acc0 = acc0 + lo
        acc1 = acc1 + hi
    return acc0, acc1


# --------------------------------------------------------------------------
# CTA-wide helpers
# --------------------------------------------------------------------------


@traced
def block_excl_scan(v, lane, wave, buf):
    """(exclusive prefix of v over the CTA in thread order, CTA total); `buf`
    needs WAVES words."""
    incl = wave_incl_scan(v, lane)
    gpu.barrier()
    if lane == 63:
        fx.ptr_store(incl, buf + wave)
    gpu.barrier()
    off = fx.Int32(0)
    tot = fx.Int32(0)
    for w in range_constexpr(WAVES):
        cnt = fx.ptr_load(buf + w)
        off = off + (w < wave).select(cnt, 0)
        tot = tot + cnt
    return incl - v + off, tot


LOAD_BATCH = 4  # rounds of a strided loop whose loads go out together


@traced
def batched_rounds(tid, n, load, use, batch=LOAD_BATCH):
    """A CTA's strided loop over elements v = tid + THREADS i < n (n a Python
    int): `load(v)` of `batch` rounds issued before any `use(v, loaded)`.
    A round whose `use` stores to global memory keeps the next round's loads
    behind it (they may alias), so the rounds' loads would go out in series. A
    round past n loads element n - 1 (in range) and uses nothing."""
    rounds = -(-n // THREADS)
    for b0 in range_constexpr(0, rounds, batch):
        idx = range(b0, min(b0 + batch, rounds))
        got = [load(fx.min(tid + THREADS * i, n - 1)) for i in idx]
        for i, g in zip(idx, got):
            v = tid + THREADS * i
            if v < n:
                use(v, g)


def rows_to_lds(tid, src, rows, row_words, dst, dst_row):
    """`rows` rows of `row_words` words (a multiple of 4) read at device scope
    from address `src` -> LDS `dst`, a row every `dst_row` words: 16 B a
    lane, the loads batched (`batched_rounds`)."""
    assert row_words % 4 == 0, row_words
    batched_rounds(
        tid, rows * row_words // 4,
        lambda v: fx.Vector(ld_dev(src, v * 4, 4)),
        lambda v, w: fx.ptr_store(
            w, dst + (v * 4 // row_words * dst_row + v * 4 % row_words)
        ),
    )  # fmt: skip


def row_sum(red, r, t, waves=None):
    """Row r, token t of the waves' 16x16 MFMA accumulators (wave w's at `red`
    + (w * 64 + lane) * 4), summed in wave order from 0 (`waves`: which, in
    order; all of them by default)."""
    tot = fx.Float32(0.0)
    for w in waves if waves is not None else range(WAVES):
        tot = tot + fx.ptr_load(red + ((w * 64 + 16 * (r // 4) + t) * 4 + r % 4))
    return tot
