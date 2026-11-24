# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: ignore-errors
"""DSpark greedy draft: the vocab-sharded argmax combine as one launch a position.

One CTA a draft row, at TP 4: the row's argmax over this rank's vocabulary
shard of the fp32 logits (ties to the lower index), then the
(value bits, global index) record pushed into every peer's mailbox with one
16 B system-scope store, the peers' records polled from this rank's mailbox,
and the global pick written as int32. The global index is
`rank * shard + in-shard index`, so the pick equals a first-maximum argmax over
the whole vocabulary, bitwise. It replaces the all-gather of the per-block keys
and the combine launch of `markov_vocab_shard`.

The record carries its own tag, derived from a device epoch that the caller
bumps once a draft step, so a stale record never matches and no mailbox is ever
zeroed (the monokernel's mailbox idiom, kernels/dev.py). A rank's position
i + 1 push comes only after its position i poll completed against every peer,
so one region a position is enough. A poll that times out records the CTA in
`health` (dev.py's `_spin_bounded`).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu, range_constexpr

from .dev import (
    CM_DEV,
    CM_SYS,
    ld_i32,
    ld_vec_if,
    mb_poll,
    rsrc,
    uniform,
    xshfl,
)

THREADS = 512
WAVES = THREADS // 64
UNROLL = 8  # vec4 loads in flight a thread; the grid is only `rows` CTAs, so
# latency hiding comes from ILP, not occupancy


def _sel_peer(pl, d):
    """`pl[d]` for a traced rank `d` (k3.sel_peer assumes 8 peers)."""
    p = pl[0]
    for r_ in range_constexpr(1, len(pl)):
        p = (d == r_).select(pl[r_], p)
    return p


def _take_max(bv, bi, ov, oi):
    take = (ov > bv) | ((ov == bv) & (oi < bi))
    return take.select(ov, bv), take.select(oi, bi)


def _wave_argmax(bv, bi, offs=(32, 16, 8, 4, 2, 1)):
    for off in offs:
        bv, bi = _take_max(bv, bi, xshfl(bv, off), xshfl(bi, off))
    return bv, bi


def compile_combine(world: int, rows: int, shard: int):
    """The launch for `rows` draft rows of a `shard`-token vocabulary shard.

    The launch takes the fp32 logits `[rows, shard]` (row stride `shard`), the
    int32 output `[rows]`, the epoch word, the health words
    (`8 + 8 * rows` int32), up to 8 peer mailbox pointers (`world` used, this
    rank's region of `16 * world * rows` bytes at each), the shard size, this
    rank and the stream.
    """
    assert world in (2, 4, 8) and shard % 4 == 0 and 1 <= rows <= 56, (
        world,
        rows,
        shard,
    )
    WORLD = world
    NCHUNK = shard // 4
    ROUNDS = -(-NCHUNK // (THREADS * UNROLL))
    _KEYSTR = f"draft_combine_w{world}_r{rows}_s{shard}_t{THREADS}_u{UNROLL}"

    @fx.struct
    class Red:
        v: fx.Array[fx.Float32, WAVES, 16]
        i: fx.Array[fx.Int32, WAVES, 16]

    @flyc.kernel(
        name=f"draft_combine_w{world}_r{rows}", known_block_size=[THREADS, 1, 1]
    )
    def kernel(
        x: fx.Int64,
        out: fx.Int64,
        epoch: fx.Int64,
        health: fx.Int64,
        peer0: fx.Int64,
        peer1: fx.Int64,
        peer2: fx.Int64,
        peer3: fx.Int64,
        peer4: fx.Int64,
        peer5: fx.Int64,
        peer6: fx.Int64,
        peer7: fx.Int64,
        i32_shard: fx.Int32,
        i32_rank: fx.Int32,
    ):
        tx, row = fx.thread_idx.x, fx.block_idx.x
        lane, wave = tx % 64, tx // 64
        rank = fx.Int32(i32_rank)
        pl = [peer0, peer1, peer2, peer3, peer4, peer5, peer6, peer7][:WORLD]
        ep = uniform(ld_i32(epoch, 0, CM_DEV))
        tag = (ep & 0x1FFFFFF) * 8 + 1

        # phase A: per-row shard argmax over fp32 bits (tie -> lower index)
        # An all -inf (or NaN) shard picks its first token, as greedy_argmax.
        bv = fx.Float32(float("-inf"))
        bi = fx.Int32(0)
        nbytes = rows * shard * 4
        for r in range_constexpr(ROUNDS):
            base = (r * UNROLL) * THREADS + tx
            vs, oks = [], []
            for j in range_constexpr(UNROLL):
                ok = base + j * THREADS < NCHUNK
                oks.append(ok)
                vs.append(
                    ld_vec_if(
                        ok,
                        x,
                        row * i32_shard + (base + j * THREADS) * 4,
                        4,
                        nbytes,
                        CM_DEV,
                    )
                )
            for j in range_constexpr(UNROLL):
                for e in range_constexpr(4):
                    idx = fx.Int32((base + j * THREADS) * 4 + e)
                    vv = vs[j][e].bitcast(fx.Float32)
                    take = oks[j] & ((vv > bv) | ((vv == bv) & (idx < bi)))
                    bv = take.select(vv, bv)
                    bi = take.select(idx, bi)
        bv, bi = _wave_argmax(bv, bi)

        red = fx.SharedAllocator().allocate(Red).peek()
        if lane == 0:
            fx.ptr_store(bv, red.v.ptr + wave)
            fx.ptr_store(bi, red.i.ptr + wave)
        gpu.barrier()
        if wave == 0:
            bv = fx.ptr_load(red.v.ptr + lane % WAVES)
            bi = fx.ptr_load(red.i.ptr + lane % WAVES)
            bv, bi = _wave_argmax(bv, bi, (4, 2, 1))
            if lane == 0:
                glo = rank * i32_shard + bi
                rec = fx.Vector.from_elements(
                    [bv.bitcast(fx.Int32), tag, fx.Int32(glo), tag], fx.Int32
                )
                for dst in range_constexpr(WORLD):
                    bo.buffer_store(
                        rec,
                        rsrc(pl[dst]),
                        (rank * rows + row) * 4,
                        cache_modifier=CM_SYS,
                    )
                own = _sel_peer(pl, rank)
                specs = [(own, (s * rows + row) * 2, 2) for s in range(WORLD)]
                poks = [fx.Int32(s) != rank for s in range(WORLD)]
                vals = mb_poll(tag, health, specs, 1, oks=poks)
                best_v, best_i = bv, glo
                for s in range_constexpr(WORLD):
                    ok = fx.Int32(s) != rank
                    pv = vals[s][0].bitcast(fx.Float32)
                    pi = fx.Int32(vals[s][1])
                    take = ok & ((pv > best_v) | ((pv == best_v) & (pi < best_i)))
                    best_v = take.select(pv, best_v)
                    best_i = take.select(pi, best_i)
                bo.buffer_store(fx.Int32(best_i), rsrc(out), row, cache_modifier=CM_DEV)

    @flyc.jit
    def launch(
        x: fx.Int64,
        out: fx.Int64,
        epoch: fx.Int64,
        health: fx.Int64,
        peer0: fx.Int64,
        peer1: fx.Int64,
        peer2: fx.Int64,
        peer3: fx.Int64,
        peer4: fx.Int64,
        peer5: fx.Int64,
        peer6: fx.Int64,
        peer7: fx.Int64,
        i32_shard: fx.Int32,
        i32_rank: fx.Int32,
        stream: fx.Stream,
    ):
        _ = _KEYSTR  # disk-cache identity (k3.build idiom)
        kernel(
            x,
            out,
            epoch,
            health,
            peer0,
            peer1,
            peer2,
            peer3,
            peer4,
            peer5,
            peer6,
            peer7,
            i32_shard,
            i32_rank,
        ).launch(grid=(rows, 1, 1), block=(THREADS, 1, 1), stream=stream)

    return launch
