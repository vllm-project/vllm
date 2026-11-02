# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A one-GPU kernel that hands data between CTAs on different XCDs with the
FFN launch's same-GPU protocol (``mono/sync.py``).

Every round each of ``BLOCKS`` resident CTAs writes a ``DWORDS`` payload to its
own buffer (32-, 64- or 16-bit stores by round, values alternating with the
round's parity), publishes it, and checks the buffer of the CTA ``k`` behind
it with 128-bit loads in another lane order; ``k`` is odd and changes every
round, so under round-robin placement over 8 XCDs producer and consumer never
share an XCD (each CTA records its XCC_ID / HW_ID to verify). Before polling,
the consumer pulls the producer's buffer into its caches with plain loads, so
a stale line would be hit; random per-CTA delays skew the rounds. The
consumer's ack frees the buffer for the next round."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import Int32, Int64, T

from vllm.models.deepseek_v41.amd.mono.common.ops import rsrc, traced
from vllm.models.deepseek_v41.amd.mono.common.sync import Space, sreg
from vllm.models.minimax_m3.amd.specialized_decode.mono.sync import (
    Mailbox,
    cta_sync,
    load_dev,
    store16,
    store_dev,
)

BLOCKS = 256
THREADS = 512
DWORDS = 8192  # 32 KB a CTA: 8 MB in all, past an XCD's 4 MB L2
ROUNDS = 64
OUT_WORDS = 8  # xcc, hw_id, bad, then the first bad: round, producer, dword, got
WAIT_TICKS = 100_000_000  # 1 s

_STREAM = fx.Stream(None)


def _hash(p, r, x):
    v = (x + 1) * (2 * p + 1) * fx.Int32(-1640531535) + r * fx.Int32(1663821227)
    return ((r & 1) != 0).select(v ^ fx.Int32(-1), v)


def _want(p, r, d):
    """Dword d of producer p's round r payload."""
    half = (_hash(p, r, 2 * d) & 0xFFFF) | (_hash(p, r, 2 * d + 1) << 16)
    return (r % 3 == 2).select(half, _hash(p, r, d))


def _hwreg(reg):
    return fx.Int32(
        _llvm.InlineAsmOp(
            T.i32, [], f"s_getreg_b32 $0, hwreg({reg})", "=s", has_side_effects=True
        ).result
    )


def _store(scoped, buf, i, words):
    if const_expr(scoped):
        store_dev(buf, i, words)
    else:
        v = words[0] if len(words) == 1 else fx.Vector.from_elements(words, fx.Int32)
        bo.buffer_store(v, rsrc(buf), i)


@traced
def _produce(scoped, buf, p, r, tid):
    mode = r % 3
    if mode == 0:
        for j in range_constexpr(DWORDS // THREADS):
            d = j * THREADS + tid
            _store(scoped, buf, d, [_hash(p, r, d)])
    if mode == 1:
        for j in range_constexpr(DWORDS // THREADS // 2):
            d = 2 * (j * THREADS + tid)
            _store(scoped, buf, d, [_hash(p, r, d), _hash(p, r, d + 1)])
    if mode == 2:
        for j in range_constexpr(2 * DWORDS // THREADS):
            e = j * THREADS + tid
            v = fx.Int16(_hash(p, r, e) & 0xFFFF).bitcast(fx.BFloat16)
            if const_expr(scoped):
                store16(buf, e, v, "agent")
            else:
                bo.buffer_store(v, rsrc(buf), e)


@traced
def _skew(p, r, seed):
    n = (_hash(p, r, seed) >> 20) & 63
    i = fx.Int32(0)
    while i < n:
        rocdl.s_sleep(2)
        i = i + 1


@traced
def _record(out, bid, r, q, d, got):
    base = bid * OUT_WORDS
    fx.llvm.atomic_add(
        fx.inttoptr(
            fx.PointerType.get(T.i32, fx.AddressSpace.Global, 4),
            out + fx.Int64(base + 2) * 4,
        ),
        fx.Int32(1),
        syncscope="agent",
    )
    for k, v in enumerate((r, q, d, got)):
        bo.buffer_store(fx.Int32(v), rsrc(out), base + 3 + k)


@traced
def _check(scoped, buf, q, r, tid, out, bid):
    for j in range_constexpr(4):
        ch = (tid ^ 0x155) + THREADS * j
        if const_expr(scoped):
            got = load_dev(buf, 4 * ch, 4)
        else:
            got = fx.Vector(bo.buffer_load(rsrc(buf), 4 * ch, vec_width=4, dtype=T.i32))
        for i in range_constexpr(4):
            d = 4 * ch + i
            g = fx.Int32(got[i])
            if g != _want(q, r, d):
                _record(out, bid, r, q, d, g)


def _mailbox(c, tag):
    return Mailbox(tag, c["err"], [c["err"]], 0, 0, WAIT_TICKS)


def _await(c, tag, i, site):
    _mailbox(c, tag).poll([(c["pairs"], i, 1)], site=site)
    _mailbox(c, tag).acquire(Space.SCRATCH)


def _publish(c, tag, i, v):
    _mailbox(c, tag).publish(c["pairs"], i, v)


@traced
def _round(c, r):
    bid, tid = c["bid"], c["tid"]
    tag = c["base"] + r + 1
    k = 2 * ((r * 37 + c["seed"]) % 64) + 1
    q = (bid + BLOCKS - k) % BLOCKS
    mine = c["buf"] + fx.Int64(bid) * (DWORDS * 4)
    theirs = c["buf"] + fx.Int64(q) * (DWORDS * 4)
    # my buffer is free once last round's consumer acked it
    if (r > 0) & (tid == 0):
        _await(c, tag - 1, BLOCKS + bid, 1)
    cta_sync()
    _skew(bid, r, c["seed"])
    _produce(c["stores"], mine, bid, r, tid)
    cta_sync(drain=True)
    if tid == 0:
        _publish(c, tag, bid, r)
    # stale copies of the consumer's next reads in this CU's L1 and XCD's L2
    old = fx.Int32(0)
    for j in range_constexpr(4):
        ch = (tid ^ 0x155) + THREADS * j
        v = fx.Vector(bo.buffer_load(rsrc(theirs), 4 * ch, vec_width=4, dtype=T.i32))
        for i in range_constexpr(4):
            old = old ^ fx.Int32(v[i])
    bo.buffer_store(old, rsrc(c["sink"]), bid * THREADS + tid)
    if tid == 0:
        _await(c, tag, q, 2)
    cta_sync()
    _check(c["loads"], theirs, q, r, tid, c["out"], bid)
    cta_sync()
    if tid == 0:
        _publish(c, tag, BLOCKS + q, r)


def build(stores=True, loads=True):
    """``stores`` / ``loads`` False: the payload's stores / loads plain cached
    accesses instead (negative controls: not store-available / load-visible)."""
    name = f"minimax_m3_xcd_stress_s{int(stores)}_l{int(loads)}"

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kernel(
        buf: Int64,
        pairs: Int64,
        err: Int64,
        out: Int64,
        sink: Int64,
        base: Int32,
        seed: Int32,
    ):
        bid, tid = fx.block_idx.x, fx.thread_idx.x
        if tid == 0:
            o = bid * OUT_WORDS
            bo.buffer_store(_hwreg("HW_REG_XCC_ID, 0, 4"), rsrc(out), o)
            bo.buffer_store(_hwreg("HW_REG_HW_ID"), rsrc(out), o + 1)
        c = {
            "stores": stores,
            "loads": loads,
            "bid": bid,
            "tid": tid,
            "base": base,
            "seed": seed,
            "err": err,
            "buf": buf,
            "out": out,
            "sink": sink,
            "pairs": sreg(pairs, 0, "pairs"),
        }
        for r in range(ROUNDS):
            _round(c, r)

    @flyc.jit
    def launch(
        buf: Int64,
        pairs: Int64,
        err: Int64,
        out: Int64,
        sink: Int64,
        base: Int32,
        seed: Int32,
        stream: fx.Stream = _STREAM,
    ):
        kernel(buf, pairs, err, out, sink, base, seed).launch(
            grid=(BLOCKS,), block=(THREADS,), stream=stream
        )

    return launch


def run(launches, stores=True, loads=True, stop_on_error=True):
    """``launches`` launches of ``ROUNDS`` rounds; returns the error word and
    the per-CTA records (host tensors)."""
    import torch

    launch = build(stores, loads)
    dev = torch.device("cuda")
    buf = torch.zeros(BLOCKS * DWORDS, dtype=torch.int32, device=dev)
    pairs = torch.zeros(2 * BLOCKS * 2, dtype=torch.int32, device=dev)
    err = torch.zeros(64, dtype=torch.int32, device=dev)
    out = torch.zeros(BLOCKS, OUT_WORDS, dtype=torch.int32, device=dev)
    sink = torch.zeros(BLOCKS * THREADS, dtype=torch.int32, device=dev)
    ptrs = [t.data_ptr() for t in (buf, pairs, err, out, sink)]
    for n in range(launches):
        launch(*ptrs, n * ROUNDS, n)
        if n % 64 == 63:
            torch.accelerator.synchronize()
            if stop_on_error and err[0].item() != 0:
                break
    torch.accelerator.synchronize()
    return err[:4].cpu(), out.cpu()
