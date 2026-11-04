# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Five seven-row draft layers using six persistent launches on gfx950."""

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import Int32, Int64, T, as_ir_value

from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.kda import (
    bf4_words,
    bf16r,
    ld_bf4,
    poll_rows_to_lds,
    step_tag,
)
from vllm.models.kimi_k3.amd.specialized_decode.mono.common.ops import (
    CM_DEV,
    bf16_pair,
    bf_hi,
    bf_lo,
    butterfly,
    hw_exp2,
    hw_rsq,
    ld_i32,
    lds_bytes,
    load_ptr64,
    mfma_bf16,
    row_sum,
    rsrc,
    traced,
    uniform,
)
from vllm.models.kimi_k3.amd.specialized_decode.mono.common.plan import pair_layout
from vllm.models.kimi_k3.amd.specialized_decode.mono.common.ranks import peer_bases
from vllm.models.kimi_k3.amd.specialized_decode.mono.common.sync import (
    Mailbox,
    poll_sent,
    preg,
    publish,
    put_sent,
    sent_word,
    sreg,
)
from vllm.models.kimi_k3.amd.specialized_decode.mono.stages import gemv

from . import attention, combine, frontend, geometry

# Sentinel all-reduces (draft run 7; the target's OPT_SENT form, sync.py's
# put_sent / poll_sent): a peer's bf16 pair goes as one plain 4 B word (not a
# 4 B word + its 4 B tag) into the receiver's region, which holds the sentinel
# (all ones; ``sent_word`` remaps a real all-ones word to another NaN pair);
# the reader polls off the sentinel and re-arms. Half the xGMI and poll bytes;
# the same values summed in the same peer order (bit-exact). Word index of
# (src rank, token t, column pair): (src S + t) H / 2 + col / 2. Regions
# alternate by step parity (``par``) and are armed at prepare (runtime).
SENT_AR = True  # o_proj -> post norm ('arout')
SENT_NEXT = True  # down -> the next layer's input norm ('arin')

S = 7
H = 7168
FF = 1792
NH = 8
BLOCKS = 256
THREADS = 512
NC = S * NH  # (head, token) pairs
# uvh (MLA run 5's combine by W_UV output-column halves, no exchange): CTAs
# b and b + NC (one XCD) both combine pair b's partials in full and run W_UV
# for half b // NC of its 128 output columns (64 KB of W_UV a CTA, not 128)
UVH = True
NCC = 2 * NC if UVH else NC  # combine + W_UV CTAs
OPC = BLOCKS - NCC  # o_proj CTAs (NCC ..): 448 tasks
O4 = H // 16 - 3 * OPC  # (uvh) o_proj CTAs with a 4th task: the last O4
assert (0 < O4 <= OPC) if UVH else (2 * OPC < H // 16 <= 3 * OPC)
# The AR consumers (``normalize_wide``): a (token, column slice) a CTA. Front
# (input norm): 8 slices on CTAs FN0.. (q_b CTAs: no weight stream waits behind
# it); back (post norm): 4 slices on CTAs BN0.. (o_proj CTAs without gate_up).
NSL_F, FN0 = 8, 139
NSL_B, BN0 = 4, BLOCKS - 4 * 7
NSL = 8  # flag / msq regions sized for the wider
NN = S * NSL
GU = 2 * FF // 16  # gate_up tasks (224): CTAs 0 .. GU
assert FN0 + S * NSL_F <= 235 and BN0 >= GU and BN0 + S * NSL_B == BLOCKS
REGIONS = tuple(frontend.REGIONS) + (
    "xrdy",
    "ffnrdy",
    "ardy",
    "core",
    "gu",
    "act",
    "arin",
    "arout",
    "msqf",
    "msqb",
)
extra = pair_layout(
    [
        ("xrdy", NN),
        ("ffnrdy", NN),
        ("ardy", BLOCKS),
        ("core", S * 1024 // 2),
        ("gu", S * 2 * FF // 2),
        ("act", S * FF // 2),
        ("msqf", NN),
        ("msqb", NN),
    ]
)
EXTRA_OFF = (frontend.NBYTES + 15) // 16 * 16
LAY = dict(frontend.LAY) | {k: (o + EXTRA_OFF, n) for k, (o, n) in extra.items()}
NBYTES = max(o + n for o, n in LAY.values())
STREAM = fx.Stream(None)
BUILDS: dict = {}
SOURCE = hashlib.sha256(
    b"".join(p.read_bytes() for p in sorted(Path(__file__).parent.glob("*.py")))
).hexdigest()


@dataclass(frozen=True)
class Key:
    tp: int = 8
    eps: float = 1e-5
    ws: int = 0
    first: bool = False
    last: bool = False  # The final down projection writes the output.


def peer_bytes(tp):
    return 2 * 2 * tp * S * (H // 2) * 8


def context(a, sm, key, bid, tid):
    mb = Mailbox(
        step_tag(a["epoch"], a["layer"]) - 1,
        a["health"],
        a["alarm"],
        {n: i for i, n in enumerate(REGIONS)},
    )
    arbytes = key.tp * S * (H // 2) * 8
    par = fx.Int64(uniform(ld_i32(a["epoch"], 0)) % 2) * 2 * arbytes
    bases = peer_bases(a["peers"], key.tp)
    own = load_ptr64(a["peers"], a["rank"])
    c = dict(
        a=a,
        args=a,
        S=S,
        tid=tid,
        bid=bid,
        lane=tid % 64,
        wave=tid // 64,
        red=sm.red.ptr,
        xl=sm.xl.ptr,
        lat=sm.lat.ptr,
        cw=sm.red.ptr,
        tp=key.tp,
        eps=key.eps,
        rank=a["rank"],
        QH=8,
        nrdy=S * NSL_F,
        qc=a["scratch"] + fx.Int64(frontend.QCOFF),
        mb=mb,
        put=mb.put,
        put_bf=mb.put_bf,
        put_words=mb.put_words,
        poll=mb.poll,
    )
    for n, (o, _) in LAY.items():
        c[n] = sreg(a["scratch"], o, n)
    for n, off in [("arin", 0), ("arout", arbytes)]:
        c[n + "_peers"] = [preg(b + par, off, n) for b in bases]
        c[n + "_own"] = preg(own + par, off, n)
    return c


def ld_bf4_dev(ptr, i):
    """`ld_bf4` at device scope (rows another CTA of the launch wrote)."""
    w = fx.Vector(
        bo.buffer_load(
            rsrc(ptr), i // 2, vec_width=2, dtype=T.i32, cache_modifier=CM_DEV
        )
    )
    return [bf_lo(w[0]), bf_hi(w[0]), bf_lo(w[1]), bf_hi(w[1])]


@traced
def normalize_wide(
    c,
    t,
    s,
    nsl,
    which,
    residual,
    weight,
    resout,
    ready,
    msq,
    first=False,
):
    """`normalize` of token t over `nsl` CTAs: this CTA's slice s (H / nsl
    columns, thread tid < H / 4 / nsl four of them). The 8 peers' rows of the
    slice (one poll of 8 specs: 229 / nsl KB a CTA of the uncached peer buffer,
    not 229 KB) summed + residual, the slice's sum of squares exchanged with the
    token's other slices (`msq`, device scope) for the row's rstd, then the
    slice's resout and x and its ready flag (t nsl + s)."""
    tid, lane, wave, tp = c["tid"], c["lane"], c["wave"], c["tp"]
    qs = H // 4 // nsl
    live = tid < qs
    e = s * qs + fx.min(tid, qs - 1)
    nw = ld_bf4(weight, 4 * e)
    if const_expr(first):
        v = ld_bf4(c["a"]["hidden"], t * H + 4 * e)
    else:
        old = ld_bf4_dev(residual, t * H + 4 * e)
        own = c[which + "_own"]
        if const_expr(SENT_AR if which == "arout" else SENT_NEXT):
            # (SENT_*) spin on one peer's word a thread (the peers in
            # rotation), then the 8 peers' 2 words each, re-armed after the
            # read (live threads only: a dead thread's clamped words are
            # thread qs - 1's, a read is destructive)
            poll_sent(
                c["mb"],
                own,
                [((tid % tp) * S + t) * H // 2 + 2 * e],
                [live],
                rearm=False,
            )
            ws = poll_sent(
                c["mb"],
                own,
                [(p * S + t) * H // 2 + 2 * e + j for p in range(tp) for j in range(2)],
                [live] * (2 * tp),
            )
            got = [[ws[2 * p], ws[2 * p + 1]] for p in range(tp)]
        else:
            # spin on one peer's spec a thread (the peers in rotation: 1/8 the
            # bytes a spin round), then the 8 peers' specs in one batch
            c["poll"]([(own, ((tid % tp) * S + t) * H // 2 + 2 * e, 2)])
            got = c["poll"](
                [(own, (p * S + t) * H // 2 + 2 * e, 2) for p in range(tp)], batch=tp
            )
        w = got[0]
        acc = [bf_lo(w[0]), bf_hi(w[0]), bf_lo(w[1]), bf_hi(w[1])]
        for p in range_constexpr(1, tp):
            w = got[p]
            z = [bf_lo(w[0]), bf_hi(w[0]), bf_lo(w[1]), bf_hi(w[1])]
            acc = [acc[j] + z[j] for j in range(4)]
        v = [bf16r(acc[j]) + old[j] for j in range(4)]
    sq = live.select(
        v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3], fx.Float32(0.0)
    )
    sq = butterfly(sq, (32, 16, 8, 4, 2, 1))
    if lane == 0:
        fx.ptr_store(sq, c["red"] + wave)
    gpu.barrier()

    part = fx.ptr_load(c["red"])
    for i in range_constexpr(1, 8):
        part = part + fx.ptr_load(c["red"] + i)
    if tid == 0:
        c["put"](c[msq], t * nsl + s, fx.Float32(part).bitcast(fx.Int32))
    # thread j < nsl polls slice j's sum -> LDS; every thread sums them in order
    if tid < nsl:
        ms = c["poll"]([(c[msq], t * nsl + tid, 1)])
        fx.ptr_store(ms[0][0].bitcast(fx.Float32), c["red"] + 8 + tid)
    gpu.barrier()

    total = fx.ptr_load(c["red"] + 8)
    for j in range_constexpr(1, nsl):
        total = total + fx.ptr_load(c["red"] + 8 + j)
    inv = hw_rsq(total * fx.Float32(1 / H) + fx.Float32(c["eps"]))
    if live:
        bo.buffer_store(
            bf4_words(v), rsrc(resout), (t * H + 4 * e) // 2, cache_modifier=CM_DEV
        )
        bo.buffer_store(
            bf4_words([v[j] * inv * nw[j] for j in range(4)]),
            rsrc(c["a"]["x"]),
            (t * H + 4 * e) // 2,
            cache_modifier=CM_DEV,
        )
    publish(c["put"], c[ready], t * nsl + s, fx.Int32(1), tid == 0)


@traced
def dense(c, task, w, k, source, width, kind):
    tid, lane, wave, red = c["tid"], c["lane"], c["wave"], c["red"]
    ops = gemv.loads(lane, wave, w, task * 16, k)
    if const_expr(source == "x"):
        if tid < S * NSL_B:
            c["poll"]([(c["ffnrdy"], tid, 1)])
        gpu.barrier()
        gemv.dev_rows_to_lds(tid, c["a"]["x"], k, S, c["xl"], k + 8)
    else:
        poll_rows_to_lds(c, c[source], k, S, c["xl"], k + 8, k, 0, 0)
    acc = gemv.mfmas(lane, wave, c["xl"], k + 8, k, S, ops)
    fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
    gpu.barrier()
    row = tid % 8
    t = fx.min(tid // 8, S - 1)
    if tid < 8 * S:
        v = [row_sum(red, 2 * row + j, t) for j in range(2)]
        col = task * 16 + 2 * row
        if const_expr(kind == "ar"):
            for p in range_constexpr(c["tp"]):
                c["put_bf"](c["arout_peers"][p], (c["rank"] * S + t) * H + col, v)
        elif const_expr(kind == "out"):
            bo.buffer_store(
                bf4_words(v + v)[0], rsrc(c["a"]["out"]), (t * H + col) // 2
            )
        else:
            c["put_bf"](c[kind], t * width + col, v)
    gpu.barrier()


@traced
def dense_n(c, tasks, lives, w, k, source, width, kind):
    """`dense` for len(tasks) tasks of 16 rows on one CTA: every task's
    weights issued before the poll, the token rows polled into LDS once.
    `lives[i]`: None (always) or a traced condition (a dead task's rows are
    clamped in range and its outputs not stored)."""
    tid, lane, wave, red = c["tid"], c["lane"], c["wave"], c["red"]
    last = H // 16 - 1
    ops = [gemv.loads(lane, wave, w, fx.min(tk, last) * 16, k) for tk in tasks]
    if const_expr(source == "x"):
        if tid < S * NSL_B:
            c["poll"]([(c["ffnrdy"], tid, 1)])
        gpu.barrier()
        gemv.dev_rows_to_lds(tid, c["a"]["x"], k, S, c["xl"], k + 8)
    else:
        poll_rows_to_lds(c, c[source], k, S, c["xl"], k + 8, k, 0, 0)

    for i in range_constexpr(len(tasks)):
        acc = gemv.mfmas(lane, wave, c["xl"], k + 8, k, S, ops[i])
        fx.ptr_store(acc, red + (i * 512 + wave * 64 + lane) * 4)
    gpu.barrier()
    row = tid % 8
    t = fx.min(tid // 8, S - 1)
    for i in range_constexpr(len(tasks)):
        go = tid < 8 * S
        if const_expr(lives[i] is not None):
            go = go & lives[i]
        if go:
            v = [row_sum(red + i * 2048, 2 * row + j, t) for j in range(2)]
            col = tasks[i] * 16 + 2 * row
            # (SENT_*) the pair's word index: (rank S + t) H / 2 + col / 2
            si = (c["rank"] * S + t) * (H // 2) + tasks[i] * 8 + row
            if const_expr(kind == "ar" and SENT_AR):
                word = sent_word(bf16_pair(v[0], v[1]))
                for p in range_constexpr(c["tp"]):
                    put_sent(c["arout_peers"][p], si, word)
            elif const_expr(kind == "ar"):
                for p in range_constexpr(c["tp"]):
                    c["put_bf"](c["arout_peers"][p], (c["rank"] * S + t) * H + col, v)
            elif const_expr(kind == "out"):
                bo.buffer_store(
                    bf4_words(v + v)[0], rsrc(c["a"]["out"]), (t * H + col) // 2
                )
            elif const_expr(kind == "next" and SENT_NEXT):
                word = sent_word(bf16_pair(v[0], v[1]))
                for p in range_constexpr(c["tp"]):
                    put_sent(c["next_peers"][p], si, word)
            elif const_expr(kind == "next"):
                for p in range_constexpr(c["tp"]):
                    c["put_next"](c["next_peers"][p], (c["rank"] * S + t) * H + col, v)
            else:
                c["put_bf"](c[kind], t * width + col, v)
    gpu.barrier()


def gu_loads(lane, wave, w, task):
    """`gemv.loads` of gate_up rows: MFMA rows 0..7 gate rows 8 task ..,
    rows 8..15 the matching up rows FF + 8 task .. (K = H)."""
    n = H // 128
    per = (n + 7) // 8
    r = lane % 16
    g = lane // 16
    row = (r < 8).select(task * 8 + r, FF + task * 8 + r - 8)
    r_w = rsrc(w)
    out = []
    for i in range(per):
        ch = fx.min(wave + 8 * i, n - 1)
        base = (row * H + ch * 128 + g * 8) // 2
        out.append(
            [
                fx.Vector(
                    bo.buffer_load(
                        r_w, base + 16 * q, vec_width=4, dtype=T.i32, cache_modifier=2
                    )
                )
                for q in range(4)
            ]
        )
    return out


@traced
def gate_up_act(c, task):
    """gate_up task `task`: gate columns 8 task .. 8 task + 8 and their up
    columns, then SiLU(gate) * up of them -> `act` directly (`activate`'s
    math on the same bf16-rounded gate / up values; no SiLU hand-off)."""
    tid, lane, wave, red = c["tid"], c["lane"], c["wave"], c["red"]
    ops = gu_loads(lane, wave, c["a"]["wgu"], task)
    if tid < S * NSL_B:
        c["poll"]([(c["ffnrdy"], tid, 1)])
    gpu.barrier()

    frontend.x_rows_to_lds(tid, c["a"]["x"], H, S, c["xl"], H + 8)
    acc = gemv.mfmas(lane, wave, c["xl"], H + 8, H, S, ops)
    fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
    gpu.barrier()

    p = tid % 4
    t = fx.min(tid // 4, S - 1)
    if tid < 4 * S:
        g = [bf16r(row_sum(red, 2 * p + j, t)) for j in range(2)]
        u = [bf16r(row_sum(red, 8 + 2 * p + j, t)) for j in range(2)]
        v = [
            g[j]
            / (fx.Float32(1.0) + hw_exp2(-g[j] * fx.Float32(1.4426950408889634)))
            * u[j]
            for j in range(2)
        ]
        c["put_bf"](c["act"], t * FF + task * 8 + 2 * p, v)
    gpu.barrier()


@traced
def combine_uv(c, task):
    """Combine + W_UV of (head task % 8, token task // 8): one token a CTA
    over 56 CTAs (`combine.combine_t1`), W_UV's operands loaded first.

    `UVH`: task // NC the half of W_UV's output columns (half 64 ..), task %
    NC the (head, token) pair. Waves w and w + 4 share the 16 rows half 64 +
    16 (w % 4) ..: wave w loads K steps 0..7 of them, wave w + 4 steps 8..15
    (8 dwordx4 a lane, not 16). Every wave runs its 8-MFMA chain from 0, hands
    the accumulator over LDS (`red`, free after the combine's last barrier),
    and runs its chain again from its mate's (w + 4) % 8: waves 4..7 then hold
    the 16-MFMA chain of the block in one wave's order (bit-exact) and store."""
    if const_expr(UVH):
        half = task // NC
        pt = task % NC
    else:
        half = 0
        pt = task
    h = pt % NH
    t0 = pt // NH
    lane, wave = c["lane"], c["wave"]
    g = lane // 16
    if const_expr(UVH):
        row = half * 64 + wave % 4 * 16 + lane % 16
        q0 = wave // 4 * 8
        nq = 8
    else:
        row = wave * 16 + lane % 16
        q0 = 0
        nq = 16
    ws = [
        fx.Vector(
            bo.buffer_load(
                rsrc(c["a"]["wuv"]),
                (h * 128 * 512 + row * 512 + (q0 + q) * 32 + g * 8) // 2,
                vec_width=4,
                dtype=T.i32,
            )
        )
        for q in range(nq)
    ]
    d = dict(
        c,
        S=8,
        Q=8,
        R=1,
        splits=geometry.splits(8),
        TC=1,
        m1c_ht=geometry.geometry(8)[0],
        args=dict(o=c["a"]["part_o"], lse=c["a"]["part_lse"]),
    )
    combine.combine_t1(d, t0, h)

    def chain(acc):
        for q in range_constexpr(nq):
            x = fx.Vector(
                fx.ptr_load(
                    c["lat"] + ((q0 + q) * 32 + g * 8) // 2,
                    result_type=fx.Vector.make_type(4, fx.Int32),
                )
            )
            acc = mfma_bf16(ws[q].bitcast(fx.BFloat16), x.bitcast(fx.BFloat16), acc)
        return acc

    acc = chain(fx.Vector.filled(4, 0.0, fx.Float32))
    if const_expr(UVH):
        hand = c["red"]
        fx.ptr_store(acc, hand + (wave * 64 + lane) * 4)
        gpu.barrier()
        mate = hand + ((wave + 4) % 8 * 64 + lane) * 4
        acc = chain(
            fx.Vector.from_elements(
                [fx.ptr_load(mate + i) for i in range(4)], fx.Float32
            )
        )
        keep = (lane % 16 == 0) & (wave >= 4)
    else:
        keep = lane % 16 == 0
    if keep:
        col = h * 128 + row - lane % 16 + g * 4
        c["put_bf"](c["core"], t0 * 1024 + col, [acc[j] for j in range(4)])
    gpu.barrier()


def structures():
    @fx.struct
    class Dense:
        xl: fx.Array[fx.Int32, S * (H + 8) // 2, 16]
        red: fx.Array[
            fx.Float32, 8192 if UVH else 6144, 16
        ]  # 4 (3) tasks' accumulators (dense_n)
        lat: fx.Array[fx.Int32, 4 * combine.LROW, 16]

    Attn = attention.smem(8, 2, 0)

    @fx.union
    class Both:
        dense: Dense
        attention: Attn  # type: ignore[valid-type]  # FlyDSL shared-memory type

    assert lds_bytes(Both) <= 160 * 1024
    return Dense, Both


# Explicit kernel/launch signatures are generated below from a single ABI.
NAMES = [
    "hidden",
    "residual",
    "in_weight",
    "post_weight",
    "wqkv",
    "wqb",
    "wuk",
    "wuv",
    "wo",
    "wgu",
    "wd",
    "qnw",
    "kvnw",
    "positions",
    "cos_sin",
    "slots",
    "cache",
    "qsi",
    "ksi",
    "query",
    "part_o",
    "part_lse",
    "pages",
    "indptr",
    "scale",
    "vscale",
    "x",
    "residual_mid",
    "residual_out",
    "out",
    "scratch",
    "peers",
    "rank",
    "epoch",
    "layer",
    "health",
    "alarm",
    "npeers",
]
INT32 = {"rank", "layer"}


@traced
def front_body(a, lds, key):
    tid = fx.thread_idx.x
    bid = fx.block_idx.x
    c = context(a, lds, key, bid, tid)

    # Not first: the previous layer's down already pushed its rows into this
    # layer's AR inbox (``dense_n`` kind 'next'): no push_input pass.
    # the input norm over 56 CTAs FN0..: (token, column slice of 8)
    if (bid >= FN0) & (bid < FN0 + S * NSL_F):
        normalize_wide(
            c,
            (bid - FN0) // NSL_F,
            (bid - FN0) % NSL_F,
            NSL_F,
            "arin",
            a["residual"],
            a["in_weight"],
            a["residual_mid"],
            "xrdy",
            "msqf",
            key.first,
        )
    # Frontend expects its output pointer under the name 'out'.
    c["a"] = dict(a, out=a["query"])
    if bid < 132:
        frontend.proj(c, bid)

    if (bid >= 132) & (bid < 139):
        frontend.norm(c, bid - 132)
    if (bid >= 139) & (bid < 235):
        frontend.qb(c, bid - 139)

    if bid >= 235:
        frontend.uk(c, (bid - 235) * 8 + tid // 64)
    if bid < 11:
        frontend.uk(c, (bid + 21) * 8 + tid // 64)


@traced
def back_body(a, lds, key):
    tid = fx.thread_idx.x
    bid = fx.block_idx.x
    sm = lds.dense.peek()
    am = lds.attention.peek()

    mb = Mailbox(
        step_tag(a["epoch"], a["layer"]) - 1,
        a["health"],
        a["alarm"],
        {n: i for i, n in enumerate(REGIONS)},
    )
    ardy = sreg(a["scratch"], LAY["ardy"][0], "ardy")

    def done(i):
        publish(mb.put, ardy, i, fx.Int32(1), tid == 0)

    # qearly (MLA attention run 3's M1c v001): the Q rows DMA'd before the
    # indptr loads, Q's split under block 0's page-index trip (bit-exact)
    ac = dict(
        S=8,
        Q=8,
        dbg=0,
        rt=2,
        wide=True,
        ws=key.ws,
        bid=bid,
        tid=tid,
        cm=CM_DEV,
        qearly=True,
        ks=am.ks.ptr,
        xm=am.xm.ptr,
        xp=am.xp.ptr,
        xi=am.xi.ptr,
        done=done,
        args=dict(
            qb=a["query"],
            kv=a["cache"],
            pages=a["pages"],
            indptr=a["indptr"],
            part_o=a["part_o"],
            part_lse=a["part_lse"],
            scale=a["scale"],
            vscale=a["vscale"],
        ),
    )
    attention.cta(ac)

    gpu.barrier()
    # The v1 ISA hoisted lane%16 across attention, spilled it at entry and
    # reloaded it in each later GEMV. Keep later-stage indices independent.
    fresh = fx.Int32(
        llvm.inline_asm(
            T.i32, [as_ir_value(tid)], "v_mov_b32 $0, $1", "=v,v", has_side_effects=True
        )
    )
    c = context(a, sm, key, bid, fresh)
    # combine + W_UV: a (head, token) a CTA on CTAs 0..55 (uvh: a (half,
    # head, token) on 0..111); o_proj's 448 tasks on the rest (56..255: 3 a CTA
    # on 56..103, else 2; uvh 112..255: 3 a CTA, 4 on 240..255), all weights
    # before the poll
    if bid < NCC:
        combine_uv(c, bid)
    if const_expr(UVH):
        # o_proj on CTAs NCC..255 (144): 3 tasks a CTA, a 4th on the last O4
        # (post-norm CTAs: no gate_up stream waits behind them)
        o = bid - NCC
        if (bid >= NCC) & (bid < BLOCKS - O4):
            dense_n(
                c,
                [o, o + OPC, o + 2 * OPC],
                [None, None, None],
                a["wo"],
                1024,
                "core",
                H,
                "ar",
            )
        if bid >= BLOCKS - O4:
            dense_n(
                c,
                [o, o + OPC, o + 2 * OPC, o + 2 * OPC + O4],
                [None, None, None, None],
                a["wo"],
                1024,
                "core",
                H,
                "ar",
            )
    else:
        if bid >= NC:
            o = bid - NC
            dense_n(
                c,
                [o, o + OPC, o + 2 * OPC],
                [None, None, o + 2 * OPC < H // 16],
                a["wo"],
                1024,
                "core",
                H,
                "ar",
            )
    # the post norm over 28 CTAs BN0.. (o_proj CTAs without a gate_up task):
    # (token, column slice of 4)
    if bid >= BN0:
        normalize_wide(
            c,
            (bid - BN0) // NSL_B,
            (bid - BN0) % NSL_B,
            NSL_B,
            "arout",
            a["residual_mid"],
            a["post_weight"],
            a["residual_out"],
            "ffnrdy",
            "msqb",
            False,
        )
    # gate_up with SiLU * up fused (no activate hand-off): CTAs 0 .. GU
    if bid < GU:
        gate_up_act(c, bid)
    # down: tasks bid and bid + 256 (< 448), both tasks' weights before the poll;
    # except in the last layer its rows go straight to the next layer's AR
    # inbox (the next layer's peers and tag): the xGMI push overlaps the launch
    # gap and the next front has no push_input pass
    if const_expr(not key.last):
        arbytes = key.tp * S * (H // 2) * 8
        par = fx.Int64(uniform(ld_i32(a["epoch"], 0)) % 2) * 2 * arbytes
        c["next_peers"] = [
            preg(b + par, 0, "arin") for b in peer_bases(a["npeers"], key.tp)
        ]
        c["put_next"] = Mailbox(
            step_tag(a["epoch"], a["layer"] + 1) - 1,
            a["health"],
            a["alarm"],
            {n: i for i, n in enumerate(REGIONS)},
        ).put_bf
    dense_n(
        c,
        [bid, bid + 256],
        [None, bid + 256 < H // 16],
        a["wd"],
        FF,
        "act",
        H,
        "out" if key.last else "next",
    )


# Layer i + 1's own front arguments in the fused (back i, front i + 1) launch;
# the rest is layer i's (positions, rank, epoch; residual = its residual_out).
NEXT = [
    "in_weight",
    "wqkv",
    "wqb",
    "wuk",
    "qnw",
    "kvnw",
    "cos_sin",
    "slots",
    "cache",
    "qsi",
    "ksi",
    "query",
    "x",
    "residual_mid",
    "scratch",
    "peers",
    "health",
    "alarm",
]


def precompile():
    """Compile every launch before ranks enter the peer collectives or capture."""
    args = [0] * len(NAMES)
    stream = torch.cuda.current_stream()
    old = os.environ.get("COMPILE_ONLY")
    os.environ["COMPILE_ONLY"] = "1"
    try:
        build(Key(first=True))[0](*args, stream=stream)
        build_bf(Key())(*args, *([0] * len(NEXT)), stream=stream)
        build(Key(last=True))[1](*args, stream=stream)
    finally:
        if old is None:
            del os.environ["COMPILE_ONLY"]
        else:
            os.environ["COMPILE_ONLY"] = old


def run_step(args):
    """The layers' launches: front 0, then (back i, front i + 1) fused, then the
    last back (`len(args)` + 1 launches instead of 2 `len(args)`)."""
    stream = torch.cuda.current_stream()
    values = [[a[n] if n in INT32 else a[n].data_ptr() for n in NAMES] for a in args]
    build(Key(first=True))[0](*values[0], stream=stream)
    fused = build_bf(Key())
    for i in range(len(args) - 1):
        fused(*values[i], *[args[i + 1][n].data_ptr() for n in NEXT], stream=stream)
    build(Key(last=True))[1](*values[-1], stream=stream)


def fused_body(a, n, sm, key):
    back_body(a, sm, key)
    gpu.barrier()
    b = dict(a, residual=a["residual_out"], layer=a["layer"] + 1, **n)
    front_body(b, sm.dense.peek(), key)


def build(key):
    if key in BUILDS:
        return BUILDS[key]
    assert key.tp in (1, 8) and key.ws == 0
    Dense, Both = structures()
    # front kernels differ by ``first`` only, back kernels by ``last`` only
    keyed_f = (key.tp, key.eps, key.ws, key.first, SOURCE)
    keyed_b = (key.tp, key.eps, key.ws, key.last, SOURCE)

    @flyc.kernel(
        name=f"k3_draft_layer_front_tp{key.tp}_v1", known_block_size=[512, 1, 1]
    )
    def front_kernel(
        hidden: Int64,
        residual: Int64,
        in_weight: Int64,
        post_weight: Int64,
        wqkv: Int64,
        wqb: Int64,
        wuk: Int64,
        wuv: Int64,
        wo: Int64,
        wgu: Int64,
        wd: Int64,
        qnw: Int64,
        kvnw: Int64,
        positions: Int64,
        cos_sin: Int64,
        slots: Int64,
        cache: Int64,
        qsi: Int64,
        ksi: Int64,
        query: Int64,
        part_o: Int64,
        part_lse: Int64,
        pages: Int64,
        indptr: Int64,
        scale: Int64,
        vscale: Int64,
        x: Int64,
        residual_mid: Int64,
        residual_out: Int64,
        out: Int64,
        scratch: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        layer: Int32,
        health: Int64,
        alarm: Int64,
        npeers: Int64,
    ):
        _ = keyed_f
        sm = fx.SharedAllocator().allocate(Dense).peek()
        front_body(
            dict(
                hidden=hidden,
                residual=residual,
                in_weight=in_weight,
                post_weight=post_weight,
                wqkv=wqkv,
                wqb=wqb,
                wuk=wuk,
                wuv=wuv,
                wo=wo,
                wgu=wgu,
                wd=wd,
                qnw=qnw,
                kvnw=kvnw,
                positions=positions,
                cos_sin=cos_sin,
                slots=slots,
                cache=cache,
                qsi=qsi,
                ksi=ksi,
                query=query,
                part_o=part_o,
                part_lse=part_lse,
                pages=pages,
                indptr=indptr,
                scale=scale,
                vscale=vscale,
                x=x,
                residual_mid=residual_mid,
                residual_out=residual_out,
                out=out,
                scratch=scratch,
                peers=peers,
                rank=rank,
                epoch=epoch,
                layer=layer,
                health=health,
                alarm=alarm,
                npeers=npeers,
            ),
            sm,
            key,
        )

    @flyc.jit
    def front_launch(
        hidden: Int64,
        residual: Int64,
        in_weight: Int64,
        post_weight: Int64,
        wqkv: Int64,
        wqb: Int64,
        wuk: Int64,
        wuv: Int64,
        wo: Int64,
        wgu: Int64,
        wd: Int64,
        qnw: Int64,
        kvnw: Int64,
        positions: Int64,
        cos_sin: Int64,
        slots: Int64,
        cache: Int64,
        qsi: Int64,
        ksi: Int64,
        query: Int64,
        part_o: Int64,
        part_lse: Int64,
        pages: Int64,
        indptr: Int64,
        scale: Int64,
        vscale: Int64,
        x: Int64,
        residual_mid: Int64,
        residual_out: Int64,
        out: Int64,
        scratch: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        layer: Int32,
        health: Int64,
        alarm: Int64,
        npeers: Int64,
        stream: fx.Stream = STREAM,
    ):
        _ = keyed_f
        front_kernel(
            hidden,
            residual,
            in_weight,
            post_weight,
            wqkv,
            wqb,
            wuk,
            wuv,
            wo,
            wgu,
            wd,
            qnw,
            kvnw,
            positions,
            cos_sin,
            slots,
            cache,
            qsi,
            ksi,
            query,
            part_o,
            part_lse,
            pages,
            indptr,
            scale,
            vscale,
            x,
            residual_mid,
            residual_out,
            out,
            scratch,
            peers,
            rank,
            epoch,
            layer,
            health,
            alarm,
            npeers,
        ).launch(grid=(256,), block=(512,), stream=stream)

    @flyc.kernel(
        name=f"k3_draft_layer_back_tp{key.tp}_v1", known_block_size=[512, 1, 1]
    )
    def back_kernel(
        hidden: Int64,
        residual: Int64,
        in_weight: Int64,
        post_weight: Int64,
        wqkv: Int64,
        wqb: Int64,
        wuk: Int64,
        wuv: Int64,
        wo: Int64,
        wgu: Int64,
        wd: Int64,
        qnw: Int64,
        kvnw: Int64,
        positions: Int64,
        cos_sin: Int64,
        slots: Int64,
        cache: Int64,
        qsi: Int64,
        ksi: Int64,
        query: Int64,
        part_o: Int64,
        part_lse: Int64,
        pages: Int64,
        indptr: Int64,
        scale: Int64,
        vscale: Int64,
        x: Int64,
        residual_mid: Int64,
        residual_out: Int64,
        out: Int64,
        scratch: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        layer: Int32,
        health: Int64,
        alarm: Int64,
        npeers: Int64,
    ):
        _ = keyed_b
        sm = fx.SharedAllocator().allocate(Both)
        back_body(
            dict(
                hidden=hidden,
                residual=residual,
                in_weight=in_weight,
                post_weight=post_weight,
                wqkv=wqkv,
                wqb=wqb,
                wuk=wuk,
                wuv=wuv,
                wo=wo,
                wgu=wgu,
                wd=wd,
                qnw=qnw,
                kvnw=kvnw,
                positions=positions,
                cos_sin=cos_sin,
                slots=slots,
                cache=cache,
                qsi=qsi,
                ksi=ksi,
                query=query,
                part_o=part_o,
                part_lse=part_lse,
                pages=pages,
                indptr=indptr,
                scale=scale,
                vscale=vscale,
                x=x,
                residual_mid=residual_mid,
                residual_out=residual_out,
                out=out,
                scratch=scratch,
                peers=peers,
                rank=rank,
                epoch=epoch,
                layer=layer,
                health=health,
                alarm=alarm,
                npeers=npeers,
            ),
            sm,
            key,
        )

    @flyc.jit
    def back_launch(
        hidden: Int64,
        residual: Int64,
        in_weight: Int64,
        post_weight: Int64,
        wqkv: Int64,
        wqb: Int64,
        wuk: Int64,
        wuv: Int64,
        wo: Int64,
        wgu: Int64,
        wd: Int64,
        qnw: Int64,
        kvnw: Int64,
        positions: Int64,
        cos_sin: Int64,
        slots: Int64,
        cache: Int64,
        qsi: Int64,
        ksi: Int64,
        query: Int64,
        part_o: Int64,
        part_lse: Int64,
        pages: Int64,
        indptr: Int64,
        scale: Int64,
        vscale: Int64,
        x: Int64,
        residual_mid: Int64,
        residual_out: Int64,
        out: Int64,
        scratch: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        layer: Int32,
        health: Int64,
        alarm: Int64,
        npeers: Int64,
        stream: fx.Stream = STREAM,
    ):
        _ = keyed_b
        back_kernel(
            hidden,
            residual,
            in_weight,
            post_weight,
            wqkv,
            wqb,
            wuk,
            wuv,
            wo,
            wgu,
            wd,
            qnw,
            kvnw,
            positions,
            cos_sin,
            slots,
            cache,
            qsi,
            ksi,
            query,
            part_o,
            part_lse,
            pages,
            indptr,
            scale,
            vscale,
            x,
            residual_mid,
            residual_out,
            out,
            scratch,
            peers,
            rank,
            epoch,
            layer,
            health,
            alarm,
            npeers,
        ).launch(grid=(256,), block=(512,), stream=stream)

    BUILDS[key] = (front_launch, back_launch)
    return BUILDS[key]


FUSED: dict = {}


def build_bf(key):
    """The fused launch: layer i's back, then layer i + 1's front (its own
    arguments `NEXT`, prefixed n_)."""
    if key in FUSED:
        return FUSED[key]
    assert key.tp in (1, 8) and key.ws == 0 and not key.first and not key.last
    Dense, Both = structures()
    keyed_bf = (key.tp, key.eps, key.ws, SOURCE)

    @flyc.kernel(
        name=f"k3_draft_layer_backfront_tp{key.tp}_v1", known_block_size=[512, 1, 1]
    )
    def bf_kernel(
        hidden: Int64,
        residual: Int64,
        in_weight: Int64,
        post_weight: Int64,
        wqkv: Int64,
        wqb: Int64,
        wuk: Int64,
        wuv: Int64,
        wo: Int64,
        wgu: Int64,
        wd: Int64,
        qnw: Int64,
        kvnw: Int64,
        positions: Int64,
        cos_sin: Int64,
        slots: Int64,
        cache: Int64,
        qsi: Int64,
        ksi: Int64,
        query: Int64,
        part_o: Int64,
        part_lse: Int64,
        pages: Int64,
        indptr: Int64,
        scale: Int64,
        vscale: Int64,
        x: Int64,
        residual_mid: Int64,
        residual_out: Int64,
        out: Int64,
        scratch: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        layer: Int32,
        health: Int64,
        alarm: Int64,
        npeers: Int64,
        n_in_weight: Int64,
        n_wqkv: Int64,
        n_wqb: Int64,
        n_wuk: Int64,
        n_qnw: Int64,
        n_kvnw: Int64,
        n_cos_sin: Int64,
        n_slots: Int64,
        n_cache: Int64,
        n_qsi: Int64,
        n_ksi: Int64,
        n_query: Int64,
        n_x: Int64,
        n_residual_mid: Int64,
        n_scratch: Int64,
        n_peers: Int64,
        n_health: Int64,
        n_alarm: Int64,
    ):
        _ = keyed_bf
        sm = fx.SharedAllocator().allocate(Both)
        fused_body(
            dict(
                hidden=hidden,
                residual=residual,
                in_weight=in_weight,
                post_weight=post_weight,
                wqkv=wqkv,
                wqb=wqb,
                wuk=wuk,
                wuv=wuv,
                wo=wo,
                wgu=wgu,
                wd=wd,
                qnw=qnw,
                kvnw=kvnw,
                positions=positions,
                cos_sin=cos_sin,
                slots=slots,
                cache=cache,
                qsi=qsi,
                ksi=ksi,
                query=query,
                part_o=part_o,
                part_lse=part_lse,
                pages=pages,
                indptr=indptr,
                scale=scale,
                vscale=vscale,
                x=x,
                residual_mid=residual_mid,
                residual_out=residual_out,
                out=out,
                scratch=scratch,
                peers=peers,
                rank=rank,
                epoch=epoch,
                layer=layer,
                health=health,
                alarm=alarm,
                npeers=npeers,
            ),
            dict(
                in_weight=n_in_weight,
                wqkv=n_wqkv,
                wqb=n_wqb,
                wuk=n_wuk,
                qnw=n_qnw,
                kvnw=n_kvnw,
                cos_sin=n_cos_sin,
                slots=n_slots,
                cache=n_cache,
                qsi=n_qsi,
                ksi=n_ksi,
                query=n_query,
                x=n_x,
                residual_mid=n_residual_mid,
                scratch=n_scratch,
                peers=n_peers,
                health=n_health,
                alarm=n_alarm,
            ),
            sm,
            key,
        )

    @flyc.jit
    def bf_launch(
        hidden: Int64,
        residual: Int64,
        in_weight: Int64,
        post_weight: Int64,
        wqkv: Int64,
        wqb: Int64,
        wuk: Int64,
        wuv: Int64,
        wo: Int64,
        wgu: Int64,
        wd: Int64,
        qnw: Int64,
        kvnw: Int64,
        positions: Int64,
        cos_sin: Int64,
        slots: Int64,
        cache: Int64,
        qsi: Int64,
        ksi: Int64,
        query: Int64,
        part_o: Int64,
        part_lse: Int64,
        pages: Int64,
        indptr: Int64,
        scale: Int64,
        vscale: Int64,
        x: Int64,
        residual_mid: Int64,
        residual_out: Int64,
        out: Int64,
        scratch: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        layer: Int32,
        health: Int64,
        alarm: Int64,
        npeers: Int64,
        n_in_weight: Int64,
        n_wqkv: Int64,
        n_wqb: Int64,
        n_wuk: Int64,
        n_qnw: Int64,
        n_kvnw: Int64,
        n_cos_sin: Int64,
        n_slots: Int64,
        n_cache: Int64,
        n_qsi: Int64,
        n_ksi: Int64,
        n_query: Int64,
        n_x: Int64,
        n_residual_mid: Int64,
        n_scratch: Int64,
        n_peers: Int64,
        n_health: Int64,
        n_alarm: Int64,
        stream: fx.Stream = STREAM,
    ):
        _ = keyed_bf
        bf_kernel(
            hidden,
            residual,
            in_weight,
            post_weight,
            wqkv,
            wqb,
            wuk,
            wuv,
            wo,
            wgu,
            wd,
            qnw,
            kvnw,
            positions,
            cos_sin,
            slots,
            cache,
            qsi,
            ksi,
            query,
            part_o,
            part_lse,
            pages,
            indptr,
            scale,
            vscale,
            x,
            residual_mid,
            residual_out,
            out,
            scratch,
            peers,
            rank,
            epoch,
            layer,
            health,
            alarm,
            npeers,
            n_in_weight,
            n_wqkv,
            n_wqb,
            n_wuk,
            n_qnw,
            n_kvnw,
            n_cos_sin,
            n_slots,
            n_cache,
            n_qsi,
            n_ksi,
            n_query,
            n_x,
            n_residual_mid,
            n_scratch,
            n_peers,
            n_health,
            n_alarm,
        ).launch(grid=(256,), block=(512,), stream=stream)

    FUSED[key] = bf_launch
    return bf_launch
