# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K2a ``attn_post``: one Kimi-K3 KDA layer from the gated core output to the
MoE's input row -- o_proj, its TP all-reduce inside the kernel, and the MLP
AttnRes seam.

One launch a layer and rank, ``BLOCKS`` x ``THREADS``:

    gemv CTAs 56 ..   o_proj (448 x 16 rows, K = 1536 local): bf16 GEMV of the
                      core rows (plain loads: the previous launch wrote them) ->
                      bf16 partial -> pushed to every rank's AR region (system
                      scope, this rank's slot)
    attn CTAs 0 .. 55 per 128 columns: the 8 ranks' partials summed in rank
                      order (fp32 -> bf16: the custom all-reduce's result) ->
                      LDS delta; then K1's AttnRes stages with the MLP's weights:
                      prefix + delta -> prefix out, the sources' softmax mix,
                      post_attention_layernorm -> the MoE input x

The AR region is double-buffered by the step epoch's parity: a rank a step
ahead writes the other half, so no peer's unread partials are overwritten.
"""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import T

from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.kda import (
    COLS,
    HIDDEN,
    LDS_PAD,
    NA,
    PROJ,
    QC,
    ROWS,
    bf4_words,
    bf16r,
)
from vllm.models.kimi_k3.amd.specialized_decode.mono.common.ops import (
    CM_NT,
    bf2_f32,
    bf16_pair,
    bf16_round,
    bf_hi,
    bf_lo,
    butterfly,
    mfma_bf16,
    row_sum,
    rsrc,
    traced,
)
from vllm.models.kimi_k3.amd.specialized_decode.mono.common.plan import (
    BLOCKS,
    THREADS,
    WAVES,
    pair_layout,
)
from vllm.models.kimi_k3.amd.specialized_decode.mono.common.ranks import sum_partials
from vllm.models.kimi_k3.amd.specialized_decode.mono.common.sync import (
    Addr,
    poll_sent,
    put_sent,
    rearm_sent,
    sent_word,
)

TP = 8
OWN_COLS = HIDDEN // TP  # two-shot: the columns a rank reduces (7 slices)


def two_shot(s):
    """Above 8 rows the AR scatters each partial to its columns' owner rank,
    which reduces and broadcasts: 2/8 of the one-shot's bytes a link."""
    return s > 8


def peer_at(peers, p):
    """``peers[p]`` (an Addr a rank) for a traced rank ``p``."""
    v = peers[0].value
    for q in range(1, TP):
        v = (p == q).select(peers[q].value, v)
    return Addr(peers[0].region, peers[0].space, v)


OROWS = HIDDEN  # o_proj rows (full hidden, a rank's K slice)
OTASKS = OROWS // ROWS  # 448
OK = PROJ  # K = 1536 a rank
OKCH = OK // 128  # 12 chunks
GEMV0 = NA  # the first gemv CTA
GEMV_N = BLOCKS - NA  # 200
TL_POINTS = 6  # start, gemv, pushed, reduced, slice/gate, mix

_STREAM = fx.Stream(None)


def sent_on(c, which="ar"):
    """(OPT_SENTS) This build's all-reduce ``which`` ("ar": o_proj's,
    "latr", "final") goes through its sentinel region (``layer.k2_sent``; a
    plain call: a method call on ``c`` inside a traced branch would make FlyDSL
    carry ``c`` as branch state)."""
    return c.get("sent_" + which, False)


def sent_ix(src, col, t, s, width):
    """(OPT_SENT) Word index of (src rank, pair at ``col``, token t), blocked
    [src][16-col group][token][8 pairs]; shared by push and reduce."""
    return ((src * (width // 16) + col // 16) * s + t) * 8 + (col % 16) // 2


def sent_item(idx, s):
    """(OPT_SENT) A reduce item's (token, column pair), so consecutive items
    read consecutive ``sent_ix`` words."""
    g = idx // (8 * s)
    r = idx % (8 * s)
    return r // 8, g * 8 + r % 8


def sent_sum(c, own, col, t, width, rearm=True):
    """The ``TP`` ranks' sentinel words of (col pair, t) in ``own`` summed in
    fp32 in rank order (as ``sum_partials``), re-armed unless ``rearm`` is
    off (the caller re-arms them later)."""
    s = c["S"]
    ws = poll_sent(
        c["mb"], own, [sent_ix(src, col, t, s, width) for src in range(TP)], rearm=rearm
    )
    acc0, acc1 = bf2_f32(ws[0])
    for src in range_constexpr(1, TP):
        lo, hi = bf2_f32(ws[src])
        acc0 = acc0 + lo
        acc1 = acc1 + hi
    return acc0, acc1


def ar_half_pairs(s):
    """One parity's AR region: ``[src rank][token][column pair]``."""
    return TP * s * HIDDEN // 2


def scratch_layout(s: int, nblocks: int) -> dict:
    ns = nblocks + 1
    pairs = {
        "part": NA * s * ns * 2,
        "wgt": s * (ns + 1),
        "msq": NA * s,
        "xrdy": NA,
    }
    return pair_layout(pairs.items())


def o_loads(c, rg):
    """Rows 16 rg .. of o_proj, this wave's K chunks (wave, wave + 8 < 12): a lane
    64 B of its row a chunk."""
    lane, wave = c["lane"], c["wave"]
    r_w = rsrc(c["args"]["w_o"])
    row = rg * ROWS + lane % ROWS
    g = lane // ROWS
    out = []
    for i in range(2):
        ch = fx.min(wave + WAVES * i, OKCH - 1)
        base = (row * OK + ch * 128 + g * 8) // 2
        out.append(
            [
                fx.Vector(
                    bo.buffer_load(
                        r_w,
                        base + 16 * q,
                        vec_width=4,
                        dtype=T.i32,
                        cache_modifier=CM_NT,
                    )
                )
                for q in range(4)
            ]
        )
    return out


@traced
def load_core(c, t0, n, cm=0):
    """Core rows t0 .. t0 + n (of S x 1536 bf16) -> LDS, rows padded; ``cm``
    CM_DEV when this launch's own CTAs wrote them (K2's fused M1b)."""
    tid = c["tid"]
    words = n * OK // 8
    row = OK + LDS_PAD
    for i in range_constexpr((words + THREADS - 1) // THREADS):
        e = fx.min(tid + THREADS * i, words - 1)
        v = fx.Vector(
            bo.buffer_load(
                rsrc(c["args"]["core"]),
                (t0 * (OK // 8) + e) * 4,
                vec_width=4,
                dtype=T.i32,
                cache_modifier=cm,
            )
        )
        t = e // (OK // 8)
        k = e % (OK // 8) * 8
        if tid + THREADS * i < words:
            fx.ptr_store(v, c["xl"] + (t * row + k) // 2)
    gpu.barrier()


@traced
def stage_o(c, rg, ops, t0, n):
    """Rows 16 rg .. for the core tile t0 .. t0 + n: GEMV -> bf16 partials ->
    every rank's AR region, this rank's slot (thread: token, row pair; a put a
    peer)."""
    s, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    g = lane // ROWS
    t = fx.min(lane % ROWS, n - 1)
    row = OK + LDS_PAD
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    for i in range_constexpr(2):
        ch = wave + WAVES * i
        live = ch < OKCH
        k = fx.min(ch, OKCH - 1) * 128 + g * 8
        for q in range_constexpr(4):
            xb = fx.Vector(
                fx.ptr_load(
                    c["xl"] + (t * row + k + 32 * q) // 2,
                    result_type=fx.Vector.make_type(4, fx.Int32),
                )
            )
            xb = fx.Vector.from_elements(
                [live.select(xb[d], fx.Int32(0)) for d in range(4)], fx.Int32
            )
            acc = mfma_bf16(
                ops[i][q].bitcast(fx.BFloat16), xb.bitcast(fx.BFloat16), acc
            )
    fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
    gpu.barrier()
    if tid < n * (ROWS // 2):
        tt = tid // (ROWS // 2)
        rp = tid % (ROWS // 2)
        v0 = bf16_round(row_sum(red, 2 * rp, tt))
        v1 = bf16_round(row_sum(red, 2 * rp + 1, tt))
        at = (c["rank"] * s + t0 + tt) * HIDDEN + rg * ROWS + 2 * rp
        if const_expr(sent_on(c)):
            # (OPT_SENT) A plain word to the sentinel region
            si = sent_ix(c["rank"], rg * ROWS + 2 * rp, t0 + tt, s, HIDDEN)
            word = sent_word(bf16_pair(v0, v1))
            if const_expr(two_shot(s)):
                owner = rg * ROWS // OWN_COLS
                put_sent(peer_at(c["ars_peer"], owner), si, word)
            else:
                for p in range_constexpr(TP):
                    put_sent(c["ars_peer"][p], si, word)
        elif const_expr(two_shot(s)):
            owner = rg * ROWS // OWN_COLS
            c["put_bf"](peer_at(c["ar_peer"], owner), at, [v0, v1])
        else:
            for p in range_constexpr(TP):
                c["put_bf"](c["ar_peer"][p], at, [v0, v1])
    gpu.barrier()


@traced
def stage_reduce(c, task):
    """Columns 128 task ..: the 8 ranks' partials, rank order, -> bf16 delta in
    LDS (thread: token, column pair)."""
    s, tid = c["S"], c["tid"]
    n = s * (COLS // 2)
    for i in range_constexpr((n + THREADS - 1) // THREADS):
        idx = tid + THREADS * i
        if idx < n:
            if const_expr(sent_on(c)):
                t, cp = sent_item(idx, s)
            else:
                t = idx // (COLS // 2)
                cp = idx % (COLS // 2)
            col = task * COLS + 2 * cp
            if const_expr(two_shot(s)):
                lo, hi = reduce_two_shot(c, task, t, col)
            elif const_expr(sent_on(c)):
                # re-armed after the mix (``rearm_ar``)
                lo, hi = sent_sum(c, c["ars_own"], col, t, HIDDEN, rearm=False)
            else:
                lo, hi = sum_partials(
                    c["poll"],
                    c["ar_own"],
                    lambda src, t=t, col=col: ((src * s + t) * HIDDEN + col) // 2,
                    TP,
                )
            fx.ptr_store(bf16_round(lo), c["delta_lds"] + (t * COLS + 2 * cp))
            fx.ptr_store(bf16_round(hi), c["delta_lds"] + (t * COLS + 2 * cp + 1))
    gpu.barrier()


@traced
def reduce_two_shot(c, task, t, col):
    """The owner of ``col`` sums the 8 scattered partials (rank order, as the
    one-shot) and puts the bf16 sum to the other ranks' slot [owner]; they
    poll it there (a slot the scatter leaves free off the owner)."""
    s = c["S"]
    owner = task * COLS // OWN_COLS
    at = (owner * s + t) * HIDDEN + col
    lo, hi = fx.Float32(0.0), fx.Float32(0.0)
    if const_expr(sent_on(c)):
        # (OPT_SENT) The scatter and the broadcast through the
        # sentinel region, the same slots as the tagged pairs'
        si = sent_ix(owner, col, t, s, HIDDEN)
        if owner == c["rank"]:
            a0, a1 = sent_sum(c, c["ars_own"], col, t, HIDDEN, rearm=False)
            word = sent_word(bf16_pair(a0, a1))
            for p in range_constexpr(TP):
                if p != c["rank"]:
                    put_sent(c["ars_peer"][p], si, word)
            lo, hi = a0, a1
        else:
            lo, hi = bf2_f32(poll_sent(c["mb"], c["ars_own"], [si], rearm=False)[0])
        return lo, hi
    if owner == c["rank"]:
        a0, a1 = sum_partials(
            c["poll"],
            c["ar_own"],
            lambda src: ((src * s + t) * HIDDEN + col) // 2,
            TP,
        )
        for p in range_constexpr(TP):
            if p != c["rank"]:
                c["put_bf"](c["ar_peer"][p], at, [a0, a1])
        lo, hi = a0, a1
    else:
        lo, hi = bf2_f32(c["poll"]([(c["ar_own"], at // 2, 1)])[0][0])
    return lo, hi


# ---------------------------------------------------------------- pre-AR preload
# (``layer.OPT_PRELOAD``)
# The slice's block / prefix / norm / qk loads (independent of the
# all-reduce) issued before ``stage_reduce``'s poll; ``stage_slice_pre`` is
# ``kda.stage_slice`` on the preloaded words, same arithmetic and order.


def _raw4(ptr, i):
    """bf16 elements i .. i + 4 as two raw words (``kda.ld_bf4``'s load)."""
    return fx.Vector(bo.buffer_load(rsrc(ptr), i // 2, vec_width=2, dtype=T.i32))


def _f4(w):
    return [bf_lo(w[0]), bf_hi(w[0]), bf_lo(w[1]), bf_hi(w[1])]


def slice_loads(c, task):
    """``stage_slice``'s loads of slice ``task`` that need no hand-off: each
    round's block and prefix words, then the round-invariant ares_nw / ares_qk
    words (a live thread's columns are the same every round: THREADS % QC == 0)."""
    s, ns, nb, tid = c["S"], c["NS"], c["NB"], c["tid"]
    a = c["args"]
    c0 = task * COLS
    n = s * ns * QC
    blk, pre = [], []
    for i in range(-(-n // THREADS)):
        e = fx.min(tid + THREADS * i, n - 1)
        row = e // QC
        t = row // ns
        src = row % ns
        col = c0 + (e % QC) * 4
        blk.append(
            _raw4(
                a["blocks"], t * a["blk_sm"] + fx.min(src, nb - 1) * a["blk_sr"] + col
            )
        )
        if const_expr(not c.get("reset", False)):
            pre.append(_raw4(a["prefix"], t * HIDDEN + col))
    col0 = c0 + (fx.min(tid, n - 1) % QC) * 4
    return {
        "blk": blk,
        "pre": pre,
        "w": _raw4(a["ares_nw"], col0),
        "q": _raw4(a["ares_qk"], col0),
    }


@traced
def stage_slice_pre(c, task, ld):
    """``kda.stage_slice`` for K2 (delta in LDS, no block write) on the
    preloaded words ``ld`` (``slice_loads``)."""
    s, ns, nb, tid, lane = c["S"], c["NS"], c["NB"], c["tid"], c["lane"]
    a = c["args"]
    assert c["delta"] and c.get("delta_lds") is not None
    assert c.get("write_idx", -1) < 0
    c0 = task * COLS
    n = s * ns * QC
    w = _f4(ld["w"])
    q = _f4(ld["q"])
    for i in range_constexpr((n + THREADS - 1) // THREADS):
        e = fx.min(tid + THREADS * i, n - 1)
        live = (tid + THREADS * i) < n
        row = e // QC
        t = row // ns
        src = row % ns
        col = c0 + (e % QC) * 4
        is_pre = src == nb
        blk = _f4(ld["blk"][i])
        dl = [
            fx.ptr_load(c["delta_lds"] + (t * COLS + (e % QC) * 4 + j))
            for j in range(4)
        ]
        if const_expr(c.get("reset", False)):
            pre = dl
        else:
            pre = _f4(ld["pre"][i])
            pre = [bf16r(pre[j] + dl[j]) for j in range(4)]
        if live & is_pre:
            bo.buffer_store(bf4_words(pre), rsrc(a["prefix"]), (t * HIDDEN + col) // 2)
        v = [
            live.select(is_pre.select(pre[j], blk[j]), fx.Float32(0.0))
            for j in range(4)
        ]
        if const_expr(c["vals_all"]):
            if live:
                for j in range_constexpr(4):
                    fx.ptr_store(v[j], c["vals"] + (row * COLS + (e % QC) * 4 + j))
        else:
            if live & is_pre:
                for j in range_constexpr(4):
                    fx.ptr_store(v[j], c["vals"] + (t * COLS + (e % QC) * 4 + j))
        sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3]
        dt = (
            v[0] * (w[0] * q[0])
            + v[1] * (w[1] * q[1])
            + v[2] * (w[2] * q[2])
            + v[3] * (w[3] * q[3])
        )
        sq = butterfly(sq, (16, 8, 4, 2, 1))
        dt = butterfly(dt, (16, 8, 4, 2, 1))
        if live & (lane % 32 == 0):
            at = ((task * s + t) * ns + src) * 2
            c["put"](c["part"], at, sq)
            c["put"](c["part"], at + 1, dt)
    gpu.barrier()


@traced
def rearm_ar(c, task):
    """(OPT_SENT) Re-arm the words ``stage_reduce`` read for ``task``,
    after the mix (off the critical path)."""
    s, tid = c["S"], c["tid"]
    n = s * (COLS // 2)
    for i in range_constexpr((n + THREADS - 1) // THREADS):
        idx = tid + THREADS * i
        if idx < n:
            t, cp = sent_item(idx, s)
            col = task * COLS + 2 * cp
            if const_expr(two_shot(s)):
                owner = task * COLS // OWN_COLS
                if owner == c["rank"]:
                    rearm_sent(
                        c["ars_own"],
                        [sent_ix(src, col, t, s, HIDDEN) for src in range(TP)],
                    )
                else:
                    rearm_sent(c["ars_own"], [sent_ix(owner, col, t, s, HIDDEN)])
            else:
                rearm_sent(
                    c["ars_own"], [sent_ix(src, col, t, s, HIDDEN) for src in range(TP)]
                )
