# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""M1b ``mla_post``: one Kimi-K3 MLA layer from AITER's decode output to
o_proj's input, at TP8 + DCP8 (the DCP group the TP group) or TP8 + DCP1.

AITER leaves every rank the partial attention of its KV shard for all the
gathered heads (``o`` [S, ranks x 12, 512] bf16 and ``lse`` [S, ranks x 12]
f32, strided views). As ``dcp_a2a_lse_reduce`` -> ``_v_up_proj`` -> the gate:

    push (a wave a (token, head), over every CTA)
                        o / lse of rank p's heads -> rank p's CMB: the o row
                        untagged, then the LSE pair as its flag (an LSE of
                        -inf for a row with no local KV or past the batch, as
                        ``_dcp_a2a_pack_send_kernel`` masks it)
    combine (a CTA a (head, 4 tokens))
                        the ranks' partials in rank order, LSE-weighted in
                        fp32 -> bf16 latents in LDS
    W_UV (a wave 16 of the head's 128 rows)
                        the latents -> MXFP4 (aiter's even scale), four FP4
                        MFMAs against W_V -> bf16, times the gate -> ``core``

At DCP1 (``ranks`` 1) ``o`` [S, 12, 512] is already the attention's output, as
vLLM hands it to ``_v_up_proj``: no push, no LSE; the combine stage loads it.
With ``splits`` (M1c ``mla_attn``'s partials: ``o`` / ``lse`` split 0 of dense
[splits, S, 12, 512] bf16 / [splits, S, 12] f32) the combine stage merges a
token's request's splits, LSE-weighted.

K2 (``layer``, ``K2Build.m1b_qlen``) runs these stages in its own launch.
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int32, Int64, T

from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.kda import (
    LOG2E,
    MAX_S,
    bf16r,
    step_tag,
)
from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.mla import (
    GROWS,
    KVL,
    NH,
    even_code,
    pk_fp4_bf16,
)
from vllm.models.kimi_k3.amd.specialized_decode.mono.common.debug import (
    region_ids,
    stamp,
    stamp_begin,
    stamp_flush,
)
from vllm.models.kimi_k3.amd.specialized_decode.mono.common.mx import (
    FP4,
    mfma_scaled,
    pow2,
)
from vllm.models.kimi_k3.amd.specialized_decode.mono.common.ops import (
    CM_SYS,
    bf16_pair,
    bf_hi,
    bf_lo,
    butterfly,
    hw_exp2,
    hw_rcp,
    ld_i32,
    load_ptr64,
    rsrc,
    traced,
    uniform,
)
from vllm.models.kimi_k3.amd.specialized_decode.mono.common.plan import (
    BLOCKS,
    THREADS,
    WAVES,
    KernelAbi,
    first_task,
    key_tuple,
    source_digest,
)
from vllm.models.kimi_k3.amd.specialized_decode.mono.common.ranks import peer_bases
from vllm.models.kimi_k3.amd.specialized_decode.mono.common.sync import Mailbox, preg

VD = 128  # v_head_dim
CW = KVL // 2 + 2  # CMB pairs a (source, token, head): 256 o words, the LSE, pad
OU = KVL // 4  # the LSE's pair in a CMB row
TT = 4  # tokens a combine task
KS = KVL // 128  # W_UV's 128-k MFMA steps
LROW = KVL // 2 + 4  # an LDS latent row (words, padded)
WMAX = 128  # M1c splits a request at most (an LDS weight row)
CB = 8  # combine_splits' o loads in flight a thread (1 with M1c in the launch)
# timeline points: start, push, combine, W_UV
TL_POINTS = 4

assert VD == WAVES * 16 and THREADS == TT * (KVL // 4) and KVL // 2 == 64 * 4

_STREAM = fx.Stream(None)
REGIONS = ("cmb",)
_SOURCES = source_digest(".")


@dataclass(frozen=True)
class MlaPostBuild:
    tokens: int  # S
    qlen: int  # tokens a request (1 + speculative)
    ranks: int = 8  # 1: DCP1, no peer memory
    peer_off: int = 0  # CMB's byte offset in a rank's peer buffer
    timeline: bool = False
    splits: int = 0  # DCP1: M1c's KV splits (0: ``o`` the attention's output)
    ctok: int = 0  # tokens a combine task (0: ``ctok``'s)


ABI = KernelAbi(
    (
        "o",
        "o_sb",
        "o_sh",
        "lse",
        "lse_sb",
        "lse_sh",
        "seq_lens",
        "qsl",
        "w_v",
        "wv_sh",
        "wv_sn",
        "w_vs",
        "vs_sh",
        "vs_sn",
        "vs_sk",
        "gate",
        "core",
        "peers",
        "rank",
        "layer",
        "epoch",
        "health",
        "alarm",
        "tl",
    )
)


def peer_pairs(s: int, ranks: int) -> int:
    return ranks * s * NH * CW


def peer_bytes(s: int, ranks: int) -> int:
    return peer_pairs(s, ranks) * 8


def cmb_pair(c, src, t, h):
    return ((src * c["S"] + t) * NH + h) * CW


# ---------------------------------------------------------------- push
@traced
def stage_push(c):
    """Rank p's heads of this rank's partials -> rank p's CMB: a wave a (token,
    head), the waves over every CTA first. The o row goes untagged (a lane 16 B
    a destination, half the bytes of tagged pairs: the push is xGMI-bound), then
    once the wave's stores drained, the LSE pair, its tag the row's flag."""
    s, q, bid, a = c["S"], c["Q"], c["bid"], c["args"]
    lane, wave = c["lane"], c["wave"]
    nr = s * NH
    nw = BLOCKS * WAVES
    for i in range_constexpr((nr + nw - 1) // nw):
        g = (wave + WAVES * i) * BLOCKS + bid
        if g < nr:
            h = g % NH
            t = g // NH
            # the request's local KV empty, or a row past the batch: LSE -inf
            rows = ld_i32(a["qsl"], s // q)
            empty = (ld_i32(a["seq_lens"], t // q) == 0) | (t >= rows)
            at = cmb_pair(c, c["rank"], t, h)
            lses = []
            for p in range_constexpr(c["R"]):
                gh = p * NH + h
                w = bo.buffer_load(
                    rsrc(a["o"]),
                    (t * a["o_sb"] + gh * a["o_sh"]) // 2 + 4 * lane,
                    vec_width=4,
                    dtype=T.i32,
                )
                bo.buffer_store(
                    w, rsrc(c["cmb"][p].value), at * 2 + 4 * lane, cache_modifier=CM_SYS
                )
                lse = fx.Int32(
                    bo.buffer_load(
                        rsrc(a["lse"]),
                        t * a["lse_sb"] + gh * a["lse_sh"],
                        vec_width=1,
                        dtype=T.i32,
                    )
                )
                lses.append(empty.select(fx.Int32(-8388608), lse))  # -inf
            # the wave's row stores acknowledged before its flags (a wave's
            # own stores: no barrier)
            rocdl.s_waitcnt(vmcnt=0)
            for p in range_constexpr(c["R"]):
                if lane == p:
                    c["put_words"](c["cmb"][p], at + 2 * OU, [lses[p], fx.Int32(0)])


# ---------------------------------------------------------------- combine
def sane(lse):
    """NaN / +inf -> -inf (``_dcp_a2a_unpack_combine_kernel``)."""
    bad = (lse != lse) | (lse == fx.Float32(float("inf")))
    return bad.select(fx.Float32(float("-inf")), lse)


@traced
def stage_combine(c, h, t0):
    """Tokens t0 .. t0 + 4 of head h: thread (token tid / 128, latents
    4 (tid % 128) ..) combines the ranks' partials into the LDS row."""
    s, R, tid = c["S"], c["R"], c["tid"]
    tt = tid // (KVL // 4)
    cc = tid % (KVL // 4)
    t = fx.min(t0 + tt, s - 1)
    if const_expr(R == 1 and c["splits"] > 0):
        combine_splits(c, t0, h)
    if const_expr(R == 1 and c["splits"] == 0):
        a = c["args"]
        o = fx.Vector(
            bo.buffer_load(
                rsrc(a["o"]),
                (t * a["o_sb"] + h * a["o_sh"]) // 2 + 2 * cc,
                vec_width=2,
                dtype=T.i32,
            )
        )
        live = t0 + tt < s
        base = c["lat"] + tt * LROW + 2 * cc
        fx.ptr_store(live.select(o[0], fx.Int32(0)), base)
        fx.ptr_store(live.select(o[1], fx.Int32(0)), base + 1)
    if const_expr(R > 1):
        combine_ranks(c, h, t0, tt, cc, t)
    gpu.barrier()


@traced
def combine_splits(c, t0, h):
    """``stage_combine`` at DCP1 over M1c's partials: token t's request's splits
    (``splits / requests`` of them, request-major), LSE-weighted. Wave tt < tc
    (the task's tokens) turns token t0 + tt's LSEs into weights (LDS ``cw``);
    then a token's WAVES / tc waves (parts) sum the splits part, part + parts,
    .. a thread 8 latents of a token (16 B loads, CB in flight), the parts'
    sums halved through ``cw`` (the upper half's handed over) to part 0."""
    s, q, n, a, tid, tc = c["S"], c["Q"], c["splits"], c["args"], c["tid"], c["TC"]
    per = n // (s // q)
    sp0 = t0 // q * per
    wv, lane = tid // 64, tid % 64
    if const_expr("ardy" in c):
        # M1c in this launch (its VGPRs at the limit): the wave index scalar
        wv = uniform(wv)
        # its CTAs' flags (split, head group of M1c's ``ht`` heads), a thread each
        ht = c["m1c_ht"]
        if tid < per:
            c["poll"]([(c["ardy"], (sp0 + tid) * (NH // ht) + h // ht, 1)])
        gpu.barrier()
    if wv < tc:
        t = fx.min(t0 + wv, s - 1)
        lse = []
        for k in range_constexpr((per + 63) // 64):
            i = lane + 64 * k
            v = fx.Int32(
                bo.buffer_load(
                    rsrc(a["lse"]),
                    ((sp0 + fx.min(i, per - 1)) * s + t) * NH + h,
                    vec_width=1,
                    dtype=T.i32,
                )
            ).bitcast(fx.Float32)
            lse.append((i < per).select(sane(v), fx.Float32(float("-inf"))))
        m = lse[0]
        for k in range_constexpr(1, len(lse)):
            m = fx.max(m, lse[k])
        m = butterfly(m, (32, 16, 8, 4, 2, 1), fx.max)
        m = (m == fx.Float32(float("-inf"))).select(fx.Float32(0.0), m)
        e = [hw_exp2((x - m) * fx.Float32(LOG2E)) for x in lse]
        tot = e[0]
        for k in range_constexpr(1, len(e)):
            tot = tot + e[k]
        tot = butterfly(tot, (32, 16, 8, 4, 2, 1))
        # a padded request (no KV): every split empty, its rows 0
        inv = (tot == fx.Float32(0.0)).select(fx.Float32(0.0), hw_rcp(tot))
        for k in range_constexpr(len(e)):
            fx.ptr_store(e[k] * inv, c["cw"] + wv * WMAX + lane + 64 * k)
    gpu.barrier()
    parts = WAVES // tc
    part, tt = wv // tc, wv % tc
    t = fx.min(t0 + tt, s - 1)
    nj = per // parts  # splits a part
    src = (((sp0 + part) * s + t) * NH + h) * (KVL // 2) + 4 * lane
    wrow = c["cw"] + tt * WMAX + part
    stride = parts * s * NH * KVL * 2  # bytes between a part's splits
    acc = [fx.Float32(0.0)] * 8

    def batch(acc, j0, nb):
        os_ = [
            fx.Vector(
                bo.buffer_load(
                    rsrc(a["o"]),
                    src,
                    vec_width=4,
                    dtype=T.i32,
                    soffset_bytes=(j0 + j) * stride,
                )
            )
            for j in range(nb)
        ]
        # an empty split: weight 0, its o 0
        wts = [fx.ptr_load(wrow + (j0 + j) * parts) for j in range(nb)]
        acc = list(acc)
        for j in range_constexpr(nb):
            o, wt = os_[j], wts[j]
            for k in range_constexpr(4):
                acc[2 * k] = acc[2 * k] + bf_lo(o[k]) * wt
                acc[2 * k + 1] = acc[2 * k + 1] + bf_hi(o[k]) * wt
        return acc

    if const_expr("ardy" in c):
        # with M1c: unrolled, a load a batch (a loop, or wider batches, spill
        # M1c's Q fragment in its loop)
        for j0 in range_constexpr(nj):
            acc = batch(acc, j0, 1)
    else:
        # CB loads in flight a batch (a loop over the batches: the scheduler
        # hoists no further), the rest unrolled
        jb = fx.Int32(0)
        while jb < nj // CB * CB:
            acc = batch(acc, jb, CB)
            jb = jb + CB
        if const_expr(nj % CB):
            acc = batch(acc, nj // CB * CB, nj % CB)
    # parts p >= hf hand their sums to p - hf, halving
    for hf in [parts >> k for k in range(1, parts.bit_length())]:
        red = c["cw"] + TT * WMAX + (tt * 64 + lane) * 8
        if (part >= hf) & (part < 2 * hf):
            at = red + (part - hf) * (tc * 64 * 8)
            for k in range_constexpr(2):
                fx.ptr_store(
                    fx.Vector.from_elements(acc[4 * k : 4 * k + 4], fx.Float32),
                    at + 4 * k,
                )
        gpu.barrier()
        if const_expr(hf > 1):
            at = red + fx.min(part, hf - 1) * (tc * 64 * 8)
            acc = [
                (part < hf).select(acc[i] + fx.ptr_load(at + i), acc[i])
                for i in range(8)
            ]
            gpu.barrier()  # read before the next half's stores
        else:
            acc = [acc[i] + fx.ptr_load(red + i) for i in range(8)]
    if part == 0:
        live = t0 + tt < s
        for k in range_constexpr(4):
            fx.ptr_store(
                live.select(
                    bf16_pair(acc[2 * k], acc[2 * k + 1]).bitcast(fx.Int32),
                    fx.Int32(0),
                ),
                c["lat"] + tt * LROW + 4 * lane + k,
            )


@traced
def combine_ranks(c, h, t0, tt, cc, t):
    """``stage_combine`` at DCP8: the ranks' CMB partials, LSE-weighted."""
    s, R = c["S"], c["R"]
    specs = [(c["cmb_own"], cmb_pair(c, src, t, h) + 2 * OU, 1) for src in range(R)]
    got = c["poll"](specs)
    lse = [sane(got[r][0].bitcast(fx.Float32)) for r in range(R)]
    # each source's row, untagged: its LSE flag above announced it
    rows = [
        fx.Vector(
            bo.buffer_load(
                rsrc(c["cmb_own"].value),
                cmb_pair(c, src, t, h) * 2 + 2 * cc,
                vec_width=2,
                dtype=T.i32,
                cache_modifier=CM_SYS,
            )
        )
        for src in range(R)
    ]
    m = lse[0]
    for r in range_constexpr(1, R):
        m = fx.max(m, lse[r])
    m = (m == fx.Float32(float("-inf"))).select(fx.Float32(0.0), m)
    e = [hw_exp2((lse[r] - m) * fx.Float32(LOG2E)) for r in range(R)]
    tot = e[0]
    for r in range_constexpr(1, R):
        tot = tot + e[r]
    inv = hw_rcp(tot)
    acc = [fx.Float32(0.0)] * 4
    for r in range_constexpr(R):
        wt = (e[r] == fx.Float32(0.0)).select(fx.Float32(0.0), e[r] * inv)
        o = rows[r]
        pv = [bf_lo(o[0]), bf_hi(o[0]), bf_lo(o[1]), bf_hi(o[1])]
        for j in range_constexpr(4):
            pj = (wt == fx.Float32(0.0)).select(fx.Float32(0.0), pv[j])
            acc[j] = acc[j] + pj * wt
    live = t0 + tt < s
    w0 = bf16_pair(acc[0], acc[1]).bitcast(fx.Int32)
    w1 = bf16_pair(acc[2], acc[3]).bitcast(fx.Int32)
    w0 = live.select(w0, fx.Int32(0))
    w1 = live.select(w1, fx.Int32(0))
    base = c["lat"] + tt * LROW + 2 * cc
    fx.ptr_store(w0, base)
    fx.ptr_store(w1, base + 1)


# ---------------------------------------------------------------- W_UV
def wv_loads(c, h):
    """This lane's W_V operands for the wave's 16 rows: (16 B, scale) a step."""
    a, lane, wave = c["args"], c["lane"], c["wave"]
    v = wave * 16 + lane % 16
    kb = lane // 16
    ops = []
    for st in range_constexpr(KS):
        wo = h * a["wv_sh"] + v * a["wv_sn"] + st * 64 + kb * 16
        wv = fx.Vector(
            bo.buffer_load(rsrc(a["w_v"]), wo // 4, vec_width=4, dtype=T.i32)
        )
        so = h * a["vs_sh"] + v * a["vs_sn"] + (st * 4 + kb) * a["vs_sk"]
        sw = fx.Int32(
            bo.buffer_load(rsrc(a["w_vs"]), so // 4, vec_width=1, dtype=T.i32)
        )
        ops.append((wv, (sw >> ((so % 4) * 8)) & 0xFF))
    return ops


@traced
def stage_uv(c, h, t0, ops):
    """The LDS latents of tokens t0 .. t0 + 4 against W_V[h]'s 16 rows of this
    wave -> bf16, times the gate -> ``core``."""
    s, lane, wave, a, tc = c["S"], c["lane"], c["wave"], c["args"], c["TC"]
    tl = lane % 16
    kb = lane // 16
    row = c["lat"] + fx.min(tl, tc - 1) * LROW
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    for st in range_constexpr(KS):
        ws = [
            (tl < tc).select(fx.ptr_load(row + st * 64 + kb * 16 + i), fx.Int32(0))
            for i in range(16)
        ]
        am = fx.Int32(0)
        for w in ws:
            am = fx.max(am, w & 0x7FFF)
            am = fx.max(am, (w >> 16) & 0x7FFF)
        code = fx.max(even_code(am), fx.Int32(1))
        scale = pow2(code)
        xs = []
        for q in range_constexpr(4):
            pk = fx.Int32(0)
            for j in range_constexpr(4):
                pk = pk_fp4_bf16(pk, ws[4 * q + j], scale, j)
            xs.append(pk)
        xv = fx.Vector.from_elements(xs, fx.Int32)
        wv, sa = ops[st]
        acc = mfma_scaled(wv, xv, acc, sa, code, FP4, FP4, 0, 0)
    t = t0 + tl
    v0 = wave * 16 + 4 * kb
    col = h * VD + v0
    tg = fx.min(t, s - 1)
    gw = fx.Vector(
        bo.buffer_load(rsrc(a["gate"]), (tg * GROWS + col) // 2, vec_width=2,
                       dtype=T.i32)
    )  # fmt: skip
    g = [bf_lo(gw[0]), bf_hi(gw[0]), bf_lo(gw[1]), bf_hi(gw[1])]
    out = [bf16r(acc[j]) * g[j] for j in range(4)]
    if (tl < tc) & (t < s):
        # fused into K2 (``c["core_cm"]``): read by other CTAs, device scope
        bo.buffer_store(
            fx.Vector.from_elements(
                [bf16_pair(out[0], out[1]), bf16_pair(out[2], out[3])], fx.Float32
            ),
            rsrc(a["core"]),
            (t * GROWS + col) // 2,
            **({"cache_modifier": c["core_cm"]} if "core_cm" in c else {}),
        )
    gpu.barrier()


def ctok(s: int, q: int, splits: int, ctas: int = BLOCKS) -> int:
    """Tokens a combine task: TT, or with M1c's splits the fewest of 1, 2, 4
    whose parts (WAVES / it a token) divide a request's splits and whose tasks
    fit ``ctas`` (a CTA's reads, a part's splits a thread, over more CTAs)."""
    if not splits:
        return TT
    per = splits // (s // q)
    ok = [tc for tc in (1, 2, 4) if per % (WAVES // tc) == 0]
    return next((tc for tc in ok if tasks(s, tc) <= ctas), ok[-1])


def tasks(s: int, tc: int = TT) -> int:
    """Combine tasks: a (head, ``tc`` tokens), head-minor (task h + 12 tile)."""
    return NH * ((s + tc - 1) // tc)


@traced
def combine_tasks(c, ctas=BLOCKS, done=None):
    """The combine + W_UV tasks over CTAs 0 .. ``ctas``; ``done(task)`` after
    each task's ``core`` rows are stored (K2's fused M1b: their flag)."""
    s, bid, tc = c["S"], c["bid"], c["TC"]
    for task in range(first_task(bid, 0), tasks(s, tc), ctas):
        h = task % NH
        t0 = task // NH * tc
        ops = wv_loads(c, h)
        stage_combine(c, h, t0)
        stage_uv(c, h, t0, ops)
        if const_expr(done is not None):
            done(task)


# ---------------------------------------------------------------- kernel
_BUILDS: dict = {}


def build(key: MlaPostBuild):
    if key in _BUILDS:
        return _BUILDS[key]
    s, R = key.tokens, key.ranks
    assert 1 <= s <= MAX_S and s % key.qlen == 0 and 1 <= R <= 8

    @fx.struct
    class Smem:
        lat: fx.Array[fx.Int32, TT * LROW, 16]
        cw: fx.Array[fx.Float32, TT * (WMAX + KVL) if key.splits else 1, 16]
        tls: fx.Array[fx.Int64, TL_POINTS, 16]

    keyed = key_tuple(key, _SOURCES)
    name = f"k3_mono_mla_post_s{s}_q{key.qlen}_r{R}_t{int(key.timeline)}"
    tc = key.ctok or ctok(s, key.qlen, key.splits)
    assert tc in (1, 2, 4) and (
        not key.splits or key.splits // (s // key.qlen) % (WAVES // tc) == 0
    )
    name += f"_sp{key.splits}_c{tc}" if key.splits else ""
    assert not key.splits or (R == 1 and key.splits % (s // key.qlen) == 0)
    assert key.splits // (s // key.qlen) % 2 == 0
    assert key.splits // (s // key.qlen) <= WMAX

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def mla_post(
        o: Int64,
        o_sb: Int32,
        o_sh: Int32,
        lse: Int64,
        lse_sb: Int32,
        lse_sh: Int32,
        seq_lens: Int64,
        qsl: Int64,
        w_v: Int64,
        wv_sh: Int32,
        wv_sn: Int32,
        w_vs: Int64,
        vs_sh: Int32,
        vs_sn: Int32,
        vs_sk: Int32,
        gate: Int64,
        core: Int64,
        peers: Int64,
        rank: Int32,
        layer: Int32,
        epoch: Int64,
        health: Int64,
        alarm: Int64,
        tl: Int64,
    ):
        _ = keyed
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lds = fx.SharedAllocator().allocate(Smem).peek()
        mb = Mailbox(step_tag(epoch, layer) - 1, health, alarm, region_ids(REGIONS))
        # DCP1 reads no peer memory (``peers`` may be 0)
        bases = peer_bases(peers, R) if R > 1 else []
        own = load_ptr64(peers, rank) if R > 1 else peers
        c = {
            "S": s,
            "Q": key.qlen,
            "R": R,
            "splits": key.splits,
            "TC": tc,
            "tid": tid,
            "bid": bid,
            "lane": tid % 64,
            "wave": tid // 64,
            "rank": uniform(rank),
            "put_words": mb.put_words,
            "poll": mb.poll,
            "lat": lds.lat.ptr,
            "cw": lds.cw.ptr,
            "cmb": [preg(b, key.peer_off, "cmb") for b in bases],
            "cmb_own": preg(own, key.peer_off, "cmb"),
            "args": {
                "o": o,
                "o_sb": o_sb,
                "o_sh": o_sh,
                "lse": lse,
                "lse_sb": lse_sb,
                "lse_sh": lse_sh,
                "seq_lens": seq_lens,
                "qsl": qsl,
                "w_v": w_v,
                "wv_sh": wv_sh,
                "wv_sn": wv_sn,
                "w_vs": w_vs,
                "vs_sh": vs_sh,
                "vs_sn": vs_sn,
                "vs_sk": vs_sk,
                "gate": gate,
                "core": core,
            },
        }
        on, tls = key.timeline, lds.tls.ptr
        stamp_begin(on, tls, tid, TL_POINTS)
        if const_expr(R > 1):
            stage_push(c)
        stamp(on, tls, tid, 1)
        combine_tasks(c)
        stamp(on, tls, tid, 3)
        stamp_flush(on, tls, tl, tid, bid, TL_POINTS)

    @flyc.jit
    def launch(
        o: Int64,
        o_sb: Int32,
        o_sh: Int32,
        lse: Int64,
        lse_sb: Int32,
        lse_sh: Int32,
        seq_lens: Int64,
        qsl: Int64,
        w_v: Int64,
        wv_sh: Int32,
        wv_sn: Int32,
        w_vs: Int64,
        vs_sh: Int32,
        vs_sn: Int32,
        vs_sk: Int32,
        gate: Int64,
        core: Int64,
        peers: Int64,
        rank: Int32,
        layer: Int32,
        epoch: Int64,
        health: Int64,
        alarm: Int64,
        tl: Int64,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        mla_post(
            o,
            o_sb,
            o_sh,
            lse,
            lse_sb,
            lse_sh,
            seq_lens,
            qsl,
            w_v,
            wv_sh,
            wv_sn,
            w_vs,
            vs_sh,
            vs_sn,
            vs_sk,
            gate,
            core,
            peers,
            rank,
            layer,
            epoch,
            health,
            alarm,
            tl,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    ABI.check(mla_post, launch)
    _BUILDS[key] = launch
    return launch


def host_args(ranks, o, lse, seq_lens, qsl, w_v, w_v_scale, gate) -> list:
    """ABI's ``o`` .. ``gate`` (``mla_post``'s, or K2's fused M1b's)."""
    assert o.stride(-1) == 1 and w_v.stride(-1) == 1
    assert o.dtype == torch.bfloat16
    assert lse is not None or ranks == 1  # DCP1: M1c's partials carry one
    assert lse is None or lse.dtype == torch.float32
    assert seq_lens.dtype == torch.int32 and qsl.dtype == torch.int32
    assert gate.is_contiguous()
    return [
        o.data_ptr(),
        o.stride(0),
        o.stride(1),
        0 if lse is None else lse.data_ptr(),
        0 if lse is None else lse.stride(0),
        0 if lse is None else lse.stride(1),
        seq_lens.data_ptr(),
        qsl.data_ptr(),
        w_v.data_ptr(),
        w_v.stride(0),
        w_v.stride(1),
        w_v_scale.data_ptr(),
        w_v_scale.stride(0),
        w_v_scale.stride(1),
        w_v_scale.stride(2),
        gate.data_ptr(),
    ]


def mla_post(
    key: MlaPostBuild,
    *,
    o,
    lse,
    seq_lens,
    qsl,
    w_v,
    w_v_scale,
    gate,
    core,
    peers,
    rank,
    layer,
    epoch,
    health,
    alarm,
    tl=None,
    scratch=None,
):
    """Host launcher. ``o`` [S, >= ranks x 12, 512] bf16 and ``lse`` [S, >=
    ranks x 12] f32, rows and heads strided (ranks 1: no ``lse``, ``peers``
    may be None); ``seq_lens`` [requests] and ``qsl``
    [requests + 1] int32 (the local KV lengths, the query starts); ``w_v`` (12,
    128, 256) u8 MXFP4 and ``w_v_scale`` (12, 128, 16) e8m0, any strides but
    W_V's last; ``gate`` / ``core`` [S, 1536] bf16, dense."""
    del scratch
    assert core.is_contiguous()
    f = build(key)
    f(
        *host_args(key.ranks, o, lse, seq_lens, qsl, w_v, w_v_scale, gate),
        core.data_ptr(),
        0 if peers is None else peers.data_ptr(),
        rank,
        layer,
        epoch.data_ptr(),
        health.data_ptr(),
        alarm,
        0 if tl is None else tl.data_ptr(),
        stream=torch.cuda.current_stream(),
    )
