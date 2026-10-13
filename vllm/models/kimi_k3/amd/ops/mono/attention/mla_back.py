# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""M1b ``mla_post``: one Kimi-K3 MLA layer from AITER's decode output to
o_proj's input, at TP8 + DCP8 (the DCP group the TP group) or TP8 + DCP1.

AITER leaves every rank the partial attention of its KV shard for all the
gathered heads (``o`` [S, ranks x 12, 512] bf16 and ``lse`` [S, ranks x 12]
f32, strided views). As ``dcp_a2a_lse_reduce`` -> ``_v_up_proj`` -> the gate:

    push (every CTA)    o / lse of rank p's heads -> rank p's CMB (an LSE of
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
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import Int32, Int64, T

from vllm.models.kimi_k3.amd.ops.mono.attention.kda import (
    LOG2E,
    MAX_S,
    bf16r,
    step_tag,
)
from vllm.models.kimi_k3.amd.ops.mono.attention.mla import (
    GROWS,
    KVL,
    NH,
    even_code,
    pk_fp4_bf16,
)
from vllm.models.kimi_k3.amd.ops.mono.common.debug import (
    region_ids,
    stamp,
    stamp_begin,
    stamp_flush,
)
from vllm.models.kimi_k3.amd.ops.mono.common.mx import FP4, mfma_scaled, pow2
from vllm.models.kimi_k3.amd.ops.mono.common.ops import (
    bf16_pair,
    bf_hi,
    bf_lo,
    hw_exp2,
    hw_rcp,
    ld_i32,
    load_ptr64,
    rsrc,
    traced,
    uniform,
)
from vllm.models.kimi_k3.amd.ops.mono.common.plan import (
    BLOCKS,
    THREADS,
    WAVES,
    KernelAbi,
    first_task,
    key_tuple,
    source_digest,
)
from vllm.models.kimi_k3.amd.ops.mono.common.ranks import peer_bases
from vllm.models.kimi_k3.amd.ops.mono.common.sync import Mailbox, preg

VD = 128  # v_head_dim
CW = KVL // 2 + 2  # CMB pairs a (source, token, head): 256 o words, the LSE, pad
OU = KVL // 4  # push units a (token, head): 2 o words each, then the LSE
TT = 4  # tokens a combine task
KS = KVL // 128  # W_UV's 128-k MFMA steps
LROW = KVL // 2 + 4  # an LDS latent row (words, padded)
# timeline points: start, push, combine, W_UV
TL_POINTS = 4

assert VD == WAVES * 16 and THREADS == TT * (KVL // 4)

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
    """Rank p's heads of this rank's partials -> rank p's CMB: a thread a unit
    (2 o words, or the masked LSE) a destination."""
    s, q, tid, bid, a = c["S"], c["Q"], c["tid"], c["bid"], c["args"]
    nu = s * NH * (OU + 1)
    rounds = (nu + BLOCKS * THREADS - 1) // (BLOCKS * THREADS)
    for i in range_constexpr(rounds):
        g = bid * THREADS + tid + BLOCKS * THREADS * i
        u = fx.min(g, nu - 1)
        j = u % (OU + 1)
        h = (u // (OU + 1)) % NH
        t = u // (NH * (OU + 1))
        # the request's local KV empty, or a row past the batch: LSE -inf
        rows = ld_i32(a["qsl"], s // q)
        empty = (ld_i32(a["seq_lens"], t // q) == 0) | (t >= rows)
        for p in range_constexpr(c["R"]):
            gh = p * NH + h
            at = cmb_pair(c, c["rank"], t, h)
            if (g < nu) & (j < OU):
                w = fx.Vector(
                    bo.buffer_load(
                        rsrc(a["o"]),
                        (t * a["o_sb"] + gh * a["o_sh"]) // 2 + 2 * j,
                        vec_width=2,
                        dtype=T.i32,
                    )
                )
                c["put_words"](c["cmb"][p], at + 2 * j, [w[0], w[1]])
            if (g < nu) & (j == OU):
                lse = fx.Int32(
                    bo.buffer_load(
                        rsrc(a["lse"]),
                        t * a["lse_sb"] + gh * a["lse_sh"],
                        vec_width=1,
                        dtype=T.i32,
                    )
                )
                lse = empty.select(fx.Int32(-8388608), lse)  # -inf
                c["put_words"](c["cmb"][p], at + 2 * OU, [lse, fx.Int32(0)])


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
    if const_expr(R == 1):
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
def combine_ranks(c, h, t0, tt, cc, t):
    """``stage_combine`` at DCP8: the ranks' CMB partials, LSE-weighted."""
    s, R = c["S"], c["R"]
    specs = []
    for src in range_constexpr(R):
        at = cmb_pair(c, src, t, h)
        specs += [(c["cmb_own"], at + 2 * OU, 1), (c["cmb_own"], at + 2 * cc, 2)]
    got = c["poll"](specs)
    lse = [sane(got[2 * r][0].bitcast(fx.Float32)) for r in range(R)]
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
        o = got[2 * r + 1]
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
    s, lane, wave, a = c["S"], c["lane"], c["wave"], c["args"]
    tl = lane % 16
    kb = lane // 16
    row = c["lat"] + fx.min(tl, TT - 1) * LROW
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    for st in range_constexpr(KS):
        ws = [
            (tl < TT).select(fx.ptr_load(row + st * 64 + kb * 16 + i), fx.Int32(0))
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
    tc = fx.min(t, s - 1)
    gw = fx.Vector(
        bo.buffer_load(rsrc(a["gate"]), (tc * GROWS + col) // 2, vec_width=2,
                       dtype=T.i32)
    )  # fmt: skip
    g = [bf_lo(gw[0]), bf_hi(gw[0]), bf_lo(gw[1]), bf_hi(gw[1])]
    out = [bf16r(acc[j]) * g[j] for j in range(4)]
    if (tl < TT) & (t < s):
        bo.buffer_store(
            fx.Vector.from_elements(
                [bf16_pair(out[0], out[1]), bf16_pair(out[2], out[3])], fx.Float32
            ),
            rsrc(a["core"]),
            (t * GROWS + col) // 2,
        )
    gpu.barrier()


@traced
def combine_tasks(c):
    s, bid = c["S"], c["bid"]
    tiles = (s + TT - 1) // TT
    for task in range(first_task(bid, 0), NH * tiles, BLOCKS):
        h = task % NH
        t0 = task // NH * TT
        ops = wv_loads(c, h)
        stage_combine(c, h, t0)
        stage_uv(c, h, t0, ops)


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
        tls: fx.Array[fx.Int64, TL_POINTS, 16]

    keyed = key_tuple(key, _SOURCES)
    name = f"k3_mono_mla_post_s{s}_q{key.qlen}_r{R}_t{int(key.timeline)}"

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
            "tid": tid,
            "bid": bid,
            "lane": tid % 64,
            "wave": tid // 64,
            "rank": uniform(rank),
            "put_words": mb.put_words,
            "poll": mb.poll,
            "lat": lds.lat.ptr,
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
    assert o.stride(-1) == 1 and w_v.stride(-1) == 1
    assert o.dtype == torch.bfloat16
    assert (lse is None) == (key.ranks == 1)
    assert lse is None or lse.dtype == torch.float32
    assert seq_lens.dtype == torch.int32 and qsl.dtype == torch.int32
    assert gate.is_contiguous() and core.is_contiguous()
    f = build(key)
    f(
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
