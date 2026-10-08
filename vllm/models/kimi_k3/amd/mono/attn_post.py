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

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import Int32, Int64, T

from vllm.models.kimi_k3.amd.mono.common.abi import KernelAbi
from vllm.models.kimi_k3.amd.mono.common.build_key import key_tuple
from vllm.models.kimi_k3.amd.mono.common.execution import BLOCKS, THREADS, WAVES
from vllm.models.kimi_k3.amd.mono.common.layout import pair_layout
from vllm.models.kimi_k3.amd.mono.common.ops import (
    CM_NT,
    bf16_round,
    ld_i32,
    mfma_bf16,
    row_sum,
    rsrc,
    traced,
    uniform,
)
from vllm.models.kimi_k3.amd.mono.common.ranks import peer_bases, sum_partials
from vllm.models.kimi_k3.amd.mono.common.stamps import stamp, stamp_begin, stamp_flush
from vllm.models.kimi_k3.amd.mono.common.sync import Mailbox, preg, shift, sreg
from vllm.models.kimi_k3.amd.mono.kda_pre import (
    _SOURCES,
    COLS,
    HIDDEN,
    LDS_PAD,
    NA,
    PROJ,
    ROWS,
    stage_gate,
    stage_mix,
    stage_slice,
    step_tag,
)

TP = 8
OROWS = HIDDEN  # o_proj rows (full hidden, a rank's K slice)
OTASKS = OROWS // ROWS  # 448
OK = PROJ  # K = 1536 a rank
OKCH = OK // 128  # 12 chunks
GEMV0 = NA  # the first gemv CTA
GEMV_N = BLOCKS - NA  # 200
TL_POINTS = 6  # start, gemv, pushed, reduced, slice/gate, mix

_STREAM = fx.Stream(None)


@dataclass(frozen=True)
class AttnPostBuild:
    tokens: int
    nblocks: int  # the MLP AttnRes block sources
    eps: float = 1e-5
    out_eps: float = 1e-5
    timeline: bool = False


ABI = KernelAbi(
    (
        "core",
        "w_o",
        "prefix",
        "blocks",
        "blk_sm",
        "blk_sr",
        "ares_nw",
        "ares_qk",
        "in_nw",
        "x_out",
        "scratch",
        "peers",
        "rank",
        "epoch",
        "layer",
        "tl",
    )
)


def ar_half_pairs(s):
    """One parity's AR region: [src rank][token][column pair]."""
    return TP * s * HIDDEN // 2


def peer_bytes(max_tokens):
    return 2 * ar_half_pairs(max_tokens) * 8


def scratch_layout(key: AttnPostBuild) -> dict:
    s, ns = key.tokens, key.nblocks + 1
    pairs = {
        "part": NA * s * ns * 2,
        "wgt": s * (ns + 1),
        "msq": NA * s,
        "xrdy": NA,
    }
    return pair_layout(pairs.items())


def scratch_bytes(key: AttnPostBuild) -> int:
    return max(o + n for o, n in scratch_layout(key).values())


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
def load_core(c):
    """The core rows (S x 1536 bf16) -> LDS, rows padded."""
    s, tid = c["S"], c["tid"]
    words = s * OK // 8
    row = OK + LDS_PAD
    for i in range_constexpr((words + THREADS - 1) // THREADS):
        e = fx.min(tid + THREADS * i, words - 1)
        v = fx.Vector(
            bo.buffer_load(rsrc(c["args"]["core"]), e * 4, vec_width=4, dtype=T.i32)
        )
        t = e // (OK // 8)
        k = e % (OK // 8) * 8
        if tid + THREADS * i < words:
            fx.ptr_store(v, c["xl"] + (t * row + k) // 2)
    gpu.barrier()


@traced
def stage_o(c, rg, ops):
    """Rows 16 rg ..: GEMV -> bf16 partials -> every rank's AR region, this
    rank's slot (thread: token, row pair; a put a peer)."""
    s, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    g = lane // ROWS
    t = fx.min(lane % ROWS, s - 1)
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
    if tid < s * (ROWS // 2):
        tt = tid // (ROWS // 2)
        rp = tid % (ROWS // 2)
        v0 = bf16_round(row_sum(red, 2 * rp, tt))
        v1 = bf16_round(row_sum(red, 2 * rp + 1, tt))
        at = (c["rank"] * s + tt) * HIDDEN + rg * ROWS + 2 * rp
        for p in range_constexpr(TP):
            c["put_bf"](c["ar_peer"][p], at, [v0, v1])
    gpu.barrier()


@traced
def stage_reduce(c, task):
    """Columns 128 task ..: the 8 ranks' partials, rank order, -> bf16 delta in
    LDS (thread: token, column pair)."""
    s, tid = c["S"], c["tid"]
    if tid < s * (COLS // 2):
        t = tid // (COLS // 2)
        cp = tid % (COLS // 2)
        col = task * COLS + 2 * cp
        lo, hi = sum_partials(
            c["poll"], c["ar_own"], lambda src: ((src * s + t) * HIDDEN + col) // 2, TP
        )
        fx.ptr_store(bf16_round(lo), c["delta_lds"] + (t * COLS + 2 * cp))
        fx.ptr_store(bf16_round(hi), c["delta_lds"] + (t * COLS + 2 * cp + 1))
    gpu.barrier()


_BUILDS: dict = {}


def build(key: AttnPostBuild):
    if key in _BUILDS:
        return _BUILDS[key]
    s, ns = key.tokens, key.nblocks + 1
    assert s * (ROWS // 2) <= THREADS and s * (COLS // 2) <= THREADS
    lay = scratch_layout(key)
    half = ar_half_pairs(s) * 8

    @fx.struct
    class Smem:
        red: fx.Array[fx.Float32, WAVES * 64 * 4, 16]
        xl: fx.Array[fx.Int32, s * (OK + LDS_PAD) // 2, 16]
        vals: fx.Array[fx.Float32, s * ns * COLS, 16]
        dl: fx.Array[fx.Float32, s * COLS, 16]
        tls: fx.Array[fx.Int64, TL_POINTS, 16]

    keyed = key_tuple(key, _SOURCES)
    name = f"k3_mono_attn_post_s{s}_nb{key.nblocks}_t{int(key.timeline)}"

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def attn_post(
        core: Int64,
        w_o: Int64,
        prefix: Int64,
        blocks: Int64,
        blk_sm: Int32,
        blk_sr: Int32,
        ares_nw: Int64,
        ares_qk: Int64,
        in_nw: Int64,
        x_out: Int64,
        scratch: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        layer: Int32,
        tl: Int64,
    ):
        _ = keyed
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lds = fx.SharedAllocator().allocate(Smem).peek()
        mb = Mailbox(step_tag(epoch, layer) - 1)
        par = uniform(ld_i32(epoch, 0)) % 2
        bases = peer_bases(peers, TP)
        c = {
            "S": s,
            "NS": ns,
            "NB": key.nblocks,
            "tid": tid,
            "bid": bid,
            "lane": tid % 64,
            "wave": tid // 64,
            "rank": rank,
            "delta": True,
            "debug": False,
            "eps": key.eps,
            "out_eps": key.out_eps,
            "put": mb.put,
            "put_bf": mb.put_bf,
            "poll": mb.poll,
            "red": lds.red.ptr,
            "xl": lds.xl.ptr,
            "vals": lds.vals.ptr,
            "delta_lds": lds.dl.ptr,
            "xbuf": x_out,
            "ar_peer": [shift(preg(b, 0, "ar"), fx.Int64(par) * half) for b in bases],
            "ar_own": shift(preg(load_own(peers, rank), 0, "ar"), fx.Int64(par) * half),
            "args": {
                "core": core,
                "w_o": w_o,
                "prefix": prefix,
                "blocks": blocks,
                "blk_sm": blk_sm,
                "blk_sr": blk_sr,
                "ares_nw": ares_nw,
                "ares_qk": ares_qk,
                "in_nw": in_nw,
                "x_dbg": 0,
            },
        }
        for region in ("part", "wgt", "msq", "xrdy"):
            c[region] = sreg(scratch, lay[region][0], region)
        on, tls = key.timeline, lds.tls.ptr
        stamp_begin(on, tls, tid, TL_POINTS)
        if bid >= GEMV0:
            gi = bid - GEMV0
            # every task's weights in flight first: nothing here polls
            ops = [o_loads(c, fx.min(gi + GEMV_N * j, OTASKS - 1)) for j in range(3)]
            load_core(c)
            stamp(on, tls, tid, 1)
            for j in range_constexpr(3):
                if gi + GEMV_N * j < OTASKS:
                    stage_o(c, gi + GEMV_N * j, ops[j])
            stamp(on, tls, tid, 2)
        if bid < NA:
            stage_reduce(c, bid)
            stamp(on, tls, tid, 3)
            stage_slice(c, bid)
            if bid < s:
                stage_gate(c, bid)
            stamp(on, tls, tid, 4)
            stage_mix(c, bid)
            stamp(on, tls, tid, 5)
        stamp_flush(on, tls, tl, tid, bid, TL_POINTS)

    @flyc.jit
    def launch(
        core: Int64,
        w_o: Int64,
        prefix: Int64,
        blocks: Int64,
        blk_sm: Int32,
        blk_sr: Int32,
        ares_nw: Int64,
        ares_qk: Int64,
        in_nw: Int64,
        x_out: Int64,
        scratch: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        layer: Int32,
        tl: Int64,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        attn_post(
            core,
            w_o,
            prefix,
            blocks,
            blk_sm,
            blk_sr,
            ares_nw,
            ares_qk,
            in_nw,
            x_out,
            scratch,
            peers,
            rank,
            epoch,
            layer,
            tl,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    ABI.check(attn_post, launch)
    _BUILDS[key] = launch
    return launch


def load_own(peers, rank):
    """This rank's own peer buffer base (its entry of the table)."""
    from vllm.models.kimi_k3.amd.mono.common.ops import load_ptr64

    return load_ptr64(peers, rank)


def attn_post(
    key: AttnPostBuild,
    *,
    core,
    w_o,
    prefix,
    blocks,
    ares_nw,
    ares_qk,
    in_nw,
    x_out,
    scratch,
    peers,
    rank,
    epoch,
    layer,
    tl=None,
):
    """Host launcher. ``peers``: the int64 table of every rank's peer buffer
    (at least ``peer_bytes(S)`` each); ``prefix`` updated in place."""
    assert w_o.shape == (OROWS, OK) and w_o.is_contiguous()
    assert prefix.is_contiguous() and core.is_contiguous() and x_out.is_contiguous()
    f = build(key)
    f(
        core.data_ptr(),
        w_o.data_ptr(),
        prefix.data_ptr(),
        blocks.data_ptr(),
        blocks.stride(0),
        blocks.stride(1),
        ares_nw.data_ptr(),
        ares_qk.data_ptr(),
        in_nw.data_ptr(),
        x_out.data_ptr(),
        scratch.data_ptr(),
        peers.data_ptr(),
        rank,
        epoch.data_ptr(),
        layer,
        0 if tl is None else tl.data_ptr(),
        stream=torch.cuda.current_stream(),
    )
