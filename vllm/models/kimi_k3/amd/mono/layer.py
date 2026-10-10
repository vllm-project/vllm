# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K2 ``layer_post``: one Kimi-K3 layer from the attention's core output (the
o_proj input) to the layer's MoE output, in one launch -- K2a's o_proj, its
all-reduce and the MLP AttnRes seam, then K2b's latent MoE (``moe``).

As ATOM's ``layer_post``, the two halves share the launch so that the MoE's
static weights stream while the AttnRes chain (partials -> reduce -> slice ->
gate -> mix) is still settling: the CTAs that will run the router, latent and
shared gate/up GEMVs issue their weights before they first wait.

    CTAs 0 .. 55     K2a's reduce + AttnRes (slice, gate, mix -> x rows + XRDY);
                     then route (0 .. S-1) and the shared SiTU (8 .. 8+S-1)
    CTAs 56 .. 159   o_proj (448 x 16 rows) -> partials to every rank; then
                     the shared gate/up (96 tasks, weights right after o_proj)
    CTAs 160 .. 243  router parts (28) and latent parts (56): weights first
    every CTA        ug; down (0 .. 223) or the shared down (224 .. 255);
                     lnorm (56 .. 83), up (0 .. 55), reduce (200 .. 255)
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import Int32, Int64

from vllm.models.kimi_k3.amd.mono.attention import back as k2a
from vllm.models.kimi_k3.amd.mono.attention.kda import (
    _SOURCES,
    COLS,
    HIDDEN,
    NA,
    stage_gate,
    stage_mix,
    stage_slice,
    step_tag,
)
from vllm.models.kimi_k3.amd.mono.common.debug import (
    region_ids,
    stamp,
    stamp_begin,
    stamp_flush,
)
from vllm.models.kimi_k3.amd.mono.common.ops import (
    ld_i32,
    lds_bytes,
    load_ptr64,
    traced,
    uniform,
)
from vllm.models.kimi_k3.amd.mono.common.plan import (
    BLOCKS,
    GFX942,
    LDS_BYTES,
    THREADS,
    WAVES,
    KernelAbi,
    key_tuple,
    pair_layout,
)
from vllm.models.kimi_k3.amd.mono.common.ranks import peer_bases
from vllm.models.kimi_k3.amd.mono.common.sync import Mailbox, preg, shift, sreg
from vllm.models.kimi_k3.amd.mono.stages import moe as k2b

O0, O1 = NA, BLOCKS  # o_proj CTAs: every CTA past AttnRes's
O_N = O1 - O0  # 200
OTASKS = k2a.OTASKS  # 448
O_ROUNDS = (OTASKS + O_N - 1) // O_N  # 5
E0 = 160  # router / latent CTAs (their weights in flight under their o_proj)
H0 = 8  # the shared SiTU's CTAs (after their AttnRes work)
SD0 = k2b.DOWN_TASKS  # 224: the shared down's CTAs
# timeline points: 0 start, 1 o_proj done / AR reduced, 2 early done / slice,
# 3 ug, 4 down/sdown, 5 lnorm, 6 up, 7 reduce, 8 gate, 9 mix (x published),
# 10 route, 11 route table, 12 latent quant, 13 route starts
TL_POINTS = 17  # + 14 (unused), 15 INTER loaded, 16 down done

assert E0 + k2b.R_TASKS + k2b.L_TASKS <= BLOCKS and k2b.G_TASKS <= E0 - O0

REGIONS = ("part", "wgt", "msq", "xrdy", "ar") + k2b.REGIONS

_STREAM = fx.Stream(None)


@dataclass(frozen=True)
class K2Build:
    tokens: int
    nblocks: int  # the MLP AttnRes block sources
    eps: float = 1e-5  # mlp_res_norm
    out_eps: float = 1e-5  # post_attention_layernorm
    ln_eps: float = 1e-5  # the latent RMSNorm
    reduce: bool = True  # the final reduce here (else the next layer's K1)
    reset: bool = False  # a block-write layer: the prefix restarts at the o_proj output
    diag: bool = False
    timeline: bool = False
    xp: int = 0  # timing experiments (7: route without its selection)


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
        "w_gate",
        "bias",
        "w_ld",
        "w_sgu",
        "w_sd",
        "w13",
        "w13s",
        "w2",
        "w2s",
        "ln_w",
        "w_up",
        "out",
        "scratch",
        "peers",
        "rank",
        "epoch",
        "layer",
        "diag",
        "tl",
    )
)


def peer_layout(s):
    """K2a's AR region, then K2b's: one parity's (name -> (byte offset, bytes))."""
    pairs = {"ar": k2a.ar_half_pairs(s), **k2b.peer_pairs(s)}
    return pair_layout(pairs.items())


def peer_bytes(s):
    lay = peer_layout(s)
    return 2 * max(o + n for o, n in lay.values())


def scratch_layout(key: K2Build) -> dict:
    s = key.tokens
    a = k2a.scratch_layout(k2a.AttnPostBuild(tokens=s, nblocks=key.nblocks))
    end = max(o + n for o, n in a.values())
    b = k2b.scratch_layout(k2b.MoeBuild(tokens=s))
    lay = dict(a)
    for name, (o, n) in b.items():
        lay[name] = ((end + 15) // 16 * 16 + o, n)
    end = max(o + n for o, n in lay.values())
    lay["xrow"] = ((end + 15) // 16 * 16, s * HIDDEN * 2)
    return lay


def scratch_bytes(key: K2Build) -> int:
    return max(o + n for o, n in scratch_layout(key).values())


@traced
def stage_oproj(c, gi):
    """o_proj tasks gi, gi + 200, gi + 400 (< 448): every task's weights in
    flight at once (one memory latency, not three), then the MFMAs."""
    ops = [k2a.o_loads(c, fx.min(gi + O_N * j, OTASKS - 1)) for j in range(O_ROUNDS)]
    k2a.load_core(c)
    for j in range_constexpr(O_ROUNDS):
        t = gi + O_N * j
        if t < OTASKS:
            k2a.stage_o(c, t, ops[j])


_BUILDS: dict = {}


def build(key: K2Build):
    if key in _BUILDS:
        return _BUILDS[key]
    s, ns = key.tokens, key.nblocks + 1
    assert 1 <= s <= 8
    lay = scratch_layout(key)
    play = peer_layout(s)
    half = peer_bytes(s) // 2
    # the early rows also hold o_proj's core rows
    RouteLds, EarlyLds, UgLds, DownLds = k2b.lds_structs(s, (k2a.OK,))

    @fx.struct
    class AttnLds:
        vals: fx.Array[fx.Float32, s * ns * COLS, 16]
        dl: fx.Array[fx.Float32, s * COLS, 16]

    @fx.union
    class StageLds:
        early: EarlyLds
        attn: AttnLds
        ug: UgLds
        down: DownLds

    @fx.struct
    class Smem:
        red: fx.Array[fx.Float32, WAVES * 64 * 4, 16]
        tls: fx.Array[fx.Int64, TL_POINTS, 16]

    used = lds_bytes(Smem) + lds_bytes(RouteLds) + lds_bytes(StageLds)
    assert used <= LDS_BYTES, f"LDS {used} > {LDS_BYTES}"
    keyed = key_tuple(key, _SOURCES)
    name = (
        f"k3_mono_k2_s{s}_nb{key.nblocks}_x{int(key.reset)}_r{int(key.reduce)}"
        f"_g{int(key.diag)}_t{int(key.timeline)}"
    )

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def k2(
        core: Int64,
        w_o: Int64,
        prefix: Int64,
        blocks: Int64,
        blk_sm: Int32,
        blk_sr: Int32,
        ares_nw: Int64,
        ares_qk: Int64,
        in_nw: Int64,
        w_gate: Int64,
        bias: Int64,
        w_ld: Int64,
        w_sgu: Int64,
        w_sd: Int64,
        w13: Int64,
        w13s: Int64,
        w2: Int64,
        w2s: Int64,
        ln_w: Int64,
        w_up: Int64,
        out: Int64,
        scratch: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        layer: Int32,
        diag: Int64,
        tl: Int64,
    ):
        _ = keyed
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        alloc = fx.SharedAllocator()
        lds = alloc.allocate(Smem).peek()
        rl = alloc.allocate(RouteLds).peek()
        st = alloc.allocate(StageLds)
        el, al, ul, dl = st.early.peek(), st.attn.peek(), st.ug.peek(), st.down.peek()
        if const_expr(key.diag):
            mb = Mailbox(step_tag(epoch, layer) - 1, diag, region_ids(REGIONS))
        else:
            mb = Mailbox(step_tag(epoch, layer) - 1)
        par = uniform(ld_i32(epoch, 0)) % 2
        bases = peer_bases(peers, TP_)
        own = load_ptr64(peers, rank)
        peer = {}
        for region in ("rlog", "lat", "latr", "final", "ar"):
            off = play[region][0]
            peer[region] = [
                shift(preg(b, off, region), fx.Int64(par) * half) for b in bases
            ]
            peer[region + "_own"] = shift(preg(own, off, region), fx.Int64(par) * half)
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
            "reset": key.reset,
            "gate_local": False,
            "debug": False,
            "eps": key.eps,
            "out_eps": key.out_eps,
            "ln_eps": key.ln_eps,
            "xp": key.xp,
            "lazy_inter": True,
            "put": mb.put,
            "put_bf": mb.put_bf,
            "put_words": mb.put_words,
            "poll": mb.poll,
            "red": lds.red.ptr,
            "peer": peer,
            "ar_peer": peer["ar"],
            "ar_own": peer["ar_own"],
            "xl": el.xl.ptr,
            "vals": al.vals.ptr,
            "delta_lds": al.dl.ptr,
            **k2b.lds_ptrs(ul, dl),
            "scan": rl.scan.ptr,
            "rt": {
                "route": rl.route.ptr,
                "flag": rl.flag.ptr,
                "expert": rl.expert.ptr,
                "kof": rl.kof.ptr,
            },
            "xbuf": scratch + fx.Int64(lay["xrow"][0]),
            "xflags": NA,
            "tl_on": key.timeline,
            "tls": lds.tls.ptr,
            "ln": scratch + fx.Int64(lay["ln"][0]),
            "shp": scratch + fx.Int64(lay["shp"][0]),
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
                "w_gate": w_gate,
                "bias": bias,
                "w_ld": w_ld,
                "w_sgu": w_sgu,
                "w_sd": w_sd,
                "w13": w13,
                "w13s": w13s,
                "w2": w2,
                "w2s": w2s,
                "ln_w": ln_w,
                "w_up": w_up,
                "out": out,
            },
        }
        for region in ["part", "wgt", "msq", "xrdy", *k2b.scratch_regions()]:
            c[region] = sreg(scratch, lay[region][0], region)
        if const_expr(GFX942):
            c["inter16"] = scratch + fx.Int64(lay["inter16"][0])
        on, tls = key.timeline, lds.tls.ptr
        stamp_begin(on, tls, tid, TL_POINTS)
        # ---- o_proj on every CTA past AttnRes's
        eb = bid - E0
        lb = eb - k2b.R_TASKS
        is_r = (bid >= E0) & (eb < k2b.R_TASKS)
        is_l = (bid >= E0 + k2b.R_TASKS) & (lb < k2b.L_TASKS)
        # o_proj first (every AttnRes CTA waits on the slowest partial; loads
        # retire in order), the early weights after: they land before x does
        if is_r:
            stage_oproj(c, bid - O0)
            stamp(on, tls, tid, 1)
            rops = k2b.early_loads(c, 0, eb // k2b.R_PARTS, eb % k2b.R_PARTS)
            k2b.early_compute(c, 0, eb // k2b.R_PARTS, eb % k2b.R_PARTS, rops)
        if is_l:
            stage_oproj(c, bid - O0)
            stamp(on, tls, tid, 1)
            lops = k2b.early_loads(c, 1, lb // k2b.L_PARTS, lb % k2b.L_PARTS)
            k2b.early_compute(c, 1, lb // k2b.L_PARTS, lb % k2b.L_PARTS, lops)
        if (bid >= O0) & ~is_r & ~is_l:
            stage_oproj(c, bid - O0)
            stamp(on, tls, tid, 1)
            gi = bid - O0
            if gi < k2b.G_TASKS:
                k2b.early_compute(c, 2, gi, 0, k2b.early_loads(c, 2, gi, 0))
        # ---- AttnRes: the reduced o_proj -> x; then route / SiTU
        if bid < NA:
            k2a.stage_reduce(c, bid)
            stamp(on, tls, tid, 1)
            stage_slice(c, bid)
            stamp(on, tls, tid, 2)
            if bid < s:
                stage_gate(c, bid)
            stamp(on, tls, tid, 8)
            stage_mix(c, bid)
            stamp(on, tls, tid, 9)
            stamp(on, tls, tid, 13)
            if bid < s:
                k2b.stage_route(c, bid)
            stamp(on, tls, tid, 10)
            if (bid >= H0) & (bid < H0 + s):
                k2b.stage_h(c, bid - H0)
        if bid >= NA:
            stamp(on, tls, tid, 2)
        # ---- the routed experts
        # the latent (ready ~10 us before the route) quantized first (gfx942:
        # its first K phase into LDS)
        if const_expr(GFX942):
            k2b.load_lat(c, 0)
        else:
            k2b.load_latq(c)
        stamp(on, tls, tid, 12)
        nu = k2b.load_route(c)
        stamp(on, tls, tid, 11)
        if const_expr(GFX942):
            k2b.run_ug_i4(c, bid, nu)
        else:
            total = nu * k2b.UG_GROUPS
            last = total - 1
            # reversed placement: the round's leftover tasks go to the high CTAs,
            # whose early work ends first (the AttnRes / route CTAs start ug last)
            task = BLOCKS - 1 - bid
            ops = k2b.ug_loads(c, fx.min(task, last))
            while task < total:
                nxt = task + BLOCKS
                ops_n = k2b.ug_loads(c, fx.min(nxt, last))
                k2b.stage_ug(c, task, ops)
                task = nxt
                ops = ops_n
        stamp(on, tls, tid, 3)
        if const_expr(GFX942):
            k2b.run_down_i4(c, bid, nu)
            stamp(on, tls, tid, 16)
        else:
            if bid < k2b.DOWN_TASKS:
                dops = k2b.down_loads(c, bid, nu, fx.min(c["wave"], nu - 1))
                k2b.load_inter(c)
                stamp(on, tls, tid, 15)
                k2b.stage_down(c, bid, nu, dops)
                stamp(on, tls, tid, 16)
        k2b.run_sdown_rest(c, bid, w_sd)
        k2b.run_sdown(c, bid, w_sd)
        stamp(on, tls, tid, 4)
        if (bid >= k2b.LN0) & (bid < k2b.LN0 + k2b.LN_TASKS):
            k2b.stage_lnorm(c, bid - k2b.LN0)
        stamp(on, tls, tid, 5)
        if bid < k2b.UP_TASKS:
            k2b.stage_up(c, bid)
        stamp(on, tls, tid, 6)
        if const_expr(key.reduce):  # noqa: SIM102 (traced: `and` does not combine traced values)
            if (bid >= k2b.RED0) & (bid < k2b.RED0 + k2b.RED_TASKS):
                k2b.stage_reduce(c, bid - k2b.RED0)
        stamp(on, tls, tid, 7)
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
        w_gate: Int64,
        bias: Int64,
        w_ld: Int64,
        w_sgu: Int64,
        w_sd: Int64,
        w13: Int64,
        w13s: Int64,
        w2: Int64,
        w2s: Int64,
        ln_w: Int64,
        w_up: Int64,
        out: Int64,
        scratch: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        layer: Int32,
        diag: Int64,
        tl: Int64,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        k2(
            core,
            w_o,
            prefix,
            blocks,
            blk_sm,
            blk_sr,
            ares_nw,
            ares_qk,
            in_nw,
            w_gate,
            bias,
            w_ld,
            w_sgu,
            w_sd,
            w13,
            w13s,
            w2,
            w2s,
            ln_w,
            w_up,
            out,
            scratch,
            peers,
            rank,
            epoch,
            layer,
            diag,
            tl,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    ABI.check(k2, launch)
    _BUILDS[key] = launch
    return launch


TP_ = k2b.TP


def ptr(t):
    return 0 if t is None else t.data_ptr()


def k2_launch(
    key: K2Build,
    *,
    core,
    w_o,
    prefix,
    blocks,
    ares_nw,
    ares_qk,
    in_nw,
    w_gate,
    bias,
    w_ld,
    w_sgu,
    w_sd,
    w13,
    w13s,
    w2,
    w2s,
    ln_w,
    w_up,
    out,
    scratch,
    peers,
    rank,
    epoch,
    layer,
    diag=None,
    tl=None,
):
    """Host launcher: ``core`` [S, 1536] (o_proj's input), ``prefix`` updated in
    place, ``out`` the MoE output [S, 7168]; MoE weights as ``moe``."""
    assert (
        w_o.shape == (HIDDEN, k2a.OK)
        and core.is_contiguous()
        and prefix.is_contiguous()
    )
    assert w_up.shape == (k2b.UP_N, k2b.LAT) and w_up.is_contiguous()
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
        w_gate.data_ptr(),
        bias.data_ptr(),
        w_ld.data_ptr(),
        w_sgu.data_ptr(),
        w_sd.data_ptr(),
        w13.data_ptr(),
        w13s.data_ptr(),
        w2.data_ptr(),
        w2s.data_ptr(),
        ln_w.data_ptr(),
        w_up.data_ptr(),
        out.data_ptr(),
        scratch.data_ptr(),
        peers.data_ptr(),
        rank,
        epoch.data_ptr(),
        layer,
        ptr(diag),
        ptr(tl),
        stream=torch.cuda.current_stream(),
    )
