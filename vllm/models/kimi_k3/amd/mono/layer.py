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
from flydsl.expr import const_expr, gpu, range_constexpr
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
from vllm.models.kimi_k3.amd.mono.common.ops import ld_i32, load_ptr64, traced, uniform
from vllm.models.kimi_k3.amd.mono.common.plan import (
    BLOCKS,
    THREADS,
    WAVES,
    KernelAbi,
    key_tuple,
    pair_layout,
)
from vllm.models.kimi_k3.amd.mono.common.ranks import peer_bases
from vllm.models.kimi_k3.amd.mono.common.sync import Mailbox, preg, publish, shift, sreg
from vllm.models.kimi_k3.amd.mono.common.tiles import MAX_TOKENS
from vllm.models.kimi_k3.amd.mono.stages import gemv as gemv
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
TL_POINTS = 21  # 15 first down input, 16 down done, 18 down entered, 20 logits ready

assert E0 + k2b.R_TASKS + k2b.L_TASKS <= BLOCKS and k2b.G_TASKS <= E0 - O0

REGIONS = (
    ("part", "wgt", "msq", "xrdy", "ar")
    + k2b.REGIONS
    + (
        "shared_latq",
        "shared_latsc",
        "local_lat",
        "qlat",
        "qlatsc",
        "xpanels",
    )
)

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
    shared_down_workers: int = 0
    parallel_shared_activation: bool = False
    late_shared_wait: bool = False
    latent_quant_workers: int = 0
    producer_latent_quant: bool = False
    down_workers: int = 0
    full_shared_pipeline: bool = False
    down_batch: int = 5
    inter_vec: int = 1
    latq_vec: int = 1
    trim_oproj: bool = False
    stream_oproj: bool = False
    balanced_oproj: bool = False
    wide: bool = False
    compact_lds: bool = False
    cache_wide_inter: bool = False
    separate_wide_lds: bool = False
    stream_attn: bool = False
    stream_latent: bool = False
    sparse_routes: bool = False
    stream_x: bool = False
    front_panels: bool = False


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
    pairs = {
        "ar": k2a.ar_half_pairs(s),
        **k2b.peer_pairs(s),
        "qlat": s * k2b.LAT // 8,
        "qlatsc": s * k2b.LAT // 32,
    }
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
    if key.latent_quant_workers:
        end = lay["xrow"][0] + lay["xrow"][1]
        for name, count in (
            ("shared_latq", s * k2b.LAT // 8),
            ("shared_latsc", s * k2b.LAT // 32),
        ):
            offset = (end + 15) // 16 * 16
            lay[name] = (offset, count * 8)
            end = offset + count * 8
    if key.producer_latent_quant:
        end = max(o + n for o, n in lay.values())
        lay["local_lat"] = ((end + 15) // 16 * 16, s * (k2b.LAT // k2b.TP) * 4)
    if key.stream_x:
        end = max(o + n for o, n in lay.values())
        lay["xpanels"] = ((end + 15) // 16 * 16, ((s + 15) // 16) * NA * 8)
    return lay


def scratch_bytes(key: K2Build) -> int:
    return max(o + n for o, n in scratch_layout(key).values())


@traced
def stage_oproj(c, gi):
    workers = BLOCKS if c.get("balanced_oproj", False) else O_N
    rounds = (OTASKS + workers - 1) // workers
    if const_expr(c.get("trim_oproj", False)):
        if gi < OTASKS - workers * (rounds - 1):
            stage_oproj_tasks(c, gi, rounds, workers)
        else:
            stage_oproj_tasks(c, gi, rounds - 1, workers)
    else:
        stage_oproj_tasks(c, gi, rounds, workers)


@traced
def stage_oproj_tasks(c, gi, rounds, workers):
    """Prefetch this CTA's assigned output tiles together, then run their MFMAs."""
    ops = [k2a.o_loads(c, fx.min(gi + workers * j, OTASKS - 1)) for j in range(rounds)]
    if const_expr(c.get("core_ready") is not None):
        if c["tid"] < c["core_writers"]:
            c["poll"]([(c["core_ready"], c["tid"], 1)])
        gpu.barrier()
        stamp(c["tl_on"], c["tls"], c["tid"], 17)
    if const_expr(c.get("stream_oproj", False)):
        k2a.stage_o_stream(c, gi, workers, ops)
    else:
        for first in range_constexpr(0, c["S"], 16):
            panel = dict(c, panel_first=first, panel_rows=min(16, c["S"] - first))
            k2a.load_core(panel)
            if const_expr(c.get("core_mailbox") is not None):
                stamp(c["tl_on"], c["tls"], c["tid"], 17)
            for j in range_constexpr(rounds):
                t = gi + workers * j
                if t < OTASKS:
                    k2a.stage_o(panel, t, ops[j])


@traced
def stage_front_panels(c):
    """Stream all-reduce and AttnRes through disjoint token-panel mailboxes."""
    s, ns, bid, tid = c["S"], c["NS"], c["bid"], c["tid"]
    a = c["args"]
    on, tls = c["tl_on"], c["tls"]
    for first in range_constexpr(0, s, 16):
        count = min(16, s - first)
        panel = dict(
            c,
            S=count,
            ar_tokens=s,
            ar_first=first,
            stream_attn=False,
            stream_x=False,
            xbuf=c["xbuf"] + fx.Int64(first * HIDDEN * 2),
            part=shift(c["part"], first * NA * ns * 2 * 8),
            wgt=shift(c["wgt"], first * (ns + 1) * 8),
            msq=shift(c["msq"], first * NA * 8),
            xrdy=shift(c["xpanels"], first // 16 * NA * 8),
            args=dict(
                a,
                prefix=a["prefix"] + fx.Int64(first * HIDDEN * 2),
                blocks=a["blocks"] + fx.Int64(first) * fx.Int64(a["blk_sm"]) * 2,
            ),
        )
        k2a.stage_reduce(panel, bid)
        stamp(on, tls, tid, 1)
        stage_slice(panel, bid)
        stamp(on, tls, tid, 2)
        if bid < count:
            stage_gate(panel, bid)
        stamp(on, tls, tid, 8)
        stage_mix(panel, bid)
        stamp(on, tls, tid, 9)
    publish(c["put"], c["xrdy"], bid, fx.Int32(1), tid == 0)
    gpu.barrier()


@traced
def run_stages(c, key):
    bid, tid, rank, s = c["bid"], c["tid"], c["rank"], c["S"]
    w_sd = c["args"]["w_sd"]
    shared_start = BLOCKS - key.shared_down_workers
    activation_start = max(H0, s) if key.parallel_shared_activation else H0
    on, tls = key.timeline, c["tls"]
    stamp_begin(on, tls, tid, c.get("tl_points", TL_POINTS))
    # ---- o_proj on every CTA past AttnRes's
    eb = bid - E0
    lb = eb - k2b.R_TASKS
    is_r = (bid >= E0) & (eb < k2b.R_TASKS)
    is_l = (bid >= E0 + k2b.R_TASKS) & (lb < k2b.L_TASKS)
    # o_proj first (every AttnRes CTA waits on the slowest partial; loads
    # retire in order), the early weights after: they land before x does
    if const_expr(key.balanced_oproj):
        stage_oproj(c, bid)
        stamp(on, tls, tid, 1)
    if is_r:
        if const_expr(not key.balanced_oproj):
            stage_oproj(c, bid - O0)
        stamp(on, tls, tid, 1)
        rops = k2b.early_loads(c, 0, eb // k2b.R_PARTS, eb % k2b.R_PARTS)
        k2b.early_compute(c, 0, eb // k2b.R_PARTS, eb % k2b.R_PARTS, rops)
    if is_l:
        if const_expr(not key.balanced_oproj):
            stage_oproj(c, bid - O0)
        stamp(on, tls, tid, 1)
        lops = k2b.early_loads(c, 1, lb // k2b.L_PARTS, lb % k2b.L_PARTS)
        k2b.early_compute(c, 1, lb // k2b.L_PARTS, lb % k2b.L_PARTS, lops)
    if (bid >= O0) & ~is_r & ~is_l:
        if const_expr(not key.balanced_oproj):
            stage_oproj(c, bid - O0)
        stamp(on, tls, tid, 1)
        gi = bid - O0
        if gi < k2b.G_TASKS:
            k2b.early_compute(c, 2, gi, 0, k2b.early_loads(c, 2, gi, 0))
    # ---- AttnRes: the reduced o_proj -> x; then route / SiTU
    if bid < NA:
        if const_expr(key.front_panels):
            stage_front_panels(c)
        else:
            k2a.stage_reduce(c, bid)
            stamp(on, tls, tid, 1)
            stage_slice(c, bid)
            stamp(on, tls, tid, 2)
            for token in range(bid, s, NA):
                stage_gate(c, token)
            stamp(on, tls, tid, 8)
            stage_mix(c, bid)
            stamp(on, tls, tid, 9)
        stamp(on, tls, tid, 13)
        for token in range(bid, s, NA):
            k2b.stage_route(c, token)
        stamp(on, tls, tid, 10)
        if const_expr(s > NA - H0):
            for token in range(bid, s, NA):
                k2b.stage_h(c, token)
        else:
            if (bid >= activation_start) & (bid < activation_start + s):
                k2b.stage_h(c, bid - activation_start)
    if bid >= NA:
        stamp(on, tls, tid, 2)
    if const_expr(key.shared_down_workers > 0):  # noqa: SIM102
        if bid >= shared_start:
            k2b.load_h(c)
            for task in range(
                bid - shared_start,
                k2b.SD_TASKS if key.full_shared_pipeline else k2b.UP_TASKS,
                key.shared_down_workers,
            ):
                row = (task + rank * k2b.UP_TASKS) % k2b.SD_TASKS
                k2b.stage_sdown(
                    c,
                    row,
                    gemv.loads(c["lane"], c["wave"], w_sd, row * k2b.ROWS, k2b.SI),
                )
            stamp(on, tls, tid, 14)
    # ---- the routed experts
    # the latent (ready ~10 us before the route) quantized first
    if const_expr(key.producer_latent_quant and not key.stream_latent):
        k2b.load_shared_latq(c)
    elif const_expr(key.latent_quant_workers > 0):
        quant_start = E0 + k2b.R_TASKS + k2b.L_TASKS
        if (bid >= quant_start) & (bid < quant_start + key.latent_quant_workers):
            k2b.publish_latq(c, bid - quant_start, key.latent_quant_workers)
        k2b.load_shared_latq(c)
    elif const_expr(not key.stream_latent):
        k2b.load_latq(c)
    stamp(on, tls, tid, 12)
    nu = k2b.load_route(c)
    stamp(on, tls, tid, 11)
    k2b.run_routed(c, nu, BLOCKS, key.down_workers)
    if const_expr(not key.full_shared_pipeline):
        k2b.run_sdown_rest(c, bid, w_sd)
    if const_expr(key.shared_down_workers == 0):
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


_BUILDS: dict = {}


def build(key: K2Build):
    if key in _BUILDS:
        return _BUILDS[key]
    s, ns = key.tokens, key.nblocks + 1
    assert 1 <= s <= (MAX_TOKENS if key.wide else k2b.MOE_TILE_TOKENS)
    assert key.shared_down_workers in (0, 16, 32, 64)
    assert key.latent_quant_workers in (0, 1, 2, 4, 8)
    assert not (key.latent_quant_workers and key.producer_latent_quant)
    assert key.down_workers in (0, 16, 32, 64, 96, 112)
    assert not key.full_shared_pipeline or key.shared_down_workers > 0
    assert 1 <= key.down_batch <= 10
    assert key.inter_vec in (1, 2)
    assert key.latq_vec in (1, 2)
    assert not key.wide or not key.stream_oproj
    assert not key.cache_wide_inter or key.wide and key.compact_lds
    assert not key.stream_attn or key.wide and s > 16
    assert not key.stream_latent or key.wide and key.producer_latent_quant
    assert not key.sparse_routes or key.wide and key.compact_lds
    assert not key.stream_x or key.wide and s > 16
    assert not key.front_panels or key.stream_x
    assert not key.stream_x or s * (ns + 1) + 128 <= WAVES * 64 * 4
    stream_inter = key.down_workers > 0 or key.wide and not key.cache_wide_inter
    route_type = fx.Int8 if key.compact_lds else fx.Int32
    scale_type = fx.Uint8 if key.compact_lds else fx.Int32
    contrib_type = fx.BFloat16 if key.compact_lds else fx.Float32
    max_u = min(s * k2b.TOPK, k2b.E) if key.wide else s * k2b.TOPK
    route_rows = s if key.wide else 8 if s <= 8 else 16
    lay = scratch_layout(key)
    play = peer_layout(s)
    half = peer_bytes(s) // 2

    @fx.struct
    class RouteLds:
        route: fx.Array[fx.Int32, s * 2 * k2b.TOPK, 16]
        flag: fx.Array[fx.Int32, k2b.E, 16]
        expert: fx.Array[fx.Int32, max_u, 16]
        kof: fx.Array[route_type, 4 if key.sparse_routes else max_u * route_rows, 16]
        begins: fx.Array[fx.Int32, max_u if key.sparse_routes else 4, 16]
        counts: fx.Array[fx.Int32, max_u if key.sparse_routes else 4, 16]
        cursor: fx.Array[fx.Int32, max_u if key.sparse_routes else 4, 16]
        picks: fx.Array[fx.Uint16, s * k2b.TOPK if key.sparse_routes else 4, 16]
        scan: fx.Array[fx.Int32, WAVES, 16]

    @fx.struct
    class EarlyLds:
        xl: fx.Array[
            fx.Int32,
            min(s, 16) * ((HIDDEN if s <= 8 else 4096) + gemv.LDS_PAD) // 2,
            16,
        ]

    @fx.struct
    class AttnLds:
        vals: fx.Array[
            fx.Float32,
            (
                min(s, 16) * ns
                if key.front_panels
                else s * (1 if key.stream_attn else ns)
            )
            * COLS,
            16,
        ]
        dl: fx.Array[fx.Float32, (min(s, 16) if key.front_panels else s) * COLS, 16]

    @fx.struct
    class UgLds:
        latq: fx.Array[fx.Int32, 4 if key.stream_latent else s * k2b.LAT // 8, 16]
        latsc: fx.Array[scale_type, 4 if key.stream_latent else s * k2b.LAT // 32, 16]
        av: fx.Array[fx.Float32, THREADS, 16]

    @fx.struct
    class DownLds:
        interq: fx.Array[fx.Int32, 4 if stream_inter else s * k2b.TOPK * 48, 16]
        intsc: fx.Array[
            scale_type,
            4 if stream_inter else s * k2b.TOPK * k2b.UG_GROUPS,
            16,
        ]
        contrib: fx.Array[contrib_type, s * k2b.TOPK * k2b.ROWS, 16]

    @fx.union
    class RoutedStages:
        ug: UgLds
        down: DownLds

    @fx.struct
    class RoutedLds:
        route: RouteLds
        stages: RoutedStages

    @fx.union
    class StageLds:
        early: EarlyLds
        attn: AttnLds
        routed: RoutedLds

    @fx.union
    class LegacyStages:
        early: EarlyLds
        attn: AttnLds
        ug: UgLds
        down: DownLds

    @fx.struct
    class Smem:
        red: fx.Array[fx.Float32, WAVES * 64 * 4, 16]
        tls: fx.Array[fx.Int64, TL_POINTS, 16]

    keyed = key_tuple(key, _SOURCES)
    name = (
        f"k3_mono_k2_s{s}_nb{key.nblocks}_x{int(key.reset)}_r{int(key.reduce)}"
        f"_g{int(key.diag)}_t{int(key.timeline)}"
        f"_sd{key.shared_down_workers}_ha{int(key.parallel_shared_activation)}"
        f"_lw{int(key.late_shared_wait)}"
        f"_lq{key.latent_quant_workers}"
        f"_lqp{int(key.producer_latent_quant)}"
        f"_dw{key.down_workers}_db{key.down_batch}_iv{key.inter_vec}"
        f"_lv{key.latq_vec}"
        f"_sx{int(key.stream_x)}"
        f"_fp{int(key.front_panels)}"
        f"_optrim{int(key.trim_oproj)}_opstream{int(key.stream_oproj)}"
        f"_opbalanced{int(key.balanced_oproj)}_wide{int(key.wide)}"
        f"_compact{int(key.compact_lds)}_cached{int(key.cache_wide_inter)}"
        f"_separate{int(key.separate_wide_lds)}"
        f"_attnstream{int(key.stream_attn)}"
        f"_latstream{int(key.stream_latent)}"
        f"_sparse{int(key.sparse_routes)}"
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
        if const_expr(key.wide and not key.separate_wide_lds):
            st = alloc.allocate(StageLds)
            rl = st.routed.route.peek()
            el, al = st.early.peek(), st.attn.peek()
            ul, dl = st.routed.stages.ug.peek(), st.routed.stages.down.peek()
        else:
            rl = alloc.allocate(RouteLds).peek()
            st = alloc.allocate(LegacyStages)
            el, al = st.early.peek(), st.attn.peek()
            ul, dl = st.ug.peek(), st.down.peek()
        if const_expr(key.diag):
            mb = Mailbox(step_tag(epoch, layer) - 1, diag, region_ids(REGIONS))
        else:
            mb = Mailbox(step_tag(epoch, layer) - 1)
        par = uniform(ld_i32(epoch, 0)) % 2
        bases = peer_bases(peers, TP_)
        own = load_ptr64(peers, rank)
        peer = {}
        for region in ("rlog", "lat", "latr", "final", "ar", "qlat", "qlatsc"):
            off = play[region][0]
            peer[region] = [
                shift(preg(b, off, region), fx.Int64(par) * half) for b in bases
            ]
            peer[region + "_own"] = shift(preg(own, off, region), fx.Int64(par) * half)
        c = {
            "S": s,
            "max_u": max_u,
            "route_rows": route_rows,
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
            "stream_inter": stream_inter,
            "compact_lds": key.compact_lds,
            "stream_attn": key.stream_attn,
            "stream_latent": key.stream_latent,
            "sparse_routes": key.sparse_routes,
            "stream_x": key.stream_x,
            "down_batch": key.down_batch,
            "inter_vec": key.inter_vec,
            "latq_vec": key.latq_vec,
            "trim_oproj": key.trim_oproj,
            "stream_oproj": key.stream_oproj,
            "balanced_oproj": key.balanced_oproj,
            "route_timeline": key.timeline,
            "late_shared_wait": key.late_shared_wait,
            "producer_quant": key.producer_latent_quant,
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
            "latq": ul.latq.ptr,
            "latsc": ul.latsc.ptr,
            "av": ul.av.ptr,
            "interq": dl.interq.ptr,
            "intsc": dl.intsc.ptr,
            "contrib": dl.contrib.ptr,
            "scan": rl.scan.ptr,
            "rt": {
                "route": rl.route.ptr,
                "flag": rl.flag.ptr,
                "expert": rl.expert.ptr,
                "kof": rl.kof.ptr,
                "begins": rl.begins.ptr,
                "counts": rl.counts.ptr,
                "cursor": rl.cursor.ptr,
                "picks": rl.picks.ptr,
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
        for region in (
            "part",
            "wgt",
            "msq",
            "xrdy",
            "sgu",
            "h",
            "rpart",
            "lpart",
            "route",
            "inter",
            "isc",
            "lmsq",
            "lnrdy",
            "shrdy",
        ):
            c[region] = sreg(scratch, lay[region][0], region)
        if const_expr(key.latent_quant_workers > 0):
            for region in ("shared_latq", "shared_latsc"):
                c[region] = sreg(scratch, lay[region][0], region)
        if const_expr(key.producer_latent_quant):
            c["local_lat"] = sreg(scratch, lay["local_lat"][0], "local_lat")
            c["shared_latq"] = peer["qlat_own"]
            c["shared_latsc"] = peer["qlatsc_own"]
        if const_expr(key.stream_x):
            c["xpanels"] = sreg(scratch, lay["xpanels"][0], "xpanels")
        run_stages(c, key)
        on, tls = key.timeline, lds.tls.ptr
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
