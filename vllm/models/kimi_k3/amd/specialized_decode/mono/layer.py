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

from vllm.models.kimi_k3.amd.specialized_decode.mono.attention import back as k2a
from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.kda import (
    _SOURCES,
    COLS,
    HIDDEN,
    NA,
    stage_gate,
    stage_mix,
    stage_slice,
    step_tag,
)
from vllm.models.kimi_k3.amd.specialized_decode.mono.common.debug import (
    region_ids,
)
from vllm.models.kimi_k3.amd.specialized_decode.mono.common.ops import (
    ld_i32,
    load_ptr64,
    traced,
    uniform,
)
from vllm.models.kimi_k3.amd.specialized_decode.mono.common.plan import (
    BLOCKS,
    THREADS,
    WAVES,
    KernelAbi,
    key_tuple,
    pair_layout,
)
from vllm.models.kimi_k3.amd.specialized_decode.mono.common.ranks import peer_bases
from vllm.models.kimi_k3.amd.specialized_decode.mono.common.sync import (
    Mailbox,
    preg,
    publish,
    shift,
    sreg,
    stall,
)
from vllm.models.kimi_k3.amd.specialized_decode.mono.common.tiles import MAX_TOKENS
from vllm.models.kimi_k3.amd.specialized_decode.mono.stages import gemv as gemv
from vllm.models.kimi_k3.amd.specialized_decode.mono.stages import moe as k2b

O0, O1 = NA, BLOCKS  # o_proj CTAs: every CTA past AttnRes's
O_N = O1 - O0  # 200
OTASKS = k2a.OTASKS  # 448
E0 = 160  # router / latent CTAs (their weights in flight under their o_proj)
H0 = 8  # the shared SiTU's CTAs (after their AttnRes work)
SD0 = k2b.DOWN_TASKS  # 224: the shared down's CTAs

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
    # the fault hook (sync.stall): rank fault_rank stalls fault_ticks once, in
    # slot fault_layer's launch of the step at epoch fault_epoch
    fault_rank: int = -1
    fault_epoch: int = 0
    fault_layer: int = 0
    fault_ticks: int = 0
    # the latent's producers quantize it and put it to every rank
    producer_latent_quant: bool = False
    down_batch: int = k2b.DOWN_BATCH  # a wave's experts whose w2 is in flight
    inter_vec: int = 1  # INTER / ISC words a lane polls at once (``load_inter``)
    # o_proj: the last round only on the CTAs it has tasks for (``trim_oproj``),
    # the core read once for every task of a CTA (``stream_oproj``), the tasks
    # over every CTA instead of those past AttnRes's (``balanced_oproj``)
    trim_oproj: bool = False
    stream_oproj: bool = False
    balanced_oproj: bool = False
    # past MOE_TILE_TOKENS rows: the route table shares the routed stages' LDS
    # (``wide``) in byte / bf16 types (``compact_lds``), each slot's picks
    # listed (``sparse_routes``)
    wide: bool = False
    compact_lds: bool = False
    sparse_routes: bool = False
    # AttnRes keeps only the prefix in LDS and re-reads the blocks
    stream_attn: bool = False
    # x published a 16-row panel at a time (``stream_x``); the all-reduce and
    # AttnRes run panel by panel (``front_panels``)
    stream_x: bool = False
    front_panels: bool = False


def tuned(tokens: int, nblocks: int) -> dict:
    """The measured schedule of a launch's rows and AttnRes block sources."""
    wide = tokens > k2b.MOE_TILE_TOKENS
    multi = tokens >= 2
    panels = tokens >= 28
    return {
        "producer_latent_quant": multi,
        "down_batch": 2 if multi else k2b.DOWN_BATCH,
        "inter_vec": 2 if multi else 1,
        "trim_oproj": tokens == 6,
        "stream_oproj": tokens == 6,
        "balanced_oproj": multi and tokens < 30,
        "wide": wide,
        "compact_lds": wide,
        "sparse_routes": wide,
        "stream_attn": wide and nblocks >= 8,
        "stream_x": panels,
        "front_panels": panels,
    }


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
        "health",
        "alarm",
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
    a = k2a.scratch_layout(s, key.nblocks)
    end = max(o + n for o, n in a.values())
    b = k2b.scratch_layout(s)
    lay = dict(a)
    for name, (o, n) in b.items():
        lay[name] = ((end + 15) // 16 * 16 + o, n)
    end = max(o + n for o, n in lay.values())
    lay["xrow"] = ((end + 15) // 16 * 16, s * HIDDEN * 2)
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
    if const_expr(c.get("stream_oproj", False)):
        k2a.stage_o_stream(c, gi, workers, ops)
    else:
        for first in range_constexpr(0, c["S"], 16):
            panel = dict(c, panel_first=first, panel_rows=min(16, c["S"] - first))
            k2a.load_core(panel)
            for j in range_constexpr(rounds):
                t = gi + workers * j
                if t < OTASKS:
                    k2a.stage_o(panel, t, ops[j])


@traced
def stage_front_panels(c):
    """Stream all-reduce and AttnRes through disjoint token-panel mailboxes."""
    s, ns, bid, tid = c["S"], c["NS"], c["bid"], c["tid"]
    a = c["args"]
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
        stage_slice(panel, bid)
        if bid < count:
            stage_gate(panel, bid)
        stage_mix(panel, bid)
    publish(c["put"], c["xrdy"], bid, fx.Int32(1), tid == 0)
    gpu.barrier()


@traced
def run_stages(c, key):
    bid, s = c["bid"], c["S"]
    w_sd = c["args"]["w_sd"]
    # ---- o_proj on every CTA past AttnRes's
    eb = bid - E0
    lb = eb - k2b.R_TASKS
    is_r = (bid >= E0) & (eb < k2b.R_TASKS)
    is_l = (bid >= E0 + k2b.R_TASKS) & (lb < k2b.L_TASKS)
    # o_proj first (every AttnRes CTA waits on the slowest partial; loads
    # retire in order), the early weights after: they land before x does
    if const_expr(key.balanced_oproj):
        stage_oproj(c, bid)
    if is_r:
        if const_expr(not key.balanced_oproj):
            stage_oproj(c, bid - O0)
        rops = k2b.early_loads(c, 0, eb // k2b.R_PARTS, eb % k2b.R_PARTS)
        k2b.early_compute(c, 0, eb // k2b.R_PARTS, eb % k2b.R_PARTS, rops)
    if is_l:
        if const_expr(not key.balanced_oproj):
            stage_oproj(c, bid - O0)
        lops = k2b.early_loads(c, 1, lb // k2b.L_PARTS, lb % k2b.L_PARTS)
        k2b.early_compute(c, 1, lb // k2b.L_PARTS, lb % k2b.L_PARTS, lops)
    if (bid >= O0) & ~is_r & ~is_l:
        if const_expr(not key.balanced_oproj):
            stage_oproj(c, bid - O0)
        gi = bid - O0
        if gi < k2b.G_TASKS:
            k2b.early_compute(c, 2, gi, 0, k2b.early_loads(c, 2, gi, 0))
    # ---- AttnRes: the reduced o_proj -> x; then route / SiTU
    if bid < NA:
        if const_expr(key.front_panels):
            stage_front_panels(c)
        else:
            k2a.stage_reduce(c, bid)
            stage_slice(c, bid)
            for token in range(bid, s, NA):
                stage_gate(c, token)
            stage_mix(c, bid)
        for token in range(bid, s, NA):
            k2b.stage_route(c, token)
        if (bid >= H0) & (bid < H0 + s):
            k2b.stage_h(c, bid - H0)
    # ---- the routed experts
    # the latent (ready ~10 us before the route) quantized first
    if const_expr(key.producer_latent_quant):
        k2b.load_shared_latq(c)
    else:
        k2b.load_latq(c)
    nu = k2b.load_route(c)
    k2b.run_routed(c, nu, BLOCKS)
    k2b.run_sdown_rest(c, bid, w_sd)
    k2b.run_sdown(c, bid, w_sd)
    if (bid >= k2b.LN0) & (bid < k2b.LN0 + k2b.LN_TASKS):
        k2b.stage_lnorm(c, bid - k2b.LN0)
    if bid < k2b.UP_TASKS:
        k2b.stage_up(c, bid)
    if const_expr(key.reduce):  # noqa: SIM102 (traced: `and` does not combine traced values)
        if (bid >= k2b.RED0) & (bid < k2b.RED0 + k2b.RED_TASKS):
            k2b.stage_reduce(c, bid - k2b.RED0)


_BUILDS: dict = {}


def build(key: K2Build):
    if key in _BUILDS:
        return _BUILDS[key]
    s, ns = key.tokens, key.nblocks + 1
    assert 1 <= s <= (MAX_TOKENS if key.wide else k2b.MOE_TILE_TOKENS)
    assert 1 <= key.down_batch <= 10
    assert key.inter_vec in (1, 2)
    assert not key.wide or not key.stream_oproj
    assert not key.wide or key.compact_lds
    assert not key.stream_attn or key.wide and s > 16
    assert not key.sparse_routes or key.wide
    assert not key.stream_x or key.wide and s > 16
    assert not key.front_panels or key.stream_x
    assert not key.stream_x or s * (ns + 1) + 128 <= WAVES * 64 * 4
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
        latq: fx.Array[fx.Int32, s * k2b.LAT // 8, 16]
        latsc: fx.Array[scale_type, s * k2b.LAT // 32, 16]
        av: fx.Array[fx.Float32, THREADS, 16]

    @fx.struct
    class DownLds:
        interq: fx.Array[fx.Int32, s * k2b.TOPK * 48, 16]
        intsc: fx.Array[scale_type, s * k2b.TOPK * k2b.UG_GROUPS, 16]
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
    class NarrowStages:
        early: EarlyLds
        attn: AttnLds
        ug: UgLds
        down: DownLds

    @fx.struct
    class Smem:
        red: fx.Array[fx.Float32, WAVES * 64 * 4, 16]

    keyed = key_tuple(key, _SOURCES)
    name = (
        f"k3_mono_k2_s{s}_nb{key.nblocks}_x{int(key.reset)}_r{int(key.reduce)}"
        f"_lqp{int(key.producer_latent_quant)}_db{key.down_batch}"
        f"_iv{key.inter_vec}_op{int(key.trim_oproj)}{int(key.stream_oproj)}"
        f"{int(key.balanced_oproj)}_w{int(key.wide)}{int(key.compact_lds)}"
        f"{int(key.sparse_routes)}_as{int(key.stream_attn)}"
        f"_xs{int(key.stream_x)}{int(key.front_panels)}"
    )
    if key.fault_rank >= 0:
        name += (
            f"_f{key.fault_rank}_{key.fault_epoch}_{key.fault_layer}_{key.fault_ticks}"
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
        health: Int64,
        alarm: Int64,
    ):
        _ = keyed
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        alloc = fx.SharedAllocator()
        lds = alloc.allocate(Smem).peek()
        if const_expr(key.wide):
            # the route table lives only while the routed stages run
            st = alloc.allocate(StageLds)
            rl = st.routed.route.peek()
            el, al = st.early.peek(), st.attn.peek()
            ul, dl = st.routed.stages.ug.peek(), st.routed.stages.down.peek()
        else:
            rl = alloc.allocate(RouteLds).peek()
            st = alloc.allocate(NarrowStages)
            el, al = st.early.peek(), st.attn.peek()
            ul, dl = st.ug.peek(), st.down.peek()
        if const_expr(key.fault_rank >= 0):
            stall(
                key.fault_rank,
                key.fault_epoch,
                key.fault_layer,
                key.fault_ticks,
                rank,
                epoch,
                layer,
            )
        mb = Mailbox(step_tag(epoch, layer) - 1, health, alarm, region_ids(REGIONS))
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
            "eps": key.eps,
            "out_eps": key.out_eps,
            "ln_eps": key.ln_eps,
            "compact_lds": key.compact_lds,
            "stream_attn": key.stream_attn,
            "sparse_routes": key.sparse_routes,
            "stream_x": key.stream_x,
            "down_batch": key.down_batch,
            "inter_vec": key.inter_vec,
            "trim_oproj": key.trim_oproj,
            "stream_oproj": key.stream_oproj,
            "balanced_oproj": key.balanced_oproj,
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
        if const_expr(key.producer_latent_quant):
            c["local_lat"] = sreg(scratch, lay["local_lat"][0], "local_lat")
            c["shared_latq"] = peer["qlat_own"]
            c["shared_latsc"] = peer["qlatsc_own"]
        if const_expr(key.stream_x):
            c["xpanels"] = sreg(scratch, lay["xpanels"][0], "xpanels")
        run_stages(c, key)

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
        health: Int64,
        alarm: Int64,
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
            health,
            alarm,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    ABI.check(k2, launch)
    _BUILDS[key] = launch
    return launch


TP_ = k2b.TP


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
    health,
    alarm,
):
    """Host launcher: ``core`` [S, 1536] (o_proj's input), ``prefix`` updated in
    place, ``out`` the MoE output [S, 7168]; MoE weights as ``moe``."""
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
        health.data_ptr(),
        alarm,
        stream=torch.cuda.current_stream(),
    )
