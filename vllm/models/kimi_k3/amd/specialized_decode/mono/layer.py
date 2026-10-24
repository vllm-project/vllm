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

An MLA layer's K2 runs its M1b (``mla_back``) first, in the same launch: every
CTA pushes, CTAs 0 .. 55 combine and write ``core`` (a flag a head and 4
tokens), and the o_proj CTAs, their weights in flight, poll the flags of a core
tile before they load it. That removes the M1b -> K2 launch boundary.
"""

from dataclasses import dataclass, replace

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import Int32, Int64

from vllm.models.kimi_k3.amd.specialized_decode.mono.attention import back as k2a
from vllm.models.kimi_k3.amd.specialized_decode.mono.attention import mla_attn as m1c
from vllm.models.kimi_k3.amd.specialized_decode.mono.attention import mla_back as m1b
from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.kda import (
    _SOURCES,
    COLS,
    HIDDEN,
    NA,
    stage_gates,
    stage_mix,
    stage_slice,
    step_tag,
    vals_all,
    vals_words,
)
from vllm.models.kimi_k3.amd.specialized_decode.mono.common.debug import (
    region_ids,
    stamp,
    stamp_begin,
    stamp_flush,
)
from vllm.models.kimi_k3.amd.specialized_decode.mono.common.ops import (
    CM_DEV,
    ld_i32,
    lds_bytes,
    load_ptr64,
    traced,
    uniform,
)
from vllm.models.kimi_k3.amd.specialized_decode.mono.common.plan import (
    BLOCKS,
    LDS_BYTES,
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
from vllm.models.kimi_k3.amd.specialized_decode.mono.stages import gemv as gemv
from vllm.models.kimi_k3.amd.specialized_decode.mono.stages import moe as k2b

O0, O1 = NA, BLOCKS  # o_proj CTAs: every CTA past AttnRes's
O_N = O1 - O0  # 200
OTASKS = k2a.OTASKS  # 448
O_ROUNDS = (OTASKS + O_N - 1) // O_N  # 5
E0 = 160  # router / latent CTAs (their weights in flight under their o_proj)
H0 = 8  # the shared SiTU's CTAs (after their AttnRes work)
SD0 = k2b.DOWN_TASKS  # 224: the shared down's CTAs
# timeline points: 0 start, 1 o_proj done / AR reduced, 2 early done / slice,
# 3 ug, 4 down/sdown, 5 lnorm, 6 up, 7 reduce, 8 gate, 9 mix (x published),
# 10 route, 11 route table, 12 latent quant, 13 the shared SiTU's h done
TL_POINTS = 18  # + 14 the fused M1b's push, 15 INTER loaded, 16 down done,
# 17 the fused M1b's combine

assert E0 + k2b.R_TASKS + k2b.L_TASKS <= BLOCKS and k2b.G_TASKS <= E0 - O0

REGIONS = ("part", "wgt", "msq", "xrdy", "ar", "crdy", "cmb", "ardy") + k2b.REGIONS

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
    timeline: bool = False
    # an MLA layer's M1b in this launch (``mla_back``) when m1b_qlen > 0: its
    # tokens a request, its ranks and its CMB's peer byte offset. CTAs 0 .. 55
    # combine (and push, at ranks 8) before their AttnRes work; the o_proj
    # CTAs poll a flag a (head, 4 tokens) before they load the core tile.
    m1b_qlen: int = 0
    m1b_ranks: int = 1
    m1b_off: int = 0
    m1b_splits: int = 0  # DCP1: M1c's KV splits (``MlaPostBuild.splits``)
    # DCP1: M1c (``mla_attn.cta``) in this launch too, first, on its CTAs: the
    # partials stored device-scope, a flag a CTA (``ardy``) the combine polls
    m1c: bool = False
    m1c_wide: bool = False  # M1c's ``wide`` (its KV pool over 4 GiB, ``k2_launch``)


ABI = KernelAbi(
    (
        "core",
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
        "m1c_q",
        "m1c_kv",
        "m1c_pages",
        "m1c_indptr",
        "m1c_scale",
        "m1c_vscale",
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
    a = k2a.scratch_layout(s, key.nblocks)
    end = max(o + n for o, n in a.values())
    b = k2b.scratch_layout(s)
    lay = dict(a)
    for name, (o, n) in b.items():
        lay[name] = ((end + 15) // 16 * 16 + o, n)
    end = max(o + n for o, n in lay.values())
    lay["xrow"] = ((end + 15) // 16 * 16, s * HIDDEN * 2)
    end = max(o + n for o, n in lay.values())
    # the fused M1b's flags, a combine task each
    lay["crdy"] = ((end + 15) // 16 * 16, m1b.tasks(s, m1b_ctok(key)) * 8)
    # the fused M1c's flags, a CTA each
    lay["ardy"] = ((lay["crdy"][0] + lay["crdy"][1] + 15) // 16 * 16, BLOCKS * 8)
    return lay


def m1b_ctok(key: K2Build) -> int:
    """The fused M1b's tokens a combine task (over its NA CTAs)."""
    if not key.m1b_qlen:
        return m1b.TT
    return m1b.ctok(key.tokens, key.m1b_qlen, key.m1b_splits, NA)


def scratch_bytes(key: K2Build) -> int:
    return max(o + n for o, n in scratch_layout(key).values())


def xl_words(s):
    """The int32 words of the bf16 row tile LDS ``xl`` the GEMV stages share:
    the core tile (o_proj), the early stages' x tiles, every h row (the shared
    down) and the LN tile (up)."""
    tiled = [k2a.OK, HIDDEN // k2b.R_PARTS, HIDDEN // k2b.L_PARTS, HIDDEN, k2b.LAT]
    rows = [(k, min(s, gemv.tile_rows(k))) for k in tiled] + [(k2b.SI, s)]
    return max(n * (k + gemv.LDS_PAD) // 2 for k, n in rows)


@traced
def stage_oproj(c, gi):
    """o_proj tasks gi, gi + 200, gi + 400 (< 448): every task's weights in
    flight at once (one memory latency, not three), then the MFMAs a core tile
    of 16 rows at a time."""
    ops = [k2a.o_loads(c, fx.min(gi + O_N * j, OTASKS - 1)) for j in range(O_ROUNDS)]
    for t0, n in gemv.token_tiles(c["S"]):
        cm = 0
        if const_expr(c["m1b"]):
            # the fused M1b's tasks of these rows (a head, ``TC`` tokens), a
            # thread a flag
            tc = c["TC"]
            f0 = t0 // tc * m1b.NH
            nf = ((t0 + n + tc - 1) // tc) * m1b.NH - f0
            tid = c["tid"]
            if tid < nf:
                c["poll"]([(c["crdy"], f0 + tid, 1)])
            gpu.barrier()
            cm = CM_DEV
        k2a.load_core(c, t0, n, cm)
        for j in range_constexpr(O_ROUNDS):
            t = gi + O_N * j
            if t < OTASKS:
                k2a.stage_o(c, t, ops[j], t0, n)


_BUILDS: dict = {}


def build(key: K2Build):
    if key in _BUILDS:
        return _BUILDS[key]
    s, ns = key.tokens, key.nblocks + 1
    assert 1 <= s <= k2b.MAX_S
    fused = key.m1b_qlen > 0
    assert not fused or (s % key.m1b_qlen == 0 and key.m1b_ranks in (1, TP_))
    assert not key.m1c or (fused and key.m1b_ranks == 1 and key.m1b_splits)
    assert key.m1c or not key.m1c_wide
    if key.m1c:
        m1c.check(s, key.m1b_qlen)
        assert key.m1b_splits == m1c.splits(key.m1b_qlen)
    # the AttnRes stages' delta tile beside their slices
    vextra = s * COLS * 4
    lay = scratch_layout(key)
    play = peer_layout(s)
    half = peer_bytes(s) // 2

    # the route table: a slot's kof row (``few_rows``), else its pick tiles
    few = k2b.few_rows(s)
    nkof = k2b.max_u(s) * s if few else 1
    nslot = 1 if few else k2b.max_u(s) + 1
    npick = 1 if few else s * k2b.TOPK
    ntile = 1 if few else k2b.max_tiles(s)

    @fx.struct
    class RouteLds:
        route: fx.Array[fx.Int32, s * 2 * k2b.TOPK, 16]
        eslot: fx.Array[fx.Int32, k2b.E, 16]
        expert: fx.Array[fx.Int32, k2b.max_u(s), 16]
        kof: fx.Array[fx.Int32, nkof, 16]
        first: fx.Array[fx.Int32, nslot, 16]
        tfirst: fx.Array[fx.Int32, nslot, 16]
        order: fx.Array[fx.Int32, npick, 16]
        tslot: fx.Array[fx.Int32, ntile, 16]
        scan: fx.Array[fx.Int32, WAVES, 16]

    # red, and as words the route table's token bitmaps (``load_route``)
    @fx.struct
    class RedF:
        v: fx.Array[fx.Float32, WAVES * 64 * 4, 16]

    @fx.struct
    class RedI:
        v: fx.Array[fx.Int32, WAVES * 64 * 4, 16]

    @fx.union
    class RedLds:
        f: RedF
        i: RedI

    @fx.struct
    class EarlyLds:
        xl: fx.Array[fx.Int32, xl_words(s), 16]
        av: fx.Array[fx.Float32, THREADS, 16]

    @fx.struct
    class AttnLds:
        vals: fx.Array[fx.Float32, vals_words(s, ns, vextra), 16]
        dl: fx.Array[fx.Float32, s * COLS, 16]

    @fx.struct
    class UgLds:
        latq: fx.Array[fx.Int32, s * k2b.LAT // 8, 16]
        latsc: fx.Array[fx.Int32, k2b.latsc_words(s), 16]

    # every pick's INTER in LDS, or (polled at use) none
    nq = s * k2b.TOPK if k2b.inter_lds(s) else 1

    @fx.struct
    class DownLds:
        interq: fx.Array[fx.Int32, nq * 48, 16]
        intsc: fx.Array[fx.Int32, nq * k2b.UG_GROUPS, 16]
        contrib: fx.Array[fx.Float32, s * k2b.TOPK * k2b.ROWS, 16]

    @fx.struct
    class M1bLds:
        lat: fx.Array[fx.Int32, m1b.TT * m1b.LROW if fused else 1, 16]
        cw: fx.Array[
            fx.Float32, m1b.TT * (m1b.WMAX + m1b.KVL) if key.m1b_splits else 1, 16
        ]

    @fx.struct
    class NoLds:
        v: fx.Array[fx.Int32, 1, 16]

    M1cLds = m1c.smem(key.m1b_qlen) if key.m1c else NoLds

    @fx.union
    class StageLds:
        early: EarlyLds
        attn: AttnLds
        ug: UgLds
        down: DownLds
        m1b: M1bLds

    # the stages' LDS: M1c runs first in a CTA and is done (a barrier) before
    # any of it is touched, so M1c's LDS shares it all
    @fx.struct
    class K2Lds:
        red: RedLds
        route: RouteLds
        stage: StageLds

    @fx.union
    class MainLds:
        k2: K2Lds
        m1c: M1cLds  # type: ignore[valid-type]

    @fx.struct
    class Smem:
        tls: fx.Array[fx.Int64, TL_POINTS, 16]

    # LLVM's own over-the-limit error doesn't stop a build (COMPILE_ONLY)
    assert lds_bytes(Smem) + lds_bytes(MainLds) <= LDS_BYTES, lds_bytes(
        Smem
    ) + lds_bytes(MainLds)

    keyed = key_tuple(key, _SOURCES)
    name = (
        f"k3_mono_k2_s{s}_nb{key.nblocks}_x{int(key.reset)}_r{int(key.reduce)}"
        f"_t{int(key.timeline)}"
    )
    if fused:
        name += f"_m{key.m1b_qlen}_{key.m1b_ranks}"
        name += f"_sp{key.m1b_splits}" if key.m1b_splits else ""
        name += "_c" if key.m1c else ""
        name += "w" if key.m1c_wide else ""
    if key.fault_rank >= 0:
        name += (
            f"_f{key.fault_rank}_{key.fault_epoch}_{key.fault_layer}_{key.fault_ticks}"
        )

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def k2(
        core: Int64,
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
        m1c_q: Int64,
        m1c_kv: Int64,
        m1c_pages: Int64,
        m1c_indptr: Int64,
        m1c_scale: Int32,
        m1c_vscale: Int32,
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
        tl: Int64,
    ):
        _ = keyed
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        alloc = fx.SharedAllocator()
        lds = alloc.allocate(Smem).peek()
        main = alloc.allocate(MainLds)
        rd = main.k2.red
        rl = main.k2.route.peek()
        st = main.k2.stage
        el, al, ul, dl = st.early.peek(), st.attn.peek(), st.ug.peek(), st.down.peek()
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
        for region in ("rlog", "lat", "latq4", "latr", "final", "ar"):
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
            "vals_all": vals_all(s, ns, vextra),
            "eps": key.eps,
            "out_eps": key.out_eps,
            "ln_eps": key.ln_eps,
            "put": mb.put,
            "put_bf": mb.put_bf,
            "put_words": mb.put_words,
            "poll": mb.poll,
            "red": rd.f.peek().v.ptr,
            "redi": rd.i.peek().v.ptr,
            "peer": peer,
            "ar_peer": peer["ar"],
            "ar_own": peer["ar_own"],
            "xl": el.xl.ptr,
            "vals": al.vals.ptr,
            "delta_lds": al.dl.ptr,
            "latq": ul.latq.ptr,
            "latsc": ul.latsc.ptr,
            "av": el.av.ptr,
            "interq": dl.interq.ptr,
            "intsc": dl.intsc.ptr,
            "contrib": dl.contrib.ptr,
            "scan": rl.scan.ptr,
            "rt": {
                "route": rl.route.ptr,
                "eslot": rl.eslot.ptr,
                "expert": rl.expert.ptr,
                "kof": rl.kof.ptr,
                "first": rl.first.ptr,
                "tfirst": rl.tfirst.ptr,
                "order": rl.order.ptr,
                "tslot": rl.tslot.ptr,
            },
            "xbuf": scratch + fx.Int64(lay["xrow"][0]),
            "xflags": NA,
            "tl_on": key.timeline,
            "tls": lds.tls.ptr,
            "ln": scratch + fx.Int64(lay["ln"][0]),
            "shp": scratch + fx.Int64(lay["shp"][0]),
            "m1b": fused,
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
            "lnrdy",
            "shrdy",
        ):
            c[region] = sreg(scratch, lay[region][0], region)
        on, tls = key.timeline, lds.tls.ptr
        stamp_begin(on, tls, tid, TL_POINTS)
        # ---- the fused M1b: core's rows, a flag a combine task
        if const_expr(fused):
            R = key.m1b_ranks
            c["crdy"] = sreg(scratch, lay["crdy"][0], "crdy")
            c.update(
                Q=key.m1b_qlen,
                R=R,
                splits=key.m1b_splits,
                TC=m1b_ctok(key),
                lat=st.m1b.peek().lat.ptr,
                cw=st.m1b.peek().cw.ptr,
                core_cm=CM_DEV,
                cmb=[preg(b, key.m1b_off, "cmb") for b in bases] if R > 1 else [],
                cmb_own=preg(own, key.m1b_off, "cmb"),
            )
            c["args"].update(
                o=o,
                o_sb=o_sb,
                o_sh=o_sh,
                lse=lse,
                lse_sb=lse_sb,
                lse_sh=lse_sh,
                seq_lens=seq_lens,
                qsl=qsl,
                w_v=w_v,
                wv_sh=wv_sh,
                wv_sn=wv_sn,
                w_vs=w_vs,
                vs_sh=vs_sh,
                vs_sn=vs_sn,
                vs_sk=vs_sk,
                gate=gate,
            )
            if const_expr(key.m1c):
                c["ardy"] = sreg(scratch, lay["ardy"][0], "ardy")
                c["m1c_ht"] = m1c.geometry(key.m1b_qlen)[0]
                ml = main.m1c.peek()
                m1c.cta(
                    {
                        "S": s,
                        "Q": key.m1b_qlen,
                        "stamps": False,
                        "dbg": 0,
                        "wide": key.m1c_wide,
                        # no warp specialization in here: with it the fused
                        # TP8 builds fault (memory access) where M1c alone
                        # passes (not root-caused)
                        "ws": 0,
                        "bid": bid,
                        "tid": tid,
                        "cm": CM_DEV,
                        "ks": ml.ks.ptr,
                        "xm": ml.xm.ptr,
                        "xp": ml.xp.ptr,
                        "xi": ml.xi.ptr,
                        "args": {
                            "qb": m1c_q,
                            "kv": m1c_kv,
                            "pages": m1c_pages,
                            "indptr": m1c_indptr,
                            "part_o": o,
                            "part_lse": lse,
                            "scale": m1c_scale,
                            "vscale": m1c_vscale,
                        },
                        "done": lambda f: publish(c["put"], c["ardy"], f, 1, tid == 0),
                    }
                )
                # M1c's LDS is the stages' (``MainLds``): its last DMAs (a
                # block past its end, prefetched) landed, every wave done
                m1c.wait_dma()
            if const_expr(R > 1):
                m1b.stage_push(c)
            stamp(on, tls, tid, 14)
            if bid < NA:
                m1b.combine_tasks(
                    c, NA, lambda task: publish(c["put"], c["crdy"], task, 1, tid == 0)
                )
            stamp(on, tls, tid, 17)
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
            stage_gates(c, bid)
            stamp(on, tls, tid, 8)
            stage_mix(c, bid)
            stamp(on, tls, tid, 9)
            # the shared SiTU of token t on CTA (t + H0) % 56, first: SGU lands
            # before the logits
            hb = (bid + NA - H0) % NA
            for rr in range_constexpr((s + NA - 1) // NA):
                if hb + NA * rr < s:
                    k2b.stage_h(c, hb + NA * rr)
            stamp(on, tls, tid, 13)
            for rr in range_constexpr((s + NA - 1) // NA):
                if bid + NA * rr < s:
                    k2b.stage_route(c, bid + NA * rr)
            stamp(on, tls, tid, 10)
        if bid >= NA:
            stamp(on, tls, tid, 2)
        # ---- the routed experts
        # the latent (ready ~10 us before the route) quantized first
        if const_expr(k2b.few_rows(s)):
            k2b.load_latq4(c)
        else:
            k2b.load_latq(c)
        stamp(on, tls, tid, 12)
        nt = k2b.load_route(c)
        stamp(on, tls, tid, 11)
        total = nt * k2b.UG_GROUPS
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
        if bid < k2b.DOWN_TASKS:
            dops = k2b.down_loads(c, bid, nt, fx.min(c["wave"], nt - 1))
            if const_expr(k2b.inter_lds(s)):
                k2b.load_inter(c)
            stamp(on, tls, tid, 15)
            k2b.stage_down(c, bid, nt, dops)
            stamp(on, tls, tid, 16)
        k2b.run_sdown_rest(c, bid, w_sd)
        k2b.run_sdown(c, bid, w_sd)
        stamp(on, tls, tid, 4)
        if (bid >= k2b.LN0) & (bid < k2b.LN0 + k2b.LN_TASKS):
            k2b.run_lnorm(c, bid)
        stamp(on, tls, tid, 5)
        if bid < k2b.UP_TASKS * k2b.up_splits(s):
            k2b.run_up(c, bid)
        stamp(on, tls, tid, 6)
        if const_expr(key.reduce):  # noqa: SIM102 (traced: `and` does not combine traced values)
            if (bid >= k2b.RED0) & (bid < k2b.RED0 + k2b.RED_TASKS):
                k2b.stage_reduce(c, bid - k2b.RED0)
        stamp(on, tls, tid, 7)
        stamp_flush(on, tls, tl, tid, bid, TL_POINTS)

    @flyc.jit
    def launch(
        core: Int64,
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
        m1c_q: Int64,
        m1c_kv: Int64,
        m1c_pages: Int64,
        m1c_indptr: Int64,
        m1c_scale: Int32,
        m1c_vscale: Int32,
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
        tl: Int64,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        k2(
            core,
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
            m1c_q,
            m1c_kv,
            m1c_pages,
            m1c_indptr,
            m1c_scale,
            m1c_vscale,
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
    health,
    alarm,
    tl=None,
    m1b_in=None,
    m1c_in=None,
):
    """Host launcher: ``core`` [S, 1536] (o_proj's input), ``prefix`` updated in
    place, ``out`` the MoE output [S, 7168]; MoE weights as ``moe``. With
    ``key.m1b_qlen``, ``m1b_in`` is ``mla_back.mla_post``'s (o, lse, seq_lens,
    qsl, w_v, w_v_scale, gate) and this launch writes ``core`` first. With
    ``key.m1c``, ``m1c_in`` is ``mla_attn.mla_attn``'s (q, kv, pages, indptr,
    scale, kv_scale), its partials M1b's o / lse (split 0's views)."""
    assert (m1b_in is not None) == (key.m1b_qlen > 0)
    assert (m1c_in is not None) == key.m1c
    if m1c_in is not None and m1c.wide_kv(m1c_in["kv"]):
        key = replace(key, m1c_wide=True)
    f = build(key)
    f(
        core.data_ptr(),
        *(m1b.host_args(key.m1b_ranks, **m1b_in) if m1b_in is not None else [0] * 16),
        *(m1c.host_args(**m1c_in) if m1c_in is not None else [0] * 6),
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
        ptr(tl),
        stream=torch.cuda.current_stream(),
    )
