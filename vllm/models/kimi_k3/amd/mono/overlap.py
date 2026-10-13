# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K1/K2 overlap with worker reuse or reserved workers and per-group buffers.

Every group contains complete independent requests. K2 waits for all local K1
writers before reading core/prefix. Its TP reduction then waits for peer groups.
Separate scratch and peer regions avoid slot reuse while checking this schedule.
Serving selects worker reuse for eligible steps with at most 16 rows.
"""

from dataclasses import dataclass, replace

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int32, Int64

from vllm.models.kimi_k3.amd.mono import layer as k2
from vllm.models.kimi_k3.amd.mono.attention import back as k2a
from vllm.models.kimi_k3.amd.mono.attention import kda as k1
from vllm.models.kimi_k3.amd.mono.attention import parallel
from vllm.models.kimi_k3.amd.mono.attention.kda import (
    BLOCKS,
    COLS,
    HD,
    HIDDEN,
    NA,
    NH,
    PROJ,
    THREADS,
    WAVES,
    stage_gate,
    stage_mix,
    stage_slice,
    step_tag,
)
from vllm.models.kimi_k3.amd.mono.common.debug import stamp, stamp_begin, stamp_flush
from vllm.models.kimi_k3.amd.mono.common.ops import ld_i32, load_ptr64, traced, uniform
from vllm.models.kimi_k3.amd.mono.common.plan import key_tuple
from vllm.models.kimi_k3.amd.mono.common.ranks import peer_bases
from vllm.models.kimi_k3.amd.mono.common.sync import Mailbox, preg, publish, shift, sreg
from vllm.models.kimi_k3.amd.mono.stages import gemv
from vllm.models.kimi_k3.amd.mono.stages import moe as k2b

_STREAM = fx.Stream(None)
_BUILDS: dict = {}
TL_POINTS = 21


@dataclass(frozen=True)
class OverlapBuild:
    k1: k1.KdaPreBuild
    k2: k2.K2Build
    requests_per_group: int = 1
    k1_workers: int = 128
    reuse_workers: bool = False
    tagged_core: bool = False
    ring_buffers: bool = False


def groups(key):
    assert key.k1.tokens == key.k2.tokens
    assert key.k1.tokens % key.k1.qlen == 0
    assert key.reuse_workers or not key.k2.latent_quant_workers
    if key.reuse_workers:
        assert not key.ring_buffers
        assert key.k1.tokens <= 16 and key.k1_workers == BLOCKS
        return [(0, key.k1.tokens)]
    assert key.requests_per_group in (1, 2)
    size = key.requests_per_group * key.k1.qlen
    assert size <= 16
    return [
        (first, min(size, key.k1.tokens - first))
        for first in range(0, key.k1.tokens, size)
    ]


def layouts(key):
    result: list[dict[str, int]] = []
    local, peer, trace = 0, 0, 0
    tiles = groups(key)
    capacity = max(rows for _, rows in tiles)
    for tile, (first, rows) in enumerate(tiles):
        one = replace(key.k1, tokens=rows)
        if key.ring_buffers and tile >= 2:
            item = {**result[tile % 2], "first": first, "rows": rows, "trace": trace}
        else:
            allocated = capacity if key.ring_buffers else rows
            local = (local + 255) // 256 * 256
            item = dict(first=first, rows=rows, k1=local, peer=peer, trace=trace)
            local += k1.scratch_bytes(replace(key.k1, tokens=allocated))
            local = (local + 255) // 256 * 256
            item["k2"] = local
            local += k2.scratch_bytes(replace(key.k2, tokens=allocated))
            peer += k2.peer_bytes(allocated)
        item["k1_trace_words"] = k1.timeline_words(one)
        trace += item["k1_trace_words"] + BLOCKS * TL_POINTS
        result.append(item)
    if key.ring_buffers:
        for item in result:
            item["consumed"] = peer
        peer += len(tiles) * (BLOCKS - key.k1_workers) * k2b.TP * 8
    ready = (local + 255) // 256 * 256
    total = ready + len(result) * key.k1_workers * 8
    if key.tagged_core:
        assert key.reuse_workers
        total += key.k1.tokens * PROJ * 4
    return result, ready, total, peer, trace


def scratch_bytes(key):
    return layouts(key)[2]


def peer_bytes(key):
    return layouts(key)[3]


def timeline_words(key):
    return layouts(key)[4]


K1_ARGS = tuple(
    name for name in k1.ABI.names if name not in ("scratch", "epoch", "layer", "tl")
)
K2_ARGS = tuple(
    name
    for name in k2.ABI.names
    if name not in ("scratch", "epoch", "layer", "tl", "peers", "rank", "diag")
)
ARGUMENTS = tuple("k1_" + name for name in K1_ARGS) + tuple(
    "k2_" + name for name in K2_ARGS
)
SCALARS = {"blk_sm", "blk_sr", "cs_seq", "cs_dim", "cs_tok", "rs_seq", "st_stride"}


def argument(argv, name):
    index = ARGUMENTS.index(name)
    if name[3:] in SCALARS:
        return uniform(ld_i32(argv, 2 * index))
    return load_ptr64(argv, index)


def group_tag(epoch, layer, tile, ring):
    if not ring:
        return step_tag(epoch, layer)
    # Negative tags keep grouped generations disjoint from standalone launches.
    e = uniform(ld_i32(epoch, 0))
    return fx.Int32(-(1 << 31)) + (e % (1 << 19)) * 4096 + layer * 16 + tile


@traced
def wait_consumed(control, own, tile, workers, tid):
    poll = control.poll
    if tid < workers:
        poll([(own, (tile * k2b.TP + p) * workers + tid, 1) for p in range(k2b.TP)])
    gpu.barrier()


@traced
def publish_consumed(control, destinations, tile, workers, rank, bid, tid):
    put = control.put
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    if tid == 0:
        for p in range_constexpr(k2b.TP):
            put(destinations[p], (tile * k2b.TP + rank) * workers + bid, 1)


@traced
def run_k1(c, key):
    bid, tid = c["bid"], c["tid"]
    heads = c["S"] // c["L"] * NH
    producers = key.k1_workers - heads
    c["head_workers"] = heads
    on, tls = c["on"], c["tls"]
    stamp_begin(on, tls, tid, k1.TL_POINTS)
    if bid < NA:
        stage_slice(c, bid)
        stamp(on, tls, tid, 1)
        if bid < c["S"]:
            stage_gate(c, bid)
        stamp(on, tls, tid, 2)
        stage_mix(c, bid)
        stamp(on, tls, tid, 3)
    else:
        for task in range(bid - NA, k1.GEMV_CTAS, key.k1_workers - NA):
            parallel.cached_projection([c], 2 * task, 2 * task + 1)
        stamp(on, tls, tid, 6)
    if const_expr(key.reuse_workers):
        k1.kda_tasks(c, on, tls)
    elif const_expr(not (key.k1.pipeline & k1.TOKEN_PIPELINE)):
        for task in range(bid, heads * c["NVCH"], key.k1_workers):
            k1.kda_task(c, task, on, tls)
    elif bid < producers:
        for task in range(bid, heads * c["NVCH"], producers):
            k1.kda_task(c, task, on, tls)
    else:
        head = bid - producers
        parallel.head_prepare(c, head)
        for token in range_constexpr(c["L"]):
            parallel.output(c, head * c["L"] + token)


@traced
def run_k2(c, workers, shared_workers, key):
    bid, tid = c["bid"], c["tid"]
    s, on, tls = c["S"], c["tl_on"], c["tls"]
    if bid >= NA:
        stride = workers - NA
        task = bid - NA
        o_ops = k2a.o_loads(c, task)
        if c["tid"] < c["core_writers"]:
            c["poll"]([(c["core_ready"], c["tid"], 1)])
        gpu.barrier()
        stamp(on, tls, tid, 17)
        k2a.load_core(c)
        while task < k2a.OTASKS:
            nxt = task + stride
            o_next = k2a.o_loads(c, fx.min(nxt, k2a.OTASKS - 1))
            k2a.stage_o(c, task, o_ops)
            task, o_ops = nxt, o_next
        stamp(on, tls, tid, 1)
        for task in range(bid - NA, k2b.R_TASKS + k2b.L_TASKS + k2b.G_TASKS, stride):
            if task < k2b.R_TASKS:
                group, part = task // k2b.R_PARTS, task % k2b.R_PARTS
                k2b.early_compute(c, 0, group, part, k2b.early_loads(c, 0, group, part))
            elif task < k2b.R_TASKS + k2b.L_TASKS:
                index = task - k2b.R_TASKS
                group, part = index // k2b.L_PARTS, index % k2b.L_PARTS
                k2b.early_compute(c, 1, group, part, k2b.early_loads(c, 1, group, part))
            else:
                group = task - k2b.R_TASKS - k2b.L_TASKS
                k2b.early_compute(c, 2, group, 0, k2b.early_loads(c, 2, group, 0))
        stamp(on, tls, tid, 2)
    else:
        k2a.stage_reduce(c, bid)
        stamp(on, tls, tid, 1)
        stage_slice(c, bid)
        if bid < s:
            stage_gate(c, bid)
        stage_mix(c, bid)
        stamp(on, tls, tid, 9)
        if bid < s:
            k2b.stage_route(c, bid)
        if (bid >= s) & (bid < 2 * s):
            k2b.stage_h(c, bid - s)
        stamp(on, tls, tid, 10)
    if bid >= workers - shared_workers:
        k2b.load_h(c)
        for task in range(
            bid - (workers - shared_workers),
            k2b.SD_TASKS if key.full_shared_pipeline else k2b.UP_TASKS,
            shared_workers,
        ):
            row = (task + c["rank"] * k2b.UP_TASKS) % k2b.SD_TASKS
            shared_early_ops = gemv.loads(
                c["lane"], c["wave"], c["args"]["w_sd"], row * k2b.ROWS, k2b.SI
            )
            k2b.stage_sdown(c, row, shared_early_ops)
        stamp(on, tls, tid, 14)
    if const_expr(key.producer_latent_quant):
        k2b.load_shared_latq(c)
    else:
        k2b.load_latq(c)
    stamp(on, tls, tid, 12)
    nu = k2b.load_route(c)
    stamp(on, tls, tid, 11)
    k2b.run_routed(c, nu, workers, key.down_workers)
    if const_expr(not key.full_shared_pipeline):  # noqa: SIM102
        if bid >= k2b.UP_TASKS:
            k2b.load_h(c)
            for task in range(
                k2b.UP_TASKS + bid - k2b.UP_TASKS, k2b.SD_TASKS, workers - k2b.UP_TASKS
            ):
                row = (task + c["rank"] * k2b.UP_TASKS) % k2b.SD_TASKS
                shared_rest_ops = gemv.loads(
                    c["lane"], c["wave"], c["args"]["w_sd"], row * k2b.ROWS, k2b.SI
                )
                k2b.stage_sdown(c, row, shared_rest_ops)
    stamp(on, tls, tid, 4)
    if bid < k2b.LN_TASKS:
        k2b.stage_lnorm(c, bid)
    stamp(on, tls, tid, 5)
    if bid < k2b.UP_TASKS:
        k2b.stage_up(c, bid)
    stamp(on, tls, tid, 6)
    if bid < k2b.RED_TASKS:
        k2b.stage_reduce(c, bid)
    stamp(on, tls, tid, 7)


def build(key):
    if key in _BUILDS:
        return _BUILDS[key]
    tiles = groups(key)
    layout, ready_offset, _, _, _ = layouts(key)
    s = max(rows for _, rows in tiles)
    L = key.k1.qlen
    ns = key.k1.nblocks + 1
    state_rows = k1._state_rows(key.k1)
    assert key.k1.state_rows in (16, 32, 64)
    nvch = HD // state_rows
    assert key.k1.pipeline & 24576
    if key.reuse_workers:
        assert key.k1_workers == BLOCKS
        assert not (key.k1.pipeline & (k1.TOKEN_PIPELINE | 4))
    else:
        assert NA < key.k1_workers < BLOCKS - NA
        if key.k1.pipeline & k1.TOKEN_PIPELINE:
            assert key.k1.pipeline & 4
            assert s // L * NH * nvch <= key.k1_workers - s // L * NH
        else:
            assert not (key.k1.pipeline & 12)
            assert s // L * NH * nvch <= key.k1_workers
        assert 0 <= key.k2.down_workers < BLOCKS - key.k1_workers
    assert not key.ring_buffers or len(tiles) <= 16
    assert not key.k2.balanced_oproj or key.reuse_workers
    assert key.k2.reduce
    assert not key.k2.stream_x
    assert not key.k2.front_panels
    assert key.k1.timeline == key.k2.timeline
    xcols, xrow, xbanks = 2048, 2056, 1

    @fx.struct
    class AttnLds:
        vals: fx.Array[fx.Float32, s * ns * COLS, 16]

    @fx.struct
    class GemvLds:
        xl: fx.Array[fx.Int32, xbanks * s * xrow // 2, 16]

    cached_words = 20 if key.k1.pipeline & 16384 else 16

    @fx.struct
    class CachedGemvLds:
        xl: fx.Array[fx.Int32, s * xrow // 2, 16]
        weights: fx.Array[fx.Int32, 2 * THREADS * cached_words, 16]

    GemvStorage = CachedGemvLds if key.k1.pipeline & 24576 else GemvLds

    @fx.struct
    class KdaLds:
        fa: fx.Array[fx.Float32, L * HD, 16]
        qkl: fx.Array[fx.Float32, L * 2 * HD, 16]
        decay: fx.Array[fx.Float32, L * HD, 16]
        vloc: fx.Array[fx.Float32, L * state_rows, 16]
        ol: fx.Array[fx.Float32, L * state_rows, 16]
        beta: fx.Array[fx.Float32, 16, 16]

    ns = key.k2.nblocks + 1
    max_u = s * k2b.TOPK
    route_rows = 8 if s <= 8 else 16

    @fx.struct
    class RouteLds:
        route: fx.Array[fx.Int32, s * 2 * k2b.TOPK, 16]
        flag: fx.Array[fx.Int32, k2b.E, 16]
        expert: fx.Array[fx.Int32, max_u, 16]
        kof: fx.Array[fx.Int32, max_u * route_rows, 16]
        scan: fx.Array[fx.Int32, WAVES, 16]

    @fx.struct
    class EarlyLds:
        xl: fx.Array[
            fx.Int32,
            s * ((HIDDEN if s <= 8 else 4096) + gemv.LDS_PAD) // 2,
            16,
        ]

    @fx.struct
    class K2AttnLds:
        vals: fx.Array[fx.Float32, s * ns * COLS, 16]
        dl: fx.Array[fx.Float32, s * COLS, 16]

    @fx.struct
    class UgLds:
        latq: fx.Array[fx.Int32, s * k2b.LAT // 8, 16]
        latsc: fx.Array[fx.Int32, s * k2b.LAT // 32, 16]
        av: fx.Array[fx.Float32, THREADS, 16]

    @fx.struct
    class DownLds:
        interq: fx.Array[fx.Int32, 4 if key.k2.down_workers else s * k2b.TOPK * 48, 16]
        intsc: fx.Array[
            fx.Int32, 4 if key.k2.down_workers else s * k2b.TOPK * k2b.UG_GROUPS, 16
        ]
        contrib: fx.Array[fx.Float32, s * k2b.TOPK * k2b.ROWS, 16]

    @fx.union
    class WorkerStorage:
        first_attn: AttnLds
        first_gemv: GemvStorage  # type: ignore[valid-type]
        first_kda: KdaLds
        second_early: EarlyLds
        second_attn: K2AttnLds
        second_ug: UgLds
        second_down: DownLds

    @fx.struct
    class Smem:
        red: fx.Array[fx.Float32, WAVES * 64 * 4, 16]
        tls: fx.Array[fx.Int64, TL_POINTS, 16]

    keyed = (
        ("k1", key_tuple(key.k1, k1._SOURCES)),
        ("k2", key_tuple(key.k2, k1._SOURCES)),
        ("requests_per_group", key.requests_per_group),
        ("k1_workers", key.k1_workers),
        ("reuse_workers", key.reuse_workers),
        ("tagged_core", key.tagged_core),
        ("ring_buffers", key.ring_buffers),
    )
    name = (
        f"k3_mono_overlap_m{key.k1.tokens}_l{L}_r{key.requests_per_group}"
        f"_w{key.k1_workers}_v{state_rows}_tc{int(key.tagged_core)}"
        f"_lq{key.k2.latent_quant_workers}_lqp{int(key.k2.producer_latent_quant)}"
        f"_dw{key.k2.down_workers}_ring{int(key.ring_buffers)}"
        f"_lv{key.k2.latq_vec}"
        f"_db{key.k2.down_batch}_iv{key.k2.inter_vec}"
        f"_optrim{int(key.k2.trim_oproj)}_opstream{int(key.k2.stream_oproj)}"
        f"_opbalanced{int(key.k2.balanced_oproj)}"
        f"_t{int(key.k2.timeline)}"
    )

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kernel(
        argv: Int64,
        scratch: Int64,
        peers: Int64,
        epoch: Int64,
        rank: Int32,
        layer: Int32,
        full_tl: Int64,
    ):
        _ = keyed
        tid = fx.thread_idx.x
        physical_bid = fx.block_idx.x
        alloc = fx.SharedAllocator()
        lds = alloc.allocate(Smem).peek()
        rl = alloc.allocate(RouteLds).peek()
        worker = alloc.allocate(WorkerStorage)
        first_attn = worker.first_attn.peek()
        first_gemv = worker.first_gemv.peek()
        first_kda = worker.first_kda.peek()
        second_early = worker.second_early.peek()
        second_attn = worker.second_attn.peek()
        second_ug = worker.second_ug.peek()
        second_down = worker.second_down.peek()
        ready = sreg(scratch, ready_offset, "group_core_ready")
        if const_expr(key.ring_buffers):
            consumed_bases = peer_bases(peers, k2b.TP)
            consumed_own = load_ptr64(peers, rank)
            consumed_offset = layout[0]["consumed"]
            consumed = preg(consumed_own, consumed_offset, "group_consumed")
            consumed_peers = [
                preg(b, consumed_offset, "group_consumed") for b in consumed_bases
            ]
            control = Mailbox(step_tag(epoch, layer) - 1)
        if physical_bid < key.k1_workers:
            bid = physical_bid
            al, gl, kl = first_attn, first_gemv, first_kda
            prefix = argument(argv, "k1_prefix")
            delta = argument(argv, "k1_delta")
            blocks = argument(argv, "k1_blocks")
            blk_sm = argument(argv, "k1_blk_sm")
            blk_sr = argument(argv, "k1_blk_sr")
            ares_nw = argument(argv, "k1_ares_nw")
            ares_qk = argument(argv, "k1_ares_qk")
            in_nw = argument(argv, "k1_in_nw")
            w_in = argument(argv, "k1_w_in")
            w_fb = argument(argv, "k1_w_fb")
            conv_w = argument(argv, "k1_conv_w")
            conv_st = argument(argv, "k1_conv_st")
            cs_seq = argument(argv, "k1_cs_seq")
            cs_dim = argument(argv, "k1_cs_dim")
            cs_tok = argument(argv, "k1_cs_tok")
            a_log = argument(argv, "k1_a_log")
            dt_bias = argument(argv, "k1_dt_bias")
            on_w = argument(argv, "k1_on_w")
            rstate = argument(argv, "k1_rstate")
            rs_seq = argument(argv, "k1_rs_seq")
            st_idx = argument(argv, "k1_st_idx")
            st_stride = argument(argv, "k1_st_stride")
            num_acc = argument(argv, "k1_num_acc")
            core_out = argument(argv, "k1_core_out")
            x_dbg = argument(argv, "k1_x_dbg")
            proj_dbg = argument(argv, "k1_proj_dbg")

            for tile in range_constexpr(len(tiles)):
                if const_expr(key.ring_buffers and tile >= 2):
                    wait_consumed(
                        control, consumed, tile - 2, BLOCKS - key.k1_workers, tid
                    )
                first, s = tiles[tile]
                ns = key.k1.nblocks + 1
                lay = k1.scratch_layout(replace(key.k1, tokens=s))
                tile_scratch = scratch + fx.Int64(layout[tile]["k1"])
                tl = full_tl + fx.Int64(layout[tile]["trace"] * 8)
                mb = Mailbox(group_tag(epoch, layer, tile, key.ring_buffers) - 1)
                c = {
                    "S": s,
                    "XCOLS": xcols,
                    "XROWS": max(rows for _, rows in tiles),
                    "L": L,
                    "VCH": state_rows,
                    "NVCH": nvch,
                    "NS": ns,
                    "NB": key.k1.nblocks,
                    "SL": key.k1.state_len,
                    "tid": tid,
                    "bid": bid,
                    "lane": tid % 64,
                    "wave": tid // 64,
                    "delta": key.k1.delta,
                    "debug": key.k1.debug,
                    "eps": key.k1.eps,
                    "out_eps": key.k1.out_eps,
                    "onorm_eps": key.k1.onorm_eps,
                    "lb": key.k1.lower_bound,
                    "kda0": key.k1.kda0,
                    "pipeline": key.k1.pipeline,
                    "projection_chunks": key.k1.projection_chunks,
                    "x_ready": bool(key.k1.pipeline & 224),
                    "head_workers": key.k1.head_workers,
                    "head_total": s // L * NH,
                    "head_first": 0,
                    "events": tl + fx.Int64(BLOCKS * k1.TL_POINTS * 8),
                    "event_stride": max(2 * L + 10, 16),
                    "event_prod_base": 0,
                    "event_head_base": s // L * NH * nvch,
                    "event_out_base": s // L * NH * (nvch + 1),
                    "event_init_base": s // L * NH * (nvch + 1) + s * NH,
                    "xp": key.k1.xp,
                    "write_idx": key.k1.write_idx,
                    "gate_local": False,
                    "on": key.k1.timeline,
                    "tls": lds.tls.ptr,
                    "put": mb.put,
                    "put_bf": mb.put_bf,
                    "poll": mb.poll,
                    "red": lds.red.ptr,
                    "vals": al.vals.ptr,
                    "xl": gl.xl.ptr,
                    "fa": kl.fa.ptr,
                    "qkl": kl.qkl.ptr,
                    "decay": kl.decay.ptr,
                    "vloc": kl.vloc.ptr,
                    "ol": kl.ol.ptr,
                    "beta": kl.beta.ptr,
                    "xbuf": tile_scratch + fx.Int64(lay["x"][0]),
                    "args": {
                        "prefix": prefix + fx.Int64(first * HIDDEN * 2),
                        "delta": delta + fx.Int64(first * HIDDEN * 2)
                        if key.k1.delta
                        else delta,
                        "blocks": blocks + fx.Int64(first) * fx.Int64(blk_sm) * 2,
                        "blk_sm": blk_sm,
                        "blk_sr": blk_sr,
                        "ares_nw": ares_nw,
                        "ares_qk": ares_qk,
                        "in_nw": in_nw,
                        "w_in": w_in,
                        "w_fb": w_fb,
                        "conv_w": conv_w,
                        "conv_st": conv_st,
                        "cs_seq": cs_seq,
                        "cs_dim": cs_dim,
                        "cs_tok": cs_tok,
                        "a_log": a_log,
                        "dt_bias": dt_bias,
                        "on_w": on_w,
                        "rstate": rstate,
                        "rs_seq": rs_seq,
                        "st_idx": st_idx
                        + fx.Int64(first // L) * fx.Int64(st_stride) * 4,
                        "st_stride": st_stride,
                        "num_acc": num_acc + fx.Int64(first // L * 4),
                        "core_out": core_out + fx.Int64(first * PROJ * 2),
                        "x_dbg": x_dbg + fx.Int64(first * HIDDEN * 2)
                        if key.k1.debug
                        else x_dbg,
                        "proj_dbg": proj_dbg + fx.Int64(first * k1.NPROJ * 2)
                        if key.k1.debug
                        else proj_dbg,
                    },
                }
                for region in ("part", "wgt", "msq", "xrdy", "proj", "qk", "onrm"):
                    c[region] = sreg(tile_scratch, lay[region][0], region)
                if const_expr(key.k1.pipeline & 12):
                    c["head_ready"] = sreg(
                        tile_scratch, lay["head_ready"][0], "head_ready"
                    )
                    c["head_data"] = tile_scratch + fx.Int64(lay["head_data"][0])
                    c["ybuf"] = tile_scratch + fx.Int64(lay["y"][0])
                    if const_expr(key.k1.pipeline & k1.MAILBOX_OUTPUT):
                        c["output_values"] = sreg(
                            tile_scratch, lay["y"][0], "output_values"
                        )
                if const_expr(key.k1.pipeline & 24576):
                    c["wl"] = gl.weights.ptr
                if const_expr(key.tagged_core):
                    c["core_mailbox"] = sreg(
                        scratch, ready_offset + key.k1_workers * 8, "core_mailbox"
                    )

                run_k1(c, key)
                if const_expr(not key.tagged_core):
                    publish(
                        c["put"],
                        ready,
                        tile * key.k1_workers + bid,
                        fx.Int32(1),
                        tid == 0,
                    )
                stamp(c["on"], lds.tls.ptr, tid, 14)
                stamp_flush(c["on"], lds.tls.ptr, tl, tid, bid, k1.TL_POINTS)
        if physical_bid >= (0 if key.reuse_workers else key.k1_workers):
            bid = physical_bid - (0 if key.reuse_workers else key.k1_workers)
            workers = BLOCKS if key.reuse_workers else BLOCKS - key.k1_workers
            el, al, ul, dl = second_early, second_attn, second_ug, second_down
            core = argument(argv, "k2_core")
            w_o = argument(argv, "k2_w_o")
            prefix = argument(argv, "k2_prefix")
            blocks = argument(argv, "k2_blocks")
            blk_sm = argument(argv, "k2_blk_sm")
            blk_sr = argument(argv, "k2_blk_sr")
            ares_nw = argument(argv, "k2_ares_nw")
            ares_qk = argument(argv, "k2_ares_qk")
            in_nw = argument(argv, "k2_in_nw")
            w_gate = argument(argv, "k2_w_gate")
            bias = argument(argv, "k2_bias")
            w_ld = argument(argv, "k2_w_ld")
            w_sgu = argument(argv, "k2_w_sgu")
            w_sd = argument(argv, "k2_w_sd")
            w13 = argument(argv, "k2_w13")
            w13s = argument(argv, "k2_w13s")
            w2 = argument(argv, "k2_w2")
            w2s = argument(argv, "k2_w2s")
            ln_w = argument(argv, "k2_ln_w")
            w_up = argument(argv, "k2_w_up")
            out = argument(argv, "k2_out")

            bases = peer_bases(peers, k2b.TP)
            own = load_ptr64(peers, rank)
            par = uniform(ld_i32(epoch, 0)) % 2
            for tile in range_constexpr(len(tiles)):
                if const_expr(key.ring_buffers and tile >= 2):
                    wait_consumed(control, consumed, tile - 2, workers, tid)
                first, s = tiles[tile]
                ns = key.k2.nblocks + 1
                route_rows = 8 if s <= 8 else 16
                lay = k2.scratch_layout(replace(key.k2, tokens=s))
                play = k2.peer_layout(s)
                half = k2.peer_bytes(s) // 2
                tile_scratch = scratch + fx.Int64(layout[tile]["k2"])
                tl = full_tl + fx.Int64(
                    (layout[tile]["trace"] + layout[tile]["k1_trace_words"]) * 8
                )
                mb = Mailbox(group_tag(epoch, layer, tile, key.ring_buffers) - 1)
                peer = {}
                for region in ("rlog", "lat", "latr", "final", "ar", "qlat", "qlatsc"):
                    offset = play[region][0] + layout[tile]["peer"]
                    peer[region] = [
                        shift(preg(b, offset, region), fx.Int64(par) * half)
                        for b in bases
                    ]
                    peer[region + "_own"] = shift(
                        preg(own, offset, region), fx.Int64(par) * half
                    )
                c = {
                    "S": s,
                    "max_u": s * k2b.TOPK,
                    "route_rows": route_rows,
                    "NS": ns,
                    "NB": key.k2.nblocks,
                    "tid": tid,
                    "bid": bid,
                    "lane": tid % 64,
                    "wave": tid // 64,
                    "rank": rank,
                    "delta": True,
                    "reset": key.k2.reset,
                    "gate_local": False,
                    "debug": False,
                    "eps": key.k2.eps,
                    "out_eps": key.k2.out_eps,
                    "ln_eps": key.k2.ln_eps,
                    "xp": key.k2.xp,
                    "stream_inter": key.k2.down_workers > 0,
                    "down_batch": key.k2.down_batch,
                    "inter_vec": key.k2.inter_vec,
                    "latq_vec": key.k2.latq_vec,
                    "trim_oproj": key.k2.trim_oproj,
                    "stream_oproj": key.k2.stream_oproj,
                    "balanced_oproj": key.k2.balanced_oproj,
                    "route_timeline": key.k2.timeline,
                    "late_shared_wait": key.k2.late_shared_wait,
                    "producer_quant": key.k2.producer_latent_quant,
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
                    },
                    "xbuf": tile_scratch + fx.Int64(lay["xrow"][0]),
                    "xflags": NA,
                    "tl_on": key.k2.timeline,
                    "tls": lds.tls.ptr,
                    "ln": tile_scratch + fx.Int64(lay["ln"][0]),
                    "shp": tile_scratch + fx.Int64(lay["shp"][0]),
                    "args": {
                        "core": core + fx.Int64(first * k1.PROJ * 2),
                        "w_o": w_o,
                        "prefix": prefix + fx.Int64(first * HIDDEN * 2),
                        "blocks": blocks + fx.Int64(first) * fx.Int64(blk_sm) * 2,
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
                        "out": out + fx.Int64(first * HIDDEN * 2),
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
                    c[region] = sreg(tile_scratch, lay[region][0], region)

                if const_expr(key.k2.latent_quant_workers > 0):
                    for region in ("shared_latq", "shared_latsc"):
                        c[region] = sreg(tile_scratch, lay[region][0], region)
                if const_expr(key.k2.producer_latent_quant):
                    c["local_lat"] = sreg(
                        tile_scratch, lay["local_lat"][0], "local_lat"
                    )
                    c["shared_latq"] = peer["qlat_own"]
                    c["shared_latsc"] = peer["qlatsc_own"]
                stamp_begin(key.k2.timeline, lds.tls.ptr, tid, TL_POINTS)
                if const_expr(key.reuse_workers):
                    if const_expr(key.tagged_core):
                        c["core_mailbox"] = sreg(
                            scratch, ready_offset + key.k1_workers * 8, "core_mailbox"
                        )
                    else:
                        c["core_ready"] = ready
                        c["core_writers"] = key.k1_workers
                    c["tl_points"] = TL_POINTS
                    k2.run_stages(c, key.k2)
                else:
                    c["core_ready"] = shift(ready, tile * key.k1_workers * 8)
                    c["core_writers"] = key.k1_workers
                    run_k2(c, workers, min(16, workers - NA), key.k2)
                if const_expr(key.ring_buffers):
                    publish_consumed(
                        control, consumed_peers, tile, workers, rank, bid, tid
                    )
                stamp_flush(key.k2.timeline, lds.tls.ptr, tl, tid, bid, TL_POINTS)

    @flyc.jit
    def launch(
        argv: Int64,
        scratch: Int64,
        peers: Int64,
        epoch: Int64,
        rank: Int32,
        layer: Int32,
        tl: Int64,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        kernel(argv, scratch, peers, epoch, rank, layer, tl).launch(
            grid=(BLOCKS,), block=(THREADS,), stream=stream
        )

    _BUILDS[key] = launch
    return launch


def argument_values(first, second, *, reset=False):
    """Resolve the ABI values; graph replays may reuse their device block."""

    def value_of(value):
        if value is None:
            return 0
        return value.data_ptr() if isinstance(value, torch.Tensor) else int(value)

    pairs = [("blocks", "blocks"), ("core_out", "core")]
    if not reset:
        pairs.append(("prefix", "prefix"))
    for before, after in pairs:
        assert value_of(first[before]) == value_of(second[after])
    values = []
    for name in ARGUMENTS:
        source = first if name.startswith("k1_") else second
        field = name[3:]
        value = (
            source.get(field)
            if field in ("x_dbg", "proj_dbg", "delta")
            else source[field]
        )
        values.append(value_of(value))
    return tuple(values)


def pack_arguments(first, second, device, *, reset=False):
    """Allocate the stable argument block before graph capture or timing."""
    return torch.tensor(
        argument_values(first, second, reset=reset), dtype=torch.int64, device=device
    )


def launch(key, arguments, scratch, peers, epoch, rank, layer=0, timeline=None):
    assert 0 <= layer < k1.TAG_SLOTS
    build(key)(
        arguments.data_ptr(),
        scratch.data_ptr(),
        peers.data_ptr(),
        epoch.data_ptr(),
        rank,
        layer,
        0 if timeline is None else timeline.data_ptr(),
        stream=torch.cuda.current_stream(),
    )
