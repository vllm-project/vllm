# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The mono decode layer's two launches (README): K1 = the attention seam
(``stages.seam``: slice, gate, the norm into vLLM's MXFP8) + the attention front
(``attention.front``); K2 = the attention back (``attention.back``, wo_b pushed
to every rank) + the FFN seam (the TP sum folded in) + the MoE (``stages.moe``,
its all-reduce in-kernel too).

Tags: K1's hand-offs carry 2 e + 1, K2's 2 e + 2 (e: the launch pair's epoch,
moved on by K2's CTA 0 at its end), so K2's seam may reuse K1's seam regions.
Peer regions alternate by e's parity; every rank runs the same launches, so
the epochs agree across ranks.
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int32, Int64, T

from .attention import back as mb_back
from .attention import front as mb_front
from .attention.device import BLOCKS, CM_DEV, THREADS, gstore, kernel_symbol, rsrc
from .attention.plan import back_scratch, front_scratch
from .common.mx import clamp_fp8
from .common.ops import butterfly, fp8_pack4, memrealtime, peer_bases, traced
from .common.plan import WAVES, first_task, key_tuple
from .common.sync import Mailbox, publish, sreg
from .dims import HIDDEN, INDEX_TOPK, MAX_TOKENS, X_GROUPS, X_WORDS, Dims
from .sources import SOURCES
from .stages import allreduce, moe, seam
from .stages.moe_shape import SORT_NETS, MoeBuild, route_shape

_STREAM = fx.Stream(None)
# K1's MXFP8 x and the MoE's: plain words in both, the same size, so the two
# launches share the MoE's (placed later); any other name is one region, so a
# word is never a tag to one launch and data to another
SHARED_REGIONS = ("x8", "x8s")
COUNTER_WORDS = 2 * 256  # the MoE's ug queue and down counts, a word a slot
COUNTERS = ("ugq", "dq")
# the epoch buffer, in words: the epoch, a mark a CTA (back.EPOCH_MARKS), then on
# lines of their own the MoE's counters
EPOCH_COUNTERS = -(-(mb_back.EPOCH_MARKS + BLOCKS) // 64) * 64
EPOCH_WORDS = EPOCH_COUNTERS + COUNTER_WORDS
# a timeline build's K2 stamps a CTA: the attention back's, then the MoE's
K2_POINTS = mb_back.BACK_POINTS + moe.TL_POINTS


@dataclass(frozen=True)
class MonoBuild:
    tokens: int
    tp: int
    ratio: int  # the layer's compress ratio (its attention's key sources)
    timeline: bool = False
    # K1's seam folds the previous layer's output into the residual; off for a
    # residual that arrives settled (the first layer, after an Engram)
    fold: bool = True
    # a hash-routed MoE (``moe_shape.RouteShape.hash_route``)
    hash_route: bool = False
    # a token's compressed keys at most (0: the model's top-k; ``Dims.topk``)
    topk: int = 0
    # the all-reduces in two hops (``stages.allreduce``)
    two_hop: bool = False
    # the key table's compressed keys are slots, not indices through the block
    # table (``front.stage_kt``)
    global_keys: bool = False


def _variant(key: MonoBuild) -> dict:
    """A build's variant flags past the original's, for its kernel symbol."""
    out = {"h": 1} if key.hash_route else {}
    if key.topk:
        out["k"] = key.topk
    if key.global_keys:
        out["g"] = 1
    if key.two_hop:
        out["a"] = 2
    return out


def _dims(key: MonoBuild) -> Dims:
    return Dims(key.tp, key.topk)


def _moe_key(key: MonoBuild) -> MoeBuild:
    return MoeBuild(
        key.tokens,
        key.tp,
        timeline=key.timeline,
        hash_route=key.hash_route,
        two_hop=key.two_hop,
    )


def _front_key(key: MonoBuild):
    return mb_front.FrontBuild(key.tokens, key.tp, key.ratio, key.timeline, key.topk)


def _back_key(key: MonoBuild):
    return mb_back.BackBuild(key.tokens, key.tp, key.ratio, key.timeline, key.topk)


def scratch_layout(s: int, tp: int, topk: int = 0) -> dict:
    """Every region of step width s's launches -> (byte offset, bytes), disjoint:
    K1's front and seam, K2's back, the MoE's regions, the normed rows and their
    flags. Each width has a scratch of its own (``MonoRunner.scratch``):
    under another width's layout a mailbox pair's tag word could be plain data,
    which a poll may take for a current tag. The MoE's counters, whose slots
    outlive a step width, are in the epoch buffer."""
    d = Dims(tp, topk)
    out: dict = {}
    off = 0

    def place(regions):
        nonlocal off
        base = off
        for name, (o, n) in regions.items():
            assert name not in out or name in SHARED_REGIONS, name
            out[name] = (base + o, n)
        off = max(off, base + max(o + n for o, n in regions.values()))
        off = -(-off // 256) * 256

    place(front_scratch(s, d))
    place(seam.scratch_layout(s))
    place(back_scratch(s, d, start=0))
    regions = moe.scratch_layout(MoeBuild(s, tp)).items()
    mr = {n: v for n, v in regions if n not in COUNTERS}
    lo = min(o for o, _ in mr.values())
    place({n: (o - lo, n_) for n, (o, n_) in mr.items()})
    row = s * HIDDEN * 2
    place({"normed": (0, row), "xrdy_moe": (row + 256, s * 8)})
    place({"krdy": (0, s * 8)})
    return out


def scratch_bytes(s: int, tp: int, topk: int = 0) -> int:
    return max(o + n for o, n in scratch_layout(s, tp, topk).values())


def peer_half_bytes(tp: int) -> int:
    """One parity's peer regions: the attention's partials, then the MoE's."""
    d, s = Dims(tp), MAX_TOKENS
    return seam.attn_peer_bytes(s, d) + moe.peer_bytes(s, d)


def _epoch(epoch):
    return fx.Int32(bo.buffer_load(rsrc(epoch), 0, vec_width=1, dtype=T.i32))


# ---------------------------------------------------------------- K1


@traced
def store_x8(c, t, col, ys, live):
    """The attention seam's norm output (``seam.stage_norm``'s 8 columns at
    ``col``) -> vLLM's MXFP8 of each 32 (the attention's input quant) -> the
    front's X8 / X8S, plain at device scope."""
    amax = abs(ys[0])
    for y in ys[1:]:
        amax = fx.max(amax, abs(y))
    code = moe.vllm_mx_code(butterfly(amax, (1, 2), fx.max))
    mul = ((254 - code) << 23).bitcast(fx.Float32)
    w0 = fp8_pack4(*[clamp_fp8(ys[i] * mul) for i in range(4)])
    w1 = fp8_pack4(*[clamp_fp8(ys[4 + i] * mul) for i in range(4)])
    if live:
        bo.buffer_store(
            fx.Vector.from_elements([w0, w1], fx.Int32),
            rsrc(c["x8"]),
            t * X_WORDS + col // 4,
            cache_modifier=CM_DEV,
        )
        if c["tid"] % 4 == 0:
            bo.buffer_store(
                code,
                rsrc(c["x8s"]),
                t * X_GROUPS + col // 32,
                cache_modifier=CM_DEV,
            )


@traced
def publish_x8(c, t):
    """Token t's two XRDY flags (one a wqkv K half) once its X8 / X8S landed."""
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    if c["tid"] < 2:
        c["mb"].put(c["xrdy"], 2 * t + c["tid"], fx.Int32(1))
    gpu.barrier()


def build_mono_k1(key: MonoBuild):
    s, tp, d = key.tokens, key.tp, _dims(key)
    assert 1 <= s <= MAX_TOKENS
    layout = scratch_layout(s, tp, key.topk)
    fkey = _front_key(key)
    FrontLds = mb_front.front_lds(s, d)
    n_slices = seam.SLICES
    # a norm collapses with its token's gate: the gates first
    GATE0 = n_slices
    NORM0 = n_slices + s
    # past the grid a CTA runs its stages in order: a gate never waits on a norm
    assert n_slices <= BLOCKS

    @fx.struct
    class SeamLds:
        red: fx.Array[fx.Float32, WAVES * 2 * 64 * 4, 16]
        rl: fx.Array[fx.Float32, s * seam.KT, 16]
        fl: fx.Array[fx.Float32, seam.MIX * seam.KT, 16]

    @fx.union
    class K1Lds:
        seam: SeamLds
        front: FrontLds  # type: ignore[valid-type]

    name = kernel_symbol(
        "dsv4_mono_k1",
        s=s,
        tp=tp,
        r=key.ratio,
        tl=key.timeline,
        f=key.fold,
        **_variant(key),
    )
    keyed = key_tuple(key, SOURCES)

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def k1(
        res_in: Int64,
        pend: Int64,
        post_in: Int64,
        comb_in: Int64,
        # unread: the seam collapses with its own pre-mix; dropping it moves
        # the kernels' register allocation
        pre_in: Int64,
        hc_fn: Int64,
        hc_scale: Int64,
        hc_base: Int64,
        attn_w: Int64,
        res_out: Int64,
        post_out: Int64,
        comb_out: Int64,
        pre_out: Int64,
        wqkv: Int64,
        wqkv_s: Int64,
        qn_w: Int64,
        kvn_w: Int64,
        wqb: Int64,
        wqb_s: Int64,
        cos_sin: Int64,
        pos: Int64,
        slot: Int64,
        swa: Int64,
        swa_stride: Int64,
        swa_block: Int32,
        q: Int64,
        swa_idx: Int64,
        swa_lens: Int64,
        comp_block: Int32,
        comp_bt: Int64,
        bt_stride: Int32,
        topk: Int64,
        t2r: Int64,
        kt: Int64,
        klen: Int64,
        normed_out: Int64,
        qr_out: Int64,
        scratch: Int64,
        epoch: Int64,
        tl: Int64,
    ):
        _ = keyed
        bid = fx.block_idx.x
        lds = fx.SharedAllocator().allocate(K1Lds)
        sl = lds.seam.peek()
        ep = _epoch(epoch)
        tag = (ep << 1) + 1
        fargs = {
            "wqkv": wqkv,
            "wqkv_s": wqkv_s,
            "qn_w": qn_w,
            "kvn_w": kvn_w,
            "wqb": wqb,
            "wqb_s": wqb_s,
            "cos_sin": cos_sin,
            "pos": pos,
            "slot": slot,
            "swa": swa,
            "swa_stride": swa_stride,
            "swa_block": swa_block,
            "q": q,
            "swa_idx": swa_idx,
            "swa_lens": swa_lens,
            "comp_block": comp_block,
            "comp_bt": comp_bt,
            "bt_stride": bt_stride,
            "topk": topk,
            "t2r": t2r,
            "kt": kt,
            "klen": klen,
            "qr_out": qr_out,
        }
        flayout = {n: layout[n] for n in front_scratch(s, d)}
        c = mb_front.front_context(fkey, lds.front.peek(), fargs, scratch, tag, flayout)
        amb = Mailbox(tag)
        tid = fx.thread_idx.x
        ca = {
            "S": s,
            "tid": tid,
            "bid": bid,
            "lane": tid % 64,
            "wave": tid // 64,
            "rl": sl.rl.ptr,
            "fl": sl.fl.ptr,
            "red": sl.red.ptr,
            "put": amb.put,
            "put_bf": amb.put_bf,
            "put_words": amb.put_words,
            "poll": amb.poll,
            "args": {
                "res_in": res_in,
                "pend": pend,
                "post_in": post_in,
                "comb_in": comb_in,
                "pre_in": pre_in,
                "hc_fn": hc_fn,
                "hc_scale": hc_scale,
                "hc_base": hc_base,
                "norm_w": attn_w,
                "res_out": res_out,
                "post_out": post_out,
                "comb_out": comb_out,
                "pre_out": pre_out,
            },
            "d": d,
            "fold": key.fold,
        }
        for region in seam.REGIONS:
            ca[region] = sreg(scratch, layout[region][0], region)
        if const_expr(key.timeline):
            c["tl"], c["tl_points"] = tl, mb_front.FRONT_POINTS

        def seam_slice():
            for task in range(first_task(bid, 0), n_slices, BLOCKS):
                seam.stage_slice(ca, task)
            gpu.barrier()

        def store(*v):
            store_x8(c, *v)
            store_normed(normed_out, HIDDEN, *v)

        def norms():
            for t in range(first_task(bid, NORM0), s, BLOCKS):
                seam.stage_norm(ca, t, store)
                publish_x8(c, t)

        def gates():
            for t in range(first_task(bid, GATE0), s, BLOCKS):
                seam.stage_gate(ca, t)

        def seam_rest():
            gates()
            norms()
            gpu.barrier()

        mb_front.run_front(c, fkey, bid, before_wqkv=seam_slice, after_wqkv=seam_rest)

    @flyc.jit
    def launch(
        res_in: Int64,
        pend: Int64,
        post_in: Int64,
        comb_in: Int64,
        pre_in: Int64,
        hc_fn: Int64,
        hc_scale: Int64,
        hc_base: Int64,
        attn_w: Int64,
        res_out: Int64,
        post_out: Int64,
        comb_out: Int64,
        pre_out: Int64,
        wqkv: Int64,
        wqkv_s: Int64,
        qn_w: Int64,
        kvn_w: Int64,
        wqb: Int64,
        wqb_s: Int64,
        cos_sin: Int64,
        pos: Int64,
        slot: Int64,
        swa: Int64,
        swa_stride: Int64,
        swa_block: Int32,
        q: Int64,
        swa_idx: Int64,
        swa_lens: Int64,
        comp_block: Int32,
        comp_bt: Int64,
        bt_stride: Int32,
        topk: Int64,
        t2r: Int64,
        kt: Int64,
        klen: Int64,
        normed_out: Int64,
        qr_out: Int64,
        scratch: Int64,
        epoch: Int64,
        tl: Int64,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        k1(
            res_in,
            pend,
            post_in,
            comb_in,
            pre_in,
            hc_fn,
            hc_scale,
            hc_base,
            attn_w,
            res_out,
            post_out,
            comb_out,
            pre_out,
            wqkv,
            wqkv_s,
            qn_w,
            kvn_w,
            wqb,
            wqb_s,
            cos_sin,
            pos,
            slot,
            swa,
            swa_stride,
            swa_block,
            q,
            swa_idx,
            swa_lens,
            comp_block,
            comp_bt,
            bt_stride,
            topk,
            t2r,
            kt,
            klen,
            normed_out,
            qr_out,
            scratch,
            epoch,
            tl,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    return launch


# ---------------------------------------------------------------- K2


@traced
def store_normed(normed, hidden, t, col, ys, live):
    """The FFN seam's norm output -> the MoE input row (bf16, device scope:
    this launch reads it)."""
    if live:
        for j in range_constexpr(8):
            bo.buffer_store(
                ys[j].to(fx.BFloat16),
                rsrc(normed),
                t * hidden + col + j,
                cache_modifier=CM_DEV,
            )


def _seam_lds(s):
    @fx.struct
    class SeamLds:
        red: fx.Array[fx.Float32, WAVES * 2 * 64 * 4, 16]
        rl: fx.Array[fx.Float32, s * seam.KT, 16]
        fl: fx.Array[fx.Float32, seam.MIX * seam.KT, 16]
        pl: fx.Array[fx.Float32, s * seam.COLS, 16]

    return SeamLds


@traced
def run_ffn(key: MonoBuild, lds, a: dict, amb, bases, own, rank, ep):
    """K2's FFN half on this CTA: the FFN seam (the attention's TP partials
    summed and folded into the residual), the MoE and its all-reduce into
    ``a["out"]``, then the launch pair's epoch moved on."""
    s, tp, d = key.tokens, key.tp, _dims(key)
    layout = scratch_layout(s, tp, key.topk)
    mkey = _moe_key(key)
    rs = route_shape(mkey)
    n_slices = seam.SLICES
    GATE0 = n_slices
    NORM0 = n_slices + s
    bid = fx.block_idx.x
    tid = fx.thread_idx.x
    sl = lds.seam.peek()
    scratch = a["scratch"]
    seam_args = (
        "res_in",
        "post_in",
        "comb_in",
        "pre_in",
        "hc_fn",
        "hc_scale",
        "hc_base",
        "res_out",
        "post_out",
        "comb_out",
        "pre_out",
    )
    ca = {
        "S": s,
        "tid": tid,
        "bid": bid,
        "lane": tid % 64,
        "wave": tid // 64,
        "rl": sl.rl.ptr,
        "fl": sl.fl.ptr,
        "red": sl.red.ptr,
        "pend_lds": sl.pl.ptr,
        "put": amb.put,
        "put_bf": amb.put_bf,
        "put_words": amb.put_words,
        "poll": amb.poll,
        "peer_addr": lambda p: bases[p],
        "rank": rank,
        "sym": own,
        "args": {n: a[n] for n in seam_args} | {"norm_w": a["ffn_w"]},
        "d": d,
        "two_hop": key.two_hop,
    }
    normed = scratch + fx.Int64(layout["normed"][0])
    for region in seam.REGIONS:
        ca[region] = sreg(scratch, layout[region][0], region)
    mlayout = {n: layout[n] for n in moe.scratch_layout(mkey) if n not in COUNTERS}
    mlayout["xrdy"] = layout["xrdy_moe"]
    margs = moe.moe_args(
        normed,
        a["gate_w"],
        a["bias"],
        a["w13"],
        a["w13_s"],
        a["w2"],
        a["w2_s"],
        a["sgu"],
        a["sgu_s"],
        a["sw2"],
        a["sw2_s"],
        a["out"],
    )
    cb = moe.moe_context(
        s, rs, lds.moe.peek(), amb, bases, mlayout, scratch, margs, rank, own
    )
    cb["x_cm"], cb["x_ready"] = CM_DEV, True
    cb["two_hop"] = key.two_hop
    cb["tag"] = ep & 255
    for i, region in enumerate(COUNTERS):
        cb[region] = sreg(a["epoch"], 4 * (EPOCH_COUNTERS + i * moe.TAGS), region)
    for task in range(first_task(bid, 0), n_slices, BLOCKS):
        seam.stage_reduce(ca, task)
        seam.stage_slice(ca, task)
    for t in range(first_task(bid, GATE0), s, BLOCKS):
        seam.stage_gate(ca, t)
    for t in range(first_task(bid, NORM0), s, BLOCKS):
        seam.stage_norm(ca, t, lambda *v: store_normed(normed, HIDDEN, *v))
        publish(cb["put"], cb["xrdy"], t, 1, tid == 0)
    gpu.barrier()

    # ---- the MoE, its all-reduce into ``out``
    if const_expr(key.timeline):
        # the seam's end, then the MoE's stages' (stamped in LDS), after the
        # attention back's points
        tl = a["tl"]
        tls = lds.moe.peek().tls.ptr
        if tid == 0:
            fx.ptr_store(memrealtime(), tls + 0)
        moe.run_moe(cb, mkey, bid, tls, 0)
        if tid < moe.TL_POINTS:
            t64 = fx.Int64(fx.ptr_load(tls + tid))
            w = fx.Vector.from_elements([fx.Int32(t64), fx.Int32(t64 >> 32)], fx.Int32)
            bo.buffer_store(
                w, rsrc(tl), (bid * K2_POINTS + mb_back.BACK_POINTS + tid) * 2
            )
    else:
        moe.run_moe(cb, mkey, bid, 0, 0)

    def reset():
        # the MoE counter slot 128 launch pairs ahead: its last use long
        # done, its next use far off
        slot = (ep + 128) & 255
        for region in COUNTERS:
            gstore(cb[region].value + fx.Int64(slot * 4), fx.Int32(0), words=1)

    mb_back.epoch_end({"bid": bid, "tid": tid}, a["epoch"], ep, reset)


def build_mono_k2(key: MonoBuild):
    s, tp, d = key.tokens, key.tp, _dims(key)
    assert 1 <= s <= MAX_TOKENS
    layout = scratch_layout(s, tp, key.topk)
    bkey = _back_key(key)
    mkey = _moe_key(key)
    rs = route_shape(mkey)
    assert rs.experts // 64 in SORT_NETS
    BackM = mb_back.back_lds_members(s, d)
    SplitLds, GemvLds = BackM["split"], BackM["gemv"]
    MoeLds = moe.moe_smem(s, rs, key.timeline)
    half = peer_half_bytes(tp)
    assert seam.SLICES <= BLOCKS
    SeamLds = _seam_lds(s)

    @fx.union
    class K2Lds:
        split: SplitLds  # type: ignore[valid-type]
        gemv: GemvLds  # type: ignore[valid-type]
        seam: SeamLds  # type: ignore[valid-type]
        moe: MoeLds  # type: ignore[valid-type]

    name = kernel_symbol(
        "dsv4_mono_k2",
        s=s,
        tp=tp,
        r=key.ratio,
        tl=key.timeline,
        **_variant(key),
    )
    keyed = key_tuple(key, SOURCES)

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def k2(
        q: Int64,
        swa: Int64,
        swa_stride: Int64,
        swa_block: Int32,
        comp: Int64,
        comp_stride: Int64,
        comp_block: Int32,
        kt: Int64,
        klen: Int64,
        slot: Int64,
        swa_idx: Int64,
        swa_lens: Int64,
        t2r: Int64,
        topk: Int64,
        comp_bt: Int64,
        bt_stride: Int32,
        pos: Int64,
        sink: Int64,
        qk_scale: Int32,
        cos_sin: Int64,
        woa: Int64,
        woa_s: Int64,
        wob: Int64,
        wob_s: Int64,
        zrec: Int64,
        res_in: Int64,
        post_in: Int64,
        comb_in: Int64,
        pre_in: Int64,  # unread, as K1's
        hc_fn: Int64,
        hc_scale: Int64,
        hc_base: Int64,
        ffn_w: Int64,
        res_out: Int64,
        post_out: Int64,
        comb_out: Int64,
        pre_out: Int64,
        gate_w: Int64,
        bias: Int64,
        w13: Int64,
        w13_s: Int64,
        w2: Int64,
        w2_s: Int64,
        sgu: Int64,
        sgu_s: Int64,
        sw2: Int64,
        sw2_s: Int64,
        out: Int64,
        scratch: Int64,
        sym: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        tl: Int64,
    ):
        _ = keyed
        bid = fx.block_idx.x
        lds = fx.SharedAllocator().allocate(K2Lds)
        ep = _epoch(epoch)
        tag = (ep << 1) + 2
        par = fx.Int64(ep & 1) * fx.Int64(half)
        bases = [b + par for b in peer_bases(peers, tp)]
        own = sym + par
        bargs = {
            "q": q,
            "swa": swa,
            "swa_stride": swa_stride,
            "swa_block": swa_block,
            "comp": comp,
            "comp_stride": comp_stride,
            "comp_block": comp_block,
            "kt": kt,
            "klen": klen,
            "pos": pos,
            "sink": sink,
            "qk_scale": qk_scale,
            "cos_sin": cos_sin,
            "woa": woa,
            "woa_s": woa_s,
            "wob": wob,
            "wob_s": wob_s,
            "out": 0,
            "out_stride": 0,
            "zrec": zrec,
        }
        blayout = {n: layout[n] for n in back_scratch(s, d, start=0)}
        views = {"split": lds.split.peek(), "gemv": lds.gemv.peek()}
        c = mb_back.back_context(bkey, views, bargs, scratch, tag, blayout)
        # a key table wider than a token's top-k: chunks past a token's keys
        # skipped from its position
        c["bounded"] = d.topk_width > INDEX_TOPK
        amb = Mailbox(tag)

        def push(t, col, v0, v1):
            # this rank's bf16 partial pair to the ATTN region of every rank,
            # or of the column's owner (``allreduce``)
            allreduce.push(
                lambda dst, pair, vals: amb.put_bf(dst, 2 * pair, vals),
                lambda p: bases[p],
                0,
                "attn",
                rank,
                t,
                s,
                col,
                [v0, v1],
                d,
                key.two_hop,
            )

        c["wob_out"] = push
        if const_expr(key.timeline):
            c["tl"], c["tl_points"] = tl, K2_POINTS
        mb_back.epoch_begin(c, epoch, ep)
        # the key table, from the top-k vLLM's indexer selected after K1,
        # on the last CTAs: each split task waits on its token's
        ckt = {
            "S": s,
            "d": d,
            "tid": c["tid"],
            "dev": True,
            "ratio": key.ratio,
            "global_keys": key.global_keys,
            "args": {
                "slot": slot,
                "swa_idx": swa_idx,
                "swa_lens": swa_lens,
                "pos": pos,
                "t2r": t2r,
                "topk": topk,
                "comp_block": comp_block,
                "comp_bt": comp_bt,
                "bt_stride": bt_stride,
                "kt": kt,
                "klen": klen,
            },
        }
        c["krdy"] = sreg(scratch, layout["krdy"][0], "krdy")
        for t in range(first_task(bid, BLOCKS - s), s, BLOCKS):
            mb_front.stage_kt(ckt, t)
            mb_back.publish(c, c["krdy"], t)
        mb_back.run_back(c, bkey, bid)
        gpu.barrier()

        ffn = dict(
            res_in=res_in,
            post_in=post_in,
            comb_in=comb_in,
            pre_in=pre_in,
            hc_fn=hc_fn,
            hc_scale=hc_scale,
            hc_base=hc_base,
            ffn_w=ffn_w,
            res_out=res_out,
            post_out=post_out,
            comb_out=comb_out,
            pre_out=pre_out,
            gate_w=gate_w,
            bias=bias,
            w13=w13,
            w13_s=w13_s,
            w2=w2,
            w2_s=w2_s,
            sgu=sgu,
            sgu_s=sgu_s,
            sw2=sw2,
            sw2_s=sw2_s,
            out=out,
            scratch=scratch,
            epoch=epoch,
            tl=tl,
        )
        run_ffn(key, lds, ffn, amb, bases, own, rank, ep)

    @flyc.jit
    def launch(
        q: Int64,
        swa: Int64,
        swa_stride: Int64,
        swa_block: Int32,
        comp: Int64,
        comp_stride: Int64,
        comp_block: Int32,
        kt: Int64,
        klen: Int64,
        slot: Int64,
        swa_idx: Int64,
        swa_lens: Int64,
        t2r: Int64,
        topk: Int64,
        comp_bt: Int64,
        bt_stride: Int32,
        pos: Int64,
        sink: Int64,
        qk_scale: Int32,
        cos_sin: Int64,
        woa: Int64,
        woa_s: Int64,
        wob: Int64,
        wob_s: Int64,
        zrec: Int64,
        res_in: Int64,
        post_in: Int64,
        comb_in: Int64,
        pre_in: Int64,
        hc_fn: Int64,
        hc_scale: Int64,
        hc_base: Int64,
        ffn_w: Int64,
        res_out: Int64,
        post_out: Int64,
        comb_out: Int64,
        pre_out: Int64,
        gate_w: Int64,
        bias: Int64,
        w13: Int64,
        w13_s: Int64,
        w2: Int64,
        w2_s: Int64,
        sgu: Int64,
        sgu_s: Int64,
        sw2: Int64,
        sw2_s: Int64,
        out: Int64,
        scratch: Int64,
        sym: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        tl: Int64,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        k2(
            q,
            swa,
            swa_stride,
            swa_block,
            comp,
            comp_stride,
            comp_block,
            kt,
            klen,
            slot,
            swa_idx,
            swa_lens,
            t2r,
            topk,
            comp_bt,
            bt_stride,
            pos,
            sink,
            qk_scale,
            cos_sin,
            woa,
            woa_s,
            wob,
            wob_s,
            zrec,
            res_in,
            post_in,
            comb_in,
            pre_in,
            hc_fn,
            hc_scale,
            hc_base,
            ffn_w,
            res_out,
            post_out,
            comb_out,
            pre_out,
            gate_w,
            bias,
            w13,
            w13_s,
            w2,
            w2_s,
            sgu,
            sgu_s,
            sw2,
            sw2_s,
            out,
            scratch,
            sym,
            peers,
            rank,
            epoch,
            tl,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    return launch
