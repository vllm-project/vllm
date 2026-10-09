# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The mono decode layer's two launches (README): K1 = the attention seam
(``stages.seam``: slice, gate, the norm into vLLM's MXFP8) + the attention front
(``attention.front``); K2 = the attention back (``attention.back``, wo_b pushed
to every rank) + the FFN seam (the TP sum folded in) + the MoE (``stages.moe``,
its all-reduce in-kernel too). A layer whose attention vLLM runs takes the FFN
launch instead: K2 with the attention back replaced by a push of vLLM's
unreduced wo_b output (``seam.stage_push``).

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
from .attention.plan import HIDDEN, Dims, back_scratch, front_scratch
from .common.mx import clamp_fp8
from .common.ops import butterfly, fp8_pack4, peer_bases, traced
from .common.plan import WAVES, first_task, key_tuple
from .common.sync import Mailbox, preg, publish, sreg
from .sources import SOURCES
from .stages import moe, seam
from .stages.dims import Dims as StageDims
from .stages.moe_shape import SORT_NETS, MoeBuild, route_shape

_STREAM = fx.Stream(None)
MAX_TOKENS = 48
X_WORDS = HIDDEN // 4
X_GROUPS = HIDDEN // 32
SLICES = seam.SLICES  # 160
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


@dataclass(frozen=True)
class MonoBuild:
    tokens: int
    tp: int
    ratio: int  # the layer's compress ratio (its attention's key sources)
    timeline: bool = False


def _moe_key(key: MonoBuild) -> MoeBuild:
    return MoeBuild(tokens=key.tokens, tp=key.tp)


def _front_key(key: MonoBuild):
    return mb_front.FrontBuild(key.tokens, key.tp, key.ratio, key.timeline)


def _back_key(key: MonoBuild):
    return mb_back.BackBuild(key.tokens, key.tp, key.ratio, key.timeline)


def scratch_layout(s: int, tp: int) -> dict:
    """Every region of step width s's launches -> (byte offset, bytes), disjoint:
    K1's front and seam, K2's back, the MoE's regions, the normed rows and their
    flags. Each width has a scratch of its own (``DSV41MonoLayer.scratch``):
    under another width's layout a mailbox pair's tag word could be plain data,
    which a poll may take for a current tag. The MoE's counters, whose slots
    outlive a step width, are in the epoch buffer."""
    d = Dims(tp)
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
    regions = moe.scratch_layout(_moe_key(MonoBuild(s, tp, 1))).items()
    mr = {n: v for n, v in regions if n not in COUNTERS}
    lo = min(o for o, _ in mr.values())
    place({n: (o - lo, n_) for n, (o, n_) in mr.items()})
    place({"normed": (0, s * HIDDEN * 2), "xrdy_moe": (s * HIDDEN * 2 + 256, s * 8)})
    return out


def scratch_bytes(s: int = MAX_TOKENS, tp: int = 2) -> int:
    return max(o + n for o, n in scratch_layout(s, tp).values())


def peer_half_bytes(tp: int) -> int:
    """One parity's peer regions: the attention's partials, then the MoE's."""
    return seam.attn_peer_bytes(MAX_TOKENS, tp) + moe.peer_bytes(MAX_TOKENS, tp)


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
                code, rsrc(c["x8s"]), t * X_GROUPS + col // 32, cache_modifier=CM_DEV
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
    s, tp = key.tokens, key.tp
    assert 1 <= s <= MAX_TOKENS
    layout = scratch_layout(s, tp)
    fkey = _front_key(key)
    FrontLds = mb_front.front_lds(s, Dims(tp))
    NORM0 = SLICES  # the norms on the CTAs past the slices: the wqkv waits on them
    GATE0 = SLICES + s
    assert GATE0 + s <= BLOCKS

    @fx.struct
    class SeamLds:
        red: fx.Array[fx.Float32, WAVES * 2 * 64 * 4, 16]
        rl: fx.Array[fx.Float32, s * seam.KT, 16]
        fl: fx.Array[fx.Float32, seam.MIX * seam.KT, 16]

    @fx.union
    class K1Lds:
        seam: SeamLds
        front: FrontLds  # type: ignore[valid-type]

    name = kernel_symbol("dsv41_mono_k1", s=s, tp=tp, r=key.ratio, tl=key.timeline)
    keyed = key_tuple(key, SOURCES)

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def k1(
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
            "x": 0,
            "x_stride": 0,
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
        }
        flayout = {n: layout[n] for n in front_scratch(s, Dims(tp))}
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
            "d": StageDims(tp),
        }
        for region in ("lin", "pmix"):
            ca[region] = sreg(scratch, layout[region][0], region)
        if const_expr(key.timeline):
            c["tl"], c["tl_points"] = tl, mb_front.FRONT_POINTS

        def seam_slice():
            for task in range(first_task(bid, 0), SLICES, BLOCKS):
                seam.stage_slice(ca, task)
            gpu.barrier()

        def seam_rest():
            for t in range(first_task(bid, NORM0), s, BLOCKS):
                seam.stage_norm(ca, t, lambda *v: store_x8(c, *v))
                publish_x8(c, t)
            for t in range(first_task(bid, GATE0), s, BLOCKS):
                seam.stage_gate(ca, t)
            gpu.barrier()

        mb_front.run_front(
            c, fkey, bid, x8_given=True, before_wqkv=seam_slice, after_wqkv=seam_rest
        )

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
            scratch,
            epoch,
            tl,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    return launch


# ---------------------------------------------------------------- K2


@traced
def store_normed(normed, t, col, ys, live):
    """The FFN seam's norm output -> the MoE input row (bf16, device scope:
    this launch reads it)."""
    if live:
        for j in range_constexpr(8):
            bo.buffer_store(
                ys[j].to(fx.BFloat16),
                rsrc(normed),
                t * HIDDEN + col + j,
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
def run_ffn(key: MonoBuild, lds, a: dict, amb, bases, own, rank, ep, part=None):
    """K2's FFN half on this CTA: the FFN seam (the attention's TP partials --
    ``part`` pushed to every rank's ATTN region first -- summed and folded into
    the residual), the MoE and its all-reduce into ``a["out"]``, then the
    launch pair's epoch moved on."""
    s, tp = key.tokens, key.tp
    layout = scratch_layout(s, tp)
    mkey = _moe_key(key)
    rs = route_shape(mkey)
    GATE0 = SLICES
    NORM0 = SLICES + s
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
        "d": StageDims(tp),
    }
    normed = scratch + fx.Int64(layout["normed"][0])
    for region in ("lin", "pmix"):
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
    cb["tag"] = ep & 255
    for i, region in enumerate(COUNTERS):
        cb[region] = sreg(a["epoch"], 4 * (EPOCH_COUNTERS + i * moe.TAGS), region)
    for task in range(first_task(bid, 0), SLICES, BLOCKS):
        if const_expr(part is not None):
            seam.stage_push(ca, task, part)
        seam.stage_reduce(ca, task)
        seam.stage_slice(ca, task)
    for t in range(first_task(bid, GATE0), s, BLOCKS):
        seam.stage_gate(ca, t)
    for t in range(first_task(bid, NORM0), s, BLOCKS):
        seam.stage_norm(ca, t, lambda *v: store_normed(normed, *v))
        publish(cb["put"], cb["xrdy"], t, 1, tid == 0)
    gpu.barrier()

    # ---- the MoE, its all-reduce into ``out``
    moe.run_moe(cb, mkey, bid, 0, 0)

    def reset():
        # the MoE counter slot 128 launch pairs ahead: its last use long
        # done, its next use far off
        slot = (ep + 128) & 255
        for region in COUNTERS:
            gstore(cb[region].value + fx.Int64(slot * 4), fx.Int32(0), words=1)

    mb_back.epoch_end({"bid": bid, "tid": tid}, a["epoch"], ep, reset)


def build_mono_k2(key: MonoBuild):
    s, tp = key.tokens, key.tp
    assert 1 <= s <= MAX_TOKENS
    layout = scratch_layout(s, tp)
    bkey = _back_key(key)
    mkey = _moe_key(key)
    rs = route_shape(mkey)
    assert rs.experts // 64 in SORT_NETS
    BackM = mb_back.back_lds_members(s, Dims(tp))
    SplitLds, GemvLds = BackM["split"], BackM["gemv"]
    MoeLds = moe.moe_smem(s, rs, False)
    half = peer_half_bytes(tp)
    assert SLICES + 2 * s <= BLOCKS
    SeamLds = _seam_lds(s)

    @fx.union
    class K2Lds:
        split: SplitLds  # type: ignore[valid-type]
        gemv: GemvLds  # type: ignore[valid-type]
        seam: SeamLds  # type: ignore[valid-type]
        moe: MoeLds  # type: ignore[valid-type]

    name = kernel_symbol("dsv41_mono_k2", s=s, tp=tp, r=key.ratio, tl=key.timeline)
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
        blayout = {n: layout[n] for n in back_scratch(s, Dims(tp), start=0)}
        views = {"split": lds.split.peek(), "gemv": lds.gemv.peek()}
        c = mb_back.back_context(bkey, views, bargs, scratch, tag, blayout)
        amb = Mailbox(tag)

        def push(t, col, v0, v1):
            # this rank's bf16 partial pair to every rank's ATTN region
            for p in range_constexpr(tp):
                dst = preg(bases[p], 0, "attn")
                amb.put_bf(dst, 2 * seam.attn_region_pair(rank, t, s, col), [v0, v1])

        c["wob_out"] = push
        if const_expr(key.timeline):
            c["tl"], c["tl_points"] = tl, mb_back.BACK_POINTS
        mb_back.epoch_begin(c, epoch, ep)
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


# ---------------------------------------------------------------- the FFN launch


def build_mono_ffn(key: MonoBuild):
    """The FFN launch of a layer whose attention vLLM ran (``part``: this rank's
    unreduced wo_b output, bf16 [S, HIDDEN]): K2's FFN half, the partial pushed
    to every rank first. It takes a launch pair's epoch, as K2 does."""
    s, tp = key.tokens, key.tp
    assert 1 <= s <= MAX_TOKENS
    rs = route_shape(_moe_key(key))
    assert rs.experts // 64 in SORT_NETS
    MoeLds = moe.moe_smem(s, rs, False)
    half = peer_half_bytes(tp)
    assert SLICES + 2 * s <= BLOCKS
    SeamLds = _seam_lds(s)

    @fx.union
    class FfnLds:
        seam: SeamLds  # type: ignore[valid-type]
        moe: MoeLds  # type: ignore[valid-type]

    name = kernel_symbol("dsv41_mono_ffn", s=s, tp=tp)
    keyed = key_tuple(key, SOURCES)

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kffn(
        part: Int64,
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
    ):
        _ = keyed
        lds = fx.SharedAllocator().allocate(FfnLds)
        ep = _epoch(epoch)
        par = fx.Int64(ep & 1) * fx.Int64(half)
        bases = [b + par for b in peer_bases(peers, tp)]
        mb_back.epoch_begin({"bid": fx.block_idx.x, "tid": fx.thread_idx.x}, epoch, ep)
        a = dict(
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
        )
        amb = Mailbox((ep << 1) + 2)
        run_ffn(key, lds, a, amb, bases, sym + par, rank, ep, part=part)

    @flyc.jit
    def launch(
        part: Int64,
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
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        kffn(
            part,
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
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    return launch
