# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The wide decode's launches: one persistent kernel per GDN layer, and two per
QSA layer around vLLM's attention (``LayerBuild.kind``):

    gdn        [stats] attn mix -> GDN proj + conv -> recurrence + gated norm
               -> out_proj -> TP exchange -> attn combine
               -> mlp mix -> router -> route + quantize -> experts
               -> TP exchange -> mlp combine
    qsa_front  [stats] attn mix -> x -> qkv_proj + index_qk_proj
               (vLLM's QSA attention follows)
    qsa_back   o_proj (gated) -> TP exchange -> attn combine -> the MLP half

Phases are separated by grid barriers; an exchange's readers wait on its line
flags instead. Up to ``tiled.RESIDENT_MAX`` rows a mix is one phase and so is
GDN proj -> out_proj -> attn combine: their hand-offs are tagged pairs
(``stages.tagged``), and each GEMM's first weights are issued inside the
barrier before it. A launch reads the multi-stream state, updates it in place
(``state``, [rows, HC * HIDDEN] bf16, materialized: the next launch, a PLE
layer and the final mixer read it as is) and leaves ``rrms`` of every (row,
stream) in its scratch for the next launch's mix; ``stats`` computes them
first when the state comes from vLLM (layer 0, after a PLE layer).
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import const_expr
from flydsl.expr.typing import Int32, Int64

from vllm.models.qwen4_exp.amd.specialized_decode.wide.common.grid import (
    Grid,
    TraceInvariant,
    grid_sync,
    prologue,
)
from vllm.models.qwen4_exp.amd.specialized_decode.wide.common.ops import (
    ld,
    ld_ptr64,
    memrealtime,
    st,
)
from vllm.models.qwen4_exp.amd.specialized_decode.wide.common.plan import (
    KernelAbi,
    key_tuple,
)
from vllm.models.qwen4_exp.amd.specialized_decode.wide.layout import (
    BLOCKS,
    HC,
    HIDDEN,
    LOWRANK,
    MAX_TOKENS,
    QSA_CORE,
    SLOTS,
    THREADS,
    VALUE_DIM,
)
from vllm.models.qwen4_exp.amd.specialized_decode.wide.stages import exchange as ex_
from vllm.models.qwen4_exp.amd.specialized_decode.wide.stages import (
    gdn,
    hc,
    moe,
    tagged,
    tiled,
)
from vllm.models.qwen4_exp.amd.specialized_decode.wide.stages import qsa as qsa_

KINDS = ("gdn", "qsa_front", "qsa_back")
ATTN, MLP = 0, 1  # a layer's two exchanges


@dataclass(frozen=True)
class LayerBuild:
    kind: str
    tokens: int
    tp: int
    stats: bool = False  # the input state came from vLLM: compute its rrms first
    qkvz_shuffled: bool = False  # in_proj_qkvz on AITER's ASM FP4 GEMM layout
    out_shuffled: bool = False  # out_proj likewise
    rip: int = 384  # the experts' padded intermediate rows a rank
    hc_eps: float = 1e-6
    gdn_eps: float = 1e-6
    trace: bool = False  # CTA 0 stamps the clock at every phase end (``phase_names``)
    allreduce: str = "bf16"  # the TP exchange's line codec (``exchange.CODECS``)

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"layer kind {self.kind!r}")
        if self.allreduce not in ex_.CODECS:
            raise ValueError(f"all-reduce codec {self.allreduce!r}")
        if not 1 <= self.tokens <= MAX_TOKENS:
            raise ValueError(f"{self.tokens} rows: 1 to {MAX_TOKENS}")


ABI = KernelAbi(
    (
        "state",
        "x",
        "scratch",
        "a_norm",
        "a_di",
        "a_up",
        "m_norm",
        "m_di",
        "m_up",
        "w_qkvz",
        "w_qkvz_s",
        "w_ba",
        "w_conv",
        "a_log",
        "dt_bias",
        "gdn_norm",
        "w_out",
        "w_out_s",
        "conv_state",
        "conv_stride",
        "ssm",
        "ssm_stride",
        "st_idx",
        "attn",
        "gate",
        "w_gate",
        "w13",
        "w13s",
        "w2",
        "w2s",
        "peers",
        "rank",
        "epoch",
        "layer",
        "sync",
        "sync_next",
        "folds",
        "health",
        "alarm",
        "trace",
        "w_qkv",
        "w_ix",
        "qkv_out",
        "ix_out",
    )
)
TRACE_WORDS = 32  # i64 stamps a layer in the trace buffer
TRACE_BACK = 8  # where a QSA layer's second launch stamps in its layer's row


def phase_names(key) -> list[str]:
    """The stamps of a traced build (``LayerBuild.trace``), in order: each
    marks the end of the named phase ("start": the kernel's start)."""
    parts = ["mix_down", "mix_reduce", "mix_up"]
    if tiled.resident(key.tokens):
        parts = ["mix_down+up"]  # one phase (hc.mix_resident)
    mix = lambda t: [f"{t}.{p}" for p in parts]  # noqa: E731
    mlp = mix("mlp") + moe.phase_names(key.tokens)
    out = ["start"] + (["stats"] if key.stats else [])
    if key.kind != "qsa_back":
        out += mix("attn")
    if key.kind == "qsa_front":
        out += ["qsa_proj"]
    if key.kind == "gdn":
        gdn_ = ["gdn_proj+conv", "gdn_recur", "out+allreduce+combine"]
        if tiled.resident(key.tokens):
            gdn_ = ["gdn+out+allreduce+combine"]  # one phase (tagged hand-offs)
        out += gdn_ + mlp
    if key.kind == "qsa_back":
        out += ["out+allreduce+combine"] + mlp
    return out


# --------------------------------------------------------------------------
# scratch: a width's intermediate buffers at static offsets
# --------------------------------------------------------------------------


def _part_words(rows: int) -> int:
    plans = [
        hc.down_plan(rows),
        gdn.proj_plan(),
        gdn.out_plan(),
        gdn.out_plan(QSA_CORE),
        moe.router_plan(),
    ]
    return max(p.partial_words(rows) for p in plans)


def _resident_chunks() -> int:
    """K chunks of the longest GEMM unit, which a resident stage holds."""
    plans = [
        hc.down_plan(1),
        gdn.proj_plan(),
        gdn.ba_plan(),
        gdn.out_plan(),
        gdn.out_plan(QSA_CORE),
        moe.router_plan(),
        qsa_.qkv_plan(),
        qsa_.ix_plan(),
    ]
    return max([p.per for p in plans] + [hc.UP_CHUNKS])


def scratch_layout(rows: int) -> dict[str, tuple[int, int]]:
    """Name -> (byte offset, bytes) of a width's scratch, 256-byte aligned.
    ``split`` (the MoE's split-unit marks) comes first and is sized for the
    widest step, so every width's marks share one zeroed region that nothing
    else writes."""
    sizes = {
        "split": MAX_TOKENS * SLOTS * 4,
        "rrms": rows * HC * 4,
        "inj_a": rows * HC * 4,
        "inj_m": rows * HC * 4,
        "down": rows * LOWRANK * 2,
        "qkv": gdn.handoff_words(rows)[0] * 4,
        "z": rows * VALUE_DIM * 2,
        "ba": gdn.handoff_words(rows)[1] * 4,
        "core": gdn.handoff_words(rows)[2] * 4,
        "part": _part_words(rows) * 4,
        "route": 2 * rows * SLOTS * 4,
        "x4": rows * HIDDEN // 8 * 4,
        "x4s": rows * HIDDEN // 32 * 4,
        "mpart": rows * SLOTS * HIDDEN * 4,
        "mpart2": rows * SLOTS * HIDDEN * 4,
        "mixp": hc.mix_words(rows)[0] * 4,
        "mixd": hc.mix_words(rows)[1] * 4,
    }
    out, off = {}, 0
    for name, n in sizes.items():
        out[name] = (off, n)
        off = (off + n + 255) // 256 * 256
    return out


def scratch_bytes(rows: int) -> int:
    return max(o + n for o, n in scratch_layout(rows).values())


class Scratch(TraceInvariant):
    """A width's scratch buffers (``scratch_layout``) as byte addresses."""

    rrms: fx.Int64
    inj_a: fx.Int64
    inj_m: fx.Int64
    down: fx.Int64
    qkv: fx.Int64
    z: fx.Int64
    ba: fx.Int64
    core: fx.Int64
    part: fx.Int64
    route: fx.Int64
    x4: fx.Int64
    x4s: fx.Int64
    mpart: fx.Int64
    mpart2: fx.Int64
    split: fx.Int64
    mixp: fx.Int64
    mixd: fx.Int64

    def __init__(self, base, rows: int):
        for name, (off, _) in scratch_layout(rows).items():
            setattr(self, name, fx.Int64(base) + fx.Int64(off))


# --------------------------------------------------------------------------
# the kernels
# --------------------------------------------------------------------------

_BUILDS: dict = {}
_STREAM = fx.Stream(None)


def build(key: LayerBuild):
    if key in _BUILDS:
        return _BUILDS[key]
    S, tp = key.tokens, key.tp
    shape = moe.Shape(S, key.rip)
    MoeLds = moe.lds_struct(shape)
    aux_words = max(hc.up_lds_words(S), ex_.GroupTile.words(S), gdn.recur_lds_words())
    chunks = _resident_chunks()

    @fx.struct
    class GemmLds:
        stage: fx.Array[fx.Int32, tiled.stage_words(S, chunks), 16]
        aux: fx.Array[fx.Int32, aux_words, 16]

    @fx.union
    class PhaseLds:
        gemm: GemmLds
        moe: MoeLds  # type: ignore[valid-type]

    @fx.struct
    class Smem:
        word: fx.Array[fx.Int32, 4, 16]

    if key.kind == "qsa_front":
        part_bytes = scratch_layout(S)["part"][1]
        assert qsa_.partial_words(S) * 4 <= part_bytes, "qsa_proj partials"
    keyed = key_tuple(key)
    name = f"q4w_{key.kind}_s{S}_tp{tp}" + ("_st" if key.stats else "")
    name += "" if key.allreduce == "bf16" else f"_{key.allreduce}"

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kernel(
        state: Int64,
        x: Int64,
        scratch: Int64,
        a_norm: Int64,
        a_di: Int64,
        a_up: Int64,
        m_norm: Int64,
        m_di: Int64,
        m_up: Int64,
        w_qkvz: Int64,
        w_qkvz_s: Int64,
        w_ba: Int64,
        w_conv: Int64,
        a_log: Int64,
        dt_bias: Int64,
        gdn_norm: Int64,
        w_out: Int64,
        w_out_s: Int64,
        conv_state: Int64,
        conv_stride: Int64,
        ssm: Int64,
        ssm_stride: Int64,
        st_idx: Int64,
        attn: Int64,
        gate: Int64,
        w_gate: Int64,
        w13: Int64,
        w13s: Int64,
        w2: Int64,
        w2s: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        layer: Int32,
        sync: Int64,
        sync_next: Int64,
        folds: Int64,
        health: Int64,
        alarm: Int64,
        trace: Int64,
        w_qkv: Int64,
        w_ix: Int64,
        qkv_out: Int64,
        ix_out: Int64,
    ):
        _ = keyed
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        alloc = fx.SharedAllocator()
        sm = alloc.allocate(Smem).peek()
        ph = alloc.allocate(PhaseLds)
        gl, ml = ph.gemm.peek(), ph.moe.peek()
        g = Grid(tid, bid, sync, sync_next, folds, health, alarm, sm.word.ptr)
        prologue(g)
        scr = Scratch(scratch, S)
        aux_f32 = fx.recast_iter(fx.Float32, gl.aux.ptr)
        gemm = tiled.Gemm(gl.stage.ptr, S, tid, chunks=chunks)
        up_gemm = tiled.Gemm(gl.stage.ptr, S, tid, k_total=LOWRANK, chunks=chunks)
        bases = [ld_ptr64(peers, r) for r in range(tp)]
        own = bases[0]
        for r in fx.range_constexpr(1, tp):
            own = (rank == r).select(bases[r], own)
        ep = ld(epoch, 0, fx.Int32)
        names: list[str] = []
        first = TRACE_BACK if key.kind == "qsa_back" else 0

        def stamp(name):
            if const_expr(key.trace):
                if (bid == 0) & (tid == 0):
                    st(trace, layer * TRACE_WORDS + first + len(names), memrealtime())
                names.append(name)

        def end_phase(name, issue=None):
            """The phase's grid barrier; at resident widths ``issue()`` (the
            next phase's first weights, or the small MoE path's
            ``stage_rows``) runs inside it, its result returned."""
            out = grid_sync(g, issue if tiled.resident(S) else None)
            stamp(name)
            return out

        stamp("start")

        def exchange(kind: int):
            slot = (layer % 2) * 2 + kind
            tag = ep * 128 + layer * 2 + kind + 1
            return ex_.Exchange(bases, own, rank, tp, slot, tag, S, key.allreduce)

        # Past ``tiled.RESIDENT_MAX`` and up to ``tiled.PRELOAD_ROWS``, a GEMM
        # phase's static weights are issued in an earlier phase, where the grid
        # waits (``tiled.unit_preload``), and reach the GEMM as its
        # ``pending``: mix_up's while mix_down runs (they land in its GEMM, not
        # at the reduce's barrier), the GDN input projection's and the router's
        # while mix_reduce runs, out_proj's while the recurrence runs, the MLP
        # mix's W_di while the attention combine waits for the peer. Above it,
        # mix_up's loads are issued as mix_reduce starts (``hc.UpLoads``).
        early = tiled.RESIDENT_MAX < S <= tiled.PRELOAD_ROWS

        def ahead(weights, plan):
            return tiled.unit_preload(gemm, weights, plan, g.bid) if early else None

        def mix(tag, kind, norm, di, up, inj, pending=None, then=None, nxt=None):
            """``pending``: the down GEMM's first weights in flight, at the
            resident widths ``hc.mix_issue``'s, else ``ahead``'s. The next
            GEMM's are issued by ``then()`` inside the last barrier at the
            resident widths, by ``ahead(*nxt)`` as mix_reduce starts at the
            others, and returned."""
            if const_expr(tiled.resident(S)):
                hc.mix_resident(
                    g,
                    gemm,
                    up_gemm,
                    aux_f32,
                    state,
                    scr.rrms,
                    norm,
                    di,
                    up,
                    x,
                    inj,
                    (scr.mixp, scr.mixd),
                    ep * 128 + layer * 2 + kind + 1,
                    S,
                    pending,
                )
                return end_phase(f"{tag}.mix_down+up", then)
            if const_expr(early):
                pre = hc.UpLoads(g, up_gemm, up, state, scr.rrms, norm, S)
            hc.mix_down(g, gemm, state, scr.rrms, norm, di, scr.part, pending)
            end_phase(f"{tag}.mix_down")
            pre_next = None if nxt is None else ahead(*nxt)
            if const_expr(not early):
                pre = hc.UpLoads(g, up_gemm, up, state, scr.rrms, norm, S)
            hc.mix_reduce(g, scr.part, scr.down, inj, S, True)
            end_phase(f"{tag}.mix_reduce")
            hc.mix_up(g, up_gemm, aux_f32, scr.down, up, x, S, pre)
            end_phase(f"{tag}.mix_up")
            return pre_next

        def back(weights, source, k, pending=None, name="out+allreduce+combine"):
            xch = exchange(ATTN)
            tile = ex_.GroupTile(gl.aux.ptr, S)
            gdn.out(g, gemm, weights, source, scr.part, tile, xch, k, 0, pending)
            pre_down = ahead(hc.down_weights(m_di), hc.down_plan(S))
            hc.combine(g, xch, state, scr.inj_a, scr.rrms, S, key.hc_eps)
            issued = end_phase(name, lambda: hc.mix_issue(g, m_di, m_up, S))
            return issued if tiled.resident(S) else pre_down

        def mlp_half(pending):
            mw = moe.Weights(w_gate, w13, w13s, w2, w2s)
            pre = moe.prefetch(g, shape, mw)  # in flight through the mix
            router_w = (moe.router_weights(w_gate), moe.router_plan())
            routed = mix(
                "mlp", MLP, m_norm, m_di, m_up, scr.inj_m, pending, nxt=router_w
            )
            xch = exchange(MLP)
            moe.mlp(
                g,
                shape,
                gemm,
                ml,
                x,
                mw,
                moe.Bufs(
                    scr.part,
                    scr.route,
                    scr.x4,
                    scr.x4s,
                    scr.mpart,
                    scr.mpart2,
                    scr.split,
                ),
                xch,
                end_phase,
                pre if moe.small(S) else routed,
            )
            hc.combine(g, xch, state, scr.inj_m, scr.rrms, S, key.hc_eps)
            if const_expr(key.trace):
                end_phase(moe.phase_names(S)[-1])

        down_w = None
        if const_expr(key.stats):
            pre_attn = ahead(hc.down_weights(a_di), hc.down_plan(S))
            hc.stats(g, state, scr.rrms, S, key.hc_eps)
            issued = end_phase("stats", lambda: hc.mix_issue(g, a_di, a_up, S))
            down_w = issued if tiled.resident(S) else pre_attn
        if const_expr(key.kind == "qsa_front"):
            mix("attn", ATTN, a_norm, a_di, a_up, scr.inj_a, down_w)
            qsa_.proj(g, gemm, w_qkv, w_ix, x, qkv_out, ix_out, scr.part, S)
            if const_expr(key.trace):
                end_phase("qsa_proj")
        if const_expr(key.kind == "gdn"):
            w = gdn.Bufs(
                qkvz=w_qkvz,
                qkvz_s=w_qkvz_s,
                shuffled=key.qkvz_shuffled,
                ba=w_ba,
                conv=w_conv,
                a_log=a_log,
                dt_bias=dt_bias,
                norm=gdn_norm,
            )
            b = gdn.Bufs(
                x=x,
                qkv=scr.qkv,
                z=scr.z,
                ba=scr.ba,
                core=scr.core,
                part=scr.part,
                st_idx=st_idx,
                conv_state=conv_state,
                conv_stride=conv_stride,
                ssm=ssm,
                ssm_stride=ssm_stride,
            )
            proj_w = mix(
                "attn",
                ATTN,
                a_norm,
                a_di,
                a_up,
                scr.inj_a,
                down_w,
                lambda: gdn.proj_issue(g, w),
                (gdn.proj_weights(w), gdn.proj_plan()),
            )
            out_w = tiled.Fp4Weights(
                w_out, w_out_s, HIDDEN, VALUE_DIM, key.out_shuffled
            )
            if const_expr(tiled.resident(S)):
                # one phase: the recurrence polls the projection's tagged
                # outputs, out_proj the recurrence's (its weights issued
                # before the recurrence, to land under it)
                gtag = ep * 128 + layer * 2 + 1
                gdn.proj(g, gemm, w, b, S, 0, proj_w, gtag)
                pending = tiled.issue_unit(bid, tid, out_w, gdn.out_plan())
                gdn.recur(g, w, b, aux_f32, S, key.gdn_eps, None, gtag)
                source = tagged.Pairs(g, scr.core, VALUE_DIM, gtag)
                name = "gdn+out+allreduce+combine"
            else:
                gdn.proj(g, gemm, w, b, S, 0, proj_w)
                end_phase("gdn_proj+conv")
                pending = ahead(out_w, gdn.out_plan())  # lands under the recurrence
                gdn.recur(g, w, b, aux_f32, S, key.gdn_eps)
                end_phase("gdn_recur")
                source = tiled.Source(scr.core, VALUE_DIM)  # type: ignore[assignment]
                name = "out+allreduce+combine"
            mlp_half(back(out_w, source, VALUE_DIM, pending, name))
        if const_expr(key.kind == "qsa_back"):
            pending = back(
                tiled.Bf16Weights(w_out, HIDDEN, QSA_CORE),
                gdn.GatedCore(attn, gate, QSA_CORE),
                QSA_CORE,
            )
            mlp_half(pending)
        if const_expr(key.trace):
            assert names == phase_names(key), (names, phase_names(key))

    @flyc.jit
    def launch(
        state: Int64,
        x: Int64,
        scratch: Int64,
        a_norm: Int64,
        a_di: Int64,
        a_up: Int64,
        m_norm: Int64,
        m_di: Int64,
        m_up: Int64,
        w_qkvz: Int64,
        w_qkvz_s: Int64,
        w_ba: Int64,
        w_conv: Int64,
        a_log: Int64,
        dt_bias: Int64,
        gdn_norm: Int64,
        w_out: Int64,
        w_out_s: Int64,
        conv_state: Int64,
        conv_stride: Int64,
        ssm: Int64,
        ssm_stride: Int64,
        st_idx: Int64,
        attn: Int64,
        gate: Int64,
        w_gate: Int64,
        w13: Int64,
        w13s: Int64,
        w2: Int64,
        w2s: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        layer: Int32,
        sync: Int64,
        sync_next: Int64,
        folds: Int64,
        health: Int64,
        alarm: Int64,
        trace: Int64,
        w_qkv: Int64,
        w_ix: Int64,
        qkv_out: Int64,
        ix_out: Int64,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        kernel(
            state,
            x,
            scratch,
            a_norm,
            a_di,
            a_up,
            m_norm,
            m_di,
            m_up,
            w_qkvz,
            w_qkvz_s,
            w_ba,
            w_conv,
            a_log,
            dt_bias,
            gdn_norm,
            w_out,
            w_out_s,
            conv_state,
            conv_stride,
            ssm,
            ssm_stride,
            st_idx,
            attn,
            gate,
            w_gate,
            w13,
            w13s,
            w2,
            w2s,
            peers,
            rank,
            epoch,
            layer,
            sync,
            sync_next,
            folds,
            health,
            alarm,
            trace,
            w_qkv,
            w_ix,
            qkv_out,
            ix_out,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    ABI.check(kernel, launch)
    _BUILDS[key] = launch
    return launch


def run(key: LayerBuild, args: dict) -> None:
    """Launch ``key``'s kernel on the current stream; ``args``: every ABI name
    (tensors by data pointer, ints as is)."""
    from vllm.v1.attention.ops.flydsl_kernels import _run_compiled

    vals = [
        v.data_ptr() if isinstance(v, torch.Tensor) else int(v) for v in ABI.pack(args)
    ]
    _run_compiled(build(key), *vals, torch.cuda.current_stream())


def compile_only(key: LayerBuild) -> None:
    """Compile ``key``'s kernel without launching it (FlyDSL's compile-only
    mode returns before the launch)."""
    import os

    old = os.environ.get("COMPILE_ONLY")
    os.environ["COMPILE_ONLY"] = "1"
    try:
        f = build(key)
        f(*ABI.zeros(), stream=torch.cuda.current_stream())
    finally:
        if old is None:
            del os.environ["COMPILE_ONLY"]
        else:
            os.environ["COMPILE_ONLY"] = old
