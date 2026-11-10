# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The wide decode's launches: one persistent kernel per GDN layer, one per
full-attention layer after vLLM's attention (``LayerBuild.kind``):

    gdn   [first: embedding -> input norm -> x]
          GDN proj + conv -> recurrence + gated norm -> out_proj
          -> TP exchange -> post-attention combine (x, x4)
          -> router -> route -> experts up -> experts down
          -> TP exchange -> next layer's input combine (or the final norm)
          [-> the next (attention) layer's qkv_proj]
    attn  o_proj (gated) -> TP exchange -> post-attention combine
          -> the MLP half as above

Phases are separated by grid barriers; an exchange's readers wait on its line
flags instead. Up to ``tiled.RESIDENT_MAX`` rows GDN proj -> recurrence ->
out_proj is one phase: their hand-offs are tagged pairs (``stages.tagged``).
A launch reads the residual and the normed rows ``x`` the previous one left in
its scratch and leaves the next ones there; the step's last leaves the final
norm's rows in ``x``.
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import const_expr
from flydsl.expr.typing import Int32, Int64

from vllm.models.qwen3_5.amd.specialized_decode.wide.common.grid import (
    Grid,
    TraceInvariant,
    grid_sync,
    prologue,
)
from vllm.models.qwen3_5.amd.specialized_decode.wide.common.ops import (
    ld,
    ld_ptr64,
    memrealtime,
    st,
)
from vllm.models.qwen3_5.amd.specialized_decode.wide.common.plan import (
    KernelAbi,
    key_tuple,
)
from vllm.models.qwen3_5.amd.specialized_decode.wide.layout import (
    ATTN_CORE,
    BLOCKS,
    HIDDEN,
    MAX_TOKENS,
    QKV,
    THREADS,
    VALUE_DIM,
    WAVES,
)
from vllm.models.qwen3_5.amd.specialized_decode.wide.stages import attn as attn_
from vllm.models.qwen3_5.amd.specialized_decode.wide.stages import exchange as ex_
from vllm.models.qwen3_5.amd.specialized_decode.wide.stages import (
    gdn,
    moe,
    norm,
    tagged,
    tiled,
)

KINDS = ("gdn", "attn")
ATTN, MLP = 0, 1  # a layer's two exchanges
FORMATS = ("bf16", "mxfp4", "mxfp4_h32")  # a GDN projection's weight format
# fold counter bases: the GDN projection's and out_proj's share a phase
FOLD_PROJ, FOLD_OUT, FOLD_PIECES = 0, 128, 256
TAG_LAYERS = 256  # tags a step: (layer, use) pairs below it


def q4_of(fmt: str):
    """The MXFP4 rows a GEMM of weight format ``fmt`` reads (``norm.Q4_MODES``)."""
    return {"bf16": None, "mxfp4": "plain", "mxfp4_h32": "h32"}[fmt]


@dataclass(frozen=True)
class LayerBuild:
    kind: str
    tokens: int
    tp: int
    first: bool = False  # the step's first layer: its input combine comes first
    qkvz: str = "bf16"  # in_proj_qkvz's format (``FORMATS``)
    qkvz_shuffled: bool = False  # MXFP4 in_proj_qkvz on AITER's ASM GEMM layout
    out: str = "bf16"  # out_proj's format
    out_shuffled: bool = False
    first_q4: str = "none"  # the first combine's MXFP4 rows ("none", "plain", "h32")
    next_q4: str = "none"  # the closing combine's, for the next layer's qkvz
    qkv_tail: bool = False  # the next layer is full attention: run its qkv_proj
    rip: int = 256  # the experts' padded intermediate rows a rank
    eps: float = 1e-6
    gdn_eps: float = 1e-6
    trace: bool = False  # CTA 0 stamps the clock at every phase end (``phase_names``)

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"layer kind {self.kind!r}")
        if not 1 <= self.tokens <= MAX_TOKENS:
            raise ValueError(f"{self.tokens} rows: 1 to {MAX_TOKENS}")
        if self.qkvz not in FORMATS or self.out not in FORMATS:
            raise ValueError(f"GDN projection formats {self.qkvz!r}, {self.out!r}")
        if self.first and self.kind != "gdn":
            raise ValueError("the step's first layer is a GDN layer")
        for q in (self.first_q4, self.next_q4):
            if q not in ("none", "plain", "h32"):
                raise ValueError(f"MXFP4 rows {q!r}")


def _q4(q: str):
    return None if q == "none" else q


ABI = KernelAbi(
    (
        "scratch",
        "emb",
        "n_in",
        "n_post",
        "n_next",
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
        "w_qkv",
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
    )
)
TRACE_WORDS = 32  # i64 stamps a layer in the trace buffer


def phase_names(key) -> list[str]:
    """The stamps of a traced build (``LayerBuild.trace``), in order: each
    marks the end of the named phase ("start": the kernel's start)."""
    out = ["start"] + (["first_combine"] if key.first else [])
    if key.kind == "gdn":
        if fused_gdn(key):
            out += ["gdn+out+allreduce+combine"]
        elif tiled.resident(key.tokens):
            out += ["gdn_proj+recur", "out+allreduce+combine"]
        else:
            out += ["gdn_proj+conv", "gdn_recur", "out+allreduce+combine"]
    else:
        out += ["o_proj+allreduce+combine"]
    out += moe.phase_names(key.tokens)[:-1] + ["experts_down+allreduce+combine"]
    if key.qkv_tail:
        out += ["qkv_proj"]
    return out


def fused_gdn(key) -> bool:
    """GDN proj -> recurrence -> out_proj as one phase (tagged hand-offs)."""
    return tiled.resident(key.tokens) and key.out == "bf16"


# --------------------------------------------------------------------------
# scratch: a width's buffers at static offsets
# --------------------------------------------------------------------------


def _part_words(rows: int) -> int:
    plans = [
        gdn.proj_plan(),
        gdn.out_plan(),
        gdn.out_plan(ATTN_CORE),
        moe.router_plan(),
        attn_.qkv_plan(),
    ]
    return max(p.partial_words(rows) for p in plans)


def _resident_chunks() -> int:
    """K chunks of the longest GEMM unit, which a resident stage holds."""
    plans = [
        gdn.proj_plan(),
        gdn.ba_plan(),
        gdn.out_plan(),
        gdn.out_plan(ATTN_CORE),
        moe.router_plan(),
        attn_.qkv_plan(),
    ]
    return max(p.max_per for p in plans)


def scratch_layout(rows: int, rip: int = 256) -> dict[str, tuple[int, int]]:
    """Name -> (byte offset, bytes) of a width's scratch, 256-byte aligned."""
    qkv, ba, core = gdn.handoff_words(rows)
    sizes = {
        "res": rows * HIDDEN * 2,
        "x": rows * HIDDEN * 2,
        "x4": rows * HIDDEN // 2,
        "x4s": rows * HIDDEN // 32 * 4,
        "qkv": qkv * 4,
        "z": rows * VALUE_DIM * 2,
        "ba": ba * 4,
        "core": core * 4,
        "core_s": rows * VALUE_DIM // 32 * 4,
        "part": _part_words(rows) * 4,
        "qkv_out": rows * QKV * 2,
    }
    for k, n in moe.scratch_words(moe.Shape(rows, rip)).items():
        sizes[k] = n * 4
    out, off = {}, 0
    for name, n in sizes.items():
        out[name] = (off, n)
        off = (off + n + 255) // 256 * 256
    return out


def scratch_bytes(rows: int, rip: int = 256) -> int:
    return max(o + n for o, n in scratch_layout(rows, rip).values())


class Scratch(TraceInvariant):
    """A width's scratch buffers (``scratch_layout``) as byte addresses."""

    def __init__(self, base, rows: int, rip: int):
        for name, (off, _) in scratch_layout(rows, rip).items():
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
    aux_words = max(
        ex_.GroupTile.words(S), gdn.recur_lds_words(), norm.LDS_WORDS
    )
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

    keyed = key_tuple(key)
    name = f"q38w_{key.kind}_s{S}_tp{tp}"
    name += "_first" if key.first else ""
    name += "" if key.qkvz == "bf16" else f"_qkvz_{key.qkvz}"
    name += "" if key.out == "bf16" else f"_out_{key.out}"
    name += "_qkv" if key.qkv_tail else ""

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kernel(
        scratch: Int64,
        emb: Int64,
        n_in: Int64,
        n_post: Int64,
        n_next: Int64,
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
        w_qkv: Int64,
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
        scr = Scratch(scratch, S, key.rip)
        aux_f32 = fx.recast_iter(fx.Float32, gl.aux.ptr)
        gemm = tiled.Gemm(gl.stage.ptr, S, tid, chunks=chunks)
        bases = [ld_ptr64(peers, r) for r in range(tp)]
        own = bases[0]
        for r in fx.range_constexpr(1, tp):
            own = (rank == r).select(bases[r], own)
        ep = ld(epoch, 0, fx.Int32)
        names: list[str] = []

        def stamp(name):
            if const_expr(key.trace):
                if (bid == 0) & (tid == 0):
                    st(trace, layer * TRACE_WORDS + len(names), memrealtime())
                names.append(name)

        def end_phase(name, issue=None):
            """The phase's grid barrier; at resident widths ``issue()`` (the
            next phase's first weights) runs inside it, its result returned."""
            out = grid_sync(g, issue if tiled.resident(S) else None)
            stamp(name)
            return out

        stamp("start")
        two_shot = not ex_.one_shot(S)
        tag0 = ep * TAG_LAYERS + layer * 2

        def exchange(kind: int, mode: str):
            return ex_.Exchange(
                bases, own, rank, tp, kind, mode, layer % 2, tag0 + kind + 1, S
            )

        def all_reduce(kind: int):
            """The exchange a producer pushes to, and the one its combine reads."""
            if const_expr(two_shot):
                return exchange(kind, "scatter"), exchange(kind, "gather")
            x = exchange(kind, "sum")
            return x, x

        def reduce(push):
            """The owner half of a two-shot exchange, after the producers'
            pushes (every wave of the grid)."""
            if const_expr(two_shot):
                ex_.reduce_lines(
                    g, push, all_reduce(push.kind)[1], g.spread, BLOCKS * WAVES
                )

        def combine(xch, weight, q4, first=False):
            out = norm.Out(scr.res, scr.x, scr.x4, scr.x4s, q4)
            norm.combine(
                g,
                None if first else xch,
                out,
                weight,
                aux_f32,
                S,
                key.eps,
                emb if first else None,
            )

        if const_expr(key.first):
            combine(None, n_in, _q4(key.first_q4), first=True)
            end_phase("first_combine")

        def mlp_half():
            mw = moe.Weights(w_gate, w13, w13s, w2, w2s)
            b = moe.Bufs(
                part=scr.part,
                logits=scr.logits,
                route=scr.route,
                x4=scr.x4,
                x4s=scr.x4s,
                act4=scr.act4,
                act4s=scr.act4s,
                upart=scr.upart,
                y=scr.y,
            )
            push, read = all_reduce(MLP)
            moe.mlp(
                g, shape, gemm, ml, scr.x, mw, b, push, end_phase,
                (FOLD_PROJ, FOLD_PIECES),
            )
            reduce(push)
            combine(read, n_next, _q4(key.next_q4))
            if const_expr(key.trace or key.qkv_tail):
                end_phase("experts_down+allreduce+combine")
            if const_expr(key.qkv_tail):
                attn_.qkv_proj(
                    g, gemm, w_qkv, scr.x, scr.qkv_out, scr.part, S, FOLD_PROJ
                )
                if const_expr(key.trace):
                    end_phase("qkv_proj")

        def back(weights, source, k, pending=None, name="out+allreduce+combine"):
            push, read = all_reduce(ATTN)
            tile = ex_.GroupTile(gl.aux.ptr, S)
            gdn.out(g, gemm, weights, source, scr.part, tile, push, k, FOLD_OUT, pending)
            reduce(push)
            combine(read, n_post, "plain")
            end_phase(name)

        if const_expr(key.kind == "gdn"):
            w = gdn.Bufs(
                qkvz=w_qkvz,
                qkvz_s=w_qkvz_s,
                qkvz_fmt="bf16" if key.qkvz == "bf16" else "mxfp4",
                shuffled=key.qkvz_shuffled,
                ba=w_ba,
                conv=w_conv,
                a_log=a_log,
                dt_bias=dt_bias,
                norm=gdn_norm,
            )
            b = gdn.Bufs(
                x=scr.x,
                x4=scr.x4,
                x4s=scr.x4s,
                qkv=scr.qkv,
                z=scr.z,
                ba=scr.ba,
                core=scr.core,
                core_s=scr.core_s,
                part=scr.part,
                st_idx=st_idx,
                conv_state=conv_state,
                conv_stride=conv_stride,
                ssm=ssm,
                ssm_stride=ssm_stride,
            )
            out_w = gdn.out_weights(w_out, w_out_s, key.out, key.out_shuffled)
            if const_expr(tiled.resident(S)):
                # the recurrence polls the projection's tagged outputs; with a
                # bf16 core out_proj polls the recurrence's too (its weights
                # issued before the recurrence, to land under it), an MXFP4
                # core is quantized a 32-group a thread and waits a barrier
                gtag = tag0 + 1
                gdn.proj(g, gemm, w, b, S, FOLD_PROJ, None, gtag)
                if const_expr(fused_gdn(key)):
                    pending = tiled.issue_unit(bid, tid, out_w, gdn.out_plan())
                    gdn.recur(g, w, b, aux_f32, S, key.gdn_eps, "bf16", None, gtag)
                    source = tagged.Pairs(g, scr.core, VALUE_DIM, gtag)
                    back(out_w, source, VALUE_DIM, pending, "gdn+out+allreduce+combine")
                else:
                    gdn.recur(g, w, b, aux_f32, S, key.gdn_eps, key.out, None, gtag)
                    pending = end_phase(
                        "gdn_proj+recur",
                        lambda: tiled.issue_unit(bid, tid, out_w, gdn.out_plan()),
                    )
                    source = tiled.Mx4Source(scr.core, scr.core_s, VALUE_DIM)
                    back(out_w, source, VALUE_DIM, pending)
            else:
                gdn.proj(g, gemm, w, b, S, FOLD_PROJ)
                end_phase("gdn_proj+conv")
                gdn.recur(g, w, b, aux_f32, S, key.gdn_eps, key.out)
                end_phase("gdn_recur")
                if const_expr(key.out == "bf16"):
                    source = tiled.Source(scr.core, VALUE_DIM)
                else:
                    source = tiled.Mx4Source(scr.core, scr.core_s, VALUE_DIM)
                back(out_w, source, VALUE_DIM)
            mlp_half()
        if const_expr(key.kind == "attn"):
            back(
                tiled.Bf16Weights(w_out, HIDDEN, ATTN_CORE),
                gdn.GatedCore(attn, gate, ATTN_CORE),
                ATTN_CORE,
                name="o_proj+allreduce+combine",
            )
            mlp_half()
        if const_expr(key.trace):
            assert names == phase_names(key), (names, phase_names(key))

    @flyc.jit
    def launch(
        scratch: Int64,
        emb: Int64,
        n_in: Int64,
        n_post: Int64,
        n_next: Int64,
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
        w_qkv: Int64,
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
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        kernel(
            scratch,
            emb,
            n_in,
            n_post,
            n_next,
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
            w_qkv,
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
