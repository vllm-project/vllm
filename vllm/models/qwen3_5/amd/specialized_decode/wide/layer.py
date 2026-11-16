# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The wide decode's launches: one persistent kernel per GDN layer, one per
full-attention layer after vLLM's attention (`LayerBuild.kind`):

    gdn   [first: embedding -> input norm -> x]
          GDN proj + conv -> recurrence + gated norm -> out_proj
          -> TP exchange -> post-attention combine (x, x4)
          -> router -> route -> experts up -> experts down
          -> TP exchange -> next layer's input combine (or the final norm)
          [-> the next (attention) layer's qkv_proj]
    attn  o_proj (gated) -> TP exchange -> post-attention combine
          -> the MLP half as above

Phases are separated by grid barriers; an exchange's readers wait on its line
flags instead. A launch reads the residual and the normed rows `x` the
previous one left in its scratch and leaves the next ones there; the step's
last leaves the final norm's rows in `x`.

Up to `SPLIT_MAX` rows a layer is two launches (`LayerBuild.split`), its
MLP half `mono.blocks.layer_block` (out_proj / o_proj -> all-reduce ->
post-attention norm -> MoE -> all-reduce, dataflow hand-offs, no grid
barriers) after:

    gdn   [embedding or] the previous layer's output + residual -> input norm
          -> GDN proj + conv -> recurrence + gated norm -> core
    attn  the previous layer's output + residual -> input norm -> qkv_proj
          (then vLLM's attention)

where GDN proj -> recurrence is one phase: its hand-offs are tagged pairs
(`stages.tagged`).
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import const_expr
from flydsl.expr.typing import Int32, Int64

from vllm.models.qwen3_5.amd.specialized_decode.abi import KernelAbi, key_tuple
from vllm.models.qwen3_5.amd.specialized_decode.layout import (
    ATTN_CORE,
    BLOCKS,
    HIDDEN,
    MAX_TOKENS,
    QKV,
    SPLIT_MAX,
    THREADS,
    VALUE_DIM,
    WAVES,
)
from vllm.models.qwen3_5.amd.specialized_decode.ops import ld, ld_ptr64, memrealtime, st
from vllm.models.qwen3_5.amd.specialized_decode.wide.grid import (
    Grid,
    TraceInvariant,
    grid_sync,
    prologue,
)
from vllm.models.qwen3_5.amd.specialized_decode.wide.stages import attn as attn_
from vllm.models.qwen3_5.amd.specialized_decode.wide.stages import exchange as ex_
from vllm.models.qwen3_5.amd.specialized_decode.wide.stages import (
    gdn,
    moe,
    norm,
    tiled,
)

KINDS = ("gdn", "attn")
ATTN, MLP = 0, 1  # a layer's two exchanges
FORMATS = ("bf16", "mxfp4", "mxfp4_h32")  # a GDN projection's weight format
# fold counter bases: the GDN projection's and out_proj's share a phase
FOLD_PROJ, FOLD_OUT = 0, 128
TAG_LAYERS = 256  # tags a step: (layer, use) pairs below it


def q4_of(fmt: str, shuffled: bool = True):
    """The MXFP4 rows a GEMM of weight format `fmt` reads (`norm.Q4_MODES`),
    unless `shuffled` (the ASM GEMM) as the Triton GEMM quantizes them."""
    q = {"bf16": None, "mxfp4": "plain", "mxfp4_h32": "h32"}[fmt]
    return q if q is None or shuffled else q + "_t"


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
    first_q4: str = "none"  # the first combine's MXFP4 rows ("none", ``norm.Q4_MODES``)
    next_q4: str = "none"  # the closing combine's, for the next layer's qkvz
    qkv_tail: bool = False  # the next layer is full attention: run its qkv_proj
    rip: int = 256  # the experts' padded intermediate rows a rank
    eps: float = 1e-6
    gdn_eps: float = 1e-6
    trace: bool = False  # CTA 0 stamps the clock at every phase end (``phase_names``)

    @property
    def split(self) -> bool:
        """The launch ends at the GDN core (`core_out` [rows, VALUE_DIM]
        bf16, Hadamard-rotated for an "mxfp4_h32" out_proj), or for "attn"
        at its qkv_proj (`qkv_out` in the scratch, of `w_qkv`); its input
        combine adds `emb` to `res_in` (the first layer: `emb` alone), its
        MXFP4 rows `first_q4`."""
        return self.tokens <= SPLIT_MAX

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
            if q != "none" and q not in norm.Q4_MODES:
                raise ValueError(f"MXFP4 rows {q!r}")


def core_of(key: LayerBuild) -> str:
    """The GDN core's format for out_proj (`gdn.CORE_MODES`)."""
    if key.out == "bf16" or key.out_shuffled:
        return key.out
    return key.out + "_t"


def _q4(q: str):
    return None if q == "none" else q


ABI = KernelAbi(
    (
        "scratch",
        "emb",
        "res_in",
        "core_out",
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
    """The stamps of a traced build (`LayerBuild.trace`), in order: each
    marks the end of the named phase ("start": the kernel's start)."""
    if key.split:
        return [
            "start",
            "combine",
            "gdn_proj+recur" if key.kind == "gdn" else "qkv_proj",
        ]
    out = ["start"] + (["first_combine"] if key.first else [])
    if key.kind == "gdn":
        out += ["gdn_proj+conv", "gdn_recur", "out"]
    else:
        out += ["o_proj"]
    out += ["allreduce+combine"]
    out += moe.PHASES + ["allreduce+combine"]
    if key.qkv_tail:
        out += ["qkv_proj"]
    return out


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
    if rows > SPLIT_MAX:
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
    """A width's scratch buffers (`scratch_layout`) as byte addresses."""

    def __init__(self, base, rows: int, rip: int):
        for name, (off, _) in scratch_layout(rows, rip).items():
            setattr(self, name, fx.Int64(base) + fx.Int64(off))

    def __getattr__(self, name: str) -> fx.Int64:
        # the attributes are the width's layout names, set in __init__
        raise AttributeError(name)


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
    aux_words = max(ex_.GroupTile.words(S), gdn.recur_lds_words(), norm.LDS_WORDS)
    chunks = _resident_chunks()

    @fx.struct
    class GemmLds:
        stage: fx.Array[fx.Int32, tiled.stage_words(S, chunks), 16]
        aux: fx.Array[fx.Int32, aux_words, 16]

    if key.split:
        PhaseLds = GemmLds
    else:
        MoeLds = moe.lds_struct(shape)

        @fx.union
        class PhaseLds:  # type: ignore[no-redef]
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
    name += "_split" if key.split else ""

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kernel(
        scratch: Int64,
        emb: Int64,
        res_in: Int64,
        core_out: Int64,
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
        gl = ph.peek() if key.split else ph.gemm.peek()
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
            """The phase's grid barrier; `issue()` (the next phase's first
            weights, a split launch's) runs inside it, its result returned."""
            out = grid_sync(g, issue)
            stamp(name)
            return out

        stamp("start")
        tag0 = ep * TAG_LAYERS + layer * 2

        def exchange(kind: int, mode: str):
            return ex_.Exchange(
                bases, own, rank, tp, kind, mode, layer % 2, tag0 + kind + 1, S
            )

        def all_reduce(kind: int):
            """The exchange a producer pushes to, and the one its combine reads."""
            return exchange(kind, "scatter"), exchange(kind, "gather")

        def reduce(push):
            """The owner half of the exchange, after the producers' pushes
            (every wave of the grid)."""
            gather = all_reduce(push.kind)[1]
            ex_.reduce_lines(g, push, gather, g.spread, BLOCKS * WAVES)

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

        if const_expr(key.first and not key.split):
            combine(None, n_in, _q4(key.first_q4), first=True)
            end_phase("first_combine")

        def mlp_half():
            mw = moe.Weights(w_gate, w13, w13s, w2, w2s)
            b = moe.Bufs(
                part=scr.part,
                x4=scr.x4,
                x4s=scr.x4s,
                **{k: getattr(scr, k) for k in moe.scratch_words(shape)},
            )
            push, read = all_reduce(MLP)
            ml = ph.moe.peek()
            moe.mlp(g, shape, gemm, ml, scr.x, mw, b, push, end_phase)
            if const_expr(key.trace):
                end_phase(moe.PHASES[-1])
            reduce(push)
            combine(read, n_next, _q4(key.next_q4))
            if const_expr(key.trace or key.qkv_tail):
                end_phase("allreduce+combine" if key.trace else "mlp+allreduce+combine")
            if const_expr(key.qkv_tail):
                attn_.qkv_proj(
                    g, gemm, w_qkv, scr.x, scr.qkv_out, scr.part, S, FOLD_PROJ
                )
                if const_expr(key.trace):
                    end_phase("qkv_proj")

        def back(weights, source, k, name="out+allreduce+combine"):
            push, read = all_reduce(ATTN)
            tile = ex_.GroupTile(gl.aux.ptr, S)
            gdn.out(g, gemm, weights, source, scr.part, tile, push, k, FOLD_OUT)
            if const_expr(key.trace):
                end_phase(name.removesuffix("+allreduce+combine"))
            reduce(push)
            combine(read, n_post, "plain")
            end_phase("allreduce+combine" if key.trace else name)
            mlp_half()

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
                core=core_out if key.split else scr.core,
                core_s=scr.core_s,
                part=scr.part,
                st_idx=st_idx,
                conv_state=conv_state,
                conv_stride=conv_stride,
                ssm=ssm,
                ssm_stride=ssm_stride,
            )
            out_w = gdn.out_weights(w_out, w_out_s, key.out, key.out_shuffled)
            if const_expr(key.split):
                out = norm.Out(scr.res, scr.x, scr.x4, scr.x4s, _q4(key.first_q4))
                res = None if key.first else res_in
                norm.combine(g, None, out, n_in, aux_f32, S, key.eps, emb, res)
                # the projection's first weights, in the barrier's wait
                pending = end_phase("combine", lambda: gdn.proj_issue(g, w))
                gtag = tag0 + 1
                gdn.proj(g, gemm, w, b, S, FOLD_PROJ, pending, gtag)
                core_fmt = "bf16_h32" if key.out == "mxfp4_h32" else "bf16"
                gdn.recur(
                    g,
                    w,
                    b,
                    aux_f32,
                    S,
                    key.gdn_eps,
                    core_fmt,
                    None,
                    gtag,
                )
                stamp("gdn_proj+recur")
            else:
                gdn.proj(g, gemm, w, b, S, FOLD_PROJ)
                end_phase("gdn_proj+conv")
                gdn.recur(g, w, b, aux_f32, S, key.gdn_eps, core_of(key))
                end_phase("gdn_recur")
                source: tiled.Source = tiled.Source(scr.core, VALUE_DIM)
                if const_expr(key.out != "bf16"):
                    source = tiled.Mx4Source(scr.core, scr.core_s, VALUE_DIM)
                back(out_w, source, VALUE_DIM)
        if const_expr(key.kind == "attn" and key.split):
            out = norm.Out(scr.res, scr.x, scr.x4, scr.x4s, None)
            norm.combine(g, None, out, n_in, aux_f32, S, key.eps, emb, res_in)
            qw = attn_.qkv_weights(w_qkv)
            pending = end_phase(
                "combine",
                lambda: tiled.issue_unit(bid, tid, qw, attn_.qkv_plan()),
            )
            attn_.qkv_proj(
                g, gemm, w_qkv, scr.x, scr.qkv_out, scr.part, S, FOLD_PROJ, pending
            )
            stamp("qkv_proj")
        if const_expr(key.kind == "attn" and not key.split):
            back(
                tiled.Bf16Weights(w_out, HIDDEN, ATTN_CORE),
                gdn.GatedCore(attn, gate, ATTN_CORE),
                ATTN_CORE,
                name="o_proj+allreduce+combine",
            )
        if const_expr(key.trace):
            assert names == phase_names(key), (names, phase_names(key))

    @flyc.jit
    def launch(
        scratch: Int64,
        emb: Int64,
        res_in: Int64,
        core_out: Int64,
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
            res_in,
            core_out,
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
    """Launch `key`'s kernel on the current stream; `args`: every ABI name
    (tensors by data pointer, ints as is)."""
    from vllm.v1.attention.ops.flydsl_kernels import _run_compiled

    vals = [
        v.data_ptr() if isinstance(v, torch.Tensor) else int(v) for v in ABI.pack(args)
    ]
    _run_compiled(build(key), *vals, torch.cuda.current_stream())


def compile_only(key: LayerBuild) -> None:
    """Compile `key`'s kernel without launching it (FlyDSL's compile-only
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
