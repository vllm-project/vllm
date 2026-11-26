# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kernel registry of the MiniMax-M3 specialized decode (NVIDIA, SM100 family).

Maps each decode stage of `specialized_decode/stages.py` to the
implementation that serves it and the envelope it is valid for: target arch,
TP size and rows per rank (rows = c * (K + 1), c concurrent requests, K draft
tokens). `select` returns `None` for any (stage, arch, tp, rows) outside
every envelope; the caller then runs the stock path for that stage.

Nothing is imported or compiled at import time: `KernelImpl.load()` imports
the implementing module lazily.

The kernels cover the sparse MoE layers 3-59; layers 0-2 run stock. The executor runs
steps of up to `MAX_ROWS` = 64 rows
(`min(64, 8 * (K + 1))`); each stage's envelope below is how far its kernel
goes, and a step with more rows runs that stage stock (`select` returns
None, `fallback_stages` names them for the build log):

====================  =======================================================
stage                 implementation
====================  =======================================================
front                 AR land + residual + RMSNorm + MXFP8 + QKV/index GEMV
                      (tcgen05) + q/k norm + RoPE + FP8 KV / index-K insert
index_score           upstream `minimax_m3_index_decode_score_cutedsl`
page_topk             top-16 pages per (row, index head)
sparse_decode         16-CTA cluster FP8 sparse decode with MXFP8 output
o_proj                MXFP8 o_proj GEMV + router CTAs, Lamport push
allreduce_norm        AR land + residual + norm + NVFP4/MXFP8 quant + top-4
moe                   fused NVFP4 routed + MXFP8 shared experts, push
dense_mlp             stock (no entry)
====================  =======================================================

Rows per stage (`max_rows`) and what serves the rows above it:

====================  ====  ==================================================
stage                 rows  above it
====================  ====  ==================================================
front                 16    the "wide" variant to 64 (the kernel's activation
                            tile and its 20 boundary clusters are sized for 16
                            rows): stock AR + norm, MXFP8 quantize, the GEMV
                            QKV / index projection (rows = MMA N), the stock
                            q/k norm + RoPE + cache insert op
index_score           64    (grid = requests; only heads x qlen <= 32 binds)
page_topk             64    (grid = rows x heads)
sparse_decode         64    (grid = rows x heads clusters; beyond one wave of
                            clusters it runs in more than one wave)
o_proj                8     the "wide" variant (`wide.py`) to 64: the same
                            MXFP8 GEMV with the rows as the MMA's N (16 / 32 /
                            48 / 64), each weight tile read once; no router
                            CTAs, no push (BF16 partial)
allreduce_norm        8     stock (FlashInfer AR + Gemma norm, the stock
                            router GEMM)
moe                   8     stock (at 32-64 rows x top-4 most experts are hit
                            and the stock grouped GEMMs stream each weight
                            once, at the HBM roofline; the fused engine's
                            slots stop at 8)
dense_mlp             -     stock at every row count
====================  ====  ==================================================

A stage's implementations are tried in order; the "wide" ones (`variant`)
serve only the rows above the fused pair's (`min_rows`).
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass

STAGES = (
    "front",
    "index_score",
    "page_topk",
    "sparse_decode",
    "o_proj",
    "allreduce_norm",
    "moe",
    "dense_mlp",
)

BLACKWELL = ("sm_100a", "sm_103a")
# Largest executor step (specialized_decode.MAX_ROWS): 8 requests x (7 + 1).
MAX_ROWS = 64
# Rows of the o_proj router CTAs, the post and the MoE engine (one build per power of
# two); the front's activation tile.
PAIR_MAX_ROWS = 8
FRONT_MAX_ROWS = 16


@dataclass(frozen=True)
class KernelImpl:
    stage: str
    name: str
    target: str  # "module:attr", module relative to this package
    archs: tuple[str, ...]
    tps: tuple[int, ...]
    max_rows: int
    provenance: str  # how it is built: "CuTe DSL" or "upstream"
    cache_formats: tuple[str, ...] = ()  # KV / indexer cache formats it reads/writes
    notes: str = ""

    pow2_rows: bool = False
    min_rows: int = 1
    variant: str = ""  # "" = the stage's primary kernel; "wide" = wide.py

    def supports(self, arch: str, tp: int, rows: int) -> bool:
        if self.pow2_rows and rows & (rows - 1):
            return False
        return (
            arch in self.archs
            and tp in self.tps
            and self.min_rows <= rows <= self.max_rows
        )

    def load(self):
        mod, attr = self.target.split(":")
        return getattr(importlib.import_module(f"{__name__}.{mod}"), attr)


REGISTRY: dict[str, tuple[KernelImpl, ...]] = {
    "front": (
        KernelImpl(
            "front",
            "qkv-front-mxfp8-smem-a",
            "front:QkvFrontStage",
            BLACKWELL,
            (4,),
            FRONT_MAX_ROWS,
            "CuTe DSL",
            ("fp8_e4m3",),
        ),
        KernelImpl(
            "front",
            "mxfp8-gemv-qkv-wide",
            "wide:WideFront",
            BLACKWELL,
            (4,),
            MAX_ROWS,
            "MXFP8 quantize + GEMV (CuTe DSL); stock norm / RoPE / cache insert op",
            ("fp8_e4m3",),
            min_rows=FRONT_MAX_ROWS + 1,
            variant="wide",
        ),
    ),
    "index_score": (
        KernelImpl(
            "index_score",
            "stock-cutedsl",
            "index_score:StockIndexScoreStage",
            BLACKWELL,
            (4,),
            MAX_ROWS,
            "upstream (vLLM minimax_m3_index_decode_score_cutedsl)",
            ("fp8_e4m3",),
        ),
    ),
    "page_topk": (
        KernelImpl(
            "page_topk",
            "page-top16",
            "page_topk:PageTopkStage",
            BLACKWELL,
            (4,),
            MAX_ROWS,
            "CuTe DSL",
        ),
    ),
    "sparse_decode": (
        KernelImpl(
            "sparse_decode",
            "cluster16-mxfp8-out",
            "sparse_decode:ClusterSparseDecodeStage",
            BLACKWELL,
            (4,),
            MAX_ROWS,
            "CuTe DSL, fp16-P online softmax",
            ("fp8_e4m3",),
        ),
    ),
    "o_proj": (
        KernelImpl(
            "o_proj",
            "mxfp8-gemv-router-tp4",
            "o_proj:OProjRouterStage",
            BLACKWELL,
            (4,),
            PAIR_MAX_ROWS,
            "CuTe DSL",
            pow2_rows=True,
            notes="rows must be a power of two (1, 2, 4, 8); every o_proj "
            "call must be followed by its allreduce_norm call",
        ),
        KernelImpl(
            "o_proj",
            "mxfp8-gemv-wide",
            "wide:WideOProj",
            BLACKWELL,
            (4,),
            MAX_ROWS,
            "GEMV (CuTe DSL), no router CTAs / push",
            min_rows=PAIR_MAX_ROWS + 1,
            variant="wide",
        ),
    ),
    "allreduce_norm": (
        KernelImpl(
            "allreduce_norm",
            "route-warp-tp4-fc1-per-token",
            "allreduce_norm:PostRouteStage",
            BLACKWELL,
            (4, 8),
            PAIR_MAX_ROWS,
            "CuTe DSL",
        ),
    ),
    "moe": (
        KernelImpl(
            "moe",
            "fused-nvfp4-shared-mxfp8",
            "moe:FusedMoeStage",
            BLACKWELL,
            (4,),
            PAIR_MAX_ROWS,
            "CuTe DSL",
        ),
    ),
    "dense_mlp": (),
}


def select(
    stage: str, arch: str, tp: int, rows: int, variant: str | None = None
) -> KernelImpl | None:
    """The implementation serving `stage` for this envelope (of
    `variant`, when given), or None (run the stock path)."""
    if stage not in STAGES:
        raise KeyError(f"unknown stage {stage!r}; stages: {STAGES}")
    for impl in REGISTRY.get(stage, ()):
        if variant is not None and impl.variant != variant:
            continue
        if impl.supports(arch, tp, rows):
            return impl
    return None


def fallback_stages(arch: str, tp: int, rows: int) -> tuple[str, ...]:
    """Stages with an implementation that does not cover `rows` at
    (arch, tp): they run stock for a step of that many rows."""
    return tuple(
        stage
        for stage in STAGES
        if any(arch in i.archs and tp in i.tps for i in REGISTRY.get(stage, ()))
        and select(stage, arch, tp, rows) is None
    )


def supported_envelope(
    stage: str,
) -> list[tuple[tuple[str, ...], tuple[int, ...], int]]:
    return [(i.archs, i.tps, i.max_rows) for i in REGISTRY.get(stage, ())]


def max_rows(stage: str, variant: str = "") -> int:
    """The largest rows any `variant` implementation of `stage` runs."""
    return max(
        (i.max_rows for i in REGISTRY.get(stage, ()) if i.variant == variant), default=0
    )


__all__ = [
    "BLACKWELL",
    "FRONT_MAX_ROWS",
    "MAX_ROWS",
    "PAIR_MAX_ROWS",
    "REGISTRY",
    "STAGES",
    "KernelImpl",
    "fallback_stages",
    "max_rows",
    "select",
    "supported_envelope",
]
