# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compile-only viability prototypes for checkpoint-specific mono stages."""

from __future__ import annotations

from dataclasses import dataclass

import torch

HIDDEN = 6144
INTER = 768
EXPERTS = 129


@dataclass(frozen=True)
class StageResource:
    stage: str
    threads: int
    lds_bytes: int
    grid_ctas: int
    residency_proven: bool


def bf16_gemv_resource(output_rows: int = 2560) -> StageResource:
    return StageResource(
        stage="bf16_projection",
        threads=512,
        lds_bytes=0,
        grid_ctas=(output_rows + 255) // 256,
        residency_proven=False,
    )


def fp32_router_resource(width: int) -> StageResource:
    block_m = 1 if width <= 4 else 4
    return StageResource(
        stage="fp32_router",
        threads=512,
        lds_bytes=0,
        grid_ctas=((width + block_m - 1) // block_m) * 128,
        residency_proven=False,
    )


def a4w4_resource() -> StageResource:
    from aiter.ops.flydsl.kernels.mxfp4_gemm1 import (
        _bm_constants,
        default_epi_splits,
        default_k_stages,
    )

    bm, bn, bk, waves = 16, 256, 256, 4
    k_tiles = HIDDEN // bk
    k_half = bk // 2
    epi = default_epi_splits(bm, bn, 1, waves)
    stages = default_k_stages(bm, bn, k_half, k_tiles, 2 * INTER, 1, epi)
    *_, lds = _bm_constants(bm, bn, k_half, k_tiles, 1, epi, stages)
    return StageResource(
        stage="a4w4_expert",
        threads=waves * 64,
        lds_bytes=lds,
        grid_ctas=256,
        residency_proven=False,
    )


def run_bf16_gemv_prototype(hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    from aiter.ops.flydsl.kernels.gemm_a16w16_gfx950 import gemm_a16w16

    return gemm_a16w16(hidden, weight.t(), layout="nt")


def run_fp32_router_prototype(
    hidden: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    from vllm.model_executor.layers.fused_moe.router.rocm_fp32_router_gemm import (
        rocm_fp32_router_gemm,
    )

    return rocm_fp32_router_gemm(hidden, weight)


def compile_a4w4_prototype() -> None:
    from aiter.ops.flydsl.kernels.mxfp4_gemm1 import compile_gemm1_a4w4_port

    launch = compile_gemm1_a4w4_port(
        BM=16,
        use_nt=True,
        inline_quant=True,
        D_HIDDEN=HIDDEN,
        D_INTER=INTER,
        NE=EXPERTS,
        BN=256,
        BK=256,
        a_dtype="fp4",
        out_dtype="fp4",
        act="swiglu",
        swiglu_limit=7.0,
    )
    launch(
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        16,
        256,
        0,
        0,
        0,
        0,
        stream=torch.cuda.current_stream(),
    )
