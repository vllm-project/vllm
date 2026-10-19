# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compile-only viability prototypes for checkpoint-specific mono stages."""

from __future__ import annotations

from dataclasses import dataclass

import torch

HIDDEN = 6144


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
