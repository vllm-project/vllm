# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd.

"""Shape-specific autotune configuration for the TileLang Hadamard kernel."""

from __future__ import annotations

from typing import Any

from vllm.tilelang_utils import tilelang

from .utils import GEMM_HADAMARD_ORDER

AUTOTUNE_TOKENS = 256


def get_composite_hadamard_configs(
    hidden_size: int,
    base_order: int,
    inverse: bool,
    dtype: str = "bfloat16",
    num_threads: int = 256,
    mma_rows: int = 0,
    gemm_policy: str = "square",
) -> list[dict[str, int | str]]:
    """Build ``threads x neighboring M tiles x policies`` candidates."""
    del inverse, dtype, num_threads, mma_rows, gemm_policy

    if hidden_size // base_order < 32:
        return [{"num_threads": 256, "mma_rows": 0, "gemm_policy": "square"}]

    rows_per_token = hidden_size // GEMM_HADAMARD_ORDER
    floor_rows = (rows_per_token // 32) * 32
    ceil_rows = ((rows_per_token + 31) // 32) * 32
    row_candidates = []
    for rows in (floor_rows, ceil_rows):
        if rows > 0 and rows not in row_candidates:
            row_candidates.append(rows)

    return [
        {
            "num_threads": threads,
            "mma_rows": rows,
            "gemm_policy": policy,
        }
        for threads in (512, 256)
        for rows in row_candidates
        for policy in ("square", "full_col")
    ]


def supply_composite_hadamard_inputs(params: list[Any]) -> list[Any]:
    """Create profiling inputs with a fixed dynamic token dimension."""
    supply = tilelang.utils.tensor.get_tensor_supply(tilelang.TensorSupplyType.Auto)
    inputs = []
    for param in params:
        if param.is_scalar():
            inputs.append(AUTOTUNE_TOKENS)
            continue
        shape = [
            dim if isinstance(dim, int) else AUTOTUNE_TOKENS for dim in param.shape
        ]
        inputs.append(
            supply(tilelang.engine.param.KernelParam(dtype=param.dtype, shape=shape))
        )
    return inputs


__all__ = [
    "AUTOTUNE_TOKENS",
    "get_composite_hadamard_configs",
    "supply_composite_hadamard_inputs",
]
