# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3 decode GEMM selection for unquantized BF16 on SM90/SM100/SM103.

Dispatch is purely by local ``(N, K)`` shape and token count ``M`` -- the module
name plays no role. Each measured shape maps to the token counts where
``dsv3_fused_a_gemm`` beat the default unquantized GEMM; the projection names in
the comments are debug labels only.

The supported capabilities carry separate measured tables:
:data:`KIMI_K3_PROJECTIONS` was tuned on B300 (SM103),
:data:`KIMI_K3_PROJECTIONS_SM100` on B200 (SM100), and
:data:`KIMI_K3_PROJECTIONS_SM90` on H200 (SM90). The per-(shape, M) winners
genuinely differ between the parts, so the tables must not be merged.
"""

from __future__ import annotations

import torch
from torch import nn

from vllm.models.common.ops.low_latency_linear import (
    FusedATable,
    install_fused_a_linear,
)
from vllm.platforms import current_platform

_M1_TO_16 = frozenset(range(1, 17))

# B300 (SM103), keyed by local (N, K). TP8 shapes first, then TP16-only ones.
# Re-measured against the CuTeDSL BF16 default with
# benchmarks/kernels/benchmark_fused_a_vs_flashinfer_bf16.py: every (shape, M)
# below is one the fused-A GEMM still wins. Shapes the default now wins at
# every token count (3216x7168, 3584x7168, 4224x7168) were dropped, and the
# entries whose low-M end the default took were trimmed.
KIMI_K3_PROJECTIONS: FusedATable = {
    (1536, 128): _M1_TO_16,  # f_b_proj
    (3072, 128): _M1_TO_16,  # f_b_proj
    # 1536x7168 is shared by shared_gate_up_proj and mla_g_proj.
    (1536, 7168): _M1_TO_16,
    (2112, 7168): frozenset(range(3, 17)),  # fused_qkv_a_proj
    (2304, 1536): frozenset(range(2, 17)),  # q_b_proj
    (4608, 1536): _M1_TO_16,  # q_b_proj
    (7168, 768): _M1_TO_16,  # shared_down_proj
    # TP16.
    (768, 7168): frozenset(range(5, 9)),  # mla_g_proj/shared_gate_up_proj
    (1152, 1536): frozenset(range(3, 17)),  # q_b_proj
    (768, 128): _M1_TO_16,  # f_b_proj
    (7168, 384): frozenset(range(1, 9)),  # shared_down_proj
}

# B200 (SM100). Measured independently: the SM103 winners do not carry over.
KIMI_K3_PROJECTIONS_SM100: FusedATable = {
    (1536, 128): frozenset({1, 16}),  # f_b_proj
    (3072, 128): frozenset({8}),  # f_b_proj
    (1536, 7168): frozenset({4, 8}),  # shared_gate_up_proj/mla_g_proj
    (2112, 7168): frozenset({4, 16}),  # fused_qkv_a_proj
    (2304, 1536): _M1_TO_16,  # q_b_proj
    (4608, 1536): frozenset({1, 2, 4}),  # q_b_proj
    (7168, 768): frozenset({1}),  # shared_down_proj
}

# H200 (SM90).
KIMI_K3_PROJECTIONS_SM90: FusedATable = {
    (1536, 128): frozenset(range(1, 9)),  # f_b_proj
    (3072, 128): frozenset({1, 2, 5, 6, 7, 8, 9}),  # f_b_proj
    (1536, 7168): frozenset(range(7, 17)),  # shared_gate_up_proj/mla_g_proj
    (2112, 7168): frozenset(range(5, 17)),  # fused_qkv_a_proj
    (2304, 1536): frozenset(range(3, 9)),  # q_b_proj
}


def _low_latency_table() -> FusedATable | None:
    """Measured dispatch table for the current device, or None if unsupported."""
    if current_platform.is_device_capability((10, 3)):
        return KIMI_K3_PROJECTIONS
    if current_platform.is_device_capability((10, 0)):
        return KIMI_K3_PROJECTIONS_SM100
    if current_platform.is_device_capability((9, 0)):
        return KIMI_K3_PROJECTIONS_SM90
    return None


def select_kimi_k3_tokens(n: int, k: int) -> frozenset[int]:
    """Token counts taking the fused-A GEMM for a local ``(N, K)``."""
    table = _low_latency_table()
    return frozenset() if table is None else table.get((n, k), frozenset())


def enable_kimi_k3_low_latency_gemm(module: nn.Module, dtype: torch.dtype) -> None:
    """Install the fused-A GEMM on every layer the current device's table covers."""
    table = _low_latency_table()
    if table is not None:
        install_fused_a_linear(module, dtype, table)
