# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Intel XPU fused ops for MiniMax-M3."""

from vllm.models.minimax_m3.xpu.ops.gemma_rmsnorm import (
    gemma_fused_add_rmsnorm,
    gemma_rmsnorm,
)

__all__ = [
    "gemma_rmsnorm",
    "gemma_fused_add_rmsnorm",
]
