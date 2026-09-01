# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.2 decode GEMM selection for unquantized BF16 on SM103."""

from __future__ import annotations

import torch
from torch import nn

from vllm.models.common.ops.low_latency_linear import (
    FusedATable,
    install_fused_a_linear,
)
from vllm.platforms import current_platform

# Measured on SM103: the fused-A GEMM wins over cuBLAS from M=3 up.
GLM52_FUSED_A_TABLE: FusedATable = {
    (2624, 6144): frozenset(range(3, 17)),  # qkv_a_proj
    (2048, 2048): frozenset(range(3, 17)),  # q_b_proj
}


def enable_glm52_low_latency_gemm(module: nn.Module, dtype: torch.dtype) -> None:
    if current_platform.is_device_capability((10, 3)):
        install_fused_a_linear(module, dtype, GLM52_FUSED_A_TABLE)
