# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AMD/ROCm Gemma RMSNorm — re-exports the shared Triton implementation.

The kernels live in ``common/ops/gemma_rmsnorm.py``; this shim keeps the AMD
import path stable while avoiding duplication.
"""

from vllm.models.minimax_m3.common.ops.gemma_rmsnorm import (
    gemma_fused_add_rmsnorm,
    gemma_rmsnorm,
)

__all__ = ["gemma_rmsnorm", "gemma_fused_add_rmsnorm"]
