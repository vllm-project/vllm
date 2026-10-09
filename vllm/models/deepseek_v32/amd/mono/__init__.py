# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from ROCm/aiter#6173 at b02df0db8 (MIT License),
# Copyright (C) 2026, Advanced Micro Devices, Inc.:
# aiter/ops/flydsl/glm5_mono.py

"""GLM-5 fused decode-layer MonoKernel for gfx950.

One persistent launch per MoE layer and rank covers input RMSNorm, the MLA
projections, RoPE and KV-cache insert, sparse attention over precomputed
indices, o_proj with an in-kernel all-reduce, the router, and the MXFP4
experts with an in-kernel all-reduce. Device code from ROCm/FlyDSL#1204,
host wrapper and TP4 paths from ROCm/ATOM#2435.
"""

import torch

from vllm.models.deepseek_v32.amd.mono.config import (
    GLM5_KERNEL_SAMPLES,
    AttentionWeight,
    KvCacheLayout,
    Mxfp4ScaleLayout,
    Mxfp4WeightLayout,
    glm5_kernel_samples,
    glm5_tp_config,
)
from vllm.models.deepseek_v32.amd.mono.glm.op import (
    Glm5MonoKernel,
    prepare_glm5_weights,
)
from vllm.models.deepseek_v32.amd.mono.weights import LayerWeights, pack_dense_mlp
from vllm.models.deepseek_v32.amd.mono.weights import (
    _unshuffle_linear_weight as _unshuffle,
)

__all__ = [
    "GLM5_KERNEL_SAMPLES",
    "AttentionWeight",
    "Glm5MonoKernel",
    "KvCacheLayout",
    "LayerWeights",
    "Mxfp4ScaleLayout",
    "Mxfp4WeightLayout",
    "glm5_kernel_samples",
    "glm5_mono_launch_rows",
    "glm5_tp_config",
    "pack_dense_mlp",
    "prepare_glm5_weights",
    "unshuffle_linear_weight",
]


def unshuffle_linear_weight(weight: torch.Tensor) -> torch.Tensor:
    """Invert ``shuffle_weight(layout=(16, 16))`` for a one-byte 2D weight."""
    view = weight.view(weight.dtype)
    view.is_shuffled = True
    return _unshuffle(view)


def glm5_mono_launch_rows(rows: int, query_length: int = 1) -> tuple[int, int]:
    """Return ``(padded_rows, chunk)`` for a decode step of ``rows`` query rows.

    A one-row launch never completes on gfx950, so one row is padded to two.
    The pad row must carry slot -1 and an empty sparse-index range, so it
    writes no cache entry and attends to nothing.
    """
    padded = 2 if rows == 1 and query_length == 1 else rows
    return padded, glm5_kernel_samples(padded, query_length)
