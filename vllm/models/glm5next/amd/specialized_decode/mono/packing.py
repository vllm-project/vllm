# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from the GLM-5 MonoKernel of ROCm/FlyDSL#1204 (@coderfeli),
# kernels/monokernel/packing.py, and ROCm/ATOM#2435 (@XiaobingSuper),
# atom/model_ops/monokernel/packing.py (Apache-2.0 License),
# Copyright (c) 2025 FlyDSL Project Contributors.

"""Host-side weight packing for the FFN launch."""

from __future__ import annotations

import torch


def pack_bf16(w: torch.Tensor) -> torch.Tensor:
    """Pack BF16 ``[N, K]`` for the kernel's 16-row, 64-K MFMA tiles."""
    if w.ndim != 2:
        raise ValueError(f"BF16 packing expects a matrix, got shape {tuple(w.shape)}")
    rows, k = w.shape
    if rows % 16 or k % 64:
        raise ValueError(
            f"BF16 matrix dimensions must be divisible by (16, 64), got {(rows, k)}"
        )
    w16 = w.view(torch.int16).reshape(rows // 16, 16, k // 64, 2, 4, 8)
    return w16.permute(0, 2, 3, 4, 1, 5).contiguous().view(-1)


def pack_shared_expert(
    gate_up: torch.Tensor,
    gate_up_scale: torch.Tensor,
    down: torch.Tensor,
    down_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """A row-major MXFP4 SwiGLU MLP as one expert of the AITER-shuffled MoE layout.

    ``gate_up`` ``[2 * I, K // 2]`` (gate rows, then up rows), ``down``
    ``[N, I // 2]``, each scale one E8M0 per 32 inputs, ``[rows, inputs // 32]``.
    Returns ``(w_ug, s_ug, w_dn, s_dn)`` as bytes."""
    from aiter.ops.shuffle import shuffle_scale, shuffle_weight

    def weight(t):
        fp4 = t.view(torch.uint8).contiguous().view(torch.float4_e2m1fn_x2)
        return shuffle_weight(fp4, layout=(16, 16)).view(torch.uint8)

    def scale(t):
        return shuffle_scale(t.view(torch.uint8).contiguous()).view(torch.uint8)

    return weight(gate_up), scale(gate_up_scale), weight(down), scale(down_scale)
