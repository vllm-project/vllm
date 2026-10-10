# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from ROCm/aiter#6173 at c39b56c36 (Apache-2.0 License),
# Copyright (c) 2025 FlyDSL Project Contributors:
# aiter/ops/flydsl/kernels/glm5_mono/packing.py

"""Host-side weight packing for fused model-layer launch wrappers."""

from __future__ import annotations

import torch

from vllm.models.deepseek_v32.amd.mono.config import (
    GLM5_CONFIG,
    AttentionWeight,
    ExpertWeight,
    LayerConfig,
    MoeMode,
    Mxfp4ScaleLayout,
    Mxfp4WeightLayout,
    RouterWeightLayout,
    as_layer_config,
    as_mxfp4_scale_layout,
    as_mxfp4_weight_layout,
    as_router_weight_layout,
    moe_format,
)


def pack_fp8(q: torch.Tensor) -> torch.Tensor:
    """Pack FP8 ``[..., N, K]`` for the kernel's 16-row, 64-K MFMA tiles."""
    *lead, rows, k = q.shape
    if rows % 16 or k % 64:
        raise ValueError(
            f"FP8 matrix dimensions must be divisible by (16, 64), got {(rows, k)}"
        )
    w8 = q.view(torch.uint8).reshape(*lead, rows // 16, 16, k // 64, 2, 4, 8)
    nlead = len(lead)
    order = list(range(nlead)) + [nlead + position for position in (0, 2, 4, 1, 3, 5)]
    return w8.permute(*order).contiguous().view(-1)


def pack_ptpc_fp8(q: torch.Tensor) -> torch.Tensor:
    """Pack row-major PTPC FP8 weights in AITER's 16x16 preshuffle layout."""
    *lead, rows, k = q.shape
    if rows % 16 or k % 32:
        raise ValueError(
            f"PTPC FP8 matrix dimensions must be divisible by (16, 32), got {(rows, k)}"
        )
    values = q.view(torch.uint8).reshape(*lead, rows // 16, 16, k // 32, 2, 16)
    nlead = len(lead)
    order = list(range(nlead)) + [nlead + position for position in (0, 2, 3, 1, 4)]
    return values.permute(*order).contiguous().view(-1)


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


def pack_bf16_atom(w: torch.Tensor) -> torch.Tensor:
    """Keep the row-major BF16 layout used by ATOM's unquantized router."""
    if w.ndim != 2:
        raise ValueError(f"BF16 packing expects a matrix, got shape {tuple(w.shape)}")
    return w.contiguous().view(-1)


def pack_mxfp4(q: torch.Tensor) -> torch.Tensor:
    """Pack MXFP4 for four BF16 MFMA K32 steps in each 128-K tile."""
    q = q.view(torch.uint8)
    *lead, rows, packed_k = q.shape
    k = packed_k * 2
    if rows % 16 or k % 128:
        raise ValueError(
            f"MXFP4 matrix dimensions must be divisible by (16, 128), got {(rows, k)}"
        )
    w4 = (
        q.reshape(*lead, rows // 16, 16, k // 128, 4, 4, 4)
        .view(torch.int32)
        .squeeze(-1)
    )
    nlead = len(lead)
    order = list(range(nlead)) + [nlead + position for position in (0, 2, 4, 1, 3)]
    return w4.permute(*order).contiguous().view(torch.uint8).view(-1)


def pack_a16w4_weight(q: torch.Tensor) -> torch.Tensor:
    """Pack MXFP4 values in the ATOM/AITER 16-row by 16-byte tile order."""
    q = q.view(torch.uint8)
    *lead, rows, packed_k = q.shape
    if rows % 16 or packed_k % 32:
        raise ValueError(
            "A16W4 weights require rows divisible by 16 and logical K divisible "
            f"by 64, got rows={rows}, K={packed_k * 2}"
        )
    tiled = q.reshape(*lead, rows // 16, 16, packed_k // 32, 2, 16)
    nlead = len(lead)
    order = list(range(nlead)) + [nlead + position for position in (0, 2, 3, 1, 4)]
    return tiled.permute(*order).contiguous().view(torch.uint8).view(-1)


def pack_a16w4_scale(scale: torch.Tensor) -> torch.Tensor:
    """Pack E8M0 scales in the ATOM/AITER 256-row by 8-group tile order."""
    scale = scale.view(torch.uint8)
    if scale.ndim < 2:
        raise ValueError(
            f"A16W4 scales must have at least two dimensions, got {scale.ndim}"
        )
    groups = scale.shape[-1]
    rows = scale.numel() // groups
    flat = scale.reshape(rows, groups)
    padded_rows = (rows + 255) // 256 * 256
    padded_groups = (groups + 7) // 8 * 8
    padded = torch.zeros(
        padded_rows, padded_groups, dtype=torch.uint8, device=scale.device
    )
    padded[:rows, :groups] = flat
    packed = padded.view(padded_rows // 32, 2, 16, padded_groups // 8, 2, 4)
    return packed.permute(0, 3, 5, 2, 4, 1).contiguous().view(-1)


def pack_layer_weights(
    tensors: dict[str, torch.Tensor],
    moe_mode: MoeMode | str = MoeMode.W8A8,
    model_config: LayerConfig | str = GLM5_CONFIG,
    attention_only: bool = False,
    *,
    mxfp4_weight_layout: Mxfp4WeightLayout | str | None = None,
    mxfp4_scale_layout: Mxfp4ScaleLayout | str | None = None,
    router_weight_layout: RouterWeightLayout | str | None = None,
) -> dict[str, torch.Tensor]:
    """Pack weights for a model profile and the selected kernel storage contract.

    Native layouts are the shared default. Model wrappers may select alternate
    physical layouts explicitly without coupling them to model geometry.
    """
    config = as_layer_config(model_config)
    attention_names = ("w_qkv_a", "w_q_b", "w_uk", "w_uv", "w_o")
    expert_names = ("w_ug", "w_dn")
    required = (
        attention_names if attention_only else (*attention_names, *expert_names, "w_r")
    )
    missing = [name for name in required if name not in tensors]
    if missing:
        raise ValueError(f"missing layer weights: {', '.join(missing)}")

    pack_attention = (
        pack_bf16 if config.attention_weight is AttentionWeight.BF16 else pack_fp8
    )
    packed = {name: pack_attention(tensors[name]) for name in attention_names}
    if attention_only:
        return packed

    weight = moe_format(moe_mode).weight
    weight_layout = (
        Mxfp4WeightLayout.NATIVE
        if mxfp4_weight_layout is None
        else as_mxfp4_weight_layout(mxfp4_weight_layout)
    )
    scale_layout = (
        Mxfp4ScaleLayout.NATIVE
        if mxfp4_scale_layout is None
        else as_mxfp4_scale_layout(mxfp4_scale_layout)
    )
    router_layout = (
        RouterWeightLayout.NATIVE
        if router_weight_layout is None
        else as_router_weight_layout(router_weight_layout)
    )

    if weight is ExpertWeight.MXFP4_BLOCK32:
        pack_expert = (
            pack_a16w4_weight if weight_layout is Mxfp4WeightLayout.ATOM else pack_mxfp4
        )
    else:
        if (
            weight_layout is not Mxfp4WeightLayout.NATIVE
            or scale_layout is not Mxfp4ScaleLayout.NATIVE
        ):
            raise ValueError("ATOM MXFP4 layouts require an MXFP4 expert mode")
        pack_expert = pack_fp8
    packed.update({name: pack_expert(tensors[name]) for name in expert_names})
    if scale_layout is Mxfp4ScaleLayout.ATOM:
        packed.update(
            {name: pack_a16w4_scale(tensors[name]) for name in ("s_ug", "s_dn")}
        )
    packed["w_r"] = (
        pack_bf16_atom(tensors["w_r"])
        if router_layout is RouterWeightLayout.ATOM
        else pack_bf16(tensors["w_r"])
    )
    return packed
