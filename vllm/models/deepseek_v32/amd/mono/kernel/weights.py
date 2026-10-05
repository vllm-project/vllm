# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 FlyDSL Project Contributors
# mypy: ignore-errors
#
# This file contains code copied from FlyDSL (ROCm/FlyDSL PR #1204 at 21a3d1ee), as vendored by
# ROCm/ATOM PR #2435 (head 45e4b55d, atom/model_ops/monokernel/weights.py). The original source code was
# licensed under the Apache License 2.0 and included the following copyright notice:
# Copyright (c) 2026 FlyDSL Project Contributors
# Modified by the vLLM project contributors (Apache-2.0 sec. 4(b)): import paths rewritten to this package;
#   MonoUnsupported defined here (ATOM's mode selector, dispatch.py, is not vendored).

"""Shared host-side weight container for model-specific MonoKernels."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm.models.deepseek_v32.amd.mono.kernel.config import (
    GLM5_CONFIG,
    LayerConfig,
    Mxfp4ScaleLayout,
    Mxfp4WeightLayout,
)
from vllm.models.deepseek_v32.amd.mono.kernel.formats import dequantize_mxfp4


class MonoUnsupported(Exception):
    """The loaded weights cannot be used by the MonoKernel."""


def _need(ok: bool, what: str) -> None:
    if not ok:
        raise MonoUnsupported(what)


def _unshuffle_linear_weight(weight: torch.Tensor) -> torch.Tensor:
    """Invert AITER's MI355 16x16 weight preshuffle."""

    if not getattr(weight, "is_shuffled", False):
        return weight.contiguous()
    source_dtype = weight.dtype
    raw = weight if weight.dtype == torch.uint8 else weight.view(torch.uint8)
    *lead, rows, packed_k = raw.shape
    lane_k = 16 // raw.element_size()
    block_k = 32
    shuffled = raw.reshape(
        *lead,
        rows // 16,
        packed_k // block_k,
        block_k // lane_k,
        16,
        lane_k,
    )
    nlead = len(lead)
    order = list(range(nlead)) + [nlead, nlead + 3, nlead + 1, nlead + 2, nlead + 4]
    return (
        shuffled.permute(*order)
        .contiguous()
        .reshape(*lead, rows, packed_k)
        .view(source_dtype)
    )


def _unshuffle_linear_scale(
    scale: torch.Tensor,
    *,
    experts: int,
    rows: int,
    groups: int,
) -> torch.Tensor:
    """Invert the non-GUGU E8M0 scale layout used by FlyDSL linears."""

    flat_rows = experts * rows
    padded_rows = (flat_rows + 255) // 256 * 256
    padded_groups = (groups + 7) // 8 * 8
    _need(
        scale.numel() == padded_rows * padded_groups,
        f"linear scale bytes {scale.numel()} != {padded_rows * padded_groups}",
    )
    packed = scale.reshape(
        padded_rows // 32,
        padded_groups // 8,
        4,
        16,
        2,
        2,
    )
    native = packed.permute(0, 5, 3, 1, 4, 2).contiguous()
    return native.reshape(padded_rows, padded_groups)[:flat_rows, :groups].reshape(
        experts, rows, groups
    )


def linear_ptpc_fp8(
    linear,
    *,
    name: str,
    logical_rows: int,
    logical_cols: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a preshuffled PTPC FP8 weight and its per-output-channel scales."""

    weight = linear.weight
    scale = getattr(linear, "weight_scale", None)
    fp8_dtypes = tuple(
        dtype
        for dtype in (
            getattr(torch, "float8_e4m3fn", None),
            getattr(torch, "float8_e4m3fnuz", None),
        )
        if dtype is not None
    )
    _need(
        getattr(getattr(linear, "quant_type", None), "name", None) == "per_Token",
        f"{name} must use per-token FP8",
    )
    _need(
        getattr(linear, "params_dtype", None) in fp8_dtypes
        and weight.dtype == getattr(linear, "params_dtype", None),
        f"{name} must use E4M3 FP8 weights",
    )
    _need(
        bool(getattr(weight, "is_shuffled", False)),
        f"{name} FP8 weight must use the AITER preshuffle",
    )
    _need(
        weight.ndim == 2
        and weight.shape[0] >= logical_rows
        and weight.shape[1] == logical_cols,
        f"{name} weight shape {tuple(weight.shape)}",
    )
    _need(
        not getattr(linear, "is_output_padded", False)
        or getattr(linear, "_output_size_before_padding", None) == logical_rows,
        f"{name} output padding does not preserve {logical_rows} logical rows",
    )
    _need(
        scale is not None
        and scale.dtype == torch.float32
        and scale.numel() == weight.shape[0],
        f"{name} per-output-channel scale",
    )
    return weight.data, scale.data.view(-1)


def linear_bf16(
    linear,
    *,
    name: str,
    logical_rows: int,
    logical_cols: int,
    row_start: int = 0,
    row_count: int | None = None,
) -> torch.Tensor:
    """Return one logical linear matrix as contiguous BF16 storage."""

    row_count = logical_rows - row_start if row_count is None else row_count
    _need(
        getattr(linear, "input_size", None) == logical_cols,
        f"{name} declares input_size={getattr(linear, 'input_size', None)}, expected {logical_cols}",
    )
    _need(
        0 <= row_start < logical_rows and 0 < row_count <= logical_rows - row_start,
        f"{name} row slice [{row_start}, {row_start + row_count}) is outside {logical_rows} rows",
    )

    weight = linear.weight
    quant_name = getattr(getattr(linear, "quant_type", None), "name", None)
    params_dtype = getattr(linear, "params_dtype", None)
    shuffled = bool(getattr(weight, "is_shuffled", False))
    if quant_name == "No":
        _need(
            params_dtype == torch.bfloat16,
            f"{name} unquantized params must be BF16, got {params_dtype}",
        )
        _need(not shuffled, f"{name} BF16 weight must not be preshuffled")
        _need(
            weight.dtype == torch.bfloat16,
            f"{name} BF16 weight has dtype {weight.dtype}",
        )
        _need(
            weight.shape == (logical_rows, logical_cols),
            f"{name} weight shape {tuple(weight.shape)}",
        )
        _need(weight.is_contiguous(), f"{name} BF16 weight must be contiguous")
        if row_start == 0 and row_count == logical_rows:
            return weight
        return weight.narrow(0, row_start, row_count).contiguous()

    fp8_dtypes = tuple(
        dtype
        for dtype in (
            getattr(torch, "float8_e4m3fn", None),
            getattr(torch, "float8_e4m3fnuz", None),
        )
        if dtype is not None
    )
    if quant_name == "per_Token":
        _need(
            params_dtype in fp8_dtypes,
            f"{name} per_Token params must be E4M3 FP8, got {params_dtype}",
        )
        _need(
            weight.dtype == params_dtype,
            f"{name} FP8 weight has dtype {weight.dtype}, expected {params_dtype}",
        )
        padded = bool(getattr(linear, "is_output_padded", False))
        storage_rows = weight.shape[0] if weight.ndim == 2 else 0
        if padded:
            _need(
                getattr(linear, "_output_size_before_padding", None) == logical_rows,
                f"{name} padded logical rows do not match {logical_rows}",
            )
            _need(
                storage_rows >= logical_rows,
                f"{name} padded FP8 rows {storage_rows} < logical rows {logical_rows}",
            )
        else:
            _need(
                storage_rows == logical_rows,
                f"{name} FP8 rows {storage_rows}, expected {logical_rows}",
            )
        _need(
            weight.shape == (storage_rows, logical_cols),
            f"{name} FP8 weight shape {tuple(weight.shape)}",
        )
        scale = getattr(linear, "weight_scale", None)
        _need(scale is not None, f"{name} per_Token weight_scale is missing")
        _need(
            scale.shape == (storage_rows, 1),
            f"{name} weight_scale shape {tuple(scale.shape)}, expected {(storage_rows, 1)}",
        )
        _need(
            scale.dtype == torch.float32,
            f"{name} weight_scale must be FP32, got {scale.dtype}",
        )
        if shuffled:
            _need(
                storage_rows % 16 == 0 and logical_cols % 32 == 0,
                f"{name} preshuffled FP8 shape must align to (16, 32), got {(storage_rows, logical_cols)}",
            )
            weight = _unshuffle_linear_weight(weight)
        weight = weight.narrow(0, row_start, row_count)
        scale = scale.narrow(0, row_start, row_count)
        return (weight.float() * scale.float()).to(torch.bfloat16).contiguous()

    fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
    _need(
        quant_name == "per_1x32"
        and fp4_dtype is not None
        and params_dtype == fp4_dtype,
        f"{name} unsupported quantization: quant_type={quant_name}, params_dtype={params_dtype}",
    )
    _need(shuffled, f"{name} per_1x32 FP4 weight must be preshuffled")
    _need(
        logical_cols % 32 == 0,
        f"{name} logical K={logical_cols} must be divisible by 32",
    )
    expected_weight = (logical_rows, logical_cols // 2)
    _need(
        weight.dtype == fp4_dtype,
        f"{name} packed weight has dtype {weight.dtype}, expected {fp4_dtype}",
    )
    _need(
        weight.shape == expected_weight,
        f"{name} packed weight shape {tuple(weight.shape)}, expected {expected_weight}",
    )

    scale = getattr(linear, "weight_scale", None)
    groups = logical_cols // 32
    padded_rows = (logical_rows + 255) // 256 * 256
    padded_groups = (groups + 7) // 8 * 8
    expected_scale_shape = (padded_rows, padded_groups)
    _need(scale is not None, f"{name} per_1x32 weight_scale is missing")
    _need(
        scale.element_size() == 1, f"{name} weight_scale must use one-byte E8M0 values"
    )
    _need(
        scale.shape == expected_scale_shape,
        f"{name} weight_scale shape {tuple(scale.shape)}, expected {expected_scale_shape}",
    )
    packed = _unshuffle_linear_weight(weight).narrow(0, row_start, row_count)
    native_scale = _unshuffle_linear_scale(
        scale, experts=1, rows=logical_rows, groups=groups
    )[0]
    native_scale = native_scale.narrow(0, row_start, row_count)
    result = dequantize_mxfp4(packed, native_scale).to(torch.bfloat16).contiguous()
    _need(
        result.shape == (row_count, logical_cols),
        f"{name} dequantized shape {tuple(result.shape)}, expected {(row_count, logical_cols)}",
    )
    return result


@dataclass
class LayerWeights:
    """One tensor-parallel rank's weights and model geometry."""

    heads: int
    t: dict[str, torch.Tensor]
    config: LayerConfig = GLM5_CONFIG
    rank: int = 0
    npes: int = 1
    mxfp4_weight_layout: Mxfp4WeightLayout = Mxfp4WeightLayout.NATIVE
    mxfp4_scale_layout: Mxfp4ScaleLayout = Mxfp4ScaleLayout.NATIVE
    physical_experts: int | None = None


def atom_mxfp4_storage_view(
    tensor: torch.Tensor,
    *,
    name: str,
    logical_rows: int,
    logical_k: int,
    scale: bool,
) -> torch.Tensor:
    if logical_rows <= 0 or logical_k <= 0 or logical_k % 32:
        raise ValueError(f"{name} invalid logical shape [{logical_rows}, {logical_k}]")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} ATOM storage must be contiguous")
    if tensor.element_size() != 1:
        raise ValueError(f"{name} ATOM storage must use one-byte values")
    if scale:
        groups = logical_k // 32
        padded_rows = (logical_rows + 255) // 256 * 256
        padded_groups = (groups + 7) // 8 * 8
        expected_bytes = padded_rows * padded_groups
    else:
        if not getattr(tensor, "is_shuffled", False):
            raise ValueError(f"{name} ATOM weight must carry the preshuffled marker")
        expected_bytes = logical_rows * logical_k // 2
    if tensor.numel() != expected_bytes:
        raise ValueError(
            f"{name} has {tensor.numel()} bytes, expected {expected_bytes}"
        )
    return tensor.view(torch.uint8).view(-1)


def prepare_mxfp4_expert_storage(
    weights: LayerWeights, *, canonical: bool = True
) -> tuple[torch.Tensor, ...]:
    config = weights.config
    tensors = weights.t
    experts = (
        config.n_experts
        if weights.physical_experts is None
        else weights.physical_experts
    )
    expert_hidden = (
        config.hidden if config.routed_hidden is None else config.routed_hidden
    )
    ug_rows = 2 * config.inter
    dn_rows = expert_hidden

    if not canonical:
        if (
            weights.mxfp4_weight_layout is not Mxfp4WeightLayout.ATOM
            or weights.mxfp4_scale_layout is not Mxfp4ScaleLayout.ATOM
        ):
            raise ValueError("zero-copy MXFP4 storage requires ATOM values and scales")
        return (
            atom_mxfp4_storage_view(
                tensors["w_ug"],
                name="w_ug",
                logical_rows=experts * ug_rows,
                logical_k=expert_hidden,
                scale=False,
            ),
            atom_mxfp4_storage_view(
                tensors["s_ug"],
                name="s_ug",
                logical_rows=experts * ug_rows,
                logical_k=expert_hidden,
                scale=True,
            ),
            atom_mxfp4_storage_view(
                tensors["w_dn"],
                name="w_dn",
                logical_rows=experts * dn_rows,
                logical_k=config.inter,
                scale=False,
            ),
            atom_mxfp4_storage_view(
                tensors["s_dn"],
                name="s_dn",
                logical_rows=experts * dn_rows,
                logical_k=config.inter,
                scale=True,
            ),
        )

    from vllm.models.deepseek_v32.amd.mono.kernel.packing import pack_mxfp4

    def values(name: str, rows: int, k: int) -> torch.Tensor:
        tensor = tensors[name]
        if weights.mxfp4_weight_layout is Mxfp4WeightLayout.ATOM:
            packed = atom_mxfp4_storage_view(
                tensor, name=name, logical_rows=experts * rows, logical_k=k, scale=False
            )
            shuffled = packed.view(experts, rows, k // 2)
            shuffled.is_shuffled = True
            tensor = _unshuffle_linear_weight(shuffled)
        elif weights.mxfp4_weight_layout is not Mxfp4WeightLayout.NATIVE:
            raise ValueError(
                f"unsupported MXFP4 weight layout {weights.mxfp4_weight_layout!r}"
            )
        return pack_mxfp4(tensor)

    def scales(name: str, rows: int, k: int) -> torch.Tensor:
        tensor = tensors[name]
        groups = k // 32
        if weights.mxfp4_scale_layout is Mxfp4ScaleLayout.ATOM:
            packed = atom_mxfp4_storage_view(
                tensor, name=name, logical_rows=experts * rows, logical_k=k, scale=True
            )
            tensor = _unshuffle_linear_scale(
                packed, experts=experts, rows=rows, groups=groups
            )
        elif weights.mxfp4_scale_layout is not Mxfp4ScaleLayout.NATIVE:
            raise ValueError(
                f"unsupported MXFP4 scale layout {weights.mxfp4_scale_layout!r}"
            )
        return tensor.view(torch.uint8).contiguous().view(-1)

    return (
        values("w_ug", ug_rows, expert_hidden),
        scales("s_ug", ug_rows, expert_hidden),
        values("w_dn", dn_rows, config.inter),
        scales("s_dn", dn_rows, config.inter),
    )


def prepare_aiter_mxfp4_expert_storage(
    weights: LayerWeights,
) -> tuple[torch.Tensor, ...]:
    if (
        weights.mxfp4_weight_layout is Mxfp4WeightLayout.ATOM
        and weights.mxfp4_scale_layout is Mxfp4ScaleLayout.ATOM
    ):
        return prepare_mxfp4_expert_storage(weights, canonical=False)
    if (
        weights.mxfp4_weight_layout is Mxfp4WeightLayout.NATIVE
        and weights.mxfp4_scale_layout is Mxfp4ScaleLayout.NATIVE
    ):
        from vllm.models.deepseek_v32.amd.mono.kernel.packing import (
            pack_a16w4_scale,
            pack_a16w4_weight,
        )

        tensors = weights.t
        return (
            pack_a16w4_weight(tensors["w_ug"]),
            pack_a16w4_scale(tensors["s_ug"]),
            pack_a16w4_weight(tensors["w_dn"]),
            pack_a16w4_scale(tensors["s_dn"]),
        )
    raise ValueError("AITER MXFP4 storage requires matching value/scale layouts")


__all__ = [
    "LayerWeights",
    "atom_mxfp4_storage_view",
    "linear_bf16",
    "prepare_aiter_mxfp4_expert_storage",
    "prepare_mxfp4_expert_storage",
]
