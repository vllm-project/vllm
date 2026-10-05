# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 FlyDSL Project Contributors
# mypy: ignore-errors
#
# This file contains code copied from FlyDSL (ROCm/FlyDSL PR #1204 at 21a3d1ee), as vendored by
# ROCm/ATOM PR #2435 (head 45e4b55d, atom/model_ops/monokernel/formats.py). The original source code was
# licensed under the Apache License 2.0 and included the following copyright notice:
# Copyright (c) 2026 FlyDSL Project Contributors
# Modified by the vLLM project contributors (Apache-2.0 sec. 4(b)): import paths rewritten to this package.

"""Torch-side MXFP4 and MXFP8 conversion helpers shared by MoE wrappers."""

from __future__ import annotations

import torch


def float_to_e8m0(x: torch.Tensor) -> torch.Tensor:
    """Round positive FP32 values to E8M0 exponent bytes."""

    bits = x.float().contiguous().view(torch.int32)
    exponent = ((bits >> 23) & 0xFF).to(torch.uint8)
    round_up = ((bits & 0x400000) != 0) & (
        ((bits & 0x200000) != 0) | ((bits & 0x1FFFFF) != 0) | (exponent != 0)
    )
    exponent = exponent + round_up.to(torch.uint8)
    return torch.where(exponent == 0xFF, torch.full_like(exponent, 0xFF), exponent)


def e8m0_to_float(scale: torch.Tensor) -> torch.Tensor:
    """Decode E8M0 exponent bytes to FP32 power-of-two scales."""

    exponent = scale.view(torch.uint8)
    bits = exponent.to(torch.int32) << 23
    bits = torch.where(exponent == 0, torch.full_like(bits, 0x00400000), bits)
    bits = torch.where(exponent == 0xFF, torch.full_like(bits, 0x7F800001), bits)
    return bits.view(torch.float32)


def _float_to_fp4_codes(x: torch.Tensor) -> torch.Tensor:
    """Round finite FP32 values to E2M1 codes, with ties to even."""

    magnitude = x.float().abs().clamp(max=6.0)
    boundaries = torch.tensor((0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0), device=x.device)
    code = torch.bucketize(magnitude, boundaries, right=False).to(torch.uint8)
    # At these midpoints the upper code has an even mantissa bit.
    code = code + ((magnitude == 0.75) | (magnitude == 1.75) | (magnitude == 3.5)).to(
        torch.uint8
    )
    return code | ((x < 0).to(torch.uint8) << 3)


def quantize_mxfp4(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``[..., K]`` to packed MXFP4 plus row-major per-1x32 E8M0 scales."""

    if w.shape[-1] % 32:
        raise ValueError(
            f"MXFP4 K dimension must be divisible by 32, got {w.shape[-1]}"
        )
    shape = w.shape
    blocks = w.float().reshape(*shape[:-1], shape[-1] // 32, 32)
    amax = blocks.abs().amax(dim=-1)
    scale = float_to_e8m0(amax / 4.0)
    scale_f32 = e8m0_to_float(scale).clamp_min(torch.finfo(torch.float32).tiny)
    codes = _float_to_fp4_codes(blocks / scale_f32.unsqueeze(-1)).reshape(shape)
    packed = (codes[..., 0::2] | (codes[..., 1::2] << 4)).contiguous()
    return packed, scale.contiguous()


def dequantize_mxfp4(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Decode row-major packed MXFP4 values and per-1x32 E8M0 scales."""

    q = q.view(torch.uint8)
    codes = q.repeat_interleave(2, dim=-1)
    codes[..., 0::2] &= 0xF
    codes[..., 1::2] >>= 4
    values = torch.tensor(
        (
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ),
        dtype=torch.float32,
        device=q.device,
    )
    return values[codes.long()] * e8m0_to_float(scale).repeat_interleave(32, dim=-1)


def quant_dequant_mxfp8(x: torch.Tensor) -> torch.Tensor:
    """Per-1x32 MXFP8 E4M3 quantization, returned dequantized in FP32."""

    if x.shape[-1] % 32:
        raise ValueError(
            f"MXFP8 K dimension must be divisible by 32, got {x.shape[-1]}"
        )
    shape = x.shape
    blocks = x.float().reshape(*shape[:-1], shape[-1] // 32, 32)
    amax = blocks.abs().amax(dim=-1)
    scale = float_to_e8m0(amax / 448.0)
    scale_f32 = e8m0_to_float(scale).clamp_min(torch.finfo(torch.float32).tiny)
    q = (
        (blocks / scale_f32.unsqueeze(-1))
        .clamp(-448.0, 448.0)
        .to(torch.float8_e4m3fn)
        .float()
    )
    return (q * scale_f32.unsqueeze(-1)).reshape(shape)


def quantize_mxfp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``[..., K]`` to FP8 values plus row-major per-1x32 E8M0 scales."""

    if x.shape[-1] % 32:
        raise ValueError(
            f"MXFP8 K dimension must be divisible by 32, got {x.shape[-1]}"
        )
    shape = x.shape
    blocks = x.float().reshape(*shape[:-1], shape[-1] // 32, 32)
    amax = blocks.abs().amax(dim=-1)
    scale = float_to_e8m0(amax / 448.0)
    scale_f32 = e8m0_to_float(scale).clamp_min(torch.finfo(torch.float32).tiny)
    quantized = (
        (blocks / scale_f32.unsqueeze(-1)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    )
    return quantized.reshape(shape).contiguous(), scale.contiguous()


def dequantize_mxfp8(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Decode row-major MXFP8 values and per-1x32 E8M0 scales."""

    if q.shape[-1] % 32:
        raise ValueError(
            f"MXFP8 K dimension must be divisible by 32, got {q.shape[-1]}"
        )
    expected_scale_shape = (*q.shape[:-1], q.shape[-1] // 32)
    if scale.shape != expected_scale_shape:
        raise ValueError(
            f"MXFP8 scale shape must be {expected_scale_shape}, got {tuple(scale.shape)}"
        )
    return q.float() * e8m0_to_float(scale).repeat_interleave(32, dim=-1)
