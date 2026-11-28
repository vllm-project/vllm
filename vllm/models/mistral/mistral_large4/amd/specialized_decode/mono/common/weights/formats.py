# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
# Copyright (c) 2026 FlyDSL Project Contributors

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
