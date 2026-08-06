# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numeric helpers for online FP8_ptpc / block FP8 / MXFP8 weight quantization.

Ports the subset of compressed-tensors math used by the online quantize-at-load
path so online and offline exports stay bitwise-aligned, without importing
compressed-tensors APIs from the online path.
"""

from __future__ import annotations

import torch

# e4m3 representable range (matches compressed-tensors FP8_E4M3_DATA).
_FP8_E4M3_MAX = 448.0
_FP8_E4M3_MIN = -448.0

# Element-format offsets for MX scale generation (CT _MX_ELEM_OFFSET).
_MX_ELEM_OFFSET = {4: 2, 8: 8}

# Floating-point layout: (mantissa_bits, exponent_bits).
_FLOAT_LAYOUT: dict[torch.dtype, tuple[int, int]] = {
    torch.bfloat16: (7, 8),
    torch.float16: (10, 5),
    torch.float32: (23, 8),
    torch.float64: (52, 11),
}
# FP4 e2m1 mantissa used by CT round_to_power_2's VAL_TO_ADD.
_FP4_E2M1_MANTISSA = 1


def _symmetric_fp8_scale(
    min_vals: torch.Tensor,
    max_vals: torch.Tensor,
) -> torch.Tensor:
    """Per-group symmetric FP8 scale: ``amax / 448`` (CT calculate_qparams)."""
    # Ensure 0 is representable in the quantized range.
    min_vals = torch.min(min_vals, torch.zeros_like(min_vals))
    max_vals = torch.max(max_vals, torch.zeros_like(max_vals))
    amax = torch.max(torch.abs(min_vals), torch.abs(max_vals))
    scales = amax / _FP8_E4M3_MAX
    # Replace zeros so division by scale never NaNs (CT dtype eps).
    eps = torch.finfo(scales.dtype).eps
    return torch.where(
        scales == 0,
        torch.tensor(eps, dtype=scales.dtype, device=scales.device),
        scales,
    )


def _quantize_fp8(
    x: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Divide by scale, clamp to e4m3 range, cast (CT ``_quantize`` for FP8)."""
    scaled = x / scale
    return torch.clamp(scaled, _FP8_E4M3_MIN, _FP8_E4M3_MAX).to(dtype)


def _round_to_power_2(x: torch.Tensor) -> torch.Tensor:
    """Round each value to the closest power of 2 (CT ``round_to_power_2``)."""
    scale_dtype = x.dtype
    if scale_dtype not in _FLOAT_LAYOUT:
        raise TypeError(f"Unsupported dtype {scale_dtype}")
    mantissa, exponent = _FLOAT_LAYOUT[scale_dtype]
    if scale_dtype is torch.float64:
        int_dtype: torch.dtype = torch.uint64
        work = x.view(int_dtype).to(torch.int64)
    elif scale_dtype is torch.float32:
        int_dtype = torch.uint32
        work = x.view(int_dtype).to(torch.int32)
    else:
        int_dtype = torch.uint16
        work = x.view(int_dtype).to(torch.int32)

    val_to_add = 1 << (mantissa - _FP4_E2M1_MANTISSA - 1)
    sign_exponent_mask = ((1 << (exponent + 1)) - 1) << mantissa
    block_max_uint = torch.bitwise_and(work + val_to_add, sign_exponent_mask)
    if int_dtype is torch.uint16:
        return block_max_uint.to(int_dtype).view(scale_dtype)
    return block_max_uint.view(scale_dtype)


def _generate_mx_scales(amax: torch.Tensor, num_bits: int = 8) -> torch.Tensor:
    """E8M0-biased MX scales from per-group amax (CT ``generate_mx_scales``)."""
    offset = _MX_ELEM_OFFSET[num_bits]
    scale_power_2 = _round_to_power_2(amax)
    return 127 + torch.floor(torch.log2(scale_power_2)) - offset


def _round_to_uint8(x: torch.Tensor) -> torch.Tensor:
    """Clamp/round to uint8 (CT ``round_to_quantized_type_dtype``)."""
    iinfo = torch.iinfo(torch.uint8)
    return torch.round(torch.clamp(x, iinfo.min, iinfo.max)).to(torch.uint8)
