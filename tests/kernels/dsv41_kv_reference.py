# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Torch ports of FlashMLA ``tests/quant.py`` for the DeepSeek V4.1 KV formats.

V4.1 fp8 (528 B/token): 512 e4m3 values (RoPE dims quantized too) in 16 tiles
of 32 with a ue8m0 scale ``2**ceil(log2(clamp_min(amax / 448, 1e-4)))`` each.
V4.1 fp4 (288 B/token): 512 e2m1 values packed two per byte (even element in
the low nibble) in 32 tiles of 16 with an e4m3 scale ``clamp(amax / 6, 2^-9,
448)`` each; the e2m1 x e4m3 product is exact in bf16.
"""

import torch

_E2M1_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def quantize_v41_fp8(k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``k [T, 512]`` -> (values uint8 ``[T, 512]``, ue8m0 scales ``[T, 16]``)."""
    tiles = k.float().reshape(-1, 16, 32)
    scale_inv = (tiles.abs().amax(-1) / 448.0).clamp_min(1e-4)
    exponent = torch.ceil(torch.log2(scale_inv))
    values = (tiles / torch.exp2(exponent).unsqueeze(-1)).clamp(-448.0, 448.0)
    values = values.to(torch.float8_e4m3fn).reshape(-1, 512).view(torch.uint8)
    return values, (exponent + 127).to(torch.uint8)


def dequantize_v41_fp8(values: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    scale = torch.exp2(scales.float() - 127.0)
    v = values.view(torch.float8_e4m3fn).float().reshape(-1, 16, 32)
    return (v * scale.unsqueeze(-1)).reshape(-1, 512).to(torch.bfloat16)


def quantize_e2m1_codes(x: torch.Tensor) -> torch.Tensor:
    """Round-to-nearest-even e2m1 codes (sign in bit 3), saturating at 6."""
    values = _E2M1_VALUES.to(x.device)
    mag = x.abs().clamp(max=6.0)
    # values[idx - 1] < mag <= values[idx]
    idx = torch.bucketize(mag, values).clamp(max=7)
    lo_idx = (idx - 1).clamp(min=0)
    lo, hi = values[lo_idx], values[idx]
    d_lo, d_hi = mag - lo, hi - mag
    pick_hi = (d_hi < d_lo) | ((d_hi == d_lo) & (idx % 2 == 0))
    code = torch.where(pick_hi, idx, lo_idx)
    code = torch.where(mag == 0, torch.zeros_like(code), code)
    return (code + 8 * (x < 0).to(code.dtype)).to(torch.uint8)


def quantize_v41_fp4(k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``k [T, 512]`` -> (packed e2m1 uint8 ``[T, 256]``, e4m3 scales ``[T, 32]``)."""
    tiles = k.float().reshape(-1, 32, 16)
    amax = tiles.abs().amax(-1)
    scale = (amax / 6.0).clamp(2.0**-9, 448.0).to(torch.float8_e4m3fn)
    codes = quantize_e2m1_codes(tiles / scale.float().unsqueeze(-1))
    codes = codes.reshape(-1, 256, 2)
    packed = codes[..., 0] | (codes[..., 1] << 4)
    return packed.to(torch.uint8), scale.view(torch.uint8)


def dequantize_v41_fp4(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    lo, hi = packed & 0xF, packed >> 4
    codes = torch.stack([lo, hi], -1).reshape(-1, 512).long()
    mag = _E2M1_VALUES.to(packed.device)[codes & 7]
    vals = torch.where(codes >= 8, -mag, mag).reshape(-1, 32, 16)
    scale = scales.view(torch.float8_e4m3fn).float()
    return (vals * scale.unsqueeze(-1)).reshape(-1, 512).to(torch.bfloat16)
