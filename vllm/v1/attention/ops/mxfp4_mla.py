# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PyTorch reference for the mxfp4_mla KV cache layout.

OCP MXFP4, group 32: E2M1 elements packed two per byte (the low nibble holds
the lower index) and one E8M0 scale per group. A row is the packed data
followed by its scales, ``row_bytes(d) == d // 2 + d // 32``.

Scales round up, ``ceil_pow2(absmax / 6.0)``, as AITER's default does, so the
scaled peak lands in ``(3, 6]`` and never clamps.
"""

from __future__ import annotations

import torch

from vllm.model_executor.layers.quantization.utils.ocp_mx_utils import (
    OCP_MX_BLOCK_SIZE,
)
from vllm.v1.attention.ops.ultraquant.reference import (
    _pack_nibbles_last_dim,
    _unpack_nibbles_last_dim,
)

# E2M1 magnitudes by code index 0..7. Code is ``sign << 3 | index``.
E2M1_VALUES: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E2M1_MAX = 6.0
# Midpoints between consecutive magnitudes, for round-to-nearest via bucketize.
_E2M1_MIDPOINTS: tuple[float, ...] = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)

GROUP_SIZE = OCP_MX_BLOCK_SIZE
E8M0_BIAS = 127
# X = 0 is the E8M0 encoding for a zero scale; the smallest usable exponent is 1.
_E8M0_MIN_X = 1
_E8M0_MAX_X = 254

_LUT_CACHE: dict[tuple[torch.device, torch.dtype], tuple[torch.Tensor, ...]] = {}


def _luts(
    device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    key = (device, dtype)
    if key not in _LUT_CACHE:
        _LUT_CACHE[key] = (
            torch.tensor(_E2M1_MIDPOINTS, device=device, dtype=dtype),
            torch.tensor(E2M1_VALUES, device=device, dtype=dtype),
        )
    return _LUT_CACHE[key]


def ceil_pow2_exponent(v: torch.Tensor) -> torch.Tensor:
    """Smallest integer ``e`` with ``2**e >= v``, computed exactly.

    ``torch.ceil(torch.log2(v))`` is off by one for some exact powers of two
    because ``log2`` is not exactly representable there. ``frexp`` gives
    ``v = m * 2**e`` with ``m`` in ``[0.5, 1)``, so ``v <= 2**e`` always and
    ``v == 2**(e-1)`` exactly when ``m == 0.5``.
    """
    mantissa, exponent = torch.frexp(v)
    return torch.where(mantissa == 0.5, exponent - 1, exponent)


def compute_e8m0_scales(x: torch.Tensor, group: int = GROUP_SIZE) -> torch.Tensor:
    """Per-group E8M0 shared exponents, RoundUp, as ``uint8`` ``X`` bytes.

    ``x`` is ``[..., D]`` with ``D`` divisible by ``group``. Returns
    ``[..., D // group]`` of ``uint8``.
    """
    if x.shape[-1] % group:
        raise ValueError(f"last dim {x.shape[-1]} not divisible by group {group}")
    grouped = x.detach().to(torch.float32).reshape(*x.shape[:-1], -1, group)
    absmax = grouped.abs().amax(dim=-1)

    shared_exp = ceil_pow2_exponent(absmax / E2M1_MAX)
    encoded = (shared_exp + E8M0_BIAS).clamp(_E8M0_MIN_X, _E8M0_MAX_X)
    # An all-zero group has absmax 0; frexp gives exponent 0, and the clamp
    # above would encode 2**-126. The scale is irrelevant there (every element
    # quantizes to the zero code) but pin it to 2**0 so the encoding is
    # deterministic and round-trips bit-exactly.
    return torch.where(absmax == 0, torch.full_like(encoded, E8M0_BIAS), encoded).to(
        torch.uint8
    )


def decode_e8m0_scales(encoded: torch.Tensor) -> torch.Tensor:
    """``X`` bytes -> fp32 scale factors ``2**(X - 127)``."""
    return torch.exp2(encoded.to(torch.float32) - float(E8M0_BIAS))


def quantize_to_e2m1_codes(
    x: torch.Tensor, scales_encoded: torch.Tensor, group: int = GROUP_SIZE
) -> torch.Tensor:
    """Quantize to 4-bit E2M1 codes. Returns ``uint8`` of the same shape as ``x``.

    Codes are ``sign << 3 | magnitude_index``; magnitudes saturate at 6.0.
    """
    scale = decode_e8m0_scales(scales_encoded).unsqueeze(-1)
    grouped = x.detach().to(torch.float32).reshape(*x.shape[:-1], -1, group)
    normalized = grouped / scale

    midpoints, _ = _luts(x.device, torch.float32)
    index = torch.bucketize(normalized.abs(), midpoints).to(torch.uint8)
    sign = (normalized < 0).to(torch.uint8) << 3
    # The sign bit is kept even when the magnitude rounds to zero, so a small
    # negative value encodes as -0.0 (code 8) rather than +0.0 (code 0). Both
    # dequantize to zero and dot_scaled cannot tell them apart, but AITER's
    # reference preserves the sign and matching it bit-for-bit is what lets
    # ``per_1x32_f4_quant`` serve as a strict oracle -- and keeps the door open
    # to using AITER's quantizer as the writer. Note ``-0.0 < 0`` is False, so
    # a negative-zero *input* still encodes as code 0.
    return (index | sign).reshape(x.shape)


def dequantize_e2m1_codes(
    codes: torch.Tensor,
    scales_encoded: torch.Tensor,
    group: int = GROUP_SIZE,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """E2M1 codes + E8M0 scales -> real values."""
    _, values = _luts(codes.device, torch.float32)
    magnitude = values[(codes & 0x07).long()]
    signed = torch.where(codes & 0x08 != 0, -magnitude, magnitude)

    scale = decode_e8m0_scales(scales_encoded).unsqueeze(-1)
    grouped = signed.reshape(*signed.shape[:-1], -1, group)
    return (grouped * scale).reshape(signed.shape).to(out_dtype)


def quantize_pack(
    x: torch.Tensor, group: int = GROUP_SIZE
) -> tuple[torch.Tensor, torch.Tensor]:
    """``[..., D]`` -> ``(packed [..., D // 2], scales [..., D // group])``, uint8."""
    scales = compute_e8m0_scales(x, group)
    codes = quantize_to_e2m1_codes(x, scales, group)
    return _pack_nibbles_last_dim(codes), scales


def unpack_dequantize(
    packed: torch.Tensor,
    scales: torch.Tensor,
    group: int = GROUP_SIZE,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Inverse of :func:`quantize_pack`."""
    codes = _unpack_nibbles_last_dim(packed, 2 * packed.shape[-1])
    return dequantize_e2m1_codes(codes, scales, group, out_dtype)


def quantize_dequantize(x: torch.Tensor, group: int = GROUP_SIZE) -> torch.Tensor:
    """Round-trip through MXFP4 without materializing the packed form.

    This is the fake-quantization path: use it to measure accuracy with an
    unmodified bf16 cache.
    """
    packed, scales = quantize_pack(x, group)
    return unpack_dequantize(packed, scales, group, out_dtype=x.dtype)


def row_bytes(latent_dim: int, group: int = GROUP_SIZE) -> int:
    """Packed row size: data bytes + scale bytes. 272 for a 512-wide latent."""
    if latent_dim % group:
        raise ValueError(f"latent dim {latent_dim} not divisible by group {group}")
    return latent_dim // 2 + latent_dim // group


def scale_region_offset(latent_dim: int) -> int:
    """Byte offset of the scale region within a row. 256 for a 512-wide latent."""
    return latent_dim // 2


def scale_view(cache: torch.Tensor, latent_dim: int, group: int = GROUP_SIZE):
    """A ``uint8`` view of the scale region of every slot in a paged cache.

    ``cache`` is the packed ``uint8`` cache viewed as
    ``[num_blocks, block_size, row_bytes]``. Returns
    ``[num_blocks, block_size, latent_dim // group]`` aliasing the same storage,
    so the read kernel takes two clean pointers instead of doing byte
    arithmetic inside its gather loop.

    This mirrors ``int4_per_token_head``'s approach of carving strided views
    over inline scale padding (see ``triton_attn.py::_ensure_scale_caches``).
    """
    if cache.dtype != torch.uint8:
        raise ValueError(f"expected a uint8 cache, got {cache.dtype}")
    num_blocks, block_size, stride = cache.shape[0], cache.shape[1], cache.shape[2]
    expected = row_bytes(latent_dim, group)
    if stride != expected:
        raise ValueError(f"expected row stride {expected}, got {stride}")
    return torch.as_strided(
        cache,
        size=(num_blocks, block_size, latent_dim // group),
        stride=(cache.stride(0), stride, 1),
        storage_offset=cache.storage_offset() + scale_region_offset(latent_dim),
    )
