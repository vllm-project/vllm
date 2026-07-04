# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
DeepSeek-V4 activation quantization helpers (PR1 torch fallbacks).

Three ops the V4 reference inference uses for Quantization-Aware Training (QAT)
simulation on activations:

    act_quant_inplace        — block-wise FP8 e4m3 round-trip (BF16 -> FP8 -> BF16)
    fp4_act_quant_inplace    — block-wise FP4 e2m1 round-trip (BF16 -> FP4 -> BF16)
    rotate_activation        — Walsh-Hadamard transform with 1/sqrt(N) scaling

The reference TileLang kernels live in /data/DeepSeek-V4-Pro/inference/kernel.py.
PR1 ships pure-torch fallbacks for numerical correctness; production-perf paths
land alongside the AITER `sparse_attn` kernel in PR4.

These ops do NOT change tensor shape or dtype — they round-trip the values
in-place to simulate the precision loss of low-bit storage.
"""

from typing import Optional

import torch

# FP4 e2m1 representable magnitudes (positive half). Symmetric around 0.
# Reference: /data/DeepSeek-V4-Pro/inference/convert.py:11-14
_FP4_MAGNITUDES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
)

# FP4 e2m1 full lookup table (16 entries: low nibble 0..7 = positive,
# low nibble 8..15 = negative). Matches convert.py:11-14 exactly.
_FP4_LOOKUP = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)


def dequant_fp4_e2m1(
    packed: torch.Tensor,
    scale: torch.Tensor,
    fp4_block_size: int = 32,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize packed FP4 e2m1 weights with per-block ue8m0 scale.

    On-disk format used by DeepSeek-V4-Pro expert weights:
      - `packed`: int8 [..., out, in/2]. Each byte holds 2 FP4 values:
            byte = (high_nibble << 4) | low_nibble
            position 2*j   ← FP4_LOOKUP[low_nibble]
            position 2*j+1 ← FP4_LOOKUP[high_nibble]
      - `scale`: float8_e8m0fnu [..., out, in/fp4_block_size]. Power-of-2
            scaling factor for each contiguous block of `fp4_block_size`
            values along the input dim.

    Reference: convert.py:cast_e2m1fn_to_e4m3fn (lines 17-52). The first
    half of that function does the unpack; we then apply the per-block
    scale directly to BF16 instead of repacking into FP8 e4m3.

    Args:
        packed: int8 tensor with shape [..., out, in/2]
        scale: float8_e8m0fnu (or any float) tensor with shape [..., out, in/fp4_block_size]
        fp4_block_size: scaling block size along input dim (default 32)
        out_dtype: dtype to return (default bfloat16)

    Returns:
        Dequantized tensor with shape [..., out, in], dtype=out_dtype.
    """
    assert packed.dtype == torch.int8, f"packed must be int8, got {packed.dtype}"
    assert packed.dim() >= 2, f"packed must be ≥2D, got shape {packed.shape}"

    *prefix, out_dim, in_packed = packed.shape
    in_dim = in_packed * 2
    assert (
        in_dim % fp4_block_size == 0
    ), f"unpacked in_dim {in_dim} not divisible by fp4_block_size {fp4_block_size}"
    expected_scale = (*prefix, out_dim, in_dim // fp4_block_size)
    assert (
        tuple(scale.shape) == expected_scale
    ), f"scale shape {tuple(scale.shape)} != expected {expected_scale}"

    # Unpack: each byte → 2 FP4 values via lookup table.
    table = _FP4_LOOKUP.to(packed.device)  # [16] FP32
    u = packed.view(torch.uint8)
    low = (u & 0x0F).long()  # [..., out, in/2]
    high = ((u >> 4) & 0x0F).long()
    # Stack so adjacent positions are (low, high), then flatten the trailing pair.
    unpacked = torch.stack([table[low], table[high]], dim=-1)  # [..., out, in/2, 2]
    unpacked = unpacked.reshape(*prefix, out_dim, in_dim)  # [..., out, in]

    # Apply per-block scale. ue8m0 cast to float gives the linear scale value
    # (since float8_e8m0fnu represents pure powers of 2).
    s = scale.float()  # [..., out, in/block]
    s_expanded = s.repeat_interleave(fp4_block_size, dim=-1)  # [..., out, in]
    dequant = unpacked * s_expanded

    return dequant.to(out_dtype)


def act_quant_inplace(
    x: torch.Tensor, block_size: int = 128, scale_fmt: Optional[str] = None
) -> None:
    """In-place BF16 -> FP8 e4m3 -> BF16 round-trip, blocked along the last dim.

    Reference: inference/kernel.py:act_quant with `inplace=True`.

    Args:
        x:          tensor to quantize in-place; last dim must be divisible by block_size
        block_size: number of elements per scaling block (typical: 64 or 128)
        scale_fmt:  None         -> FP32 scale (no special rounding)
                    "ue8m0"      -> round scale UP to nearest power of 2 (MXFP-style)
    """
    fp8_max = 448.0
    fp8_max_inv = 1.0 / fp8_max

    *prefix, n = x.shape
    assert n % block_size == 0, f"last dim {n} not divisible by block_size {block_size}"

    blocks = x.reshape(*prefix, n // block_size, block_size).float()
    amax = blocks.abs().amax(dim=-1, keepdim=True).clamp(min=1e-4)
    scale = amax * fp8_max_inv
    if scale_fmt == "ue8m0":
        # Match reference (ref_full_generate / aiter): round-to-even via
        # f32_to_e8m0 + e8m0_to_f32. Earlier `ceil(log2(scale))` matched the
        # TileLang reference but differed from the aiter-backed ref_full_generate
        # path by up to 1 binade — see notes/17_root_cause_input_quant.md.
        from aiter.utility import fp4_utils as _fp4u

        e8m0 = _fp4u.f32_to_e8m0(scale.contiguous())
        scale = _fp4u.e8m0_to_f32(e8m0)

    # Quantize -> FP8 -> dequantize
    quant_fp8 = (
        (blocks / scale).clamp(min=-fp8_max, max=fp8_max).to(torch.float8_e4m3fn)
    )
    dequant = quant_fp8.float() * scale

    x.copy_(dequant.reshape(*prefix, n).to(x.dtype))


def fp4_act_quant_inplace(x: torch.Tensor, block_size: int = 32) -> None:
    """In-place BF16 -> FP4 e2m1 -> BF16 round-trip with per-block ue8m0 scale.

    Reference: inference/kernel.py:fp4_act_quant with `inplace=True`. FP4 e2m1
    representable magnitudes (positive half) are {0, 0.5, 1, 1.5, 2, 3, 4, 6};
    we snap each value to the nearest representable point after scale-down.

    Args:
        x:          tensor to quantize in-place; last dim must be divisible by block_size
        block_size: number of elements per scaling block (default 32 for FP4)
    """
    fp4_max = 6.0
    fp4_max_inv = 1.0 / fp4_max
    eps_amax = 6.0 * (2.0**-126)  # matches reference's clamp floor

    *prefix, n = x.shape
    assert n % block_size == 0, f"last dim {n} not divisible by block_size {block_size}"

    blocks = x.reshape(*prefix, n // block_size, block_size).float()
    amax = blocks.abs().amax(dim=-1, keepdim=True).clamp(min=eps_amax)
    # ue8m0 scale: round up to nearest power of 2
    scale = torch.pow(2.0, torch.ceil(torch.log2(amax * fp4_max_inv)))

    normalized = (blocks / scale).clamp(min=-fp4_max, max=fp4_max)

    # Snap abs(normalized) to nearest representable FP4 magnitude.
    fp4_vals = _FP4_MAGNITUDES.to(normalized.device)  # [8]
    diff = (normalized.abs().unsqueeze(-1) - fp4_vals).abs()  # [..., 8]
    snapped_mag = fp4_vals[diff.argmin(dim=-1)]  # [...]
    snapped = torch.where(normalized < 0, -snapped_mag, snapped_mag)

    dequant = snapped * scale
    x.copy_(dequant.reshape(*prefix, n).to(x.dtype))


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    """Apply Walsh-Hadamard transform along last dim with 1/sqrt(N) scaling.

    Reference: inference/model.py:rotate_activation, which delegates to the
    `fast_hadamard_transform` package. We provide a pure-torch fallback since
    that package fails to build on AMD ROCm.

    Iterative radix-2 butterfly (FFT-style): O(N log N) ops, log2(N) passes.
    For each pass `h` = 1, 2, 4, ..., N/2: pair (x[k+j], x[k+j+h]) becomes
    (a+b, a-b). After all passes, multiply by 1/sqrt(N) for normalization.

    Args:
        x: tensor whose last dim is a power of 2 (typically 128 or 512)
    Returns:
        Hadamard-transformed tensor, same shape and dtype as x
    """
    n = x.shape[-1]
    assert n > 0 and (n & (n - 1)) == 0, f"last dim {n} must be a power of 2"

    orig_dtype = x.dtype
    *prefix, _ = x.shape
    flat = x.reshape(-1, n).float().contiguous()

    h = 1
    while h < n:
        # Group consecutive 2h-element segments; pair element j with element j+h.
        view = flat.view(-1, n // (2 * h), 2, h)
        a = view[..., 0, :]
        b = view[..., 1, :]
        flat = torch.stack([a + b, a - b], dim=-2).reshape(-1, n)
        h *= 2

    flat = flat * (n**-0.5)
    return flat.reshape(*prefix, n).to(orig_dtype)


# ---------------------------------------------------------------------------
# Self-test (run as `python -m atom.model_ops.quant_v4`)
# ---------------------------------------------------------------------------


def _selftest():
    torch.manual_seed(0)

    # ---- FP8 round-trip: error bounded by ~1/448 per block ----
    x = torch.randn(2, 16, 256, dtype=torch.bfloat16) * 3.0
    x_orig = x.clone()
    act_quant_inplace(x, block_size=128, scale_fmt=None)
    rel_err = (
        ((x.float() - x_orig.float()).abs() / x_orig.float().abs().clamp(min=1e-3))
        .mean()
        .item()
    )
    # FP8 e4m3 has ~3-bit mantissa => ~6% per-element ULP, ~2-3% mean rel error.
    assert rel_err < 0.05, f"FP8 round-trip relative error too large: {rel_err}"
    print(f"[act_quant_inplace fp32-scale]  OK  mean_rel_err={rel_err:.2e}")

    x = x_orig.clone()
    act_quant_inplace(x, block_size=128, scale_fmt="ue8m0")
    rel_err_ue = (
        ((x.float() - x_orig.float()).abs() / x_orig.float().abs().clamp(min=1e-3))
        .mean()
        .item()
    )
    assert (
        rel_err_ue < 0.05
    ), f"FP8 ue8m0 round-trip relative error too large: {rel_err_ue}"
    print(f"[act_quant_inplace ue8m0-scale] OK  mean_rel_err={rel_err_ue:.2e}")

    # ---- FP4 round-trip: error bounded by ~1/6 per block ----
    x = torch.randn(2, 16, 64, dtype=torch.bfloat16) * 2.0
    x_orig = x.clone()
    fp4_act_quant_inplace(x, block_size=32)
    rel_err = (
        ((x.float() - x_orig.float()).abs() / x_orig.float().abs().clamp(min=1e-3))
        .mean()
        .item()
    )
    # FP4 e2m1 has ~1-bit mantissa => ~25% per-element ULP is expected.
    assert rel_err < 0.30, f"FP4 round-trip relative error too large: {rel_err}"
    # Check all values land on valid FP4 grid (after rescaling per block)
    blocks = x.reshape(2, 16, 2, 32).float()
    amax = blocks.abs().amax(dim=-1, keepdim=True).clamp(min=6 * 2**-126)
    scale = torch.pow(2.0, torch.ceil(torch.log2(amax / 6.0)))
    normalized = (blocks / scale).abs()
    valid_grid = _FP4_MAGNITUDES.to(normalized.device)
    on_grid = (
        (normalized.unsqueeze(-1) - valid_grid).abs().min(dim=-1).values.max().item()
    )
    assert on_grid < 1e-4, f"FP4 values off grid by {on_grid}"
    print(
        f"[fp4_act_quant_inplace] OK  mean_rel_err={rel_err:.2e}  off_grid={on_grid:.2e}"
    )

    # ---- Hadamard transform: orthogonality H @ H^T = I ----
    n = 128
    eye = torch.eye(n, dtype=torch.float32)
    h = rotate_activation(eye)
    # H @ H^T should be identity for an orthogonal transform
    hht = h @ h.T
    err = (hht - torch.eye(n)).abs().max().item()
    assert err < 1e-5, f"Hadamard not orthogonal: max abs err = {err}"
    print(f"[rotate_activation orthogonality] OK  max_abs_err={err:.2e}")

    # Hadamard inverse: applying twice = identity (Hadamard is involutive after normalization)
    x = torch.randn(2, 4, 64, dtype=torch.float32)
    twice = rotate_activation(rotate_activation(x))
    err = (twice - x).abs().max().item()
    assert err < 1e-5, f"Hadamard not involutive: max abs err = {err}"
    print(f"[rotate_activation involution]   OK  max_abs_err={err:.2e}")

    print("ALL OK")


if __name__ == "__main__":
    _selftest()
