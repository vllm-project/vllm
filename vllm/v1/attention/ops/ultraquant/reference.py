# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PyTorch reference for the UltraQuant KV cache format.

Scale is UE8M0 (power of two): after ``s = c * absmax`` snap
``s = 2^round(log2(s))`` and store one byte. Q is Hadamard-rotated then
cast to FP8 E4M3 before the QK matmul, matching the kernel launcher.

Same FP4 E2M1 grid. ``c = 0.156``. No per-token L2 norm. No V rotation.
K is rotated at encode; Q arrives pre-rotated at decode.

This module is the bit-comparison ground truth for the kernels.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm.v1.attention.ops.ultraquant.format import (
    FP4_LEVELS_SORTED,
    GROUP_SIZE,
    MIDPOINTS_SORTED,
    SORTED_TO_BITS,
    UE8M0_BIAS,
    get_constant_c,
)

_LEVELS_CACHE: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}
_MIDPOINTS_CACHE: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}
_SORTED_TO_BITS_CACHE: dict[torch.device, torch.Tensor] = {}


def _get_levels(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (device, dtype)
    t = _LEVELS_CACHE.get(key)
    if t is None:
        t = torch.tensor(FP4_LEVELS_SORTED, device=device, dtype=dtype)
        _LEVELS_CACHE[key] = t
    return t


def _get_midpoints(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (device, dtype)
    t = _MIDPOINTS_CACHE.get(key)
    if t is None:
        t = torch.tensor(MIDPOINTS_SORTED, device=device, dtype=dtype)
        _MIDPOINTS_CACHE[key] = t
    return t


def _get_sorted_to_bits(device: torch.device) -> torch.Tensor:
    t = _SORTED_TO_BITS_CACHE.get(device)
    if t is None:
        t = torch.tensor(SORTED_TO_BITS, device=device, dtype=torch.uint8)
        _SORTED_TO_BITS_CACHE[device] = t
    return t


_HADAMARD_CACHE: dict[tuple[int, torch.device, torch.dtype], torch.Tensor] = {}


def hadamard_matrix(
    dim: int, device: torch.device, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Sylvester Hadamard, normalised so H @ H.T = I."""
    if dim <= 0 or (dim & (dim - 1)) != 0:
        raise ValueError(f"hadamard_matrix requires power-of-two dim, got {dim}")
    key = (dim, device, dtype)
    cached = _HADAMARD_CACHE.get(key)
    if cached is not None:
        return cached
    H = torch.tensor([[1.0]], dtype=torch.float64)
    while H.shape[0] < dim:
        H = torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)
    H = (H / (dim**0.5)).to(device=device, dtype=dtype).contiguous()
    _HADAMARD_CACHE[key] = H
    return H


# ── Core quantization primitives ───────────────────────────────────────────
def _snap_to_sorted_idx(
    x_norm: torch.Tensor, *, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Snap normalised values to nearest FP4 level, returning sorted index
    in [0, 14]. Same semantic as `torch.bucketize` against midpoints
    (reproduced by 14 `tl.where` in the kernel)."""
    boundaries = _get_midpoints(x_norm.device, dtype)
    x = x_norm.to(dtype)
    # ROCm's bucketize uses int32 element indexing and raises
    # hipErrorInvalidConfiguration once numel >= 2**31 (e.g. a [B, Hk, N, D]
    # K tensor at B=128, N=16384, Hk=8, D=128 is exactly 2**31). Chunk the
    # flattened input so each call stays under the limit (bit-identical output).
    _INT32_LIMIT = 2**31
    if x.numel() < _INT32_LIMIT:
        return torch.bucketize(x, boundaries)
    flat = x.reshape(-1)
    out = torch.empty_like(flat, dtype=torch.long)
    chunk = _INT32_LIMIT // 2
    for i in range(0, flat.numel(), chunk):
        out[i : i + chunk] = torch.bucketize(flat[i : i + chunk], boundaries)
    return out.reshape(x.shape)


def _pack_nibbles_last_dim(codes: torch.Tensor) -> torch.Tensor:
    if codes.shape[-1] % 2 != 0:
        raise ValueError(f"pack_nibbles last dim must be even, got {codes.shape[-1]}")
    lo = codes[..., 0::2] & 0xF
    hi = codes[..., 1::2] & 0xF
    return (lo | (hi << 4)).to(torch.uint8)


def _unpack_nibbles_last_dim(packed: torch.Tensor, out_dim: int) -> torch.Tensor:
    if out_dim != 2 * packed.shape[-1]:
        packed_dim = packed.shape[-1]
        raise ValueError(
            f"unpack_nibbles: out_dim={out_dim} must be 2*packed.last={2 * packed_dim}"
        )
    p = packed.to(torch.int32)
    lo = (p & 0xF).to(torch.uint8)
    hi = ((p >> 4) & 0xF).to(torch.uint8)
    out_shape = packed.shape[:-1] + (out_dim,)
    out = torch.empty(out_shape, dtype=torch.uint8, device=packed.device)
    out[..., 0::2] = lo
    out[..., 1::2] = hi
    return out


def _ue8m0_snap_tensor(
    s_raw: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Snap a positive-scale tensor `s_raw` to a power of two and return
    ``(s_snapped_fp32, byte_uint8)``.

    Matches the kernel: zero / non-finite inputs map to byte=0 / value=0.0.
    Round-half-up via ``floor(log2(s) + 0.5)``, matching Triton's
    ``tl.floor(... + 0.5)`` idiom.
    """
    is_zero = (s_raw <= 0) | ~torch.isfinite(s_raw)
    s_safe = torch.where(is_zero, torch.ones_like(s_raw), s_raw)
    log2_s = torch.log2(s_safe)
    exp = torch.floor(log2_s + 0.5).to(torch.int32)
    # Clamp to UE8M0's representable exponent range (signed pre-bias).
    exp = torch.clamp(exp, min=-126, max=127)
    byte = (exp + UE8M0_BIAS).to(torch.int32)
    byte = torch.where(is_zero, torch.zeros_like(byte), byte).to(torch.uint8)
    s_snapped = torch.where(
        is_zero, torch.zeros_like(s_raw), torch.exp2(exp.to(torch.float32))
    )
    return s_snapped, byte


@dataclass
class UltraQuantEncoded:
    """Encoded ultraquant representation of one tensor along its last axis.

    Shapes assume input shape `[..., D]`:
    - codes_packed: `[..., D // 2]` uint8 — 2 FP4 nibbles per byte
    - scale_bytes:  `[..., D // GROUP_SIZE]` uint8 — E8M0 byte per group
    Dequant is ``code · 2^(byte - 127)`` (`c` is folded into the byte).
    """

    codes_packed: torch.Tensor  # uint8, [..., D//2]
    scale_bytes: torch.Tensor  # uint8, [..., D//GROUP_SIZE]

    @property
    def scales_fp32(self) -> torch.Tensor:
        """Decode the E8M0 bytes back to fp32 (for dequant / inspection)."""
        b = self.scale_bytes.to(torch.int32)
        exp = b - UE8M0_BIAS
        is_zero = b == 0
        s = torch.where(
            is_zero,
            torch.zeros_like(b, dtype=torch.float32),
            torch.exp2(exp.to(torch.float32)),
        )
        return s


def ultraquant_encode(
    x: torch.Tensor,
    *,
    rotate: bool = True,
    constant_c: float | None = None,
) -> UltraQuantEncoded:
    """Encode `x` (shape [..., D]) to FP4 codes + E8M0 per-group-of-32 scales.

    Steps:
    1. Optional Hadamard rotation: `x_rot = x @ H.T`
    2. Reshape to `[..., G, GROUP_SIZE]`.
    3. `absmax = max(|x_rot|)` per group.
    4. `s_raw = c · absmax`; `s_snapped = 2^round(log2(s_raw))`.
       Zero-amax groups encode byte=0.
    5. `sorted_idx = bucketize(x_g / s_snapped, midpoints)` ∈ [0, 14].
    6. Remap sorted_idx → FP4 E2M1 bit pattern via SORTED_TO_BITS.
    7. Pack pairs of 4-bit codes into bytes.
    """
    if x.shape[-1] % GROUP_SIZE != 0:
        raise ValueError(
            f"ultraquant_encode: last dim {x.shape[-1]} must be a multiple of "
            f"GROUP_SIZE={GROUP_SIZE}"
        )
    D = x.shape[-1]
    G = D // GROUP_SIZE
    c = constant_c if constant_c is not None else get_constant_c()

    x_f32 = x.to(torch.float32)
    if rotate:
        H = hadamard_matrix(D, x.device, torch.float32)
        x_rot = x_f32 @ H.T
    else:
        x_rot = x_f32

    x_g = x_rot.reshape(*x.shape[:-1], G, GROUP_SIZE)
    absmax = x_g.abs().amax(dim=-1, keepdim=True)  # [..., G, 1]
    s_raw = absmax * c
    s_snapped, scale_bytes = _ue8m0_snap_tensor(s_raw.squeeze(-1))
    s_snapped = s_snapped.unsqueeze(-1)  # [..., G, 1]
    s_for_div = torch.where(s_snapped == 0, torch.ones_like(s_snapped), s_snapped)

    sorted_idx = _snap_to_sorted_idx(x_g / s_for_div)
    sorted_idx = torch.where(
        (s_snapped == 0).expand_as(sorted_idx),
        torch.full_like(sorted_idx, 7),  # sorted idx 7 → +0.0
        sorted_idx,
    )

    sorted_to_bits = _get_sorted_to_bits(x.device)
    fp4_bits = sorted_to_bits[sorted_idx]  # [..., G, GS]
    fp4_bits_flat = fp4_bits.reshape(*x.shape[:-1], D)
    codes_packed = _pack_nibbles_last_dim(fp4_bits_flat)  # [..., D//2]

    return UltraQuantEncoded(
        codes_packed=codes_packed,
        scale_bytes=scale_bytes,
    )


def ultraquant_dequant(
    encoded: UltraQuantEncoded,
    head_dim: int,
) -> torch.Tensor:
    """Decode `encoded` to fp32 (no inverse rotation — result in the
    rotated basis if rotation was applied at encode).

    Dequant is ``code · 2^(byte - 127)`` (`c` is already folded into the byte).
    """
    G = head_dim // GROUP_SIZE
    codes = _unpack_nibbles_last_dim(encoded.codes_packed, head_dim)
    bits_to_val = torch.tensor(
        [
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
        ],
        device=encoded.codes_packed.device,
        dtype=torch.float32,
    )
    decoded = bits_to_val[codes.to(torch.long)]  # [..., D] fp32
    decoded_g = decoded.reshape(*decoded.shape[:-1], G, GROUP_SIZE)
    scales = encoded.scales_fp32.unsqueeze(-1)  # [..., G, 1]
    return (decoded_g * scales).reshape(*decoded.shape[:-1], head_dim)


def ultraquant_encode_decode(
    x: torch.Tensor,
    *,
    rotate: bool = True,
    constant_c: float | None = None,
) -> torch.Tensor:
    """Full encode→decode round-trip in the rotated basis. Returns fp32."""
    enc = ultraquant_encode(
        x,
        rotate=rotate,
        constant_c=constant_c,
    )
    return ultraquant_dequant(enc, x.shape[-1])


# ── Reference attention against the ultraquant encoded cache ──────────────────
def reference_ultraquant_attention(
    query: torch.Tensor,  # [B, Hq, D] bf16 — RAW (will be rotated + cast to FP8 here)
    key: torch.Tensor,  # [B, N, Hk, D] bf16
    value: torch.Tensor,  # [B, N, Hk, D] bf16
    *,
    scale: float,
    sinks: torch.Tensor | None = None,
    constant_c: float | None = None,
) -> torch.Tensor:
    """End-to-end reference attention with ultraquant K/V encode→decode in
    the rotated basis. Q is Hadamard-rotated AND cast through FP8 E4M3
    (scale = 1) before the QK matmul — mirrors what the launcher does.

    Output shape: [B, Hq, D] in query.dtype.
    """
    B, Hq, D = query.shape
    B2, N, Hk, D2 = key.shape
    assert B == B2 and D == D2, (B, B2, D, D2)
    assert value.shape == key.shape
    assert Hq % Hk == 0, f"Hq={Hq} must be a multiple of Hk={Hk}"
    g = Hq // Hk

    H = hadamard_matrix(D, query.device, torch.float32)
    q_rot = query.to(torch.float32) @ H.T
    # Round-trip Q through FP8 E4M3 with scale=1 — same precision haircut
    # as the launcher (`q_rot.to(torch.float8_e4m3fn)`).
    q_fp8 = q_rot.to(torch.float8_e4m3fn)
    q_for_matmul = q_fp8.to(torch.float32)

    k_recon_rot = ultraquant_encode_decode(
        key.transpose(1, 2).contiguous(),
        rotate=True,
        constant_c=constant_c,
    )
    v_recon = ultraquant_encode_decode(
        value.transpose(1, 2).contiguous(),
        rotate=False,
        constant_c=constant_c,
    )

    k_recon_rot = k_recon_rot.unsqueeze(2).expand(B, Hk, g, N, D).reshape(B, Hq, N, D)
    v_recon = v_recon.unsqueeze(2).expand(B, Hk, g, N, D).reshape(B, Hq, N, D)

    qk = torch.einsum("bhd,bhnd->bhn", q_for_matmul, k_recon_rot) * scale  # [B, Hq, N]

    if sinks is not None:
        sink_log = sinks.to(torch.float32).reshape(1, Hq, 1).expand(B, Hq, 1)
        qk_full = torch.cat([qk, sink_log], dim=-1)
        p_full = torch.softmax(qk_full, dim=-1)
        p = p_full[..., :N]
    else:
        p = torch.softmax(qk, dim=-1)

    out = torch.einsum("bhn,bhnd->bhd", p, v_recon)
    return out.to(query.dtype)
