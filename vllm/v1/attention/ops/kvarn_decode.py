# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KVarN tile-level dequant reference (pure PyTorch).

Inverse of ``kvarn_store_tile_{k,v}``. Produces the dequantized tile in the
**rotated** frame; the caller is responsible for the inverse Hadamard
(matmul with H, which is its own inverse) and any subsequent attention math.

The Triton port (Stage 4) lives in ``triton_kvarn_decode.py`` and must
produce numerically equivalent outputs (cosine ≥ 0.999 vs this reference).
"""

from __future__ import annotations

import torch


def _unpack_4bit(packed: torch.Tensor, original_last_dim: int) -> torch.Tensor:
    """Inverse of ``kvarn_store::_pack_4bit``.

    ``packed`` has shape ``[..., original_last_dim // 2]`` uint8; returns
    shape ``[..., original_last_dim]`` uint8 with values in [0, 15].
    """
    assert original_last_dim % 2 == 0
    lo = packed & 0xF
    hi = (packed >> 4) & 0xF
    out = torch.empty(
        *packed.shape[:-1],
        original_last_dim,
        dtype=torch.uint8,
        device=packed.device,
    )
    out[..., 0::2] = lo
    out[..., 1::2] = hi
    return out


def _unpack_lowbit(
    packed: torch.Tensor, original_last_dim: int, bits: int
) -> torch.Tensor:
    """Inverse of ``kvarn_store::_pack_lowbit`` for any ``bits`` in {2, 4}.

    Mirrors the pack/decode-kernel convention exactly: value at original index
    ``c`` is stored in byte ``c // pack`` at shift ``(c % pack) * bits``, where
    ``pack = 8 // bits`` values share one byte. ``packed`` has shape
    ``[..., original_last_dim // pack]`` uint8; returns ``[..., original_last_dim]``
    uint8 with values in ``[0, 2**bits - 1]``.
    """
    if bits == 4:
        return _unpack_4bit(packed, original_last_dim)
    pack = 8 // bits
    mask = (1 << bits) - 1
    assert original_last_dim % pack == 0, (
        f"last dim {original_last_dim} must be divisible by {pack} for {bits}-bit"
    )
    out = torch.empty(
        *packed.shape[:-1],
        original_last_dim,
        dtype=torch.uint8,
        device=packed.device,
    )
    for j in range(pack):
        out[..., j::pack] = (packed >> (j * bits)) & mask
    return out


def kvarn_dequant_tile_k(
    q_packed_uint8: torch.Tensor,
    s_col_K: torch.Tensor,
    zp_K: torch.Tensor,
    s_row_K: torch.Tensor,
    group: int,
    bits: int = 4,
) -> torch.Tensor:
    """Dequantize one K tile back to the rotated ``[D, group]`` frame.

    Args:
        q_packed_uint8 : ``[D, group // (8//bits)]`` uint8.
        s_col_K        : ``[D]`` fp16  — absorbed per-channel scale.
        zp_K           : ``[D]`` fp16  — absorbed per-channel zero.
        s_row_K        : ``[group]`` fp16 — per-token-in-tile sinkhorn scale.
        group          : tile width in tokens.
        bits           : quant bit-width of K (default 4).

    Returns:
        ``[D, group]`` fp32 dequantized tile in the rotated frame.
        Identity: ``out[r,c] = (q[r,c] * s_col_K[r] + zp_K[r]) * s_row_K[c]``.
    """
    q = _unpack_lowbit(q_packed_uint8, group, bits).float()  # [D, group]
    s_col = s_col_K.float().unsqueeze(-1)  # [D, 1]
    zp = zp_K.float().unsqueeze(-1)  # [D, 1]
    s_row = s_row_K.float().unsqueeze(0)  # [1, group]
    return (q * s_col + zp) * s_row


def kvarn_dequant_tile_v(
    q_packed_uint8: torch.Tensor,
    s_col_V: torch.Tensor,
    s_row_V: torch.Tensor,
    zp_V: torch.Tensor,
    head_dim: int,
    bits: int = 4,
) -> torch.Tensor:
    """Dequantize one V tile back to the rotated ``[group, D]`` frame.

    Args:
        q_packed_uint8 : ``[group, D // (8//bits)]`` uint8.
        s_col_V        : ``[D]`` fp16  — per-channel sinkhorn scale (untouched).
        s_row_V        : ``[group]`` fp16 — absorbed per-token-in-tile scale.
        zp_V           : ``[group]`` fp16 — absorbed per-token-in-tile zero.
        head_dim       : tile width in channels.
        bits           : quant bit-width of V (default 4; k4v2 uses 2).

    Returns:
        ``[group, head_dim]`` fp32 dequantized tile in the rotated frame.
        Identity: ``out[t,c] = (q[t,c] * s_row_V[t] + zp_V[t]) * s_col_V[c]``.
    """
    q = _unpack_lowbit(q_packed_uint8, head_dim, bits).float()  # [group, D]
    s_row = s_row_V.float().unsqueeze(-1)  # [group, 1]
    zp = zp_V.float().unsqueeze(-1)  # [group, 1]
    s_col = s_col_V.float().unsqueeze(0)  # [1, D]
    return (q * s_row + zp) * s_col
