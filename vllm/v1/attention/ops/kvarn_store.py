# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KVarN tile-level store reference (pure PyTorch).

Quantizes one tile of K (or V) per call. The Triton port (Stage 4) lives in
``triton_kvarn_store.py`` and must produce byte-identical outputs.

Inputs to the K path are tile-shaped ``[D, group]`` (channels × tokens — the
KIVI K-axis orientation) **after** Hadamard rotation. Inputs to the V path
are tile-shaped ``[group, D]`` (tokens × channels — the KIVI V-axis
orientation) also after Hadamard rotation. The Hadamard rotation is applied
externally via a cuBLAS GEMM, identically to TurboQuant's MSE path.

The output is a packed record matching the cache layout from
``KVarNConfig`` — see that file for byte offsets.
"""

from __future__ import annotations

import os

import torch

from vllm.model_executor.layers.quantization.kvarn.sinkhorn import (
    variance_normalize,
)


def _rtn_range(t: torch.Tensor, dim: int):
    """Per-row range. With KVARN_RTN_QUANTILE=q > 0 (e.g. 0.005), uses
    percentiles [q, 1-q] instead of min/max — values outside get clamped at
    quantize time, sacrificing outliers for finer bulk resolution. Critical
    for k2v2 on models like Qwen3-30B-A3B-Thinking where K outliers
    (max/std ≈ 6) waste 2-bit resolution.
    """
    q_str = os.environ.get("KVARN_RTN_QUANTILE", "")
    if q_str and float(q_str) > 0:
        q = float(q_str)
        lo = torch.quantile(t, q, dim=dim, keepdim=True)
        hi = torch.quantile(t, 1.0 - q, dim=dim, keepdim=True)
        return lo, hi
    return t.amin(dim=dim, keepdim=True), t.amax(dim=dim, keepdim=True)


# ──────────────────────────────────────────────────────────────────────────────
# Asymmetric per-row RTN
# ──────────────────────────────────────────────────────────────────────────────


def _asymmetric_rtn_per_row(
    tile: torch.Tensor, bits: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-row asymmetric RTN over the full row (no sub-grouping).

    Args:
        tile: [R, C] fp32.
        bits: 2, 3 or 4.

    Returns:
        q     [R, C] int32 in [0, 2^bits - 1]
        scale [R, 1] fp32
        zp    [R, 1] fp32  (= row minimum)
    """
    qmax = (1 << bits) - 1
    lo = tile.amin(dim=1, keepdim=True)
    hi = tile.amax(dim=1, keepdim=True)
    scale = ((hi - lo) / qmax).clamp_min(1e-10)
    zp = lo
    q = torch.clamp(torch.round((tile - zp) / scale), 0, qmax).to(torch.int32)
    return q, scale, zp


def _pack_4bit(q: torch.Tensor) -> torch.Tensor:
    """Pack 4-bit ints (last dim even) into uint8 pairs, two-per-byte.

    Layout: low nibble = even-indexed, high nibble = odd-indexed.
    """
    assert q.shape[-1] % 2 == 0, "last dim must be even for 4-bit pairing"
    q = q.to(torch.uint8) & 0xF
    lo = q[..., 0::2]
    hi = q[..., 1::2]
    return (lo | (hi << 4)).to(torch.uint8)


def _pack_lowbit(q: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack `bits`-bit ints into uint8, PACK=8//bits values per byte.

    Value at last-dim index c lands in byte c//PACK at bit-shift (c%PACK)*bits —
    matching the decode kernel's unpack (byte=idx//PACK, shift=(idx%PACK)*bits).
    bits=4 reduces to _pack_4bit; bits=2 packs 4 values/byte.
    """
    pack = 8 // bits
    C = q.shape[-1]
    assert C % pack == 0, f"last dim {C} must be divisible by {pack} for {bits}-bit"
    q = (q.to(torch.uint8) & ((1 << bits) - 1)).reshape(*q.shape[:-1], C // pack, pack)
    out = q[..., 0].clone()
    for j in range(1, pack):
        out = out | (q[..., j] << (j * bits))
    return out.to(torch.uint8)


# ──────────────────────────────────────────────────────────────────────────────
# K tile store: per-channel RTN, [D, group] orientation
# ──────────────────────────────────────────────────────────────────────────────


def kvarn_store_tile_k(
    k_tile_rotated: torch.Tensor,
    bits: int,
    sinkhorn_iters: int = 16,
) -> dict[str, torch.Tensor]:
    """Quantize one rotated K tile.

    Args:
        k_tile_rotated: ``[D, group]`` fp32 / fp16 — channels × tokens, *after*
            Hadamard rotation along head_dim. Caller is responsible for the
            external ``(K @ H).T`` GEMM.
        bits: key bit-width (typically 4).
        sinkhorn_iters: log-domain iterations.

    Returns dict with packed cache record:
        q_packed_uint8 : ``[D, group/2]`` uint8 — 4-bit pairs (low=even, high=odd)
        s_col_K        : ``[D]``        fp16   — absorbed per-channel scale
        zp_K           : ``[D]``        fp16   — absorbed per-channel zero
        s_row_K        : ``[group]``    fp16   — per-token-in-tile sinkhorn scale
    """
    assert bits == 4, "Stage 3a only validates 4-bit; lower-bit follow-ups TBD"
    tile = k_tile_rotated.float()
    D, G = tile.shape

    balanced, s_col_sinkhorn, s_row_sinkhorn = variance_normalize(
        tile, iterations=sinkhorn_iters
    )
    # In [D, group] orientation:
    #   s_col_sinkhorn is [1, G] = per-token-in-tile
    #   s_row_sinkhorn is [D, 1] = per-channel
    s_chan = s_row_sinkhorn  # [D, 1]
    s_tok = s_col_sinkhorn  # [1, G]

    q, rtn_scale, rtn_zp = _asymmetric_rtn_per_row(balanced, bits=bits)
    # rtn_scale [D, 1], rtn_zp [D, 1]

    # Absorb per-channel RTN into the per-channel sinkhorn scale.
    s_col_K = (s_chan * rtn_scale).squeeze(-1)  # [D]
    zp_K = (s_chan * rtn_zp).squeeze(-1)  # [D]
    s_row_K = s_tok.squeeze(0)  # [G]

    q_packed = _pack_4bit(q)  # [D, G/2]

    return {
        "q_packed_uint8": q_packed,
        "s_col_K": s_col_K.to(torch.float16),
        "zp_K": zp_K.to(torch.float16),
        "s_row_K": s_row_K.to(torch.float16),
    }


# ──────────────────────────────────────────────────────────────────────────────
# V tile store: per-token RTN, [group, D] orientation
# ──────────────────────────────────────────────────────────────────────────────


def kvarn_store_tile_k_batch_from_sinkhorn(
    balanced: torch.Tensor,
    s_col: torch.Tensor,
    s_row: torch.Tensor,
    bits: int,
) -> dict[str, torch.Tensor]:
    """Batched K-path RTN + scale absorption + 4-bit packing.

    Assumes the sinkhorn step already ran (e.g. via the Triton kernel).

    Args:
        balanced : ``[N, D, group]`` fp32 — sinkhorn-balanced K tiles.
        s_col    : ``[N, group]`` fp32 — per-token sinkhorn scale (axis-1).
        s_row    : ``[N, D]``     fp32 — per-channel sinkhorn scale (axis-0).
        bits     : key bit-width (4).

    Returns dict of per-tile (N-batched) tensors:
        q_packed_uint8 : ``[N, D, group/2]`` uint8
        s_col_K        : ``[N, D]``         fp16 — absorbed per-channel scale
        zp_K           : ``[N, D]``         fp16 — absorbed per-channel zero
        s_row_K        : ``[N, group]``     fp16 — per-token sinkhorn scale
    """
    qmax = (1 << bits) - 1
    N, R, C = balanced.shape
    lo, hi = _rtn_range(balanced, dim=2)  # [N, R, 1]
    scale = ((hi - lo) / qmax).clamp_min(1e-10)
    zp = lo
    q = torch.clamp(torch.round((balanced - zp) / scale), 0, qmax).to(torch.int32)
    s_col_K = (s_row * scale.squeeze(-1)).to(torch.float16)  # [N, R=D]
    zp_K = (s_row * zp.squeeze(-1)).to(torch.float16)
    s_row_K = s_col.to(torch.float16)  # [N, C=group]
    q_packed = _pack_lowbit(q, bits)  # [N, R, C/pack]
    return {
        "q_packed_uint8": q_packed,
        "s_col_K": s_col_K,
        "zp_K": zp_K,
        "s_row_K": s_row_K,
    }


def kvarn_store_tile_v_batch_from_sinkhorn(
    balanced: torch.Tensor,
    s_col: torch.Tensor,
    s_row: torch.Tensor,
    bits: int,
) -> dict[str, torch.Tensor]:
    """Batched V-path RTN + scale absorption + 4-bit packing.

    Args:
        balanced : ``[N, group, D]`` fp32 — sinkhorn-balanced V tiles.
        s_col    : ``[N, D]``     fp32 — per-channel sinkhorn scale (axis-1).
        s_row    : ``[N, group]`` fp32 — per-token-in-tile sinkhorn scale (axis-0).
        bits     : value bit-width (4).

    Returns dict of per-tile (N-batched) tensors mirroring `kvarn_store_tile_v`.
    """
    qmax = (1 << bits) - 1
    N, R, C = balanced.shape
    lo, hi = _rtn_range(balanced, dim=2)  # [N, R, 1]
    scale = ((hi - lo) / qmax).clamp_min(1e-10)
    zp = lo
    q = torch.clamp(torch.round((balanced - zp) / scale), 0, qmax).to(torch.int32)
    s_row_V = (s_row * scale.squeeze(-1)).to(torch.float16)  # [N, R=group]
    zp_V = (s_row * zp.squeeze(-1)).to(torch.float16)
    s_col_V = s_col.to(torch.float16)  # [N, C=D]
    q_packed = _pack_lowbit(q, bits)  # [N, R, C/pack]
    return {
        "q_packed_uint8": q_packed,
        "s_col_V": s_col_V,
        "s_row_V": s_row_V,
        "zp_V": zp_V,
    }


def kvarn_store_tile_v(
    v_tile_rotated: torch.Tensor,
    bits: int,
    sinkhorn_iters: int = 16,
) -> dict[str, torch.Tensor]:
    """Quantize one rotated V tile.

    Args:
        v_tile_rotated: ``[group, D]`` fp32 / fp16 — tokens × channels, *after*
            Hadamard rotation along head_dim. Caller is responsible for the
            external ``V @ H`` GEMM.
        bits: value bit-width (typically 4).
        sinkhorn_iters: log-domain iterations.

    Returns dict with packed cache record:
        q_packed_uint8 : ``[group, D/2]`` uint8 — 4-bit pairs
        s_col_V        : ``[D]``          fp16   — per-channel scale (untouched)
        s_row_V        : ``[group]``      fp16   — absorbed per-token-in-tile scale
        zp_V           : ``[group]``      fp16   — absorbed per-token-in-tile zero
    """
    assert bits == 4, "Stage 3a only validates 4-bit; lower-bit follow-ups TBD"
    tile = v_tile_rotated.float()
    G, D = tile.shape

    balanced, s_col_sinkhorn, s_row_sinkhorn = variance_normalize(
        tile, iterations=sinkhorn_iters
    )
    # In [group, D] orientation:
    #   s_col_sinkhorn is [1, D] = per-channel
    #   s_row_sinkhorn is [G, 1] = per-token-in-tile
    s_chan = s_col_sinkhorn  # [1, D]
    s_tok = s_row_sinkhorn  # [G, 1]

    q, rtn_scale, rtn_zp = _asymmetric_rtn_per_row(balanced, bits=bits)
    # rtn_scale [G, 1], rtn_zp [G, 1]

    # Absorb per-token RTN into the per-token sinkhorn scale.
    s_row_V = (s_tok * rtn_scale).squeeze(-1)  # [G]
    zp_V = (s_tok * rtn_zp).squeeze(-1)  # [G]
    s_col_V = s_chan.squeeze(0)  # [D]

    q_packed = _pack_4bit(q)  # [G, D/2]

    return {
        "q_packed_uint8": q_packed,
        "s_col_V": s_col_V.to(torch.float16),
        "s_row_V": s_row_V.to(torch.float16),
        "zp_V": zp_V.to(torch.float16),
    }
