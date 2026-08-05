# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KIVI 4-bit (per-quest-chunk per-channel) key cache packing."""

from __future__ import annotations

import torch


def _bits_constants(bits: int) -> tuple[int, int]:
    if bits not in (2, 4, 8):
        raise ValueError(f"bits must be one of (2, 4, 8); got {bits}")
    if 32 % bits != 0:
        raise ValueError(f"bits ({bits}) must divide 32")
    levels = (1 << bits) - 1
    feat_per_int = 32 // bits
    return levels, feat_per_int


def chunk_minmax_to_scale_zero(
    chunk_min: torch.Tensor,
    chunk_max: torch.Tensor,
    bits: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    levels, _ = _bits_constants(bits)
    scale = (chunk_max - chunk_min) / float(levels)
    return scale, chunk_min


def _expand_chunk_meta_to_tokens(
    meta: torch.Tensor,
    group_size: int,
    T: int,
) -> torch.Tensor:
    bs, kv, n_chunks, D = meta.shape
    assert n_chunks * group_size >= T, (
        f"chunk meta covers {n_chunks * group_size} tokens, < T={T}"
    )
    expanded = meta.unsqueeze(3).expand(bs, kv, n_chunks, group_size, D)
    return expanded.reshape(bs, kv, n_chunks * group_size, D)[:, :, :T, :].contiguous()


def quantize_kcache_4bit(
    K: torch.Tensor,
    chunk_min: torch.Tensor,
    chunk_max: torch.Tensor,
    group_size: int,
    bits: int = 4,
) -> torch.Tensor:
    bs, kv, T, D = K.shape
    n_chunks = T // group_size
    T_trunc = n_chunks * group_size
    levels, _ = _bits_constants(bits)
    mn = chunk_min[..., :n_chunks, :]
    mx = chunk_max[..., :n_chunks, :]
    scale = (mx - mn) / float(levels)
    scale_safe = scale.clamp(min=1e-8)
    scale_t = _expand_chunk_meta_to_tokens(scale_safe, group_size, T_trunc)
    mn_t = _expand_chunk_meta_to_tokens(mn, group_size, T_trunc)
    K_use = K[..., :T_trunc, :]
    q = ((K_use - mn_t) / scale_t).clamp_(0.0, float(levels)).round_()
    return q.to(torch.int32)


def _pack_along_last_dim(codes_int32: torch.Tensor, bits: int) -> torch.Tensor:
    _, feat_per_int = _bits_constants(bits)
    shape = codes_int32.shape
    D = shape[-1]
    assert D % feat_per_int == 0, (
        f"D={D} not divisible by feat_per_int={feat_per_int} (bits={bits})"
    )
    n_pack = D // feat_per_int
    view = codes_int32.view(*shape[:-1], n_pack, feat_per_int)
    shifts = torch.arange(
        feat_per_int, device=codes_int32.device, dtype=torch.int32
    ) * int(bits)
    packed = (view << shifts).sum(dim=-1, dtype=torch.int32)
    return packed.contiguous()


def pack_kcache_4bit(
    K: torch.Tensor,
    chunk_min: torch.Tensor,
    chunk_max: torch.Tensor,
    group_size: int,
    bits: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    codes = quantize_kcache_4bit(K, chunk_min, chunk_max, group_size, bits=bits)
    packed = _pack_along_last_dim(codes, bits=bits)
    scale, _ = chunk_minmax_to_scale_zero(chunk_min, chunk_max, bits=bits)
    return packed, scale


def pack_block_kcache_4bit(
    K_block: torch.Tensor,
    bits: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack a single physical block of keys.

    Args:
        K_block: [num_kv_heads, block_size, head_dim]
    Returns:
        packed_chunk_major: [num_kv_heads, n_pack, block_size] int32
        chunk_min / chunk_max / centroid: [num_kv_heads, head_dim]
    """
    assert K_block.dim() == 3
    kv, g, D = K_block.shape
    K = K_block.unsqueeze(0)  # [1, kv, g, D]
    chunk_min = K.amin(dim=2)  # [1, kv, D]
    chunk_max = K.amax(dim=2)
    centroid = K.mean(dim=2)
    packed_tok, _ = pack_kcache_4bit(
        K,
        chunk_min.unsqueeze(2),
        chunk_max.unsqueeze(2),
        group_size=g,
        bits=bits,
    )  # [1, kv, g, n_pack]
    # Chunk-major layout expected by KIVI rerank: [kv, n_pack, g]
    packed_cm = packed_tok.squeeze(0).transpose(-1, -2).contiguous()
    return (
        packed_cm,
        chunk_min.squeeze(0).contiguous(),
        chunk_max.squeeze(0).contiguous(),
        centroid.squeeze(0).contiguous(),
    )


def _unpack_along_last_dim(packed: torch.Tensor, bits: int) -> torch.Tensor:
    _, feat_per_int = _bits_constants(bits)
    levels = (1 << bits) - 1
    shifts = torch.arange(feat_per_int, device=packed.device, dtype=torch.int32) * int(
        bits
    )
    codes = (packed.unsqueeze(-1) >> shifts) & levels
    out_shape = packed.shape[:-1] + (packed.shape[-1] * feat_per_int,)
    return codes.view(*out_shape).contiguous()


def unpack_kcache_4bit(
    packed: torch.Tensor,
    chunk_min: torch.Tensor,
    chunk_max: torch.Tensor,
    group_size: int,
    bits: int = 4,
) -> torch.Tensor:
    codes = _unpack_along_last_dim(packed, bits=bits)
    bs, kv, T, D = codes.shape
    n_chunks = (T + group_size - 1) // group_size
    scale, mn = chunk_minmax_to_scale_zero(
        chunk_min[..., :n_chunks, :],
        chunk_max[..., :n_chunks, :],
        bits=bits,
    )
    scale_t = _expand_chunk_meta_to_tokens(scale, group_size, T)
    mn_t = _expand_chunk_meta_to_tokens(mn, group_size, T)
    K_approx = codes.to(scale_t.dtype) * scale_t + mn_t
    return K_approx.to(chunk_min.dtype)
