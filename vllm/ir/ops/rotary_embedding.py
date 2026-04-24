# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
from torch import Tensor

from ..op import register_op


@register_op
def rotary_embedding_neox(
    positions: Tensor,
    query: Tensor,
    key: Tensor,
    head_size: int,
    rotary_dim: int,
    cos_sin_cache: Tensor,
) -> tuple[Tensor, Tensor]:
    """Rotary positional embedding – Neox-style (first-half / second-half split).

    Applies RoPE to the first ``rotary_dim`` dimensions of each attention head::

        cos, sin = cos_sin_cache[pos].chunk(2)   # each [rotary_dim//2]
        x1, x2  = q[..., :half], q[..., half:]
        out      = cat(x1*cos - x2*sin,  x2*cos + x1*sin)

    ``cos_sin_cache`` layout (matches ``RotaryEmbedding._compute_cos_sin_cache``)::

        cos_sin_cache[pos, :rotary_dim//2]  = cos values
        cos_sin_cache[pos, rotary_dim//2:]  = sin values

    Args:
        positions:      [num_tokens] int64 position indices.
        query:          [num_tokens, num_q_heads * head_size].
        key:            [num_tokens, num_kv_heads * head_size].
        head_size:      Full head dimension.
        rotary_dim:     Number of head dims to rotate (≤ head_size).
        cos_sin_cache:  [max_seq_len, rotary_dim] packed cos|sin cache.

    Returns:
        (query_out, key_out) with RoPE applied (out-of-place).
    """
    num_tokens = positions.shape[0]
    rot_half = rotary_dim // 2

    cos_sin = cos_sin_cache.index_select(0, positions)  # [num_tokens, rotary_dim]
    cos = cos_sin[:, :rot_half]   # [num_tokens, rot_half]
    sin = cos_sin[:, rot_half:]   # [num_tokens, rot_half]

    def _apply(x_flat: Tensor, n_heads: int) -> Tensor:
        x = x_flat.reshape(num_tokens, n_heads, head_size)
        x_rot = x[:, :, :rotary_dim]
        x_pass = x[:, :, rotary_dim:]
        cos_h = cos.unsqueeze(1).to(x.dtype)  # [num_tokens, 1, rot_half]
        sin_h = sin.unsqueeze(1).to(x.dtype)
        x1 = x_rot[:, :, :rot_half]
        x2 = x_rot[:, :, rot_half:]
        x_rot_out = torch.cat((x1 * cos_h - x2 * sin_h,
                               x2 * cos_h + x1 * sin_h), dim=-1)
        return torch.cat((x_rot_out, x_pass), dim=-1).reshape_as(x_flat)

    num_q_heads = query.shape[1] // head_size
    num_kv_heads = key.shape[1] // head_size
    return _apply(query, num_q_heads), _apply(key, num_kv_heads)


@register_op
def rotary_embedding_gptj(
    positions: Tensor,
    query: Tensor,
    key: Tensor,
    head_size: int,
    rotary_dim: int,
    cos_sin_cache: Tensor,
) -> tuple[Tensor, Tensor]:
    """Rotary positional embedding – GPT-J-style (interleaved even/odd).

    Applies RoPE to the first ``rotary_dim`` dimensions of each attention head::

        cos, sin = cos_sin_cache[pos].chunk(2)   # each [rotary_dim//2]
        x1, x2  = q[..., ::2], q[..., 1::2]
        out      = stack(x1*cos - x2*sin,  x2*cos + x1*sin).flatten(-2)

    ``cos_sin_cache`` layout is identical to ``rotary_embedding_neox``.

    Args:
        positions:      [num_tokens] int64 position indices.
        query:          [num_tokens, num_q_heads * head_size].
        key:            [num_tokens, num_kv_heads * head_size].
        head_size:      Full head dimension.
        rotary_dim:     Number of head dims to rotate (≤ head_size).
        cos_sin_cache:  [max_seq_len, rotary_dim] packed cos|sin cache.

    Returns:
        (query_out, key_out) with RoPE applied (out-of-place).
    """
    num_tokens = positions.shape[0]
    rot_half = rotary_dim // 2

    cos_sin = cos_sin_cache.index_select(0, positions)
    cos = cos_sin[:, :rot_half]
    sin = cos_sin[:, rot_half:]

    def _apply(x_flat: Tensor, n_heads: int) -> Tensor:
        x = x_flat.reshape(num_tokens, n_heads, head_size)
        x_rot = x[:, :, :rotary_dim]
        x_pass = x[:, :, rotary_dim:]
        cos_h = cos.unsqueeze(1).to(x.dtype)
        sin_h = sin.unsqueeze(1).to(x.dtype)
        x1 = x_rot[:, :, ::2]
        x2 = x_rot[:, :, 1::2]
        x_rot_out = torch.stack((x1 * cos_h - x2 * sin_h,
                                 x2 * cos_h + x1 * sin_h), dim=-1).flatten(-2)
        return torch.cat((x_rot_out, x_pass), dim=-1).reshape_as(x_flat)

    num_q_heads = query.shape[1] // head_size
    num_kv_heads = key.shape[1] // head_size
    return _apply(query, num_q_heads), _apply(key, num_kv_heads)
