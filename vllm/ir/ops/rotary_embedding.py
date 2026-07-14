# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
from torch import Tensor

from ..op import register_op


def _apply_rotary_emb(
    x: Tensor,
    cos: Tensor,
    sin: Tensor,
    is_neox_style: bool,
) -> Tensor:
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)

    if is_neox_style:
        x1, x2 = torch.chunk(x, 2, dim=-1)
    else:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]

    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin

    if is_neox_style:
        return torch.cat((o1, o2), dim=-1)
    return torch.stack((o1, o2), dim=-1).flatten(-2)


def _apply_standard_rope(
    x: Tensor,
    num_tokens: int,
    head_size: int,
    rotary_dim: int,
    cos: Tensor,
    sin: Tensor,
    is_neox_style: bool,
) -> Tensor:
    x_shape = x.shape
    x = x.view(num_tokens, -1, head_size)
    x_rot = x[..., :rotary_dim]
    x_pass = x[..., rotary_dim:]

    x_rot = _apply_rotary_emb(
        x_rot,
        cos,
        sin,
        is_neox_style,
    )

    return torch.cat((x_rot, x_pass), dim=-1).reshape(x_shape)


@register_op(activations=["query", "key"], allow_inplace=True)
def rotary_embedding(  # type: ignore[misc]
    positions: Tensor,
    query: Tensor,
    key: Tensor,
    head_size: int,
    rotary_dim: int,
    cos_sin_cache: Tensor,
    is_neox_style: bool,
) -> tuple[Tensor, Tensor]:
    """Apply rotary position embedding to query and key."""
    positions = positions.flatten()
    cos_sin = cos_sin_cache.index_select(0, positions)
    cos, sin = cos_sin.chunk(2, dim=-1)
    query_out = _apply_standard_rope(
        query,
        positions.shape[0],
        head_size,
        rotary_dim,
        cos,
        sin,
        is_neox_style,
    )
    key_out = _apply_standard_rope(
        key,
        positions.shape[0],
        head_size,
        rotary_dim,
        cos,
        sin,
        is_neox_style,
    )
    return query_out, key_out


@rotary_embedding.register_input_generator
def _rotary_embedding_input_generator(
    num_tokens: int = 16,
    num_heads: int = 8,
    num_kv_heads: int = 2,
    head_size: int = 64,
    rotary_dim: int = 64,
    max_pos: int = 4096,
    dtype: torch.dtype = torch.float16,
    is_neox_style: bool = True,
) -> tuple:
    positions = torch.randint(0, max_pos, (num_tokens,), dtype=torch.int64)
    query = torch.randn(num_tokens, num_heads * head_size, dtype=dtype)
    key = torch.randn(num_tokens, num_kv_heads * head_size, dtype=dtype)
    cos_sin_cache = torch.randn(max_pos, rotary_dim, dtype=dtype)
    return positions, query, key, head_size, rotary_dim, cos_sin_cache, is_neox_style


@register_op
def rotary_embedding_query_only(
    positions: Tensor,
    query: Tensor,
    head_size: int,
    rotary_dim: int,
    cos_sin_cache: Tensor,
    is_neox_style: bool,
) -> Tensor:
    """A PyTorch-native implementation of forward()."""
    positions = positions.flatten()
    cos_sin = cos_sin_cache.index_select(0, positions)
    cos, sin = cos_sin.chunk(2, dim=-1)

    query_out = _apply_standard_rope(
        query,
        positions.shape[0],
        head_size,
        rotary_dim,
        cos,
        sin,
        is_neox_style,
    )
    return query_out


@rotary_embedding_query_only.register_input_generator
def _rotary_embedding_query_only_input_generator(
    num_tokens: int = 16,
    num_heads: int = 8,
    head_size: int = 64,
    rotary_dim: int = 64,
    max_pos: int = 4096,
    dtype: torch.dtype = torch.float16,
    is_neox_style: bool = True,
) -> tuple:
    positions = torch.randint(0, max_pos, (num_tokens,), dtype=torch.int64)
    query = torch.randn(num_tokens, num_heads * head_size, dtype=dtype)
    cos_sin_cache = torch.randn(max_pos, rotary_dim, dtype=dtype)
    return positions, query, head_size, rotary_dim, cos_sin_cache, is_neox_style
