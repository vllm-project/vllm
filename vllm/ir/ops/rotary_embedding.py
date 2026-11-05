# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM IR ops for rotary positional embedding (RoPE).

The rotary dimension is the width of ``cos_sin_cache``, matching the vLLM C
kernel. Keeping it out of the signature lets fusion patterns match partial
rotary layers.
"""

import torch
from torch import Tensor

from ..op import register_op


def _rotate(x: Tensor, cos: Tensor, sin: Tensor, is_neox: bool) -> Tensor:
    """Rotate contiguous halves (NeoX) or interleaved pairs (GPT-J)."""
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)

    if is_neox:
        x1, x2 = torch.chunk(x, 2, dim=-1)
    else:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]

    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin

    if is_neox:
        return torch.cat((o1, o2), dim=-1)
    return torch.stack((o1, o2), dim=-1).flatten(-2)


def _apply_rope(
    x: Tensor, cos: Tensor, sin: Tensor, head_size: int, is_neox: bool
) -> Tensor:
    rotary_dim = 2 * cos.shape[-1]
    shape = x.shape
    x = x.view(cos.shape[0], -1, head_size)
    x_rot = _rotate(x[..., :rotary_dim], cos, sin, is_neox)
    return torch.cat((x_rot, x[..., rotary_dim:]), dim=-1).reshape(shape)


def _lookup_cos_sin(positions: Tensor, cos_sin_cache: Tensor) -> tuple[Tensor, Tensor]:
    cos_sin = cos_sin_cache.index_select(0, positions.flatten())
    cos, sin = cos_sin.chunk(2, dim=-1)
    return cos, sin


@register_op(activations=["query", "key"], allow_inplace=True)
def rotary_embedding(
    positions: Tensor,
    query: Tensor,
    key: Tensor,
    head_size: int,
    cos_sin_cache: Tensor,
    is_neox: bool,
) -> tuple[Tensor, Tensor]:
    """Rotate the first ``cos_sin_cache.shape[-1]`` channels of each head."""
    cos, sin = _lookup_cos_sin(positions, cos_sin_cache)
    return (
        _apply_rope(query, cos, sin, head_size, is_neox),
        _apply_rope(key, cos, sin, head_size, is_neox),
    )


rotary_embedding.override_tolerance(torch.float16, atol=5e-3, rtol=2e-3)
rotary_embedding.override_tolerance(torch.bfloat16, atol=3e-2, rtol=2e-2)


@rotary_embedding.register_fake
def _rotary_embedding_fake(
    positions: Tensor,
    query: Tensor,
    key: Tensor,
    head_size: int,
    cos_sin_cache: Tensor,
    is_neox: bool,
) -> tuple[Tensor, Tensor]:
    return torch.empty_like(query), torch.empty_like(key)


@register_op(activations=["query"], allow_inplace=True)
def rotary_embedding_query_only(
    positions: Tensor,
    query: Tensor,
    head_size: int,
    cos_sin_cache: Tensor,
    is_neox: bool,
) -> Tensor:
    """RoPE over query only, for layers that reuse another layer's KV cache."""
    cos, sin = _lookup_cos_sin(positions, cos_sin_cache)
    return _apply_rope(query, cos, sin, head_size, is_neox)


rotary_embedding_query_only.override_tolerance(torch.float16, atol=5e-3, rtol=2e-3)
rotary_embedding_query_only.override_tolerance(torch.bfloat16, atol=3e-2, rtol=2e-2)


@rotary_embedding_query_only.register_fake
def _rotary_embedding_query_only_fake(
    positions: Tensor,
    query: Tensor,
    head_size: int,
    cos_sin_cache: Tensor,
    is_neox: bool,
) -> Tensor:
    return torch.empty_like(query)


def _cos_sin_cache(
    rotary_dim: int, dtype: torch.dtype, device: torch.device | str | None
) -> Tensor:
    inv_freq = 1.0 / (
        10000
        ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float, device=device)
            / rotary_dim
        )
    )
    freqs = torch.outer(torch.arange(4096, dtype=torch.float, device=device), inv_freq)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(dtype)


@rotary_embedding.register_input_generator
def _rotary_embedding_input_generator(
    num_tokens: int,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    rotary_dim: int,
    dtype: torch.dtype,
    is_neox: bool = True,
    device: torch.device | str | None = None,
) -> tuple:
    positions = torch.randint(0, 4096, (num_tokens,), dtype=torch.int64, device=device)
    query = torch.randn(num_tokens, num_heads * head_size, dtype=dtype, device=device)
    key = torch.randn(num_tokens, num_kv_heads * head_size, dtype=dtype, device=device)
    cos_sin_cache = _cos_sin_cache(rotary_dim, dtype, device)
    return positions, query, key, head_size, cos_sin_cache, is_neox


@rotary_embedding_query_only.register_input_generator
def _rotary_embedding_query_only_input_generator(
    num_tokens: int,
    num_heads: int,
    head_size: int,
    rotary_dim: int,
    dtype: torch.dtype,
    is_neox: bool = True,
    device: torch.device | str | None = None,
) -> tuple:
    positions = torch.randint(0, 4096, (num_tokens,), dtype=torch.int64, device=device)
    query = torch.randn(num_tokens, num_heads * head_size, dtype=dtype, device=device)
    cos_sin_cache = _cos_sin_cache(rotary_dim, dtype, device)
    return positions, query, head_size, cos_sin_cache, is_neox
