# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM IR op for rotary positional embedding (RoPE).

Prototype migration of ``rotary_embedding`` from a plain ``CustomOp`` to a vLLM
IR op, mirroring ``rms_norm``. Making RoPE a stable IR node lets the QK-norm +
RoPE(+KVCache) fusion patterns match it *before* lowering, so the fusion no
longer needs ``custom_ops=["+rotary_embedding"]`` force-enabled (issue #28042).

Scope: the base ``RotaryEmbedding`` (Neox and GPT-J styles, full or partial
rope) with ``key`` present. The ``key is None`` case and the flashinfer / ROCm
AITER variants stay on their existing paths for now.
"""

import torch
from torch import Tensor

from ..op import register_op


def _rotate(x: Tensor, cos: Tensor, sin: Tensor, is_neox: bool) -> Tensor:
    """Apply the rotation to the rope slice ``x``.

    Args:
        x: ``[num_tokens, num_heads, rotary_dim]`` rope portion.
        cos: ``[num_tokens, rotary_dim // 2]`` cosine table for the positions.
        sin: ``[num_tokens, rotary_dim // 2]`` sine table for the positions.
        is_neox: Neox-style (contiguous halves) vs GPT-J-style (interleaved).

    Mirrors ``ApplyRotaryEmb.forward_static`` (no fp32 upcast, matching the
    base ``RotaryEmbedding`` native path).
    """
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


@register_op(activations=["query", "key"], allow_inplace=True)
def rotary_embedding(
    positions: Tensor,
    query: Tensor,
    key: Tensor,
    head_size: int,
    rotary_dim: int,
    cos_sin_cache: Tensor,
    is_neox: bool,
) -> tuple[Tensor, Tensor]:
    """Functional RoPE over query and key.

    Semantics match ``RotaryEmbedding.forward_static`` for ``key`` present:
    look up cos/sin by ``positions``, rotate the first ``rotary_dim`` channels
    of each head, and pass the remainder through unchanged.
    """
    positions = positions.flatten()
    num_tokens = positions.shape[0]
    cos_sin = cos_sin_cache.index_select(0, positions)
    cos, sin = cos_sin.chunk(2, dim=-1)

    query_shape = query.shape
    query = query.view(num_tokens, -1, head_size)
    query_rot = _rotate(query[..., :rotary_dim], cos, sin, is_neox)
    query = torch.cat((query_rot, query[..., rotary_dim:]), dim=-1).reshape(
        query_shape
    )

    key_shape = key.shape
    key = key.view(num_tokens, -1, head_size)
    key_rot = _rotate(key[..., :rotary_dim], cos, sin, is_neox)
    key = torch.cat((key_rot, key[..., rotary_dim:]), dim=-1).reshape(key_shape)

    return query, key


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
    cos_sin_cache = torch.randn(4096, rotary_dim, dtype=dtype, device=device)
    return positions, query, key, head_size, rotary_dim, cos_sin_cache, is_neox
