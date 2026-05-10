# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ops for GDN linear attention.

_compiled_rearrange_mixed_qkv and _rearrange_mixed_qkv are
implemented differently in torch as each other because of
the different ways they are used (eager vs cudagraph).

"""

import torch
from einops import rearrange


@torch.compile(fullgraph=True)
def _compiled_rearrange_mixed_qkv(
    mixed_qkv: torch.Tensor,
    key_dim: int,
    value_dim: int,
    tp_size: int,
    head_k_dim: int,
    head_v_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split packed qkv into contiguous (1, seq, heads, dim) tensors.

    The original code used ``rearrange(x, "l (h d) -> 1 l h d", d=...)``
    followed by ``.contiguous()`` on each tensor.  This version flattens
    all three splits into a single buffer via ``torch.cat`` so that
    torch.compile emits one Triton copy kernel instead of three separate
    contiguous() calls.
    """
    seq_len = mixed_qkv.shape[0]
    q_dim = key_dim // tp_size
    k_dim = key_dim // tp_size
    v_dim = value_dim // tp_size

    query, key, value = torch.split(mixed_qkv, [q_dim, k_dim, v_dim], dim=-1)

    fused = torch.cat([query.reshape(-1), key.reshape(-1), value.reshape(-1)], dim=0)

    q_size = seq_len * q_dim
    k_size = seq_len * k_dim

    q_contig = fused[0:q_size]
    k_contig = fused[q_size : q_size + k_size]
    v_contig = fused[q_size + k_size :]

    query = q_contig.view(1, seq_len, -1, head_k_dim)
    key = k_contig.view(1, seq_len, -1, head_k_dim)
    value = v_contig.view(1, seq_len, -1, head_v_dim)

    return query, key, value


def _rearrange_mixed_qkv(
    mixed_qkv: torch.Tensor,
    key_dim: int,
    value_dim: int,
    tp_size: int,
    head_k_dim: int,
    head_v_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    query, key, value = torch.split(
        mixed_qkv,
        [
            key_dim // tp_size,
            key_dim // tp_size,
            value_dim // tp_size,
        ],
        dim=-1,
    )
    query, key = map(
        lambda x: rearrange(x, "l (h d) -> 1 l h d", d=head_k_dim),
        (query, key),
    )
    value = rearrange(value, "l (h d) -> 1 l h d", d=head_v_dim)
    return query.contiguous(), key.contiguous(), value.contiguous()


def rearrange_mixed_qkv(
    mixed_qkv: torch.Tensor | None,
    key_dim: int,
    value_dim: int,
    tp_size: int,
    head_k_dim: int,
    head_v_dim: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    if mixed_qkv is None:
        return None, None, None

    # TODO: Find a better way to initialize the function so that
    # we always use the torch compiled function
    # For now, we are mixing two kernels implementation to
    # get the best out of both worlds.
    if mixed_qkv.is_cuda and torch.cuda.is_current_stream_capturing():
        return _rearrange_mixed_qkv(
            mixed_qkv,
            key_dim,
            value_dim,
            tp_size,
            head_k_dim,
            head_v_dim,
        )

    return _compiled_rearrange_mixed_qkv(
        mixed_qkv,
        key_dim,
        value_dim,
        tp_size,
        head_k_dim,
        head_v_dim,
    )
