# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Layout permutations for FlashMLA's fused sparse-attention kernel.

The fused kernel reads Q with 16-element head-dim chunks interleaved across
heads and writes O with 32-element chunks interleaved across the 8 heads of a
``wo_a`` group. Every permutation ``perm`` here satisfies
``fused = standard[perm]``; ``wq_b`` rows and ``wo_a`` columns are permuted
once at load time so the GEMMs produce and consume the fused layouts directly.
"""

import torch

HEAD_DIM = 512
Q_CHUNK = 16
O_CHUNK = 32
WV_GROUP_SIZE = 8


def q_fused_permutation(num_heads: int, head_dim: int = HEAD_DIM) -> torch.Tensor:
    """``fused[(d // 16) * (H * 16) + h * 16 + d % 16] = standard[h * D + d]``."""
    h = torch.arange(num_heads).view(num_heads, 1)
    d = torch.arange(head_dim).view(1, head_dim)
    fused_index = (d // Q_CHUNK) * (num_heads * Q_CHUNK) + h * Q_CHUNK + d % Q_CHUNK
    perm = torch.empty(num_heads * head_dim, dtype=torch.long)
    perm[fused_index.reshape(-1)] = torch.arange(num_heads * head_dim)
    return perm


def o_fused_permutation(
    heads_per_group: int = WV_GROUP_SIZE, head_dim: int = HEAD_DIM
) -> torch.Tensor:
    """``fused[(c * G + h) * 32 + j] = standard[h * D + c * 32 + j]`` per group."""
    h = torch.arange(heads_per_group).view(-1, 1, 1)
    c = torch.arange(head_dim // O_CHUNK).view(1, -1, 1)
    j = torch.arange(O_CHUNK).view(1, 1, -1)
    fused_index = (c * heads_per_group + h) * O_CHUNK + j
    perm = torch.empty(heads_per_group * head_dim, dtype=torch.long)
    perm[fused_index.reshape(-1)] = torch.arange(heads_per_group * head_dim)
    return perm


def o_fused_chunk_permutation(
    heads_per_group: int = WV_GROUP_SIZE, head_dim: int = HEAD_DIM
) -> torch.Tensor:
    """Per-32-element-chunk form of :func:`o_fused_permutation` (for scales)."""
    return o_fused_permutation(heads_per_group, head_dim)[::O_CHUNK] // O_CHUNK


def permute_q_to_fused(q: torch.Tensor) -> torch.Tensor:
    """``[N, H, D]`` standard layout -> the same nominal shape, fused layout."""
    n, h, d = q.shape
    perm = q_fused_permutation(h, d).to(q.device)
    return q.reshape(n, h * d)[:, perm].view(n, h, d)


def _bytes_view(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.uint8) if t.element_size() == 1 else t


def permute_wq_b_(
    weight: torch.Tensor, weight_scale: torch.Tensor, num_local_heads: int
) -> None:
    """Permute the rows of an MXFP8 ``wq_b`` shard and its per-row scale in place."""
    head_dim = weight.shape[0] // num_local_heads
    perm = q_fused_permutation(num_local_heads, head_dim).to(weight.device)
    for t in (weight, weight_scale):
        b = _bytes_view(t)
        b.copy_(b[perm])


def permute_wo_a_(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    heads_per_group: int = WV_GROUP_SIZE,
) -> None:
    """Permute the input columns of an MXFP8 ``wo_a`` shard and its per-32 scale."""
    head_dim = weight.shape[1] // heads_per_group
    perm = o_fused_permutation(heads_per_group, head_dim).to(weight.device)
    w = _bytes_view(weight)
    w.copy_(w[:, perm])
    chunk_perm = o_fused_chunk_permutation(heads_per_group, head_dim)
    s = _bytes_view(weight_scale)
    s.copy_(s[:, chunk_perm.to(weight.device)])
