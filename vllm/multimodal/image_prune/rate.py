# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# RATE (Redundancy-Aware Token Eviction) image token pruning.
# Tokens are scored by their similarity to a small pivot set of
# representative tokens, and the most redundant ones are evicted.

from functools import lru_cache

import torch
import torch.nn.functional as F


def compute_retained_tokens_count(num_tokens: int, q: float) -> int:
    """Number of image tokens retained after RATE pruning.

    The target is `(1 - q) * num_tokens`; at least one token is always
    kept so the placeholder for the image never becomes empty.
    """
    return max(1, int(num_tokens * (1 - q)))


def compute_retention_mask_rate(
    embeds: torch.Tensor,
    q: float,
    pivot_block_size: int = 32,
    skip_ratio: float = 0.01,
    num_proj: int = 8,
) -> torch.Tensor:
    """Compute the RATE retention mask for a single image.

    Args:
        embeds: `(num_tokens, hidden_size)` post-ViT image token features.
        q: Pruning fraction in `[0, 1)`; retention ratio is `1 - q`.
        pivot_block_size: Minimum token count required to run RATE; smaller
            images fall back to deterministic prefix retention.
        skip_ratio: Fraction of extreme-scoring tokens skipped when
            normalizing projection scores.
        num_proj: Number of random projection directions.

    Returns:
        Flat bool tensor of shape `(num_tokens,)`, True for retained
        tokens. The True count equals `compute_retained_tokens_count` so
        placeholders sized at prompt-processing time match exactly.

    """
    num_tokens = embeds.shape[0]
    device = embeds.device
    if num_tokens == 0:
        return torch.zeros(0, dtype=torch.bool, device=device)

    retain_num_tokens = compute_retained_tokens_count(num_tokens, q)
    if retain_num_tokens >= num_tokens:
        return torch.ones(num_tokens, dtype=torch.bool, device=device)

    scores = _get_diversity_scores(
        embeds,
        pivot_block_size=pivot_block_size,
        skip_num=max(1, int(skip_ratio * num_tokens)),
        num_proj=num_proj,
    )
    # Lowest similarity to the pivot set == most diverse: retain first.
    order = torch.argsort(scores, descending=False, stable=True)
    retention_mask = torch.zeros(num_tokens, dtype=torch.bool, device=device)
    retention_mask[order[:retain_num_tokens]] = True
    return retention_mask


@lru_cache
def _get_random_projections(
    num_proj: int, embed_dim: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Unit-norm random projection matrix of shape `(num_proj, embed_dim)`."""
    return F.normalize(
        torch.randn(num_proj, embed_dim, dtype=dtype, device=device), p=2, dim=-1
    )


def _get_diversity_scores(
    embeds: torch.Tensor,
    pivot_block_size: int,
    skip_num: int,
    num_proj: int,
) -> torch.Tensor:
    """Similarity-to-pivot score per token; lower means more diverse.

    Images with fewer than `pivot_block_size` tokens (and degenerate ones
    with no score spread) get constant scores, which makes RATE retain a
    deterministic prefix of the tokens.
    """
    num_tokens = embeds.shape[0]
    if num_tokens < pivot_block_size:
        return torch.ones(num_tokens, dtype=embeds.dtype, device=embeds.device)

    seq_norm_embeds = F.normalize(embeds, p=2, dim=-1)
    rand_proj = _get_random_projections(
        num_proj=num_proj,
        embed_dim=embeds.shape[-1],
        dtype=embeds.dtype,
        device=embeds.device,
    )
    init_scores = torch.einsum("ik,jk->i", seq_norm_embeds, rand_proj)
    sorted_values, sorted_idx = init_scores.sort(dim=-1, stable=True)

    denom = sorted_values[skip_num] - sorted_values[-skip_num]
    if denom <= 0:
        return torch.zeros(num_tokens, dtype=embeds.dtype, device=embeds.device)

    # Quantize the projection scores so roughly `num_tokens /
    # pivot_block_size` distinct buckets exist; bucket boundaries form the
    # pivot set.
    q_scale = (num_tokens / pivot_block_size) / denom
    sorted_q_values = (sorted_values * q_scale).round()
    pivot_mask = sorted_q_values[:-1] > sorted_q_values[1:]
    pivot_idx = torch.cat([sorted_idx[:1], sorted_idx[1:][pivot_mask]])

    pivot_set = seq_norm_embeds[pivot_idx]
    scores = torch.einsum("ik,jk->i", seq_norm_embeds, pivot_set)
    scores /= pivot_set.shape[0]
    scores[pivot_idx] = 0.0
    return scores
