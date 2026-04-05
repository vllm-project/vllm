# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utilities for pruning visual tokens based on attention scores.

Two pruning strategies:
  1. Dominant-only: keep top-k tokens by importance score.
  2. Dominant + merge: keep dominant tokens and merge remaining contextual
     tokens into uniformly sampled anchors via cosine similarity.
"""

import torch


def prune_visual_tokens_dominant_only(
    embeddings: torch.Tensor,
    scores: torch.Tensor,
    pruning_rate: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep top-k visual tokens ranked by attention importance scores.

    Args:
        embeddings: [N, hidden_size] post-merger embeddings for one image.
        scores: [N] importance scores per token.
        pruning_rate: fraction of tokens to prune, in [0, 1).
                      0.0 means no pruning; 0.6 means prune 60% of tokens.

    Returns:
        pruned_embeddings: [K, hidden_size].
        keep_indices: [K] kept token indices in original order.
    """
    N = embeddings.shape[0]
    if pruning_rate <= 0.0:
        return embeddings, torch.arange(N, device=embeddings.device)

    keep_num = max(1, int(N * (1.0 - pruning_rate)))
    _, topk_indices = torch.topk(scores, keep_num, sorted=False)
    keep_indices = topk_indices.sort().values
    return embeddings[keep_indices], keep_indices


def prune_visual_tokens_with_merge(
    embeddings: torch.Tensor,
    scores: torch.Tensor,
    pruning_rate: float,
    merge_ratio: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dominant token selection + contextual token merging.

    Selects dominant tokens by score, then uniformly samples anchor tokens
    from the remaining contextual tokens. Non-anchor contextual tokens are
    merged into their most similar anchor via cosine similarity.

    Args:
        embeddings: [N, hidden_size] post-merger embeddings for one image.
        scores: [N] importance scores per token.
        pruning_rate: fraction of tokens to prune, in [0, 1).
                      0.0 means no pruning; 0.6 means prune 60% of tokens.
        merge_ratio: fraction reserved for merge anchor tokens (default 0.1).

    Returns:
        pruned_embeddings: [K, hidden_size].
        keep_indices: [K] kept token indices in original order.
    """
    N = embeddings.shape[0]
    if pruning_rate <= 0.0:
        return embeddings, torch.arange(N, device=embeddings.device)

    keep_ratio = 1.0 - pruning_rate
    keep_total = max(1, int(N * keep_ratio))

    dominant_ratio = keep_ratio - merge_ratio
    if dominant_ratio <= 0:
        return prune_visual_tokens_dominant_only(embeddings, scores, pruning_rate)

    anchor_num = max(1, int(N * merge_ratio))
    anchor_num = min(anchor_num, keep_total - 1)
    dominant_num = keep_total - anchor_num

    _, topk_indices = torch.topk(scores, dominant_num, sorted=False)
    dominant_mask = torch.zeros(N, dtype=torch.bool, device=embeddings.device)
    dominant_mask[topk_indices] = True

    contextual_indices = torch.where(~dominant_mask)[0]
    if contextual_indices.numel() == 0:
        keep_indices = topk_indices.sort().values
        return embeddings[keep_indices], keep_indices

    # Uniformly sample anchors from contextual tokens
    step = max(1, contextual_indices.size(0) // anchor_num)
    anchor_pos_in_ctx = torch.arange(
        0, contextual_indices.size(0), step, device=embeddings.device
    )[:anchor_num]
    anchor_global_indices = contextual_indices[anchor_pos_in_ctx]

    is_anchor_in_ctx = torch.zeros(
        contextual_indices.size(0), dtype=torch.bool, device=embeddings.device
    )
    is_anchor_in_ctx[anchor_pos_in_ctx] = True

    # Cosine similarity between non-anchor contextual tokens and anchors
    ctx_embeds = embeddings[contextual_indices]
    ctx_norm = ctx_embeds / ctx_embeds.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    anchor_norm = ctx_norm[anchor_pos_in_ctx]
    to_merge_norm = ctx_norm[~is_anchor_in_ctx]

    if to_merge_norm.size(0) > 0:
        similarity = torch.mm(to_merge_norm, anchor_norm.t())
        assignments = similarity.argmax(dim=1)
        assign_onehot = torch.zeros(
            to_merge_norm.size(0),
            anchor_num,
            dtype=embeddings.dtype,
            device=embeddings.device,
        )
        assign_onehot.scatter_(1, assignments.unsqueeze(-1), 1)
        counts = assign_onehot.sum(dim=0).clamp(min=1).unsqueeze(-1)

        to_merge_embeds = ctx_embeds[~is_anchor_in_ctx]
        aggregated = (
            assign_onehot.t().to(to_merge_embeds.dtype) @ to_merge_embeds
        ) / counts.to(to_merge_embeds.dtype)
        anchor_embeds = embeddings[anchor_global_indices] + aggregated
    else:
        anchor_embeds = embeddings[anchor_global_indices]

    # Build result: dominant tokens (original) + merged anchor tokens
    result = embeddings.clone()
    result[anchor_global_indices] = anchor_embeds

    keep_mask = dominant_mask.clone()
    keep_mask[anchor_global_indices] = True
    keep_indices = torch.where(keep_mask)[0]

    return result[keep_indices], keep_indices
