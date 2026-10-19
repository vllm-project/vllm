# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerical contracts shared by the MiniMax mono stages and parity tests."""

from __future__ import annotations

import torch


def select_index_blocks(
    token_scores: torch.Tensor,
    forced_blocks: torch.Tensor,
    *,
    top_k: int = 16,
) -> torch.Tensor:
    """Independent head-wise block selection; max is over tokens in a block."""
    if token_scores.ndim != 4 or token_scores.shape[-1] != 128:
        raise ValueError("token_scores must be [heads, rows, blocks, 128]")
    block_scores = token_scores.float().amax(dim=-1)
    if forced_blocks.shape != block_scores.shape:
        raise ValueError("forced_blocks must match [heads, rows, blocks]")
    forced_count = forced_blocks.sum(dim=-1)
    if bool((forced_count > top_k).any()):
        raise ValueError("forced blocks exceed top-k cardinality")
    block_scores = block_scores.masked_fill(forced_blocks, torch.inf)
    return torch.topk(block_scores, k=top_k, dim=-1, sorted=False).indices.to(
        torch.int32
    )


def gemma_norm_stage(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    variance = hidden.float().square().mean(dim=-1, keepdim=True)
    normalized = hidden.float() * torch.rsqrt(variance + epsilon)
    return (normalized * (1.0 + weight.float())).to(hidden.dtype)


def fp32_router_stage(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    correction_bias: torch.Tensor,
    *,
    top_k: int,
    routed_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = torch.mm(hidden.float(), weight.t())
    scores = logits.sigmoid()
    selected = torch.topk(scores + correction_bias, top_k, dim=-1, sorted=False)[1]
    selected_weights = scores.gather(1, selected)
    selected_weights /= selected_weights.sum(dim=-1, keepdim=True)
    selected_weights *= routed_scale
    return logits, selected_weights, selected.to(torch.int32)
