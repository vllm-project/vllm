# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Entropy-gated deferred verification for FLy speculative decoding.

FLy defers a high-entropy rejection when the next `W` draft tokens would be
accepted natively. The rejection kernel applies these decisions.
"""

import math

import torch


def compute_fly_entropy(
    values: torch.Tensor, entropy_top_k: int, *, from_logits: bool = False
) -> torch.Tensor:
    """Normalized top-k entropy in `[0, 1]`.

    Takes the largest `k` target scores, renormalizes them into a
    distribution over those `k` candidates only, then divides the Shannon
    entropy by `log(k)` so that `fly_entropy_threshold` keeps the same
    meaning for every `fly_entropy_top_k`.
    """
    if values.ndim != 2:
        raise ValueError("FLy expects 2-D target probabilities or logits")
    if values.shape[-1] == 0:
        raise ValueError("FLy requires a non-empty target vocabulary")

    if entropy_top_k <= 0:
        raise ValueError("fly_entropy_top_k must be greater than zero")
    top_k = min(entropy_top_k, values.shape[-1])
    values = values.to(torch.float32)
    # k=1 is degenerate after renormalization (always [1.0]) and log(1)=0.
    if top_k <= 1:
        return torch.zeros(values.shape[0], dtype=torch.float32, device=values.device)

    top_values = torch.topk(values, k=top_k, dim=-1).values
    if from_logits:
        top_probs = torch.softmax(top_values, dim=-1)
    else:
        mass = top_values.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        top_probs = top_values / mass

    entropy_terms = torch.where(
        top_probs > 0,
        top_probs * top_probs.log(),
        torch.zeros_like(top_probs),
    )
    return (-entropy_terms.sum(dim=-1)) / math.log(top_k)
