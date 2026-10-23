# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Entropy-gated deferred verification for FLy speculative decoding.

FLy defers a high-entropy rejection when the next ``W`` draft tokens would be
accepted natively. ModelRunnerV1 encodes these decisions in a pre-pass by
replacing target argmaxes or uniform samples. ModelRunnerV2 applies them
inside its rejection kernel.
"""

import torch


def compute_fly_entropy(
    values: torch.Tensor, entropy_top_k: int, *, from_logits: bool = False
) -> torch.Tensor:
    """Compute FLy's top-k entropy from processed target probabilities or logits."""
    if values.ndim != 2:
        raise ValueError("FLy expects 2-D target probabilities or logits")
    if values.shape[-1] == 0:
        raise ValueError("FLy requires a non-empty target vocabulary")

    if entropy_top_k <= 0:
        raise ValueError("fly_entropy_top_k must be greater than zero")
    top_k = min(entropy_top_k, values.shape[-1])
    values = values.to(torch.float32)
    top_values = torch.topk(values, k=top_k, dim=-1).values
    if from_logits:
        top_log_probs = top_values - values.logsumexp(dim=-1, keepdim=True)
        top_probs = top_log_probs.exp()
    else:
        top_probs = top_values
        top_log_probs = top_probs.log()

    entropy_terms = torch.where(
        top_probs > 0,
        top_probs * top_log_probs,
        torch.zeros_like(top_probs),
    )
    return -entropy_terms.sum(dim=-1)
