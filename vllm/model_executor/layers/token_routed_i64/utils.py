# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright INL Dynamics / Complexity-ML
"""
Utility functions for I64 token-routed experts.
"""

import torch


def compute_expert_ids(
    token_ids: torch.Tensor,
    num_experts: int,
    vocab_size: int,
) -> torch.Tensor:
    """
    Compute deterministic expert assignment: expert_id = token_id % num_experts.

    Args:
        token_ids: (num_tokens,) — input token IDs
        num_experts: total number of experts
        vocab_size: vocabulary size for clamping

    Returns:
        expert_ids: (num_tokens,) long
    """
    clamped = token_ids.clamp(0, vocab_size - 1)
    return (clamped % num_experts).long()


def expert_token_counts(
    expert_ids: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """
    Count tokens per expert.

    Args:
        expert_ids: (num_tokens,) long
        num_experts: total number of experts

    Returns:
        counts: (num_experts,) long
    """
    return torch.bincount(expert_ids, minlength=num_experts)
