# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerical contracts shared by the MiniMax mono stages and parity tests."""

from __future__ import annotations

import torch


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
