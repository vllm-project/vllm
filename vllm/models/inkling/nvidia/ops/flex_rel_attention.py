# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from collections.abc import Callable

import torch


def make_inkling_flex_score_mod(
    rel_logits: torch.Tensor,
    rel_extent: int,
) -> Callable[..., torch.Tensor]:
    """Return a FlexAttention score modifier for Inkling relative bias."""

    def score_mod_rel_bias(
        score: torch.Tensor,
        batch_index: torch.Tensor,
        head_index: torch.Tensor,
        logical_query_index: torch.Tensor,
        logical_kv_index: torch.Tensor,
        *,
        physical_q: torch.Tensor,
    ) -> torch.Tensor:
        del batch_index
        relative_distance = logical_query_index - logical_kv_index
        relative_index = torch.clamp(relative_distance, min=0, max=rel_extent - 1)
        safe_query_index = torch.clamp(
            physical_q,
            min=0,
            max=rel_logits.shape[0] - 1,
        )
        relative_bias = rel_logits[
            safe_query_index,
            head_index,
            relative_index,
        ].to(torch.float32)
        return score + torch.where(
            (relative_distance >= 0) & (relative_distance < rel_extent),
            relative_bias,
            torch.zeros_like(score),
        )

    return score_mod_rel_bias
