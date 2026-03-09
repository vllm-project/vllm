# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable

import torch


def compute_prompt_scores_for_mode(
    logits: torch.Tensor,
    logprobs_mode: str,
    compute_logprobs_fn: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Select prompt-side score tensor based on logprobs_mode."""
    if logprobs_mode.endswith("logits"):
        return logits.to(torch.float32)
    return compute_logprobs_fn(logits)
