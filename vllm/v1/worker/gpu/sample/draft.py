# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sampling draft tokens for speculative decoding."""

from typing import TYPE_CHECKING

import torch

from vllm.v1.worker.gpu.sample.greedy_argmax import greedy_argmax
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample

if TYPE_CHECKING:
    from vllm.v1.watermarking.spec_decode import DraftWatermarker
    from vllm.v1.worker.gpu.spec_decode.draft_logits import DraftLogitsCache


def sample_independently(
    logits: torch.Tensor,
    sample_src_positions: torch.Tensor,
    idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
    seeds: torch.Tensor,
    draft_step: torch.Tensor,
    draft_logits: "DraftLogitsCache | None",
    watermarker: "DraftWatermarker | None",
    use_fp64_gumbel: bool,
    num_steps: int = 1,
) -> torch.Tensor:
    """Sample a draft token from each row of `logits`, caching them in
    `draft_logits` for probabilistic rejection sampling."""
    if draft_logits is not None and watermarker is not None:
        return watermarker.sample(
            logits,
            idx_mapping,
            temperature,
            seeds,
            sample_src_positions,
            apply_temperature=True,
            is_drafting=True,
            logits_cache=draft_logits.values,
            logits_cache_col=draft_step,
            use_fp64=use_fp64_gumbel,
            num_steps=num_steps,
        )
    if draft_logits is not None:
        return gumbel_sample(
            logits,
            idx_mapping,
            temperature,
            seeds,
            sample_src_positions,
            apply_temperature=True,
            is_drafting=True,
            logits_cache=draft_logits.values,
            logits_cache_col=draft_step,
            use_fp64=use_fp64_gumbel,
            logits_cache_row_mapping=draft_logits.row_mapping,
        )
    return greedy_argmax(logits)
