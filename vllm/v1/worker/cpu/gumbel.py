# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm import _custom_ops as ops


def gumbel_sample(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
    seed: torch.Tensor,
    pos: torch.Tensor,
    apply_temperature: bool,
    is_drafting: bool,
    logits_cache: torch.Tensor | None = None,
    logits_cache_col: torch.Tensor | None = None,
    use_fp64: bool = False,
    logits_cache_source: torch.Tensor | None = None,
) -> torch.Tensor:
    if (
        logits_cache is not None
        or logits_cache_col is not None
        or logits_cache_source is not None
    ):
        raise NotImplementedError(
            "CPU Gumbel sampling does not yet support the draft-logits cache."
        )

    return ops.gumbel_sample_cpu(
        logits,
        expanded_idx_mapping.contiguous(),
        temperature,
        seed,
        pos.contiguous(),
        apply_temperature,
        is_drafting,
        use_fp64,
    )
