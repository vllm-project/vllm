# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paged FP8 scoring for HY4 Flash's compressed IndexPool."""

import torch

from vllm.utils.deep_gemm import (
    fp8_fp4_paged_mqa_logits,
    get_paged_mqa_logits_metadata,
)


def score_indexpool_groups(
    q: torch.Tensor,
    q_scales: torch.Tensor | None,
    cache: torch.Tensor,
    weights: torch.Tensor,
    lengths: torch.Tensor,
    table: torch.Tensor,
    width: int,
    num_sms: int,
    indices: torch.Tensor | None,
    *,
    use_triton: bool,
) -> torch.Tensor:
    """Score compressed groups without changing the IndexPool cache layout."""
    kv = cache.unsqueeze(2)
    if use_triton:
        from vllm.v1.attention.ops.mqa_logits_triton import fp8_paged_mqa_logits_triton

        # On Ada the query scale is folded into weights, and each cache
        # page stores FP8 keys followed by FP32 scales. The block table
        # already has one row per query, so no request-owner map is needed.
        assert q_scales is None and indices is None
        return fp8_paged_mqa_logits_triton(q, kv, weights, lengths, table, width)

    schedule = get_paged_mqa_logits_metadata(
        lengths, cache.shape[1], num_sms, indices=indices
    )
    return fp8_fp4_paged_mqa_logits(
        (q, q_scales),
        kv,
        weights,
        lengths,
        table,
        schedule,
        max_model_len=width,
        indices=indices,
    )
