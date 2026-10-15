# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Vocab-parallel argmax reduction for DSpark drafting."""

from typing import Any

import torch

MAX_EXACT_FP32_TOKEN_ID = 1 << 24


def reduce_local_argmax(
    local_scores: torch.Tensor,
    *,
    vocab_start: int,
    vocab_end: int,
    vocab_size: int,
    tp_group: Any,
) -> torch.Tensor:
    """Reduce local vocabulary argmax candidates to global token IDs.

    Args:
        local_scores: Scores for this rank's padded vocabulary shard.
        vocab_start: First unpadded global vocabulary index on this rank.
        vocab_end: End of this rank's unpadded global vocabulary range.
        vocab_size: Size of the full vocabulary.
        tp_group: Tensor-parallel group for the LM head.

    Returns:
        The global argmax token ID for each row in ``local_scores``.

    """
    if vocab_size - 1 > MAX_EXACT_FP32_TOKEN_ID:
        raise ValueError(
            "DSpark local argmax packs token IDs into FP32, but vocabulary "
            f"size {vocab_size} exceeds the exact FP32 integer range."
        )

    valid_local_vocab = vocab_end - vocab_start
    local_scores = local_scores[..., :valid_local_vocab]
    local_values, local_indices = local_scores.max(dim=-1)
    global_indices = local_indices + vocab_start

    tp_size = tp_group.world_size
    if tp_size == 1:
        return global_indices.to(torch.int64)

    local_pair = torch.stack(
        (local_values.float(), global_indices.float()),
        dim=-1,
    )
    gathered = tp_group.all_gather(local_pair, dim=-1)
    gathered = gathered.view(*local_scores.shape[:-1], tp_size, 2)

    # TP ranks own contiguous vocabulary shards in increasing token order, so
    # the first matching rank preserves full-vocabulary argmax tie-breaking.
    winner_rank = gathered[..., 0].argmax(dim=-1, keepdim=True)
    return (
        gathered[..., 1].gather(dim=-1, index=winner_rank).squeeze(-1).to(torch.int64)
    )
