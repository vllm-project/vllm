# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Redistribute head-sharded Engram rows to sequence-parallel token owners."""

from collections.abc import Callable

import torch


def exchange_heads_for_tokens(
    rows: torch.Tensor,
    tp_size: int,
    num_heads: int,
    exchange: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Return this rank's padded token chunk with all non-padding heads.

    Args:
        rows: All tokens and this rank's padded head shard, shaped [T, H/P, D].
        tp_size: Number of head and token owners.
        num_heads: Unpadded global head count.
        exchange: Equal-split all-to-all in rank order along the token dimension.
    """
    num_tokens, local_heads, dim = rows.shape
    if not num_tokens:
        return rows.new_empty((0, num_heads, dim))
    chunk = (num_tokens + tp_size - 1) // tp_size
    padding = chunk * tp_size - num_tokens
    if padding:
        rows = torch.nn.functional.pad(rows, (0, 0, 0, 0, 0, padding))
    else:
        rows = rows.contiguous()
    received = exchange(rows)
    return (
        received.view(tp_size, chunk, local_heads, dim)
        .permute(1, 0, 2, 3)
        .reshape(chunk, tp_size * local_heads, dim)[:, :num_heads]
    )
