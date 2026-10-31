# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Global top-k merge for the DSA indexer under decode context parallelism.

Under DCP every rank holds 1/N of a request's indexer keys, so each rank's
top-k only covers its own shard. A token in the global top-k is also in the
local top-k of the rank that owns it, because at most ``topk - 1`` tokens
rank above it globally, and so at most that many on its own rank. Exchanging
only the local candidates is therefore exact.

This module does the exchange with plain torch ops so that it runs on every
platform. Each candidate is packed into one int64 key that orders by score
and then by the lower global token id, which is the tie rule of the
single-rank top-k. One all-gather of the keys and one ``torch.topk`` select
the global winners.
"""

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm.distributed.parallel_state import GroupCoordinator

# Key layout (int64, always >= 0 for a valid candidate):
#   bits 31..62: the fp32 score, transformed so that integer order is float
#                order
#   bits  0..30: _ID_MASK - global_id, so that the lower id wins a tie
# Invalid candidates use -1, which is below every valid key.
_ID_BITS = 31
_ID_MASK = (1 << _ID_BITS) - 1
_INVALID_KEY = -1

# Bound the all-gather buffer of a long prefill chunk. Each row gathers
# ``dcp_world_size * topk`` int64 keys, so 1024 rows of GLM-5.x at DCP8 is
# 128 MiB.
_MAX_ROWS_PER_EXCHANGE = 1024


def _ordered_score_bits(scores: torch.Tensor) -> torch.Tensor:
    """Map fp32 scores to non-negative int64 values with the same order."""
    # NaN must never win, and -0.0 must compare equal to 0.0.
    scores = torch.nan_to_num(scores, nan=float("-inf")) + 0.0
    bits = scores.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    negative = bits >= 0x80000000
    # Negative floats: flip every bit. Positive floats: flip only the sign.
    return torch.where(negative, bits ^ 0xFFFFFFFF, bits ^ 0x80000000)


def pack_dcp_topk_keys(
    logits: torch.Tensor,
    topk_indices: torch.Tensor,
    dcp_rank: int,
    dcp_world_size: int,
    cp_interleave: int,
    row_starts: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pack this rank's local top-k into int64 keys with global token ids.

    ``topk_indices`` holds positions into this rank's KV shard, ``-1`` for
    padding. Column ``row_starts[r] + j`` of ``logits`` holds the score of
    local position ``j`` of row ``r``.
    """
    valid = topk_indices >= 0
    local_idx = topk_indices.clamp(min=0).to(torch.int64)
    score_col = local_idx
    if row_starts is not None:
        score_col = score_col + row_starts.to(torch.int64).view(-1, 1)
    if logits.shape[1] > 0:
        score_col = score_col.clamp(max=logits.shape[1] - 1)
        scores = logits.gather(1, score_col)
    else:
        scores = torch.full_like(local_idx, float("-inf"), dtype=torch.float32)
    # Local position j of this rank is global position
    # (j // I) * (N * I) + rank * I + j % I for interleave I and N ranks.
    global_id = (
        (local_idx // cp_interleave) * (dcp_world_size * cp_interleave)
        + dcp_rank * cp_interleave
        + local_idx % cp_interleave
    )
    key = (_ordered_score_bits(scores.float()) << _ID_BITS) | (_ID_MASK - global_id)
    return torch.where(valid, key, _INVALID_KEY)


def select_topk_from_keys(keys: torch.Tensor, topk: int, out: torch.Tensor) -> None:
    """Write the global token ids of the ``topk`` largest keys into ``out``."""
    # sorted=True puts the valid ids first. The ROCm backend reads only the
    # first min(context, topk) columns of a row.
    top = keys.topk(topk, dim=1, sorted=True).values
    ids = _ID_MASK - (top & _ID_MASK)
    out.copy_(torch.where(top >= 0, ids, -1).to(out.dtype))


def merge_dcp_topk_global(
    logits: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_tokens: int,
    dcp_rank: int,
    dcp_world_size: int,
    cp_interleave: int,
    group: "GroupCoordinator",
    row_starts: torch.Tensor | None = None,
) -> None:
    """Replace each rank's local top-k with the global top-k, in place.

    After the call ``topk_indices`` holds global (request-level) token ids,
    ``-1`` for padding, and is the same on every rank of ``group``. The
    attention backend maps them to this rank's own KV slots. Every rank of
    the group must call this with the same number of rows.
    """
    if dcp_world_size <= 1:
        return
    assert topk_indices.shape[1] == topk_tokens
    num_rows = topk_indices.shape[0]
    for start in range(0, num_rows, _MAX_ROWS_PER_EXCHANGE):
        end = min(start + _MAX_ROWS_PER_EXCHANGE, num_rows)
        local_keys = pack_dcp_topk_keys(
            logits[start:end],
            topk_indices[start:end],
            dcp_rank,
            dcp_world_size,
            cp_interleave,
            None if row_starts is None else row_starts[start:end],
        )
        keys = group.all_gather(local_keys.contiguous(), dim=1)
        select_topk_from_keys(keys, topk_tokens, topk_indices[start:end])
