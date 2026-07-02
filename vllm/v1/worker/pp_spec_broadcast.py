# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PP sampled-token broadcast helpers for speculative decoding.

Under PP + async scheduling the last rank broadcasts its sampled token ids to the
other ranks (which never run the sampler) so they can advance positions and build
the next input batch. Without spec the broadcast carries shape ``[num_reqs, 1]``;
with MTP/EAGLE spec the sampler emits ``[num_reqs, num_spec + 1]`` (accepted
drafts + bonus, ``-1``-padded). These helpers keep the transport width-agnostic so
both cases share one path, and expose the per-request valid count the receiver
uses to advance each request by the right number of tokens (not always 1).

Kept in a CUDA-free module so the shape/transport logic is unit-testable over a
plain gloo CPU group.
"""

import torch
import torch.distributed as dist


def count_valid_sampled_tokens_per_req(sampled_token_ids: torch.Tensor) -> torch.Tensor:
    """Per-request count of valid sampled tokens in a ``[num_reqs, width]`` grid.

    Valid tokens are the non-``-1`` entries (rejected/padded positions are ``-1``);
    this mirrors the accepted-count idiom used elsewhere in the runner
    (``(sampled_token_ids != -1).sum(dim=1)``). For the non-spec width-1 case every
    row holds one real token, so the count is 1 per request.
    """
    return (sampled_token_ids != -1).sum(dim=1)


def select_latest_sampled_token_per_req(
    sampled_token_ids: torch.Tensor,
) -> torch.Tensor:
    """Per-request latest *valid* sampled token from a ``[num_reqs, width]`` grid.

    Rejected/padded positions are ``-1``; a request that advanced by ``v`` valid
    tokens has its latest token in the last valid column ``recv[i, v - 1]`` (reject
    ``v=1`` -> col 0; accept + bonus ``v=2`` -> col 1). This is the real value the
    non-last rank must persist as its next input instead of a ``-1`` placeholder
    (the C4 value-back-write that fixes the ``indexSelectSmallIndex`` break: every
    confirmed request has ``v >= 1`` so the result is never ``-1``).
    """
    counts = count_valid_sampled_tokens_per_req(sampled_token_ids)
    last_idx = (counts - 1).clamp(min=0)
    return sampled_token_ids.gather(1, last_idx.unsqueeze(1)).squeeze(1)


def gather_valid_sampled_tokens_per_req(
    sampled_token_ids: torch.Tensor,
) -> list[list[int]]:
    """Per-request list of ALL valid sampled tokens ``recv[i, 0:v]``, in order.

    ``select_latest_sampled_token_per_req`` keeps only the single latest token;
    the holistic C4 back-write needs every confirmed token (accepted drafts +
    bonus) so the non-last rank can fill the ``v`` ``token_ids_cpu`` positions the
    next step will read (indexed by ``num_computed_tokens``, which includes the
    spec tokens) — not just one slot. Valid entries are the leading non-``-1``
    columns; a fully padded row yields ``[]`` (advance the cursor by 0).
    """
    counts = count_valid_sampled_tokens_per_req(sampled_token_ids).tolist()
    rows = sampled_token_ids.tolist()
    return [row[:v] for row, v in zip(rows, counts)]


def num_computed_tokens_drift_correction(
    prev_num_draft_len: int, valid_sampled_count: int
) -> int:
    """Amount to subtract from the optimistic ``num_computed_tokens`` on a non-last
    rank to undo async spec-decode drift after a (partial) draft rejection.

    Async spec decode advances ``num_computed_tokens`` optimistically by
    ``1 (bonus) + prev_num_draft_len (drafts assumed accepted)``; the true advance is
    the broadcast ``valid_sampled_count`` (accepted drafts + the bonus). The
    difference is the number of optimistically-counted drafts that were actually
    rejected. On the last rank this correction is applied by the GPU kernel
    ``update_num_computed_tokens_for_batch_change`` from the sampler's valid count;
    the non-last rank never runs the sampler, so it reconstructs the same correction
    from the broadcast valid count instead — keeping rope/KV positions identical on
    every rank (the invariant: advance ``num_computed_tokens`` by the valid count).
    Non-negative (you cannot accept more drafts than were proposed); ``0`` when every
    draft was accepted or none were proposed.
    """
    return (1 + prev_num_draft_len) - valid_sampled_count


def broadcast_sampled_token_ids(
    sampled_token_ids: torch.Tensor, group, src: int
) -> None:
    """Broadcast the (possibly multi-column) sampled token ids from ``src``."""
    assert sampled_token_ids.dim() == 2, (
        f"expected 2-D [num_reqs, width], got {tuple(sampled_token_ids.shape)}"
    )
    dist.broadcast(sampled_token_ids, src=src, group=group)


def receive_sampled_token_ids(
    num_reqs: int,
    width: int,
    group,
    src: int,
    device,
    dtype: torch.dtype = torch.int32,
) -> torch.Tensor:
    """Receive a ``[num_reqs, width]`` sampled-token grid broadcast from ``src``."""
    recv = torch.empty((num_reqs, width), dtype=dtype, device=device)
    dist.broadcast(recv, src=src, group=group)
    return recv
