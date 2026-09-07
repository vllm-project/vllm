# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Torch equivalents of the ``gpu/input_batch.py`` Triton kernels.

Ragged per-request work is expressed with run-length indexing rather than a
host loop: a loop reads request metadata back with ``.item()``, and those
syncs, not the arithmetic, are what cost a measurable slice of the step.
"""

from typing import Any

import torch


def _run_offsets(counts: torch.Tensor) -> tuple[int, torch.Tensor]:
    """Total length, and each element's offset within its own run."""
    total = int(counts.sum())
    starts = torch.cumsum(counts, 0) - counts
    offsets = torch.arange(total, device=counts.device) - torch.repeat_interleave(
        starts, counts
    )
    return total, offsets


def prepare_prefill_inputs(
    grid: tuple[int, ...],
    input_ids: torch.Tensor,
    next_prefill_tokens: torch.Tensor,
    next_prefill_tokens_stride: int,
    num_lookahead: int,
    idx_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    all_token_ids: torch.Tensor,
    all_token_ids_stride: int,
    prefill_len: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    **kwargs: Any,
) -> None:
    num_reqs = idx_mapping.shape[0]
    req = idx_mapping[:num_reqs].long()
    total_prefill = prefill_len[req].long()
    num_computed = num_computed_tokens[req].long()

    # Every request is past its prompt on a decode step, which is the common
    # case; leaving early keeps it off the ragged path below.
    prefilling = (num_computed < total_prefill).nonzero(as_tuple=True)[0]
    if prefilling.numel() == 0:
        return

    req = req[prefilling]
    total_prefill = total_prefill[prefilling]
    num_computed = num_computed[prefilling]
    starts = query_start_loc[prefilling].long()
    query_lens = query_start_loc[prefilling + 1].long() - starts

    total, offsets = _run_offsets(query_lens)
    rows = torch.repeat_interleave(req, query_lens)
    cols = torch.repeat_interleave(num_computed, query_lens) + offsets
    # These buffers are a mix of int32 and int64, and an indexed write, unlike
    # the slice it replaces, will not cast for us.
    input_ids[torch.repeat_interleave(starts, query_lens) + offsets] = all_token_ids[
        rows, cols
    ].to(input_ids.dtype)

    if num_lookahead > 0:
        # [num_lookahead, num_prefilling]: the tokens the next step will need.
        pos = (num_computed + query_lens).unsqueeze(0) + torch.arange(
            num_lookahead, device=input_ids.device
        ).unsqueeze(1)
        inside = pos < total_prefill.unsqueeze(0)
        gathered = all_token_ids[
            req.unsqueeze(0).expand_as(pos),
            pos.clamp_max(all_token_ids.shape[1] - 1),
        ]
        next_prefill_tokens[:, req] = torch.where(
            inside, gathered, torch.zeros_like(gathered)
        ).to(next_prefill_tokens.dtype)


def prepare_pos_seq_lens(
    grid: tuple[int, ...],
    pos: torch.Tensor,
    seq_lens: torch.Tensor,
    idx_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    max_num_reqs: int,
    **kwargs: Any,
) -> None:
    # The kernel spends its last program on this padding, hence grid - 1.
    num_reqs = grid[0] - 1
    seq_lens[num_reqs:max_num_reqs] = 0

    req = idx_mapping[:num_reqs].long()
    num_computed = num_computed_tokens[req].long()
    starts = query_start_loc[:num_reqs].long()
    query_lens = query_start_loc[1 : num_reqs + 1].long() - starts

    seq_lens[:num_reqs] = (num_computed + query_lens).to(seq_lens.dtype)

    total, offsets = _run_offsets(query_lens)
    dst = torch.repeat_interleave(starts, query_lens) + offsets
    pos[dst] = (torch.repeat_interleave(num_computed, query_lens) + offsets).to(
        pos.dtype
    )


def combine_sampled_and_draft_tokens(
    grid: tuple[int, ...],
    input_ids: torch.Tensor,
    idx_mapping: torch.Tensor,
    last_sampled_tokens: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    prefill_len: torch.Tensor,
    draft_tokens: torch.Tensor,
    draft_tokens_stride: int,
    cu_num_logits: torch.Tensor,
    logits_indices: torch.Tensor,
    NUM_NEW_SAMPLED_TOKENS: int = 1,
    **kwargs: Any,
) -> None:
    num_reqs = idx_mapping.shape[0]
    req = idx_mapping[:num_reqs].long()
    starts = cu_num_logits[:num_reqs].long()
    num_logits = cu_num_logits[1 : num_reqs + 1].long() - starts
    query_end = query_start_loc[1 : num_reqs + 1].long()
    logits_start = query_end - num_logits

    total, offsets = _run_offsets(num_logits)
    dst = torch.repeat_interleave(starts, num_logits) + offsets
    logits_indices[dst] = (
        torch.repeat_interleave(logits_start, num_logits) + offsets
    ).to(logits_indices.dtype)

    # Prompt-tail slots keep the prompt's tokens; only generated slots are
    # rewritten, so a chunked prefill contributes nothing here.
    generating = seq_lens[:num_reqs].long() > prefill_len[req].long()
    if NUM_NEW_SAMPLED_TOKENS > 0:
        resumed = generating & (
            seq_lens[:num_reqs].long() - num_logits >= prefill_len[req].long()
        )
        rows = resumed.nonzero(as_tuple=True)[0]
        # last_sampled_tokens is [max_num_reqs, 1]; flatten it so one token
        # lands in one slot.
        last = last_sampled_tokens.view(last_sampled_tokens.shape[0], -1)[:, 0]
        input_ids[logits_start[rows]] = last[req[rows]].to(input_ids.dtype)

    num_draft = num_logits - NUM_NEW_SAMPLED_TOKENS
    drafted = (generating & (num_draft > 0)).nonzero(as_tuple=True)[0]
    if drafted.numel():
        counts = num_draft[drafted]
        _, off = _run_offsets(counts)
        src_rows = torch.repeat_interleave(req[drafted], counts)
        ends = torch.repeat_interleave(query_end[drafted] - counts, counts)
        input_ids[ends + off] = draft_tokens[src_rows, off].to(input_ids.dtype)


def get_num_sampled_and_rejected(
    grid: tuple[int, ...],
    num_sampled: torch.Tensor,
    num_rejected: torch.Tensor,
    seq_lens: torch.Tensor,
    cu_num_logits: torch.Tensor,
    idx_mapping: torch.Tensor,
    prefill_len: torch.Tensor,
    **kwargs: Any,
) -> None:
    num_reqs = idx_mapping.shape[0]
    req = idx_mapping[:num_reqs].long()
    chunked_prefilling = seq_lens[:num_reqs] < prefill_len[req]

    sampled = torch.where(chunked_prefilling, 0, num_sampled[:num_reqs].to(torch.int64))
    num_logits = cu_num_logits[1 : num_reqs + 1] - cu_num_logits[:num_reqs]
    rejected = torch.where(chunked_prefilling, 0, num_logits.to(torch.int64) - sampled)

    num_sampled[:num_reqs] = sampled.to(num_sampled.dtype)
    num_rejected[:num_reqs] = rejected.to(num_rejected.dtype)


def post_update(
    grid: tuple[int, ...],
    idx_mapping: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    last_sampled_tokens: torch.Tensor,
    output_bin_counts: torch.Tensor | None,
    output_bin_counts_stride: int,
    sampled_tokens: torch.Tensor,
    sampled_tokens_stride: int,
    num_sampled: torch.Tensor,
    num_rejected: torch.Tensor,
    query_start_loc: torch.Tensor | None,
    all_token_ids: torch.Tensor,
    all_token_ids_stride: int,
    total_len: torch.Tensor,
    **kwargs: Any,
) -> None:
    num_reqs = idx_mapping.shape[0]
    batch = (idx_mapping[:num_reqs] >= 0).nonzero(as_tuple=True)[0]
    if batch.numel() == 0:
        return
    req = idx_mapping[batch].long()

    counts = num_sampled[batch].long()
    accepted = (counts > 0).nonzero(as_tuple=True)[0]
    if accepted.numel():
        rows = req[accepted]
        src = batch[accepted]
        counts = counts[accepted]
        lengths = total_len[rows].long()

        _, offsets = _run_offsets(counts)
        tokens = sampled_tokens[torch.repeat_interleave(src, counts), offsets]
        all_token_ids[
            torch.repeat_interleave(rows, counts),
            torch.repeat_interleave(lengths, counts) + offsets,
        ] = tokens.to(all_token_ids.dtype)
        # A flat view of the [max_num_reqs, 1] buffer, so this writes one token
        # per request.
        last = last_sampled_tokens.view(last_sampled_tokens.shape[0], -1)[:, 0]
        last[rows] = sampled_tokens[src, counts - 1].to(last.dtype)
        total_len[rows] = (lengths + counts).to(total_len.dtype)
        if output_bin_counts is not None:
            output_bin_counts.index_put_(
                (torch.repeat_interleave(rows, counts), tokens.long()),
                torch.ones_like(tokens, dtype=output_bin_counts.dtype),
                accumulate=True,
            )

    query_lens: torch.Tensor | int = 0
    if query_start_loc is not None:
        query_lens = query_start_loc[batch + 1].long() - query_start_loc[batch].long()
    delta = query_lens - num_rejected[batch].long()
    num_computed_tokens.index_add_(0, req, delta.to(num_computed_tokens.dtype))


def post_update_num_computed_tokens(
    grid: tuple[int, ...],
    idx_mapping: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    query_start_loc: torch.Tensor,
    **kwargs: Any,
) -> None:
    num_reqs = idx_mapping.shape[0]
    query_lens = query_start_loc[1 : num_reqs + 1] - query_start_loc[:num_reqs]
    num_computed_tokens.index_add_(
        0, idx_mapping[:num_reqs].long(), query_lens.to(num_computed_tokens.dtype)
    )


def expand_idx_mapping(
    grid: tuple[int, ...],
    idx_mapping: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    expanded_local_pos: torch.Tensor,
    cu_num_logits: torch.Tensor,
    **kwargs: Any,
) -> None:
    num_reqs = idx_mapping.shape[0]
    starts = cu_num_logits[:num_reqs].long()
    counts = cu_num_logits[1 : num_reqs + 1].long() - starts
    total = int(counts.sum())

    expanded_idx_mapping[:total] = torch.repeat_interleave(
        idx_mapping[:num_reqs], counts
    )
    expanded_local_pos[:total] = torch.arange(
        total, dtype=expanded_local_pos.dtype
    ) - torch.repeat_interleave(starts, counts).to(expanded_local_pos.dtype)
