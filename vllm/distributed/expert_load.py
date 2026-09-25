# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def record_logical_expert_load(
    expert_ids,
    token_ids,
    mask,
    counts,
    num_valid_tokens,
    START,
    END,
    TOKEN_OFFSET,
    NUM_EXPERTS: tl.constexpr,
    NUM_BINS: tl.constexpr,
):
    num_valid = tl.load(num_valid_tokens)
    valid = (
        mask
        & (token_ids >= START)
        & (token_ids < END)
        & (token_ids - START + TOKEN_OFFSET < num_valid)
        & (expert_ids >= 0)
        & (expert_ids < NUM_EXPERTS)
    )
    # Aggregate within the CTA before atomics, especially for skewed decode.
    # Triton 3.7.1 can wrap out-of-range inputs into valid bins: exclude lanes
    # with an explicit mask, not an out-of-range sentinel value.
    histogram = tl.histogram(expert_ids.to(tl.int32), NUM_BINS, mask=valid)
    bins = tl.arange(0, NUM_BINS)
    tl.atomic_add(
        counts + bins,
        histogram.to(tl.int64),
        mask=(bins < NUM_EXPERTS) & (histogram != 0),
        sem="relaxed",
    )


@triton.jit(do_not_specialize=["NUM_TOKENS", "START", "END", "TOKEN_OFFSET"])
def _record_expert_load(
    ids,
    counts,
    num_valid_tokens,
    NUM_TOKENS,
    TOP_K: tl.constexpr,
    STRIDE_TOKEN: tl.constexpr,
    STRIDE_EXPERT: tl.constexpr,
    START,
    END,
    TOKEN_OFFSET,
    NUM_EXPERTS: tl.constexpr,
    NUM_BINS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tokens = offsets // TOP_K
    mask = tokens < NUM_TOKENS
    experts = tl.load(
        ids + tokens * STRIDE_TOKEN + (offsets % TOP_K) * STRIDE_EXPERT,
        mask=mask,
        other=-1,
    )
    record_logical_expert_load(
        experts,
        tokens,
        mask,
        counts,
        num_valid_tokens,
        START,
        END,
        TOKEN_OFFSET,
        NUM_EXPERTS,
        NUM_BINS,
    )


@triton.jit
def finish_expert_load_iteration(
    counts,
    summary,
    trace,
    SIZE: tl.constexpr,
    TRACE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < SIZE
    values = tl.load(counts + offsets, mask=mask, other=0)
    old = tl.load(summary + offsets, mask=mask, other=0)
    tl.store(summary + offsets, old + values, mask=mask)
    if TRACE:
        tl.store(trace + offsets, values, mask=mask)
    tl.store(counts + offsets, 0, mask=mask)


def owned_token_span(
    num_rows: int,
    dp_rank: int,
    tp_rank: int,
    sp_size: int,
    naive_dispatch: bool,
    dp_sizes: list[int] | None,
    shard_sizes: list[int] | None,
) -> tuple[int, int, int]:
    """Return router row bounds and the corresponding local token offset.

    Replicated routing is reported on TP rank zero only. SP routing counts
    each TP shard locally, without an observability all-gather.
    """
    if naive_dispatch:
        if tp_rank != 0:
            return 0, 0, 0
        if dp_sizes is None:
            if sp_size != 1:
                raise ValueError("Expert-load SP dispatch requires DP metadata")
            return 0, num_rows, 0
        sizes = shard_sizes if shard_sizes is not None else dp_sizes
        if sum(sizes) != num_rows:
            raise ValueError("Expert-load routing rows do not match dispatch metadata")
        shards_per_dp = len(sizes) // len(dp_sizes)
        start = sum(sizes[: dp_rank * shards_per_dp])
        return start, start + dp_sizes[dp_rank], 0
    if sp_size > 1:
        return 0, num_rows, tp_rank * num_rows
    return (0, num_rows, 0) if tp_rank == 0 else (0, 0, 0)


@dataclass
class ExpertLoadLayer:
    counts: torch.Tensor
    num_valid_tokens: torch.Tensor
    dp_rank: int
    tp_rank: int
    sp_size: int
    naive_dispatch: bool

    def token_span(self, topk_ids: torch.Tensor) -> tuple[int, int, int]:
        from vllm.forward_context import get_forward_context

        metadata = get_forward_context().dp_metadata if self.naive_dispatch else None
        return owned_token_span(
            topk_ids.shape[0],
            self.dp_rank,
            self.tp_rank,
            self.sp_size,
            self.naive_dispatch,
            metadata.num_tokens_across_dp_list if metadata else None,
            metadata.local_sizes if metadata else None,
        )

    def record(self, topk_ids: torch.Tensor) -> None:
        if topk_ids.numel() == 0:
            return
        start, end, offset = self.token_span(topk_ids)
        _record_expert_load[(triton.cdiv(topk_ids.numel(), 256),)](
            topk_ids,
            self.counts,
            self.num_valid_tokens,
            topk_ids.shape[0],
            topk_ids.shape[1],
            topk_ids.stride(0),
            topk_ids.stride(1),
            start,
            end,
            offset,
            self.counts.numel(),
            triton.next_power_of_2(self.counts.numel()),
            256,
        )
