# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _indexer_metadata(
    QueryStart,
    SeqLens,
    Positions,
    Slots,
    Context,
    ValidPages,
    IndexSlots,
    num_reqs,
    num_tokens,
    num_slots,
    rank: tl.constexpr,
    tp: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    q0 = tl.load(QueryStart + row, row < num_reqs, other=0)
    q1 = tl.load(QueryStart + row + 1, row < num_reqs, other=0)
    seq = tl.load(SeqLens + row, row < num_reqs, other=0)
    tl.store(Context + row, seq - (q1 - q0), row < num_reqs)
    position = tl.load(Positions + row, row < num_tokens, other=-1)
    slot = tl.load(Slots + row, row < num_slots, other=-1)
    tl.store(
        ValidPages + row,
        tl.where((position >= 0) & (slot >= 0), position // 128 + 1, 0),
        row < num_tokens,
    )
    offset = slot % PAGE_SIZE
    if PAGE_SIZE == 256:
        owner = offset // (PAGE_SIZE // tp)
        local = offset % (PAGE_SIZE // tp)
    else:
        owner = offset % tp
        local = offset // tp
    sharded = slot // PAGE_SIZE * (PAGE_SIZE // tp) + local
    tl.store(
        IndexSlots + row,
        tl.where((slot >= 0) & (owner == rank), sharded, -1),
        row < num_slots,
    )


def prepare_indexer_metadata(
    query_start: torch.Tensor,
    seq_lens: torch.Tensor,
    positions: torch.Tensor,
    slots: torch.Tensor,
    context: torch.Tensor,
    valid_pages: torch.Tensor,
    index_slots: torch.Tensor,
    rank: int,
    tp: int,
    page_size: int,
) -> None:
    """Write stable index-K slots, causal page counts and prefix lengths on device."""
    assert page_size in (128, 256) and tp in (2, 4)
    assert slots.numel() <= index_slots.numel()
    rows = max(seq_lens.numel(), slots.numel(), valid_pages.numel())
    _indexer_metadata[(triton.cdiv(rows, 256),)](
        query_start,
        seq_lens,
        positions,
        slots,
        context,
        valid_pages,
        index_slots,
        seq_lens.numel(),
        valid_pages.numel(),
        slots.numel(),
        rank,
        tp,
        page_size,
        256,
    )


@triton.jit
def _prefill_window_metadata(
    QueryStart,
    SeqLens,
    Metadata,
    stride,
    num_reqs,
    start,
    end,
    BLOCK: tl.constexpr,
):
    request = tl.arange(0, BLOCK)
    mask = request < num_reqs
    base = tl.load(QueryStart)
    begin = tl.load(QueryStart + request, mask, other=0) - base
    finish = tl.load(QueryStart + request + 1, mask, other=0) - base
    local_begin = tl.minimum(tl.maximum(begin - start, 0), end - start)
    local_end = tl.minimum(tl.maximum(finish - start, 0), end - start)
    query_len = local_end - local_begin
    prefix = tl.load(SeqLens + request, mask, other=0) - (finish - begin)
    prefix += tl.minimum(tl.maximum(start - begin, 0), finish - begin)
    seq_len = prefix + query_len
    kv_end = tl.cumsum(tl.where(mask, seq_len, 0))
    tl.store(Metadata + request, local_begin, mask)
    tl.store(Metadata + stride + request, query_len, mask)
    tl.store(Metadata + 2 * stride + request, seq_len, mask)
    tl.store(Metadata + 3 * stride + request, prefix, mask)
    tl.store(Metadata + 4 * stride + request, kv_end - seq_len, mask)
    tl.store(Metadata + num_reqs, end - start)
    tl.store(Metadata + 4 * stride + num_reqs, tl.sum(tl.where(mask, seq_len, 0)))


def prepare_prefill_window(
    query_start: torch.Tensor,
    seq_lens: torch.Tensor,
    start: int,
    end: int,
    metadata: torch.Tensor,
) -> None:
    """Clip a token window against all requests without reading device lengths."""
    num_reqs = seq_lens.numel()
    assert query_start.numel() == num_reqs + 1
    assert metadata.shape[0] == 5 and metadata.shape[1] >= num_reqs + 1
    _prefill_window_metadata[(1,)](
        query_start,
        seq_lens,
        metadata,
        metadata.stride(0),
        num_reqs,
        start,
        end,
        triton.next_power_of_2(num_reqs),
    )
