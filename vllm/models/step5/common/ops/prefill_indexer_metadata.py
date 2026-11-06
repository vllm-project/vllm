# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Step-local indexer metadata shared across sparse attention layers."""

from typing import NamedTuple

import torch

from vllm.triton_utils import tl, triton

from .sparse_indexer import (
    PagedPrefillMetadata,
    _emit_all_regions,
    _emit_indexer_schedule,
    _emit_prefill_pages,
)


class PrefillIndexerMetadata(NamedTuple):
    current: torch.Tensor
    lengths: torch.Tensor
    topk_lengths: torch.Tensor
    block_table: torch.Tensor
    schedule: torch.Tensor | None
    selected: torch.Tensor | None


# Row, page and region counts follow the prompt; keep them runtime values so
# each new prompt length does not recompile.
@triton.jit(do_not_specialize=["ROWS", "PAGES", "REGIONS"])
def _prepare_prefill_indexer(
    Owners,
    Positions,
    Current,
    Lengths,
    TopKLengths,
    Table,
    Schedules,
    Selected,
    ROWS,
    PAGES,
    OWNER_STRIDE: tl.constexpr,
    POSITION_STRIDE: tl.constexpr,
    NUM_SMS: tl.constexpr,
    BLOCK_SMS: tl.constexpr,
    NEED_SCORES: tl.constexpr,
    REGIONS,
    BLOCK_REGIONS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
    PREPARE_PAGES: tl.constexpr = False,
    PageOwners=None,
    PageTable=None,
    PageIndices=None,
    PageIndptr=None,
    PAGE_OWNER_STRIDE: tl.constexpr = 0,
    PAGE_TABLE_ROW: tl.constexpr = 0,
    PAGE_TABLE_COL: tl.constexpr = 0,
    PAGE_INDPTR_STRIDE: tl.constexpr = 1,
    BLOCK_SIZE: tl.constexpr = 8,
    PAGE_STRIDE: tl.constexpr = 1,
):
    page_tile = tl.program_id(0)
    row = tl.program_id(1)
    if page_tile == 0:
        position = tl.load(Positions + row * POSITION_STRIDE).to(tl.int64)
        current = position >> 3
        tl.store(Current + row, current)
        tl.store(Lengths + row, current + 1)
        tl.store(TopKLengths + row, current)
        if not NEED_SCORES:
            _emit_all_regions(
                current,
                Selected + row.to(tl.int64) * (REGIONS + 1),
                REGIONS,
                BLOCK_REGIONS,
            )
        if PREPARE_PAGES:
            col = tl.arange(0, BLOCK_REGIONS)
            region = col - (REGIONS - current)
            selected = tl.where(region >= 0, region, -1)
            _emit_prefill_pages(
                selected,
                col,
                row,
                PageOwners,
                PageTable,
                PageIndices,
                PageIndptr,
                REGIONS,
                PAGE_OWNER_STRIDE,
                PAGE_TABLE_ROW,
                PAGE_TABLE_COL,
                PAGE_INDPTR_STRIDE,
                BLOCK_SIZE,
                PAGE_STRIDE,
            )
        if NEED_SCORES and row % CHUNK_SIZE == 0:
            query = row + tl.arange(0, CHUNK_SIZE)
            pos = tl.load(
                Positions + query * POSITION_STRIDE, query < ROWS, other=0
            ).to(tl.int64)
            lengths = ((pos >> 3) + 1).to(tl.int32)
            _emit_indexer_schedule(
                lengths,
                Schedules + (row // CHUNK_SIZE) * (NUM_SMS + 1) * 2,
                tl.minimum(ROWS - row, CHUNK_SIZE),
                CHUNK_SIZE,
                NUM_SMS,
                BLOCK_SMS,
                SEARCH_STEPS,
            )
    if NEED_SCORES:
        col = page_tile * 256 + tl.arange(0, 256)
        owner = tl.load(Owners + row * OWNER_STRIDE)
        tl.store(
            Table + row.to(tl.int64) * PAGES + col,
            owner * PAGES + col,
            col < PAGES,
        )


def prefill_indexer_chunk_size(regions: int) -> int:
    """Bound FP32 logits to 128 MiB for the model's 1M-token context.

    Each chunk costs a few host launches per DSA layer, so chunks are as large
    as the bound allows.
    """
    padded_regions = triton.cdiv(max(1, regions), 256) * 256
    rows = max(1, (128 << 20) // (4 * padded_regions))
    return max(256, min(4096, 1 << (rows.bit_length() - 1)))


def prepare_prefill_indexer(
    owners: torch.Tensor,
    positions: torch.Tensor,
    regions: int,
    topk: int,
    num_sms: int,
    *,
    paged_prefill: PagedPrefillMetadata | None = None,
) -> list[PrefillIndexerMetadata]:
    """Prepare bounded FI query chunks once; reuse them for every DSA layer."""
    rows = positions.numel()
    chunk_size = prefill_indexer_chunk_size(regions)
    need_scores = regions > topk
    paged_kwargs = {}
    if paged_prefill is not None:
        if need_scores or paged_prefill.block_size not in (16, 32, 64, 128):
            raise ValueError("Shared pages require unscored prefill")
        paged_kwargs = dict(
            PREPARE_PAGES=True,
            PageOwners=paged_prefill.owners,
            PageTable=paged_prefill.block_table,
            PageIndices=paged_prefill.indices,
            PageIndptr=paged_prefill.indptr,
            PAGE_OWNER_STRIDE=paged_prefill.owners.stride(0),
            PAGE_TABLE_ROW=paged_prefill.block_table.stride(0),
            PAGE_TABLE_COL=paged_prefill.block_table.stride(1),
            PAGE_INDPTR_STRIDE=paged_prefill.indptr.stride(0),
            BLOCK_SIZE=paged_prefill.block_size,
            PAGE_STRIDE=paged_prefill.page_stride,
        )
    pages = triton.cdiv(regions, 128) * 4 if need_scores else 0
    current = torch.empty(rows, device=positions.device, dtype=torch.int64)
    lengths = torch.empty(rows, device=positions.device, dtype=torch.int32)
    topk_lengths = torch.empty_like(lengths)
    table = torch.empty((rows, pages), device=positions.device, dtype=torch.int32)
    schedules = (
        torch.empty(
            (triton.cdiv(rows, chunk_size), num_sms + 1, 2),
            device=positions.device,
            dtype=torch.int32,
        )
        if need_scores
        else None
    )
    selected = (
        torch.empty((rows, regions + 1), device=positions.device, dtype=torch.int64)
        if not need_scores
        else None
    )
    if rows:
        _prepare_prefill_indexer[(max(1, triton.cdiv(pages, 256)), rows)](
            owners,
            positions,
            current,
            lengths,
            topk_lengths,
            table,
            schedules,
            selected,
            rows,
            pages,
            owners.stride(0),
            positions.stride(0),
            num_sms,
            triton.next_power_of_2(num_sms + 1),
            need_scores,
            regions if not need_scores else 0,
            # Sized by topk, not the prompt: every short prompt reuses one variant.
            triton.next_power_of_2(topk + 1) if not need_scores else 1,
            chunk_size,
            chunk_size.bit_length(),
            num_warps=4,
            **paged_kwargs,
        )
    return [
        PrefillIndexerMetadata(
            current[start : start + chunk_size],
            lengths[start : start + chunk_size],
            topk_lengths[start : start + chunk_size],
            table[start : start + chunk_size],
            schedules[start // chunk_size] if schedules is not None else None,
            selected[start : start + chunk_size] if selected is not None else None,
        )
        for start in range(0, rows, chunk_size)
    ]
