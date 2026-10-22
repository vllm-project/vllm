# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packing and stable region selection for the sparse indexer."""

from typing import NamedTuple

import torch

from vllm.model_executor.layers.indexer_topk import (
    get_indexer_topk,
    is_deep_select_supported,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton


class DecodeIndexerMetadata(NamedTuple):
    active: torch.Tensor
    positions: torch.Tensor
    current: torch.Tensor
    lengths: torch.Tensor
    block_table: torch.Tensor
    topk_lengths: torch.Tensor
    schedule: torch.Tensor | None = None


class PagedPrefillMetadata(NamedTuple):
    owners: torch.Tensor
    block_table: torch.Tensor
    indices: torch.Tensor
    indptr: torch.Tensor
    block_size: int
    # Eight-token pages between consecutive cache blocks.
    page_stride: int


@triton.jit
def _emit_prefill_pages(
    selected,
    col,
    row,
    Owners,
    Table,
    Pages,
    Indptr,
    K,
    OWNER_STRIDE: tl.constexpr,
    TABLE_ROW: tl.constexpr,
    TABLE_COL: tl.constexpr,
    INDPTR_STRIDE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    PAGE_STRIDE: tl.constexpr,
):
    start = tl.load(Indptr + row * INDPTR_STRIDE)
    end = tl.load(Indptr + (row + 1) * INDPTR_STRIDE)
    compact_col = col - (K + 1 - (end - start))
    valid = (col <= K) & (compact_col >= 0)
    owner = tl.load(Owners + row * OWNER_STRIDE)
    physical = tl.load(
        Table + owner * TABLE_ROW + selected // (BLOCK_SIZE // 8) * TABLE_COL,
        valid,
        other=0,
    )
    page = physical * PAGE_STRIDE + selected % (BLOCK_SIZE // 8)
    tl.store(Pages + start + compact_col, page, valid)


@triton.jit
def _emit_indexer_schedule(
    context,
    Schedule,
    batch,
    BLOCK_BATCH: tl.constexpr,
    NUM_SMS: tl.constexpr,
    BLOCK_SMS: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
):
    row = tl.arange(0, BLOCK_BATCH)
    segments = tl.where(row < batch, (context + 255) // 256, 0)
    prefix = tl.cumsum(segments)
    total = tl.sum(segments, 0)
    sm = tl.arange(0, BLOCK_SMS)
    target = sm * (total // NUM_SMS) + tl.minimum(sm, total % NUM_SMS)
    lo = tl.full((BLOCK_SMS,), 0, tl.int32)
    count = tl.full((BLOCK_SMS,), 0, tl.int32) + batch
    for _ in range(SEARCH_STEPS):
        half = count // 2
        mid = lo + half
        boundary = tl.gather(prefix, tl.minimum(mid, BLOCK_BATCH - 1), 0)
        take = (count > 0) & (boundary <= target)
        lo = tl.where(take, mid + 1, lo)
        count = tl.where(take, count - half - 1, half)
    previous = tl.gather(prefix, tl.maximum(lo - 1, 0), 0)
    local = target - tl.where(lo > 0, previous, 0)
    tl.store(Schedule + sm * 2, lo, sm <= NUM_SMS)
    tl.store(Schedule + sm * 2 + 1, local, sm <= NUM_SMS)


@triton.jit(do_not_specialize=["BATCH"])
def _emit_decode_schedule(
    Slots,
    SeqLens,
    Schedule,
    SLOT_STRIDE: tl.constexpr,
    SEQ_STRIDE: tl.constexpr,
    BATCH,
    BLOCK_BATCH: tl.constexpr,
    NUM_SMS: tl.constexpr,
    BLOCK_SMS: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
):
    # FlashInfer's public schedule_meta: ceil(context_lens / 256) partitions.
    row = tl.arange(0, BLOCK_BATCH)
    active = tl.load(Slots + row * SLOT_STRIDE, row < BATCH, other=-1) >= 0
    length = tl.load(SeqLens + row * SEQ_STRIDE, row < BATCH, other=0).to(tl.int64)
    position = tl.where(active, length - 1, 0)
    context = ((position >> 3) + 1).to(tl.int32)
    _emit_indexer_schedule(
        context, Schedule, BATCH, BLOCK_BATCH, NUM_SMS, BLOCK_SMS, SEARCH_STEPS
    )


@triton.jit
def _copy_summary_page(
    Mean,
    Table,
    Packed,
    Scales,
    request,
    page,
    REGIONS,
    PAGES,
    CACHE_PAGES: tl.constexpr,
    REGIONS_PER_BLOCK: tl.constexpr,
    MEAN_BLOCK: tl.constexpr,
    TABLE_COLS: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
):
    region = page * 32 + tl.arange(0, 32)
    col = region // REGIONS_PER_BLOCK
    physical = tl.load(
        Table + request * TABLE_STRIDE + col,
        col < TABLE_COLS,
        other=-1,
    )
    valid = (region < REGIONS) & (physical >= 0) & (physical < CACHE_PAGES)
    dim = tl.arange(0, 256)
    values = tl.load(
        Mean
        + physical[:, None].to(tl.int64) * MEAN_BLOCK
        + region[:, None] % REGIONS_PER_BLOCK * 256
        + dim[None, :],
        valid[:, None],
        other=0,
    )
    # Large graph batches can pack more than 2 GiB of summary bytes.
    output_page = request.to(tl.int64) * PAGES + page
    tl.store(
        Packed
        + output_page * (32 * 260)
        + tl.arange(0, 32)[:, None] * 256
        + dim[None, :],
        values,
    )
    tl.store(Scales + output_page * (32 * 260 // 4) + tl.arange(0, 32), 1.0)


@triton.jit
def _copy_summary_tile(
    Mean,
    Table,
    Packed,
    Scales,
    request,
    page,
    REGIONS,
    PAGES,
    CACHE_PAGES: tl.constexpr,
    REGIONS_PER_BLOCK: tl.constexpr,
    MEAN_BLOCK: tl.constexpr,
    TABLE_COLS: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    TILE_PAGES: tl.constexpr,
):
    offset = tl.arange(0, TILE_PAGES * 32)
    region = page * 32 + offset
    packed_page = page + offset // 32
    col = region // REGIONS_PER_BLOCK
    physical = tl.load(Table + request * TABLE_STRIDE + col, col < TABLE_COLS, other=-1)
    valid = (region < REGIONS) & (physical >= 0) & (physical < CACHE_PAGES)
    dim = tl.arange(0, 64)
    values = tl.load(
        Mean.to(tl.pointer_type(tl.uint32))
        + physical[:, None].to(tl.int64) * (MEAN_BLOCK // 4)
        + region[:, None] % REGIONS_PER_BLOCK * 64
        + dim[None, :],
        valid[:, None],
        other=0,
    )
    output_page = request.to(tl.int64) * PAGES + packed_page
    tl.store(
        Packed.to(tl.pointer_type(tl.uint32))
        + output_page[:, None] * (32 * 65)
        + (offset[:, None] % 32) * 64
        + dim[None, :],
        values,
        (packed_page < PAGES)[:, None],
    )
    tl.store(Scales + output_page * (32 * 65) + offset % 32, 1.0, packed_page < PAGES)


# Context-dependent sizes stay runtime values so new prompt lengths reuse code.
@triton.jit(do_not_specialize=["REGIONS", "PAGES", "BATCH"])
def _pack_summary_pages(
    Mean,
    Table,
    Packed,
    Scales,
    Slots,
    SeqLens,
    Active,
    Positions,
    Current,
    Lengths,
    IndexTable,
    TopKLengths,
    Schedule,
    REGIONS,
    PAGES,
    CACHE_PAGES: tl.constexpr,
    REGIONS_PER_BLOCK: tl.constexpr,
    MEAN_BLOCK: tl.constexpr,
    TABLE_COLS: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    DECODE: tl.constexpr,
    SLOT_STRIDE: tl.constexpr,
    SEQ_STRIDE: tl.constexpr,
    BATCH,
    BLOCK_BATCH: tl.constexpr,
    NUM_SMS: tl.constexpr,
    BLOCK_SMS: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
    LIVE_ONLY: tl.constexpr = False,
    BLOCK_PAGES: tl.constexpr = 1,
    TILE_PAGES: tl.constexpr = 1,
):
    page = tl.program_id(0)
    request = tl.program_id(1)
    if DECODE and NUM_SMS > 0 and page == 0 and request == 0:
        _emit_decode_schedule(
            Slots,
            SeqLens,
            Schedule,
            SLOT_STRIDE,
            SEQ_STRIDE,
            BATCH,
            BLOCK_BATCH,
            NUM_SMS,
            BLOCK_SMS,
            SEARCH_STEPS,
        )
    if LIVE_ONLY:
        is_active = tl.load(Slots + request * SLOT_STRIDE) >= 0
        seq_len = tl.load(SeqLens + request * SEQ_STRIDE).to(tl.int64)
        pos = tl.where(is_active, seq_len - 1, 0)
        context = (pos >> 3) + 1
        live_pages = tl.minimum(PAGES, tl.maximum((context + 127) // 128, 0) * 4)
    if DECODE:
        if page == 0:
            active = tl.load(Slots + request * SLOT_STRIDE) >= 0
            length = tl.load(SeqLens + request * SEQ_STRIDE).to(tl.int64)
            position = tl.where(active, length - 1, 0)
            # Arithmetic shift preserves signed floor division, including -1.
            current = position >> 3
            tl.store(Active + request, active)
            tl.store(Positions + request, position)
            tl.store(Current + request, current)
            tl.store(Lengths + request, current + 1)
            tl.store(TopKLengths + request, current)
        if LIVE_ONLY:
            if page == 0:
                cols = tl.arange(0, BLOCK_PAGES)
                indices = request.to(tl.int64) * PAGES + cols
                tl.store(IndexTable + indices, indices, cols < PAGES)
        elif page < PAGES:
            index = request.to(tl.int64) * PAGES + page
            tl.store(IndexTable + index, index)
    if LIVE_ONLY:
        if page * TILE_PAGES < live_pages:
            packed_page = page * TILE_PAGES
            _copy_summary_tile(
                Mean,
                Table,
                Packed,
                Scales,
                request,
                packed_page,
                REGIONS,
                PAGES,
                CACHE_PAGES,
                REGIONS_PER_BLOCK,
                MEAN_BLOCK,
                TABLE_COLS,
                TABLE_STRIDE,
                TILE_PAGES,
            )
    elif page < PAGES:
        _copy_summary_page(
            Mean,
            Table,
            Packed,
            Scales,
            request,
            page,
            REGIONS,
            PAGES,
            CACHE_PAGES,
            REGIONS_PER_BLOCK,
            MEAN_BLOCK,
            TABLE_COLS,
            TABLE_STRIDE,
        )


def _pack_summaries(
    mean, block_table, regions, slots=None, seq_lens=None, num_sms=0, *, live_only=False
):
    # Blocks may be strided (block-outer KV layout); regions within one are not.
    assert mean.stride(1) == 256 and mean.stride(2) == 1
    pages = triton.cdiv(regions, 128) * 4
    batch = block_table.shape[0]
    packed = torch.empty(
        (batch * pages, 32 * 260), device=mean.device, dtype=torch.uint8
    )
    metadata = None
    if slots is not None:
        assert seq_lens is not None
        metadata = DecodeIndexerMetadata(
            torch.empty(batch, device=mean.device, dtype=torch.bool),
            torch.empty(batch, device=mean.device, dtype=torch.int64),
            torch.empty(batch, device=mean.device, dtype=torch.int64),
            torch.empty(batch, device=mean.device, dtype=torch.int32),
            torch.empty((batch, pages), device=mean.device, dtype=torch.int32),
            torch.empty(batch, device=mean.device, dtype=torch.int32),
            torch.empty((num_sms + 1, 2), device=mean.device, dtype=torch.int32)
            if num_sms
            else None,
        )
    if live_only and slots is None:
        raise ValueError("Live packing needs decode slots")
    tile_pages = 4 if batch >= 16 and pages >= 256 else 2
    workers = max(triton.cdiv(pages, tile_pages), 1) if live_only else max(pages, 1)
    # Constexprs only the decode schedule and live packing read. Prefill
    # passes constants: specializing on the context length or batch would
    # JIT a new variant mid-serving (~0.4 s) each time one crosses a power of 2.
    decode = metadata is not None
    block_batch = triton.next_power_of_2(max(batch, 1)) if decode else 1
    search_steps = max(batch, 1).bit_length() if decode else 1
    block_pages = triton.next_power_of_2(max(pages, 1)) if live_only else 1
    if batch and (pages or metadata is not None):
        _pack_summary_pages[(workers, batch)](
            mean,
            block_table,
            packed,
            packed[:, 32 * 256 :].view(torch.float32),
            slots,
            seq_lens,
            *(metadata if metadata is not None else (None,) * 7),
            regions,
            pages,
            mean.shape[0],
            mean.shape[1],
            mean.stride(0),
            block_table.shape[1],
            block_table.stride(0),
            metadata is not None,
            slots.stride(0) if slots is not None else 0,
            seq_lens.stride(0) if seq_lens is not None else 0,
            batch,
            block_batch,
            num_sms,
            triton.next_power_of_2(num_sms + 1),
            search_steps,
            LIVE_ONLY=live_only,
            BLOCK_PAGES=block_pages,
            TILE_PAGES=tile_pages if live_only else 1,
        )
    return packed.view(-1, 32, 1, 260), pages, metadata


def pack_summary_pages(
    mean: torch.Tensor, block_table: torch.Tensor, regions: int
) -> tuple[torch.Tensor, int]:
    packed, pages, _ = _pack_summaries(mean, block_table, regions)
    return packed, pages


def pack_decode_summary_pages(
    mean: torch.Tensor,
    block_table: torch.Tensor,
    regions: int,
    slots: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    num_sms: int = 0,
    live_only: bool = False,
) -> tuple[torch.Tensor, DecodeIndexerMetadata]:
    """Pack summaries and emit decode metadata in the same launch."""
    packed, _, metadata = _pack_summaries(
        mean, block_table, regions, slots, seq_lens, num_sms, live_only=live_only
    )
    assert metadata is not None
    return packed, metadata


@triton.jit
def _emit_all_regions(current, Output, K, BLOCK: tl.constexpr):
    col = tl.arange(0, BLOCK)
    # Left-pad with -1, then include every historical region and the current one.
    region = col - (K - current)
    tl.store(Output + col, tl.where(region >= 0, region, -1), col <= K)


@triton.jit
def _all_regions(Current, Output, K: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    current = tl.load(Current + row)
    _emit_all_regions(current, Output + row * (K + 1), K, BLOCK)


def select_all_regions(current: torch.Tensor, capacity: int) -> torch.Tensor:
    """Select all history and current without computing unused indexer scores."""
    rows = current.shape[0]
    output = torch.empty((rows, capacity + 1), device=current.device, dtype=torch.long)
    if rows:
        _all_regions[(rows,)](
            current, output, capacity, triton.next_power_of_2(capacity + 1)
        )
    return output


@triton.jit(do_not_specialize=["SCORE_STRIDE"])
def _stable_topk_regions(
    Scores,
    Current,
    Indices,
    Scratch,
    Output,
    SCORE_STRIDE,
    INDEX_STRIDE: tl.constexpr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_SCAN: tl.constexpr,
    BLOCK_SORT: tl.constexpr,
    Table=None,
    Pages=None,
    Indptr=None,
    TABLE_ROW: tl.constexpr = 0,
    TABLE_COL: tl.constexpr = 0,
    BLOCK_SIZE: tl.constexpr = 8,
    PAGE_STRIDE: tl.constexpr = 1,
    PREPARE_PREFILL: tl.constexpr = False,
    Owners=None,
    OWNER_STRIDE: tl.constexpr = 0,
    INDPTR_STRIDE: tl.constexpr = 1,
    SORT: tl.constexpr = True,
):
    row = tl.program_id(0)
    length = tl.load(Current + row).to(tl.int32)
    col = tl.arange(0, BLOCK_K)
    chosen = tl.load(Indices + row * INDEX_STRIDE + col, col < K, other=-1)
    valid = (col < K) & (chosen >= 0) & (chosen < length)
    values = tl.load(Scores + row * SCORE_STRIDE + chosen, valid, other=float("inf"))
    # When Top-K contains every historical region, its ordering is irrelevant.
    # Non-finite scores or an incomplete selection take the general path.
    complete = tl.sum(valid.to(tl.int32), 0) == length
    short_finite = False
    if (length <= K) & complete:
        history = tl.load(Scores + row * SCORE_STRIDE + col, col < length, other=0)
        nonfinite = (history != history) | (tl.abs(history) == float("inf"))
        short_finite = tl.sum(nonfinite.to(tl.int32), 0) == 0
    if short_finite:
        output_col = tl.arange(0, BLOCK_SORT)
        region = output_col - (K - length)
        selected = tl.where(region >= 0, region, -1)
    else:
        threshold = tl.min(values, 0)
        above = valid & (values > threshold)
        above_rank = tl.cumsum(above.to(tl.int32))
        num_above = tl.sum(above.to(tl.int32), 0)
        # Unsorted output keeps the decode contract (-1 left padding, current
        # last) in DeepSelect's deterministic order, skipping the 1024 sort.
        if SORT:
            output = Scratch + row * (K + 1)
            pad = 0
        else:
            output = Output + row * (K + 1)
            pad = K - tl.minimum(length, K)
        # Also initialize capture/dummy rows whose scores can be NaN.
        tl.store(output + col, -1, col < K)
        tl.store(output + K, length)
        tl.debug_barrier()
        tl.store(output + pad + above_rank - 1, chosen, above)
        # Public top-k backends need not prefer smaller IDs at the cutoff.
        # Keep every score above the cutoff and fill ties in increasing region order.
        needed = tl.minimum(length, K) - num_above
        found = 0
        start = 0
        offsets = tl.arange(0, BLOCK_SCAN)
        while (found < needed) & (start < length):
            region = start + offsets
            scores = tl.load(
                Scores + row * SCORE_STRIDE + region,
                region < length,
                other=float("inf"),
            )
            equal = (region < length) & (scores == threshold)
            rank = tl.cumsum(equal.to(tl.int32))
            tl.store(
                output + pad + num_above + found + rank - 1,
                region,
                equal & (found + rank <= needed),
            )
            found += tl.sum(equal.to(tl.int32), 0)
            start += BLOCK_SCAN

        if SORT:
            tl.debug_barrier()
            sorted_col = tl.arange(0, BLOCK_SORT)
            selected = tl.load(output + sorted_col, sorted_col <= K, other=2147483647)
            selected = tl.sort(selected, descending=False)
        elif PREPARE_PREFILL:
            # Page metadata reads the row back; the unsorted layout (-1 left
            # padding, current last) already matches its compact-page contract.
            tl.debug_barrier()
            sorted_col = tl.arange(0, BLOCK_SORT)
            selected = tl.load(output + sorted_col, sorted_col <= K, other=-1)
            selected = selected.to(tl.int32)
        else:
            selected = tl.full([BLOCK_SORT], -1, tl.int32)

    col = tl.arange(0, BLOCK_SORT)
    if SORT or short_finite:
        tl.store(Output + row * (K + 1) + col, selected, col <= K)
    if PREPARE_PREFILL:
        _emit_prefill_pages(
            selected,
            col,
            row,
            Owners,
            Table,
            Pages,
            Indptr,
            K,
            OWNER_STRIDE,
            TABLE_ROW,
            TABLE_COL,
            INDPTR_STRIDE,
            BLOCK_SIZE,
            PAGE_STRIDE,
        )


def select_regions(
    scores: torch.Tensor,
    current: torch.Tensor,
    topk: int,
    backend: str = "auto",
    *,
    topk_lengths: torch.Tensor | None = None,
    paged_prefill: PagedPrefillMetadata | None = None,
    sort: bool = True,
) -> torch.Tensor:
    """Select stable top-k historical regions and append current.

    With ``sort=False`` selected regions keep top-k order, left-padded with -1.
    """
    rows, regions = scores.shape
    k = min(topk, regions)
    if paged_prefill is not None:
        if regions <= topk:
            raise ValueError("Fused prefill pages require scored prefill only")
        if paged_prefill.block_size not in (16, 32, 64, 128):
            raise ValueError("Unsupported prefill cache block size")
    if regions <= topk:
        return select_all_regions(current, k)
    output = torch.empty((rows, k + 1), device=scores.device, dtype=torch.long)
    if not rows:
        return output
    # Step5 requires exact FP32 ordering. The cooperative/persistent backends
    # can select the wrong cutoff for scores separated by one FP32 epsilon.
    if backend == "auto":
        backend = (
            "deep_select"
            if current_platform.is_device_capability_family(100)
            and is_deep_select_supported(scores, k)
            else "per_row"
        )
    if backend in ("cooperative", "persistent"):
        raise ValueError(
            "Step5 requires exact FP32 top-k; use auto, deep_select, or per_row."
        )
    # Only DeepSelect guarantees a deterministic unsorted selection order.
    sort = sort or backend != "deep_select"
    # DeepSelect needs 32-byte row alignment; per_row writes contiguous rows.
    stride = triton.cdiv(k, 8) * 8 if backend == "deep_select" else k
    indices = torch.empty((rows, stride), device=scores.device, dtype=torch.int32)[
        :, :k
    ]
    get_indexer_topk(backend)(
        scores,
        current.to(torch.int32).contiguous() if topk_lengths is None else topk_lengths,
        1,
        indices,
        k,
        regions,
    )
    scratch = torch.empty((rows, k + 1), device=scores.device, dtype=torch.int32)
    paged_kwargs = {}
    if paged_prefill is not None:
        paged_kwargs = dict(
            PREPARE_PREFILL=True,
            Owners=paged_prefill.owners,
            Table=paged_prefill.block_table,
            Pages=paged_prefill.indices,
            Indptr=paged_prefill.indptr,
            OWNER_STRIDE=paged_prefill.owners.stride(0),
            TABLE_ROW=paged_prefill.block_table.stride(0),
            TABLE_COL=paged_prefill.block_table.stride(1),
            INDPTR_STRIDE=paged_prefill.indptr.stride(0),
            BLOCK_SIZE=paged_prefill.block_size,
            PAGE_STRIDE=paged_prefill.page_stride,
        )
    _stable_topk_regions[(rows,)](
        scores,
        current,
        indices,
        scratch,
        output,
        scores.stride(0),
        indices.stride(0),
        k,
        triton.next_power_of_2(k),
        1024,
        triton.next_power_of_2(k + 1),
        **paged_kwargs,
        SORT=sort,
    )
    return output
