# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compact eight-token pages for sparse prefill attention."""

from typing import NamedTuple

import numpy as np
import torch

from vllm.triton_utils import tl, triton

from ...common.ops.sparse_indexer import PagedPrefillMetadata


@triton.jit(do_not_specialize=["REGION_COUNT", "REGION_ROW"])
def _compact_prefill_pages(
    Regions,
    Table,
    Owners,
    Indptr,
    Pages,
    Query,
    ContiguousQuery,
    QUERY_ROW: tl.constexpr,
    QUERY_HEAD: tl.constexpr,
    QUERY_DIM: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    QUERY_WIDTH: tl.constexpr,
    REGION_COUNT,
    REGION_ROW,
    REGION_COL: tl.constexpr,
    TABLE_ROW: tl.constexpr,
    TABLE_COL: tl.constexpr,
    OWNER_STRIDE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    PAGE_STRIDE: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(1)
    col = tl.program_id(0) * TILE + tl.arange(0, TILE)
    start = tl.load(Indptr + row)
    end = tl.load(Indptr + row + 1)
    count = end - start
    # The prefill reference removes all left padding before attention.
    region = tl.load(
        Regions + row * REGION_ROW + (REGION_COUNT - count + col) * REGION_COL,
        col < count,
        other=0,
    )
    owner = tl.load(Owners + row * OWNER_STRIDE)
    physical = tl.load(
        Table + owner * TABLE_ROW + region // (BLOCK_SIZE // 8) * TABLE_COL,
        col < count,
        other=0,
    )
    page = physical * PAGE_STRIDE + region % (BLOCK_SIZE // 8)
    tl.store(Pages + start + col, page, col < count)
    q_col = tl.program_id(0) * 256 + tl.arange(0, 256)
    value = tl.load(
        Query
        + row * QUERY_ROW
        + (q_col // HEAD_DIM) * QUERY_HEAD
        + (q_col % HEAD_DIM) * QUERY_DIM,
        q_col < QUERY_WIDTH,
        other=0,
    )
    tl.store(
        ContiguousQuery + row * QUERY_WIDTH + q_col,
        value,
        q_col < QUERY_WIDTH,
    )


def fill_prefill_pages(
    regions,
    table,
    owners,
    indptr,
    pages,
    block_size,
    query,
    contiguous_query,
    page_stride,
):
    """Emit compact pages and copy ``query`` into ``contiguous_query``."""
    rows, count = regions.shape
    width = query.shape[1] * query.shape[2]
    tiles = max(triton.cdiv(count, 128), triton.cdiv(width, 256))
    _compact_prefill_pages[(tiles, rows)](
        regions,
        table,
        owners,
        indptr,
        pages,
        query.view(torch.int16),
        contiguous_query.view(torch.int16),
        *query.stride(),
        query.shape[2],
        width,
        count,
        *regions.stride(),
        *table.stride(),
        owners.stride(0),
        block_size,
        page_stride,
        128,
        num_warps=4,
    )


def kv_page_stride(cache: torch.Tensor) -> int:
    """Eight-token pages between consecutive blocks of a ``[B, H, N, C]`` cache.

    Blocks may be strided (sliced KV groups, block-outer layouts) as long as
    the block stride is a whole number of eight-token pages.
    """
    page = 8 * cache.stride(2)
    if cache.stride(0) % page:
        raise ValueError(
            f"KV block stride {cache.stride(0)} is not a multiple of the "
            f"eight-token page ({page})"
        )
    return cache.stride(0) // page


def view_sparse_kv_pages(cache):
    """View eight-token pages, retaining gaps between blocks and KV groups."""
    dim = cache.shape[-1] // 2
    kv_page_stride(cache)
    page_stride = 8 * cache.stride(2)
    size = (
        (cache.shape[0] - 1) * cache.stride(0) + cache.shape[2] * cache.stride(2)
    ) // page_stride
    pages = cache.as_strided(
        (size, 8, 1, 2 * dim),
        (page_stride, cache.stride(2), 2 * dim, 1),
    )
    return pages.split(dim, -1)


def compact_page_metadata(positions, region_count):
    positions = np.asarray(positions, dtype=np.int64)
    if region_count < 1 or (positions < 0).any():
        raise ValueError("Prefill requires nonnegative positions and a current region")
    indptr = np.zeros(positions.size + 1, dtype=np.int32)
    np.cumsum(np.minimum(positions // 8 + 1, region_count), out=indptr[1:])
    return (
        torch.from_numpy(indptr),
        torch.from_numpy((positions % 8 + 1).astype(np.int32)),
    )


class CompactPagedPrefill:
    """One native attention plan for a full prefill step, shared by DSA layers."""

    def __init__(self, workspace):
        from flashinfer import BatchPrefillWithPagedKVCacheWrapper

        self.wrapper = BatchPrefillWithPagedKVCacheWrapper(
            workspace, "NHD", backend="fa2"
        )
        self._plan_event = torch.cuda.Event()

    def plan(
        self,
        query_shape,
        dtype,
        device,
        positions,
        region_count,
        scale,
        block_size,
        *,
        contiguous_query=True,
    ):
        """Plan from host positions, which may be per-row upper bounds.

        Runs before the forward (in the metadata builder) and issues no
        blocking copies; see ``fixup_positions`` for exact device positions.
        """
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Step5 prefill planning requires a breakable graph")
        rows, heads, dim = query_shape
        assert rows == len(positions) and rows > 0
        indptr, last_page_len = compact_page_metadata(positions, region_count)
        indptr, last_page_len = indptr.pin_memory(), last_page_len.pin_memory()
        self.pages = torch.empty(int(indptr[-1]), device=device, dtype=torch.int32)
        self.indptr = indptr.to(device, non_blocking=True)
        self.region_count = region_count
        self.block_size = block_size
        # The native planner reuses pinned memory for asynchronous HtoD copies.
        # Plans alternate between two instances, so this only waits if the GPU
        # is two steps behind.
        self._plan_event.synchronize()
        self.wrapper.plan(
            torch.arange(rows + 1, dtype=torch.int32).pin_memory(),
            indptr,
            self.pages,
            last_page_len,
            heads,
            1,
            dim,
            8,
            head_dim_vo=dim,
            causal=False,
            sm_scale=scale,
            q_data_type=dtype,
            kv_data_type=dtype,
            disable_split_kv=True,
            # Skips FlashInfer's Python builtin max over the query lengths
            # (about 20 ms at 16K rows). Its KV-length counterpart cannot be
            # passed: FlashInfer 0.7.0 then leaves paged_kv_indptr_host unset.
            max_token_per_sequence=1,
        )
        self._plan_event.record(torch.cuda.current_stream(device))

        self.query_shape = torch.Size(query_shape)
        self.query_buffer = (
            None
            if contiguous_query
            else torch.empty(query_shape, dtype=dtype, device=device)
        )

    def fixup_positions(self, positions: torch.Tensor) -> None:
        """Replace upper-bound page ranges with exact device positions.

        With split KV disabled each FA2 CTA reads its row's range from these
        device buffers at run time, so shorter exact ranges are safe.
        """
        counts = torch.clamp(positions // 8 + 1, max=self.region_count)
        self.indptr[1:].copy_(torch.cumsum(counts, 0))
        self.indptr[:1].zero_()
        self.wrapper._paged_kv_indptr_buf.copy_(self.indptr)
        self.wrapper._paged_kv_last_page_len_buf.copy_(positions % 8 + 1)

    def page_metadata(self, table, owners, page_stride, start=0, end=None):
        if self.query_buffer is not None:
            return None
        end = self.query_shape[0] if end is None else end
        return PagedPrefillMetadata(
            owners[start:end],
            table,
            self.pages,
            self.indptr[start : end + 1],
            self.block_size,
            page_stride,
        )

    def prepare_chunk(self, start, query, cache, regions, table, owners):
        assert self.query_buffer is not None
        end = start + query.shape[0]
        assert 0 <= start < end <= self.query_shape[0]
        assert query.shape[1:] == self.query_shape[1:]
        assert regions.shape == (query.shape[0], self.region_count)
        fill_prefill_pages(
            regions,
            table,
            owners,
            self.indptr[start : end + 1],
            self.pages,
            cache.shape[2],
            query,
            self.query_buffer[start:end],
            kv_page_stride(cache),
        )

    def finish(self, query, cache, out):
        assert query.shape == self.query_shape
        key, value = view_sparse_kv_pages(cache)
        q = self.query_buffer if self.query_buffer is not None else query
        self.wrapper.run(q, (key, value), out=out)


class StagedPrefillPages(NamedTuple):
    """Request pages remapped into a contiguous single-group BF16 buffer."""

    pages: torch.Tensor
    table: torch.Tensor


def plan_staged_prefill_pages(
    table: torch.Tensor, lengths: list[int], block_size: int
) -> StagedPrefillPages:
    """Map every live page of each request to a dense staging page."""
    counts = [(length + block_size - 1) // block_size for length in lengths]
    # Keep the block table's width: kernels specialize on its row stride, and a
    # length-dependent width would recompile them for every prompt length.
    width = table.shape[1]
    flat: list[int] = []
    remapped: list[list[int]] = []
    base = 0
    for row, count in enumerate(counts):
        flat.extend(range(row * table.shape[1], row * table.shape[1] + count))
        remapped.append(list(range(base, base + count)) + [0] * (width - count))
        base += count

    def upload(values):
        return (
            torch.tensor(values, dtype=torch.int32)
            .pin_memory()
            .to(table.device, non_blocking=True)
        )

    flat_table = table.reshape(-1) if table.is_contiguous() else table.flatten()
    return StagedPrefillPages(flat_table[upload(flat).long()], upload(remapped))


@triton.jit
def _stage_prefill_pages(
    Cache,
    Pages,
    Output,
    KScale,
    VScale,
    CACHE_PAGE: tl.constexpr,
    CACHE_TOKEN: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    FP8_CACHE: tl.constexpr,
):
    page = tl.program_id(0)
    token = tl.program_id(1)
    physical = tl.load(Pages + page).to(tl.int64)
    dim = tl.arange(0, BLOCK_D)
    mask = dim < DIM
    source = Cache + physical * CACHE_PAGE + token * CACHE_TOKEN
    key = tl.load(source + dim, mask)
    value = tl.load(source + DIM + dim, mask)
    if FP8_CACHE:
        key = key.to(tl.float32) * tl.load(KScale)
        value = value.to(tl.float32) * tl.load(VScale)
    destination = Output + (page * BLOCK_SIZE + token).to(tl.int64) * (2 * DIM)
    tl.store(destination + dim, key, mask)
    tl.store(destination + DIM + dim, value, mask)


def stage_prefill_pages(
    cache: torch.Tensor,
    staged: StagedPrefillPages,
    dtype: torch.dtype,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Copy one KV group's pages into ``[pages, 1, block, 2 * dim]`` ``dtype``.

    This lets FP8 and multi-group (strided) caches use the paged prefill
    kernels, which need contiguous model-dtype pages.
    """
    pages, groups, block_size, width = cache.shape
    assert groups == 1 and cache.stride(-1) == 1
    fp8_cache = cache.dtype in (torch.uint8, torch.float8_e4m3fn)
    if fp8_cache:
        assert k_scale is not None and v_scale is not None
        cache = cache.view(torch.float8_e4m3fn)
    output = torch.empty(
        (staged.pages.numel(), 1, block_size, width), dtype=dtype, device=cache.device
    )
    if staged.pages.numel():
        _stage_prefill_pages[(staged.pages.numel(), block_size)](
            cache,
            staged.pages,
            output,
            k_scale,
            v_scale,
            cache.stride(0),
            cache.stride(2),
            block_size,
            width // 2,
            triton.next_power_of_2(width // 2),
            fp8_cache,
        )
    return output
