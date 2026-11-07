# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence

import pytest
import torch

from vllm.models.qwen4_exp.nvidia.ops import qsa_indexer as qsa_indexer_ops
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON, triton

requires_qsa_kernels = pytest.mark.skipif(
    not current_platform.is_cuda() or not HAS_TRITON,
    reason="QSA kernels require CUDA and Triton",
)

DEVICE = "cuda"
PAGE_SIZE = 16
PAGES = 6
HEADS = 4
HEAD_DIM = 32
# (rows per request, query_offset, num_queries); num_queries=None means all rows.
LAYOUTS = {
    "one_request": ((200,), 0, None),
    "eight_equal_chunks": ((96,) * 8, 0, None),
    "mixed_step": ((300,) + (1, 2, 3, 4) * 25, 0, None),
    "zero_row_requests": ((0, 70, 0, 0, 5, 0), 0, None),
    "request_longer_than_tile": ((130, 3), 0, None),
    "offset_into_chunk": ((70, 0, 0, 5, 1), 30, 40),
    "offset_across_requests": ((64, 64), 10, 100),
    "leading_empty_run": ((0, 0, 33), 0, None),
    "long_request_last": ((1, 2, 3, 4) * 40 + (300,), 0, None),
    "window_over_many_requests": ((7,) * 60, 50, 200),
    "requests_outside_the_window": ((10,) * 20 + (100,) + (10,) * 20, 200, 100),
}
CONFIGS = {
    torch.bfloat16: dict(TILE_R=64, BLOCK_N=64, K_TILES=16, STAGES=2, num_warps=4),
    torch.float8_e4m3fn: dict(TILE_R=32, BLOCK_N=64, K_TILES=16, STAGES=2, num_warps=8),
}
CACHE_DTYPES = [
    torch.bfloat16,
    pytest.param(torch.float8_e4m3fn, id="fp8"),
]
SENTINEL = -12345.0


def _make_batch(
    rows_per_request: Sequence[int], dtype: torch.dtype, seed: int = 0
) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    requests = len(rows_per_request)
    total = sum(rows_per_request)
    q = torch.randn((total, HEADS, HEAD_DIM), generator=generator)
    cache = torch.randn((requests * PAGES, PAGE_SIZE, 1, HEAD_DIM), generator=generator)
    table = torch.randperm(requests * PAGES, generator=generator).view(requests, PAGES)
    # Offsets share a non-zero base, as in a slice of a larger backing tensor.
    starts = torch.tensor([0, *rows_per_request]).cumsum(0) + 3
    visible = torch.randint(
        1, PAGES * PAGE_SIZE + 1, (total,), generator=generator, dtype=torch.int32
    )
    return (
        q.to(dtype).to(DEVICE),
        cache.to(dtype).to(DEVICE),
        table.to(torch.int32).to(DEVICE),
        starts.to(torch.int32).to(DEVICE),
        visible.to(DEVICE),
    )


def _launch(
    batch: tuple[torch.Tensor, ...],
    query_offset: int,
    num_queries: int,
    config: dict[str, int],
    compact: bool,
) -> torch.Tensor:
    q, cache, table, starts, visible = batch
    requests = table.shape[0]
    width = table.shape[1] * cache.shape[1]
    logits = torch.full((num_queries, width), SENTINEL, device=q.device)
    tile_r = config["TILE_R"]
    k_programs = triton.cdiv(width, config["BLOCK_N"] * config["K_TILES"])
    if compact:
        grid: tuple[int, ...] = (num_queries // tile_r + requests, 1, k_programs)
    else:
        longest = int((starts[1:] - starts[:-1]).max())
        grid = (requests, triton.cdiv(min(num_queries, longest), tile_r), k_programs)
    qsa_indexer_ops._qsa_mqa_paged_prefill_kernel[grid](
        q,
        cache,
        table,
        starts,
        visible,
        logits,
        *q.stride()[:-1],
        *cache.stride()[:2],
        *table.stride()[:-1],
        *logits.stride()[:-1],
        num_queries,
        query_offset,
        table.shape[1],
        requests,
        PAGE_SIZE=cache.shape[1],
        NUM_HEADS=q.shape[1],
        HEAD_DIM=q.shape[2],
        COMPACT=compact,
        STAGES=config["STAGES"],
        TILE_R=tile_r,
        BLOCK_N=config["BLOCK_N"],
        K_TILES=config["K_TILES"],
        num_warps=config["num_warps"],
    )
    return logits


def _window(layout: str) -> tuple[Sequence[int], int, int]:
    rows_per_request, offset, count = LAYOUTS[layout]
    return rows_per_request, offset, sum(rows_per_request) if count is None else count


@requires_qsa_kernels
@pytest.mark.parametrize("cache_dtype", CACHE_DTYPES)
@pytest.mark.parametrize("layout", list(LAYOUTS))
def test_qsa_prefill_compact_grid_is_bit_equal_to_the_request_grid(
    layout: str, cache_dtype: torch.dtype
) -> None:
    rows_per_request, offset, count = _window(layout)
    config = CONFIGS[cache_dtype]
    batch = _make_batch(rows_per_request, cache_dtype, seed=len(layout))
    stock = _launch(batch, offset, count, config, compact=False)
    compact = _launch(batch, offset, count, config, compact=True)
    assert torch.equal(compact, stock)
    # The reference must have written something, or the comparison is empty.
    assert bool((stock != SENTINEL).any())
    assert bool((stock != 0).any())
