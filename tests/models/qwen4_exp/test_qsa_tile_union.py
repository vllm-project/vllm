# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tile-union QSA prefill kernel (RFC #55394) against the split-K kernel."""

import dataclasses

import pytest
import torch

from vllm.models.qwen4_exp.nvidia.ops import qsa as qsa_ops
from vllm.models.qwen4_exp.nvidia.ops import qsa_indexer as qsa_indexer_ops
from vllm.models.qwen4_exp.nvidia.ops import qsa_tile_union as tile_union
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON

requires_qsa_kernels = pytest.mark.skipif(
    not current_platform.is_cuda() or not HAS_TRITON,
    reason="QSA kernels require CUDA and Triton",
)

HEAD_DIM = 256
NUM_QUERY_HEADS = 24
NUM_KV_HEADS = 2
PAGE_SIZE = 1600
TOKEN_TOPK = 2048
COMPRESS_RATIO = 4
BLOCK_TOPK = TOKEN_TOPK // COMPRESS_RATIO
SELECTION_WIDTH = TOKEN_TOPK + COMPRESS_RATIO - 1
# With |q| ~ 2 the softmax is peaked: a dropped or leaked token moves the
# output by O(0.1), far above the bf16 summation-order noise (~1e-3).
QUERY_SCALE = 2.0
TOLERANCE = 2e-2
NEGATIVE_CONTROL_MIN = 5e-2


@pytest.fixture
def tile_union_forced(monkeypatch: pytest.MonkeyPatch):
    """Force the SM121 tile on whatever GPU runs the test."""
    monkeypatch.setenv("VLLM_QSA_TILE_UNION", "1")
    tile_union.qsa_tile_union_config.cache_clear()
    yield tile_union.qsa_tile_union_config()
    tile_union.qsa_tile_union_config.cache_clear()


class Case:
    """A prefill batch: contiguous rows per request, uneven request lengths,
    a permuted physical page table, selections with a high neighbour overlap,
    and every causal-tail length 0..CR-1 across the rows."""

    def __init__(
        self,
        request_lengths: list[int],
        seed: int = 0,
        context_pages: int = 5,
        invalid_rows: list[int] | None = None,
        padding_rows: int = 0,
        token_topk: int = TOKEN_TOPK,
    ) -> None:
        g = torch.Generator(device="cuda").manual_seed(seed)
        device = "cuda"
        self.token_topk = token_topk
        block_topk = token_topk // COMPRESS_RATIO
        selection_width = token_topk + COMPRESS_RATIO - 1
        self.num_requests = num_requests = len(request_lengths)
        lengths = torch.tensor(request_lengths, device=device, dtype=torch.int32)
        real_rows = int(lengths.sum())
        # padding_rows: CUDA-graph style rows past query_start_loc[-1] with an
        # invalid request id; both kernels must write zeros there.
        self.num_rows = num_rows = real_rows + padding_rows
        self.query_start_loc = torch.cat(
            [torch.zeros(1, device=device, dtype=torch.int32), lengths.cumsum(0)]
        ).to(torch.int32)
        self.token_to_req = torch.cat(
            [
                torch.repeat_interleave(
                    torch.arange(num_requests, device=device, dtype=torch.int32),
                    lengths,
                ),
                torch.full((padding_rows,), -1, device=device, dtype=torch.int32),
            ]
        )
        if invalid_rows:
            self.token_to_req[invalid_rows] = -1
        row = torch.arange(num_rows, device=device)
        req = torch.searchsorted(
            self.query_start_loc[1:], row.to(torch.int32), right=True
        ).clamp_(max=num_requests - 1)
        row_in_request = row - self.query_start_loc[req].long()
        # Each request's rows end at its context length; the context is a few
        # pages per request so the physical table spans many pages.
        context = torch.full((num_requests,), context_pages * PAGE_SIZE, device=device)
        context = context - (torch.arange(num_requests, device=device) % 7)
        # int64: the dtype of the production QSA metadata buffer.
        self.query_positions = (context[req] - lengths[req].long() + row_in_request).to(
            torch.int64
        )
        self.visible_blocks = (
            torch.minimum(
                (self.query_positions + 1) // COMPRESS_RATIO,
                context[req] // COMPRESS_RATIO,
            )
        ).to(torch.int32)
        # Selection: per request a random permutation of its blocks; each
        # row takes a window shifted by row (unique ids per row, adjacent
        # rows overlap on all but 8 blocks -- the prefill regime); ranks at
        # or beyond the row's visible-block count are padding, as the
        # indexer leaves them.
        blocks_per_page = PAGE_SIZE // COMPRESS_RATIO
        num_blocks = context_pages * blocks_per_page
        # A context shorter than the window leaves -1 holes inside the
        # ranked selection (the expansion and the union both skip them).
        perms = torch.full(
            (num_requests, block_topk + 8), -1, device=device, dtype=torch.int64
        )
        take = min(num_blocks, block_topk + 8)
        for request in range(num_requests):
            perms[request, :take] = torch.randperm(
                num_blocks, device=device, generator=g
            )[:take]
        window = (
            torch.arange(block_topk, device=device)[None, :]
            + (row_in_request % 8)[:, None]
        )
        block_indices = torch.gather(perms[req], 1, window)
        rank = torch.arange(block_topk, device=device)[None, :]
        block_indices = torch.where(
            rank < self.visible_blocks[:, None].long(), block_indices, -1
        )
        self.block_indices = block_indices.to(torch.int32).contiguous()
        self.logical_indices = torch.empty(
            (num_rows, selection_width + 1), device=device, dtype=torch.int32
        )
        qsa_indexer_ops.expand_qsa_block_indices(
            self.block_indices,
            self.query_positions,
            self.visible_blocks,
            COMPRESS_RATIO,
            token_topk,
            self.logical_indices,
        )
        num_cache_blocks = context_pages * num_requests + 3
        self.q = (
            torch.randn(num_rows, NUM_QUERY_HEADS, HEAD_DIM, device=device, generator=g)
            * QUERY_SCALE
        ).to(torch.bfloat16)
        # The server's layout: one [blocks, kv, page, 2D] tensor; K and V are
        # transposed views of it, so block stride != page * token stride.
        self.kv_cache = torch.randn(
            num_cache_blocks,
            NUM_KV_HEADS,
            PAGE_SIZE,
            2 * HEAD_DIM,
            device=device,
            generator=g,
        ).to(torch.bfloat16)
        self.k_cache, self.v_cache = self.kv_cache.transpose(1, 2).split(
            HEAD_DIM, dim=-1
        )
        self.block_table = (
            torch.randperm(num_cache_blocks, device=device, generator=g)[
                : num_requests * context_pages
            ]
            .view(num_requests, context_pages)
            .to(torch.int32)
            .contiguous()
        )
        self.inputs = tile_union.QSATileUnionInputs(
            block_indices=self.block_indices,
            logical_positions=self.query_positions,
            query_start_loc=self.query_start_loc,
            num_decode_tokens=0,
            num_prefills=num_requests,
            compress_ratio=COMPRESS_RATIO,
            token_topk=token_topk,
        )

    def run(self, tile_union: bool, logical_indices=None, inputs=None):
        return qsa_ops.qsa_sparse_paged_attention(
            self.q,
            self.k_cache,
            self.v_cache,
            self.logical_indices if logical_indices is None else logical_indices,
            self.block_table,
            self.token_to_req,
            use_prefill_config=True,
            tile_union=(inputs or self.inputs) if tile_union else None,
        )


def _max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max())


@requires_qsa_kernels
@pytest.mark.parametrize(
    ("request_lengths", "context_pages"),
    [
        pytest.param([1100], 5, id="single_request"),
        pytest.param([513, 1, 300, 257, 40], 2, id="uneven_requests_odd_lengths"),
        pytest.param([64] * 18, 5, id="many_requests_at_the_per_request_gate"),
        pytest.param([1030], 1, id="short_context_padded_selection"),
    ],
)
def test_tile_union_matches_split_k(
    tile_union_forced, request_lengths, context_pages
) -> None:
    assert tile_union_forced is not None
    case = Case(request_lengths, seed=len(request_lengths), context_pages=context_pages)
    expected = case.run(tile_union=False)
    actual = case.run(tile_union=True)
    diff = _max_diff(actual, expected)
    assert diff < TOLERANCE, f"tile-union vs split-K: max|diff| {diff:.4f}"
    # Every tail length occurred (the map covers the open block's 0..CR-1).
    tails = (case.query_positions + 1) % COMPRESS_RATIO
    assert set(tails.tolist()) == set(range(COMPRESS_RATIO))


@requires_qsa_kernels
def test_tile_union_runs_on_production_dtypes(
    monkeypatch: pytest.MonkeyPatch, tile_union_forced
) -> None:
    """int64 positions, int32 start locs, transposed cache views: the path must
    actually execute (a silent fallback would still pass the numeric check)."""
    calls = []
    real = tile_union.qsa_tile_union_attention

    def spy(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(qsa_ops, "qsa_tile_union_attention", spy)
    case = Case([2048], seed=11)
    assert case.inputs.logical_positions.dtype == torch.int64
    assert case.inputs.query_start_loc.dtype == torch.int32
    expected = case.run(tile_union=False)
    actual = case.run(tile_union=True)
    assert len(calls) == 1
    assert _max_diff(actual, expected) < TOLERANCE


@requires_qsa_kernels
def test_tile_union_zero_length_requests_and_padding_rows(tile_union_forced) -> None:
    """Zero-length requests in the layout and trailing padding rows past
    query_start_loc[-1] (invalid request id): exact zeros, neighbours intact."""
    case = Case([513, 0, 1, 300, 0, 257], seed=9, padding_rows=37)
    expected = case.run(tile_union=False)
    actual = case.run(tile_union=True)
    assert float(actual[-37:].float().abs().max()) == 0.0
    assert _max_diff(actual, expected) < TOLERANCE


@requires_qsa_kernels
def test_tile_union_production_padding_rows(tile_union_forced) -> None:
    """Padding rows exactly as _build_qsa_metadata_kernel leaves them for a
    single-request batch: token_to_req 0 (looks live), logical position -1,
    block ids -1. Output must be exactly zero there and the real rows intact."""
    case = Case([1200], seed=13, padding_rows=29)
    case.token_to_req[-29:] = 0
    positions = case.inputs.logical_positions.clone()
    positions[-29:] = -1
    block_indices = case.inputs.block_indices.clone()
    block_indices[-29:] = -1
    logical = case.logical_indices.clone()
    logical[-29:] = -1
    logical[-29:, -1] = 0  # packed count column: nothing to attend
    inputs = dataclasses.replace(
        case.inputs, logical_positions=positions, block_indices=block_indices
    )
    expected = case.run(tile_union=False, logical_indices=logical)
    actual = case.run(tile_union=True, inputs=inputs)
    assert float(actual[-29:].float().abs().max()) == 0.0
    assert float(expected[-29:].float().abs().max()) == 0.0
    assert _max_diff(actual, expected) < TOLERANCE


@requires_qsa_kernels
def test_tile_union_shared_layout(tile_union_forced) -> None:
    """The owner computes the row -> tile layout once per forward and hands
    it to every QSA layer; a precomputed layout gives the same result."""
    case = Case([513, 1, 300, 257, 40], seed=6, context_pages=2)
    layout = tile_union.qsa_tile_union_layout(
        case.query_start_loc, case.num_rows, case.num_requests, 2
    )
    tile_row0, tile_request, num_tiles = layout
    assert num_tiles >= 257 + 1 + 150 + 129 + 20
    assert int(tile_row0[0]) == 0 and int(tile_request[0]) == 0
    shared = dataclasses.replace(case.inputs, layout=layout)
    expected = case.run(tile_union=False)
    assert _max_diff(case.run(tile_union=True, inputs=shared), expected) < TOLERANCE


@requires_qsa_kernels
def test_tile_union_static_contract(tile_union_forced) -> None:
    ok = tile_union.qsa_tile_union_static_ok
    assert ok(4, 2048, 1600, 64)
    assert not ok(1, 2048, 1600, 64)  # ratio 1: nothing to union
    assert not ok(3, 2049, 1600, 64)  # not a power of two
    assert not ok(4, 0, 1600, 64)
    assert not ok(4, 2048, 1602, 64)  # a block would straddle a page
    assert not ok(4, 2048, 1600, 1 << 26)  # block ids reach the sentinel
    case = Case([1100], seed=1)
    config = tile_union_forced
    bad = dataclasses.replace(case.inputs, compress_ratio=3, token_topk=1536)
    assert not tile_union.qsa_tile_union_eligible(
        bad, case.num_rows, case.k_cache, case.block_table, config
    )
    bad = dataclasses.replace(
        case.inputs, logical_positions=case.inputs.logical_positions.to(torch.int32)
    )
    assert not tile_union.qsa_tile_union_eligible(
        bad, case.num_rows, case.k_cache, case.block_table, config
    )
    bad = dataclasses.replace(case.inputs, num_prefills=0)
    assert not tile_union.qsa_tile_union_eligible(
        bad, case.num_rows, case.k_cache, case.block_table, config
    )
    assert tile_union.qsa_tile_union_eligible(
        case.inputs, case.num_rows, case.k_cache, case.block_table, config
    )


@requires_qsa_kernels
def test_tile_union_non_power_of_two_budget(tile_union_forced) -> None:
    """block_topk 320 (budget 1,280): the pack kernel pads 640 keys to 1,024
    with a non-power-of-two pad width, which needs the masked arange."""
    case = Case([1100], seed=17, token_topk=1280)
    expected = case.run(tile_union=False)
    actual = case.run(tile_union=True)
    assert _max_diff(actual, expected) < TOLERANCE


@requires_qsa_kernels
def test_tile_union_test_has_power(tile_union_forced) -> None:
    """One swapped block per row must move the split-K output by >> tolerance,
    otherwise the comparison above proves nothing."""
    case = Case([1100], seed=7)
    expected = case.run(tile_union=False)
    corrupted = case.logical_indices.clone()
    victim = (case.query_positions // COMPRESS_RATIO) // 2
    corrupted[:, :COMPRESS_RATIO] = (
        victim[:, None] * COMPRESS_RATIO
        + torch.arange(COMPRESS_RATIO, device="cuda")[None, :]
    ).to(torch.int32)
    diff = _max_diff(case.run(tile_union=False, logical_indices=corrupted), expected)
    assert diff > NEGATIVE_CONTROL_MIN


@requires_qsa_kernels
def test_tile_union_masks_invalid_request_rows(tile_union_forced) -> None:
    case = Case([600, 500], seed=3, invalid_rows=[0, 7, 599, 600, 1001])
    expected = case.run(tile_union=False)
    actual = case.run(tile_union=True)
    for row in (0, 7, 599, 600, 1001):
        assert float(actual[row].float().abs().max()) == 0.0
    assert _max_diff(actual, expected) < TOLERANCE


@requires_qsa_kernels
def test_tile_union_gate(monkeypatch: pytest.MonkeyPatch, tile_union_forced) -> None:
    calls = []
    real = tile_union.qsa_tile_union_attention

    def spy(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(qsa_ops, "qsa_tile_union_attention", spy)
    case = Case([1100], seed=1)
    case.run(tile_union=True)
    assert len(calls) == 1
    # The owner decides eligibility before the indexer runs (it may skip the
    # expansion), so inputs for an ineligible batch are an error, not a
    # fallback: decode rows, too few rows, a fragmented batch, env off.
    decode_inputs = dataclasses.replace(case.inputs, num_decode_tokens=4)
    with pytest.raises(RuntimeError):
        case.run(tile_union=True, inputs=decode_inputs)
    small = Case([300], seed=2)
    with pytest.raises(RuntimeError):
        small.run(tile_union=True)
    fragmented = Case([8] * 140, seed=4)  # 1120 rows, 8 per request
    with pytest.raises(RuntimeError):
        fragmented.run(tile_union=True)
    case.run(tile_union=False)
    assert len(calls) == 1
    monkeypatch.setenv("VLLM_QSA_TILE_UNION", "0")
    tile_union.qsa_tile_union_config.cache_clear()
    assert tile_union.qsa_tile_union_config() is None
    with pytest.raises(RuntimeError):
        case.run(tile_union=True)
    assert len(calls) == 1


def test_tile_union_config_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    """auto / 0 / 1 / an explicit "R,BNB,warps,min_rows" tile for bring-up."""
    monkeypatch.setenv("VLLM_QSA_TILE_UNION", "0")
    tile_union.qsa_tile_union_config.cache_clear()
    assert tile_union.qsa_tile_union_config() is None
    monkeypatch.setenv("VLLM_QSA_TILE_UNION", "2,16,8,512")
    tile_union.qsa_tile_union_config.cache_clear()
    assert tile_union.qsa_tile_union_config() == tile_union.QSATileUnionConfig(
        rows_per_tile=2, blocks_per_step=16, num_warps=8, min_rows=512
    )
    monkeypatch.setenv("VLLM_QSA_TILE_UNION", "2,8,4,256,16")
    tile_union.qsa_tile_union_config.cache_clear()
    assert tile_union.qsa_tile_union_config().min_rows_per_request == 16
    for bad in (
        "2,16",
        "16,16,8,512",
        "0,8,4,512",
        "3,8,4,512",
        "2,12,4,512",
        "2,8,3,512",
        "2,64,4,512",
        "yes",
    ):
        monkeypatch.setenv("VLLM_QSA_TILE_UNION", bad)
        tile_union.qsa_tile_union_config.cache_clear()
        with pytest.raises(ValueError):
            tile_union.qsa_tile_union_config()
    tile_union.qsa_tile_union_config.cache_clear()


@requires_qsa_kernels
def test_tile_union_warmup(tile_union_forced) -> None:
    case = Case([64], seed=5)
    profile = tile_union.warmup_qsa_tile_union(
        case.kv_cache,
        case.block_table,
        num_query_heads=NUM_QUERY_HEADS,
        compress_ratio=COMPRESS_RATIO,
        token_topk=TOKEN_TOPK,
        config=tile_union_forced,
    )
    assert profile == (2, 32, 4)
