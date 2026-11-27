# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HYV4 Flash ROCm row contract: AITER sparse MLA reads the first
`paged_kv_indptr` delta entries of each row and maps -1 to slot 0, so builder
counts must equal the indexers' live entries and rows must be front-packed.
"""

import pytest
import torch

from vllm.config import VllmConfig
from vllm.forward_context import set_forward_context
from vllm.platforms import current_platform
from vllm.utils.math_utils import round_up

if not current_platform.is_rocm():
    pytest.skip("HYV4 Flash ROCm sparse indices", allow_module_level=True)

from vllm.models.hy_v4_flash.amd import swa
from vllm.models.hy_v4_flash.amd.indexpool_ops import expand_groups, select_groups
from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
    ROCMAiterMLASparseMetadata,
)

DEVICE = "cuda"
# (seq_len, query_len) per request: decode, short and long extend, an empty
# padded request, a full prefill, and a decode past the compressed budget.
REQUESTS = [(1, 1), (300, 3), (0, 0), (64, 64), (2000, 1), (150, 64)]


def _logical_positions() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    seq_lens = torch.tensor([s for s, _ in REQUESTS], dtype=torch.int32)
    query_lens = torch.tensor([q for _, q in REQUESTS], dtype=torch.int32)
    query_start_loc = torch.zeros(len(REQUESTS) + 1, dtype=torch.int32)
    torch.cumsum(query_lens, 0, out=query_start_loc[1:])
    positions = torch.cat(
        [torch.arange(s - q, s, dtype=torch.int64) for s, q in REQUESTS]
    )
    return (
        seq_lens.to(DEVICE),
        query_start_loc.to(DEVICE),
        positions.to(DEVICE),
    )


def _assert_front_packed(rows: torch.Tensor, counts: torch.Tensor) -> None:
    columns = torch.arange(rows.shape[1], device=rows.device)
    head = columns < counts[:, None].long()
    assert torch.all((rows >= 0) == head)


@pytest.mark.parametrize("window", [0, 128], ids=["compressed", "swa"])
def test_live_counts_match_indexer_rows(window):
    ratio, budget, width = 16, 7, 128
    seq_lens, query_start_loc, positions = _logical_positions()
    num_tokens = positions.numel()
    if window:
        rows = swa.window_indices(positions, window, width)
    else:
        upper = (positions // ratio).int()
        torch.manual_seed(0)
        logits = torch.randn(num_tokens, int(upper.max()) + 1, device=DEVICE)
        rows = expand_groups(select_groups(logits, upper, budget), positions, ratio)

    counts = swa.live_counts(
        seq_lens,
        query_start_loc,
        num_tokens,
        max(q for _, q in REQUESTS),
        window,
        ratio,
        budget,
    )

    torch.testing.assert_close(counts, (rows >= 0).sum(-1).int(), rtol=0, atol=0)
    _assert_front_packed(rows, counts)


def _live_counts_reference(seq_lens, query_lens, window, ratio, budget):
    counts = []
    for seq_len, query_len in zip(seq_lens, query_lens):
        for logical in range(seq_len - query_len, seq_len):
            if seq_len <= 0 or logical < 0:
                counts.append(0)
            elif window:
                counts.append(min(logical + 1, window))
            else:
                groups = min(logical // ratio, budget)
                counts.append(groups * ratio + logical % ratio + 1)
    return torch.tensor(counts, dtype=torch.int32)


# (seq_len, query_len): decode, multi-block prefill (>64 rows per request),
# chunked extend, budget-saturated decode, then padded empty requests.
COUNT_REQUESTS = [(1, 1), (200, 200), (517, 130), (5000, 1), (0, 0), (0, 0)]


@pytest.mark.parametrize(
    "window,ratio,budget",
    [(0, 4, 511), (0, 16, 3), (1, 4, 511), (512, 4, 511)],
    ids=["compressed-r4", "compressed-r16", "w1", "w512"],
)
def test_live_counts_match_reference_formula(window, ratio, budget):
    """Builder counts follow the per-token window / group-budget formula."""
    seq_lens = [s for s, _ in COUNT_REQUESTS]
    query_lens = [q for _, q in COUNT_REQUESTS]
    starts = torch.zeros(len(COUNT_REQUESTS) + 1, dtype=torch.int32)
    torch.cumsum(torch.tensor(query_lens, dtype=torch.int32), 0, out=starts[1:])

    counts = swa.live_counts(
        torch.tensor(seq_lens, dtype=torch.int32, device=DEVICE),
        starts.to(DEVICE),
        sum(query_lens),
        max(query_lens),
        window,
        ratio,
        budget,
    )

    expected = _live_counts_reference(seq_lens, query_lens, window, ratio, budget)
    torch.testing.assert_close(counts.cpu(), expected, rtol=0, atol=0)


def test_live_counts_empty_batch_is_empty():
    starts = torch.zeros(1, dtype=torch.int32, device=DEVICE)
    seq_lens = torch.zeros(0, dtype=torch.int32, device=DEVICE)
    assert swa.live_counts(seq_lens, starts, 0, 0, 128, 4, 511).numel() == 0


def _window_rows_reference(seq_lens, starts, num_actual, rows, window, width):
    expected = torch.full((rows, width), -1, dtype=torch.int32)
    for seq_len, start, end in zip(seq_lens, starts[:-1], starts[1:]):
        for row in range(start, min(end, num_actual)):
            logical = seq_len - (end - start) + row - start
            first = max(0, logical - window + 1)
            live = torch.arange(first, logical + 1, dtype=torch.int32)
            expected[row, : live.numel()] = live
    return expected


def _window_metadata(seq_lens, query_lens, num_actual):
    starts = [0]
    for query_len in query_lens:
        starts.append(starts[-1] + query_len)
    zeros = torch.zeros(starts[-1], dtype=torch.int32, device=DEVICE)
    return ROCMAiterMLASparseMetadata(
        num_reqs=len(seq_lens),
        max_query_len=max(query_lens),
        max_seq_len=max(seq_lens),
        num_actual_tokens=num_actual,
        query_start_loc=torch.tensor(starts, dtype=torch.int32, device=DEVICE),
        slot_mapping=zeros.long(),
        block_table=torch.zeros(len(seq_lens), 1, dtype=torch.int32, device=DEVICE),
        req_id_per_token=zeros,
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=DEVICE),
    ), starts


def _run_window_indexer(indexer, metadata, rows):
    hidden = torch.empty(rows, 8, device=DEVICE)
    with set_forward_context({"layer.attn": metadata}, VllmConfig()):
        return indexer(hidden, None, None)


@pytest.mark.parametrize("window", [1, 100, 128])
def test_window_indexer_matches_reference(window):
    """Each live row holds its causal window front-packed; padding rows and
    rows past `num_actual_tokens` stay -1."""
    width = round_up(window, 128)
    buffer = torch.full((64, 256), 7, dtype=torch.int32, device=DEVICE)
    indexer = swa.FlashWindowIndexer(window, buffer, "layer.attn")
    # Prefill, decode deep in history, an empty request, a short extend; the
    # last live token is cut by num_actual_tokens and 4 padding rows follow.
    seq_lens, query_lens, rows = [9, 700, 0, 130], [9, 1, 0, 3], 17
    metadata, starts = _window_metadata(seq_lens, query_lens, 12)

    out = _run_window_indexer(indexer, metadata, rows)

    expected = _window_rows_reference(seq_lens, starts, 12, rows, window, width)
    assert out.shape == (rows, width)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


def test_window_indexer_cuda_graph_replay():
    """A captured window indexer follows new metadata contents on replay."""
    window, rows = 200, 8
    indexer = swa.FlashWindowIndexer(
        window, torch.empty(rows, 256, dtype=torch.int32, device=DEVICE), "layer.attn"
    )
    # Graph capture keeps one padded token count; the live boundary comes from
    # query_start_loc on the device.
    metadata, _ = _window_metadata([5, 6, 7, 8], [1, 1, 1, 1], rows)
    _run_window_indexer(indexer, metadata, rows)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = _run_window_indexer(indexer, metadata, rows)

    for seq_lens, query_lens in (
        ([1, 300, 199, 201], [1, 1, 1, 1]),
        ([40, 1000, 3, 0], [3, 2, 1, 0]),
    ):
        new, starts = _window_metadata(seq_lens, query_lens, rows)
        metadata.seq_lens.copy_(new.seq_lens)
        metadata.query_start_loc.copy_(new.query_start_loc)
        graph.replay()
        expected = _window_rows_reference(
            seq_lens, starts, rows, rows, window, out.shape[1]
        )
        torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
