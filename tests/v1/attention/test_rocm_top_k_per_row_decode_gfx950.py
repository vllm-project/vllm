# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the gfx950 top-512 path of `top_k_per_row_decode`.

The ROCm sparse-MLA indexer calls this op for decode. The tests skip on every
other platform.
"""

import pytest
import torch

from vllm.platforms import current_platform


def _on_gfx950() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx950

    return on_gfx950()


requires_gfx950 = pytest.mark.skipif(
    not _on_gfx950(), reason="This test exercises the gfx950 launch configuration"
)


def _top_k_per_row_decode(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    indices: torch.Tensor,
    top_k: int,
    next_n: int = 1,
) -> None:
    torch.ops._C.top_k_per_row_decode(
        logits,
        next_n,
        lengths,
        indices,
        indices.shape[0],
        logits.stride(0),
        logits.stride(1),
        top_k,
    )


def _assert_exact_topk(
    logits: torch.Tensor, indices: torch.Tensor, row_ends: torch.Tensor
) -> None:
    ends = row_ends.reshape(-1, 1).clamp_min(0)
    valid = (indices >= 0) & (indices < ends)
    assert torch.equal(valid.sum(1), ends.clamp_max(indices.shape[1]).flatten())
    assert torch.all(valid | (indices == -1))
    ordered = indices.sort(1).values
    assert not ((ordered[:, 1:] == ordered[:, :-1]) & (ordered[:, 1:] >= 0)).any()

    width = int(ends.max())
    visible = logits[:, :width].clone()
    columns = torch.arange(width, device=logits.device)
    visible.masked_fill_(columns[None, :] >= ends, float("-inf"))
    expected = visible.topk(min(indices.shape[1], width), dim=1).values
    selected = logits.gather(1, indices.clamp_min(0).long())
    selected.masked_fill_(~valid, float("-inf"))
    selected = selected.sort(1, descending=True).values
    assert torch.equal(selected[:, : expected.shape[1]], expected)


@requires_gfx950
@torch.inference_mode()
def test_top_k_per_row_decode_gfx950_k512_1d_seq_lens() -> None:
    """Honor causal offsets and exact compressed lengths, including short rows."""
    next_n, width, top_k = 6, 16_384, 512
    lengths = torch.tensor(
        [0, 1, 511, 512, 513, 1023, 1024, 16_000],
        dtype=torch.int32,
        device="cuda",
    )
    offsets = torch.arange(next_n, dtype=torch.int32, device="cuda")
    row_ends = (lengths[:, None] - next_n + offsets + 1).clamp_min(0).flatten()
    rows = row_ends.numel()
    storage = torch.full((rows, width + 8), 1e20, dtype=torch.float32, device="cuda")
    logits = storage[:, 1 : width + 1]
    logits[:, :16_000] = torch.arange(16_000, dtype=torch.float32, device="cuda")
    indices = torch.full((rows, top_k), -777, dtype=torch.int32, device="cuda")
    _top_k_per_row_decode(logits, lengths, indices, top_k, next_n)
    _assert_exact_topk(logits, indices, row_ends)


@pytest.mark.parametrize(
    ("rows", "row_length"),
    [
        pytest.param(128, 65_536, id="single-before-64k"),
        pytest.param(128, 65_537, id="four-splits-after-64k"),
        pytest.param(129, 262_144, id="single-before-three-split-length"),
        pytest.param(129, 262_145, id="three-splits-after-256k"),
        pytest.param(160, 262_145, id="three-splits-upper-row-bound"),
        pytest.param(161, 262_144, id="single-before-two-split-length"),
        pytest.param(161, 262_145, id="two-splits-after-256k"),
        pytest.param(256, 262_145, id="two-splits-upper-row-bound"),
        pytest.param(257, 262_145, id="large-row-single-block"),
        pytest.param(384, 262_145, id="optimized-row-limit"),
        pytest.param(385, 262_145, id="native-fallback"),
    ],
)
@requires_gfx950
@torch.inference_mode()
def test_top_k_per_row_decode_gfx950_k512_split_policy_boundaries(
    rows: int, row_length: int
) -> None:
    width, top_k = 262_145, 512
    row = torch.arange(width, dtype=torch.float32, device="cuda")
    logits = row.expand(rows, -1)
    lengths = torch.full((rows,), row_length, dtype=torch.int32, device="cuda")
    indices = torch.empty((rows, top_k), dtype=torch.int32, device="cuda")

    _top_k_per_row_decode(logits, lengths, indices, top_k)

    expected = torch.arange(
        row_length - top_k, row_length, dtype=torch.int32, device="cuda"
    ).expand_as(indices)
    assert torch.equal(indices.sort(1).values, expected)


@requires_gfx950
@torch.inference_mode()
def test_top_k_per_row_decode_gfx950_k512_masked_graph_replay() -> None:
    """Preserve visible top-k scores when masks and lengths change on replay."""
    rows, next_n, width, top_k = 24, 6, 262_145, 512
    logits = torch.full((rows, width), 1e20, dtype=torch.float32, device="cuda")
    lengths = torch.full(
        (rows // next_n, next_n), 70_000, dtype=torch.int32, device="cuda"
    )
    indices = torch.full((rows, top_k), -777, dtype=torch.int32, device="cuda")
    logits[:, :70_000] = float("-inf")
    bin_sizes = [128, 129, 1024, 1025]
    for row in range(rows):
        pattern = row % 9
        if pattern < 4:
            count = [0, 1, 511, 513][pattern]
            logits[row, :count] = torch.arange(
                count, dtype=torch.float32, device="cuda"
            )
        elif pattern == 4:
            logits[row, :511] = 0
            logits[row, 511] = torch.finfo(torch.float32).min
        else:
            count = bin_sizes[pattern - 5]
            higher = max(0, top_k - count // 2)
            logits[row, :higher] = 2
            logits[row, higher : higher + count] = 1
            logits[row, higher] = 1 + 2**-23

    def run() -> None:
        _top_k_per_row_decode(logits, lengths, indices, top_k, next_n)

    run()
    _assert_exact_topk(logits, indices, lengths)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    indices.fill_(-777)
    graph.replay()
    _assert_exact_topk(logits, indices, lengths)

    bounds = [
        0,
        511,
        512,
        513,
        16_384,
        16_385,
        131_072,
        131_073,
        262_144,
        262_145,
    ]
    row_bounds = torch.tensor(bounds, dtype=torch.int32, device="cuda")
    repeats = (rows + len(bounds) - 1) // len(bounds)
    lengths.flatten().copy_(row_bounds.repeat(repeats)[:rows])
    logits[:] = torch.arange(width, dtype=torch.float32, device="cuda")
    indices.fill_(-777)
    graph.replay()
    _assert_exact_topk(logits, indices, lengths)


@requires_gfx950
@torch.inference_mode()
def test_top_k_per_row_decode_gfx950_k512_few_rows_one_block() -> None:
    """Keep rows of few-row calls up to 36K positions in one block."""
    next_n, width, narrow_width, top_k = 7, 524_288, 36_864, 512
    gen = torch.Generator(device="cuda").manual_seed(0)
    lengths = torch.tensor([20_000, 36_864], dtype=torch.int32, device="cuda")
    offsets = torch.arange(next_n, dtype=torch.int32, device="cuda")
    rows = lengths.numel() * next_n
    logits = torch.randn(rows, width, generator=gen, dtype=torch.float32, device="cuda")
    indices = torch.full((rows, top_k), -777, dtype=torch.int32, device="cuda")

    narrow = logits[:, :narrow_width]
    torch.accelerator.reset_peak_memory_stats()
    allocated = torch.accelerator.memory_allocated()
    _top_k_per_row_decode(narrow, lengths, indices, top_k, next_n)
    assert torch.accelerator.max_memory_allocated() == allocated
    row_ends = (lengths[:, None] - next_n + offsets + 1).flatten()
    _assert_exact_topk(narrow, indices, row_ends)

    def run() -> None:
        _top_k_per_row_decode(logits, lengths, indices, top_k, next_n)

    run()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()

    for step_lengths in ([20_000, 36_865], [16_391, 70_000], [20_000, 36_864]):
        lengths.copy_(torch.tensor(step_lengths, dtype=torch.int32))
        indices.fill_(-777)
        graph.replay()
        row_ends = (lengths[:, None] - next_n + offsets + 1).flatten()
        _assert_exact_topk(logits, indices, row_ends)
