# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.attention.backends.mla.sparse_utils import (
    TopologyIndexConfig,
    apply_topology_tail_index_policy,
    merge_topology_tail_indices,
    merge_topology_tail_indices_reference,
)


def test_topology_tail_merge_benchmark_formats_markdown_row():
    from benchmarks.kernels.benchmark_sparse_mla_topology_index import (
        format_markdown_row,
    )

    row = format_markdown_row(
        {
            "rows": 256,
            "topk": 2048,
            "topology_width": 64,
            "learned_keep": 1536,
            "max_replacements": 64,
            "triton_us": 12.3456,
            "reference_us": 456.789,
            "speedup": 37.0001,
        }
    )

    assert row == "| 256 | 2048 | 64 | 1536 | 64 | 12.346 | 456.789 | 37.00x |"


def test_topology_tail_index_disabled_is_passthrough():
    learned_indices = torch.tensor([[0, 1, 4, -1]], dtype=torch.int32)
    scores = torch.arange(6, dtype=torch.float32).reshape(1, 6)
    segment_ids = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.int32)

    result = apply_topology_tail_index_policy(
        learned_indices,
        scores,
        segment_ids,
        config=TopologyIndexConfig(enabled=False),
    )

    assert result.indices is learned_indices
    assert result.applied is False
    assert result.fallback_reason == "disabled"


def test_topology_tail_index_falls_back_on_coordinate_mismatch():
    learned_indices = torch.tensor([[0, 1, 4, -1]], dtype=torch.int32)
    scores = torch.arange(6, dtype=torch.float32).reshape(1, 6)
    segment_ids = torch.tensor([0, 0, 1], dtype=torch.int32)

    result = apply_topology_tail_index_policy(
        learned_indices,
        scores,
        segment_ids,
        config=TopologyIndexConfig(enabled=True),
    )

    assert result.indices is learned_indices
    assert result.applied is False
    assert result.fallback_reason == "segment_shape"


def test_topology_tail_index_preserves_learned_prefix_and_injects_witnesses():
    learned_indices = torch.tensor([[0, 1, 2, 3, -1, -1]], dtype=torch.int32)
    scores = torch.tensor([[10.0, 9.0, 8.0, 1.0, 0.9, 0.8]], dtype=torch.float32)
    segment_ids = torch.tensor([0, 0, 0, 1, 1, 2], dtype=torch.int32)

    result = apply_topology_tail_index_policy(
        learned_indices,
        scores,
        segment_ids,
        config=TopologyIndexConfig(
            enabled=True,
            learned_fraction=0.5,
            max_segments=3,
            barrier_strength=20.0,
            diversity_strength=0.0,
            max_replacements=2,
        ),
    )

    assert result.applied is True
    assert result.indices.dtype == torch.int32
    assert result.indices.shape == learned_indices.shape
    assert result.indices[0, :3].tolist() == [0, 1, 2]

    selected = set(result.indices[0].tolist())
    assert {3, 4}.intersection(selected)
    assert 5 in selected
    assert result.learned_retained == 3
    assert result.structural_inserted == 2


def test_topology_tail_merge_reference_skips_duplicates_and_padding():
    learned_indices = torch.tensor(
        [[0, 1, 2, 3, 4, -1], [8, 9, 10, 11, -1, -1]],
        dtype=torch.int32,
    )
    topology_indices = torch.tensor(
        [[1, 5, 5, -1], [10, 12, 13, -1]],
        dtype=torch.int32,
    )

    merged = merge_topology_tail_indices_reference(
        learned_indices,
        topology_indices,
        learned_keep=3,
        max_replacements=2,
    )

    assert merged.tolist() == [
        [0, 1, 2, 5, 4, -1],
        [8, 9, 10, 12, 13, -1],
    ]


def test_topology_tail_merge_keeps_learned_tail_when_topology_budget_expires():
    learned_indices = torch.tensor([[0, 1, 2, 7, -1, -1]], dtype=torch.int32)
    topology_indices = torch.tensor([[6, 7]], dtype=torch.int32)

    merged = merge_topology_tail_indices_reference(
        learned_indices,
        topology_indices,
        learned_keep=2,
        max_replacements=1,
    )

    assert merged.tolist() == [[0, 1, 6, 7, -1, -1]]


def test_topology_tail_merge_cpu_dispatch_matches_reference():
    learned_indices = torch.tensor([[0, 1, 2, 3, -1, -1]], dtype=torch.int32)
    topology_indices = torch.tensor([[2, 4, 5]], dtype=torch.int32)

    expected = merge_topology_tail_indices_reference(
        learned_indices,
        topology_indices,
        learned_keep=2,
        max_replacements=3,
    )
    actual = merge_topology_tail_indices(
        learned_indices,
        topology_indices,
        learned_keep=2,
        max_replacements=3,
    )

    assert actual.dtype == torch.int32
    assert actual.shape == learned_indices.shape
    assert torch.equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_topology_tail_merge_triton_matches_reference_on_cuda():
    learned_indices = torch.tensor(
        [[0, 1, 2, 3, 4, -1], [8, 9, 10, 11, -1, -1]],
        dtype=torch.int32,
        device="cuda",
    )
    topology_indices = torch.tensor(
        [[1, 5, 5, -1], [10, 12, 13, -1]],
        dtype=torch.int32,
        device="cuda",
    )

    expected = merge_topology_tail_indices_reference(
        learned_indices.cpu(),
        topology_indices.cpu(),
        learned_keep=3,
        max_replacements=2,
    ).cuda()
    actual = merge_topology_tail_indices(
        learned_indices,
        topology_indices,
        learned_keep=3,
        max_replacements=2,
    )

    assert torch.equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_topology_tail_merge_triton_keeps_uninserted_topology_from_learned_tail():
    learned_indices = torch.tensor(
        [[0, 1, 2, 7, -1, -1]],
        dtype=torch.int32,
        device="cuda",
    )
    topology_indices = torch.tensor([[6, 7]], dtype=torch.int32, device="cuda")

    expected = merge_topology_tail_indices_reference(
        learned_indices.cpu(),
        topology_indices.cpu(),
        learned_keep=2,
        max_replacements=1,
    ).cuda()
    actual = merge_topology_tail_indices(
        learned_indices,
        topology_indices,
        learned_keep=2,
        max_replacements=1,
    )

    assert torch.equal(actual, expected)
