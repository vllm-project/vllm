# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for dcp_localize_seq_lens, the elementwise interleaved-ownership
map used by the DeepSeek-V3.2 sparse indexer to convert per-token global
causal bounds into per-DCP-rank local KV counts."""

import pytest
import torch

from vllm.v1.attention.backends.utils import (
    dcp_localize_seq_lens,
    get_dcp_local_seq_lens,
)


def _brute_force_local_count(bound: int, rank: int, world: int, ilv: int) -> int:
    """Count positions < bound owned by `rank` under interleaved ownership:
    position j belongs to rank (j // ilv) % world."""
    return sum(1 for j in range(max(bound, 0)) if (j // ilv) % world == rank)


@pytest.mark.parametrize("world", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("ilv", [1, 2, 64])
def test_matches_reference_helper(world: int, ilv: int):
    bounds = torch.arange(0, 4 * world * ilv + 7, dtype=torch.int32)
    for rank in range(world):
        ours = dcp_localize_seq_lens(bounds, rank, world, ilv)
        ref = get_dcp_local_seq_lens(bounds, world, rank, ilv)
        assert torch.equal(ours, ref.to(ours.dtype))


@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("ilv", [1, 2])
def test_matches_brute_force_ownership(world: int, ilv: int):
    bounds = torch.arange(0, 3 * world * ilv + 5, dtype=torch.int32)
    for rank in range(world):
        ours = dcp_localize_seq_lens(bounds, rank, world, ilv)
        expected = [_brute_force_local_count(int(g), rank, world, ilv) for g in bounds]
        assert ours.tolist() == expected


@pytest.mark.parametrize("world", [1, 2, 4, 8])
def test_rank_sum_partitions_bound(world: int):
    # The per-rank local counts of any bound must partition it exactly.
    bounds = torch.arange(0, 6 * world + 3, dtype=torch.int32)
    total = sum(dcp_localize_seq_lens(bounds, rank, world) for rank in range(world))
    assert torch.equal(total, bounds)


def test_identity_without_dcp():
    bounds = torch.arange(0, 100, dtype=torch.int32)
    assert torch.equal(dcp_localize_seq_lens(bounds, 0, 1), bounds)
    # Identity holds for any interleave size when world == 1.
    assert torch.equal(dcp_localize_seq_lens(bounds, 0, 1, 64), bounds)


def test_arbitrary_shape_and_dtype():
    # The decode path applies the map to 2D (batch, next_n) native-MTP
    # bounds; the formula must be purely elementwise.
    bounds_2d = torch.tensor([[7, 8], [0, 1], [15, 16]], dtype=torch.int32)
    out = dcp_localize_seq_lens(bounds_2d, 1, 4)
    assert out.shape == bounds_2d.shape
    assert out.dtype == torch.int32
    flat = dcp_localize_seq_lens(bounds_2d.flatten(), 1, 4)
    assert torch.equal(out.flatten(), flat)


def test_padding_rows_stay_empty():
    # Padding entries on the expanded decode paths carry bound <= 0
    # (seq_len 0 expands to 0 - max_decode_len + j + 1); the localized
    # count must never go positive for them.
    bounds = torch.tensor([0, -1, -7], dtype=torch.int32)
    for rank in range(4):
        out = dcp_localize_seq_lens(bounds, rank, 4)
        assert (out <= 0).all()
