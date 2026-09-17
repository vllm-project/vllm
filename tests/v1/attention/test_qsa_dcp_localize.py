# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Localizing a global QSA selection to one DCP rank's owned positions.

The reference here is a plain Python enumeration, not a second implementation
of the same formula, so a shared mistake cannot pass both.

The union property is the one that matters: across all ranks, the localized
selections must reconstruct the original global selection exactly once. A rank
that keeps a position it does not own double counts it in the merge, and a
position no rank keeps is silently dropped from the attention.
"""

import pytest
import torch

pytestmark = pytest.mark.cpu_test

PAD = -1


def _reference_localize(row, world, rank, interleave):
    """Plain enumeration. Returns (localized prefix, kept count)."""
    kept = []
    for g in row:
        if g < 0:
            continue
        if (g // interleave) % world != rank:
            continue
        kept.append((g // (world * interleave)) * interleave + (g % interleave))
    return kept, len(kept)


def _pack(rows, width):
    """Build the packed buffer: selection columns plus a trailing count."""
    out = torch.full((len(rows), width + 1), PAD, dtype=torch.int32)
    for i, row in enumerate(rows):
        for j, g in enumerate(row[:width]):
            out[i, j] = g
        out[i, width] = len(row[:width])
    return out


@pytest.mark.parametrize("world", [2, 4, 8])
@pytest.mark.parametrize("interleave", [1, 4, 16])
def test_union_across_ranks_reconstructs_the_selection(world, interleave):
    """Every selected position is kept by exactly one rank."""
    selection = list(range(0, world * interleave * 5, 3))
    seen: dict[int, int] = {}
    for rank in range(world):
        kept, count = _reference_localize(selection, world, rank, interleave)
        assert count == len(kept)
        for g in selection:
            if (g // interleave) % world == rank:
                seen[g] = seen.get(g, 0) + 1
    assert set(seen) == set(selection), "a selected position was dropped"
    assert all(n == 1 for n in seen.values()), "a position was kept twice"


@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("interleave", [1, 4])
def test_local_ids_are_dense_from_zero(world, interleave):
    """The kernel walks a dense prefix, so the ids must start at 0 with no gap."""
    selection = list(range(world * interleave * 6))
    for rank in range(world):
        kept, _ = _reference_localize(selection, world, rank, interleave)
        assert kept == sorted(kept), "localization must preserve order"
        assert kept == list(range(len(kept))), "local ids are not dense"


def test_padding_is_ignored():
    kept, count = _reference_localize([0, PAD, 8, PAD, 16], 2, 0, 4)
    assert count == len(kept)
    assert PAD not in kept


def test_a_rank_can_own_nothing():
    """The empty-owner row. Short contexts produce these routinely."""
    kept, count = _reference_localize([0, 1, 2, 3], world := 4, 3, 4)
    assert kept == [] and count == 0, "rank 3 owns none of positions 0-3"
    assert world == 4


def test_count_bounds_the_kernel_not_the_width():
    """The trailing count is the tile bound. Stale ids past it must not matter."""
    width = 8
    packed = _pack([[0, 4, 8, 12]], width)
    assert int(packed[0, width]) == 4
    assert int(packed[0, 4]) == PAD


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernel needs a GPU")
@pytest.mark.parametrize("world,rank,interleave", [(2, 0, 4), (2, 1, 4), (4, 2, 1)])
def test_kernel_matches_the_reference(world, rank, interleave):
    from vllm.models.qwen4_exp.nvidia.ops.qsa_dcp import qsa_localize_dcp_indices

    width = 32
    rows = [
        list(range(0, 64, 2)),
        list(range(1, 40)),
        [],
        [7],
    ]
    packed = _pack(rows, width).cuda()
    qsa_localize_dcp_indices(packed, world, rank, interleave)
    for i, row in enumerate(rows):
        expected, count = _reference_localize(row[:width], world, rank, interleave)
        assert int(packed[i, width]) == count
        got = [int(v) for v in packed[i, :count]]
        assert got == expected
        # everything past the count must be padding
        assert all(int(v) == PAD for v in packed[i, count:width])
