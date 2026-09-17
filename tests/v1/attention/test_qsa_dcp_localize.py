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


# --- empty-owner rows -------------------------------------------------------


def test_empty_owner_comes_from_the_count_not_a_scan():
    """A reused buffer holds stale ids past its count.

    The attention kernel bounds its tile loop by the count, so a row with
    count 0 contributes nothing even though its columns still hold ids. A scan
    for `-1` would call that row non-empty and skip the neutralization, and the
    difference only appears under a captured graph.
    """
    from vllm.models.qwen4_exp.nvidia.ops.qsa_dcp import qsa_dcp_empty_owner_rows

    width = 4
    packed = torch.tensor([[7, 9, 11, 13, 0]], dtype=torch.int32)  # stale ids, count 0
    empty = qsa_dcp_empty_owner_rows(packed)
    assert bool(empty[0]) is True, "count 0 means empty owner"
    scan_says_empty = bool((packed[0, :width] == PAD).all())
    assert scan_says_empty is False, "a scan would disagree; that is the bug"


def test_empty_owner_mask_tracks_the_count():
    from vllm.models.qwen4_exp.nvidia.ops.qsa_dcp import qsa_dcp_empty_owner_rows

    packed = torch.tensor(
        [[0, 1, PAD, 2], [PAD, PAD, PAD, 0], [4, PAD, PAD, 1]], dtype=torch.int32
    )
    got = qsa_dcp_empty_owner_rows(packed).tolist()
    assert got == [False, True, False]


def test_neutral_values_are_zero_and_negative_infinity():
    """The identity of the LSE merge."""
    from vllm.models.qwen4_exp.nvidia.ops.qsa_dcp import qsa_neutralize_empty_owner_

    out = torch.full((3, 2, 4), 5.0)
    lse = torch.full((3, 2), 1.5)
    empty = torch.tensor([False, True, False])
    qsa_neutralize_empty_owner_(out, lse, empty)

    assert torch.equal(out[1], torch.zeros_like(out[1]))
    assert torch.isneginf(lse[1]).all()
    assert torch.equal(out[0], torch.full_like(out[0], 5.0)), "row 0 untouched"
    assert torch.equal(lse[2], torch.full_like(lse[2], 1.5)), "row 2 untouched"


def test_neutralizing_clears_a_poisoned_payload():
    """`NaN * 0 = NaN`. Zeroing the output is what stops it spreading.

    A sparse kernel can leave an unwritten row undefined. Relying on the `-inf`
    weight alone leaves the NaN inside the product, and the reduction then
    carries it to every rank.

    Modelled with two ranks, because that is the case that matters: one rank
    owns nothing and carries a poisoned payload, the other has a real result.
    The merged answer must be the real one.
    """
    from vllm.models.qwen4_exp.nvidia.ops.qsa_dcp import qsa_neutralize_empty_owner_

    empty_out = torch.tensor([[[float("nan"), 1e30]]])
    empty_lse = torch.tensor([[0.0]])
    qsa_neutralize_empty_owner_(empty_out, empty_lse, torch.tensor([True]))
    assert torch.isfinite(empty_out).all(), "a poisoned payload survived"

    good_out = torch.tensor([[[2.0, 4.0]]])
    good_lse = torch.tensor([[1.0]])

    lses = torch.stack([empty_lse, good_lse])
    outs = torch.stack([empty_out, good_out])
    lse_max = lses.max(dim=0).values
    weights = torch.exp(lses - lse_max)
    merged = (outs * weights.unsqueeze(-1)).sum(0) / weights.sum(0).unsqueeze(-1)

    assert torch.isfinite(merged).all(), "the empty rank poisoned the merge"
    torch.testing.assert_close(merged, good_out)
