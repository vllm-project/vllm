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
    source = packed.clone()
    local = torch.empty_like(packed)
    qsa_localize_dcp_indices(packed, local, world, rank, interleave)
    # The layer's selection buffer must survive: MTP steps read it again.
    assert torch.equal(packed, source)
    for i, row in enumerate(rows):
        expected, count = _reference_localize(row[:width], world, rank, interleave)
        assert int(local[i, width]) == count
        got = [int(v) for v in local[i, :count]]
        assert got == expected
        # everything past the count must be padding
        assert all(int(v) == PAD for v in local[i, count:width])


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


def test_the_neutral_value_is_negative_infinity():
    """The identity of the LSE merge."""
    from vllm.models.qwen4_exp.nvidia.ops.qsa_dcp import (
        qsa_neutralize_empty_owner_lse_,
    )

    lse = torch.full((3, 2), 1.5)
    qsa_neutralize_empty_owner_lse_(lse, torch.tensor([False, True, False]))

    assert torch.isneginf(lse[1]).all()
    assert torch.equal(lse[0], torch.full_like(lse[0], 1.5)), "row 0 untouched"
    assert torch.equal(lse[2], torch.full_like(lse[2], 1.5)), "row 2 untouched"


def _reduce_like_correct_attn_out(outs, lses):
    """The arithmetic of correct_attn_out in vllm/v1/attention/ops/dcp.py.

    Base 2, because QSA scales its scores into log2 before the softmax. The
    guards are the point: they are what lets the caller skip zeroing an empty
    rank's output, so they are transcribed rather than assumed.
    """
    lses = torch.where(torch.isnan(lses) | torch.isposinf(lses), -torch.inf, lses)
    lse_max = lses.max(dim=0).values
    lse_max = torch.where(torch.isneginf(lse_max), torch.zeros_like(lse_max), lse_max)
    global_lse = torch.log2(torch.exp2(lses - lse_max).sum(dim=0)) + lse_max

    total = torch.zeros_like(outs[0])
    for rank in range(outs.shape[0]):
        exponent = lses[rank] - global_lse
        exponent = torch.where(
            torch.isnan(exponent) | torch.isposinf(exponent), -torch.inf, exponent
        )
        factor = torch.exp2(exponent)
        corrected = outs[rank] * factor.unsqueeze(-1)
        corrected = torch.where(
            (factor == 0.0).unsqueeze(-1), torch.zeros_like(corrected), corrected
        )
        total = total + corrected
    return total


def test_the_reducer_clears_an_empty_ranks_poisoned_payload():
    """Why the caller only has to fix the LSE.

    A sparse kernel leaves an unwritten row undefined, and `NaN * 0` is still
    NaN. The reducer forces the row to zero wherever the weight is zero, so the
    `-inf` alone is enough. If that ever stops being true, this test fails
    instead of a benchmark quietly returning NaN.
    """
    empty_out = torch.tensor([[[float("nan"), 1e30]]])
    empty_lse = torch.tensor([[-torch.inf]])
    good_out = torch.tensor([[[2.0, 4.0]]])
    good_lse = torch.tensor([[1.0]])

    merged = _reduce_like_correct_attn_out(
        torch.stack([empty_out, good_out]), torch.stack([empty_lse, good_lse])
    )
    assert torch.isfinite(merged).all(), "the empty rank poisoned the merge"
    torch.testing.assert_close(merged, good_out)


def test_every_rank_empty_still_reduces_to_a_finite_row():
    """A padding row: no rank owns anything, and nothing may become NaN."""
    outs = torch.stack([torch.full((1, 1, 2), float("nan"))] * 2)
    lses = torch.stack([torch.tensor([[-torch.inf]])] * 2)

    merged = _reduce_like_correct_attn_out(outs, lses)
    assert torch.equal(merged, torch.zeros_like(merged))


# --- contracts that keep the gate from being applied twice -------------------


def test_localize_refuses_a_mismatched_output():
    from vllm.models.qwen4_exp.nvidia.ops.qsa_dcp import qsa_localize_dcp_indices

    packed = _pack([[1, 2, 3]], 8)
    for bad in (torch.empty(1, 4, dtype=torch.int32), packed.float()):
        with pytest.raises(ValueError):
            qsa_localize_dcp_indices(packed, bad, 2, 0, 1)


def test_localize_leaves_the_source_alone_without_dcp():
    """World size 1 still copies, so one call site covers both paths."""
    from vllm.models.qwen4_exp.nvidia.ops.qsa_dcp import qsa_localize_dcp_indices

    packed = _pack([[5, 6], [7]], 8)
    source = packed.clone()
    local = torch.empty_like(packed)
    qsa_localize_dcp_indices(packed, local, 1, 0, 1)
    assert torch.equal(packed, source)
    assert torch.equal(local, source)


def test_the_kernel_wrapper_rejects_a_gate_with_return_lse():
    """Guards the one error that has no symptom: sigmoid applied twice."""
    from vllm.models.qwen4_exp.nvidia.ops import qsa as qsa_ops

    query = torch.zeros(2, 4, 64, dtype=torch.bfloat16)
    cache = torch.zeros(1, 16, 2, 64, dtype=torch.bfloat16)
    indices = _pack([[0], [0]], 8)
    block_table = torch.zeros(1, 1, dtype=torch.int32)
    token_to_req = torch.zeros(2, dtype=torch.int32)
    args = (query, cache, cache, indices, block_table, token_to_req, False)

    with pytest.raises(ValueError, match="do not pass one"):
        qsa_ops.qsa_sparse_paged_attention(
            *args, output_gate=torch.zeros_like(query), return_lse=True
        )
    with pytest.raises(ValueError, match="requires an output gate"):
        qsa_ops.qsa_sparse_paged_attention(*args)


# --- agreement with the slot mapping ----------------------------------------


def _slot_mapping_owner_and_local(g, world, rank, interleave, manager_block, page):
    """Transcribed from the DCP branch of vllm/v1/worker/block_table.py.

    That kernel decides which rank a position is written to, so it, not this
    module, defines ownership. Two block sizes matter and they are not always
    the same: the manager block it shards in (KV_CACHE_BLOCK_SIZE) and the
    kernel page the block table is indexed in (block_size).

    Returns (owned, block_table_index, slot_offset).
    """
    blocks_per_kv_block = manager_block // page
    virtual_block_size = manager_block * world
    vbi = g // virtual_block_size
    vbo = g - vbi * virtual_block_size
    owned = (vbo // interleave) % world == rank
    lbo = (vbo // (world * interleave)) * interleave + (vbo % interleave)
    return owned, vbi * blocks_per_kv_block + lbo // page, lbo % page


@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("page", [16, 64])
@pytest.mark.parametrize("blocks_per_kv_block", [1, 2, 4])
@pytest.mark.parametrize("interleave", [1, 4, 16])
def test_local_ids_address_the_same_slot_as_the_slot_mapping(
    world, page, blocks_per_kv_block, interleave
):
    """The one check that catches a wrong-keys bug with no other symptom.

    The invariant is that the interleave divides the MANAGER block, which is
    what vLLM asserts for DCP. blocks_per_kv_block > 1 is the case where the
    manager block and the kernel page come apart.
    """
    manager_block = page * blocks_per_kv_block
    if manager_block % interleave:
        pytest.skip("vLLM forbids this combination under DCP")

    for rank in range(world):
        for g in range(world * manager_block * 3):
            owned, block_index, slot_offset = _slot_mapping_owner_and_local(
                g, world, rank, interleave, manager_block, page
            )
            mine, _ = _reference_localize([g], world, rank, interleave)
            assert bool(mine) == owned, f"ownership disagrees at g={g}"
            if not owned:
                continue
            # The QSA kernel splits a local id exactly this way.
            local_id = mine[0]
            assert local_id // page == block_index
            assert local_id % page == slot_offset
