# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""QSA DCP selection localization and empty-owner behavior."""

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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernel needs a GPU")
@pytest.mark.parametrize("world,interleave", [(2, 1), (2, 4), (4, 16)])
def test_wide_selection_keeps_prefix_and_padding_stable(world, interleave):
    """A full-width selection must not race its padding stores."""
    from vllm.models.qwen4_exp.nvidia.ops.qsa_dcp import qsa_localize_dcp_indices

    width = 2051  # 2048-key budget plus the compression-group margin.
    rows = [list(range(width)), [0], []]
    packed = _pack(rows, width).cuda()
    local = torch.empty_like(packed)
    for rank in range(world):
        expected = [_reference_localize(row, world, rank, interleave) for row in rows]
        for _ in range(5):
            local.fill_(123)
            qsa_localize_dcp_indices(packed, local, world, rank, interleave)
            for row, (ids, count) in enumerate(expected):
                assert int(local[row, width]) == count
                assert local[row, :count].tolist() == ids
                assert bool((local[row, count:width] == PAD).all())


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


# --- contracts that keep the gate from being applied twice -------------------


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
    """The output gate must be applied after DCP merging, not by the kernel."""
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
