# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Localizing a frozen MTP selection, step after step, under DCP.

MTP selects once on step 0 and reuses those rows for the draft steps that
follow; ``compact_topk_indices`` reorders them between steps. Under DCP each
rank then has to keep its own share of that shared selection, every step,
without the shared buffer drifting and without a previous step's scratch
leaking into this one.

These tests work on the buffers, which is where that risk lives. They do not
run a model: a full metadata and indexer path with two real ranks is still
outstanding, so DCP with MTP is not claimed as supported on this evidence.
"""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="the localization kernel needs CUDA"
)

PAD = -1
WIDTH = 32


def _pack(rows, width=WIDTH):
    packed = torch.full((len(rows), width + 1), PAD, dtype=torch.int32, device="cuda")
    for i, row in enumerate(rows):
        for j, g in enumerate(row[:width]):
            packed[i, j] = g
        packed[i, width] = len(row[:width])
    return packed


def _localized_rows(source, world, rank, interleave, scratch=None):
    """One step: localize the shared selection into this rank's scratch."""
    from vllm.models.qwen4_exp.nvidia.ops.qsa_dcp import qsa_localize_dcp_indices

    if scratch is None:
        scratch = torch.empty_like(source)
    local = qsa_localize_dcp_indices(source, scratch, world, rank, interleave)
    return [
        [int(v) for v in local[i, : int(local[i, WIDTH])]]
        for i in range(local.shape[0])
    ]


def _globalize(local_ids, world, rank, interleave):
    """Invert the compact-local id, so ranks can be compared on one scale."""
    return [
        (lid // interleave) * world * interleave
        + rank * interleave
        + (lid % interleave)
        for lid in local_ids
    ]


# Prior lengths chosen to straddle compression-group boundaries at 32 and 48.
STEP0 = [
    list(range(0, 31)),  # a prefill row ending just before 32
    list(range(0, 46, 2)),  # a prefill row straddling 48
    [3, 9, 17, 40],  # a decode row
]


@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("interleave", [1, 4])
def test_the_shared_selection_survives_every_rank(world, interleave):
    """Localizing must not disturb the buffer the next MTP step reads."""
    source = _pack(STEP0)
    frozen = source.clone()

    for rank in range(world):
        _localized_rows(source, world, rank, interleave)
        assert torch.equal(source, frozen), f"rank {rank} mutated the shared selection"


@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("interleave", [1, 4])
def test_the_ranks_partition_each_frozen_row_once(world, interleave):
    source = _pack(STEP0)

    for row, expected in enumerate(STEP0):
        seen = []
        for rank in range(world):
            local = _localized_rows(source, world, rank, interleave)[row]
            seen += _globalize(local, world, rank, interleave)
        assert sorted(seen) == sorted(expected), f"row {row} was not partitioned"


@pytest.mark.parametrize("world", [2, 4])
def test_a_later_step_does_not_inherit_the_earlier_scratch(world):
    """The claim under attack: reusing one scratch buffer makes steps couple."""
    interleave = 4
    step0 = _pack(STEP0)
    # MTP keeps each request's target-aligned row, which reorders and shortens.
    compacted = _pack([STEP0[2], STEP0[0]])

    for rank in range(world):
        scratch = torch.empty_like(step0)
        _localized_rows(step0, world, rank, interleave, scratch)

        # Step 1 reads the compacted rows through the same scratch. Its first
        # two rows now hold different content than step 0 left there.
        fresh = _localized_rows(
            compacted, world, rank, interleave, scratch[: compacted.shape[0]]
        )
        independent = _localized_rows(compacted, world, rank, interleave)
        assert fresh == independent, "step 1 inherited step 0's scratch"


@pytest.mark.parametrize("world", [2, 4])
def test_a_shrinking_batch_cannot_read_a_stale_tail(world):
    """Fewer rows next step: the tail stays, and must never be read.

    The count column is the kernel's loop bound, so a stale row past the batch
    is only safe while nothing reads past the batch. Pin that.
    """
    interleave = 1
    big = _pack([list(range(0, 20)), list(range(1, 21)), list(range(2, 22))])
    small = _pack([list(range(5, 15))])

    for rank in range(world):
        scratch = torch.empty_like(big)
        _localized_rows(big, world, rank, interleave, scratch)
        stale_tail = scratch[1:].clone()

        got = _localized_rows(small, world, rank, interleave, scratch[:1])
        assert got == _localized_rows(small, world, rank, interleave)
        assert torch.equal(scratch[1:], stale_tail), "the tail should be untouched"
