# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU planning of Triton ReplaySSM standard decode (plan_replayssm_decode).

Each decode step's write_pos/is_flush comes from per-slot ring counters that
prefill resets and each planned step advances. These tests drive the counters
through whole decode runs and compare every step with the closed form the
runner used to compute on the CPU from each request's admission length.
"""

import pytest
import torch

from vllm.model_executor.layers.mamba.ops.ssu_dispatch import (
    plan_replayssm_decode,
    reset_replayssm_ring_trackers,
)
from vllm.platforms import current_platform
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="Triton GPU kernel"
)

DEVICE = "cuda"
NUM_SLOTS = 8  # slot 0 is NULL_BLOCK_ID (padding)


def _closed_form(
    num_computed: int, decode_base: int, buffer_len: int, align_block_size: int
) -> tuple[int, int]:
    """The previous CPU derivation of one standard-decode row's plan."""
    base = decode_base
    if align_block_size:
        base = max(base, (num_computed // align_block_size) * align_block_size)
    write_pos = (num_computed - base) % buffer_len
    is_flush = write_pos == buffer_len - 1
    if align_block_size and (num_computed + 1) % align_block_size == 0:
        is_flush = True
    return write_pos, int(is_flush)


class _Planner:
    def __init__(self, buffer_len: int, align_block_size: int = 0):
        self.buffer_len = buffer_len
        self.align_block_size = align_block_size
        self.trackers = [
            torch.zeros(NUM_SLOTS, dtype=torch.int32, device=DEVICE) for _ in range(3)
        ]
        self.write_pos = torch.full((NUM_SLOTS,), -1, dtype=torch.int32, device=DEVICE)
        self.is_flush = torch.full((NUM_SLOTS,), -1, dtype=torch.int8, device=DEVICE)

    @property
    def ring_pos(self) -> torch.Tensor:
        return self.trackers[0]

    def reset(self, slots: list[int]) -> None:
        """What a prefill of these slots does."""
        ring_start, prev_num_accepted, prev_query_len = self.trackers
        reset_replayssm_ring_trackers(
            ring_start,
            prev_num_accepted,
            prev_query_len,
            torch.tensor(slots, dtype=torch.int32, device=DEVICE),
        )

    def step(
        self, slots: list[int], seq_lens: list[int]
    ) -> tuple[list[int], list[int]]:
        state_indices = torch.tensor(slots, dtype=torch.int32, device=DEVICE)
        plan_replayssm_decode(
            self.ring_pos,
            state_indices.unsqueeze(-1),
            self.write_pos,
            self.is_flush,
            buffer_len=self.buffer_len,
            seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=DEVICE),
            align_block_size=self.align_block_size,
        )
        n = len(slots)
        return self.write_pos[:n].tolist(), self.is_flush[:n].tolist()


@pytest.mark.parametrize(
    "buffer_len,align_block_size",
    [(16, 0), (4, 0), (6, 0), (16, 16), (16, 32), (6, 16), (4, 64)],
)
@pytest.mark.parametrize("prompt_len", [100, 112, 127, 128])
def test_plan_matches_closed_form_over_decode_run(
    buffer_len, align_block_size, prompt_len
):
    planner = _Planner(buffer_len, align_block_size)
    planner.reset([3])
    for num_computed in range(prompt_len, prompt_len + 3 * max(64, buffer_len)):
        got = planner.step([3], [num_computed + 1])
        expected = _closed_form(num_computed, prompt_len, buffer_len, align_block_size)
        assert got == ([expected[0]], [expected[1]]), num_computed


@pytest.mark.parametrize("align_block_size", [0, 16])
def test_rows_plan_independently(align_block_size):
    """Rows at different points of their runs; NULL padding rows are no-ops."""
    buffer_len = 16
    planner = _Planner(buffer_len, align_block_size)
    bases = {1: 100, 4: 105, 6: 200}
    planner.reset(list(bases))
    # Advance each slot to its own num_computed with single-row steps.
    targets = {1: 103, 4: 105, 6: 214}
    for slot, base in bases.items():
        for num_computed in range(base, targets[slot]):
            planner.step([slot], [num_computed + 1])
    untouched_before = planner.ring_pos[[0, 2]].clone()

    slots = [1, 4, 6, NULL_BLOCK_ID, NULL_BLOCK_ID]
    seq_lens = [targets[1] + 1, targets[4] + 1, targets[6] + 1, 0, 0]
    write_pos, is_flush = planner.step(slots, seq_lens)

    expected = [
        _closed_form(targets[s], bases[s], buffer_len, align_block_size)
        for s in (1, 4, 6)
    ]
    assert write_pos == [wp for wp, _ in expected] + [0, 0]
    assert is_flush == [fl for _, fl in expected] + [0, 0]
    assert torch.equal(planner.ring_pos[[0, 2]], untouched_before)


def test_prefill_reset_reanchors_resumed_request():
    """A preempted request re-runs prefill on readmission, so its ring
    restarts at the resume point rather than the original prompt end."""
    planner = _Planner(16)
    planner.reset([2])
    for num_computed in range(100, 105):
        planner.step([2], [num_computed + 1])
    assert planner.step([2], [106]) == ([5], [0])

    planner.reset([2])  # readmitted with 105 tokens of context
    assert planner.step([2], [106]) == ([0], [0])


def test_align_block_start_ignores_stale_slot_counter():
    """A prefix-cache hit lands on a block boundary without prefilling the
    slot, so its counter may be stale; the block start restarts it."""
    planner = _Planner(16, align_block_size=16)
    planner.ring_pos[5] = 9  # left over from the slot's previous owner
    assert planner.step([5], [113]) == ([0], [0])
    assert planner.step([5], [114]) == ([1], [0])


def test_stale_counter_beyond_buffer_flushes():
    planner = _Planner(16)
    planner.ring_pos[1] = 40
    assert planner.step([1], [500]) == ([15], [1])
    assert planner.ring_pos[1].item() == 0


def test_single_token_prompt_tail_appends_after_chunk_reset():
    """A one-token final prefill chunk runs as decode on the ring that the
    preceding chunk's prefill left empty."""
    planner = _Planner(16)
    planner.ring_pos[2] = 7
    planner.reset([2])  # earlier prompt chunk
    assert planner.step([2], [100]) == ([0], [0])
    assert planner.step([2], [101]) == ([1], [0])
