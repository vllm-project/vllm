# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for reorder_batch_to_split_cp_and_normal."""

from dataclasses import dataclass

import pytest


# ---------------------------------------------------------------------------
# Inline the function under test to avoid heavy vllm imports (torch, etc.)
# ---------------------------------------------------------------------------

def reorder_batch_to_split_cp_and_normal(input_batch, scheduler_output):
    """Move CP requests to the front while preserving in-group order.

    Copied from vllm/v1/attention/backends/utils.py for isolated testing.
    """
    req_ids = input_batch.req_ids
    num_reqs = len(req_ids)
    cp_indices = [
        idx
        for idx, req_id in enumerate(req_ids)
        if scheduler_output.cp_rank_scheduled_tokens[req_id] > 1
    ]
    ncp_indices = [
        idx
        for idx, req_id in enumerate(req_ids)
        if scheduler_output.cp_rank_scheduled_tokens[req_id] <= 1
    ]
    target_order = cp_indices + ncp_indices

    if target_order == list(range(num_reqs)):
        return False

    if hasattr(input_batch, "apply_permutation"):
        return input_batch.apply_permutation(target_order)

    items_at_pos = list(range(num_reqs))
    pos_of_item = list(range(num_reqs))

    for dst_pos, target_item in enumerate(target_order):
        src_pos = pos_of_item[target_item]
        if src_pos == dst_pos:
            continue

        input_batch.swap_states(dst_pos, src_pos)

        displaced_item = items_at_pos[dst_pos]
        items_at_pos[dst_pos], items_at_pos[src_pos] = (
            items_at_pos[src_pos],
            items_at_pos[dst_pos],
        )
        pos_of_item[target_item] = dst_pos
        pos_of_item[displaced_item] = src_pos
    return True


# ---------------------------------------------------------------------------
# Mocks
# ---------------------------------------------------------------------------

class MockInputBatch:
    """Minimal InputBatch that tracks req_ids ordering via swap_states."""

    def __init__(self, req_ids: list[str], *, has_apply_permutation: bool = False):
        self.req_ids = list(req_ids)
        self.swap_log: list[tuple[int, int]] = []
        self._has_apply_permutation = has_apply_permutation

    if True:  # always define; conditionally exposed via __getattr__

        def _apply_permutation(self, permutation):
            self.req_ids[:] = [self.req_ids[i] for i in permutation]
            return True

    def __getattr__(self, name):
        if name == "apply_permutation":
            if self._has_apply_permutation:
                return self._apply_permutation
            raise AttributeError(name)
        raise AttributeError(name)

    def swap_states(self, i: int, j: int) -> None:
        self.swap_log.append((i, j))
        self.req_ids[i], self.req_ids[j] = self.req_ids[j], self.req_ids[i]


class MockSchedulerOutput:
    def __init__(self, cp_rank_scheduled_tokens: dict[str, int]):
        self.cp_rank_scheduled_tokens = cp_rank_scheduled_tokens


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

@dataclass
class CPReorderTestCase:
    # cp_rank value per request: >1 means CP, <=1 means non-CP
    cp_ranks: list[int]
    expected_order: list[int]
    expected_modified: bool
    description: str = ""


TEST_CASES: dict[str, CPReorderTestCase] = {
    "all_cp": CPReorderTestCase(
        cp_ranks=[4, 4, 4],
        expected_order=[0, 1, 2],
        expected_modified=False,
        description="All CP, already in order",
    ),
    "all_ncp": CPReorderTestCase(
        cp_ranks=[1, 1, 1],
        expected_order=[0, 1, 2],
        expected_modified=False,
        description="All non-CP, already in order",
    ),
    "already_ordered": CPReorderTestCase(
        cp_ranks=[8, 4, 1, 1],
        expected_order=[0, 1, 2, 3],
        expected_modified=False,
        description="CP already at front",
    ),
    "single_request": CPReorderTestCase(
        cp_ranks=[4],
        expected_order=[0],
        expected_modified=False,
        description="Only one request",
    ),
    "swap_one_pair": CPReorderTestCase(
        # positions: ncp, cp -> should become cp, ncp
        cp_ranks=[1, 4],
        expected_order=[1, 0],
        expected_modified=True,
        description="Simple swap of two",
    ),
    "ncp_before_cp": CPReorderTestCase(
        # [ncp0, ncp1, cp0, cp1] -> [cp0, cp1, ncp0, ncp1]
        cp_ranks=[1, 1, 4, 4],
        expected_order=[2, 3, 0, 1],
        expected_modified=True,
        description="All non-CP before all CP",
    ),
    "interleaved": CPReorderTestCase(
        # [cp0, ncp0, cp1, ncp1] -> [cp0, cp1, ncp0, ncp1]
        cp_ranks=[4, 1, 4, 1],
        expected_order=[0, 2, 1, 3],
        expected_modified=True,
        description="Interleaved CP and non-CP",
    ),
    "complex_mixed": CPReorderTestCase(
        # [ncp, cp, ncp, cp, cp, ncp]
        # CP indices: [1, 3, 4], NCP indices: [0, 2, 5]
        # target: [1, 3, 4, 0, 2, 5]
        cp_ranks=[1, 8, 1, 4, 2, 1],
        expected_order=[1, 3, 4, 0, 2, 5],
        expected_modified=True,
        description="Complex interleaving preserves in-group order",
    ),
    "large_batch": CPReorderTestCase(
        # 8 requests: indices 0,2,4,6 are CP; 1,3,5,7 are NCP
        # target: [0,2,4,6, 1,3,5,7]
        cp_ranks=[4, 1, 4, 1, 4, 1, 4, 1],
        expected_order=[0, 2, 4, 6, 1, 3, 5, 7],
        expected_modified=True,
        description="Large alternating batch",
    ),
    "single_cp_at_end": CPReorderTestCase(
        # [ncp, ncp, ncp, cp] -> [cp, ncp, ncp, ncp]
        cp_ranks=[1, 1, 1, 4],
        expected_order=[3, 0, 1, 2],
        expected_modified=True,
        description="Single CP at end moves to front",
    ),
    "single_ncp_at_start": CPReorderTestCase(
        # [ncp, cp, cp, cp] -> [cp, cp, cp, ncp]
        cp_ranks=[1, 4, 4, 4],
        expected_order=[1, 2, 3, 0],
        expected_modified=True,
        description="Single NCP at start moves to end",
    ),
}


# ---------------------------------------------------------------------------
# Tests using swap_states path (no apply_permutation)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "test_case", TEST_CASES.values(), ids=TEST_CASES.keys()
)
def test_reorder_swap_path(test_case: CPReorderTestCase):
    """Exercise the swap_states-based reordering path."""
    num_reqs = len(test_case.cp_ranks)
    req_ids = [f"r{i}" for i in range(num_reqs)]
    cp_map = {f"r{i}": cr for i, cr in enumerate(test_case.cp_ranks)}

    batch = MockInputBatch(req_ids, has_apply_permutation=False)
    sched = MockSchedulerOutput(cp_map)

    modified = reorder_batch_to_split_cp_and_normal(batch, sched)

    expected_ids = [f"r{i}" for i in test_case.expected_order]
    assert modified == test_case.expected_modified, (
        f"Expected modified={test_case.expected_modified}, got {modified}"
    )
    assert batch.req_ids == expected_ids, (
        f"Expected {expected_ids}, got {batch.req_ids}"
    )


# ---------------------------------------------------------------------------
# Tests using apply_permutation path
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "test_case", TEST_CASES.values(), ids=TEST_CASES.keys()
)
def test_reorder_apply_permutation_path(test_case: CPReorderTestCase):
    """Exercise the apply_permutation fast path."""
    num_reqs = len(test_case.cp_ranks)
    req_ids = [f"r{i}" for i in range(num_reqs)]
    cp_map = {f"r{i}": cr for i, cr in enumerate(test_case.cp_ranks)}

    batch = MockInputBatch(req_ids, has_apply_permutation=True)
    sched = MockSchedulerOutput(cp_map)

    modified = reorder_batch_to_split_cp_and_normal(batch, sched)

    expected_ids = [f"r{i}" for i in test_case.expected_order]
    assert modified == test_case.expected_modified, (
        f"Expected modified={test_case.expected_modified}, got {modified}"
    )
    assert batch.req_ids == expected_ids, (
        f"Expected {expected_ids}, got {batch.req_ids}"
    )


# ---------------------------------------------------------------------------
# Both paths must produce identical results
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "test_case", TEST_CASES.values(), ids=TEST_CASES.keys()
)
def test_swap_and_permutation_agree(test_case: CPReorderTestCase):
    """Both code paths must yield the exact same final order."""
    num_reqs = len(test_case.cp_ranks)
    cp_map = {f"r{i}": cr for i, cr in enumerate(test_case.cp_ranks)}

    batch_swap = MockInputBatch(
        [f"r{i}" for i in range(num_reqs)], has_apply_permutation=False
    )
    batch_perm = MockInputBatch(
        [f"r{i}" for i in range(num_reqs)], has_apply_permutation=True
    )

    mod_swap = reorder_batch_to_split_cp_and_normal(
        batch_swap, MockSchedulerOutput(cp_map)
    )
    mod_perm = reorder_batch_to_split_cp_and_normal(
        batch_perm, MockSchedulerOutput(cp_map)
    )

    assert mod_swap == mod_perm
    assert batch_swap.req_ids == batch_perm.req_ids


# ---------------------------------------------------------------------------
# Invariant: CP requests always before non-CP after reorder
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "test_case", TEST_CASES.values(), ids=TEST_CASES.keys()
)
def test_cp_before_ncp_invariant(test_case: CPReorderTestCase):
    """After reorder, all CP requests must precede all non-CP requests."""
    num_reqs = len(test_case.cp_ranks)
    req_ids = [f"r{i}" for i in range(num_reqs)]
    cp_map = {f"r{i}": cr for i, cr in enumerate(test_case.cp_ranks)}

    batch = MockInputBatch(req_ids, has_apply_permutation=False)
    sched = MockSchedulerOutput(cp_map)
    reorder_batch_to_split_cp_and_normal(batch, sched)

    seen_ncp = False
    for rid in batch.req_ids:
        is_cp = cp_map[rid] > 1
        if seen_ncp and is_cp:
            pytest.fail(
                f"CP request {rid} found after non-CP; "
                f"final order: {batch.req_ids}"
            )
        if not is_cp:
            seen_ncp = True
