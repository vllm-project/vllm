# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.distributed.eplb.eplb_state import (
    apply_transfer_cap,
    find_cycles,
)


def test_find_cycles_no_changes():
    old = torch.tensor([0, 1, 2, 3, 4])
    new = torch.tensor([0, 1, 2, 3, 4])
    cycles = find_cycles(old, new)
    assert cycles == []


def test_find_cycles_single_swap():
    # Swap positions 1 and 3
    old = torch.tensor([0, 1, 2, 3, 4])
    new = torch.tensor([0, 3, 2, 1, 4])
    cycles = find_cycles(old, new)
    assert len(cycles) == 1
    assert set(cycles[0]) == {1, 3}


def test_find_cycles_multiple_swaps():
    # Swap positions (0, 1) and (2, 3)
    old = torch.tensor([0, 1, 2, 3, 4])
    new = torch.tensor([1, 0, 3, 2, 4])
    cycles = find_cycles(old, new)
    assert len(cycles) == 2
    cycle_sets = [set(c) for c in cycles]
    assert {0, 1} in cycle_sets
    assert {2, 3} in cycle_sets


def test_find_cycles_three_cycle():
    old = torch.tensor([3, 0, 2, 1, 4])
    new = torch.tensor([1, 0, 3, 2, 4])
    # Cycle 0->3->2->0
    cycles = find_cycles(old, new)
    assert len(cycles) == 1
    assert set(cycles[0]) == {0, 2, 3}


def test_find_cycles_four_cycle():
    old = torch.tensor([0, 1, 2, 3, 4])
    new = torch.tensor([1, 2, 3, 0, 4])
    # Cycle 0->1->2->3->0
    cycles = find_cycles(old, new)
    assert len(cycles) == 1
    assert set(cycles[0]) == {0, 1, 2, 3}


def test_find_cycles_mixed():
    old = torch.tensor([0, 1, 2, 3, 4, 5])
    new = torch.tensor([1, 0, 3, 4, 2, 5])
    # Cycle 1: 0->1
    # Cycle 2: 2->3->4->2
    cycles = find_cycles(old, new)
    assert len(cycles) == 2
    cycle_sets = [set(c) for c in cycles]
    assert {0, 1} in cycle_sets
    assert {2, 3, 4} in cycle_sets


def test_apply_transfer_cap_under_limit():
    old = torch.tensor([0, 1, 2, 3, 4])
    new = torch.tensor([1, 0, 2, 3, 4])
    apply_transfer_cap(old, new, max_transfers=4)
    expected = torch.tensor([1, 0, 2, 3, 4])
    assert torch.equal(new, expected)


def test_apply_transfer_cap_exact_limit():
    old = torch.tensor([0, 1, 2, 3, 4])
    new = torch.tensor([1, 0, 2, 3, 4])
    apply_transfer_cap(old, new, max_transfers=2)
    expected = torch.tensor([1, 0, 2, 3, 4])
    assert torch.equal(new, expected)


def test_apply_transfer_cap_undo_one_swap():
    old = torch.tensor([0, 1, 2, 3, 4])
    new = torch.tensor([1, 0, 3, 2, 4])
    apply_transfer_cap(old, new, max_transfers=2)

    # Should undo one cycle (0->1)
    num_transfers = (old != new).sum().item()
    assert num_transfers == 2
    assert sorted(new.tolist()) == [0, 1, 2, 3, 4]


def test_apply_transfer_cap_undo_three_cycle():
    old = torch.tensor([3, 0, 2, 1, 4])
    new = torch.tensor([1, 0, 3, 2, 4])
    apply_transfer_cap(old, new, max_transfers=1)

    # Should undo all cycles
    assert torch.equal(new, old)

    """Test undoing cycles when over cap."""
    old = torch.tensor([0, 1, 2, 3, 4, 5])
    new = torch.tensor([1, 0, 3, 4, 2, 5])
    # 2-cycle at positions 0,1 (2 transfers)
    # 3-cycle at positions 2,3,4 (3 transfers)
    # Total: 5 transfers

    apply_transfer_cap(old, new, max_transfers=3)

    # Will undo the 2-cycle first (smallest), leaving 3 transfers
    num_transfers = (old != new).sum().item()
    assert num_transfers == 3
    # Positions 0 and 1 should be reverted
    assert new[0].item() == 0
    assert new[1].item() == 1
    # Positions 2, 3, 4 should still have the 3-cycle
    assert new[2].item() == 3
    assert new[3].item() == 4
    assert new[4].item() == 2


def test_apply_transfer_cap_all_cycles_same_size():
    old = torch.tensor([0, 1, 2, 3, 4, 5])
    new = torch.tensor([1, 0, 3, 2, 5, 4])
    apply_transfer_cap(old, new, max_transfers=4)

    # Should keep 2 cycles (4 transfers), undo 1 cycle
    num_transfers = (old != new).sum().item()
    assert num_transfers == 4
    assert sorted(new.tolist()) == [0, 1, 2, 3, 4, 5]


def test_apply_transfer_cap_maintains_permutation():
    old = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
    new = torch.tensor([1, 0, 3, 2, 5, 4, 7, 6])
    apply_transfer_cap(old, new, max_transfers=5)

    # Should have at most 6 transfers (3 swaps, rounded up)
    num_transfers = (old != new).sum().item()
    assert num_transfers <= 6

    # Must be valid permutation
    assert sorted(new.tolist()) == list(range(8))


def test_apply_transfer_cap_single_large_cycle():
    # Cycle 0->1->2->3->4->0
    old = torch.tensor([0, 1, 2, 3, 4])
    new = torch.tensor([4, 0, 1, 2, 3])

    apply_transfer_cap(old, new, max_transfers=3)

    assert torch.equal(new, old)
    num_transfers = (old != new).sum().item()
    assert num_transfers == 0
