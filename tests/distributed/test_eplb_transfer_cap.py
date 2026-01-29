# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.distributed.eplb.eplb_state import (
    apply_transfer_cap,
    find_cycles,
)


def test_find_cycles_no_changes():
    """Test when old and new are identical."""
    old = torch.tensor([0, 1, 2, 3, 4])
    new = torch.tensor([0, 1, 2, 3, 4])
    cycles = find_cycles(old, new)
    assert cycles == []


def test_find_cycles_single_swap():
    """Test a 2-cycle (swap)."""
    old = torch.tensor([0, 1, 2, 3, 4])
    new = torch.tensor([0, 3, 2, 1, 4])  # Swap experts at positions 1 and 3
    cycles = find_cycles(old, new)
    assert len(cycles) == 1
    assert set(cycles[0]) == {1, 3}


def test_find_cycles_multiple_swaps():
    """Test multiple independent 2-cycles."""
    old = torch.tensor([0, 1, 2, 3, 4])
    new = torch.tensor([1, 0, 3, 2, 4])  # Swap at 0↔1 and 2↔3
    cycles = find_cycles(old, new)
    assert len(cycles) == 2
    cycle_sets = [set(c) for c in cycles]
    assert {0, 1} in cycle_sets
    assert {2, 3} in cycle_sets


def test_find_cycles_three_cycle():
    """Test a 3-cycle."""
    old = torch.tensor([3, 0, 2, 1, 4])
    new = torch.tensor([1, 0, 3, 2, 4])
    # Position 0: expert 3 → expert 1
    # Position 2: expert 2 → expert 3
    # Position 3: expert 1 → expert 2
    # This is a 3-cycle: 0→3→2→0
    cycles = find_cycles(old, new)
    assert len(cycles) == 1
    assert set(cycles[0]) == {0, 2, 3}


def test_find_cycles_four_cycle():
    """Test a 4-cycle."""
    old = torch.tensor([0, 1, 2, 3, 4])
    new = torch.tensor([1, 2, 3, 0, 4])
    # Position 0: 0→1, Position 1: 1→2, Position 2: 2→3, Position 3: 3→0
    cycles = find_cycles(old, new)
    assert len(cycles) == 1
    assert set(cycles[0]) == {0, 1, 2, 3}


def test_find_cycles_mixed():
    """Test mixture of 2-cycle and 3-cycle."""
    old = torch.tensor([0, 1, 2, 3, 4, 5])
    new = torch.tensor([1, 0, 3, 4, 2, 5])
    # Positions 0↔1 form a 2-cycle
    # Positions 2→3→4→2 form a 3-cycle
    # Position 5 unchanged
    cycles = find_cycles(old, new)
    assert len(cycles) == 2
    cycle_sets = [set(c) for c in cycles]
    assert {0, 1} in cycle_sets
    assert {2, 3, 4} in cycle_sets


def test_apply_transfer_cap_under_limit():
    """Test when transfers are already under the cap."""
    old = torch.tensor([0, 1, 2, 3, 4])
    new = torch.tensor([1, 0, 2, 3, 4])  # 2 transfers (1 swap)
    apply_transfer_cap(old, new, max_transfers=4)
    expected = torch.tensor([1, 0, 2, 3, 4])
    assert torch.equal(new, expected)


def test_apply_transfer_cap_exact_limit():
    """Test when transfers exactly match the cap."""
    old = torch.tensor([0, 1, 2, 3, 4])
    new = torch.tensor([1, 0, 2, 3, 4])  # 2 transfers
    apply_transfer_cap(old, new, max_transfers=2)
    expected = torch.tensor([1, 0, 2, 3, 4])
    assert torch.equal(new, expected)


def test_apply_transfer_cap_undo_one_swap():
    """Test undoing one 2-cycle."""
    old = torch.tensor([0, 1, 2, 3, 4])
    new = torch.tensor([1, 0, 3, 2, 4])  # Two 2-cycles (4 transfers)
    apply_transfer_cap(old, new, max_transfers=2)

    # Should undo one cycle (the smaller indices one: 0↔1)
    num_transfers = (old != new).sum().item()
    assert num_transfers == 2
    assert sorted(new.tolist()) == [0, 1, 2, 3, 4]


def test_apply_transfer_cap_undo_three_cycle():
    """Test undoing a 3-cycle."""
    old = torch.tensor([3, 0, 2, 1, 4])
    new = torch.tensor([1, 0, 3, 2, 4])  # 3-cycle (3 transfers)
    apply_transfer_cap(old, new, max_transfers=1)

    # Should undo the entire 3-cycle (can't partially undo)
    assert torch.equal(new, old)


def test_apply_transfer_cap_keeps_smallest_cycles():
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
    """Test when all cycles are the same size."""
    old = torch.tensor([0, 1, 2, 3, 4, 5])
    new = torch.tensor([1, 0, 3, 2, 5, 4])  # Three 2-cycles
    apply_transfer_cap(old, new, max_transfers=4)

    # Should keep 2 cycles (4 transfers), undo 1 cycle
    num_transfers = (old != new).sum().item()
    assert num_transfers == 4
    assert sorted(new.tolist()) == [0, 1, 2, 3, 4, 5]


def test_apply_transfer_cap_maintains_permutation():
    """Test that result is always a valid permutation."""
    old = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
    new = torch.tensor([1, 0, 3, 2, 5, 4, 7, 6])  # 4 swaps, 8 transfers
    apply_transfer_cap(old, new, max_transfers=5)

    # Should have at most 6 transfers (3 swaps, rounded up)
    num_transfers = (old != new).sum().item()
    assert num_transfers <= 6

    # Must be valid permutation
    assert sorted(new.tolist()) == list(range(8))


def test_apply_transfer_cap_single_large_cycle():
    """Test worst case: single cycle that includes all expert transfers."""
    old = torch.tensor([0, 1, 2, 3, 4])
    new = torch.tensor([4, 0, 1, 2, 3])  # 5-cycle

    apply_transfer_cap(old, new, max_transfers=3)

    # Can't partially undo a cycle, so entire cycle is undone
    # Result: 0 transfers instead of the desired 3
    assert torch.equal(new, old)
    num_transfers = (old != new).sum().item()
    assert num_transfers == 0
