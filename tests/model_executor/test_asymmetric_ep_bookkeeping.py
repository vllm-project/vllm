"""Tests for the asymmetric expert-parallelism bookkeeping fix.

deepseek_v2.py and qwen3_moe.py computed::

    n_local_physical_experts = n_physical_experts // ep_size
    physical_expert_start = ep_rank * n_local_physical_experts

This is wrong when ``n_physical_experts % ep_size != 0``: every rank uses the
floor count, so the bookkeeping disagrees with the actual weight distribution
that ``determine_expert_map()`` in expert_map_manager.py uses.

The fix applies the same floor/ceil formula that expert_map_manager.py uses:

    base    = n_physical_experts // ep_size
    rem     = n_physical_experts % ep_size
    n_local = base + (1 if ep_rank < rem else 0)
    start   = ep_rank * base + min(ep_rank, rem)

These tests verify the corrected formula in pure Python (no GPU required) and
check consistency against expert_map_manager.py's reference implementation.

Repro: 5 experts, 2 GPUs → old formula gives both ranks n_local=2 (sum=4≠5).
"""
import pytest


def _bookkeeping_fixed(n_physical_experts: int, ep_size: int,
                       ep_rank: int) -> tuple[int, int, int]:
    """Fixed formula (matches expert_map_manager.py)."""
    base = n_physical_experts // ep_size
    rem = n_physical_experts % ep_size
    n_local = base + (1 if ep_rank < rem else 0)
    start = ep_rank * base + min(ep_rank, rem)
    end = start + n_local
    return n_local, start, end


def _bookkeeping_buggy(n_physical_experts: int, ep_size: int,
                       ep_rank: int) -> tuple[int, int, int]:
    """Original buggy formula."""
    n_local = n_physical_experts // ep_size
    start = ep_rank * n_local
    end = start + n_local
    return n_local, start, end


def _expert_map_manager_formula(n_experts: int, ep_size: int,
                                ep_rank: int) -> tuple[int, int]:
    """Inline the linear-strategy formula from expert_map_manager.py."""
    base = n_experts // ep_size
    rem = n_experts % ep_size
    n_local = base + 1 if ep_rank < rem else base
    start = ep_rank * base + min(ep_rank, rem)
    return n_local, start


# ---------------------------------------------------------------------------
# Repro case: 5 experts on 2 GPUs
# ---------------------------------------------------------------------------

def test_repro_5_experts_2_gpus_old_formula_is_wrong():
    """Old formula: both ranks claim floor(5/2)=2 experts → total 4 ≠ 5."""
    n_local_0, start_0, _ = _bookkeeping_buggy(5, 2, ep_rank=0)
    n_local_1, start_1, _ = _bookkeeping_buggy(5, 2, ep_rank=1)
    assert n_local_0 + n_local_1 != 5, "old formula sum should NOT equal 5"


def test_repro_5_experts_2_gpus_fixed():
    """Fixed formula: rank 0 gets 3 experts, rank 1 gets 2 → total 5."""
    n_local_0, start_0, end_0 = _bookkeeping_fixed(5, 2, ep_rank=0)
    n_local_1, start_1, end_1 = _bookkeeping_fixed(5, 2, ep_rank=1)
    assert n_local_0 == 3
    assert n_local_1 == 2
    assert n_local_0 + n_local_1 == 5
    assert start_0 == 0
    assert start_1 == 3
    assert end_0 == 3
    assert end_1 == 5


# ---------------------------------------------------------------------------
# Parametrised correctness checks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_experts,ep_size", [
    # Asymmetric: remainder != 0
    (5, 2),     # repro case from the issue
    (128, 5),   # Qwen3-30B-A3B on 5 GPUs
    (128, 6),   # Qwen3-30B-A3B on 6 GPUs
    (256, 5),   # MiniMax M2.7 / DeepSeek V3 on 5 GPUs
    (256, 6),   # DeepSeek V3 on 6 GPUs
    (256, 3),   # 3 GPUs
    (7, 3),     # small prime
    # Symmetric: remainder == 0 — must not regress
    (128, 4),
    (128, 8),
    (256, 4),
    (256, 8),
    (64, 4),
    (1, 1),
])
def test_fixed_formula_counts_sum_to_total(n_experts, ep_size):
    """All ranks' n_local counts must sum to n_experts."""
    total = sum(
        _bookkeeping_fixed(n_experts, ep_size, r)[0]
        for r in range(ep_size)
    )
    assert total == n_experts, (
        f"n_experts={n_experts} ep_size={ep_size}: sum of local counts "
        f"= {total}, expected {n_experts}"
    )


@pytest.mark.parametrize("n_experts,ep_size", [
    (5, 2), (128, 5), (128, 6), (256, 5), (256, 6), (256, 3), (7, 3),
    (128, 4), (128, 8), (256, 4), (256, 8), (64, 4), (1, 1),
])
def test_fixed_formula_ranges_partition_experts(n_experts, ep_size):
    """[start, end) ranges for all ranks are contiguous and cover [0, n_experts)."""
    ranges = [
        (_bookkeeping_fixed(n_experts, ep_size, r)[1],
         _bookkeeping_fixed(n_experts, ep_size, r)[2])
        for r in range(ep_size)
    ]
    expected_start = 0
    for start, end in ranges:
        assert start == expected_start, (
            f"gap/overlap: expected start={expected_start}, got {start}"
        )
        expected_start = end
    assert expected_start == n_experts


@pytest.mark.parametrize("n_experts,ep_size", [
    (5, 2), (128, 5), (128, 6), (256, 5), (256, 6), (256, 3), (7, 3),
    (128, 4), (128, 8), (256, 4), (256, 8), (64, 4), (1, 1),
])
def test_fixed_formula_matches_expert_map_manager(n_experts, ep_size):
    """n_local and start must agree with expert_map_manager.py's formula."""
    for r in range(ep_size):
        n_local_fixed, start_fixed, _ = _bookkeeping_fixed(n_experts, ep_size, r)
        n_local_ref, start_ref = _expert_map_manager_formula(n_experts, ep_size, r)
        assert n_local_fixed == n_local_ref, (
            f"n_experts={n_experts} ep_size={ep_size} rank={r}: "
            f"fixed n_local={n_local_fixed} vs ref={n_local_ref}"
        )
        assert start_fixed == start_ref, (
            f"n_experts={n_experts} ep_size={ep_size} rank={r}: "
            f"fixed start={start_fixed} vs ref={start_ref}"
        )


@pytest.mark.parametrize("n_experts,ep_size", [
    (128, 5), (128, 6), (256, 5), (256, 6), (5, 2),
])
def test_buggy_formula_wrong_for_asymmetric(n_experts, ep_size):
    """Document that the old formula is wrong for non-divisor ep_size."""
    total = sum(
        _bookkeeping_buggy(n_experts, ep_size, r)[0]
        for r in range(ep_size)
    )
    assert total != n_experts, (
        f"Expected old formula to be wrong for n_experts={n_experts} "
        f"ep_size={ep_size}, but sum={total} == n_experts"
    )


@pytest.mark.parametrize("n_experts,ep_size", [
    (128, 4), (128, 8), (256, 4), (256, 8),
])
def test_symmetric_case_unchanged(n_experts, ep_size):
    """For symmetric EP (remainder==0), both formulae give identical results."""
    for r in range(ep_size):
        fixed = _bookkeeping_fixed(n_experts, ep_size, r)
        buggy = _bookkeeping_buggy(n_experts, ep_size, r)
        assert fixed == buggy, (
            f"n_experts={n_experts} ep_size={ep_size} rank={r}: "
            f"symmetric case should be unchanged; fixed={fixed} buggy={buggy}"
        )
