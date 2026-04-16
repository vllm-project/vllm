# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for OMP-aware multiprocessing manager.

Validates that the OMPProcessManager correctly detects CPU topology
and generates appropriate OMP_PLACES for different architectures.
"""

from pathlib import Path

from vllm.utils.ompmultiprocessing import OMPProcessManager, OMPStrategy

MOCK_DIR = Path(__file__).parent


class TestS390XBookTopology:
    """S390X reports a single NUMA node but multiple books.
    Without book-aware grouping, all CPUs land in one OMP_PLACES entry,
    causing threads to migrate across books and ~61% throughput loss.
    """

    def test_groups_by_book_when_single_numa(self):
        """CPUs should be split into per-book OMP_PLACES groups."""
        mock_file = str(MOCK_DIR / "mock_lscpu_s390x.json")
        all_cpus = set(range(0, 50, 2))  # {0,2,4,...,48}
        om = OMPProcessManager(mock=mock_file, affinity=all_cpus)

        # Should have 4 OMP_PLACES (one per book), not 1
        assert len(om.omp_places) == 4, (
            f"Expected 4 OMP_PLACES (one per book), got {len(om.omp_places)}: "
            f"{om.omp_places}"
        )

    def test_book_groups_are_disjoint(self):
        """Each book's CPU set should be non-overlapping."""
        mock_file = str(MOCK_DIR / "mock_lscpu_s390x.json")
        all_cpus = set(range(0, 50, 2))
        om = OMPProcessManager(mock=mock_file, affinity=all_cpus)

        seen: set[int] = set()
        for place in om.omp_places:
            overlap = seen & place["mask"]
            assert not overlap, f"Overlapping CPUs in OMP_PLACES: {overlap}"
            seen |= place["mask"]

    def test_all_cpus_assigned(self):
        """Every available CPU should appear in exactly one OMP_PLACES group."""
        mock_file = str(MOCK_DIR / "mock_lscpu_s390x.json")
        all_cpus = set(range(0, 50, 2))
        om = OMPProcessManager(mock=mock_file, affinity=all_cpus)

        assigned = set()
        for place in om.omp_places:
            assigned |= place["mask"]
        assert assigned == all_cpus, (
            f"Not all CPUs assigned. Missing: {all_cpus - assigned}"
        )


class TestMultiNumaUnchanged:
    """When lscpu already reports multiple NUMA nodes, the grouping
    should use node (not book) — existing behavior preserved.
    """

    def test_groups_by_node(self):
        mock_file = str(MOCK_DIR / "mock_lscpu_x86_multi_numa.json")
        all_cpus = set(range(8))
        om = OMPProcessManager(mock=mock_file, affinity=all_cpus)

        assert len(om.omp_places) == 2, (
            f"Expected 2 OMP_PLACES (one per NUMA node), got {len(om.omp_places)}"
        )

    def test_node_split_is_correct(self):
        mock_file = str(MOCK_DIR / "mock_lscpu_x86_multi_numa.json")
        all_cpus = set(range(8))
        om = OMPProcessManager(mock=mock_file, affinity=all_cpus)

        masks = [place["mask"] for place in om.omp_places]
        assert {0, 1, 2, 3} in masks
        assert {4, 5, 6, 7} in masks


class TestS390XWithStrategy:
    """Verify that OMPStrategy merge/split still works on top of
    book-based grouping.
    """

    def test_merge_books(self):
        """Merging 2 books should yield 2 OMP_PLACES from 4 books."""
        mock_file = str(MOCK_DIR / "mock_lscpu_s390x.json")
        all_cpus = set(range(0, 50, 2))
        strategy = OMPStrategy(merge=2)
        om = OMPProcessManager(mock=mock_file, affinity=all_cpus, strategy=strategy)

        assert len(om.omp_places) == 2, (
            f"Expected 2 OMP_PLACES after merging pairs of books, "
            f"got {len(om.omp_places)}"
        )

    def test_reserve_cores(self):
        """Reserving 1 core per book should reduce compute CPUs."""
        mock_file = str(MOCK_DIR / "mock_lscpu_s390x.json")
        all_cpus = set(range(0, 50, 2))
        strategy = OMPStrategy(reserve=1)
        om = OMPProcessManager(mock=mock_file, affinity=all_cpus, strategy=strategy)

        compute = om.compute_cpus()
        total = om.total_cpus()
        assert compute == total - 4, (
            f"Expected {total - 4} compute CPUs (1 reserved per book), got {compute}"
        )
