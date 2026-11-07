# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest


def _free_hugepages(page: int) -> int:
    path = f"/sys/kernel/mm/hugepages/hugepages-{page >> 10}kB/free_hugepages"
    try:
        with open(path) as f:
            return int(f.read())
    except OSError:
        return -1


@pytest.fixture
def free_hugepages():
    """Free pages in the host's HugeTLB pool of a size; -1 if it has none."""
    return _free_hugepages


@pytest.fixture
def hugepage_with_free():
    """Return a HugeTLB page size with at least ``n`` free pages, else skip."""

    def pick(n: int) -> int:
        pages = [p for p in (2 << 20, 1 << 30) if _free_hugepages(p) >= n]
        if not pages:
            pytest.skip(f"needs {n} free hugepages of some size")
        return pages[0]

    return pick
