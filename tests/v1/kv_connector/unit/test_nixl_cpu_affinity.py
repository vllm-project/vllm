# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for NUMA topology filtering vs cgroup cpuset."""

from vllm.platforms.cpu import filter_numa_topology_by_affinity


def test_filter_numa_topology_intersects_cgroup_cpuset():
    # Host sysfs: two NUMA nodes 0-23 and 24-47; container only 1-4,25-28.
    numa_core_list = [list(range(24)), list(range(24, 48))]
    allowed = frozenset({1, 2, 3, 4, 25, 26, 27, 28})

    filtered = filter_numa_topology_by_affinity(numa_core_list, allowed)
    assert filtered == [[1, 2, 3, 4], [25, 26, 27, 28]]
    assert [max(cores) for cores in filtered] == [4, 28]


def test_filter_numa_topology_skips_nodes_with_no_overlap():
    numa_core_list = [list(range(24)), list(range(24, 48))]
    allowed = frozenset({1, 2, 3, 4})

    filtered = filter_numa_topology_by_affinity(numa_core_list, allowed)
    assert filtered == [[1, 2, 3, 4]]
    assert [max(cores) for cores in filtered] == [4]


def test_filter_numa_topology_empty_when_disjoint():
    numa_core_list = [list(range(24)), list(range(24, 48))]
    allowed = frozenset({100, 101})

    assert filter_numa_topology_by_affinity(numa_core_list, allowed) == []
