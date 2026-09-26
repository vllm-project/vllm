# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for NIXL CPU KV-transfer core reservation vs cgroup cpuset."""

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker import (
    _cpu_kv_transfer_cores,
)


def test_cpu_kv_transfer_cores_intersects_cgroup_cpuset():
    # Host sysfs: two NUMA nodes 0-23 and 24-47; container only 1-4,25-28.
    numa_core_list = [list(range(24)), list(range(24, 48))]
    allowed = frozenset({1, 2, 3, 4, 25, 26, 27, 28})

    assert _cpu_kv_transfer_cores(numa_core_list, allowed) == [4, 28]


def test_cpu_kv_transfer_cores_skips_nodes_with_no_overlap():
    numa_core_list = [list(range(24)), list(range(24, 48))]
    allowed = frozenset({1, 2, 3, 4})

    assert _cpu_kv_transfer_cores(numa_core_list, allowed) == [4]


def test_cpu_kv_transfer_cores_empty_when_disjoint():
    numa_core_list = [list(range(24)), list(range(24, 48))]
    allowed = frozenset({100, 101})

    assert _cpu_kv_transfer_cores(numa_core_list, allowed) == []
