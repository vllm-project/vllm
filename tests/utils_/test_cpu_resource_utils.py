# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cgroup memory readers and SHM allocation preflight checks."""

import sys
from io import StringIO
from types import SimpleNamespace
from unittest import mock

import pytest

from vllm.platforms import CpuArchEnum
from vllm.utils import cpu_resource_utils as cru
from vllm.utils import ompmultiprocessing as omp
from vllm.utils.mem_constants import GiB_bytes

_V2_LIMIT_PATH = "/sys/fs/cgroup/memory.max"
_V2_USAGE_PATH = "/sys/fs/cgroup/memory.current"
_V1_LIMIT_PATH = "/sys/fs/cgroup/memory/memory.limit_in_bytes"
_V1_USAGE_PATH = "/sys/fs/cgroup/memory/memory.usage_in_bytes"


@pytest.fixture(autouse=True)
def _clear_cgroup_cache():
    cru.get_cgroup_memory_limit.cache_clear()
    yield
    cru.get_cgroup_memory_limit.cache_clear()


def _stub_files(monkeypatch, files: dict):
    """Stub ``open()`` for a fixed set of paths; ``None`` -> OSError."""
    real_open = open

    def fake_open(path, *args, **kwargs):
        if path in files:
            content = files[path]
            if content is None:
                raise OSError(f"no such file: {path}")
            return StringIO(content)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", fake_open)


def test_cgroup_v2_limit_and_usage(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    _stub_files(
        monkeypatch,
        {
            _V2_LIMIT_PATH: f"{20 * GiB_bytes}\n",
            _V2_USAGE_PATH: f"{5 * GiB_bytes}\n",
        },
    )

    assert cru.get_cgroup_memory_limit() == 20 * GiB_bytes
    assert cru.get_cgroup_memory_usage() == 5 * GiB_bytes


def test_cgroup_v2_unlimited_falls_back_to_v1(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    _stub_files(
        monkeypatch,
        {
            _V2_LIMIT_PATH: "max\n",
            _V1_LIMIT_PATH: f"{8 * GiB_bytes}\n",
            _V1_USAGE_PATH: f"{1 * GiB_bytes}\n",
        },
    )

    assert cru.get_cgroup_memory_limit() == 8 * GiB_bytes
    assert cru.get_cgroup_memory_usage() == 1 * GiB_bytes


def test_cgroup_v1_limit_and_usage(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    _stub_files(
        monkeypatch,
        {
            _V2_LIMIT_PATH: None,
            _V1_LIMIT_PATH: f"{8 * GiB_bytes}\n",
            _V1_USAGE_PATH: f"{2 * GiB_bytes}\n",
        },
    )

    assert cru.get_cgroup_memory_limit() == 8 * GiB_bytes
    assert cru.get_cgroup_memory_usage() == 2 * GiB_bytes


def test_cgroup_v1_unlimited_sentinel_is_ignored(monkeypatch):
    """An unlimited cgroup v1 sentinel must not be treated as a real limit."""
    monkeypatch.setattr(sys, "platform", "linux")
    _stub_files(
        monkeypatch,
        {_V2_LIMIT_PATH: None, _V1_LIMIT_PATH: f"{(1 << 63) - 1}\n"},
    )

    assert cru.get_cgroup_memory_limit() is None
    assert cru.get_cgroup_memory_usage() is None


def test_cgroup_no_limit_files_present(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    _stub_files(monkeypatch, {_V2_LIMIT_PATH: None, _V1_LIMIT_PATH: None})

    assert cru.get_cgroup_memory_limit() is None
    assert cru.get_cgroup_memory_usage() is None


def test_cgroup_skipped_on_non_linux(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    _stub_files(monkeypatch, {_V2_LIMIT_PATH: f"{1 * GiB_bytes}\n"})

    assert cru.get_cgroup_memory_limit() is None
    assert cru.get_cgroup_memory_usage() is None


def test_cgroup_limit_is_cached_until_explicitly_cleared(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    files = {
        _V2_LIMIT_PATH: f"{1 * GiB_bytes}\n",
        _V2_USAGE_PATH: f"{512 << 20}\n",
    }
    _stub_files(monkeypatch, files)

    assert cru.get_cgroup_memory_limit() == 1 * GiB_bytes
    files[_V2_LIMIT_PATH] = f"{2 * GiB_bytes}\n"
    assert cru.get_cgroup_memory_limit() == 1 * GiB_bytes

    cru.get_cgroup_memory_limit.cache_clear()
    assert cru.get_cgroup_memory_limit() == 2 * GiB_bytes


def test_cgroup_usage_is_read_without_cache(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    files = {
        _V2_LIMIT_PATH: f"{1 * GiB_bytes}\n",
        _V2_USAGE_PATH: f"{512 << 20}\n",
    }
    _stub_files(monkeypatch, files)

    assert cru.get_cgroup_memory_usage() == 512 << 20
    files[_V2_USAGE_PATH] = f"{768 << 20}\n"
    assert cru.get_cgroup_memory_usage() == 768 << 20


@pytest.mark.parametrize("cgroup_version", [1, 2])
@pytest.mark.parametrize(
    ("limit_gib", "usage_gib", "expected_total_gib", "expected_available_gib"),
    [
        (4, 3, 4, 1),
        (8, 7, 8, 1),
        (16, 15, 8, 1),
        (16, 17, 8, 0),
        (16, 0, 8, 6),
        (None, None, 8, 6),
    ],
)
def test_numa_memory_respects_cgroup_headroom(
    monkeypatch,
    cgroup_version,
    limit_gib,
    usage_gib,
    expected_total_gib,
    expected_available_gib,
):
    """Cgroup usage bounds available memory even when a NUMA node is smaller."""
    monkeypatch.setattr(sys, "platform", "linux")
    meminfo_path = "/sys/devices/system/node/node0/meminfo"
    monkeypatch.setattr(cru.os.path, "exists", lambda path: path == meminfo_path)
    files = {
        meminfo_path: (
            f"Node 0 MemTotal: {8 * GiB_bytes // 1024} kB\n"
            f"Node 0 MemFree: {6 * GiB_bytes // 1024} kB\n"
            "Node 0 Active(file): 0 kB\n"
            "Node 0 Inactive(file): 0 kB\n"
            "Node 0 SReclaimable: 0 kB\n"
        ),
        _V2_LIMIT_PATH: None,
        _V1_LIMIT_PATH: None,
    }
    if limit_gib is not None:
        limit_path, usage_path = (
            (_V2_LIMIT_PATH, _V2_USAGE_PATH)
            if cgroup_version == 2
            else (_V1_LIMIT_PATH, _V1_USAGE_PATH)
        )
        files[limit_path] = f"{limit_gib * GiB_bytes}\n"
        files[usage_path] = f"{usage_gib * GiB_bytes}\n"
    _stub_files(monkeypatch, files)

    info = cru.get_memory_node_info(0)
    assert info.total_memory == expected_total_gib * GiB_bytes
    assert info.available_memory == expected_available_gib * GiB_bytes


def test_check_cgroup_memory_available_warns_on_low_headroom(monkeypatch):
    monkeypatch.setattr(cru, "get_cgroup_memory_limit", lambda: 1 << 30)
    monkeypatch.setattr(cru, "get_cgroup_memory_usage", lambda: 512 << 20)

    with (
        mock.patch.object(cru.logger, "debug") as log_debug,
        mock.patch.object(cru.logger, "warning") as log_warning,
    ):
        cru.check_cgroup_memory_available(600 << 20, "mmap")

    log_debug.assert_not_called()
    log_warning.assert_called_once()
    assert "current headroom is below" in log_warning.call_args.args[-1]


def test_check_cgroup_memory_available_logs_success_at_debug(monkeypatch):
    monkeypatch.setattr(cru, "get_cgroup_memory_limit", lambda: 1 << 30)
    monkeypatch.setattr(cru, "get_cgroup_memory_usage", lambda: 512 << 20)

    with (
        mock.patch.object(cru.logger, "debug") as log_debug,
        mock.patch.object(cru.logger, "warning") as log_warning,
    ):
        cru.check_cgroup_memory_available(256 << 20, "mmap")

    log_debug.assert_called_once()
    log_warning.assert_not_called()


@pytest.mark.parametrize(
    ("limit", "usage"),
    [(None, 512 << 20), (1 << 30, None)],
)
def test_check_cgroup_memory_available_skips_without_snapshot(
    monkeypatch, limit, usage
):
    monkeypatch.setattr(cru, "get_cgroup_memory_limit", lambda: limit)
    monkeypatch.setattr(cru, "get_cgroup_memory_usage", lambda: usage)

    with (
        mock.patch.object(cru.logger, "debug") as log_debug,
        mock.patch.object(cru.logger, "warning") as log_warning,
    ):
        cru.check_cgroup_memory_available(1 << 60, "mmap")

    log_debug.assert_not_called()
    log_warning.assert_not_called()


@pytest.fixture
def six_numa_nodes(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(cru, "get_memory_affinity", lambda: [0, 1, 2, 3, 4, 5])
    monkeypatch.delenv(cru.DEVICE_CONTROL_ENV_VAR, raising=False)
    monkeypatch.delenv("VLLM_CPU_SIM_MULTI_NUMA", raising=False)


def test_visible_memory_node_uses_env_var(six_numa_nodes, monkeypatch):
    monkeypatch.setenv(cru.DEVICE_CONTROL_ENV_VAR, "5,3")
    assert cru.get_visible_memory_node() == [5, 3]


def test_visible_memory_node_assigned_ids_select_dp_rank_nodes(six_numa_nodes):
    """Each DP rank sees only the nodes assigned to it."""
    assert cru.get_visible_memory_node([0, 1]) == [0, 1]
    assert cru.get_visible_memory_node([2, 3]) == [2, 3]


def test_visible_memory_node_assigned_ids_override_env_var(six_numa_nodes, monkeypatch):
    monkeypatch.setenv(cru.DEVICE_CONTROL_ENV_VAR, "5,3")
    assert cru.get_visible_memory_node([2]) == [2]


def test_visible_memory_node_assigned_ids_are_limited_to_allowed_nodes(
    six_numa_nodes, monkeypatch
):
    monkeypatch.setattr(cru, "get_memory_affinity", lambda: [0, 1])
    assert cru.get_visible_memory_node([1, 4]) == [1]


def test_visible_memory_node_sim_multi_numa_ignores_assigned_ids(
    six_numa_nodes, monkeypatch
):
    monkeypatch.setenv("VLLM_CPU_SIM_MULTI_NUMA", "1")
    assert cru.get_visible_memory_node([2]) == [0, 1, 2, 3, 4, 5]


def test_omp_manager_binds_each_dp_rank_to_its_assigned_node(monkeypatch):
    """DP ranks with different assigned NUMA nodes get disjoint cores (#47328)."""
    # 2 nodes x 4 physical cores x 2 logical CPUs
    cpus = [
        cru.LogicalCPUInfo(id=i, physical_core=i // 2, numa_node=i // 8)
        for i in range(16)
    ]
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(cru, "get_allowed_cpu_list", lambda: cpus)
    monkeypatch.setattr(cru, "get_memory_affinity", lambda: [0, 1])
    monkeypatch.setattr(
        omp,
        "current_platform",
        SimpleNamespace(
            is_cpu=lambda: True, get_cpu_architecture=lambda: CpuArchEnum.X86
        ),
    )
    monkeypatch.setenv("VLLM_CPU_OMP_THREADS_BIND", "auto")
    monkeypatch.delenv("VLLM_CPU_SIM_MULTI_NUMA", raising=False)
    monkeypatch.delenv(cru.DEVICE_CONTROL_ENV_VAR, raising=False)

    def cpu_ids(dp_rank: int) -> set[int]:
        parallel_config = SimpleNamespace(
            local_world_size=1,
            data_parallel_rank_local=dp_rank,
            _api_process_count=2,
            assigned_physical_gpu_ids=[dp_rank],
        )
        manager = omp.OMPProcessManager(
            SimpleNamespace(parallel_config=parallel_config, kv_transfer_config=None)
        )
        return set(manager.cpu_lists[0])

    rank0, rank1 = cpu_ids(0), cpu_ids(1)
    assert rank0 and rank1
    assert rank0 <= set(range(8))
    assert rank1 <= set(range(8, 16))
