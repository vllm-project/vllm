# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cgroup memory readers and SHM allocation preflight checks."""

import os
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
    """Stub `open()` for a fixed set of paths; `None` -> OSError."""
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


def test_worker_memory_nodes_keep_available_assigned_nodes(six_numa_nodes):
    assert cru.get_worker_memory_nodes([2, 3]) == [2, 3]


def test_worker_memory_nodes_raise_when_assigned_nodes_are_missing(
    six_numa_nodes, monkeypatch
):
    """DP=2 x TP=2 on a 2-node host assigns DP rank 1 nodes [2, 3]. The worker
    must stop with an error that names the cause, not an IndexError."""
    monkeypatch.setattr(cru, "get_memory_affinity", lambda: [0, 1])
    assert cru.get_visible_memory_node([2, 3]) == []
    with pytest.raises(ValueError, match="VLLM_CPU_SIM_MULTI_NUMA"):
        cru.get_worker_memory_nodes([2, 3])


def test_worker_memory_nodes_sim_multi_numa_shares_nodes(six_numa_nodes, monkeypatch):
    monkeypatch.setattr(cru, "get_memory_affinity", lambda: [0, 1])
    monkeypatch.setenv("VLLM_CPU_SIM_MULTI_NUMA", "1")
    assert cru.get_worker_memory_nodes([2, 3]) == [0, 1]


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


def _omp_manager_for_topology(
    monkeypatch, cpus, nodes, world_size=1, reserved_cpu=None
):
    """OMPProcessManager against a fake x86 topology.

    `cpus` is a list of LogicalCPUInfo; `nodes` are the visible NUMA nodes.
    `reserved_cpu` sets VLLM_CPU_NUM_OF_RESERVED_CPU explicitly.
    """
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(cru, "get_allowed_cpu_list", lambda: cpus)
    monkeypatch.setattr(cru, "get_memory_affinity", lambda: nodes)
    monkeypatch.setattr(
        omp,
        "current_platform",
        SimpleNamespace(
            is_cpu=lambda: True, get_cpu_architecture=lambda: CpuArchEnum.X86
        ),
    )
    monkeypatch.setenv("VLLM_CPU_OMP_THREADS_BIND", "auto")
    if reserved_cpu is None:
        monkeypatch.delenv("VLLM_CPU_NUM_OF_RESERVED_CPU", raising=False)
    else:
        monkeypatch.setenv("VLLM_CPU_NUM_OF_RESERVED_CPU", reserved_cpu)
    monkeypatch.delenv("VLLM_CPU_SIM_MULTI_NUMA", raising=False)
    monkeypatch.delenv(cru.DEVICE_CONTROL_ENV_VAR, raising=False)
    parallel_config = SimpleNamespace(
        local_world_size=world_size,
        data_parallel_rank_local=None,
        _api_process_count=1,
        assigned_physical_gpu_ids=list(range(world_size)) if world_size > 1 else None,
    )
    return omp.OMPProcessManager(
        SimpleNamespace(parallel_config=parallel_config, kv_transfer_config=None)
    )


def _smt_topology(num_physical_cores, num_nodes=1):
    """2-way SMT x86 topology: logical CPUs 0..2N-1, siblings {2i, 2i+1}."""
    cpus_per_node = 2 * num_physical_cores // num_nodes
    return [
        cru.LogicalCPUInfo(id=i, physical_core=i // 2, numa_node=i // cpus_per_node)
        for i in range(2 * num_physical_cores)
    ]


def test_omp_manager_skips_reservation_when_worker_would_keep_one_core(monkeypatch):
    """2 physical cores: reserving one for the scheduler would leave the
    only worker with a single core, so no CPU is reserved (#69464)."""
    manager = _omp_manager_for_topology(monkeypatch, _smt_topology(2), nodes=[0])
    assert manager.cpu_lists == [[1, 3]]
    assert manager.reserved_cpu_list == []


def test_omp_manager_keeps_reservation_with_enough_cores(monkeypatch):
    """4 physical cores: the worker keeps 3 cores after reserving one."""
    manager = _omp_manager_for_topology(monkeypatch, _smt_topology(4), nodes=[0])
    assert manager.cpu_lists == [[1, 3, 5]]
    assert manager.reserved_cpu_list == [7]


def test_omp_manager_skips_reservation_when_any_rank_would_keep_one_core(monkeypatch):
    """TP=2 with 2 physical cores per NUMA node: each rank would be left
    with a single core, so the reservation is skipped for both ranks."""
    manager = _omp_manager_for_topology(
        monkeypatch, _smt_topology(4, num_nodes=2), nodes=[0, 1], world_size=2
    )
    assert manager.cpu_lists == [[1, 3], [5, 7]]
    assert manager.reserved_cpu_list == []


def test_omp_manager_explicit_reserved_cpu_is_honored_on_small_machine(monkeypatch):
    """An explicit VLLM_CPU_NUM_OF_RESERVED_CPU still reserves a CPU on a
    2-core machine; only the default reservation is skipped."""
    manager = _omp_manager_for_topology(
        monkeypatch, _smt_topology(2), nodes=[0], reserved_cpu="1"
    )
    assert manager.cpu_lists == [[1]]
    assert manager.reserved_cpu_list == [3]


def test_omp_manager_sim_multi_numa_requires_auto_binding(monkeypatch):
    """Explicit binds with VLLM_CPU_SIM_MULTI_NUMA stop with a message, not a
    bare AssertionError."""
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(
        omp,
        "current_platform",
        SimpleNamespace(
            is_cpu=lambda: True, get_cpu_architecture=lambda: CpuArchEnum.X86
        ),
    )
    monkeypatch.setenv("VLLM_CPU_OMP_THREADS_BIND", "0-3|8-11")
    monkeypatch.setenv("VLLM_CPU_SIM_MULTI_NUMA", "1")
    parallel_config = SimpleNamespace(
        local_world_size=2,
        data_parallel_rank_local=0,
        _api_process_count=1,
        assigned_physical_gpu_ids=[0, 1],
    )
    with pytest.raises(AssertionError, match="VLLM_CPU_OMP_THREADS_BIND=auto"):
        omp.OMPProcessManager(
            SimpleNamespace(parallel_config=parallel_config, kv_transfer_config=None)
        )


def _iomp_manager(monkeypatch):
    """OMPProcessManager for one worker bound to CPUs 0-3, with Intel OpenMP."""
    monkeypatch.setattr(
        omp,
        "current_platform",
        SimpleNamespace(
            is_cpu=lambda: True, get_cpu_architecture=lambda: CpuArchEnum.X86
        ),
    )
    monkeypatch.setenv("LD_PRELOAD", "/opt/venv/lib/libiomp5.so")
    monkeypatch.setenv("VLLM_CPU_OMP_THREADS_BIND", "0-3")
    monkeypatch.delenv("VLLM_CPU_SIM_MULTI_NUMA", raising=False)
    parallel_config = SimpleNamespace(
        local_world_size=1,
        data_parallel_rank_local=None,
        _api_process_count=1,
        assigned_physical_gpu_ids=None,
    )
    return omp.OMPProcessManager(
        SimpleNamespace(parallel_config=parallel_config, kv_transfer_config=None)
    )


@pytest.mark.parametrize("unset", [None, ""], ids=["missing", "empty"])
def test_omp_envs_set_kmp_defaults(monkeypatch, unset):
    for key in ("KMP_BLOCKTIME", "KMP_TPAUSE"):
        if unset is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, unset)
    manager = _iomp_manager(monkeypatch)
    with manager.configure_omp_envs(rank=0, local_rank=0):
        assert os.environ["KMP_AFFINITY"] == (
            "granularity=fine,explicit,proclist=[0,1,2,3]"
        )
        assert os.environ["KMP_BLOCKTIME"] == "5"
        assert os.environ["KMP_TPAUSE"] == "0"
    # The previous state (missing or empty) is restored.
    assert os.environ.get("KMP_BLOCKTIME") == unset
    assert os.environ.get("KMP_TPAUSE") == unset


def test_omp_envs_keep_user_kmp_settings(monkeypatch):
    """KMP_BLOCKTIME / KMP_TPAUSE exported by the user reach the worker."""
    monkeypatch.setenv("KMP_BLOCKTIME", "0")
    monkeypatch.setenv("KMP_TPAUSE", "1")
    manager = _iomp_manager(monkeypatch)
    with manager.configure_omp_envs(rank=0, local_rank=0):
        assert os.environ["KMP_BLOCKTIME"] == "0"
        assert os.environ["KMP_TPAUSE"] == "1"
        assert os.environ["OMP_NUM_THREADS"] == "4"
    assert os.environ["KMP_BLOCKTIME"] == "0"
    assert os.environ["KMP_TPAUSE"] == "1"
