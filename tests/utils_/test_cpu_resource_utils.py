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


_PROC_SELF_CGROUP = "/stub/proc/self/cgroup"


@pytest.fixture(autouse=True)
def _clear_cgroup_cache():
    # Resolve the process cgroup from a stub path so the real /proc/self/cgroup
    # of the test host never leaks into these tests. Set directly rather than
    # through monkeypatch so this fixture is torn down after the tests' own
    # monkeypatches are undone.
    original = getattr(cru, "_PROC_SELF_CGROUP", None)
    cru._PROC_SELF_CGROUP = _PROC_SELF_CGROUP
    cru.get_cgroup_memory_limit.cache_clear()
    yield
    cru._PROC_SELF_CGROUP = original
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


def test_cgroup_v2_limit_on_parent_cgroup_is_found(monkeypatch):
    """Kubernetes sets the pod limit on the parent cgroup; the container is a
    child whose own memory.max is `max`. Limit and usage come from the parent."""
    monkeypatch.setenv("VLLM_RESOLVE_PROCESS_CGROUP", "1")
    monkeypatch.setattr(sys, "platform", "linux")
    _stub_files(
        monkeypatch,
        {
            _PROC_SELF_CGROUP: "0::/kubepods/pod1/ctr\n",
            _V2_LIMIT_PATH: "max\n",
            _V2_USAGE_PATH: f"{10 * GiB_bytes}\n",
            "/sys/fs/cgroup/kubepods/memory.max": "max\n",
            "/sys/fs/cgroup/kubepods/memory.current": f"{9 * GiB_bytes}\n",
            "/sys/fs/cgroup/kubepods/pod1/memory.max": f"{8 * GiB_bytes}\n",
            "/sys/fs/cgroup/kubepods/pod1/memory.current": f"{2 * GiB_bytes}\n",
            "/sys/fs/cgroup/kubepods/pod1/ctr/memory.max": "max\n",
            "/sys/fs/cgroup/kubepods/pod1/ctr/memory.current": f"{1 * GiB_bytes}\n",
        },
    )

    assert cru.get_cgroup_memory_limit() == 8 * GiB_bytes
    assert cru.get_cgroup_memory_usage() == 2 * GiB_bytes


def test_cgroup_v2_smallest_limit_in_hierarchy_wins(monkeypatch):
    monkeypatch.setenv("VLLM_RESOLVE_PROCESS_CGROUP", "1")
    monkeypatch.setattr(sys, "platform", "linux")
    _stub_files(
        monkeypatch,
        {
            _PROC_SELF_CGROUP: "0::/pod1/ctr\n",
            _V2_LIMIT_PATH: f"{20 * GiB_bytes}\n",
            _V2_USAGE_PATH: f"{15 * GiB_bytes}\n",
            "/sys/fs/cgroup/pod1/memory.max": f"{8 * GiB_bytes}\n",
            "/sys/fs/cgroup/pod1/memory.current": f"{3 * GiB_bytes}\n",
            "/sys/fs/cgroup/pod1/ctr/memory.max": f"{16 * GiB_bytes}\n",
            "/sys/fs/cgroup/pod1/ctr/memory.current": f"{1 * GiB_bytes}\n",
        },
    )

    assert cru.get_cgroup_memory_limit() == 8 * GiB_bytes
    assert cru.get_cgroup_memory_usage() == 3 * GiB_bytes


def test_cgroup_v2_bare_host_limit_on_scope(monkeypatch):
    """On a bare host the root has no memory.max at all; a limit on the
    process's systemd scope (or Slurm step) must still be found."""
    monkeypatch.setenv("VLLM_RESOLVE_PROCESS_CGROUP", "1")
    monkeypatch.setattr(sys, "platform", "linux")
    _stub_files(
        monkeypatch,
        {
            _PROC_SELF_CGROUP: "0::/user.slice/run-1.scope\n",
            _V2_LIMIT_PATH: None,
            _V2_USAGE_PATH: None,
            "/sys/fs/cgroup/user.slice/memory.max": "max\n",
            "/sys/fs/cgroup/user.slice/memory.current": f"{6 * GiB_bytes}\n",
            "/sys/fs/cgroup/user.slice/run-1.scope/memory.max": f"{1 * GiB_bytes}\n",
            "/sys/fs/cgroup/user.slice/run-1.scope/memory.current": f"{100 << 20}\n",
        },
    )

    assert cru.get_cgroup_memory_limit() == 1 * GiB_bytes
    assert cru.get_cgroup_memory_usage() == 100 << 20


def test_cgroup_v2_path_not_visible_in_mount_uses_root(monkeypatch):
    """Host cgroup namespace: /proc/self/cgroup names a path that does not
    exist inside the container's mount, so the root is used as before."""
    monkeypatch.setenv("VLLM_RESOLVE_PROCESS_CGROUP", "1")
    monkeypatch.setattr(sys, "platform", "linux")
    _stub_files(
        monkeypatch,
        {
            _PROC_SELF_CGROUP: "0::/docker/0123abcd\n",
            _V2_LIMIT_PATH: f"{4 * GiB_bytes}\n",
            _V2_USAGE_PATH: f"{1 * GiB_bytes}\n",
            "/sys/fs/cgroup/docker/0123abcd/memory.max": None,
            "/sys/fs/cgroup/docker/0123abcd/memory.current": None,
        },
    )

    assert cru.get_cgroup_memory_limit() == 4 * GiB_bytes
    assert cru.get_cgroup_memory_usage() == 1 * GiB_bytes


def test_cgroup_v1_limit_on_parent_cgroup_is_found(monkeypatch):
    monkeypatch.setenv("VLLM_RESOLVE_PROCESS_CGROUP", "1")
    monkeypatch.setattr(sys, "platform", "linux")
    v1 = "/sys/fs/cgroup/memory"
    _stub_files(
        monkeypatch,
        {
            _PROC_SELF_CGROUP: "12:memory:/kubepods/pod1\n"
            "3:cpu,cpuacct:/kubepods/pod1\n",
            _V2_LIMIT_PATH: None,
            _V1_LIMIT_PATH: f"{1 << 63}\n",
            _V1_USAGE_PATH: f"{10 * GiB_bytes}\n",
            f"{v1}/kubepods/memory.limit_in_bytes": f"{4 * GiB_bytes}\n",
            f"{v1}/kubepods/memory.usage_in_bytes": f"{3 * GiB_bytes}\n",
            f"{v1}/kubepods/pod1/memory.limit_in_bytes": f"{1 << 63}\n",
            f"{v1}/kubepods/pod1/memory.usage_in_bytes": f"{1 * GiB_bytes}\n",
        },
    )

    assert cru.get_cgroup_memory_limit() == 4 * GiB_bytes
    assert cru.get_cgroup_memory_usage() == 3 * GiB_bytes


_PARENT_LIMIT_FILES = {
    _PROC_SELF_CGROUP: "0::/pod1/ctr\n",
    _V2_LIMIT_PATH: "max\n",
    _V2_USAGE_PATH: f"{10 * GiB_bytes}\n",
    "/sys/fs/cgroup/pod1/memory.max": f"{8 * GiB_bytes}\n",
    "/sys/fs/cgroup/pod1/memory.current": f"{2 * GiB_bytes}\n",
}


@pytest.mark.parametrize(
    ("default", "env", "expected_limit"),
    [
        # Unset: the release default decides. Off keeps the old root-only read.
        (False, None, None),
        (True, None, 8 * GiB_bytes),
        # An explicit value overrides the default either way.
        (False, "1", 8 * GiB_bytes),
        (True, "0", None),
    ],
)
def test_cgroup_resolution_default_and_override(
    monkeypatch, default, env, expected_limit
):
    """RESOLVE_PROCESS_CGROUP_BY_DEFAULT is the one line that flips the
    behavior; VLLM_RESOLVE_PROCESS_CGROUP=1/0 overrides it."""
    monkeypatch.setattr(cru, "RESOLVE_PROCESS_CGROUP_BY_DEFAULT", default)
    if env is None:
        monkeypatch.delenv("VLLM_RESOLVE_PROCESS_CGROUP", raising=False)
    else:
        monkeypatch.setenv("VLLM_RESOLVE_PROCESS_CGROUP", env)
    monkeypatch.setattr(sys, "platform", "linux")
    _stub_files(monkeypatch, _PARENT_LIMIT_FILES)

    assert cru.get_cgroup_memory_limit() == expected_limit
    assert cru.get_cgroup_memory_usage() == (
        None if expected_limit is None else 2 * GiB_bytes
    )


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
