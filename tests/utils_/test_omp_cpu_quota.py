# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Auto OpenMP thread binding must respect the cgroup CPU quota."""

import io
import os
from types import SimpleNamespace

from vllm.platforms import CpuArchEnum
from vllm.utils import ompmultiprocessing, torch_utils
from vllm.utils.cpu_resource_utils import LogicalCPUInfo

CGROUP = "/kubepods/pod1/ctr1"


def _mock_cgroup_v2(monkeypatch, cpu_max: str, num_cpus: int):
    """Pretend to run in a cgroup v2 container whose cpu.max is `cpu_max`
    while the scheduling affinity still covers all `num_cpus` host CPUs."""
    files = {
        "/proc/self/cgroup": f"0::{CGROUP}\n",
        f"/sys/fs/cgroup{CGROUP}/cpu.max": f"{cpu_max}\n",
    }

    def fake_open(path, *args, **kwargs):
        if path not in files:
            raise FileNotFoundError(path)
        return io.StringIO(files[path])

    monkeypatch.setattr(torch_utils, "open", fake_open, raising=False)
    monkeypatch.setattr(torch_utils.sys, "platform", "linux")
    monkeypatch.setattr(
        torch_utils.os,
        "sched_getaffinity",
        lambda pid: set(range(num_cpus)),
        raising=False,
    )


def _make_manager(monkeypatch, num_nodes, cpus_per_node, world_size, dp_size=1):
    monkeypatch.setenv("VLLM_CPU_OMP_THREADS_BIND", "auto")
    monkeypatch.delenv("VLLM_CPU_NUM_OF_RESERVED_CPU", raising=False)
    monkeypatch.delenv("VLLM_CPU_SIM_MULTI_NUMA", raising=False)
    monkeypatch.delenv("LD_PRELOAD", raising=False)
    monkeypatch.setattr(
        ompmultiprocessing,
        "current_platform",
        SimpleNamespace(
            is_cpu=lambda: True,
            # No SMT selection on Arm, so every allowed CPU is a candidate.
            get_cpu_architecture=lambda: CpuArchEnum.ARM,
        ),
    )
    monkeypatch.setattr(
        ompmultiprocessing.cr_utils,
        "get_visible_memory_node",
        lambda: list(range(num_nodes)),
    )
    monkeypatch.setattr(
        ompmultiprocessing.cr_utils,
        "get_allowed_cpu_list",
        lambda: [
            LogicalCPUInfo(
                id=node * cpus_per_node + i,
                physical_core=node * cpus_per_node + i,
                numa_node=node,
            )
            for node in range(num_nodes)
            for i in range(cpus_per_node)
        ],
    )
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            local_world_size=world_size,
            data_parallel_rank_local=0,
            _api_process_count=dp_size,
        ),
        kv_transfer_config=None,
    )
    return ompmultiprocessing.OMPProcessManager(config)


def _num_threads(manager):
    return sum(len(cpus) for cpus in manager.cpu_lists)


def test_autobind_capped_by_cpu_max(monkeypatch):
    # docker run --cpus=4 / limits.cpu: 4 on a 16-CPU host.
    _mock_cgroup_v2(monkeypatch, "400000 100000", num_cpus=16)
    manager = _make_manager(monkeypatch, num_nodes=1, cpus_per_node=16, world_size=1)

    assert manager.cpu_lists == [[0, 1, 2, 3]]
    assert manager.reserved_cpu_list == [15]
    with manager.configure_omp_envs(0, 0):
        assert os.environ["OMP_NUM_THREADS"] == "4"
        assert os.environ["OMP_PLACES"] == "{0,1,2,3}"


def test_autobind_unlimited_quota_unchanged(monkeypatch):
    _mock_cgroup_v2(monkeypatch, "max 100000", num_cpus=16)
    manager = _make_manager(monkeypatch, num_nodes=1, cpus_per_node=16, world_size=1)

    assert manager.cpu_lists == [list(range(15))]
    assert manager.reserved_cpu_list == [15]


def test_autobind_quota_split_evenly_across_ranks(monkeypatch):
    # 5.5 CPUs over two NUMA nodes: 3 + 2 threads.
    _mock_cgroup_v2(monkeypatch, "550000 100000", num_cpus=16)
    manager = _make_manager(monkeypatch, num_nodes=2, cpus_per_node=8, world_size=2)

    assert manager.cpu_lists == [[0, 1, 2], [8, 9]]
    assert manager.reserved_cpu_list == [7, 15]


def test_autobind_keeps_one_thread_per_rank(monkeypatch):
    _mock_cgroup_v2(monkeypatch, "100000 100000", num_cpus=16)
    manager = _make_manager(monkeypatch, num_nodes=2, cpus_per_node=8, world_size=2)

    assert manager.cpu_lists == [[0], [8]]


def test_autobind_quota_shared_by_dp_processes(monkeypatch):
    # Two API processes in the same container split the 8-CPU quota.
    _mock_cgroup_v2(monkeypatch, "800000 100000", num_cpus=16)
    manager = _make_manager(
        monkeypatch, num_nodes=1, cpus_per_node=16, world_size=1, dp_size=2
    )

    assert _num_threads(manager) == 4
