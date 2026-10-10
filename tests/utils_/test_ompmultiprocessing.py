# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from types import SimpleNamespace

import pytest

from vllm.platforms import CpuArchEnum
from vllm.utils import ompmultiprocessing
from vllm.utils.cpu_resource_utils import LogicalCPUInfo


def _make_manager(monkeypatch, memory_nodes, cpu_nodes, world_size=None, arch=None):
    monkeypatch.setenv("VLLM_CPU_OMP_THREADS_BIND", "auto")
    monkeypatch.setenv("VLLM_CPU_NUM_OF_RESERVED_CPU", "1")
    monkeypatch.delenv("VLLM_CPU_SIM_MULTI_NUMA", raising=False)
    monkeypatch.delenv("LD_PRELOAD", raising=False)
    monkeypatch.setattr(
        ompmultiprocessing,
        "current_platform",
        SimpleNamespace(
            is_cpu=lambda: True,
            get_cpu_architecture=lambda: arch or CpuArchEnum.X86,
        ),
    )
    monkeypatch.setattr(
        ompmultiprocessing.cr_utils, "get_visible_memory_node", lambda: memory_nodes
    )
    monkeypatch.setattr(
        ompmultiprocessing.cr_utils,
        "get_allowed_cpu_list",
        lambda: [
            LogicalCPUInfo(node * 8 + i, node * 8 + i, node)
            for node in cpu_nodes
            for i in range(4)
        ],
    )
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            local_world_size=world_size or len(cpu_nodes),
            data_parallel_rank_local=0,
            _api_process_count=1,
        ),
        kv_transfer_config=None,
    )
    return ompmultiprocessing.OMPProcessManager(config)


@pytest.mark.parametrize(
    "memory_nodes,cpu_nodes,expected_cpus,expected_reserved",
    [
        ([0, 1], [0, 1], [[0, 1, 2, 3], [8, 9, 10]], [11]),
        ([0, 1], [1], [[8, 9, 10]], [11]),
        ([0, 1, 2, 3], [1, 3], [[8, 9, 10, 11], [24, 25, 26]], [27]),
        ([3, 2, 1, 0], [1, 3], [[24, 25, 26, 27], [8, 9, 10]], [11]),
    ],
    ids=[
        "unrestricted",
        "cpuset-on-node-one",
        "sparse-nodes",
        "preserve-visible-order",
    ],
)
def test_autobind_uses_nodes_with_allowed_cpus(
    monkeypatch, memory_nodes, cpu_nodes, expected_cpus, expected_reserved
):
    manager = _make_manager(monkeypatch, memory_nodes, cpu_nodes)

    assert manager.cpu_lists == expected_cpus
    assert manager.reserved_cpu_list == expected_reserved
    for rank, cpus in enumerate(expected_cpus):
        with manager.configure_omp_envs(rank, rank):
            assert os.environ["OMP_NUM_THREADS"] == str(len(cpus))
            assert os.environ["OMP_PLACES"] == "{" + ",".join(map(str, cpus)) + "}"


def test_autobind_rejects_insufficient_cpu_nodes(monkeypatch):
    with pytest.raises(AssertionError, match="Not enough allowed NUMA nodes"):
        _make_manager(monkeypatch, [0, 1], [1], world_size=2)


def test_autobind_preserves_s390x_cpu_groups(monkeypatch):
    manager = _make_manager(monkeypatch, [0], [2, 3], arch=CpuArchEnum.S390X)
    assert manager.cpu_lists == [[16, 17, 18, 19], [24, 25, 26]]
    assert manager.reserved_cpu_list == [27]
