# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Regression tests for vllm.utils.cpu_resource_utils.get_visible_memory_node().

These guard against a regression where local (non-Ray) multiprocessing DP
launches on CPU all collapsed onto the same NUMA node: the per-DP-rank
device shard computed by CoreEngineProcManager was stored on
parallel_config.assigned_physical_gpu_ids, but get_visible_memory_node()
only ever consulted the CPU_VISIBLE_MEMORY_NODES env var, which is never
set on that launch path. See vllm.platforms.interface for the in-process
mapping this now also checks.
"""

import os

import pytest

import vllm.platforms.interface as platform_interface
from vllm.utils import cpu_resource_utils


@pytest.fixture(autouse=True)
def _isolate_assigned_ids_and_env(monkeypatch):
    """Ensure the in-process global and env var don't leak across tests."""
    monkeypatch.setattr(platform_interface, "_assigned_physical_gpu_ids", None)
    monkeypatch.delenv(cpu_resource_utils.DEVICE_CONTROL_ENV_VAR, raising=False)
    monkeypatch.delenv("VLLM_CPU_SIM_MULTI_NUMA", raising=False)
    yield


def _stub_memory_affinity(monkeypatch, nodes: list[int]) -> None:
    monkeypatch.setattr(cpu_resource_utils, "get_memory_affinity", lambda: nodes)


def test_visible_memory_node_prefers_in_process_assignment(monkeypatch):
    """Per-DP-rank shard set via set_assigned_physical_gpu_ids() must win,
    even though CPU_VISIBLE_MEMORY_NODES was never written to the env."""
    _stub_memory_affinity(monkeypatch, [0, 1, 2, 3])
    platform_interface.set_assigned_physical_gpu_ids([2])

    assert cpu_resource_utils.get_visible_memory_node() == [2]


def test_two_dp_ranks_resolve_to_different_numa_nodes(monkeypatch):
    """Simulates two sibling DP EngineCore processes: each only ever sees
    its own in-process global, so they must not collapse to the same node."""
    _stub_memory_affinity(monkeypatch, [0, 1])

    platform_interface.set_assigned_physical_gpu_ids([0])
    dp_rank_0_nodes = cpu_resource_utils.get_visible_memory_node()

    # A fresh process would have its own module-level global; here we
    # simulate that by resetting it directly rather than going through
    # set_assigned_physical_gpu_ids(), which rejects conflicting values.
    monkeypatch.setattr(platform_interface, "_assigned_physical_gpu_ids", None)
    platform_interface.set_assigned_physical_gpu_ids([1])
    dp_rank_1_nodes = cpu_resource_utils.get_visible_memory_node()

    assert dp_rank_0_nodes == [0]
    assert dp_rank_1_nodes == [1]
    assert dp_rank_0_nodes != dp_rank_1_nodes


def test_visible_memory_node_falls_back_to_env_var(monkeypatch):
    """When no in-process mapping is set, the env var path still works."""
    _stub_memory_affinity(monkeypatch, [0, 1, 2, 3])
    monkeypatch.setenv(cpu_resource_utils.DEVICE_CONTROL_ENV_VAR, "1,3")

    assert cpu_resource_utils.get_visible_memory_node() == [1, 3]


def test_visible_memory_node_defaults_to_full_affinity(monkeypatch):
    """With neither mechanism set, fall back to the full affinity list."""
    _stub_memory_affinity(monkeypatch, [0, 1, 2, 3])

    assert cpu_resource_utils.get_visible_memory_node() == [0, 1, 2, 3]
