# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.utils import cpu_resource_utils as cr_utils


@pytest.fixture(autouse=True)
def _clear_cpu_memory_envs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(cr_utils.DEVICE_CONTROL_ENV_VAR, raising=False)
    monkeypatch.delenv("VLLM_CPU_SIM_MULTI_NUMA", raising=False)


def _stub_linux_memory_affinity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cr_utils.sys, "platform", "linux")
    monkeypatch.setattr(cr_utils, "get_memory_affinity", lambda: [0, 1, 2, 3])


@pytest.mark.parametrize("sim_multi_numa", [None, "0"])
def test_get_visible_memory_node_uses_visible_nodes_when_not_simulating(
    monkeypatch: pytest.MonkeyPatch, sim_multi_numa: str | None
) -> None:
    _stub_linux_memory_affinity(monkeypatch)
    monkeypatch.setenv(cr_utils.DEVICE_CONTROL_ENV_VAR, "1,3")
    if sim_multi_numa is not None:
        monkeypatch.setenv("VLLM_CPU_SIM_MULTI_NUMA", sim_multi_numa)

    assert cr_utils.get_visible_memory_node() == [1, 3]


def test_get_visible_memory_node_ignores_visible_nodes_when_simulating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_linux_memory_affinity(monkeypatch)
    monkeypatch.setenv(cr_utils.DEVICE_CONTROL_ENV_VAR, "1,3")
    monkeypatch.setenv("VLLM_CPU_SIM_MULTI_NUMA", "1")

    assert cr_utils.get_visible_memory_node() == [0, 1, 2, 3]
