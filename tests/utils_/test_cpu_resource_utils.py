# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import subprocess

import pytest

from vllm.utils import cpu_resource_utils
from vllm.utils.cpu_resource_utils import LogicalCPUInfo


@pytest.fixture
def synthesized_cpu_list(monkeypatch: pytest.MonkeyPatch) -> list[LogicalCPUInfo]:
    cpu_list = [LogicalCPUInfo(id=0, physical_core=0, numa_node=0)]
    monkeypatch.setattr(cpu_resource_utils, "_synthesize_cpu_list", lambda: cpu_list)
    return cpu_list


@pytest.mark.parametrize(
    "error",
    [
        FileNotFoundError("lscpu is unavailable"),
        subprocess.CalledProcessError(1, "lscpu"),
    ],
)
def test_get_cpu_list_falls_back_when_lscpu_fails(
    monkeypatch: pytest.MonkeyPatch,
    synthesized_cpu_list: list[LogicalCPUInfo],
    error: Exception,
) -> None:
    def raise_error(*args, **kwargs):
        raise error

    monkeypatch.setattr(subprocess, "check_output", raise_error)

    assert cpu_resource_utils._get_cpu_list() is synthesized_cpu_list


@pytest.mark.parametrize("lscpu_output", ["not-json", "{}", '{"cpus": [null]}'])
def test_get_cpu_list_falls_back_when_lscpu_output_is_unparsable(
    monkeypatch: pytest.MonkeyPatch,
    synthesized_cpu_list: list[LogicalCPUInfo],
    lscpu_output: str,
) -> None:
    monkeypatch.setattr(
        subprocess, "check_output", lambda *args, **kwargs: lscpu_output
    )

    assert cpu_resource_utils._get_cpu_list() is synthesized_cpu_list
