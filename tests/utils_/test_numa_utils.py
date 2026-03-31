# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from types import SimpleNamespace

import pytest

from vllm.config import ParallelConfig
from vllm.utils import numa_utils


def _make_config(**parallel_kwargs):
    parallel_defaults = dict(
        numa_bind=False,
        numa_bind_nodes=None,
        numa_bind_cpus=None,
        distributed_executor_backend="mp",
        data_parallel_backend="mp",
        nnodes_within_dp=1,
        data_parallel_rank_local=0,
        data_parallel_index=0,
        pipeline_parallel_size=1,
        tensor_parallel_size=1,
    )
    parallel_defaults.update(parallel_kwargs)
    parallel_config = SimpleNamespace(**parallel_defaults)
    return SimpleNamespace(parallel_config=parallel_config)


def test_get_numactl_args_with_node_binding():
    vllm_config = _make_config(numa_bind=True, numa_bind_nodes=[0, 1])
    assert (
        numa_utils._get_numactl_args(vllm_config, local_rank=1)
        == "--cpunodebind=1 --membind=1"
    )


def test_get_numactl_args_with_cpu_binding():
    vllm_config = _make_config(
        numa_bind=True,
        numa_bind_nodes=[0, 1],
        numa_bind_cpus=["0-3", "4-7"],
    )
    assert (
        numa_utils._get_numactl_args(vllm_config, local_rank=1)
        == "--physcpubind=4-7 --membind=1"
    )


def test_get_numactl_args_uses_dp_offset():
    vllm_config = _make_config(
        numa_bind=True,
        numa_bind_nodes=[0, 0, 1, 1],
        data_parallel_rank_local=1,
        pipeline_parallel_size=1,
        tensor_parallel_size=2,
    )
    assert (
        numa_utils._get_numactl_args(vllm_config, local_rank=1)
        == "--cpunodebind=1 --membind=1"
    )


def test_get_numactl_args_requires_detectable_nodes(monkeypatch):
    vllm_config = _make_config(numa_bind=True)
    monkeypatch.setattr(numa_utils, "get_auto_numa_nodes", lambda: None)
    with pytest.raises(RuntimeError):
        numa_utils._get_numactl_args(vllm_config, local_rank=0)


def test_log_numactl_show(monkeypatch):
    log_lines = []

    def fake_debug(msg, *args):
        log_lines.append(msg % args)

    monkeypatch.setattr(numa_utils.logger, "debug", fake_debug)
    monkeypatch.setattr(
        numa_utils.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout="policy: bind\nphyscpubind: 0 1 2 3\n", returncode=0
        ),
    )

    assert numa_utils._log_numactl_show("Worker_0") is True
    assert log_lines == [
        "Worker_0 affinity: policy: bind, physcpubind: 0 1 2 3",
    ]


def test_get_numactl_executable_points_to_fixed_wrapper():
    executable, debug_str = numa_utils._get_numactl_executable()
    assert executable.endswith("/vllm/utils/numa_wrapper.sh")
    assert "_VLLM_INTERNAL_NUMACTL_ARGS" in debug_str


def test_set_numa_wrapper_env_restores_previous_values():
    os.environ[numa_utils._NUMACTL_ARGS_ENV] = "old-args"
    os.environ[numa_utils._NUMACTL_PYTHON_EXECUTABLE_ENV] = "old-python"

    with numa_utils._set_numa_wrapper_env("new-args", "new-python"):
        assert os.environ[numa_utils._NUMACTL_ARGS_ENV] == "new-args"
        assert os.environ[numa_utils._NUMACTL_PYTHON_EXECUTABLE_ENV] == "new-python"

    assert os.environ[numa_utils._NUMACTL_ARGS_ENV] == "old-args"
    assert os.environ[numa_utils._NUMACTL_PYTHON_EXECUTABLE_ENV] == "old-python"


def test_set_numa_wrapper_env_clears_values_when_unset():
    os.environ.pop(numa_utils._NUMACTL_ARGS_ENV, None)
    os.environ.pop(numa_utils._NUMACTL_PYTHON_EXECUTABLE_ENV, None)

    with numa_utils._set_numa_wrapper_env("new-args", "new-python"):
        assert os.environ[numa_utils._NUMACTL_ARGS_ENV] == "new-args"
        assert os.environ[numa_utils._NUMACTL_PYTHON_EXECUTABLE_ENV] == "new-python"

    assert numa_utils._NUMACTL_ARGS_ENV not in os.environ
    assert numa_utils._NUMACTL_PYTHON_EXECUTABLE_ENV not in os.environ


def test_parallel_config_validates_numa_bind_nodes():
    with pytest.raises(ValueError, match="non-negative"):
        ParallelConfig(numa_bind_nodes=[0, -1])


@pytest.mark.parametrize("cpuset", ["", "abc", "1-", "4-1", "1,,2", "1:2"])
def test_parallel_config_rejects_invalid_numa_bind_cpus(cpuset):
    with pytest.raises(ValueError, match="numa_bind_cpus"):
        ParallelConfig(numa_bind_cpus=[cpuset])
