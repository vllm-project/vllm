# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

import vllm.utils.ompmultiprocessing as omp
from vllm.platforms.interface import CpuArchEnum


def _config(local_world_size, kv_transfer=False):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            local_world_size=local_world_size,
            data_parallel_rank_local=None,
            _api_process_count=1,
        ),
        kv_transfer_config=object() if kv_transfer else None,
    )


def _patch_cpu_platform(monkeypatch, architecture):
    monkeypatch.setattr(
        omp,
        "current_platform",
        SimpleNamespace(
            is_cpu=lambda: True,
            get_cpu_architecture=lambda: architecture,
        ),
    )


@pytest.mark.parametrize(
    ("kv_transfer", "expected"),
    [(False, 4), (True, 8)],
)
def test_x86_reserves_one_cpu_per_local_rank(monkeypatch, kv_transfer, expected):
    _patch_cpu_platform(monkeypatch, CpuArchEnum.X86)
    monkeypatch.setattr(omp.envs, "VLLM_CPU_NUM_OF_RESERVED_CPU", None)
    monkeypatch.setattr(
        omp.OMPProcessManager, "_parse_omp_threads_bind_env", lambda self: None
    )

    manager = omp.OMPProcessManager(
        _config(local_world_size=4, kv_transfer=kv_transfer)
    )

    assert manager.reserve_cpu_num == expected
