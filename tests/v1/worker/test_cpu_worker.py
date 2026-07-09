# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm.config.parallel import ParallelConfig
from vllm.forward_context import DPMetadata
from vllm.platforms import current_platform
from vllm.platforms.cpu import CpuPlatform
from vllm.v1.worker.cpu_worker import _get_cpushm_dist_ident

if not current_platform.is_cpu():
    pytest.skip("CPU-only test", allow_module_level=True)


def _make_parallel_config(dp_port: int) -> ParallelConfig:
    return ParallelConfig(
        tensor_parallel_size=1,
        data_parallel_size=2,
        data_parallel_master_ip="127.0.0.1",
        _data_parallel_master_port_list=[dp_port],
    )


def _make_platform_config(
    *,
    tensor_parallel_size: int,
    data_parallel_size: int,
    enable_expert_parallel: bool,
):
    return SimpleNamespace(
        model_config=None,
        cache_config=SimpleNamespace(
            user_specified_block_size=True,
            block_size=128,
            kv_cache_memory_bytes=0,
            mamba_ssm_cache_dtype="auto",
        ),
        scheduler_config=SimpleNamespace(async_scheduling=True),
        parallel_config=ParallelConfig(
            tensor_parallel_size=tensor_parallel_size,
            data_parallel_size=data_parallel_size,
            enable_expert_parallel=enable_expert_parallel,
            data_parallel_master_ip="127.0.0.1",
            _data_parallel_master_port_list=[26001],
            distributed_executor_backend="mp",
        ),
        compilation_config=SimpleNamespace(
            cudagraph_capture_sizes=[1],
            mode=None,
            backend=None,
            inductor_compile_config={},
            ir_enable_torch_wrap=True,
            custom_ops=[],
        ),
        lora_config=None,
        profiler_config=SimpleNamespace(torch_profiler_dump_cuda_time_total=True),
        device_config=SimpleNamespace(device_type="cpu"),
    )


@pytest.mark.parametrize(
    ("dp_ports", "expected"),
    [
        ([26001, 26001], True),
        ([26001, 26002], False),
    ],
    ids=["same-dp-rendezvous", "different-dp-rendezvous"],
)
def test_get_cpushm_dist_ident_uses_dp_rendezvous(dp_ports, expected):
    idents = [
        _get_cpushm_dist_ident(
            _make_parallel_config(dp_port),
            f"tcp://127.0.0.1:{11001 + rank}",
        )
        for rank, dp_port in enumerate(dp_ports)
    ]

    assert (idents[0] == idents[1]) is expected


def test_dp_metadata_sp_local_sizes_caches_python_sizes_and_restores():
    metadata = DPMetadata(torch.tensor([7, 8, 9], dtype=torch.int32))

    with metadata.sp_local_sizes(sequence_parallel_size=2) as sizes:
        assert sizes == [4, 4, 4, 4, 5, 5]
        assert metadata.local_sizes is sizes
        with metadata.sp_local_sizes(sequence_parallel_size=3) as nested_sizes:
            assert nested_sizes == [3, 3, 3, 3, 3, 3, 3, 3, 3]
            assert metadata.local_sizes is nested_sizes
        assert metadata.local_sizes is sizes

    assert metadata.local_sizes is None
    with metadata.sp_local_sizes(sequence_parallel_size=2) as cached_sizes:
        assert cached_sizes is sizes


def test_cpu_platform_allows_hybrid_tp_dp_ep():
    vllm_config = _make_platform_config(
        tensor_parallel_size=2,
        data_parallel_size=3,
        enable_expert_parallel=True,
    )

    with patch.dict(os.environ, {"VLLM_CPU_KVCACHE_SPACE": ""}):
        CpuPlatform.check_and_update_config(vllm_config)


def test_cpu_platform_allows_tp_only_ep():
    vllm_config = _make_platform_config(
        tensor_parallel_size=2,
        data_parallel_size=1,
        enable_expert_parallel=True,
    )

    with patch.dict(os.environ, {"VLLM_CPU_KVCACHE_SPACE": ""}):
        CpuPlatform.check_and_update_config(vllm_config)


def test_cpu_platform_allows_hybrid_tp_dp_without_ep():
    vllm_config = _make_platform_config(
        tensor_parallel_size=2,
        data_parallel_size=3,
        enable_expert_parallel=False,
    )

    with patch.dict(os.environ, {"VLLM_CPU_KVCACHE_SPACE": ""}):
        CpuPlatform.check_and_update_config(vllm_config)


def test_cpu_platform_allows_dp_only_ep():
    vllm_config = _make_platform_config(
        tensor_parallel_size=1,
        data_parallel_size=2,
        enable_expert_parallel=True,
    )

    with patch.dict(os.environ, {"VLLM_CPU_KVCACHE_SPACE": ""}):
        CpuPlatform.check_and_update_config(vllm_config)
