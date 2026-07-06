# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm.config.parallel import ParallelConfig
from vllm.platforms import current_platform
from vllm.platforms.cpu import CpuPlatform
from vllm.v1.worker.cpu_worker import _get_cpushm_dist_ident
from vllm.v1.worker.dp_utils import _synchronize_dp_ranks

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


def test_get_cpushm_dist_ident_uses_dp_rendezvous_for_single_node_dp():
    ident_rank0 = _get_cpushm_dist_ident(
        _make_parallel_config(26001),
        "tcp://127.0.0.1:11001",
    )
    ident_rank1 = _get_cpushm_dist_ident(
        _make_parallel_config(26001),
        "tcp://127.0.0.1:11002",
    )

    assert ident_rank0 == ident_rank1 == "127.0.0.1:26001"


def test_get_cpushm_dist_ident_differs_for_different_dp_rendezvous():
    ident_a = _get_cpushm_dist_ident(
        _make_parallel_config(26001),
        "tcp://127.0.0.1:11001",
    )
    ident_b = _get_cpushm_dist_ident(
        _make_parallel_config(26002),
        "tcp://127.0.0.1:11002",
    )

    assert ident_a != ident_b


def test_cpu_platform_rejects_only_hybrid_tp_dp_ep():
    vllm_config = _make_platform_config(
        tensor_parallel_size=2,
        data_parallel_size=2,
        enable_expert_parallel=True,
    )

    with (
        patch.dict(os.environ, os.environ.copy(), clear=True),
        pytest.raises(
            NotImplementedError,
            match="tensor_parallel_size > 1 with data_parallel_size > 1",
        ),
    ):
        CpuPlatform.check_and_update_config(vllm_config)


def test_cpu_platform_allows_tp_only_ep():
    vllm_config = _make_platform_config(
        tensor_parallel_size=2,
        data_parallel_size=1,
        enable_expert_parallel=True,
    )

    with patch.dict(os.environ, os.environ.copy(), clear=True):
        CpuPlatform.check_and_update_config(vllm_config)


def test_synchronize_dp_ranks_cpu_ep_moe_does_not_force_padding(monkeypatch):
    parallel_config = ParallelConfig(
        tensor_parallel_size=1,
        data_parallel_size=2,
        data_parallel_rank=0,
        enable_expert_parallel=True,
        is_moe_model=True,
    )

    def fake_run_ar(*args, **kwargs):
        return torch.tensor(
            [
                [3, 5],
                [3, 5],
                [0, 0],
                [0, 0],
            ],
            dtype=torch.int32,
        )

    monkeypatch.setattr("vllm.v1.worker.dp_utils._run_ar", fake_run_ar)

    should_ubatch, num_tokens_after_padding, synced_cudagraph_mode = (
        _synchronize_dp_ranks(
            num_tokens_unpadded=3,
            num_tokens_padded=3,
            should_attempt_ubatching=False,
            cudagraph_mode=0,
            parallel_config=parallel_config,
        )
    )

    assert not should_ubatch
    assert synced_cudagraph_mode == 0
    assert num_tokens_after_padding is not None
    assert num_tokens_after_padding.device.type == "cpu"
    torch.testing.assert_close(
        num_tokens_after_padding,
        torch.tensor([3, 5], dtype=torch.int32),
    )


def test_synchronize_dp_ranks_cpu_ep_moe_pads_small_uniform_decode_batch(
    monkeypatch,
):
    parallel_config = ParallelConfig(
        tensor_parallel_size=1,
        data_parallel_size=2,
        data_parallel_rank=0,
        enable_expert_parallel=True,
        is_moe_model=True,
    )

    def fake_run_ar(*args, **kwargs):
        return torch.tensor(
            [
                [1, 2],
                [1, 2],
                [0, 0],
                [0, 0],
            ],
            dtype=torch.int32,
        )

    monkeypatch.setattr("vllm.v1.worker.dp_utils._run_ar", fake_run_ar)

    should_ubatch, num_tokens_after_padding, synced_cudagraph_mode = (
        _synchronize_dp_ranks(
            num_tokens_unpadded=1,
            num_tokens_padded=1,
            should_attempt_ubatching=False,
            cudagraph_mode=0,
            parallel_config=parallel_config,
        )
    )

    assert not should_ubatch
    assert synced_cudagraph_mode == 0
    assert num_tokens_after_padding is not None
    assert num_tokens_after_padding.device.type == "cpu"
    torch.testing.assert_close(
        num_tokens_after_padding,
        torch.tensor([2, 2], dtype=torch.int32),
    )
