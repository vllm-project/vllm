# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import ray
import torch
import torch.distributed as dist

from vllm._aiter_ops import is_aiter_found, rocm_aiter_ops
from vllm.distributed.communication_op import tensor_model_parallel_all_reduce  # noqa
from vllm.distributed.parallel_state import get_tp_group, graph_capture
from vllm.envs import disable_envs_cache
from vllm.platforms import current_platform

from ..utils import (
    assert_rocm_custom_allreduce_backend_state,
    ensure_model_parallel_initialized,
    init_test_distributed_environment,
    multi_gpu_test,
    multi_process_parallel,
)

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="ROCm-only AITER custom allreduce tests",
)

test_cases = [
    ((2, 7168), torch.float16),
    ((2, 7168), torch.bfloat16),
    ((128, 8192), torch.float16),
    ((128, 8192), torch.bfloat16),
]


def _configure_aiter_custom_ar_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_CUSTOM_AR", "1")
    monkeypatch.setenv("VLLM_ROCM_QUICK_REDUCE_QUANTIZATION", "NONE")
    disable_envs_cache()
    rocm_aiter_ops.refresh_env_variables()


def _assert_aiter_handles_input(inp: torch.Tensor) -> None:
    aiter_ar_comm = get_tp_group().device_communicator.aiter_ar_comm
    assert aiter_ar_comm is not None
    assert aiter_ar_comm.should_custom_ar(inp), (
        f"AITER CustomAllreduce does not support input shape {inp.shape}."
    )


@ray.remote(num_gpus=1, max_calls=1)
def graph_allreduce(
    monkeypatch: pytest.MonkeyPatch,
    tp_size,
    pp_size,
    rank,
    distributed_init_port,
) -> None:
    with monkeypatch.context() as m:
        _configure_aiter_custom_ar_env(m)

        device = torch.device(f"cuda:{rank}")
        torch.accelerator.set_device_index(device)
        init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)
        ensure_model_parallel_initialized(tp_size, pp_size)
        assert_rocm_custom_allreduce_backend_state(True, "NONE")
        group = get_tp_group().device_group

        # A small all_reduce for warmup.
        # this is needed because device communicators might be created lazily
        # (e.g. NCCL). This will ensure that the communicator is initialized
        # before any communication happens, so that this group can be used for
        # graph capture immediately.
        data = torch.zeros(1)
        data = data.to(device=device)
        dist.all_reduce(data, group=group)
        torch.accelerator.synchronize()
        del data

        for shape, dtype in test_cases:
            with graph_capture(device=device) as graph_capture_context:
                inp = torch.ones(shape, dtype=dtype, device=device)
                _assert_aiter_handles_input(inp)
                expected = inp * tp_size

                torch.accelerator.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=graph_capture_context.stream):
                    out = tensor_model_parallel_all_reduce(inp)

            graph.replay()
            torch.testing.assert_close(out, expected)


@ray.remote(num_gpus=1, max_calls=1)
def eager_allreduce(
    monkeypatch: pytest.MonkeyPatch,
    tp_size,
    pp_size,
    rank,
    distributed_init_port,
) -> None:
    with monkeypatch.context() as m:
        _configure_aiter_custom_ar_env(m)

        device = torch.device(f"cuda:{rank}")
        torch.accelerator.set_device_index(device)
        init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)
        ensure_model_parallel_initialized(tp_size, pp_size)
        assert_rocm_custom_allreduce_backend_state(True, "NONE")

        for shape, dtype in test_cases:
            inp = torch.ones(shape, dtype=dtype, device=device)
            _assert_aiter_handles_input(inp)
            expected = inp * tp_size
            out = tensor_model_parallel_all_reduce(inp)
            torch.testing.assert_close(out, expected)


@pytest.mark.skipif(not is_aiter_found(), reason="AITER is not installed")
@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize("tp_size", [2])
@pytest.mark.parametrize("pipeline_parallel_size", [1])
@pytest.mark.parametrize("test_target", [eager_allreduce, graph_allreduce])
def test_rocm_aiter_custom_allreduce(
    monkeypatch: pytest.MonkeyPatch,
    tp_size,
    pipeline_parallel_size,
    test_target,
):
    multi_process_parallel(monkeypatch, tp_size, pipeline_parallel_size, test_target)
