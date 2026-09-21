# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import ray
import torch
import torch.distributed as dist

from vllm._aiter_ops import is_aiter_found, rocm_aiter_ops
from vllm.distributed.communication_op import tensor_model_parallel_all_reduce  # noqa
from vllm.distributed.device_communicators.aiter_custom_all_reduce import (
    AiterCustomAllreduce,
)
from vllm.distributed.parallel_state import get_dp_group, get_tp_group, graph_capture
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


def _get_aiter_ag_rs_comm():
    device_communicator = get_dp_group().device_communicator
    assert device_communicator.use_aiter_ag_rs, (
        "AITER custom AG/RS was not enabled on the DP group."
    )
    aiter_comm = device_communicator.aiter_ar_comm
    assert aiter_comm is not None, "AITER custom AG/RS was not initialized."
    assert not aiter_comm.disabled, "AITER custom AG/RS is disabled."
    return aiter_comm


def _assert_aiter_handles_ag(aiter_comm, inp: torch.Tensor) -> None:
    assert aiter_comm.should_custom_ag(inp), (
        f"AITER custom all-gather does not support input shape {inp.shape}."
    )


def _assert_aiter_handles_rs(aiter_comm, inp: torch.Tensor) -> None:
    assert aiter_comm.should_custom_rs(inp, dim=0), (
        f"AITER custom reduce-scatter does not support input shape {inp.shape}."
    )


@ray.remote(num_gpus=1, max_calls=1)
def eager_ag_rs(
    monkeypatch: pytest.MonkeyPatch,
    tp_size,
    pp_size,
    rank,
    distributed_init_port,
    data_parallel_size,
    data_parallel_master_port,
) -> None:
    with monkeypatch.context() as m:
        _configure_aiter_custom_ar_env(m)

        device = torch.device(f"cuda:{rank}")
        torch.accelerator.set_device_index(device)
        init_test_distributed_environment(
            tp_size,
            pp_size,
            rank,
            distributed_init_port,
            data_parallel_size=data_parallel_size,
            data_parallel_master_port=data_parallel_master_port,
        )

        dp_group = get_dp_group()
        group = dp_group.device_group
        dp_world = dp_group.world_size
        aiter_comm = _get_aiter_ag_rs_comm()

        for shape, dtype in test_cases:
            num_tokens, hidden = shape

            # all-gather: each rank contributes (num_tokens, hidden).
            inp = torch.ones(shape, dtype=dtype, device=device) * (rank + 1)
            _assert_aiter_handles_ag(aiter_comm, inp)
            expected = torch.empty(
                (num_tokens * dp_world, hidden), dtype=dtype, device=device
            )
            dist.all_gather_into_tensor(expected, inp, group=group)
            out = aiter_comm.custom_all_gather(inp, dim=0)
            assert out is not None
            torch.testing.assert_close(out, expected)

            # reduce-scatter: each rank contributes (num_tokens * dp, hidden).
            rs_in = torch.ones(
                (num_tokens * dp_world, hidden), dtype=dtype, device=device
            ) * (rank + 1)
            _assert_aiter_handles_rs(aiter_comm, rs_in)
            rs_expected = torch.empty((num_tokens, hidden), dtype=dtype, device=device)
            dist.reduce_scatter_tensor(rs_expected, rs_in, group=group)
            rs_out = torch.empty((num_tokens, hidden), dtype=dtype, device=device)
            aiter_comm.custom_reduce_scatter(rs_in, rs_out, dim=0)
            torch.testing.assert_close(rs_out, rs_expected)


@ray.remote(num_gpus=1, max_calls=1)
def graph_ag_rs(
    monkeypatch: pytest.MonkeyPatch,
    tp_size,
    pp_size,
    rank,
    distributed_init_port,
    data_parallel_size,
    data_parallel_master_port,
) -> None:
    with monkeypatch.context() as m:
        _configure_aiter_custom_ar_env(m)

        device = torch.device(f"cuda:{rank}")
        torch.accelerator.set_device_index(device)
        init_test_distributed_environment(
            tp_size,
            pp_size,
            rank,
            distributed_init_port,
            data_parallel_size=data_parallel_size,
            data_parallel_master_port=data_parallel_master_port,
        )

        dp_group = get_dp_group()
        group = dp_group.device_group
        dp_world = dp_group.world_size
        aiter_comm = _get_aiter_ag_rs_comm()

        # Warmup so DP comms is initialized before graph capture
        data = torch.zeros(1, device=device)
        dist.all_reduce(data, group=group)
        torch.accelerator.synchronize()
        del data

        for shape, dtype in test_cases:
            num_tokens, hidden = shape

            # all-gather under graph capture.
            inp = torch.ones(shape, dtype=dtype, device=device) * (rank + 1)
            _assert_aiter_handles_ag(aiter_comm, inp)
            ag_expected = torch.empty(
                (num_tokens * dp_world, hidden), dtype=dtype, device=device
            )
            dist.all_gather_into_tensor(ag_expected, inp, group=group)
            with graph_capture(device=device) as graph_capture_context:
                torch.accelerator.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=graph_capture_context.stream):
                    ag_out = aiter_comm.custom_all_gather(inp, dim=0)
            graph.replay()
            torch.testing.assert_close(ag_out, ag_expected)

            # reduce-scatter under graph capture.
            rs_in = torch.ones(
                (num_tokens * dp_world, hidden), dtype=dtype, device=device
            ) * (rank + 1)
            _assert_aiter_handles_rs(aiter_comm, rs_in)
            rs_expected = torch.empty((num_tokens, hidden), dtype=dtype, device=device)
            dist.reduce_scatter_tensor(rs_expected, rs_in, group=group)
            rs_out = torch.empty((num_tokens, hidden), dtype=dtype, device=device)
            with graph_capture(device=device) as graph_capture_context:
                torch.accelerator.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=graph_capture_context.stream):
                    aiter_comm.custom_reduce_scatter(rs_in, rs_out, dim=0)
            graph.replay()
            torch.testing.assert_close(rs_out, rs_expected)


@pytest.mark.skipif(not is_aiter_found(), reason="AITER is not installed")
@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize("tp_size", [1])
@pytest.mark.parametrize("pipeline_parallel_size", [1])
@pytest.mark.parametrize("data_parallel_size", [2])
@pytest.mark.parametrize("test_target", [eager_ag_rs, graph_ag_rs])
def test_rocm_aiter_custom_ag_rs(
    monkeypatch: pytest.MonkeyPatch,
    tp_size,
    pipeline_parallel_size,
    data_parallel_size,
    test_target,
):
    multi_process_parallel(
        monkeypatch,
        tp_size,
        pipeline_parallel_size,
        test_target,
        data_parallel_size=data_parallel_size,
    )


def _use_1stage(world_size: int, inp: torch.Tensor, fully_connected: bool = True):
    """Evaluate the launcher predicate without building a real AITER instance.

    It reads nothing but ``world_size``, ``fully_connected`` and the input's shape
    and dtype, so the boundaries are testable on the host.
    """
    comm = SimpleNamespace(
        _impl=SimpleNamespace(world_size=world_size, fully_connected=fully_connected)
    )
    return AiterCustomAllreduce.use_1stage_fused_ar_rms(comm, inp)


@pytest.mark.parametrize(
    "world_size,tokens,hidden,expected",
    [
        # TP=2 has no byte cap: the one-stage launcher wins over the whole range
        # the row and pack caps leave reachable, up to 80 x 8192 x 2 = 1280 KiB.
        (2, 80, 8192, True),
        # TP<=4 admits up to 320 KiB, i.e. 40 tokens at hidden=4096 bf16. The
        # bound is inclusive: 40 lands on the cap exactly and is a capture size.
        (4, 40, 4096, True),
        (4, 41, 4096, False),
        (3, 40, 4096, True),
        # TP<=8 admits up to 192 KiB, i.e. 24 tokens at hidden=4096 bf16.
        (8, 24, 4096, True),
        (8, 25, 4096, False),
        (5, 24, 4096, True),
        # hidden=6144 puts its 16-token capture size on the cap exactly.
        (8, 16, 6144, True),
        (8, 17, 6144, False),
        # Above 8 the fused one-stage launcher is never selected.
        (16, 1, 4096, False),
    ],
)
def test_use_1stage_fused_ar_rms_byte_caps(world_size, tokens, hidden, expected):
    inp = torch.empty((tokens, hidden), dtype=torch.bfloat16)
    assert _use_1stage(world_size, inp) is expected


def test_use_1stage_fused_ar_rms_row_cap_precedes_byte_cap():
    # 81 rows of hidden=1024 is 162 KiB, inside every byte cap, and still outside
    # the kernel's 80-row contract.
    assert _use_1stage(8, torch.empty((80, 1024), dtype=torch.bfloat16)) is True
    assert _use_1stage(8, torch.empty((81, 1024), dtype=torch.bfloat16)) is False


@pytest.mark.parametrize(
    "tokens,hidden,dtype,fully_connected,expected",
    [
        # Neither cap rescues an unsupported dtype or a row the packer cannot split.
        (8, 4096, torch.float32, True, False),
        (8, 4100, torch.bfloat16, True, False),
        (8, 16400, torch.bfloat16, True, False),
        # Above TP=2 the one-stage launcher needs an all-to-all topology.
        (8, 4096, torch.bfloat16, False, False),
    ],
)
def test_use_1stage_fused_ar_rms_other_conditions_unchanged(
    tokens, hidden, dtype, fully_connected, expected
):
    inp = torch.empty((tokens, hidden), dtype=dtype)
    assert _use_1stage(8, inp, fully_connected=fully_connected) is expected
