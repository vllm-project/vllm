# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import random
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import ray
import torch
import torch.distributed as dist

from vllm.distributed.communication_op import tensor_model_parallel_all_reduce  # noqa
from vllm.distributed.device_communicators import custom_all_reduce
from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce
from vllm.distributed.parallel_state import get_tp_group, graph_capture

from ..utils import (
    ensure_model_parallel_initialized,
    init_test_distributed_environment,
    multi_process_parallel,
)

random.seed(42)
test_sizes = [random.randint(1024, 2048 * 1024) for _ in range(8)]
for i, v in enumerate(test_sizes):
    test_sizes[i] -= v % 8


def test_sp16_dispatches_only_to_mnnvl_lamport(
    monkeypatch: pytest.MonkeyPatch,
):
    """SP16 uses the MNNVL Lamport kernels and rejects same-host dispatch."""
    comm = CustomAllreduce.__new__(CustomAllreduce)
    comm.disabled = False
    comm.world_size = 16
    comm.fully_connected = False
    comm.mnnvl_only = True
    comm._IS_CAPTURING = False
    comm.max_mnnvl_all_gather_size = 2 * 1024 * 1024
    comm.max_mnnvl_reduce_scatter_size = 16 * 1024 * 1024
    comm.mnnvl_multicast_ptr = 1
    comm.mnnvl_lamport_ag_local_ptr = 1
    comm.mnnvl_lamport_ag_multicast_ptr = 1
    comm.mnnvl_lamport_ag_epoch_ptr = 1
    comm.mnnvl_lamport_rs_local_ptr = 1
    comm.mnnvl_lamport_rs_epoch_ptr = 1
    comm.mnnvl_buffer_size = 32 * 1024 * 1024
    comm._ptr = 0

    lamport_all_gather = Mock()
    lamport_reduce_scatter = Mock()
    monkeypatch.setattr(custom_all_reduce.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        custom_all_reduce,
        "ops",
        SimpleNamespace(
            mnnvl_lamport_all_gather=lamport_all_gather,
            mnnvl_lamport_reduce_scatter=lamport_reduce_scatter,
        ),
    )

    gathered = comm.custom_all_gather(torch.empty((8, 8), dtype=torch.bfloat16))
    scattered = comm.custom_reduce_scatter(torch.empty((16, 8), dtype=torch.bfloat16))

    assert gathered is not None
    assert scattered is not None
    lamport_all_gather.assert_called_once()
    lamport_reduce_scatter.assert_called_once()
    assert not comm.should_custom_ar(torch.empty(8, dtype=torch.bfloat16))
    assert not comm.should_custom_all_gather(torch.empty((8, 8), dtype=torch.int32))
    assert not comm.should_custom_all_gather(
        torch.empty((131073, 8), dtype=torch.bfloat16)
    )

    comm.mnnvl_only = False
    assert not comm.should_custom_all_gather(torch.empty((8, 8), dtype=torch.bfloat16))
    assert not comm.should_custom_reduce_scatter(
        torch.empty((16, 8), dtype=torch.bfloat16)
    )


@ray.remote(num_gpus=1, max_calls=1)
def graph_allreduce(
    monkeypatch: pytest.MonkeyPatch,
    tp_size,
    pp_size,
    rank,
    distributed_init_port,
):
    with monkeypatch.context() as m:
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        m.delenv("HIP_VISIBLE_DEVICES", raising=False)
        device = torch.device(f"cuda:{rank}")
        torch.accelerator.set_device_index(device)
        init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)
        ensure_model_parallel_initialized(tp_size, pp_size)
        group = get_tp_group().device_group

        # A small all_reduce for warmup.
        # this is needed because device communicators might be created lazily
        # (e.g. NCCL). This will ensure that the communicator is initialized
        # before any communication happens, so that this group can be used for
        # graph capture immediately.
        data = torch.zeros(1)
        data = data.to(device=device)
        torch.distributed.all_reduce(data, group=group)
        torch.accelerator.synchronize()
        del data

        # we use the first group to communicate once
        # and the second group to communicate twice
        # and so on
        # this is used to demonstrate that each group can
        # communicate independently
        num_communication = rank // tp_size + 1

        for sz in test_sizes:
            for dtype in [torch.float32, torch.float16, torch.bfloat16]:
                with graph_capture(device=device) as graph_capture_context:
                    # use integers so result matches NCCL exactly
                    device_idx = torch.accelerator.current_device_index()
                    inp1 = torch.randint(1, 16, (sz,), dtype=dtype, device=device_idx)
                    inp2 = torch.randint(1, 16, (sz,), dtype=dtype, device=device_idx)

                    torch.accelerator.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=graph_capture_context.stream):
                        for i in range(num_communication):
                            out1 = tensor_model_parallel_all_reduce(inp1)
                            # the input buffer is immediately modified to test
                            # synchronization
                            dist.all_reduce(inp1, group=group)
                            out2 = tensor_model_parallel_all_reduce(inp2)
                            dist.all_reduce(inp2, group=group)
                graph.replay()
                torch.testing.assert_close(out1, inp1)
                torch.testing.assert_close(out2, inp2)

        fa = get_tp_group().device_communicator.ca_comm
        tp_rank = rank % tp_size
        with graph_capture(device=device) as graph_capture_context:
            local = torch.full(
                (512, 4096), tp_rank + 1, dtype=torch.bfloat16, device=device
            )
            reduce_input = torch.full(
                (512 * tp_size, 4096),
                tp_rank + 1,
                dtype=torch.bfloat16,
                device=device,
            )
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=graph_capture_context.stream):
                gathered = fa.custom_all_gather(local)
                scattered = fa.custom_reduce_scatter(reduce_input)
        graph.replay()
        assert gathered is not None
        assert scattered is not None
        expected_gather = torch.cat(
            [torch.full_like(local, peer_rank + 1) for peer_rank in range(tp_size)]
        )
        expected_scatter = torch.full_like(local, tp_size * (tp_size + 1) // 2)
        torch.testing.assert_close(gathered, expected_gather)
        torch.testing.assert_close(scattered, expected_scatter)


@ray.remote(num_gpus=1, max_calls=1)
def eager_allreduce(
    monkeypatch: pytest.MonkeyPatch,
    tp_size,
    pp_size,
    rank,
    distributed_init_port,
):
    with monkeypatch.context() as m:
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        m.delenv("HIP_VISIBLE_DEVICES", raising=False)
        device = torch.device(f"cuda:{rank}")
        torch.accelerator.set_device_index(device)
        init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)

        # we use the first group to communicate once
        # and the second group to communicate twice
        # and so on
        # this is used to demonstrate that each group can
        # communicate independently
        num_communication = rank // tp_size + 1
        sz = 1024
        fa = get_tp_group().device_communicator.ca_comm
        inp = torch.ones(sz, dtype=torch.float32, device=device)
        out = inp
        for _ in range(num_communication):
            out = fa.all_reduce(out, registered=False)
        torch.testing.assert_close(out, inp * (tp_size**num_communication))

        group = get_tp_group().device_group
        tp_rank = rank % tp_size
        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            local = torch.full((64, 4096), tp_rank + 1, dtype=dtype, device=device)
            expected_gather = torch.empty(
                (64 * tp_size, 4096), dtype=dtype, device=device
            )
            dist.all_gather_into_tensor(expected_gather, local, group=group)
            gathered = fa.custom_all_gather(local)
            assert gathered is not None
            torch.testing.assert_close(gathered, expected_gather)

            reduce_input = torch.full(
                (64 * tp_size, 4096), tp_rank + 1, dtype=dtype, device=device
            )
            expected_scatter = torch.empty((64, 4096), dtype=dtype, device=device)
            dist.reduce_scatter_tensor(
                expected_scatter, reduce_input.clone(), group=group
            )
            scattered = fa.custom_reduce_scatter(reduce_input)
            assert scattered is not None
            torch.testing.assert_close(scattered, expected_scatter)

        inp = torch.ones(sz * 4, dtype=torch.bfloat16, device=device)
        out = inp
        for _ in range(num_communication):
            out = fa.all_reduce(out, registered=False)
        torch.testing.assert_close(out, inp * (tp_size**num_communication))


@pytest.mark.parametrize("tp_size", [2])
@pytest.mark.parametrize("pipeline_parallel_size", [1, 2])
@pytest.mark.parametrize("test_target", [eager_allreduce, graph_allreduce])
def test_custom_allreduce(
    monkeypatch: pytest.MonkeyPatch,
    tp_size,
    pipeline_parallel_size,
    test_target,
):
    world_size = tp_size * pipeline_parallel_size
    if world_size > torch.accelerator.device_count():
        pytest.skip("Not enough GPUs to run the test.")
    multi_process_parallel(monkeypatch, tp_size, pipeline_parallel_size, test_target)


@pytest.mark.parametrize("test_target", [eager_allreduce, graph_allreduce])
def test_custom_collectives_world_size_four(
    monkeypatch: pytest.MonkeyPatch,
    test_target,
):
    """Exercise the four-rank kernel specialization used by Kimi SP."""
    if torch.accelerator.device_count() < 4:
        pytest.skip("Not enough GPUs to run the test.")
    multi_process_parallel(monkeypatch, 4, 1, test_target)
