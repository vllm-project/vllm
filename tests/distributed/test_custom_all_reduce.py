# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import random

import pytest
import ray
import torch
import torch.distributed as dist

from vllm import _custom_ops as ops
from vllm.distributed.communication_op import tensor_model_parallel_all_reduce  # noqa
from vllm.distributed.device_communicators.custom_all_reduce import _LifecycleState
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
        assert fa is not None
        with pytest.raises(RuntimeError, match="exactly one eager registered buffer"):
            ops.register_buffer(fa._ptr, fa.buffer_ptrs)
        inp = torch.ones(sz, dtype=torch.float32, device=device)
        out = inp
        for _ in range(num_communication):
            out = fa.all_reduce(out, registered=False)
        torch.testing.assert_close(out, inp * (tp_size**num_communication))

        if torch.version.hip is None:
            expected_peer_mappings = 2 * (fa.world_size - 1)
            assert len(fa._open_peer_ptrs) == expected_peer_mappings
            fa.prepare_for_suspend()
            assert not fa._open_peer_ptrs
            fa.prepare_for_suspend()
            with pytest.raises(RuntimeError, match="detached"):
                tensor_model_parallel_all_reduce(inp)
            fa.reinit_after_resume()
            assert len(fa._open_peer_ptrs) == expected_peer_mappings
            fa.reinit_after_resume()
            eager_out = tensor_model_parallel_all_reduce(inp)
            torch.testing.assert_close(eager_out, inp * tp_size)

        inp = torch.ones(sz * 4, dtype=torch.bfloat16, device=device)
        out = inp
        for _ in range(num_communication):
            out = fa.all_reduce(out, registered=False)
        torch.testing.assert_close(out, inp * (tp_size**num_communication))
        if torch.version.hip is None:
            fa.prepare_for_suspend()
            fa.close()
            fa.close()


@ray.remote(num_gpus=1, max_calls=1)
def graph_allreduce_suspend_resume(
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
        tp_group = get_tp_group()
        device_communicator = tp_group.device_communicator
        assert device_communicator is not None
        fa = device_communicator.ca_comm
        assert fa is not None and not fa.disabled

        selected_calls = 0
        original_custom_all_reduce = fa.custom_all_reduce

        def tracked_custom_all_reduce(input_: torch.Tensor) -> torch.Tensor | None:
            nonlocal selected_calls
            selected_calls += 1
            return original_custom_all_reduce(input_)

        m.setattr(fa, "custom_all_reduce", tracked_custom_all_reduce)
        inp1 = torch.full((1024,), fa.rank + 1, dtype=torch.float32, device=device)
        inp2 = torch.full((2048,), fa.rank + 2, dtype=torch.float32, device=device)
        with graph_capture(device=device) as graph_capture_context:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=graph_capture_context.stream):
                out1 = tensor_model_parallel_all_reduce(inp1)
                out1_repeat = tensor_model_parallel_all_reduce(inp1)
                out2 = tensor_model_parallel_all_reduce(inp2)
        assert selected_calls == 3
        assert fa._lifecycle_state == _LifecycleState.ACTIVE

        expected_peer_mappings = 2 * (fa.world_size - 1)
        assert len(fa._open_peer_ptrs) == expected_peer_mappings
        fa.prepare_for_suspend()
        assert not fa._open_peer_ptrs
        fa.prepare_for_suspend()
        with pytest.raises(RuntimeError, match="detached"):
            tensor_model_parallel_all_reduce(inp1)
        fa.reinit_after_resume()
        assert len(fa._open_peer_ptrs) == expected_peer_mappings
        fa.reinit_after_resume()

        for replay_idx in range(2):
            inp1.fill_(fa.rank + 3 + replay_idx)
            inp2.fill_(fa.rank + 4 + replay_idx)
            graph.replay()
            torch.accelerator.synchronize()
            torch.testing.assert_close(
                out1,
                torch.full_like(
                    out1, sum(range(3 + replay_idx, tp_size + 3 + replay_idx))
                ),
            )
            torch.testing.assert_close(out1_repeat, out1)
            torch.testing.assert_close(
                out2,
                torch.full_like(
                    out2, sum(range(4 + replay_idx, tp_size + 4 + replay_idx))
                ),
            )

        selected_before_eager = selected_calls
        eager_out = tensor_model_parallel_all_reduce(inp1)
        assert selected_calls == selected_before_eager + 1
        torch.testing.assert_close(eager_out, out1)
        fa.close()
        fa.close()


@pytest.mark.parametrize("tp_size", [2])
@pytest.mark.parametrize("pipeline_parallel_size", [1, 2])
@pytest.mark.parametrize(
    "test_target",
    [
        eager_allreduce,
        graph_allreduce,
        pytest.param(
            graph_allreduce_suspend_resume,
            marks=pytest.mark.skipif(
                torch.version.hip is not None,
                reason="Custom allreduce suspend/resume requires CUDA.",
            ),
        ),
    ],
)
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
