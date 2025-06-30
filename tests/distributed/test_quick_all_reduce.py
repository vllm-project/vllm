# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import random

import pytest
import ray
import torch
import torch.distributed as dist

from vllm.distributed.communication_op import (  # noqa
    tensor_model_parallel_all_reduce)
from vllm.distributed.parallel_state import (get_tensor_model_parallel_group,
                                             get_tp_group, graph_capture)
from vllm.platforms import current_platform

from ..utils import (ensure_model_parallel_initialized,
                     init_test_distributed_environment, multi_process_parallel)

torch.manual_seed(42)
random.seed(44)
# Size over 8MB is sufficient for custom quick allreduce.
test_sizes = [
    random.randint(8 * 1024 * 1024, 10 * 1024 * 1024) for _ in range(8)
]
for i, v in enumerate(test_sizes):
    test_sizes[i] -= v % 8


@ray.remote(num_gpus=1, max_calls=1)
def graph_quickreduce(
    monkeypatch: pytest.MonkeyPatch,
    tp_size,
    pp_size,
    rank,
    distributed_init_port,
):
    with monkeypatch.context() as m:
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
        init_test_distributed_environment(tp_size, pp_size, rank,
                                          distributed_init_port)
        ensure_model_parallel_initialized(tp_size, pp_size)
        group = get_tensor_model_parallel_group().device_group

        # A small all_reduce for warmup.
        # this is needed because device communicators might be created lazily
        # (e.g. NCCL). This will ensure that the communicator is initialized
        # before any communication happens, so that this group can be used for
        # graph capture immediately.
        data = torch.zeros(1)
        data = data.to(device=device)
        torch.distributed.all_reduce(data, group=group)
        torch.cuda.synchronize()
        del data

        # we use the first group to communicate once
        # and the second group to communicate twice
        # and so on
        # this is used to demonstrate that each group can
        # communicate independently
        num_communication = rank // tp_size + 1

        for sz in test_sizes:
            for dtype in [torch.float16, torch.bfloat16]:
                with graph_capture(device=device) as graph_capture_context:
                    inp1 = torch.randint(1,
                                         23, (sz, ),
                                         dtype=dtype,
                                         device=torch.cuda.current_device())
                    inp2 = torch.randint(-23,
                                         1, (sz, ),
                                         dtype=dtype,
                                         device=torch.cuda.current_device())
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph,
                                          stream=graph_capture_context.stream):
                        for _ in range(num_communication):
                            out1 = tensor_model_parallel_all_reduce(inp1)
                            dist.all_reduce(inp1, group=group)
                            out2 = tensor_model_parallel_all_reduce(inp2)
                            dist.all_reduce(inp2, group=group)
                graph.replay()
                torch.testing.assert_close(out1, inp1, atol=2.5, rtol=0.1)
                torch.testing.assert_close(out2, inp2, atol=2.5, rtol=0.1)


@ray.remote(num_gpus=1, max_calls=1)
def eager_quickreduce(
    monkeypatch: pytest.MonkeyPatch,
    tp_size,
    pp_size,
    rank,
    distributed_init_port,
):
    with monkeypatch.context() as m:
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)

        init_test_distributed_environment(tp_size, pp_size, rank,
                                          distributed_init_port)

        # Size over 8MB is sufficient for custom quick allreduce.
        sz = 16 * 1024 * 1024
        fa = get_tp_group().device_communicator.qr_comm
        inp = torch.tensor([1.0 * ((i) % 23) for i in range(sz)],
                           dtype=torch.float16,
                           device=device)
        out = fa.quick_all_reduce(inp)
        torch.testing.assert_close(out, inp * tp_size, atol=2.5, rtol=0.1)

        inp = torch.tensor([1.0 * ((i) % 23) for i in range(sz)],
                           dtype=torch.bfloat16,
                           device=device)
        out = fa.quick_all_reduce(inp)
        torch.testing.assert_close(out, inp * tp_size, atol=2.5, rtol=0.1)


@pytest.mark.skipif(not current_platform.is_rocm(),
                    reason="only test quick allreduce for rocm")
@pytest.mark.parametrize("quant_mode", ["FP", "INT8", "INT6", "INT4"])
@pytest.mark.parametrize("tp_size", [2])
@pytest.mark.parametrize("pipeline_parallel_size", [1, 2])
@pytest.mark.parametrize("test_target", [graph_quickreduce, eager_quickreduce])
def test_custom_quick_allreduce(monkeypatch: pytest.MonkeyPatch, tp_size,
                                pipeline_parallel_size, test_target,
                                quant_mode):
    world_size = tp_size * pipeline_parallel_size
    if world_size > torch.cuda.device_count():
        pytest.skip("Not enough GPUs to run the test.")

    monkeypatch.setenv("VLLM_ROCM_QUICK_REDUCE_QUANTIZATION", quant_mode)

    multi_process_parallel(monkeypatch, tp_size, pipeline_parallel_size,
                           test_target)
