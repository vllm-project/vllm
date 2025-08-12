import os
import random

import pytest
import ray
import torch
import torch.distributed as dist

from vllm.distributed.communication_op import (  # noqa
    graph_capture, tensor_model_parallel_all_reduce)
from vllm.distributed.parallel_state import (get_tensor_model_parallel_group,
                                             get_tp_ca_communicator)

from ..utils import (init_test_distributed_environment,
                     multi_process_tensor_parallel)

random.seed(42)
test_sizes = [random.randint(1024, 2048 * 1024) for _ in range(8)]
for i, v in enumerate(test_sizes):
    test_sizes[i] -= v % 8


@ray.remote(num_gpus=1, max_calls=1)
def graph_allreduce(tp_size, pp_size, rank, distributed_init_port):
    del os.environ["CUDA_VISIBLE_DEVICES"]
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    init_test_distributed_environment(tp_size, pp_size, rank,
                                      distributed_init_port)

    group = get_tensor_model_parallel_group()

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
        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            with graph_capture():
                # use integers so result matches NCCL exactly
                inp1 = torch.randint(1,
                                     16, (sz, ),
                                     dtype=dtype,
                                     device=torch.cuda.current_device())
                inp2 = torch.randint(1,
                                     16, (sz, ),
                                     dtype=dtype,
                                     device=torch.cuda.current_device())
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for i in range(num_communication):
                        out1 = tensor_model_parallel_all_reduce(inp1)
                        # the input buffer is immediately modified to test
                        # synchronization
                        dist.all_reduce(inp1, group=group)
                        out2 = tensor_model_parallel_all_reduce(inp2)
                        dist.all_reduce(inp2, group=group)
            graph.replay()
            assert torch.allclose(out1, inp1)
            assert torch.allclose(out2, inp2)


@ray.remote(num_gpus=1, max_calls=1)
def eager_allreduce(tp_size, pp_size, rank, distributed_init_port):
    del os.environ["CUDA_VISIBLE_DEVICES"]
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    init_test_distributed_environment(tp_size, pp_size, rank,
                                      distributed_init_port)

    # we use the first group to communicate once
    # and the second group to communicate twice
    # and so on
    # this is used to demonstrate that each group can
    # communicate independently
    num_communication = rank // tp_size + 1
    sz = 1024
    fa = get_tp_ca_communicator()
    inp = torch.ones(sz, dtype=torch.float32, device=device)
    out = inp
    for _ in range(num_communication):
        out = fa.all_reduce_unreg(out)
    assert torch.allclose(out, inp * (tp_size**num_communication))

    inp = torch.ones(sz * 4, dtype=torch.bfloat16, device=device)
    out = inp
    for _ in range(num_communication):
        out = fa.all_reduce_unreg(out)
    assert torch.allclose(out, inp * (tp_size**num_communication))


@pytest.mark.parametrize("tp_size", [2])
@pytest.mark.parametrize("pipeline_parallel_size", [1, 2])
@pytest.mark.parametrize("test_target", [eager_allreduce, graph_allreduce])
def test_custom_allreduce(tp_size, pipeline_parallel_size, test_target):
    world_size = tp_size * pipeline_parallel_size
    if world_size > torch.cuda.device_count():
        pytest.skip("Not enough GPUs to run the test.")
    multi_process_tensor_parallel(tp_size, pipeline_parallel_size, test_target)
