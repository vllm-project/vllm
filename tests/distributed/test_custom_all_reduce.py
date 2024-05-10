import os
import random

import pytest
import ray
import torch
import torch.distributed as dist

from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.distributed.communication_op import graph_capture_mode
from vllm.distributed.parallel_state import get_tp_ca_communicator
from vllm.test_utils import (init_test_distributed_environment,
                             multi_process_tensor_parallel)

random.seed(42)
test_sizes = [random.randint(1024, 2048 * 1024) for _ in range(8)]
for i, v in enumerate(test_sizes):
    test_sizes[i] -= v % 8


@ray.remote(num_gpus=1, max_calls=1)
def graph_allreduce(tp_size, pp_size, rank, distributed_init_port):
    del os.environ["CUDA_VISIBLE_DEVICES"]
    assert pp_size == 1
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    init_test_distributed_environment(pp_size, tp_size, rank,
                                      distributed_init_port)

    for sz in test_sizes:
        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            with graph_capture_mode():
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
                    out1 = tensor_model_parallel_all_reduce(inp1)
                    # the input buffer is immediately modified to test
                    # synchronization
                    dist.all_reduce(inp1)
                    out2 = tensor_model_parallel_all_reduce(inp2)
                    dist.all_reduce(inp2)
            graph.replay()
            assert torch.allclose(out1, inp1)
            assert torch.allclose(out2, inp2)


@ray.remote(num_gpus=1, max_calls=1)
def eager_allreduce(tp_size, pp_size, rank, distributed_init_port):
    del os.environ["CUDA_VISIBLE_DEVICES"]
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    init_test_distributed_environment(pp_size, tp_size, rank,
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


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason="Need at least 2 GPUs to run the test.")
@pytest.mark.parametrize("tensor_parallel_size", [2])
@pytest.mark.parametrize("test_target", [eager_allreduce, graph_allreduce])
def test_multi_process_tensor_parallel(tensor_parallel_size, test_target):
    multi_process_tensor_parallel(tensor_parallel_size, 1, test_target)


@pytest.mark.skipif(torch.cuda.device_count() < 4,
                    reason="Need at least 4 GPUs to run the test.")
@pytest.mark.parametrize("tensor_parallel_size", [2])
@pytest.mark.parametrize("pipeline_parallel_size", [2])
@pytest.mark.parametrize("test_target", [eager_allreduce])
def test_custom_allreduce_multiple_groups(tensor_parallel_size,
                                          pipeline_parallel_size, test_target):
    multi_process_tensor_parallel(tensor_parallel_size, pipeline_parallel_size,
                                  test_target)
