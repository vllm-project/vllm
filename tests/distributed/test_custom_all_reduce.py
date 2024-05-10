import os
import random

import pytest
import ray
import torch
import torch.distributed as dist

from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.distributed.device_communicators import custom_all_reduce
from vllm.test_utils import (init_test_distributed_environment,
                             multi_process_tensor_parallel)

random.seed(42)
test_sizes = [random.randint(1024, 2048 * 1024) for _ in range(8)]
for i, v in enumerate(test_sizes):
    test_sizes[i] -= v % 8


@ray.remote(num_gpus=1, max_calls=1)
def graph_allreduce(world_size, rank, distributed_init_port):
    del os.environ["CUDA_VISIBLE_DEVICES"]
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    init_test_distributed_environment(1, world_size, rank,
                                      distributed_init_port)

    custom_all_reduce.init_custom_ar()
    for sz in test_sizes:
        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            with custom_all_reduce.capture():
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
def eager_allreduce(world_size, rank, distributed_init_port):
    del os.environ["CUDA_VISIBLE_DEVICES"]
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    init_test_distributed_environment(1, world_size, rank,
                                      distributed_init_port)

    sz = 1024
    custom_all_reduce.init_custom_ar()
    fa = custom_all_reduce.get_handle()
    inp = torch.ones(sz, dtype=torch.float32, device=device)
    out = fa.all_reduce_unreg(inp)
    assert torch.allclose(out, inp * world_size)

    inp = torch.ones(sz * 4, dtype=torch.bfloat16, device=device)
    out = fa.all_reduce_unreg(inp)
    assert torch.allclose(out, inp * world_size)


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason="Need at least 2 GPUs to run the test.")
@pytest.mark.parametrize("tensor_parallel_size", [2])
@pytest.mark.parametrize("test_target", [eager_allreduce, graph_allreduce])
def test_multi_process_tensor_parallel(tensor_parallel_size, test_target):
    multi_process_tensor_parallel(tensor_parallel_size, test_target)


if __name__ == "__main__":
    multi_process_tensor_parallel(2, graph_allreduce)
