import torch
import random
import torch.distributed as dist
from vllm.model_executor.parallel_utils.fast_allreduce import FastAllreduce
import pytest

from tests.distributed.comm_utils import init_test_distributed_environment, multi_process_tensor_parallel

random.seed(42)
test_sizes = [random.randint(1024, 2048 * 1024) for i in range(4)]
for i, v in enumerate(test_sizes):
    test_sizes[i] -= v % 8


def graph_registration(world_size, rank, distributed_init_port):
    init_test_distributed_environment(1, world_size, rank,
                                      distributed_init_port)
    for sz in test_sizes:
        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            fa = FastAllreduce(rank, world_size)
            # use integers so result matches NCCL exactly
            inp1 = torch.ones(
                sz, dtype=dtype,
                device=torch.cuda.current_device()) * random.randint(1, 16)
            inp2 = torch.ones(
                sz, dtype=dtype,
                device=torch.cuda.current_device()) * random.randint(1, 16)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out1 = fa.all_reduce(inp1)
                out2 = fa.all_reduce(inp2)
                # the input buffer is immediately modified to test synchronization
                dist.all_reduce(inp1)
                dist.all_reduce(inp2)
            fa.register_graph_buffers()
            graph.replay()
            torch.cuda.synchronize()

            assert torch.allclose(out1, inp1)
            assert torch.allclose(out2, inp2)


def manual_registration(world_size, rank, distributed_init_port):
    init_test_distributed_environment(1, world_size, rank,
                                      distributed_init_port)
    sz = 1024
    fa = FastAllreduce(rank, world_size)
    inp = torch.ones(sz,
                     dtype=torch.float32,
                     device=torch.cuda.current_device())
    fa.register_buffer(inp)
    out = fa.all_reduce(inp)
    assert torch.allclose(out, inp * world_size)


@pytest.mark.skipif(torch.cuda.device_count() < 4,
                    reason="Need at least 4 GPUs to run the test.")
@pytest.mark.parametrize("tensor_parallel_size", [2, 4])
@pytest.mark.parametrize("test_target",
                         [manual_registration, graph_registration])
def test_multi_process_tensor_parallel(tensor_parallel_size, test_target):
    multi_process_tensor_parallel(tensor_parallel_size, test_target)
