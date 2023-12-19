"""
Run this test like this:
torchrun --standalone --nnodes=1 --nproc-per-node=4 tests/kernels/test_fast_ar.py
"""
import torch
import os
import random
import torch.distributed as dist
from vllm.model_executor.parallel_utils.fast_allreduce import FastAllreduce
from vllm.model_executor.parallel_utils.parallel_state import initialize_model_parallel

os.environ["MAX_JOBS"] = "16"
rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])
dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
torch.cuda.set_device(local_rank)
initialize_model_parallel(world_size)

test_count = 8
if rank == 0:
    test_sizes = [random.randint(1024, 2048 * 1024) for i in range(test_count)]
    for i, v in enumerate(test_sizes):
        test_sizes[i] -= v % 8
else:
    test_sizes = [0] * test_count
dist.broadcast_object_list(test_sizes, src=0)


def test_fast_ar(sz: int, dtype):
    fa = FastAllreduce(rank, world_size)
    # use integers so result matches NCCL exactly
    inp1 = torch.ones(sz, dtype=dtype,
                      device=torch.cuda.current_device()) * random.randint(
                          1, 32)
    inp2 = torch.ones(sz, dtype=dtype,
                      device=torch.cuda.current_device()) * random.randint(
                          1, 32)
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
    if rank == 0:
        print("passed", sz, dtype)


def test_manual_registration():
    sz = 1024
    fa = FastAllreduce(rank, world_size)
    inp = torch.ones(sz,
                     dtype=torch.float32,
                     device=torch.cuda.current_device())
    fa.register_buffer(inp)
    out = fa.all_reduce(inp)
    assert torch.allclose(out, inp * world_size)


if __name__ == "__main__":
    print(test_sizes)
    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        for sz in test_sizes:
            test_fast_ar(sz, dtype)
    test_manual_registration()
