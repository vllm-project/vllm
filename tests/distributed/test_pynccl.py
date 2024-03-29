import multiprocessing
import os

import pytest
import torch

from vllm.model_executor.parallel_utils.pynccl import (NCCLCommunicator,
                                                       ncclGetUniqueId)


def distributed_run(fn, world_size):
    number_of_processes = world_size
    processes = []
    for i in range(number_of_processes):
        env = os.environ.copy()
        env['RANK'] = str(i)
        env['WORLD_SIZE'] = str(number_of_processes)
        env['MASTER_ADDR'] = 'localhost'
        env['MASTER_PORT'] = '12345'
        p = multiprocessing.Process(target=fn, args=(env, ))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()


def update_env(fn):
    # `multiprocessing.Process` cannot accept environment variables directly
    # so we need to pass the environment variables as arguments
    # and update the environment variables in the function
    def wrapper(env):
        import os
        os.environ.update(env)
        fn()

    return wrapper


@update_env
def worker_fn():
    comm = NCCLCommunicator()
    tensor = torch.ones(16, 1024, 1024, dtype=torch.float32).cuda(comm.rank)
    comm.all_reduce(tensor)
    result = tensor.mean().cpu().item()
    assert result == comm.world_size


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason="Need at least 2 GPUs to run the test.")
def test_pynccl():
    distributed_run(worker_fn, 2)


@update_env
def worker_fn_with_cudagraph():
    with torch.no_grad():
        graph = torch.cuda.CUDAGraph()
        comm = NCCLCommunicator()
        # run something in the default stream to initialize torch engine
        a = torch.ones((4, 4), device=f'cuda:{comm.rank}')
        torch.cuda.synchronize()
        with torch.cuda.graph(graph, stream=comm.stream):
            # operation during the graph capture is recorded but not executed
            # see https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#creating-a-graph-using-stream-capture # noqa
            comm.all_reduce(a)
        comm.stream.synchronize()
        assert a.mean().cpu().item() == comm.world_size**0
        graph.replay()
        comm.stream.synchronize()
        assert a.mean().cpu().item() == comm.world_size**1


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason="Need at least 2 GPUs to run the test.")
def test_pynccl_with_cudagraph():
    distributed_run(worker_fn_with_cudagraph, 2)


def test_ncclGetUniqueId():
    unique_id = ncclGetUniqueId()
    # `list(unique_id.internal)` is something like this:
    # [34, -16, 23, 83, 109, -19, 59, 95, 2, 0, -86, 55, 10, -128, 0, 29, 0,
    # 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    # 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    # 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    # 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    # 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # as long as the function doesn't raise an exception, we're good
    assert unique_id is not None
