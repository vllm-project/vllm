import multiprocessing

import pytest
import torch
import torch.distributed as dist

from vllm.distributed.communication_op import (
    change_model_parallel_pynccl_allreduce, tensor_model_parallel_all_reduce)
from vllm.distributed.device_communicators.pynccl import NCCLCommunicator
from vllm.distributed.device_communicators.pynccl_wrapper import NCCLLibrary
from vllm.distributed.parallel_state import (destroy_model_parallel,
                                             init_distributed_environment,
                                             initialize_model_parallel)
from vllm.utils import update_environment_variables


def distributed_run(fn, world_size):
    number_of_processes = world_size
    processes = []
    for i in range(number_of_processes):
        env = {}
        env['RANK'] = str(i)
        env['LOCAL_RANK'] = str(i)
        env['WORLD_SIZE'] = str(number_of_processes)
        env['LOCAL_WORLD_SIZE'] = str(number_of_processes)
        env['MASTER_ADDR'] = 'localhost'
        env['MASTER_PORT'] = '12345'
        p = multiprocessing.Process(target=fn, args=(env, ))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    for p in processes:
        assert p.exitcode == 0


def worker_fn_wrapper(fn):
    # `multiprocessing.Process` cannot accept environment variables directly
    # so we need to pass the environment variables as arguments
    # and update the environment variables in the function
    def wrapped_fn(env):
        update_environment_variables(env)
        init_distributed_environment()
        fn()

    return wrapped_fn


@worker_fn_wrapper
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


@worker_fn_wrapper
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


@worker_fn_wrapper
def worker_fn_with_multiple_tp_groups():
    change_model_parallel_pynccl_allreduce(enable=True)
    initialize_model_parallel(2, 2)
    rank = dist.get_rank()

    a = torch.ones((4, 4), device=f'cuda:{rank}') * rank
    b = tensor_model_parallel_all_reduce(a)
    if rank in [0, 1]:
        assert b.mean().cpu().item() == 0 + 1
    else:
        assert b.mean().cpu().item() == 2 + 3

    destroy_model_parallel()


@pytest.mark.skipif(torch.cuda.device_count() < 4,
                    reason="Need at least 4 GPUs to run the test.")
def test_pynccl_with_multiple_tp_groups():
    distributed_run(worker_fn_with_multiple_tp_groups, 4)


def test_ncclGetUniqueId():
    library = NCCLLibrary()
    unique_id = library.ncclGetUniqueId()
    # `list(unique_id.internal)` is something like this:
    # [34, -16, 23, 83, 109, -19, 59, 95, 2, 0, -86, 55, 10, -128, 0, 29, 0,
    # 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    # 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    # 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    # 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    # 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # as long as the function doesn't raise an exception, we're good
    assert unique_id is not None
