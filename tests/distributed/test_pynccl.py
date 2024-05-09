import multiprocessing
import os

import pytest
import torch

from vllm.distributed.communication_op import (  # noqa
    graph_capture_mode, tensor_model_parallel_all_reduce)
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.device_communicators.pynccl_wrapper import NCCLLibrary
from vllm.distributed.parallel_state import (ensure_model_parallel_initialized,
                                             init_distributed_environment)
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
        local_rank = os.environ['LOCAL_RANK']
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        init_distributed_environment()
        fn()

    return wrapped_fn


@worker_fn_wrapper
def worker_fn():
    pynccl_comm = PyNcclCommunicator()
    tensor = torch.ones(16, 1024, 1024,
                        dtype=torch.float32).cuda(pynccl_comm.rank)
    with pynccl_comm.change_state(enable=True):
        pynccl_comm.all_reduce(tensor)
    result = tensor.mean().cpu().item()
    assert result == pynccl_comm.world_size


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason="Need at least 2 GPUs to run the test.")
def test_pynccl():
    distributed_run(worker_fn, 2)


@worker_fn_wrapper
def multiple_tp_worker_fn():
    device = torch.device(f"cuda:{torch.distributed.get_rank()}")
    groups = [
        torch.distributed.new_group(ranks=[0, 1], backend="gloo"),
        torch.distributed.new_group(ranks=[2, 3], backend="gloo")
    ]
    group = groups[0] if torch.distributed.get_rank() in [0, 1] else groups[1]
    pynccl_comm = PyNcclCommunicator(group=group, device=device)
    tensor = torch.ones(16, 1024, 1024, dtype=torch.float32, device=device)
    with pynccl_comm.change_state(enable=True):
        # two groups can communicate independently
        if torch.distributed.get_rank() in [0, 1]:
            pynccl_comm.all_reduce(tensor)
            pynccl_comm.all_reduce(tensor)
            result = tensor.mean().cpu().item()
            assert result == 4
        else:
            pynccl_comm.all_reduce(tensor)
            result = tensor.mean().cpu().item()
            assert result == 2


@pytest.mark.skipif(torch.cuda.device_count() < 4,
                    reason="Need at least 4 GPUs to run the test.")
def test_pynccl_multiple_tp():
    # this tests pynccl for multiple tp groups, in a standalone way
    # i.e. call `pynccl_comm.all_reduce` directly
    distributed_run(multiple_tp_worker_fn, 4)


@worker_fn_wrapper
def multiple_tp_with_vllm_worker_fn():
    device = torch.device(f"cuda:{torch.distributed.get_rank()}")
    ensure_model_parallel_initialized(2, 2)
    tensor = torch.ones(16, 1024, 1024, dtype=torch.float32, device=device)
    with graph_capture_mode():
        # two tp groups can communicate independently
        if torch.distributed.get_rank() in [0, 1]:
            tensor = tensor_model_parallel_all_reduce(tensor)
            tensor = tensor_model_parallel_all_reduce(tensor)
            result = tensor.mean().cpu().item()
            assert result == 4
        else:
            tensor = tensor_model_parallel_all_reduce(tensor)
            result = tensor.mean().cpu().item()
            assert result == 2


@pytest.mark.skipif(torch.cuda.device_count() < 4,
                    reason="Need at least 4 GPUs to run the test.")
def test_pynccl_multiple_tp_with_vllm():
    # this tests pynccl for multiple tp groups, together with vllm
    # i.e. call `tensor_model_parallel_all_reduce`
    distributed_run(multiple_tp_with_vllm_worker_fn, 4)


@worker_fn_wrapper
def worker_fn_with_cudagraph():
    with torch.no_grad():
        graph = torch.cuda.CUDAGraph()
        pynccl_comm = PyNcclCommunicator()
        # run something in the default stream to initialize torch engine
        a = torch.ones((4, 4), device=f'cuda:{pynccl_comm.rank}')
        torch.cuda.synchronize()
        with torch.cuda.graph(
                graph, stream=pynccl_comm.stream), pynccl_comm.change_state(
                    enable=True):
            # operation during the graph capture is recorded but not executed
            # see https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#creating-a-graph-using-stream-capture # noqa
            pynccl_comm.all_reduce(a)
        pynccl_comm.stream.synchronize()
        assert a.mean().cpu().item() == pynccl_comm.world_size**0
        graph.replay()
        pynccl_comm.stream.synchronize()
        assert a.mean().cpu().item() == pynccl_comm.world_size**1


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason="Need at least 2 GPUs to run the test.")
def test_pynccl_with_cudagraph():
    distributed_run(worker_fn_with_cudagraph, 2)


def test_ncclGetUniqueId():
    lib = NCCLLibrary()
    unique_id = lib.ncclGetUniqueId()
    # `list(unique_id.internal)` is something like this:
    # [34, -16, 23, 83, 109, -19, 59, 95, 2, 0, -86, 55, 10, -128, 0, 29, 0,
    # 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    # 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    # 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    # 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    # 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # as long as the function doesn't raise an exception, we're good
    assert unique_id is not None
