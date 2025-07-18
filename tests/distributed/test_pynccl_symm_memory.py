# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import multiprocessing
import os

import numpy as np
import pytest
import torch
import torch.distributed

from vllm.distributed.communication_op import (  # noqa
    tensor_model_parallel_all_reduce,
)
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.device_communicators.pynccl_wrapper import NCCLLibrary
from vllm.distributed.device_communicators.pynccl_allocator import (
    get_nccl_mem_pool,
)

from vllm.distributed.parallel_state import (
    ensure_model_parallel_initialized,
    get_world_group,
    graph_capture,
    init_distributed_environment,
)
from vllm.utils import update_environment_variables


def distributed_run(fn, world_size):
    number_of_processes = world_size
    processes: list[multiprocessing.Process] = []
    for i in range(number_of_processes):
        env: dict[str, str] = {}
        env["RANK"] = str(i)
        env["LOCAL_RANK"] = str(i)
        env["WORLD_SIZE"] = str(number_of_processes)
        env["LOCAL_WORLD_SIZE"] = str(number_of_processes)
        env["MASTER_ADDR"] = "localhost"
        env["MASTER_PORT"] = "12345"
        p = multiprocessing.Process(target=fn, args=(env,))
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
        local_rank = os.environ["LOCAL_RANK"]
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        init_distributed_environment()
        fn()

    return wrapped_fn


@worker_fn_wrapper
def multiple_allreduce_worker_fn():
    device = torch.device(f"cuda:{torch.distributed.get_rank()}")
    groups = [
        torch.distributed.new_group(ranks=[0, 1], backend="gloo"),
        torch.distributed.new_group(ranks=[2, 3], backend="gloo"),
    ]
    group = groups[0] if torch.distributed.get_rank() in [0, 1] else groups[1]
    pynccl_comm = PyNcclCommunicator(group=group, device=device)
    with torch.cuda.use_mem_pool(get_nccl_mem_pool()):
        symm_tensor = torch.ones(
            16, 1024, 1024, dtype=torch.float32, device=device
        )
    win = pynccl_comm.register_comm_window(symm_tensor)
    stream = torch.cuda.default_stream()
    # two groups can communicate independently
    if torch.distributed.get_rank() in [0, 1]:
        tensor = pynccl_comm.all_reduce(symm_tensor, stream=stream)
        tensor = pynccl_comm.all_reduce(symm_tensor, stream=stream)
        torch.cuda.synchronize()
        assert torch.all(tensor == 4).cpu().item()
    else:
        tensor = pynccl_comm.all_reduce(symm_tensor, stream=stream)
        torch.cuda.synchronize()
        assert torch.all(tensor == 2).cpu().item()
    pynccl_comm.deregister_comm_window(win)
    


@pytest.mark.skipif(
    torch.cuda.device_count() < 4,
    reason="Need at least 4 GPUs to run the test.",
)
def test_pynccl_multiple_allreduce():
    # this tests pynccl for multiple tp groups, in a standalone way
    # i.e. call `pynccl_comm.all_reduce` directly
    distributed_run(multiple_allreduce_worker_fn, 4)
