# SPDX-License-Identifier: Apache-2.0

import multiprocessing

import numpy as np
import torch.distributed as dist

from vllm.distributed.parallel_state import get_pp_group
from vllm.utils import get_ip, update_environment_variables

from ..utils import init_test_distributed_environment


def get_arrays(n: int, seed: int = 0) -> list[np.ndarray]:
    np.random.seed(seed)
    sizes = np.random.randint(1, 10_000, n)
    # on average, each array will have 5k elements
    # with int64, each array will have 40kb
    return [np.random.randint(1, 100, i) for i in sizes]


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

    def wrapped_fn(env):
        update_environment_variables(env)
        init_test_distributed_environment(1, 2, int(env['RANK']), "12345")
        fn()

    return wrapped_fn


@worker_fn_wrapper
def worker_fn():

    rank = dist.get_rank()
    ip = get_ip()
    port = 1345
    pp_group = get_pp_group()
    if rank == 1:
        broadcast_object = {"ip": ip, "port": port}
        pp_group.broadcast_object_async(broadcast_object, src=1)
    else:
        recv = pp_group.broadcast_object_async(None, src=1)
        result = recv.result()
        assert result["ip"] == ip, f"Expected {ip}, got {result['ip']}"
        assert result["port"] == port, f"Expected {port}, got {result['port']}"


def test_broadcast_async():
    distributed_run(worker_fn, 2)
