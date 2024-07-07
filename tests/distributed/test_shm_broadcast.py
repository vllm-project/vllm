import multiprocessing
import random
import time
from typing import List

import numpy as np
import torch.distributed as dist

from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
from vllm.utils import update_environment_variables


def get_arrays(n: int, seed: int = 0) -> List[np.ndarray]:
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
    # `multiprocessing.Process` cannot accept environment variables directly
    # so we need to pass the environment variables as arguments
    # and update the environment variables in the function
    def wrapped_fn(env):
        update_environment_variables(env)
        dist.init_process_group(backend="gloo")
        fn()

    return wrapped_fn


@worker_fn_wrapper
def worker_fn():
    writer_rank = 2
    broadcaster = MessageQueue.create_from_process_group(
        dist.group.WORLD, 40 * 1024, 2, writer_rank)
    if dist.get_rank() == writer_rank:
        seed = random.randint(0, 1000)
        dist.broadcast_object_list([seed], writer_rank)
    else:
        recv = [None]
        dist.broadcast_object_list(recv, writer_rank)
        seed = recv[0]  # type: ignore
    dist.barrier()
    # in case we find a race condition
    # print the seed so that we can reproduce the error
    print(f"Rank {dist.get_rank()} got seed {seed}")
    # test broadcasting with about 400MB of data
    N = 10_000
    if dist.get_rank() == writer_rank:
        arrs = get_arrays(N, seed)
        for x in arrs:
            broadcaster.broadcast_object(x)
            time.sleep(random.random() / 1000)
    else:
        arrs = get_arrays(N, seed)
        for x in arrs:
            y = broadcaster.broadcast_object(None)
            assert np.array_equal(x, y)
            time.sleep(random.random() / 1000)
    dist.barrier()


def test_shm_broadcast():
    distributed_run(worker_fn, 4)
