import multiprocessing

import torch.distributed as dist

from vllm.distributed.device_communicators.shm_broadcast import ShmRingBuffer
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
        dist.init_process_group(backend="gloo")
        fn()

    return wrapped_fn


@worker_fn_wrapper
def worker_fn():
    broadcaster = ShmRingBuffer(dist.group.WORLD, 1024, 1)
    if dist.get_rank() == 0:
        broadcaster.broadcast_object(0)
        broadcaster.broadcast_object(1)
    else:
        a = broadcaster.broadcast_object(None)
        b = broadcaster.broadcast_object(None)
        assert a == 0
        assert b == 1


def test_shm_broadcast():
    distributed_run(worker_fn, 4)
