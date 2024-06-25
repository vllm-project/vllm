import multiprocessing
import random
import time

import torch.distributed as dist

from vllm.distributed.device_communicators.shm_broadcast import (
    ShmRingBuffer, ShmRingBufferIO)
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
    writer_rank = 2
    broadcaster = ShmRingBufferIO.create_from_process_group(
        dist.group.WORLD, 1024, 2, writer_rank)
    if dist.get_rank() == writer_rank:
        time.sleep(random.random())
        broadcaster.broadcast_object(0)
        time.sleep(random.random())
        broadcaster.broadcast_object({})
        time.sleep(random.random())
        broadcaster.broadcast_object([])
    else:
        time.sleep(random.random())
        a = broadcaster.broadcast_object(None)
        time.sleep(random.random())
        b = broadcaster.broadcast_object(None)
        time.sleep(random.random())
        c = broadcaster.broadcast_object(None)
        assert a == 0
        assert b == {}
        assert c == []
    dist.barrier()


def test_shm_broadcast():
    distributed_run(worker_fn, 4)


def test_singe_process():
    buffer = ShmRingBuffer(1, 1024, 4)
    reader = ShmRingBufferIO(buffer, reader_rank=0)
    writer = ShmRingBufferIO(buffer, reader_rank=-1)
    writer.enqueue([0])
    writer.enqueue([1])
    assert reader.dequeue() == [0]
    assert reader.dequeue() == [1]
