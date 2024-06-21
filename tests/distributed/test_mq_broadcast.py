import multiprocessing
import random
import time

import torch.distributed as dist

from vllm.distributed.device_communicators.msg_queue import (
    Publisher, PublishSubscribeMsgQueue)
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
    broadcaster = PublishSubscribeMsgQueue.create_from_process_group(
        dist.group.WORLD, writer_rank)
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


def test_mq_broadcast():
    distributed_run(worker_fn, 4)


def subscriber(n_reader, reader_rank, handle):
    queue = PublishSubscribeMsgQueue(n_reader, reader_rank, handle)
    queue.wait_for_ready()
    time.sleep(random.random())
    a = queue.broadcast_object(None)
    time.sleep(random.random())
    b = queue.broadcast_object(None)
    time.sleep(random.random())
    c = queue.broadcast_object(None)
    assert a == 0
    assert b == {}
    assert c == []


def test_simple_multiprocessing():
    n_reader = 3
    publisher = Publisher(n_reader)
    handle = publisher.handle
    queue = PublishSubscribeMsgQueue(n_reader, -1, publisher)
    context = multiprocessing.get_context("spawn")
    ps = []
    for i in range(n_reader):
        p = context.Process(target=subscriber, args=(n_reader, i, handle))
        p.start()
        ps.append(p)
    queue.wait_for_ready()
    time.sleep(random.random())
    queue.broadcast_object(0)
    time.sleep(random.random())
    queue.broadcast_object({})
    time.sleep(random.random())
    queue.broadcast_object([])
    for p in ps:
        p.join()
        assert p.exitcode == 0
