# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import random
import threading
import time
from unittest import mock

import multiprocess as mp
import numpy as np
import pytest
import torch.distributed as dist

from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
from vllm.distributed.utils import StatelessProcessGroup
from vllm.utils.network_utils import get_open_port
from vllm.utils.system_utils import update_environment_variables


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
        env["RANK"] = str(i)
        env["LOCAL_RANK"] = str(i)
        env["WORLD_SIZE"] = str(number_of_processes)
        env["LOCAL_WORLD_SIZE"] = str(number_of_processes)
        env["MASTER_ADDR"] = "localhost"
        env["MASTER_PORT"] = "12345"
        p = mp.Process(target=fn, args=(env,))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    for p in processes:
        assert p.exitcode == 0


def worker_fn_wrapper(fn):
    # `mp.Process` cannot accept environment variables directly
    # so we need to pass the environment variables as arguments
    # and update the environment variables in the function
    def wrapped_fn(env):
        update_environment_variables(env)
        dist.init_process_group(backend="gloo")
        fn()

    return wrapped_fn


@worker_fn_wrapper
def worker_fn():
    rank = dist.get_rank()
    if rank == 0:
        port = get_open_port()
        ip = "127.0.0.1"
        dist.broadcast_object_list([ip, port], src=0)
    else:
        recv = [None, None]
        dist.broadcast_object_list(recv, src=0)
        ip, port = recv  # type: ignore

    stateless_pg = StatelessProcessGroup.create(ip, port, rank, dist.get_world_size())

    for pg in [dist.group.WORLD, stateless_pg]:
        writer_rank = 2
        broadcaster = MessageQueue.create_from_process_group(
            pg, 40 * 1024, 2, writer_rank
        )
        if rank == writer_rank:
            seed = random.randint(0, 1000)
            dist.broadcast_object_list([seed], writer_rank)
        else:
            recv = [None]
            dist.broadcast_object_list(recv, writer_rank)
            seed = recv[0]  # type: ignore

        if pg == dist.group.WORLD:
            dist.barrier()
        else:
            pg.barrier()

        # in case we find a race condition
        # print the seed so that we can reproduce the error
        print(f"Rank {rank} got seed {seed}")
        # test broadcasting with about 400MB of data
        N = 10_000
        if rank == writer_rank:
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

        if pg == dist.group.WORLD:
            dist.barrier()
            print(f"torch distributed passed the test! Rank {rank}")
        else:
            pg.barrier()
            print(f"StatelessProcessGroup passed the test! Rank {rank}")


def test_shm_broadcast():
    distributed_run(worker_fn, 4)


@worker_fn_wrapper
def worker_fn_test_shutdown():
    rank = dist.get_rank()
    writer_rank = 2
    message_queue = MessageQueue.create_from_process_group(
        dist.group.WORLD, 40 * 1024, 2, writer_rank
    )

    if not message_queue._is_writer:
        # Put into idle mode
        message_queue._spin_condition.last_read = 0

        shutdown_event = threading.Event()

        def shutdown_thread(mq, shutdown_event):
            shutdown_event.wait()
            mq.shutdown()

        threading.Thread(
            target=shutdown_thread, args=(message_queue, shutdown_event)
        ).start()

        with pytest.raises(TimeoutError):
            message_queue.dequeue(timeout=0.01)

        shutdown_event.set()

        with pytest.raises(RuntimeError, match="cancelled"):
            message_queue.dequeue(timeout=1)

        assert message_queue.shutting_down

    print(f"torch distributed passed the test! Rank {rank}")
    dist.barrier()


def test_message_queue_shutdown():
    distributed_run(worker_fn_test_shutdown, 4)


@worker_fn_wrapper
def worker_fn_test_idle_to_busy():
    rank = dist.get_rank()
    writer_rank = 2
    message_queue = MessageQueue.create_from_process_group(
        dist.group.WORLD, 40 * 1024, 2, writer_rank
    )

    message1 = "hello world"
    message2 = np.random.randint(1, 100, 100)
    with mock.patch.object(
        message_queue._spin_condition, "wait", wraps=message_queue._spin_condition.wait
    ) as wrapped_wait:
        if not message_queue._is_writer:
            # Put into idle mode
            message_queue._spin_condition.last_read = 0

            # no messages, so expect a TimeoutError
            with pytest.raises(TimeoutError):
                message_queue.dequeue(timeout=0.01)
            # wait should only be called once while idle
            assert wrapped_wait.call_count == 1

            # sync with the writer and wait for message1
            dist.barrier()
            recv_message = message_queue.dequeue(timeout=5)
            assert recv_message == message1
            # second call to wait, with a message read, this puts in a busy spin
            assert wrapped_wait.call_count == 2

            # sync with the writer and wait for message2
            dist.barrier()
            recv_message = message_queue.dequeue(timeout=1)
            assert np.array_equal(recv_message, message2)
            # in busy mode, we expect wait to have been called multiple times
            assert wrapped_wait.call_count > 3
        else:
            # writer writes two messages in sync with the reader
            dist.barrier()
            # sleep delays the send to ensure reader enters the read loop
            time.sleep(0.1)
            message_queue.enqueue(message1)

            dist.barrier()
            time.sleep(0.1)
            message_queue.enqueue(message2)

    message_queue.shutdown()
    assert message_queue.shutting_down
    print(f"torch distributed passed the test! Rank {rank}")


def test_message_queue_idle_wake():
    distributed_run(worker_fn_test_idle_to_busy, 4)


@worker_fn_wrapper
def worker_fn_test_busy_to_idle():
    rank = dist.get_rank()
    writer_rank = 2
    message_queue = MessageQueue.create_from_process_group(
        dist.group.WORLD, 40 * 1024, 2, writer_rank
    )

    message1 = 12345
    message2 = list(range(3))
    with mock.patch.object(
        message_queue._spin_condition, "wait", wraps=message_queue._spin_condition.wait
    ) as wrapped_wait:
        if not message_queue._is_writer:
            # Put into busy mode
            message_queue._spin_condition.busy_loop_s = 9999

            # sync with the writer and wait for message1
            dist.barrier()
            recv_message = message_queue.dequeue(timeout=1)
            assert recv_message == message1
            # in busy mode, we expect wait to have been called many times
            assert wrapped_wait.call_count > 1

            # simulate busy loop ending
            message_queue._spin_condition.busy_loop_s = 0
            # ensure we enter idle mode, then record call count
            with pytest.raises(TimeoutError):
                message_queue.dequeue(timeout=0.01)
            call_count = wrapped_wait.call_count

            # sync with the writer and wait for message2
            dist.barrier()
            recv_message = message_queue.dequeue(timeout=1)
            assert recv_message == message2

            # call to wait after idle should only happen once
            assert wrapped_wait.call_count == call_count + 1
        else:
            # writer writes two messages in sync with the reader
            dist.barrier()
            # sleep delays the send to ensure reader enters the read loop
            time.sleep(0.1)
            message_queue.enqueue(message1)

            dist.barrier()
            time.sleep(0.1)
            message_queue.enqueue(message2)

    message_queue.shutdown()
    assert message_queue.shutting_down
    print(f"torch distributed passed the test! Rank {rank}")


def test_message_queue_busy_to_idle():
    distributed_run(worker_fn_test_busy_to_idle, 4)
