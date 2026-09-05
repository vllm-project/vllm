# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import io
import pickle
import random
import threading
import time
from types import SimpleNamespace
from unittest import mock

import multiprocess as mp
import numpy as np
import pytest
import torch
import torch.distributed as dist

from vllm.distributed.device_communicators import shm_broadcast
from vllm.distributed.device_communicators.shm_broadcast import (
    MessageQueue,
    ShmRingBuffer,
    _rebuild_tensor,
    _reduce_tensor,
    check_shm_free_space,
)
from vllm.distributed.utils import StatelessProcessGroup
from vllm.utils.network_utils import get_open_port
from vllm.utils.system_utils import update_environment_variables


def get_arrays(n: int, seed: int = 0) -> list[np.ndarray]:
    np.random.seed(seed)
    sizes = np.random.randint(1, 10_000, n)
    # on average, each array will have 5k elements
    # with int64, each array will have 40kb
    return [np.random.randint(1, 100, i) for i in sizes]


def distributed_run(fn, world_size, timeout=60):
    """Run a function in multiple processes with proper error handling.

    Args:
        fn: Function to run in each process
        world_size: Number of processes to spawn
        timeout: Maximum time in seconds to wait for processes (default: 60)
    """
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

    # Monitor processes and fail fast if any process fails
    start_time = time.time()
    failed_processes = []

    # Wait for all processes, checking for failures
    while time.time() - start_time < timeout:
        all_done = True
        for i, p in enumerate(processes):
            if p.is_alive():
                all_done = False
            elif p.exitcode != 0:
                # Process failed
                failed_processes.append((i, p.exitcode))
                break

        if failed_processes or all_done:
            break
        time.sleep(0.1)  # Check every 100ms

    # Check for timeout if no failures detected yet
    for i, p in enumerate(processes):
        if p.is_alive():
            p.kill()
            p.join()

    # Report failures
    if failed_processes:
        error_msg = "Distributed test failed:\n"
        for rank, status in failed_processes:
            error_msg += f"  Rank {rank}: Exit code {status}\n"
        raise AssertionError(error_msg)


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
def worker_fn_test_shutdown_busy():
    rank = dist.get_rank()
    writer_rank = 2
    message_queue = MessageQueue.create_from_process_group(
        dist.group.WORLD, 40 * 1024, 2, writer_rank
    )

    if not message_queue._is_writer:
        # Put into busy mode
        message_queue._spin_condition.busy_loop_s = 9999

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


def test_message_queue_shutdown_busy(caplog_vllm):
    distributed_run(worker_fn_test_shutdown_busy, 4)
    print(caplog_vllm.text)


@worker_fn_wrapper
def worker_fn_test_shutdown_idle():
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


def test_message_queue_shutdown_idle():
    distributed_run(worker_fn_test_shutdown_idle, 4)


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


@worker_fn_wrapper
def worker_fn_tensor_broadcast():
    rank = dist.get_rank()
    writer_rank = 0
    message_queue = MessageQueue.create_from_process_group(
        dist.group.WORLD, 8 * 1024 * 1024, 4, writer_rank
    )

    # Both ranks construct the identical reference payload.
    torch.manual_seed(42)
    payload = {
        # 2MiB: rides the shm ring as an out-of-band buffer (the receiving
        # side must copy out of the reusable ring chunk).
        "mid": torch.randn(1024, 512),
        # 16MiB > max_chunk_bytes: overflows to the zmq socket (the
        # receiving side aliases the zmq.Frame zero-copy).
        "big": torch.randn(4096, 2048, dtype=torch.bfloat16),
        "nested": ["plain", 123, {"inner": torch.arange(5)}],
    }

    if rank == writer_rank:
        with mock.patch(
            "vllm.distributed.device_communicators.shm_broadcast._reduce_tensor",
            wraps=_reduce_tensor,
        ) as wrapped_reduce:
            message_queue.enqueue(payload)
        assert wrapped_reduce.call_count == 3
        # Cycle the ring (max_chunks=4) several times over so that aliased
        # ring chunks would be overwritten.
        for i in range(16):
            message_queue.enqueue({"junk": torch.full((1024, 512), float(i))})
    else:
        received = message_queue.dequeue(timeout=30)
        for key in ("mid", "big"):
            assert torch.equal(received[key], payload[key]), key
            assert received[key].dtype == payload[key].dtype, key
        assert torch.equal(received["nested"][2]["inner"], torch.arange(5))

        snapshot = received["mid"].clone()
        for i in range(16):
            junk = message_queue.dequeue(timeout=30)
            assert torch.equal(junk["junk"], torch.full((1024, 512), float(i)))
        # Tensors received via the shm ring must not alias chunk memory
        # that the writer has reused for subsequent messages.
        assert torch.equal(received["mid"], snapshot)
        # Rebuilt tensors must be writable, like regular tensors.
        received["mid"] += 1.0
        received["big"][0, 0] = 1.0

    dist.barrier()
    print(f"tensor broadcast passed the test! Rank {rank}")


def test_tensor_broadcast():
    distributed_run(worker_fn_tensor_broadcast, 2)


def _dumps_oob(obj) -> tuple[bytes, list]:
    """Pickle `obj` the same way `MessageQueue.enqueue` does: tensor
    dispatch table + out-of-band buffers >= 1MiB."""
    buffers = []

    def callback(buf: pickle.PickleBuffer) -> bool:
        raw = buf.raw()
        if raw.nbytes < 1024 * 1024:
            return True
        buffers.append(raw)
        return False

    bio = io.BytesIO()
    pickler = pickle.Pickler(
        bio, protocol=pickle.HIGHEST_PROTOCOL, buffer_callback=callback
    )
    pickler.dispatch_table = {torch.Tensor: _reduce_tensor}
    pickler.dump(obj)
    return bio.getvalue(), buffers


@pytest.mark.parametrize(
    "case",
    [
        "small",
        "mid",
        "bf16",
        "fp8",
        "empty",
        "scalar",
        "noncontig",
        "requires_grad",
        "conj",
        "param",
    ],
)
def test_tensor_pickle_roundtrip(case: str):
    tensor = {
        # Inlined in-band (< 1MiB) and out-of-band (>= 1MiB) buffers.
        "small": lambda: torch.randn(100, 10),
        "mid": lambda: torch.randn(1024, 512),
        # Dtypes numpy doesn't recognize.
        "bf16": lambda: torch.randn(512, 512, dtype=torch.bfloat16),
        "fp8": lambda: torch.randn(32, 32).to(torch.float8_e4m3fn),
        # Shape edge cases.
        "empty": lambda: torch.empty(0, 8),
        "scalar": lambda: torch.tensor(3.14),
        "noncontig": lambda: torch.randn(64, 64).t(),
        # These fall back to torch's default reducer.
        "requires_grad": lambda: torch.randn(8, 8, requires_grad=True),
        "conj": lambda: torch.randn(4, dtype=torch.complex64).conj(),
        "param": lambda: torch.nn.Parameter(torch.randn(4), requires_grad=False),
    }[case]()

    data, buffers = _dumps_oob({"tensor": tensor, "meta": list(range(10))})
    received = pickle.loads(data, buffers=buffers)["tensor"]

    assert received.shape == tensor.shape
    assert received.dtype == tensor.dtype
    if tensor.dtype == torch.float8_e4m3fn:
        assert torch.equal(received.view(torch.uint8), tensor.view(torch.uint8))
    else:
        assert torch.equal(received, tensor)
    assert received.requires_grad == tensor.requires_grad
    assert isinstance(received, type(tensor))
    if tensor.numel() and not tensor.requires_grad:
        # Rebuilt tensors must be writable, like regular tensors.
        received.view(-1)[0] = 1.0


@pytest.mark.parametrize("case", ["cuda", "requires_grad", "conj"])
def test_reduce_tensor_fallback(case: str):
    """Tensors the zero-copy reducer can't safely alias must fall back to
    torch's default reduction."""
    if case == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("requires CUDA")
        tensor = torch.randn(4, device="cuda")
    elif case == "requires_grad":
        tensor = torch.randn(8, requires_grad=True)
    else:
        tensor = torch.randn(4, dtype=torch.complex64).conj()

    reduced = _reduce_tensor(tensor)
    assert reduced[0] is not _rebuild_tensor


@pytest.mark.parametrize("should_warn", [False, True])
def test_reader_timeout_caps_indefinite_waits(should_warn):
    with (
        mock.patch(
            "vllm.distributed.device_communicators.shm_broadcast."
            "SHM_READER_RECHECK_INTERVAL_MS",
            new=7,
        ),
        mock.patch(
            "vllm.distributed.device_communicators.shm_broadcast."
            "VLLM_RINGBUFFER_WARNING_INTERVAL",
            new=60,
        ),
    ):
        timeout = MessageQueue.ReadTimeoutWithWarnings(
            timeout=None, should_warn=should_warn
        )
        assert timeout.timeout_ms() == 7


def test_reader_rechecks_shm_after_idle_wait_timeout_without_notify():
    writer = MessageQueue(
        n_reader=1,
        n_local_reader=1,
        max_chunk_bytes=1024 * 1024,
        max_chunks=1,
    )
    reader = MessageQueue.create_from_handle(writer.export_handle(), rank=0)
    payload = 123
    poll_started = threading.Event()
    allow_timeout = threading.Event()
    result = {}

    def acquire_read_in_thread():
        try:
            with reader.acquire_read(indefinite=True) as buf:
                result["value"] = buf[0]
        except Exception as exc:
            result["exc"] = exc

    def poll_timeout(*, timeout: int | None = None):
        poll_started.set()
        assert allow_timeout.wait(timeout=5)
        return []

    try:
        writer.wait_until_ready()
        reader.wait_until_ready()
        reader._spin_condition.last_read = 0
        reader._spin_condition.busy_loop_s = 0

        with (
            mock.patch(
                "vllm.distributed.device_communicators.shm_broadcast."
                "SHM_READER_RECHECK_INTERVAL_MS",
                new=50,
            ),
            mock.patch(
                "vllm.distributed.device_communicators.shm_broadcast."
                "VLLM_RINGBUFFER_WARNING_INTERVAL",
                new=60,
            ),
            mock.patch.object(
                reader._spin_condition.poller,
                "poll",
                side_effect=poll_timeout,
            ) as poll,
        ):
            read_thread = threading.Thread(target=acquire_read_in_thread, daemon=True)
            read_thread.start()
            assert poll_started.wait(timeout=5)
            with writer.acquire_write(timeout=0.1) as buf:
                buf[0] = payload
            allow_timeout.set()
            read_thread.join(timeout=5)

            assert not read_thread.is_alive()
            poll.assert_called_once_with(timeout=50)

        if "exc" in result:
            raise result["exc"]
        assert result["value"] == payload
        with writer.buffer.get_metadata(0) as metadata_buffer:
            assert metadata_buffer[0] == 1
            assert metadata_buffer[1] == 1
    finally:
        writer.shutdown()
        reader.shutdown()
        for socket in (
            writer.local_socket,
            writer._spin_condition.local_notify_socket,
            reader.local_socket,
            reader._spin_condition.local_notify_socket,
            reader._spin_condition.read_cancel_socket,
            reader._spin_condition.write_cancel_socket,
        ):
            socket.close(linger=0)


def test_acquire_read_releases_slot_when_reader_raises():
    writer = MessageQueue(
        n_reader=1,
        n_local_reader=1,
        max_chunk_bytes=1024 * 1024,
        max_chunks=1,
    )
    reader = MessageQueue.create_from_handle(writer.export_handle(), rank=0)
    try:
        writer.wait_until_ready()
        reader.wait_until_ready()

        writer.enqueue({"payload": "first"})

        with (
            pytest.raises(RuntimeError, match="reader failed"),
            reader.acquire_read(timeout=0.1),
        ):
            raise RuntimeError("reader failed")

        with writer.buffer.get_metadata(0) as metadata_buffer:
            assert metadata_buffer[0] == 1
            assert metadata_buffer[1] == 1

        with writer.acquire_write(timeout=0.1) as buf:
            buf[0] = 0
    finally:
        writer.shutdown()
        reader.shutdown()


def test_concurrent_readers_are_not_stranded_on_stale_ring_slot():
    """Two threads sharing one reader must both make progress.

    `acquire_read` used to bind the metadata slot for `current_idx` once,
    outside its retry loop. Once one thread consumed a block and advanced
    `current_idx`, the other kept polling the slot it had already captured and
    blocked until its own timeout, even though a later message was ready.
    """
    writer = MessageQueue(
        n_reader=1,
        n_local_reader=1,
        max_chunk_bytes=1024 * 1024,
        max_chunks=10,
    )
    reader = MessageQueue.create_from_handle(writer.export_handle(), rank=0)

    results: dict[str, object] = {}
    parked: set[str] = set()
    parked_lock = threading.Lock()
    both_parked = threading.Event()
    real_wait = reader._spin_condition.wait

    def counting_wait(*args, **kwargs):
        with parked_lock:
            parked.add(threading.current_thread().name)
            if len(parked) == 2:
                both_parked.set()
        return real_wait(*args, **kwargs)

    def dequeue_once(name: str):
        try:
            results[name] = reader.dequeue(timeout=30.0)
        except Exception as exc:  # noqa: BLE001
            results[name] = exc

    threads = [
        threading.Thread(target=dequeue_once, args=(name,), name=name, daemon=True)
        for name in ("reader-a", "reader-b")
    ]
    try:
        writer.wait_until_ready()
        reader.wait_until_ready()

        with mock.patch.object(
            reader._spin_condition, "wait", side_effect=counting_wait
        ):
            for t in threads:
                t.start()
            # Both threads have to be waiting on the same `current_idx` with no
            # message available: that is the state the bug was triggered from.
            assert both_parked.wait(timeout=10)

            writer.enqueue("first")
            writer.enqueue("second")

            for t in threads:
                t.join(timeout=15)
            assert not any(t.is_alive() for t in threads)
    finally:
        writer.shutdown()
        reader.shutdown()
        for socket in (
            writer.local_socket,
            writer._spin_condition.local_notify_socket,
            reader.local_socket,
            reader._spin_condition.local_notify_socket,
            reader._spin_condition.read_cancel_socket,
            reader._spin_condition.write_cancel_socket,
        ):
            socket.close(linger=0)

    for name, value in results.items():
        if isinstance(value, BaseException):
            raise AssertionError(f"{name} failed to dequeue") from value
    assert sorted(results.values()) == ["first", "second"]


def test_warning_logs(caplog_vllm):
    """
    Test that warning logs are emitted at VLLM_RINGBUFFER_WARNING_INTERVAL intervals
    when indefinite=False, and are not emitted when indefinite=True.
    """

    # Patch the warning log interval to every 1 ms during reads
    with mock.patch(
        "vllm.distributed.device_communicators.shm_broadcast.VLLM_RINGBUFFER_WARNING_INTERVAL",
        new=0.001,  # 1 ms
    ):
        writer = MessageQueue(
            n_reader=1,
            n_local_reader=1,
            max_chunk_bytes=1024 * 1024,  # 1MB chunks
            max_chunks=10,
        )
        reader = MessageQueue.create_from_handle(writer.export_handle(), rank=0)
        writer.wait_until_ready()
        reader.wait_until_ready()

        # We should have at least one warning log here
        # "0 seconds" expected due to rounding of 1ms test interval
        with pytest.raises(TimeoutError):
            reader.dequeue(timeout=0.01, indefinite=False)
        assert any(
            "No available shared memory broadcast block found in 0 seconds"
            in record.message
            for record in caplog_vllm.records
        )
        caplog_vllm.clear()

        # We should have no warnings this time
        with pytest.raises(TimeoutError):
            reader.dequeue(timeout=0.01, indefinite=True)
        assert all(
            "No available shared memory broadcast block found in 0 seconds"
            not in record.message
            for record in caplog_vllm.records
        )

        # Clean up when done
        writer.shutdown()
        reader.shutdown()


def _fake_disk_usage(free_bytes: int):
    return SimpleNamespace(total=free_bytes, used=0, free=free_bytes)


def test_check_shm_free_space_raises_when_insufficient(tmp_path):
    with (
        mock.patch.object(
            shm_broadcast.shutil, "disk_usage", return_value=_fake_disk_usage(32 << 20)
        ),
        pytest.raises(RuntimeError, match="Insufficient space"),
    ):
        check_shm_free_space(240 << 20, shm_path=str(tmp_path))


def test_check_shm_free_space_passes_when_sufficient(tmp_path):
    with mock.patch.object(
        shm_broadcast.shutil, "disk_usage", return_value=_fake_disk_usage(512 << 20)
    ):
        check_shm_free_space(240 << 20, shm_path=str(tmp_path))


def test_check_shm_free_space_skipped_when_path_missing(tmp_path):
    check_shm_free_space(1 << 60, shm_path=str(tmp_path / "does-not-exist"))


def test_shm_ring_buffer_creation_checks_free_space():
    with (
        mock.patch.object(
            shm_broadcast.shutil, "disk_usage", return_value=_fake_disk_usage(1 << 20)
        ),
        mock.patch.object(shm_broadcast.os.path, "isdir", return_value=True),
        pytest.raises(RuntimeError, match="Insufficient space"),
    ):
        ShmRingBuffer(n_reader=1, max_chunk_bytes=24 * 1024 * 1024, max_chunks=10)


def test_remote_subscribe_addr_unique_concurrent_writers(
    monkeypatch: pytest.MonkeyPatch,
):
    """Writers bind the remote socket to port 0 (kernel-assigned), so
    concurrent writers never race for the same probed port and the
    announced address is connectable.

    Pre-fix, the writer probed a port with get_open_port() and bound it
    afterwards; pinning the probe to one free port makes every writer
    bind the same port and fail deterministically on that code path,
    while the late-binding implementation never consults the probe."""
    from vllm.distributed.device_communicators import shm_broadcast

    colliding_port = get_open_port()
    monkeypatch.setattr(
        shm_broadcast, "get_open_port", lambda: colliding_port, raising=False
    )

    n_writers = 32
    queues: list[MessageQueue] = []
    lock = threading.Lock()

    def make_writer():
        q = MessageQueue(
            n_reader=1,
            n_local_reader=0,
            max_chunk_bytes=4096,
            max_chunks=2,
            connect_ip="127.0.0.1",
        )
        with lock:
            queues.append(q)

    threads = [threading.Thread(target=make_writer) for _ in range(n_writers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(queues) == n_writers
    addrs = [q.export_handle().remote_subscribe_addr for q in queues]
    assert all(addr and addr.startswith("tcp://") for addr in addrs)
    assert len(set(addrs)) == n_writers

    writer = queues[0]
    received = []

    def reader_main():
        reader = MessageQueue.create_from_handle(writer.export_handle(), rank=0)
        reader.wait_until_ready()
        received.append(reader.dequeue())
        reader.remote_socket.close(linger=0)

    reader_thread = threading.Thread(target=reader_main)
    reader_thread.start()
    writer.wait_until_ready()
    writer.enqueue("ping")
    reader_thread.join(timeout=30)
    assert not reader_thread.is_alive()
    assert received == ["ping"]

    for q in queues:
        q.remote_socket.close(linger=0)
