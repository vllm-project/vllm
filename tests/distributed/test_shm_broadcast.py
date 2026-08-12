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
    ShmTensorArena,
    _ArenaPickler,
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


def distributed_run(fn, world_size, timeout=60, mp_context=None):
    """Run a function in multiple processes with proper error handling.

    Args:
        fn: Function to run in each process
        world_size: Number of processes to spawn
        timeout: Maximum time in seconds to wait for processes (default: 60)
        mp_context: Optional multiprocess context (e.g. ``mp.get_context("spawn")``)
            for workers that must not inherit the parent's thread/CUDA state.
    """
    ctx = mp_context or mp
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
        p = ctx.Process(target=fn, args=(env,))
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


# --------------- ShmTensorArena (zero-copy tensor arena) tests ---------------


def _make_arena(n_reader: int = 2, slot_bytes: int = 4 << 20, n_slots: int = 3):
    """Writer arena plus attached per-reader arenas (same process)."""
    writer = ShmTensorArena(n_reader, slot_bytes, n_slots)
    readers = [
        ShmTensorArena(*writer.handle(), reader_rank=i) for i in range(n_reader)
    ]
    return writer, readers


def _get_view(reader: ShmTensorArena, idx: int, ref: torch.Tensor) -> torch.Tensor:
    return reader.get_tensor(
        idx, ref.numel() * ref.element_size(), ref.dtype, tuple(ref.shape)
    )


def _drain(reader: ShmTensorArena) -> None:
    """Flush a reader's releases to completion. On the pinned path the first
    flush defers the release behind an H2D-completion event; the second flush
    retires it once the event has fired."""
    reader.flush_releases()
    if reader._deferred_releases:
        torch.cuda.synchronize()
        reader.flush_releases()
    assert not reader._pending_release
    assert not reader._deferred_releases


def test_arena_zero_copy_roundtrip():
    writer, (r0, r1) = _make_arena(n_reader=2)
    src = torch.randn(400, 512)
    idx = writer.write_tensor(src)
    assert idx is not None
    a = _get_view(r0, idx, src)
    b = _get_view(r1, idx, src)
    assert torch.equal(a, src)
    assert torch.equal(b, src)
    assert a.dtype == src.dtype and a.shape == src.shape
    # Readers alias one shared mapping: a write through one reader's view is
    # visible through the other's (this is what "zero-copy" means here).
    a[0, 0] = 12345.0
    assert b[0, 0].item() == 12345.0
    del a, b


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_arena_dtype_roundtrip(dtype):
    """Dtypes numpy doesn't recognize traverse the arena's uint8 view path."""
    writer, (reader,) = _make_arena(n_reader=1)
    src = torch.randn(256, 256).to(dtype)
    idx = writer.write_tensor(src)
    assert idx is not None
    got = _get_view(reader, idx, src)
    assert got.dtype == dtype
    assert torch.equal(got.view(torch.uint8), src.view(torch.uint8))
    del got


def test_arena_slot_lifecycle():
    """A slot must not be reusable until EVERY reader has released it —
    the writer overwriting a slot a reader still consumes would corrupt data."""
    writer, readers = _make_arena(n_reader=2, n_slots=3)
    t = torch.ones(1000)
    first = writer.write_tensor(t)
    views = [_get_view(r, first, t) for r in readers]
    # Fill the remaining slots; the arena is now exhausted.
    assert all(writer.write_tensor(t) is not None for _ in range(2))
    assert writer.write_tensor(t) is None  # exhausted -> caller falls back
    # One of two readers releasing is NOT enough to reuse the slot.
    _drain(readers[0])
    assert writer.write_tensor(t) is None
    # Once every reader has released, the original slot is reused.
    _drain(readers[1])
    assert writer.write_tensor(t) == first
    del views


def test_arena_oversize_falls_back():
    writer, _ = _make_arena(n_reader=1, slot_bytes=1 << 20, n_slots=2)
    big = torch.empty((1 << 20) + 4096, dtype=torch.uint8)
    assert writer.write_tensor(big) is None


def _dumps_arena(obj, arena: ShmTensorArena) -> tuple[bytes, list]:
    """Pickle `obj` the same way `MessageQueue.enqueue` does when an arena is
    attached: arena diversion first (reducer_override), then the tensor
    dispatch table, with out-of-band buffers >= 1MiB."""
    buffers = []

    def callback(buf: pickle.PickleBuffer) -> bool:
        raw = buf.raw()
        if raw.nbytes < 1024 * 1024:
            return True
        buffers.append(raw)
        return False

    bio = io.BytesIO()
    pickler = _ArenaPickler(bio, arena, buffer_callback=callback)
    pickler.dispatch_table = {torch.Tensor: _reduce_tensor}
    pickler.dump(obj)
    return bio.getvalue(), buffers


def test_arena_pickler_composes(monkeypatch):
    """Large contiguous tensors are diverted into the arena; everything the
    arena declines falls through to `_reduce_tensor` unchanged."""
    writer, (reader,) = _make_arena(n_reader=1)
    monkeypatch.setattr(shm_broadcast, "_ARENA_MIN_BYTES", 1 << 20)
    monkeypatch.setitem(
        shm_broadcast._TENSOR_ARENAS, writer.shared_memory.name, reader
    )
    big = torch.randn(1024, 1024)  # 4MiB -> diverted into the arena
    small = torch.randn(16, 16)  # 1KiB -> falls through to _reduce_tensor
    data, buffers = _dumps_arena({"big": big, "small": small}, writer)
    # The diverted tensor's bytes are in the arena, not the pickle stream.
    assert len(data) + sum(b.nbytes for b in buffers) < big.numel() * 4
    out = pickle.loads(data, buffers=buffers)
    assert torch.equal(out["big"], big)
    assert torch.equal(out["small"], small)
    # "big" is a zero-copy view of the reader's slot; "small" is not.
    (idx,) = reader._pending_release
    nbytes = big.numel() * big.element_size()
    slot_ptr = torch.frombuffer(
        reader._slot(idx, nbytes), dtype=torch.uint8
    ).data_ptr()
    assert out["big"].data_ptr() == slot_ptr
    assert out["small"].data_ptr() != slot_ptr
    del out
    _drain(reader)


def test_arena_pickler_noncontig_falls_through(monkeypatch):
    writer, (reader,) = _make_arena(n_reader=1)
    monkeypatch.setattr(shm_broadcast, "_ARENA_MIN_BYTES", 1 << 20)
    monkeypatch.setitem(
        shm_broadcast._TENSOR_ARENAS, writer.shared_memory.name, reader
    )
    nc = torch.randn(2048, 1024)[:, ::2]  # non-contiguous, above threshold
    assert not nc.is_contiguous()
    data, buffers = _dumps_arena(nc, writer)
    out = pickle.loads(data, buffers=buffers)
    assert torch.equal(out, nc)
    # The arena never saw it: no slot consumed on the reader.
    assert reader._pending_release == []
    del out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_arena_event_gated_release():
    """On the pinned path a slot release is gated on an H2D-completion CUDA
    event: the writer must not see the reader's done flag until the async DMA
    sourced from the slot has been retired."""
    writer, (reader,) = _make_arena(n_reader=1)
    reader._ensure_pinned()
    if not reader._pinned:
        pytest.skip("cudaHostRegister unavailable in this environment")
    src = torch.randn(512, 1024)
    idx = writer.write_tensor(src)
    view = _get_view(reader, idx, src)
    dev = view.to("cuda", non_blocking=True)
    assert reader._pending_release == [idx]
    reader.flush_releases()
    # Deferred behind the event, not applied eagerly.
    assert len(reader._deferred_releases) == 1
    with reader._meta(idx) as meta:
        assert meta[1] == 0
    torch.cuda.synchronize()
    reader.flush_releases()
    assert reader._deferred_releases == []
    with reader._meta(idx) as meta:
        assert meta[1] == 1
    assert torch.equal(dev.cpu(), src)
    del view, dev


@worker_fn_wrapper
def worker_fn_arena_broadcast():
    rank = dist.get_rank()
    writer_rank = 0
    if rank == writer_rank:
        message_queue = MessageQueue(
            1,
            1,
            local_reader_ranks=[1],
            max_chunk_bytes=8 * 1024 * 1024,
            enable_shm_tensor_arena=True,
        )
        handles = [message_queue.export_handle()]
        dist.broadcast_object_list(handles, src=writer_rank)
    else:
        handles = [None]
        dist.broadcast_object_list(handles, src=writer_rank)
        message_queue = MessageQueue.create_from_handle(handles[0], rank)
    message_queue.wait_until_ready()

    torch.manual_seed(42)
    payload = {
        # 16MiB: above the arena divert threshold (8MiB) -> arena slot.
        "huge": torch.randn(2048, 2048),
        # 2MiB: declined by the arena -> out-of-band _reduce_tensor path.
        "mid": torch.randn(1024, 512),
    }

    if rank == writer_rank:
        assert message_queue.tensor_arena is not None
        with mock.patch(
            "vllm.distributed.device_communicators.shm_broadcast._reduce_tensor",
            wraps=_reduce_tensor,
        ) as wrapped_reduce:
            message_queue.enqueue(payload)
        # "huge" was diverted into the arena before the dispatch table was
        # consulted; only "mid" went through _reduce_tensor.
        assert wrapped_reduce.call_count == 1
    else:
        received = message_queue.dequeue(timeout=30)
        assert torch.equal(received["huge"], payload["huge"])
        assert torch.equal(received["mid"], payload["mid"])
        # The huge tensor is a zero-copy view of an arena slot.
        (arena,) = shm_broadcast._TENSOR_ARENAS.values()
        (idx,) = arena._pending_release
        nbytes = received["huge"].numel() * received["huge"].element_size()
        slot_ptr = torch.frombuffer(
            arena._slot(idx, nbytes), dtype=torch.uint8
        ).data_ptr()
        assert received["huge"].data_ptr() == slot_ptr

    dist.barrier()
    print(f"arena broadcast passed the test! Rank {rank}")


def test_arena_broadcast():
    # Spawn (not fork): by the time this test runs, the pytest process may be
    # multi-threaded / CUDA-initialized (earlier GPU tests), and a forked
    # child running the arena's slot memcpy can deadlock on inherited lock
    # state. Spawned workers start clean; the larger timeout absorbs their
    # interpreter + import startup.
    distributed_run(
        worker_fn_arena_broadcast,
        2,
        timeout=180,
        mp_context=mp.get_context("spawn"),
    )
