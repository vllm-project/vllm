# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import multiprocessing
import threading
import time
from contextlib import suppress

import pytest

from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
from vllm.v1.executor import early_kv_recv
from vllm.v1.executor.early_kv_recv import EarlyKVRecvReceiver, EarlyKVRecvSender


class _PendingConnector:
    def __init__(self, done: set[str] | None = None, error: Exception | None = None):
        self.done = done or set()
        self.error = error

    def has_pending_recving(self) -> bool:
        return True

    def poll_finished_recving(self) -> set[str]:
        if self.error is not None:
            raise self.error
        return self.done


@pytest.fixture
def rpc_queue():
    writer = MessageQueue(n_reader=1, n_local_reader=1, max_chunks=2)
    reader = MessageQueue.create_from_handle(writer.export_handle(), rank=0)
    writer.wait_until_ready()
    reader.wait_until_ready()
    yield writer, reader
    reader.shutdown()
    writer.shutdown()


@pytest.fixture
def receiver():
    receiver = EarlyKVRecvReceiver()
    yield receiver
    receiver.close()


def _sender(monkeypatch, connector, path: str) -> EarlyKVRecvSender:
    monkeypatch.setattr(early_kv_recv, "has_kv_transfer_group", lambda: True)
    monkeypatch.setattr(early_kv_recv, "get_kv_transfer_group", lambda: connector)
    return EarlyKVRecvSender(path)


def _wait_for_poll(sender: EarlyKVRecvSender, reader: MessageQueue) -> bool | None:
    """None if the worker would still be spinning after 5 s."""
    result: list[bool] = []
    thread = threading.Thread(
        target=lambda: result.append(sender.wait_for_poll(reader)), daemon=True
    )
    thread.start()
    thread.join(timeout=5)
    return result[0] if result else None


def test_sender_notifies_then_yields_to_rpc(monkeypatch, rpc_queue, receiver):
    writer, reader = rpc_queue
    sender = _sender(monkeypatch, _PendingConnector(done={"r"}), receiver.path)
    try:
        assert _wait_for_poll(sender, reader) is True
        sender.poll()
        received: list[set[str]] = []
        deadline = time.monotonic() + 5
        while not received and time.monotonic() < deadline:
            received.extend(receiver.drain())
        assert received == [{"r"}]
        assert receiver.drain() == []
        # Still waiting for the next RPC: poll again after the interval.
        assert _wait_for_poll(sender, reader) is True
        writer.enqueue("rpc")
        assert _wait_for_poll(sender, reader) is False
    finally:
        sender.close()


def test_sender_stops_polling_when_idle_or_shut_down(monkeypatch, rpc_queue, receiver):
    _, reader = rpc_queue
    sender = _sender(monkeypatch, _PendingConnector(), receiver.path)
    try:
        # A paused engine sends nothing: block in dequeue() instead of spinning.
        now = time.perf_counter()
        monkeypatch.setattr(early_kv_recv.time, "perf_counter", lambda: now + 2)
        assert _wait_for_poll(sender, reader) is False
        sender.step_rpc_done()
        assert _wait_for_poll(sender, reader) is True
        reader.shutdown()
        assert _wait_for_poll(sender, reader) is False
    finally:
        sender.close()


def test_rpc_arriving_during_poll_wait_is_not_consumed(
    monkeypatch, rpc_queue, receiver
):
    writer, reader = rpc_queue
    sender = _sender(monkeypatch, _PendingConnector(), receiver.path)
    monkeypatch.setattr(early_kv_recv, "_POLL_INTERVAL_S", 1.0)
    monkeypatch.setattr(early_kv_recv, "_ACTIVE_WINDOW_S", 5.0)
    monkeypatch.setattr(early_kv_recv.time, "sleep", lambda _: writer.enqueue("rpc"))
    try:
        sender.poll()
        assert _wait_for_poll(sender, reader) is False
        assert reader.dequeue(timeout=1) == "rpc"
    finally:
        sender.close()


def _poll_with_failure(path, broken_channel):
    with pytest.MonkeyPatch.context() as monkeypatch:
        error = None if broken_channel else RuntimeError("post-processing failed")
        sender = _sender(monkeypatch, _PendingConnector(done={"r"}, error=error), path)
        if broken_channel:
            sender.close()
        # An ordinary RPC exception can be consumed by a utility call.
        with suppress(Exception):
            sender.poll()


@pytest.mark.parametrize("broken_channel", [False, True])
def test_poll_failure_terminates_worker_without_engine_drain(receiver, broken_channel):
    process = multiprocessing.get_context("spawn").Process(
        target=_poll_with_failure,
        args=(receiver.path, broken_channel),
    )
    process.start()
    try:
        process.join(timeout=30)
        assert process.exitcode == 1
    finally:
        if process.is_alive():
            process.kill()
            process.join()
