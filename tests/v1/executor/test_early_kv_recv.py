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


@pytest.mark.parametrize("failed", [False, True])
def test_sender_notifies_then_yields_to_rpc(monkeypatch, rpc_queue, receiver, failed):
    writer, reader = rpc_queue
    error = RuntimeError("post-processing failed") if failed else None
    sender = _sender(
        monkeypatch, _PendingConnector(done={"r"}, error=error), receiver.path
    )
    try:
        assert _wait_for_poll(sender, reader) is True
        sender.poll()
        received: list[set[str] | str] = []
        deadline = time.monotonic() + 5
        while not received and time.monotonic() < deadline:
            received.extend(receiver.drain())
        assert received == (
            ["RuntimeError: post-processing failed"] if failed else [{"r"}]
        )
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


def _poll_with_broken_error_channel(path):
    with pytest.MonkeyPatch.context() as monkeypatch:
        sender = _sender(monkeypatch, _PendingConnector(done={"r"}), path)
        sender.close()  # IPC can no longer deliver completion or error.
        # An ordinary RPC exception can be consumed by a utility call.
        with suppress(Exception):
            sender.poll()


def test_broken_error_channel_terminates_worker(receiver):
    process = multiprocessing.get_context("spawn").Process(
        target=_poll_with_broken_error_channel, args=(receiver.path,)
    )
    process.start()
    try:
        process.join(timeout=30)
        assert process.exitcode == 1
    finally:
        if process.is_alive():
            process.kill()
            process.join()
