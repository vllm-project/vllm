# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import multiprocessing
import time
from contextlib import closing, suppress

import pytest

from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
from vllm.v1.executor import early_kv_recv
from vllm.v1.executor.early_kv_recv import (
    EarlyKVOutputAggregator,
    EarlyKVRecvReceiver,
    EarlyKVRecvSender,
)
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput


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
    with closing(EarlyKVRecvReceiver()) as receiver:
        yield receiver


def _sender(monkeypatch, connector, path: str) -> EarlyKVRecvSender:
    monkeypatch.setattr(early_kv_recv, "has_kv_transfer_group", lambda: True)
    monkeypatch.setattr(early_kv_recv, "get_kv_transfer_group", lambda: connector)
    return EarlyKVRecvSender(path)


@pytest.mark.timeout(10)
def test_sender_notifies_then_yields_to_rpc(monkeypatch, rpc_queue, receiver):
    writer, reader = rpc_queue
    monkeypatch.setattr(early_kv_recv.time, "perf_counter", lambda: 0)
    # Enqueue during the wait: the sender must leave the RPC for dequeue().
    monkeypatch.setattr(early_kv_recv.time, "sleep", lambda _: writer.enqueue("rpc"))
    with closing(
        _sender(monkeypatch, _PendingConnector(done={"r"}), receiver.path)
    ) as sender:
        assert sender.wait_for_poll(reader)
        sender.poll()
        received: list[set[str]] = []
        deadline = time.monotonic() + 5
        while not received and time.monotonic() < deadline:
            received.extend(receiver.drain())
        assert received == [{"r"}]
        assert receiver.drain() == []
        assert not sender.wait_for_poll(reader)
        assert reader.dequeue(timeout=1) == "rpc"


def test_sender_stops_polling_when_idle_or_shut_down(monkeypatch, rpc_queue, receiver):
    _, reader = rpc_queue
    with closing(_sender(monkeypatch, _PendingConnector(), receiver.path)) as sender:
        # A paused engine sends nothing: block in dequeue() instead of spinning.
        now = time.perf_counter()
        monkeypatch.setattr(early_kv_recv.time, "perf_counter", lambda: now + 2)
        assert not sender.wait_for_poll(reader)
        sender.step_rpc_done()
        assert sender.wait_for_poll(reader)
        reader.shutdown()
        assert not sender.wait_for_poll(reader)


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


@pytest.mark.parametrize(
    "ordinary_rank,failed",
    [(None, False), (0, False), (-1, False), (0, True), (-1, True)],
)
def test_early_receives_share_normal_quorum(ordinary_rank, failed):
    workers = 2
    aggregator = EarlyKVOutputAggregator(workers)
    early = set()
    normal = KVConnectorOutput()
    for rank in range(workers):
        if ordinary_rank is not None and rank == ordinary_rank % workers:
            output = ModelRunnerOutput(
                [],
                {},
                kv_connector_output=KVConnectorOutput(
                    finished_recving={"request"},
                    failed_recving={"request"} if failed else set(),
                ),
            )
            assert (result := aggregator.aggregate([output])) is not None
            assert (normal := result.kv_connector_output) is not None
        else:
            early |= aggregator.add_finished_recving({"request"})
        if rank < workers - 1:
            assert not early and not normal.finished_recving
    # If an early notification completes a failed quorum, recovery stays on
    # the ordinary output path, even when that output has no new receives.
    if not normal.finished_recving:
        assert aggregator.aggregate([None]) is None
        output = ModelRunnerOutput([], {})
        assert (result := aggregator.aggregate([output])) is not None
        assert (normal := result.kv_connector_output) is not None
    assert early | (normal.finished_recving or set()) == {"request"}
    assert normal.failed_recving == ({"request"} if failed else set())
    assert not early if failed else not (early and normal.finished_recving)
    assert (result := aggregator.aggregate([ModelRunnerOutput([], {})])) is not None
    assert (empty := result.kv_connector_output) is not None
    assert not empty.finished_recving and not empty.failed_recving
