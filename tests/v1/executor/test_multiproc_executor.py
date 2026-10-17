# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import weakref
from collections import deque
from types import SimpleNamespace
from typing import Any

import pytest

from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
from vllm.distributed.parallel_state import GroupCoordinator
from vllm.v1.executor import multiproc_executor
from vllm.v1.executor.multiproc_executor import MultiprocExecutor, WorkerProc
from vllm.v1.outputs import DraftTokenIds


@pytest.mark.parametrize("nnodes", [1, 2])
@pytest.mark.parametrize("concurrent_batches", [2, 9, 17])
def test_response_queue_holds_pipeline_replies(monkeypatch, nnodes, concurrent_batches):
    """A worker can publish the full RPC window before EngineCore drains it."""

    # Replace distributed bootstrap only; exercise the real SHM transport with
    # small slots since these responses contain no tensors.
    def make_queue(*args, **kwargs):
        kwargs["max_chunk_bytes"] = 1024
        return MessageQueue(*args, **kwargs)

    monkeypatch.setattr(multiproc_executor, "MessageQueue", make_queue)
    monkeypatch.setattr(
        make_queue, "create_from_handle", lambda *args: None, raising=False
    )

    def create_single_reader(pg, max_chunk_bytes, max_chunks, **kwargs):
        return make_queue(1, 1, max_chunks=max_chunks), []

    monkeypatch.setattr(
        MessageQueue, "create_from_process_group_single_reader", create_single_reader
    )
    group: Any = GroupCoordinator.__new__(GroupCoordinator)
    group.cpu_group = None
    group.ranks = [0]
    monkeypatch.setattr(group, "create_mq_broadcaster", lambda **kwargs: None)
    monkeypatch.setattr(multiproc_executor, "get_inner_dp_world_group", lambda: group)
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(nnodes_within_dp=nnodes),
        max_concurrent_batches=concurrent_batches,
    )
    worker: Any = WorkerProc.__new__(WorkerProc)
    worker.worker = SimpleNamespace(rank=0)
    worker._init_message_queues(None, config)
    writer = worker.worker_response_mq
    reader = MessageQueue.create_from_handle(writer.export_handle(), rank=0)
    try:
        writer.wait_until_ready()
        reader.wait_until_ready()
        # Repeated windows wrap the ring and verify that consumed slots can be
        # reused without overwriting or reordering any worker's replies.
        for window in range(3):
            replies = [
                (window, batch, method)
                for batch in range(concurrent_batches)
                for method in ("execute_model", "sample_tokens")
            ]
            for reply in replies:
                writer.enqueue(reply, timeout=1)
            assert [reader.dequeue(timeout=1) for _ in replies] == replies
    finally:
        writer.shutdown()
        reader.shutdown()
        for queue in (writer, reader):
            queue.local_socket.close(linger=0)
            queue._spin_condition.local_notify_socket.close(linger=0)
        reader._spin_condition.read_cancel_socket.close(linger=0)
        reader._spin_condition.write_cancel_socket.close(linger=0)


class _ExitWorkerLoop(RuntimeError):
    pass


class _RpcPayload:
    pass


class _PayloadLifetimeCheckingQueue:
    def __init__(self) -> None:
        self.payload_ref: weakref.ReferenceType[_RpcPayload] | None = None
        self.dequeue_count = 0

    def dequeue(self, *, indefinite: bool):
        assert indefinite
        self.dequeue_count += 1
        if self.dequeue_count == 1:
            payload = _RpcPayload()
            self.payload_ref = weakref.ref(payload)
            return "consume", (payload,), {}, None

        assert self.payload_ref is not None
        assert self.payload_ref() is None
        raise _ExitWorkerLoop


def test_worker_rpc_payload_released_before_next_dequeue():
    queue = _PayloadLifetimeCheckingQueue()
    worker_proc: Any = WorkerProc.__new__(WorkerProc)
    worker_proc.rpc_broadcast_mq = queue
    worker_proc.rank = 0
    worker_proc.worker = SimpleNamespace(consume=lambda payload: payload)
    worker_proc.handle_output = lambda output: None

    with pytest.raises(_ExitWorkerLoop):
        worker_proc.worker_busy_loop()

    assert queue.dequeue_count == 2


def test_execute_worker_rpc_returns_worker_exception():
    def fail():
        raise RuntimeError("test error")

    worker_proc: Any = WorkerProc.__new__(WorkerProc)
    worker_proc.rank = 0
    worker_proc.worker = SimpleNamespace(fail=fail)
    outputs: list[Any] = []
    worker_proc.handle_output = outputs.append

    worker_proc._execute_worker_rpc(("fail", (), {}, None))

    assert len(outputs) == 1
    assert isinstance(outputs[0], RuntimeError)
    assert str(outputs[0]) == "test error"


@pytest.mark.parametrize("stalled", [False, True])
@pytest.mark.parametrize(
    "method,result",
    [
        ("take_draft_token_ids", DraftTokenIds(["req"], [[1, 2, 3]])),
        ("execute_dummy_batch", None),
    ],
)
def test_model_rpc_uses_execute_model_timeout(monkeypatch, method, result, stalled):
    """A wedged model RPC must not block EngineCore forever. For
    execute_dummy_batch this is an idle DP rank stuck in a collective with a
    hung peer."""
    monkeypatch.setenv("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "7")

    def dequeue(*, timeout=None):
        assert timeout is not None and 0 < timeout <= 7
        if stalled:
            raise TimeoutError
        return WorkerProc.ResponseStatus.SUCCESS, result

    executor: Any = MultiprocExecutor.__new__(MultiprocExecutor)
    executor.is_failed = False
    executor.output_rank = 1
    executor.futures_queue = deque()
    executor.rpc_broadcast_mq = SimpleNamespace(enqueue=lambda payload: None)
    executor.response_mqs = [None, SimpleNamespace(dequeue=dequeue)]

    if stalled:
        with pytest.raises(TimeoutError, match=f"{method} timed out"):
            getattr(executor, method)()
    else:
        assert getattr(executor, method)() is result
