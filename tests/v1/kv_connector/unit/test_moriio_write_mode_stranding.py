# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for MoRI-IO WRITE-mode requests stranded in WAITING_FOR_REMOTE_KVS."""

import threading
import time
from collections import OrderedDict
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.moriio import moriio_connector
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common import (
    ROLE,
    MoRIIOConnectorMetadata,
    MoRIIOMode,
    RemoteAllocInfo,
    ReqMeta,
    TransferError,
    WriteTask,
    get_port_offset,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector import (
    MoRIIOConnectorScheduler,
    MoRIIOConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_engine import (
    MoRIIOWrapper,
    MoRIIOWriter,
)

_update_connector_output = MoRIIOConnectorScheduler.update_connector_output
_process_deferred_tasks = MoRIIOWriter._process_deferred_tasks
_fail_deferred_task = MoRIIOWriter._fail_deferred_task
_get_transfer_results = MoRIIOConnectorWorker.get_transfer_results


def _task(transfer_id="tid-1", age=0.0):
    return SimpleNamespace(
        request_id=f"req-for-{transfer_id}",
        transfer_id=transfer_id,
        enqueue_time=time.perf_counter() - age,
        transfer_state=None,
        remote_ip="10.0.0.1",
        remote_notify_port=8501,
        multi_pod_hosts=["10.0.0.1", "10.0.0.2"],
        remote_dp_size_local=2,
        remote_dp_size=4,
    )


def _writer(tasks, *, remote_ready):
    executed: list[SimpleNamespace] = []
    failed: list[tuple[SimpleNamespace, float]] = []

    def fail(task, age):
        failed.append((task, age))
        return True

    writer = SimpleNamespace(
        _deferred_tasks=list(tasks),
        _defer_timeout=60.0,
        _is_transfer_terminal=lambda tid: False,
        _is_remote_ready=lambda task: remote_ready,
        _execute_write_task=executed.append,
        _fail_deferred_task=fail,
    )
    return writer, executed, failed


def test_expired_write_is_failed_and_removed_from_the_queue():
    task = _task(age=120.0)
    writer, executed, failed = _writer([task], remote_ready=False)

    _process_deferred_tasks(writer)

    assert writer._deferred_tasks == []
    assert executed == []
    assert failed[0][0] is task
    assert failed[0][1] >= 120.0


def test_expired_write_still_runs_if_the_remote_allocation_arrived():
    task = _task(age=120.0)
    writer, executed, failed = _writer([task], remote_ready=True)

    _process_deferred_tasks(writer)

    assert writer._deferred_tasks == []
    assert executed == [task]
    assert failed == []


class _Wrapper:
    def __init__(self, events, send_notify=None):
        self.events = events
        self.lock = threading.Lock()
        self.done_req_ids = []
        self.done_remote_allocate_req_dict = {}
        self.terminal: set[str] = set()
        self.sent: list[tuple[str, int, str]] = []
        self._send_notify_override = send_notify

    fail_unallocated_write = MoRIIOWrapper.fail_unallocated_write

    def _is_transfer_terminal_locked(self, transfer_id):
        return transfer_id in self.terminal

    def _mark_transfer_terminal_locked(self, transfer_id):
        self.terminal.add(transfer_id)
        self.events.append("terminal")

    def send_notify(self, transfer_id, host, port, message_type):
        if self._send_notify_override is not None:
            return self._send_notify_override(transfer_id, host, port, message_type)
        assert transfer_id in self.terminal
        assert [ack.transfer_id for ack in self.done_req_ids] == [transfer_id]
        self.sent.append((host, port, message_type))


def _failing_writer(wrapper, events):
    writer = SimpleNamespace(
        worker=SimpleNamespace(moriio_wrapper=wrapper, tp_rank=0),
        _write_state_lock=threading.Lock(),
        _clear_transfer_state=lambda transfer_id: events.append("state_cleared"),
    )
    writer._notify_write_failed = lambda task, age: MoRIIOWriter._notify_write_failed(
        writer, task, age
    )
    writer._resolve_notify_endpoint = lambda task, rank: (
        MoRIIOWriter._resolve_notify_endpoint(writer, task, rank)
    )
    return writer


def test_timeout_notifies_every_decode_dp_rank_after_terminal_release():
    events: list[str] = []
    wrapper = _Wrapper(events)

    _fail_deferred_task(_failing_writer(wrapper, events), _task(), 120.0)

    assert events == ["terminal", "state_cleared"]
    # Global DP ranks 0-3 with two ranks per pod.
    assert wrapper.sent == [
        (host, 8501 + get_port_offset(local_rank, 0), "write_failed")
        for host in ("10.0.0.1", "10.0.0.2")
        for local_rank in (0, 1)
    ]
    assert [ack.transfer_id for ack in wrapper.done_req_ids] == ["tid-1"]


def test_timeout_releases_blocks_even_if_notification_fails():
    def send_notify(*args, **kwargs):
        raise ConnectionError

    events: list[str] = []
    wrapper = _Wrapper(events, send_notify=send_notify)

    assert _fail_deferred_task(_failing_writer(wrapper, events), _task(), 120.0)
    assert [ack.transfer_id for ack in wrapper.done_req_ids] == ["tid-1"]


def test_timeout_loses_race_to_a_remote_allocation():
    wrapper = SimpleNamespace(
        lock=threading.Lock(),
        done_req_ids=[],
        done_remote_allocate_req_dict={"tid-1": object()},
        _is_transfer_terminal_locked=lambda transfer_id: False,
        _mark_transfer_terminal_locked=lambda transfer_id: None,
    )
    wrapper.fail_unallocated_write = lambda tid: MoRIIOWrapper.fail_unallocated_write(
        wrapper, tid
    )
    writer = SimpleNamespace(
        worker=SimpleNamespace(moriio_wrapper=wrapper),
        _write_state_lock=threading.Lock(),
        _clear_transfer_state=lambda transfer_id: None,
    )

    assert not _fail_deferred_task(writer, _task(), 120.0)
    assert wrapper.done_req_ids == []
    assert "tid-1" in wrapper.done_remote_allocate_req_dict


def test_write_scheduler_does_not_release_blocks_on_its_own_timeout():
    scheduler = SimpleNamespace(
        is_producer=True,
        mode=MoRIIOMode.WRITE,
        _defer_timeout=60.0,
        _deferred_send_deadlines={"req-1": (time.monotonic() - 1.0, "tid-1")},
        _pending_sent_acks={},
        unmap_request_id=lambda *args, **kwargs: None,
    )
    output = SimpleNamespace(finished_sending=set())

    _update_connector_output(scheduler, output)

    assert output.finished_sending is None
    assert "req-1" in scheduler._deferred_send_deadlines

    output.finished_sending = {"req-1"}
    _update_connector_output(scheduler, output)

    assert output.finished_sending == {"req-1"}
    assert scheduler._deferred_send_deadlines == {}


def _consumer_worker(failed_ids, mapping):
    return SimpleNamespace(
        mode=MoRIIOMode.WRITE,
        is_producer=False,
        moriio_wrapper=SimpleNamespace(pop_failed_write_req_ids=lambda: failed_ids),
        transfer_id_to_request_id=mapping,
        _unmatched_write_failures=OrderedDict(),
        _reported_write_transfers=set(),
        _completed_write_transfers=OrderedDict(),
        get_finished=lambda: (set(), set()),
    )


def test_write_failure_is_reported_as_finished_and_failed_recving():
    worker = _consumer_worker({"tid-1", "tid-other-rank"}, {"tid-1": "req-1"})

    results = _get_transfer_results(worker)

    assert results.finished_recving == {"req-1"}
    assert results.failed_recving == {"req-1"}
    assert list(worker._unmatched_write_failures) == ["tid-other-rank"]


def test_unmatched_write_failures_are_bounded():
    worker = _consumer_worker({"tid-new"}, {})
    worker._unmatched_write_failures = OrderedDict.fromkeys(["tid-old-1", "tid-old-2"])

    with patch.object(moriio_connector, "_MAX_UNMATCHED_WRITE_FAILURES", 2):
        _get_transfer_results(worker)

    assert list(worker._unmatched_write_failures) == ["tid-old-2", "tid-new"]


@pytest.mark.parametrize("completion", ["failed", "success"])
def test_completed_write_ignores_failure_after_unmap(completion):
    worker = _consumer_worker(
        {"tid-1"} if completion == "failed" else set(), {"tid-1": "req-1"}
    )
    if completion == "success":
        worker.get_finished = lambda: (set(), {"req-1"})
    assert _get_transfer_results(worker).finished_recving == {"req-1"}
    worker.transfer_id_to_request_id = {}
    worker._reported_write_transfers.clear()
    worker.get_finished = lambda: (set(), set())
    worker.moriio_wrapper.pop_failed_write_req_ids = lambda: {"tid-1", "tid-early"}
    result = _get_transfer_results(worker)
    assert not result.finished_recving
    assert not result.failed_recving
    assert list(worker._unmatched_write_failures) == ["tid-early"]

    # A distinct transfer ID can use the same request ID; no transfer-ID reuse
    # or epoch guarantee is implied by the finite completed-ID history.
    worker.transfer_id_to_request_id = {"tid-early": "req-1"}
    worker.moriio_wrapper.pop_failed_write_req_ids = lambda: set()
    result = _get_transfer_results(worker)
    assert result.finished_recving == {"req-1"}
    assert result.failed_recving == {"req-1"}
    assert not worker._unmatched_write_failures


def test_completed_write_history_is_bounded_without_weakening_live_dedup():
    worker = _consumer_worker(set(), {})
    with patch.object(moriio_connector, "_MAX_COMPLETED_WRITE_TRANSFERS", 2):
        for index in range(3):
            tid, rid = f"tid-{index}", f"req-{index}"
            worker.transfer_id_to_request_id[tid] = rid
            worker.moriio_wrapper.pop_failed_write_req_ids = lambda tid=tid: {tid}
            assert _get_transfer_results(worker).failed_recving == {rid}
        assert list(worker._completed_write_transfers) == ["tid-1", "tid-2"]
        worker.moriio_wrapper.pop_failed_write_req_ids = lambda: {"tid-0"}
        result = _get_transfer_results(worker)
        assert not result.finished_recving
        assert not result.failed_recving
        assert not worker._unmatched_write_failures


def test_write_failure_still_wins_over_same_poll_success():
    worker = _consumer_worker({"tid-1"}, {"tid-1": "req-1"})
    worker.get_finished = lambda: (set(), {"req-1"})
    result = _get_transfer_results(worker)
    assert result.finished_recving == {"req-1"}
    assert result.failed_recving == {"req-1"}
    assert list(worker._completed_write_transfers) == ["tid-1"]


def test_evicted_unmapped_completion_uses_bounded_unknown_failure_buffer():
    worker = _consumer_worker({"tid-old"}, {"tid-old": "req-old"})
    with (
        patch.object(moriio_connector, "_MAX_COMPLETED_WRITE_TRANSFERS", 1),
        patch.object(moriio_connector, "_MAX_UNMATCHED_WRITE_FAILURES", 2),
    ):
        assert _get_transfer_results(worker).failed_recving == {"req-old"}
        worker.transfer_id_to_request_id = {"tid-new": "req-new"}
        worker.moriio_wrapper.pop_failed_write_req_ids = lambda: {"tid-new"}
        assert _get_transfer_results(worker).failed_recving == {"req-new"}
        assert list(worker._completed_write_transfers) == ["tid-new"]
        worker.transfer_id_to_request_id = {}
        worker._reported_write_transfers.clear()
        # Without an epoch, an evicted ID is indistinguishable from a genuine
        # failure racing ahead of metadata. It is retained, but stays bounded.
        for tid in ("tid-old", "unknown-1", "unknown-2"):
            worker.moriio_wrapper.pop_failed_write_req_ids = lambda tid=tid: {tid}
            result = _get_transfer_results(worker)
            assert not result.finished_recving
            assert not result.failed_recving
            assert len(worker._unmatched_write_failures) <= 2
        assert list(worker._unmatched_write_failures) == ["unknown-1", "unknown-2"]


def test_write_failed_message_is_drained_separately_from_success():
    wrapper = MoRIIOWrapper.__new__(MoRIIOWrapper)
    wrapper.lock = threading.Lock()
    wrapper.done_write_cache_req_ids = []
    wrapper.failed_write_cache_req_ids = []

    with patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_engine.get_role",
        return_value=ROLE.CONSUMER,
    ):
        wrapper._handle_structured_message(
            {"type": "write_failed", "transfer_id": "tid-1"}
        )

    assert wrapper.pop_finished_write_req_ids() == set()
    assert wrapper.pop_failed_write_req_ids() == {"tid-1"}


def _abort_setup():
    """Real writer and notification state; only CUDA/transport are replaced."""

    class Worker:
        moriio_wrapper: MoRIIOWrapper
        tp_rank = 0
        world_size = 1
        moriio_config = SimpleNamespace(defer_timeout=60.0)

        def get_engine_name_with_dp(self, engine_id, dp_rank):
            return engine_id

        def _get_built_session(self, engine_id):
            return [], None

    worker = Worker()
    wrapper = worker.moriio_wrapper = MoRIIOWrapper()
    notifications = []
    wrapper.send_notify = lambda tid, host, port, message_type: notifications.append(
        (tid, message_type)
    )
    writer = MoRIIOWriter(worker)
    writer.ensure_worker_started = lambda: None
    writer._prepare_transfer_plan = lambda *args: None
    writes = []

    def submit(*args):
        writes.append("submit")
        return []

    writer._do_layer_write = submit
    meta = ReqMeta(
        transfer_id="tid-abort",
        local_block_ids=[[1]],
        remote_block_ids=[],
        remote_host="127.0.0.2",
        remote_port=6301,
        remote_handshake_port=6301,
        remote_notify_port=61005,
        remote_engine_id="decode",
        tp_size=1,
        remote_dp_size=1,
    )
    task = WriteTask(
        request_id="req-abort",
        transfer_id=meta.transfer_id,
        dst_engine_id="decode",
        local_block_ids=[1],
        remote_block_ids_hint=None,
        layer_name="layer-0",
        event=SimpleNamespace(synchronize=lambda: None),
        remote_notify_port=61005,
        remote_ip="127.0.0.2",
    )
    return worker, writer, meta, task, writes, notifications


@pytest.mark.parametrize("allocation_arrived", [False, True])
def test_producer_abort_before_write_notifies_and_releases_once(allocation_arrived):
    worker, writer, meta, task, writes, notifications = _abort_setup()
    wrapper = worker.moriio_wrapper
    if allocation_arrived:
        wrapper.done_remote_allocate_req_dict[meta.transfer_id] = RemoteAllocInfo([2])

    assert writer.abort_before_write(task.request_id, meta)
    assert not writer.abort_before_write(task.request_id, meta)
    assert not writer.schedule_write(task)
    assert writes == []
    assert notifications == [(meta.transfer_id, "write_failed")]
    assert [ack.transfer_id for ack in wrapper.done_req_ids] == [meta.transfer_id]
    assert wrapper.done_remote_allocate_req_dict == {}


def test_cancelled_queued_write_stays_cancelled_after_terminal_history_eviction():
    worker, writer, meta, task, writes, notifications = _abort_setup()
    wrapper = worker.moriio_wrapper
    wrapper.done_remote_allocate_req_dict[meta.transfer_id] = RemoteAllocInfo([2])
    assert writer.schedule_write(task)
    assert writer.abort_before_write(task.request_id, meta)
    with (
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_engine."
            "_MAX_TERMINAL_TRANSFER_IDS",
            1,
        ),
        wrapper.lock,
    ):
        wrapper._mark_transfer_terminal_locked("other-transfer")
    assert meta.transfer_id not in wrapper._terminal_transfer_ids
    assert meta.transfer_id not in wrapper._aborted_write_endpoints

    writer._execute_write_task(writer._write_task_q.get_nowait())
    assert writes == []
    assert notifications == [(meta.transfer_id, "write_failed")]
    assert len(wrapper.done_req_ids) == 1


def test_late_allocation_gets_abort_reply_without_recreating_the_transfer():
    worker, writer, meta, task, writes, notifications = _abort_setup()
    wrapper = worker.moriio_wrapper
    assert writer.abort_before_write(task.request_id, meta)
    with patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_engine.get_role",
        return_value=ROLE.PRODUCER,
    ):
        for _ in range(2):
            wrapper._handle_remote_blocks_message(
                {
                    "transfer_id": meta.transfer_id,
                    "block_notify_list": [2],
                    "decode_rank": 0,
                }
            )
    assert len(notifications) == 3
    assert len(wrapper.done_req_ids) == 1
    assert wrapper.done_remote_allocate_req_dict == {}
    assert writes == []


@pytest.mark.parametrize("submit_wins", [False, True])
def test_abort_and_first_submit_have_one_winner(submit_wins):
    from concurrent.futures import ThreadPoolExecutor

    worker, writer, meta, task, writes, notifications = _abort_setup()
    wrapper = worker.moriio_wrapper
    wrapper.done_remote_allocate_req_dict[meta.transfer_id] = RemoteAllocInfo([2])
    reached = threading.Event()
    resume = threading.Event()

    def blocked():
        reached.set()
        assert resume.wait(5), "test failed to release the writer"

    if submit_wins:

        def submit(*args):
            writes.append("submit")
            blocked()
            return []

        writer._do_layer_write = submit
    else:
        task.event = SimpleNamespace(synchronize=blocked)
    assert writer.schedule_write(task)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(writer._execute_write_task, task)
        try:
            assert reached.wait(5), "writer did not reach the controlled race"
            assert writer.abort_before_write(task.request_id, meta) is not submit_wins
            if submit_wins:
                assert notifications == []
                assert wrapper.done_req_ids == []
        finally:
            resume.set()
        future.result(timeout=5)
    writer.seal_pending_transfers()
    if submit_wins:
        assert writes == ["submit"]
        assert notifications == [(meta.transfer_id, "write_done")]
    else:
        assert writes == []
        assert notifications == [(meta.transfer_id, "write_failed")]
    assert len(wrapper.done_req_ids) == 1


def test_submit_exception_cannot_be_reclassified_as_zero_write():
    worker, writer, meta, task, writes, notifications = _abort_setup()
    wrapper = worker.moriio_wrapper
    wrapper.done_remote_allocate_req_dict[meta.transfer_id] = RemoteAllocInfo([2])

    def partial_submit(*args):
        writes.append("possibly-posted")
        raise TransferError("failure after possible partial submit")

    writer._do_layer_write = partial_submit
    assert writer.schedule_write(task)
    with pytest.raises(TransferError):
        writer._execute_write_task(task)
    assert not writer.abort_before_write(task.request_id, meta)
    assert writes == ["possibly-posted"]
    assert notifications == []
    assert wrapper.done_req_ids == []


def test_zero_write_abort_does_not_claim_heterogeneous_tp_support():
    worker, writer, meta, task, writes, notifications = _abort_setup()
    meta.tp_size = 2
    assert not writer.abort_before_write(task.request_id, meta)
    assert notifications == []
    assert worker.moriio_wrapper.done_req_ids == []


def test_write_notifications_keep_zmq_sockets_on_the_creating_thread():
    from concurrent.futures import ThreadPoolExecutor

    wrapper = MoRIIOWrapper()
    rendezvous = threading.Barrier(2)
    owners = []

    class Socket:
        def __init__(self):
            self.owner = threading.get_ident()
            owners.append(self.owner)

        def send(self, payload):
            assert threading.get_ident() == self.owner

    def notify():
        rendezvous.wait(timeout=5)
        for _ in range(2):
            wrapper.send_notify("tid", "127.0.0.2", 61005, "write_failed")

    with (
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_engine."
            "make_zmq_socket",
            side_effect=lambda **kw: Socket(),
        ),
        ThreadPoolExecutor(max_workers=2) as executor,
    ):
        futures = [executor.submit(notify) for _ in range(2)]
        for future in futures:
            future.result(timeout=5)
    assert len(set(owners)) == 2
    assert len(wrapper.paths) == 2


def test_late_handshake_cannot_recreate_writes_after_abort_history_is_evicted():
    from concurrent.futures import Future
    from queue import Queue

    worker, writer, meta, task, writes, notifications = _abort_setup()
    futures = []

    def submit(*args):
        future: Future[Any] = Future()
        futures.append(future)
        return future

    worker._writer = writer
    worker.mode = MoRIIOMode.WRITE
    worker.is_producer = True
    worker.moriio_config.transfer_timeout = 0.0
    worker._handshake_futures = {}
    worker._handshake_initiation_executor = SimpleNamespace(submit=submit)
    worker._moriio_handshake = lambda *args: {"agent"}
    worker._handshake_lock = threading.Lock()
    worker._remote_agents = {}
    worker._ready_requests = Queue()
    worker.load_ready_flag = {}
    worker.write_ready_flags = {}
    worker._background_moriio_handshake = lambda *args: (
        MoRIIOConnectorWorker._background_moriio_handshake(worker, *args)
    )
    worker._write_blocks_for_req = lambda *args: (
        MoRIIOConnectorWorker._write_blocks_for_req(worker, *args)
    )
    worker.schedule_write_blocks = lambda **kw: (
        MoRIIOConnectorWorker.schedule_write_blocks(worker, **kw)
    )
    metadata = MoRIIOConnectorMetadata()
    metadata.reqs_to_save[task.request_id] = meta
    MoRIIOConnectorWorker.save_kv_layer(worker, metadata, "layer-0", None, None)
    assert writer.abort_before_write(task.request_id, meta)
    wrapper = worker.moriio_wrapper
    with (
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_engine."
            "_MAX_TERMINAL_TRANSFER_IDS",
            1,
        ),
        wrapper.lock,
    ):
        wrapper._mark_transfer_terminal_locked("other-transfer")
    # The actual background callback still owns the old ReqMeta.
    futures[0].set_result({"agent"})
    futures[1].set_result(True)
    with patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_engine.get_role",
        return_value=ROLE.PRODUCER,
    ):
        wrapper._handle_remote_blocks_message(
            {
                "transfer_id": meta.transfer_id,
                "block_notify_list": [2],
            }
        )
    with patch("torch.cuda.Event") as event:
        worker._write_blocks_for_req(
            *worker._ready_requests.get_nowait(), "layer-0", None
        )
    event.assert_not_called()
    assert writer._write_task_q.empty()
    assert writes == []
    assert notifications == [(meta.transfer_id, "write_failed")]
