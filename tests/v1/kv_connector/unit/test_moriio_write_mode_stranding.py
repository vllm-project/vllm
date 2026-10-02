# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for MoRI-IO WRITE-mode requests stranded in WAITING_FOR_REMOTE_KVS."""

import threading
import time
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import patch

from vllm.distributed.kv_transfer.kv_connector.v1.moriio import moriio_connector
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common import (
    ROLE,
    MoRIIOMode,
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
        if send_notify is not None:
            self.send_notify = send_notify

    def _is_transfer_terminal_locked(self, transfer_id):
        return transfer_id in self.terminal

    def _mark_transfer_terminal_locked(self, transfer_id):
        self.terminal.add(transfer_id)
        self.events.append("terminal")

    def send_notify(self, transfer_id, host, port, message_type):
        assert transfer_id in self.terminal
        assert self.done_req_ids == []
        self.sent.append((host, port, message_type))


def _failing_writer(wrapper, events):
    writer = SimpleNamespace(
        worker=SimpleNamespace(moriio_wrapper=wrapper, tp_rank=0),
        _clear_transfer_state=lambda transfer_id: events.append("state_cleared"),
    )
    writer._resolve_notify_endpoint = lambda task, rank: (
        MoRIIOWriter._resolve_notify_endpoint(writer, task, rank)
    )
    return writer


def test_timeout_notifies_every_decode_dp_rank_before_releasing_blocks():
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
    writer = SimpleNamespace(
        worker=SimpleNamespace(moriio_wrapper=wrapper),
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
