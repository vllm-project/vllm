# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the two WRITE-mode paths that can strand a decode request in
``WAITING_FOR_REMOTE_KVS``.

In WRITE mode the producer pushes KV, so the consumer parks the request until
the push lands. Both the scheduler-side admission of that wait and the
producer-side write queue must avoid leaving a request parked forever.

Neither path touches MoRI-IO state beyond a handful of plain attributes, so
they are bound to lightweight stand-ins rather than constructing a full
scheduler or writer.
"""

import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common import (
    ROLE,
    MoRIIOMode,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector import (
    MoRIIOConnectorScheduler,
    MoRIIOConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_engine import (
    MoRIIOWrapper,
    MoRIIOWriter,
)

_get_num_new_matched_tokens = MoRIIOConnectorScheduler.get_num_new_matched_tokens
_update_connector_output = MoRIIOConnectorScheduler.update_connector_output
_process_deferred_tasks = MoRIIOWriter._process_deferred_tasks
_fail_deferred_task = MoRIIOWriter._fail_deferred_task
_get_transfer_results = MoRIIOConnectorWorker.get_transfer_results

PROMPT_LEN = 8


def _consumer(mode=MoRIIOMode.WRITE):
    return SimpleNamespace(is_producer=False, mode=mode)


def _request(do_remote_prefill: bool | None):
    """A request with ``do_remote_prefill`` set, cleared, or no params at all."""
    params = (
        None if do_remote_prefill is None else {"do_remote_prefill": do_remote_prefill}
    )
    return SimpleNamespace(
        prompt_token_ids=list(range(PROMPT_LEN)),
        num_prompt_tokens=PROMPT_LEN,
        kv_transfer_params=params,
    )


def test_write_mode_requests_async_load_for_pending_remote_prefill():
    count, async_load = _get_num_new_matched_tokens(
        _consumer(), _request(do_remote_prefill=True), 0
    )
    assert (count, async_load) == (PROMPT_LEN, True)


def test_write_mode_requests_only_the_uncached_remainder():
    count, async_load = _get_num_new_matched_tokens(
        _consumer(), _request(do_remote_prefill=True), 3
    )
    assert (count, async_load) == (PROMPT_LEN - 3, True)


def test_write_mode_full_local_hit_is_not_an_async_load():
    # An async load of zero tokens trips the scheduler's
    # `assert num_external_computed_tokens > 0`.
    count, async_load = _get_num_new_matched_tokens(
        _consumer(), _request(do_remote_prefill=True), PROMPT_LEN
    )
    assert (count, async_load) == (0, False)


def test_write_mode_does_not_request_a_second_load_after_the_push_landed():
    # update_state_after_alloc clears do_remote_prefill once the transfer is
    # triggered. The scheduler promotes the request with
    # num_computed_tokens == num_tokens - 1 and asks again; answering yes here
    # would park it for a second push that the producer never sends.
    count, async_load = _get_num_new_matched_tokens(
        _consumer(), _request(do_remote_prefill=False), PROMPT_LEN - 1
    )
    assert (count, async_load) == (0, False)


def test_write_mode_without_kv_transfer_params_is_local_work():
    count, async_load = _get_num_new_matched_tokens(
        _consumer(), _request(do_remote_prefill=None), 0
    )
    assert (count, async_load) == (0, False)


def test_producer_never_requests_an_async_load():
    producer = SimpleNamespace(is_producer=True, mode=MoRIIOMode.WRITE)
    count, async_load = _get_num_new_matched_tokens(
        producer, _request(do_remote_prefill=True), 0
    )
    assert (count, async_load) == (0, False)


def test_read_mode_is_unchanged():
    count, async_load = _get_num_new_matched_tokens(
        _consumer(mode=MoRIIOMode.READ), _request(do_remote_prefill=True), 2
    )
    assert (count, async_load) == (PROMPT_LEN - 1 - 2, False)


def _task(transfer_id="tid-1", age=0.0):
    return SimpleNamespace(
        request_id=f"req-for-{transfer_id}",
        transfer_id=transfer_id,
        enqueue_time=time.perf_counter() - age,
        remote_ip="127.0.0.1",
        remote_notify_port=8501,
        remote_dp_rank=0,
        multi_pod_hosts=["127.0.0.1"],
        remote_dp_size_local=1,
    )


def _writer(tasks, *, remote_ready, terminal=()):
    executed: list[SimpleNamespace] = []
    failed: list[tuple[SimpleNamespace, float]] = []

    def fail(task, age):
        failed.append((task, age))
        return True

    writer = SimpleNamespace(
        _deferred_tasks=list(tasks),
        _defer_timeout=60.0,
        _is_transfer_terminal=lambda tid: tid in terminal,
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


def test_deferred_write_runs_once_the_remote_allocation_arrives():
    task = _task(age=120.0)
    writer, executed, failed = _writer([task], remote_ready=True)

    _process_deferred_tasks(writer)

    assert writer._deferred_tasks == []
    assert executed == [task]
    assert failed == []


def test_deferred_write_for_a_terminal_transfer_is_dropped():
    task = _task(transfer_id="tid-done")
    writer, executed, failed = _writer([task], remote_ready=True, terminal={"tid-done"})

    _process_deferred_tasks(writer)

    assert writer._deferred_tasks == []
    assert executed == []
    assert failed == []


def test_timeout_marks_terminal_before_releasing_blocks():
    events: list[str] = []

    class Wrapper:
        def __init__(self):
            self.lock = threading.Lock()
            self.done_req_ids = []
            self.done_remote_allocate_req_dict = {}
            self.terminal: set[str] = set()

        def _is_transfer_terminal_locked(self, transfer_id):
            return transfer_id in self.terminal

        def _mark_transfer_terminal_locked(self, transfer_id):
            self.terminal.add(transfer_id)
            events.append("terminal")

        def send_notify(self, transfer_id, host, port, message_type):
            assert transfer_id in self.terminal
            assert self.done_req_ids == []
            assert (host, port, message_type) == (
                "127.0.0.1",
                8501,
                "write_failed",
            )
            events.append("failure_sent")

    wrapper = Wrapper()
    writer = SimpleNamespace(
        worker=SimpleNamespace(moriio_wrapper=wrapper, tp_rank=0),
        _clear_transfer_state=lambda transfer_id: events.append("state_cleared"),
    )
    writer._resolve_notify_endpoint = lambda task, rank: (
        MoRIIOWriter._resolve_notify_endpoint(writer, task, rank)
    )

    _fail_deferred_task(writer, _task(), 120.0)

    assert events == ["terminal", "state_cleared", "failure_sent"]
    assert wrapper.done_remote_allocate_req_dict == {}
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


def test_write_failure_is_reported_as_finished_and_failed_recving():
    wrapper = SimpleNamespace(pop_failed_write_req_ids=lambda: {"tid-1"})
    worker = SimpleNamespace(
        mode=MoRIIOMode.WRITE,
        is_producer=False,
        moriio_wrapper=wrapper,
        transfer_id_to_request_id={"tid-1": "req-1"},
        _unmatched_write_failures=set(),
        get_finished=lambda: (set(), set()),
    )

    results = _get_transfer_results(worker)

    assert results.finished_recving == {"req-1"}
    assert results.failed_recving == {"req-1"}
    assert worker._unmatched_write_failures == set()


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


def test_remote_allocation_is_ignored_after_timeout_marks_terminal():
    wrapper = MoRIIOWrapper.__new__(MoRIIOWrapper)
    wrapper.lock = threading.Lock()
    wrapper.done_remote_allocate_req_dict = {}
    wrapper._terminal_transfer_ids = {"tid-1": None}

    with patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_engine.get_role",
        return_value=ROLE.PRODUCER,
    ):
        wrapper._handle_remote_blocks_message(
            {
                "transfer_id": "tid-1",
                "block_notify_list": [1],
                "decode_rank": 0,
            }
        )

    assert wrapper.done_remote_allocate_req_dict == {}
