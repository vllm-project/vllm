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

import time
from types import SimpleNamespace

from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common import (
    MoRIIOMode,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector import (
    MoRIIOConnectorScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_engine import (
    MoRIIOWriter,
)

_get_num_new_matched_tokens = MoRIIOConnectorScheduler.get_num_new_matched_tokens
_process_deferred_tasks = MoRIIOWriter._process_deferred_tasks

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
    )


def _writer(tasks, *, remote_ready, terminal=()):
    executed: list[SimpleNamespace] = []
    marked_done: list[str] = []
    writer = SimpleNamespace(
        _deferred_tasks=list(tasks),
        _defer_timeout=60.0,
        _is_transfer_terminal=lambda tid: tid in terminal,
        _is_remote_ready=lambda task: remote_ready,
        _execute_write_task=executed.append,
        _mark_request_done=marked_done.append,
    )
    return writer, executed, marked_done


def test_expired_write_keeps_waiting_for_the_remote_allocation():
    # Marking the transfer done frees the producer blocks without notifying
    # the consumer, so the consumer waits for KV that can no longer arrive.
    task = _task(age=120.0)
    writer, executed, marked_done = _writer([task], remote_ready=False)

    _process_deferred_tasks(writer)

    assert writer._deferred_tasks == [task]
    assert executed == []
    assert marked_done == []


def test_expired_write_restarts_its_interval():
    task = _task(age=120.0)
    writer, _, _ = _writer([task], remote_ready=False)

    _process_deferred_tasks(writer)

    assert time.perf_counter() - task.enqueue_time < 60.0


def test_deferred_write_runs_once_the_remote_allocation_arrives():
    task = _task(age=120.0)
    writer, executed, marked_done = _writer([task], remote_ready=True)

    _process_deferred_tasks(writer)

    assert writer._deferred_tasks == []
    assert executed == [task]
    assert marked_done == []


def test_deferred_write_for_a_terminal_transfer_is_dropped():
    task = _task(transfer_id="tid-done")
    writer, executed, marked_done = _writer(
        [task], remote_ready=True, terminal={"tid-done"}
    )

    _process_deferred_tasks(writer)

    assert writer._deferred_tasks == []
    assert executed == []
    assert marked_done == []
