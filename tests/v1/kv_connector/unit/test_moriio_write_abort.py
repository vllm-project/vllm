# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MoRIIO WRITE cancellation before the first transport submission.

Writer cases drive the real ``MoRIIOWriter``/``MoRIIOWrapper``; only the
transfer plan, sessions and notification socket are replaced. Scheduler cases
add the real producer scheduler, allocator and connector worker.
"""

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common import (
    LayerTransferPlan,
    MoRIIOMode,
    RemoteAllocInfo,
    ReqMeta,
    TransferError,
    WriteTask,
    WriteTransferState,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector import (
    MoRIIOConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_engine import (
    MoRIIOWrapper,
    MoRIIOWriter,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_tp_mapping import (
    MoRIIOWriteTPMapping,
    WriteTarget,
)
from vllm.v1.request import RequestStatus

from .test_moriio_connector import _setup_kv_transfer_request, create_vllm_config
from .utils import create_model_runner_output, create_request, create_scheduler

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


def _setup(*, targets=1, mamba=False):
    """Writer for one transfer. Two targets only exercise writer iteration;
    they are not a negotiated heterogeneous topology."""
    attempts: list[str] = []
    notifications: list[tuple[str, str]] = []
    controls = SimpleNamespace(
        before_plan=lambda task, target: None, on_submit=lambda: None
    )
    wrapper = MoRIIOWrapper()
    wrapper.local_memory_registered = True
    wrapper.moriio_engine = SimpleNamespace(allocate_transfer_uid=lambda: 1)
    wrapper.send_notify = lambda tid, host, port, message_type, **kw: (
        notifications.append((tid, message_type))
    )

    def submit(engine_id):
        attempts.append(engine_id)
        controls.on_submit()
        return SimpleNamespace(Failed=lambda: False, Succeeded=lambda: True)

    worker = MoRIIOConnectorWorker.__new__(MoRIIOConnectorWorker)
    vars(worker).update(
        world_size=1,
        tp_rank=0,
        moriio_wrapper=wrapper,
        moriio_config=SimpleNamespace(defer_timeout=60.0),
        is_producer=True,
        mode=MoRIIOMode.WRITE,
        _has_mamba=False,
        layer_to_spec={},
        transfer_id_to_request_id={},
        _pending_unmapped_acks=[],
        _consumer_notification_counts={},
        _completed_consumer_notifications=set(),
        get_engine_name_with_dp=lambda engine, dp: f"{engine}_dp{dp}",
        get_engine_name_with_dp_tp=lambda engine, dp, tp: f"{engine}_dp{dp}_tp{tp}",
        _is_mamba_layer=lambda layer_name: mamba,
        _get_built_session=lambda engine: (
            [SimpleNamespace(batch_write=lambda *args: submit(engine))],
            None,
        ),
    )
    wrapper.async_wait_reqid = lambda: None
    writer = worker._writer = MoRIIOWriter(worker)
    writer.ensure_worker_started = lambda: None

    def prepare(task, info, remote_meta, *, remote_engine_id=None, target=None):
        controls.before_plan(task, target)
        return LayerTransferPlan(
            task.request_id, task.transfer_id, task.layer_name, 0, [0], [0], [8]
        )

    writer._prepare_transfer_plan = prepare
    writer._prepare_mamba_transfer_plans = lambda task, info, target, meta: [
        prepare(task, info, meta, target=target)
    ]
    meta = ReqMeta(
        transfer_id="transfer",
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
    write_targets = tuple(
        WriteTarget(rank, 0, 0, 1, 0, 1, 0, 1, 1) for rank in range(targets)
    )
    task = WriteTask(
        request_id="request",
        transfer_id=meta.transfer_id,
        dst_engine_id="decode",
        local_block_ids=[[1]],
        remote_block_ids_hint=None,
        layer_name="layer-0",
        event=SimpleNamespace(synchronize=lambda: None),
        remote_notify_port=61005,
        remote_ip="127.0.0.2",
        remote_tp_size=1,
        tp_mapping=(
            MoRIIOWriteTPMapping(1, 1, 1, 1, write_targets) if targets > 1 else None
        ),
    )
    return SimpleNamespace(
        worker=worker,
        writer=writer,
        wrapper=wrapper,
        meta=meta,
        task=task,
        controls=controls,
        attempts=attempts,
        notifications=notifications,
    )


def _allocate(h):
    h.allocation = RemoteAllocInfo([[2]])
    h.wrapper.done_remote_allocate_req_dict[h.task.transfer_id] = h.allocation


def _acks(h):
    return [ack.transfer_id for ack in h.wrapper.done_req_ids]


def _run_deferred(h, task=None):
    assert h.writer.schedule_write(h.task if task is None else task)
    h.writer._deferred_tasks.append(h.writer._write_task_q.get_nowait())
    h.writer._process_deferred_tasks()


def _assert_held(h):
    """A possible submission keeps ownership; abort or timeout cannot release it."""
    assert h.attempts
    assert h.task.transfer_state.submit_started
    assert not h.writer.abort_before_write(h.task.request_id, h.meta)
    assert not h.writer._fail_deferred_task(h.task, age=120)
    assert h.wrapper.done_req_ids == []
    assert h.notifications == []
    assert h.wrapper.done_remote_allocate_req_dict[h.task.transfer_id] is h.allocation


@pytest.mark.parametrize("allocation_arrived", [False, True])
def test_abort_before_submit_notifies_and_acks_once(allocation_arrived):
    h = _setup()
    if allocation_arrived:
        _allocate(h)

    # Cancellation is only claimed for homogeneous TP.
    assert not h.writer.abort_before_write("request", replace(h.meta, tp_size=2))
    assert h.writer.abort_before_write(h.task.request_id, h.meta)
    assert not h.writer.abort_before_write(h.task.request_id, h.meta)
    assert not h.writer.schedule_write(h.task)
    assert h.attempts == []
    assert h.notifications == [("transfer", "write_failed")]
    assert _acks(h) == ["transfer"]
    assert h.wrapper.done_remote_allocate_req_dict == {}


@pytest.mark.parametrize("submit_wins", [False, True])
@pytest.mark.parametrize("mamba", [False, True], ids=["attention", "mamba"])
def test_abort_and_first_submit_have_one_winner(submit_wins, mamba):
    h = _setup(mamba=mamba)
    _allocate(h)
    reached, resume = threading.Event(), threading.Event()

    def pause():
        reached.set()
        assert resume.wait(5), "test failed to release the writer"

    if submit_wins:
        h.controls.on_submit = pause
    else:
        h.task.event = SimpleNamespace(synchronize=pause)
    assert h.writer.schedule_write(h.task)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(h.writer._execute_write_task, h.task)
        try:
            assert reached.wait(5), "writer did not reach the race"
            aborted = h.writer.abort_before_write(h.task.request_id, h.meta)
            assert aborted is not submit_wins
            if submit_wins:
                assert h.notifications == []
                assert h.wrapper.done_req_ids == []
        finally:
            resume.set()
        future.result(timeout=5)
    h.writer.seal_pending_transfers()

    if submit_wins:
        assert h.attempts == ["decode_dp0"]
        assert h.notifications == [("transfer", "write_done")]
    else:
        assert h.attempts == []
        assert h.notifications == [("transfer", "write_failed")]
    assert _acks(h) == ["transfer"]


@pytest.mark.parametrize(
    "failure", ["submit_raises", "second_target_prepare", "later_layer_prepare"]
)
def test_failure_after_possible_submit_keeps_ownership(failure):
    h = _setup(targets=2)
    _allocate(h)
    if failure == "submit_raises":
        # The transport may have accepted part of the write before raising.
        def raise_after_attempt():
            raise TransferError("submission outcome unknown")

        h.controls.on_submit = raise_after_attempt
        _run_deferred(h)
        assert h.attempts == ["decode_dp0_tp0"]
    elif failure == "second_target_prepare":

        def fail_second(task, target):
            if target.tp_rank == 1:
                raise TransferError("second target preparation failed")

        h.controls.before_plan = fail_second
        _run_deferred(h)
        assert h.attempts == ["decode_dp0_tp0"]
        # The first target's status is kept for the existing completion wait.
        assert len(h.allocation.transfer_statuses) == 1
    else:
        # Separate state objects for the two layers still share one gate.
        h.task.transfer_state = WriteTransferState()
        later = replace(h.task, layer_name="layer-1")
        later.transfer_state = WriteTransferState()
        assert h.writer.schedule_write(h.task)
        h.writer._execute_write_task(h.writer._write_task_q.get_nowait())
        late_meta = replace(h.meta, transfer_state=WriteTransferState())
        assert h.writer.bind_transfer(late_meta)
        assert late_meta.transfer_state is h.task.transfer_state

        def fail(task, target):
            raise TransferError("later layer preparation failed")

        h.controls.before_plan = fail
        _run_deferred(h, later)
        assert later.transfer_state is h.task.transfer_state
    _assert_held(h)


def _add_producer_request(scheduler, h, num_tokens):
    request = create_request(num_tokens=num_tokens, do_remote_decode=True)
    _setup_kv_transfer_request(request, fake_transfer_id=h.task.transfer_id)
    h.task.request_id = request.request_id
    scheduler.add_request(request)
    return request


def _deliver_ack(scheduler, h, step):
    finished_sending, _ = h.worker.get_finished()
    assert finished_sending == {h.task.request_id}
    scheduler.update_from_output(
        step, create_model_runner_output([], finished_sending=finished_sending)
    )


@pytest.mark.parametrize("path", ["never_scheduled", "queued"])
def test_producer_scheduler_frees_once_on_worker_ack(monkeypatch, path):
    scheduler = create_scheduler(
        create_vllm_config(role="kv_producer", max_num_batched_tokens=64)
    )
    h = _setup()
    worker = h.worker
    pool = scheduler.kv_cache_manager.block_pool
    free_before = pool.get_num_free_blocks()
    request = _add_producer_request(scheduler, h, 32)
    free = MagicMock(wraps=scheduler.kv_cache_manager.free)
    monkeypatch.setattr(scheduler.kv_cache_manager, "free", free)
    if path != "never_scheduled":
        worker.start_load_kv(scheduler.schedule().kv_connector_metadata)
    if path == "queued":
        assert h.writer.schedule_write(h.task)

    scheduler.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)
    # Blocks stay with the request until the worker closes the gate.
    assert request.request_id in scheduler.requests
    assert free.call_count == 0
    step = scheduler.schedule()
    assert step.total_num_scheduled_tokens == 0
    assert set(step.kv_connector_metadata.reqs_to_abort) == {request.request_id}
    worker.start_load_kv(step.kv_connector_metadata)
    _deliver_ack(scheduler, h, step)

    assert request.request_id not in scheduler.requests
    assert pool.get_num_free_blocks() == free_before
    assert not scheduler.has_requests()
    # A stale queued task adds no write, failure or ACK.
    worker.start_load_kv(scheduler.schedule().kv_connector_metadata)
    if path == "queued":
        h.writer._execute_write_task(h.writer._write_task_q.get_nowait())
    assert worker.get_finished() == (set(), set())
    assert h.attempts == []
    assert h.notifications == [("transfer", "write_failed")]
    assert free.call_count == 1


@pytest.mark.parametrize("tp_key", ["remote_tp_size", "tp_size"])
def test_heterogeneous_producer_abort_keeps_existing_behavior(tp_key):
    scheduler = create_scheduler(create_vllm_config(role="kv_producer"))
    connector = scheduler.connector.connector_scheduler
    request = create_request(num_tokens=32, do_remote_decode=True)
    _setup_kv_transfer_request(request, fake_transfer_id="hetero")
    request.kv_transfer_params.pop("remote_tp_size", None)
    request.kv_transfer_params[tp_key] = 2
    pool = scheduler.kv_cache_manager.block_pool
    free_before = pool.get_num_free_blocks()
    scheduler.add_request(request)
    scheduler.schedule()

    scheduler.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)
    assert request.request_id not in scheduler.requests
    assert pool.get_num_free_blocks() == free_before
    assert not connector._reqs_need_abort
    assert not connector._deferred_send_deadlines
