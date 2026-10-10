# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""R1/R2 regression through the real producer scheduler, worker and writer."""

import time
from unittest.mock import patch

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common import (
    MoRIIOMode,
    RemoteAllocInfo,
    TransferError,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector import (
    MoRIIOConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_engine import (
    MoRIIOWriter,
)
from vllm.v1.request import RequestStatus

from .test_moriio_connector import _setup_kv_transfer_request, create_vllm_config
from .test_moriio_write_mode_stranding import _abort_setup
from .utils import create_model_runner_output, create_request, create_scheduler


def _producer_worker():
    transport, old_writer, meta, task, writes, notifications = _abort_setup()
    worker = MoRIIOConnectorWorker.__new__(MoRIIOConnectorWorker)
    worker.moriio_wrapper = transport.moriio_wrapper
    worker.moriio_config = transport.moriio_config
    worker.world_size = 1
    worker.tp_rank = 0
    worker.is_producer = True
    worker.mode = MoRIIOMode.WRITE
    worker.transfer_id_to_request_id = {}
    worker._pending_unmapped_acks = []
    worker._consumer_notification_counts = {}
    worker._completed_consumer_notifications = set()
    # Suppress real listener/CUDA/transport startup only.
    worker.moriio_wrapper.async_wait_reqid = lambda: None
    worker.get_engine_name_with_dp = transport.get_engine_name_with_dp
    worker._get_built_session = transport._get_built_session
    writer = worker._writer = MoRIIOWriter(worker)
    writer.ensure_worker_started = lambda: None
    writer._prepare_transfer_plan = old_writer._prepare_transfer_plan
    writer._do_layer_write = old_writer._do_layer_write
    return worker, writer, meta, task, writes, notifications


class _StopWriter(BaseException):
    pass


def _drain_one_queue_task(writer):
    original_get = writer._write_task_q.get
    consumed = False

    def get_once(*args, **kwargs):
        nonlocal consumed
        if consumed:
            raise _StopWriter
        consumed = True
        return original_get(*args, **kwargs)

    with (
        patch.object(writer._write_task_q, "get", get_once),
        pytest.raises(_StopWriter),
    ):
        writer._write_worker_loop()


def _evict(wrapper):
    with (
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.moriio."
            "moriio_engine._MAX_TERMINAL_TRANSFER_IDS",
            1,
        ),
        wrapper.lock,
    ):
        wrapper._mark_transfer_terminal_locked("later-transfer")


@pytest.mark.cpu_test
@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize("num_tokens", [32, 160])
@pytest.mark.parametrize("tp_key", ["remote_tp_size", "tp_size"])
def test_unsupported_abort_does_not_take_scheduler_ownership(num_tokens, tp_key):
    scheduler = create_scheduler(
        create_vllm_config(role="kv_producer", max_num_batched_tokens=64)
    )
    connector = scheduler.connector.connector_scheduler
    request = create_request(num_tokens=num_tokens, do_remote_decode=True)
    _setup_kv_transfer_request(request, fake_transfer_id="unsupported-tp")
    request.kv_transfer_params.pop("remote_tp_size", None)
    request.kv_transfer_params[tp_key] = 2
    pool = scheduler.kv_cache_manager.block_pool
    free_before = pool.get_num_free_blocks()
    scheduler.add_request(request)
    first = scheduler.schedule()
    assert bool(first.kv_connector_metadata.reqs_to_save) is (num_tokens == 32)
    assert pool.get_num_free_blocks() < free_before
    scheduler.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)
    # This is baseline parity, not a claim about heterogeneous WRITE safety.
    assert request.request_id not in scheduler.requests
    assert pool.get_num_free_blocks() == free_before
    assert not connector._deferred_send_deadlines
    assert not connector._reqs_need_abort
    assert not scheduler.schedule().kv_connector_metadata.reqs_to_abort
    assert not scheduler.has_requests()


@pytest.mark.cpu_test
@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize("path", ["early", "queued", "deferred"])
def test_scheduler_worker_writer_ack_reclaims_once_after_history_eviction(path):
    scheduler = create_scheduler(
        create_vllm_config(role="kv_producer", max_num_batched_tokens=64)
    )
    connector = scheduler.connector.connector_scheduler
    worker, writer, _, task, writes, notifications = _producer_worker()
    pool = scheduler.kv_cache_manager.block_pool
    free_before = pool.get_num_free_blocks()
    request = create_request(
        num_tokens=160 if path == "early" else 32, do_remote_decode=True
    )
    _setup_kv_transfer_request(request, fake_transfer_id=task.transfer_id)
    task.request_id = request.request_id
    scheduler.add_request(request)
    first = scheduler.schedule()
    worker.start_load_kv(first.kv_connector_metadata)
    if path != "early":
        assert request.request_id in first.kv_connector_metadata.reqs_to_save
        assert writer.schedule_write(task)
        task.enqueue_time = time.perf_counter() - 61
        if path == "deferred":
            writer._deferred_tasks.append(writer._write_task_q.get_nowait())
    held = pool.get_num_free_blocks()
    scheduler.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)
    assert pool.get_num_free_blocks() == held < free_before
    assert request.request_id in scheduler.requests
    step = scheduler.schedule()
    worker.start_load_kv(step.kv_connector_metadata)
    assert len(worker.moriio_wrapper.done_req_ids) == 1
    finished_sending, _ = worker.get_finished()
    assert finished_sending == {request.request_id}
    scheduler.update_from_output(
        step, create_model_runner_output([], finished_sending=finished_sending)
    )
    assert request.request_id not in scheduler.requests
    assert pool.get_num_free_blocks() == free_before
    assert not connector._deferred_send_deadlines
    worker.start_load_kv(scheduler.schedule().kv_connector_metadata)
    assert not worker.transfer_id_to_request_id
    _evict(worker.moriio_wrapper)
    if path == "queued":
        _drain_one_queue_task(writer)
    else:
        writer._process_deferred_tasks()
    assert not writer._deferred_tasks
    assert writer._write_task_q.empty()
    assert writes == []
    assert notifications == [(task.transfer_id, "write_failed")]
    for _ in range(3):
        assert worker.get_finished() == (set(), set())
        assert not worker._pending_unmapped_acks
    assert not writer._transfer_states
    assert not scheduler.has_requests()


@pytest.mark.parametrize("path", ["queued", "deferred"])
def test_exception_after_cancel_and_eviction_does_not_duplicate_ack(path):
    worker, writer, meta, task, writes, notifications = _producer_worker()
    wrapper = worker.moriio_wrapper
    assert writer.schedule_write(task)
    wrapper.done_remote_allocate_req_dict[task.transfer_id] = RemoteAllocInfo([2])

    def cancel_during_preparation():
        assert writer.abort_before_write(task.request_id, meta)
        # Consume the original ACK then remove its mapping, as scheduler does.
        worker.transfer_id_to_request_id = {task.transfer_id: task.request_id}
        assert worker.get_finished() == ({task.request_id}, set())
        worker.transfer_id_to_request_id = {}
        _evict(wrapper)
        raise TransferError("pre-submit preparation failed after cancellation")

    task.event.synchronize = cancel_during_preparation
    if path == "queued":
        _drain_one_queue_task(writer)
    else:
        writer._deferred_tasks.append(writer._write_task_q.get_nowait())
        writer._process_deferred_tasks()
    assert writes == []
    assert notifications == [(task.transfer_id, "write_failed")]
    assert wrapper.done_req_ids == []
    assert worker.get_finished() == (set(), set())
    assert not worker._pending_unmapped_acks


@pytest.mark.cpu_test
@pytest.mark.skip_global_cleanup
def test_submitted_write_abort_retains_producer_blocks_until_existing_completion():
    scheduler = create_scheduler(create_vllm_config(role="kv_producer"))
    worker, writer, _, task, writes, notifications = _producer_worker()
    wrapper = worker.moriio_wrapper
    pool = scheduler.kv_cache_manager.block_pool
    free_before = pool.get_num_free_blocks()
    request = create_request(num_tokens=32, do_remote_decode=True)
    _setup_kv_transfer_request(request, fake_transfer_id=task.transfer_id)
    task.request_id = request.request_id
    scheduler.add_request(request)
    first = scheduler.schedule()
    worker.start_load_kv(first.kv_connector_metadata)
    assert writer.schedule_write(task)
    wrapper.done_remote_allocate_req_dict[task.transfer_id] = RemoteAllocInfo([2])
    writer._execute_write_task(writer._write_task_q.get_nowait())
    assert writes == ["submit"]
    assert task.transfer_state.submit_started
    assert not wrapper.done_req_ids
    scheduler.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)
    step = scheduler.schedule()
    worker.start_load_kv(step.kv_connector_metadata)
    assert notifications == []
    assert worker.get_finished() == (set(), set())
    scheduler.update_from_output(step, create_model_runner_output([]))
    assert request.request_id in scheduler.requests
    assert pool.get_num_free_blocks() < free_before
    # No synthetic timeout/abort ACK: only the existing completion may release.
    writer.seal_pending_transfers()
    finished_sending, _ = worker.get_finished()
    assert finished_sending == {request.request_id}
    scheduler.update_from_output(
        scheduler.schedule(),
        create_model_runner_output([], finished_sending=finished_sending),
    )
    assert request.request_id not in scheduler.requests
    assert pool.get_num_free_blocks() == free_before
    assert notifications == [(task.transfer_id, "write_done")]


@pytest.mark.parametrize("path", ["queued", "deferred"])
def test_cancel_between_ready_check_and_timeout_terminalization(path):
    worker, writer, meta, task, writes, notifications = _producer_worker()
    wrapper = worker.moriio_wrapper
    assert writer.schedule_write(task)
    task.enqueue_time = time.perf_counter() - 61

    def cancel_at_ready_check(_task):
        assert writer.abort_before_write(task.request_id, meta)
        worker.transfer_id_to_request_id = {task.transfer_id: task.request_id}
        assert worker.get_finished() == ({task.request_id}, set())
        worker.transfer_id_to_request_id = {}
        _evict(wrapper)
        return False

    writer._is_remote_ready = cancel_at_ready_check
    if path == "queued":
        _drain_one_queue_task(writer)
    else:
        writer._deferred_tasks.append(writer._write_task_q.get_nowait())
        writer._process_deferred_tasks()
    assert notifications == [(task.transfer_id, "write_failed")]
    assert writes == []
    assert not wrapper.done_req_ids
    assert not writer._deferred_tasks
    assert worker.get_finished() == (set(), set())
    assert not worker._pending_unmapped_acks


def test_deferred_timeout_cannot_classify_submit_attempt_as_zero_submit():
    worker, writer, _, task, writes, notifications = _producer_worker()
    assert writer.schedule_write(task)
    task.transfer_state.submit_started = True
    assert not writer._fail_deferred_task(task, age=120)
    assert not worker.moriio_wrapper.done_req_ids
    assert notifications == []
    assert not task.transfer_state.cancelled
