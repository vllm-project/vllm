# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""WRITE abort arbitration; sessions/events replace the CPU-unavailable transport."""

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace

import msgpack
import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.moriio import moriio_engine
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common import (
    ROLE,
    LayerTransferPlan,
    MoRIIOConnectorMetadata,
    MoRIIOMode,
    RemoteAllocInfo,
    ReqMeta,
    TransferError,
    WriteTask,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector import (
    MoRIIOConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_engine import (
    MoRIIOWrapper,
    MoRIIOWriter,
)
from vllm.v1.request import RequestStatus

from ..utils import create_model_runner_output, create_request, create_scheduler
from .test_moriio_connector import _setup_kv_transfer_request, create_vllm_config

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


def _setup(*, mamba=False, empty=False):
    attempts, notifications = [], []
    controls = SimpleNamespace(on_submit=lambda: None)
    wrapper = MoRIIOWrapper()
    wrapper.local_memory_registered = True
    wrapper.moriio_engine = SimpleNamespace(allocate_transfer_uid=lambda: 1)
    vars(wrapper).update(
        send_notify=lambda tid, host, port, message_type, **kw: notifications.append(
            (tid, message_type)
        )
    )

    def submit(*args):
        attempts.append("WRITE")
        controls.on_submit()
        return SimpleNamespace(Failed=lambda: False, Succeeded=lambda: True)

    worker = MoRIIOConnectorWorker.__new__(MoRIIOConnectorWorker)
    vars(worker).update(
        world_size=1,
        tp_rank=0,
        pp_rank=0,
        pp_size=1,
        moriio_wrapper=wrapper,
        moriio_config=SimpleNamespace(defer_timeout=60),
        is_producer=True,
        mode=MoRIIOMode.WRITE,
        transfer_id_to_request_id={},
        _pending_unmapped_acks=[],
        _consumer_notification_counts={},
        _completed_consumer_notifications=set(),
        _completed_notification_owners={},
        get_engine_name_with_dp=lambda engine, dp: f"{engine}_dp{dp}",
        _is_mamba_layer=lambda layer: mamba,
        _get_built_session=lambda engine: ([SimpleNamespace(batch_write=submit)], None),
    )
    vars(wrapper).update(async_wait_reqid=lambda: None)
    writer = worker._writer = MoRIIOWriter(worker)
    vars(writer).update(ensure_worker_started=lambda: None)
    meta = ReqMeta("transfer", [[1]], [], "127.0.0.2", 7000, 7000, 7000, "decode", 1, 1)
    task = WriteTask(
        "request",
        meta.transfer_id,
        "decode",
        [[1]],
        None,
        "layer",
        SimpleNamespace(synchronize=lambda: None),
        7000,
        "127.0.0.2",
        remote_tp_size=1,
    )

    def prepare(*args, **kwargs):
        return LayerTransferPlan(
            task.request_id,
            task.transfer_id,
            task.layer_name,
            0,
            [] if empty else [0],
            [] if empty else [0],
            [] if empty else [8],
        )

    vars(writer).update(
        _prepare_transfer_plan=prepare,
        _prepare_mamba_transfer_plans=lambda *args: [prepare()],
    )
    allocation = RemoteAllocInfo([[2]])
    wrapper.done_remote_allocate_req_dict[meta.transfer_id] = allocation
    return SimpleNamespace(
        worker=worker,
        writer=writer,
        wrapper=wrapper,
        meta=meta,
        task=task,
        allocation=allocation,
        controls=controls,
        attempts=attempts,
        notifications=notifications,
    )


def _run(h, task=None):
    assert h.writer.schedule_write(task or h.task)
    h.writer._execute_write_task(h.writer._write_task_q.get_nowait())


def _acks(h):
    return [ack.transfer_id for ack in h.wrapper.done_req_ids]


def test_empty_plan_does_not_close_submission_gate(monkeypatch):
    h = _setup(empty=True)
    assert h.writer.bind_transfer(h.meta)
    h.task.transfer_state = h.meta.transfer_state
    _run(h)
    assert h.writer.abort_before_write(h.task.request_id, h.meta)
    assert not h.writer.abort_before_write(h.task.request_id, h.meta)
    assert h.meta.transfer_state.cancelled
    assert not h.writer.bind_transfer(h.meta)
    h.writer.seal_pending_transfers()
    h.writer._finalize_if_complete(h.task.transfer_id, h.allocation)
    assert h.attempts == []
    assert h.notifications == [("transfer", "write_failed")]
    assert _acks(h) == ["transfer"]
    # Late allocation is ignored; it must not cause another notification/ACK.
    monkeypatch.setattr(moriio_engine, "get_role", lambda: ROLE.PRODUCER)
    h.wrapper._handle_message(
        msgpack.dumps(
            {
                "type": "remote_blocks",
                "transfer_id": "transfer",
                "block_notify_list": [2],
            }
        )
    )
    assert not h.wrapper.done_remote_allocate_req_dict
    assert h.notifications == [("transfer", "write_failed")]
    consumer = MoRIIOWrapper()
    monkeypatch.setattr(moriio_engine, "get_role", lambda: ROLE.CONSUMER)
    for kind in ["write_failed", "write_failed", "write_done"]:
        consumer._handle_message(
            msgpack.dumps({"type": kind, "transfer_id": "transfer"})
        )
    assert consumer.pop_failed_write_req_ids() == {"transfer"}
    assert not consumer.pop_failed_write_req_ids()
    assert not consumer.pop_finished_write_req_ids()


@pytest.mark.parametrize("submit_wins", [False, True])
@pytest.mark.parametrize("mamba", [False, True])
def test_abort_and_first_submit_have_one_winner(submit_wins, mamba):
    h = _setup(mamba=mamba)
    reached, resume = threading.Event(), threading.Event()

    def pause():
        reached.set()
        assert resume.wait(5)

    if submit_wins:
        h.controls.on_submit = pause
    else:
        h.task.event = SimpleNamespace(synchronize=pause)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_run, h)
        try:
            assert reached.wait(5)
            assert (
                h.writer.abort_before_write(h.task.request_id, h.meta)
                is not submit_wins
            )
        finally:
            resume.set()
        future.result(timeout=5)
    h.writer.seal_pending_transfers()
    assert len(h.attempts) == int(submit_wins)
    assert h.notifications == [
        ("transfer", "write_done" if submit_wins else "write_failed")
    ]
    assert _acks(h) == ["transfer"]


def test_empty_finalize_claim_keeps_abort_out():
    h = _setup(empty=True)
    _run(h)
    reached, resume = threading.Event(), threading.Event()

    def wait_for_transport(statuses):
        assert statuses == []
        reached.set()
        assert resume.wait(5)

    h.wrapper.waiting_for_transfer_complete = wait_for_transport
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(h.writer.seal_pending_transfers)
        try:
            assert reached.wait(5)
            assert not h.writer.abort_before_write(h.task.request_id, h.meta)
        finally:
            resume.set()
        future.result(timeout=5)
    h.writer._finalize_if_complete(h.task.transfer_id, h.allocation)
    assert h.attempts == []
    assert h.notifications == [("transfer", "write_done")]
    assert _acks(h) == ["transfer"]


@pytest.mark.parametrize("later_layer", [False, True])
def test_possible_submit_failure_releases_source_only(later_layer):
    h = _setup()
    if later_layer:
        _run(h)
        task = replace(h.task, layer_name="later", transfer_state=None)
    else:
        task = h.task

    def fail(*args, **kw):
        raise TransferError("submission outcome unknown")

    if later_layer:
        h.writer._prepare_transfer_plan = fail
    else:
        h.controls.on_submit = fail
    assert h.writer.schedule_write(task)
    h.writer._deferred_tasks.append(h.writer._write_task_q.get_nowait())
    h.writer._process_deferred_tasks()
    assert h.attempts == ["WRITE"]
    assert not h.writer.abort_before_write(h.task.request_id, h.meta)
    assert h.notifications == []
    assert _acks(h) == ["transfer"]
    h.writer._mark_request_done(h.task.transfer_id)
    assert _acks(h) == ["transfer"]


@pytest.mark.parametrize(
    ("allocated", "missing_peer_fields"), [(False, False), (False, True), (True, False)]
)
def test_abort_before_alloc_uses_configured_peer_pods(allocated, missing_peer_fields):
    config = create_vllm_config(role="kv_producer")
    config.parallel_config.tensor_parallel_size = 2
    assert config.kv_transfer_config is not None
    config.kv_transfer_config.kv_connector_extra_config.update(
        remote_hosts=["10.0.0.1", "10.0.0.2"], remote_dp_size_local=8
    )
    scheduler = create_scheduler(config)
    request = create_request(num_tokens=32, do_remote_decode=True)
    _setup_kv_transfer_request(request, fake_transfer_id="transfer", fake_port=7000)
    assert request.kv_transfer_params is not None
    request.kv_transfer_params.update(remote_dp_size=16, remote_tp_size=2)
    if missing_peer_fields:
        for key in ["remote_engine_id", "remote_block_ids"]:
            del request.kv_transfer_params[key]
    pool = scheduler.kv_cache_manager.block_pool
    free_before = pool.get_num_free_blocks()
    scheduler.add_request(request)
    if allocated:
        scheduler.schedule()
        assert pool.get_num_free_blocks() < free_before
    scheduler.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)
    step = scheduler.schedule()
    h = _setup()
    h.wrapper.done_remote_allocate_req_dict.clear()
    h.worker.world_size, h.worker.tp_rank = 2, 1
    routed = []
    h.wrapper.send_notify = lambda tid, host, port, **kw: routed.append((host, port))
    h.worker.start_load_kv(step.kv_connector_metadata)
    assert routed[9] == ("10.0.0.2", 7003)
    assert len(routed) == 16
    finished, _ = h.worker.get_finished()
    assert finished == {request.request_id}
    scheduler.update_from_output(
        step, create_model_runner_output([], finished_sending=finished)
    )
    assert request.request_id not in scheduler.requests
    assert pool.get_num_free_blocks() == free_before
    assert _acks(h) == []  # The one source ACK was consumed.
    assert h.worker.get_finished() == (set(), set())


def test_unallocated_abort_without_peer_can_finish_locally():
    scheduler = create_scheduler(create_vllm_config(role="kv_producer"))
    request = create_request(num_tokens=32, do_remote_decode=True)
    scheduler.add_request(request)
    scheduler.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)
    assert request.request_id not in scheduler.requests
    metadata = scheduler.schedule().kv_connector_metadata
    assert isinstance(metadata, MoRIIOConnectorMetadata)
    assert not metadata.reqs_to_abort
