# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for MoRIIO request mappings and aborted WRITE receive cleanup.

vLLM's input_processor mutates a request_id after the connector has stored it
(e.g. ``cmpl-abc-0`` -> ``cmpl-abc-0-956053a4``), so an exact-match unmap can
miss. These tests cover the stable-``transfer_id`` fallback that handles that
case (and the clean exact-match and miss paths).

The direct map/unmap cases use a lightweight stand-in. WRITE abort cases use
the CPU scheduler and allocator with fake completion notifications.
"""

from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.distributed.kv_transfer.kv_connector.v1.moriio import (
    moriio_connector as mc,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio import (
    moriio_engine,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common import (
    ROLE,
    MoRIIOConnectorMetadata,
    MoRIIOMode,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector import (
    MoRIIOConnectorScheduler,
)
from vllm.v1.request import RequestStatus

from .test_moriio_connector import _setup_kv_transfer_request
from .test_moriio_connector import create_vllm_config as create_moriio_config
from .test_moriio_write_mode_stranding import _abort_setup
from .utils import create_model_runner_output, create_request, create_scheduler

_map = MoRIIOConnectorScheduler.map_request_id
_unmap = MoRIIOConnectorScheduler.unmap_request_id


def _sched():
    return SimpleNamespace(
        transfer_id_to_request_id={},
        request_id_to_transfer_id={},
    )


def test_exact_match_unmap_clears_both_tables():
    s = _sched()
    _map(s, "rid-1", "tid-1")
    _unmap(s, "rid-1")
    assert s.request_id_to_transfer_id == {}
    assert s.transfer_id_to_request_id == {}


def test_unmap_via_transfer_id_after_request_id_mutation():
    s = _sched()
    # Connector stores the ORIGINAL request_id at map time.
    _map(s, "cmpl-abc-0", "tid-42")
    # input_processor later mutates the rid; exact match would miss, so the
    # stable transfer_id is used to find and clear the original entry.
    _unmap(s, "cmpl-abc-0-956053a4", transfer_id="tid-42")
    assert s.request_id_to_transfer_id == {}
    assert s.transfer_id_to_request_id == {}


def test_unmap_miss_is_noop_and_does_not_raise():
    s = _sched()
    _map(s, "rid-1", "tid-1")
    # Unknown rid and unknown/None transfer_id: must not raise, must not touch
    # the existing (unrelated) mapping.
    _unmap(s, "rid-unknown", transfer_id=None)
    _unmap(s, "rid-unknown", transfer_id="tid-unknown")
    assert s.request_id_to_transfer_id == {"rid-1": "tid-1"}
    assert s.transfer_id_to_request_id == {"tid-1": "rid-1"}


def test_exact_match_preferred_over_transfer_id_fallback():
    s = _sched()
    _map(s, "rid-1", "tid-1")
    # Exact rid is present, so it is removed directly even if a transfer_id is
    # also supplied; no stale entries left behind.
    _unmap(s, "rid-1", transfer_id="tid-1")
    assert s.request_id_to_transfer_id == {}
    assert s.transfer_id_to_request_id == {}


class _Wrapper:
    def __init__(self):
        self.write_done: set[str] = set()
        self.write_failed: set[str] = set()

    def pop_finished_write_req_ids(self):
        done, self.write_done = self.write_done, set()
        return done

    def pop_failed_write_req_ids(self):
        failed, self.write_failed = self.write_failed, set()
        return failed

    def shutdown(self):
        pass


def _worker(wrapper):
    worker = mc.MoRIIOConnectorWorker.__new__(mc.MoRIIOConnectorWorker)
    worker.is_producer = False
    worker.mode = MoRIIOMode.WRITE
    worker.moriio_wrapper = wrapper
    worker.transfer_id_to_request_id = {}
    worker._unmatched_write_completions = set()
    worker._unmatched_write_failures = OrderedDict()
    worker._reported_write_transfers = set()
    worker._completed_write_transfers = OrderedDict()
    return worker


def _step(scheduler, worker):
    out = scheduler.schedule()
    metadata = out.kv_connector_metadata
    worker.transfer_id_to_request_id = metadata.transfer_id_to_request_id.copy()
    transfer_results = worker.get_transfer_results()
    scheduled = [scheduler.requests[r] for r in out.num_scheduled_tokens]
    result = create_model_runner_output(
        reqs=scheduled,
        finished_recving=transfer_results.finished_recving or None,
    )
    if result.kv_connector_output is not None:
        result.kv_connector_output.failed_recving = transfer_results.failed_recving
    scheduler.update_from_output(out, result)


@pytest.mark.cpu_test
@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize("failure_before_metadata", [False, True])
def test_late_allocation_failure_after_decode_free_does_not_become_unmatched(
    monkeypatch, failure_before_metadata
):
    scheduler = create_scheduler(create_moriio_config(role="kv_consumer"))
    connector = scheduler.connector.connector_scheduler
    allocations = []
    monkeypatch.setattr(
        connector, "send_notify_block", lambda **kw: allocations.append(kw)
    )
    pool = scheduler.kv_cache_manager.block_pool
    free_before = pool.get_num_free_blocks()
    free = MagicMock(wraps=scheduler.kv_cache_manager.free)
    monkeypatch.setattr(scheduler.kv_cache_manager, "free", free)
    producer, writer, meta, task, writes, _ = _abort_setup()
    wrapper = _Wrapper()
    worker = _worker(wrapper)
    notifications = []

    def deliver_failure(tid, host, port, message_type):
        assert message_type == "write_failed"
        notifications.append(tid)
        wrapper.write_failed.add(tid)

    producer.moriio_wrapper.send_notify = deliver_failure
    request = create_request(
        num_tokens=4 * scheduler.block_size, do_remote_prefill=True
    )
    _setup_kv_transfer_request(request, fake_transfer_id=meta.transfer_id)
    scheduler.add_request(request)
    if not failure_before_metadata:
        _step(scheduler, worker)
        assert pool.get_num_free_blocks() < free_before

    assert writer.abort_before_write(task.request_id, meta)
    if failure_before_metadata:
        early = worker.get_transfer_results()
        assert not early.finished_recving
        assert not early.failed_recving
        assert list(worker._unmatched_write_failures) == [meta.transfer_id]

    _step(scheduler, worker)
    assert request.request_id not in scheduler.requests
    assert pool.get_num_free_blocks() == free_before
    assert not connector.transfer_id_to_request_id
    assert free.call_count == 1

    # The real metadata path prunes the active-report set after unmapping.
    worker.start_load_kv(MoRIIOConnectorMetadata())
    assert not worker._reported_write_transfers
    assert not worker.transfer_id_to_request_id
    assert len(allocations) == 1
    allocation = allocations[0]
    assert allocation["block_notify_list"]
    monkeypatch.setattr(moriio_engine, "get_role", lambda: ROLE.PRODUCER)
    producer.moriio_wrapper._handle_remote_blocks_message(allocation)
    assert notifications == [meta.transfer_id, meta.transfer_id]
    assert writes == []
    assert len(producer.moriio_wrapper.done_req_ids) == 1

    late = worker.get_transfer_results()
    assert not late.finished_recving
    assert not late.failed_recving
    assert not worker._unmatched_write_failures
    _step(scheduler, worker)
    assert free.call_count == 1
    assert pool.get_num_free_blocks() == free_before


@pytest.mark.cpu_test
@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize("completion_before_abort", [False, True])
def test_write_consumer_abort_while_waiting(monkeypatch, completion_before_abort):
    scheduler = create_scheduler(create_moriio_config(role="kv_consumer"))
    connector = scheduler.connector.connector_scheduler
    monkeypatch.setattr(connector, "send_notify_block", lambda **kw: None)
    pool = scheduler.kv_cache_manager.block_pool
    free_before = pool.get_num_free_blocks()
    transfer_id = "xfer-abort"

    num_tokens = 4 * scheduler.block_size
    request = create_request(num_tokens=num_tokens, do_remote_prefill=True)
    _setup_kv_transfer_request(request, fake_transfer_id=transfer_id)
    scheduler.add_request(request)
    wrapper = _Wrapper()
    worker = _worker(wrapper)

    _step(scheduler, worker)
    request_id = request.request_id
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    held_free = pool.get_num_free_blocks()
    assert held_free < free_before

    if not completion_before_abort:
        scheduler.finish_requests(request_id, RequestStatus.FINISHED_ABORTED)
        _step(scheduler, worker)
        assert request_id in scheduler.requests
        assert pool.get_num_free_blocks() == held_free
        assert connector.transfer_id_to_request_id == {transfer_id: request_id}
        assert connector.request_id_to_transfer_id == {request_id: transfer_id}

    wrapper.write_done.add(transfer_id)
    _step(scheduler, worker)
    if completion_before_abort:
        scheduler.finish_requests(request_id, RequestStatus.FINISHED_ABORTED)

    assert request_id not in scheduler.requests
    assert not connector.transfer_id_to_request_id
    assert not connector.request_id_to_transfer_id
    assert not connector._write_recvs_in_flight
    assert not connector._finished_write_recvs_in_flight
    assert pool.get_num_free_blocks() == free_before


@pytest.mark.cpu_test
@pytest.mark.skip_global_cleanup
def test_write_consumer_abort_then_write_failed_cleans_up_once(monkeypatch):
    scheduler = create_scheduler(create_moriio_config(role="kv_consumer"))
    connector = scheduler.connector.connector_scheduler
    monkeypatch.setattr(connector, "send_notify_block", lambda **kw: None)
    pool = scheduler.kv_cache_manager.block_pool
    free_before = pool.get_num_free_blocks()
    transfer_id = "xfer-failed"

    request = create_request(
        num_tokens=4 * scheduler.block_size, do_remote_prefill=True
    )
    _setup_kv_transfer_request(request, fake_transfer_id=transfer_id)
    scheduler.add_request(request)
    wrapper = _Wrapper()
    worker = _worker(wrapper)

    _step(scheduler, worker)
    request_id = request.request_id
    held_free = pool.get_num_free_blocks()
    scheduler.finish_requests(request_id, RequestStatus.FINISHED_ABORTED)
    _step(scheduler, worker)
    assert pool.get_num_free_blocks() == held_free

    wrapper.write_failed.add(transfer_id)
    _step(scheduler, worker)

    assert request_id not in scheduler.requests
    assert pool.get_num_free_blocks() == free_before
    assert not connector.transfer_id_to_request_id
    assert not connector.request_id_to_transfer_id

    wrapper.write_failed.add(transfer_id)
    _step(scheduler, worker)
    assert pool.get_num_free_blocks() == free_before


@pytest.mark.cpu_test
@pytest.mark.skip_global_cleanup
def test_write_producer_abort_before_final_chunk_waits_for_worker_ack():
    chunk_size = 64
    scheduler = create_scheduler(
        create_moriio_config(role="kv_producer", max_num_batched_tokens=chunk_size)
    )
    connector = scheduler.connector.connector_scheduler
    pool = scheduler.kv_cache_manager.block_pool
    free_before = pool.get_num_free_blocks()
    transfer_id = "xfer-producer-abort"
    request = create_request(
        num_tokens=2 * chunk_size + chunk_size // 2,
        do_remote_decode=True,
        do_remote_prefill=False,
    )
    _setup_kv_transfer_request(request, fake_transfer_id=transfer_id)
    scheduler.add_request(request)

    first = scheduler.schedule()
    assert not first.kv_connector_metadata.reqs_to_save
    held_free = pool.get_num_free_blocks()
    assert held_free < free_before

    request_id = request.request_id
    scheduler.finish_requests(request_id, RequestStatus.FINISHED_ABORTED)
    assert request_id in scheduler.requests
    assert pool.get_num_free_blocks() == held_free
    assert connector.transfer_id_to_request_id == {transfer_id: request_id}

    abort_step = scheduler.schedule()
    metadata = abort_step.kv_connector_metadata
    assert set(metadata.reqs_to_abort) == {request_id}
    assert request_id not in connector._reqs_need_pending_save

    next_step = scheduler.schedule()
    assert not next_step.kv_connector_metadata.reqs_to_abort

    scheduler.update_from_output(
        next_step,
        create_model_runner_output([], finished_sending={request_id}),
    )
    assert request_id not in scheduler.requests
    assert pool.get_num_free_blocks() == free_before


@pytest.mark.cpu_test
@pytest.mark.skip_global_cleanup
def test_write_failure_waits_for_all_workers_before_releasing_decode_blocks(
    monkeypatch,
):
    scheduler = create_scheduler(create_moriio_config(role="kv_consumer"))
    connector = scheduler.connector.connector_scheduler
    monkeypatch.setattr(connector, "send_notify_block", lambda **kw: None)
    pool = scheduler.kv_cache_manager.block_pool
    free_before = pool.get_num_free_blocks()
    transfer_id = "xfer-multirank-failed"
    request = create_request(
        num_tokens=4 * scheduler.block_size, do_remote_prefill=True
    )
    _setup_kv_transfer_request(request, fake_transfer_id=transfer_id)
    scheduler.add_request(request)
    wrapper = _Wrapper()
    worker = _worker(wrapper)

    _step(scheduler, worker)
    request_id = request.request_id
    held_free = pool.get_num_free_blocks()
    scheduler.finish_requests(request_id, RequestStatus.FINISHED_ABORTED)
    _step(scheduler, worker)

    rank0_wrapper = _Wrapper()
    rank1_wrapper = _Wrapper()
    rank0 = _worker(rank0_wrapper)
    rank1 = _worker(rank1_wrapper)
    for rank in (rank0, rank1):
        rank.transfer_id_to_request_id = {transfer_id: request_id}

    def transfer_output(rank):
        results = rank.get_transfer_results()
        output = create_model_runner_output(
            [], finished_recving=results.finished_recving or None
        )
        if output.kv_connector_output is not None:
            output.kv_connector_output.failed_recving = results.failed_recving
        return output

    aggregator = KVOutputAggregator(expected_finished_count=2)
    rank0_wrapper.write_failed.add(transfer_id)
    aggregated = aggregator.aggregate([transfer_output(rank0), transfer_output(rank1)])
    assert aggregated is not None
    assert aggregated.kv_connector_output is not None
    assert not aggregated.kv_connector_output.finished_recving

    wait_step = scheduler.schedule()
    scheduler.update_from_output(wait_step, aggregated)
    assert request_id in scheduler.requests
    assert pool.get_num_free_blocks() == held_free

    # A late remote_blocks notification makes the producer resend write_failed.
    # The same decode rank must not count twice across separate engine ticks.
    rank0_wrapper.write_failed.add(transfer_id)
    duplicate = aggregator.aggregate([transfer_output(rank0), transfer_output(rank1)])
    assert duplicate is not None
    assert duplicate.kv_connector_output is not None
    assert not duplicate.kv_connector_output.finished_recving
    assert request_id in scheduler.requests
    assert pool.get_num_free_blocks() == held_free

    rank1_wrapper.write_failed.add(transfer_id)
    aggregated = aggregator.aggregate([transfer_output(rank0), transfer_output(rank1)])
    assert aggregated is not None
    assert aggregated.kv_connector_output is not None
    assert aggregated.kv_connector_output.finished_recving == {request_id}
    assert aggregated.kv_connector_output.failed_recving == {request_id}

    finish_step = scheduler.schedule()
    scheduler.update_from_output(finish_step, aggregated)
    assert request_id not in scheduler.requests
    assert pool.get_num_free_blocks() == free_before


@pytest.mark.cpu_test
@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize("already_in_flight", [False, True])
def test_write_allocation_without_remote_prefill_preserves_receive_tracking(
    already_in_flight,
):
    scheduler = create_scheduler(create_moriio_config(role="kv_consumer"))
    connector = scheduler.connector.connector_scheduler
    request = create_request(num_tokens=16, do_remote_prefill=True)
    _setup_kv_transfer_request(request, fake_transfer_id="xfer-local")
    request.kv_transfer_params["do_remote_prefill"] = False
    pending = {request.request_id} if already_in_flight else set()
    connector._write_recvs_in_flight.update(pending)
    connector.send_notify_block = MagicMock()

    connector.update_state_after_alloc(request, MagicMock(), num_external_tokens=16)

    assert connector._write_recvs_in_flight == pending
    assert connector.transfer_id_to_request_id == {"xfer-local": request.request_id}
    connector.send_notify_block.assert_not_called()
