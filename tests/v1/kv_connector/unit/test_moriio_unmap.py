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

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.moriio import (
    moriio_connector as mc,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common import (
    MoRIIOMode,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector import (
    MoRIIOConnectorScheduler,
)
from vllm.v1.request import RequestStatus

from .test_moriio_connector import _setup_kv_transfer_request
from .test_moriio_connector import create_vllm_config as create_moriio_config
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

    def pop_finished_write_req_ids(self):
        done, self.write_done = self.write_done, set()
        return done

    def shutdown(self):
        pass


def _worker(wrapper):
    worker = mc.MoRIIOConnectorWorker.__new__(mc.MoRIIOConnectorWorker)
    worker.is_producer = False
    worker.mode = MoRIIOMode.WRITE
    worker.moriio_wrapper = wrapper
    worker.transfer_id_to_request_id = {}
    worker._unmatched_write_completions = set()
    return worker


def _step(scheduler, worker):
    out = scheduler.schedule()
    metadata = out.kv_connector_metadata
    worker.transfer_id_to_request_id = metadata.transfer_id_to_request_id.copy()
    _, done_recving = worker.get_finished()
    scheduled = [scheduler.requests[r] for r in out.num_scheduled_tokens]
    result = create_model_runner_output(
        reqs=scheduled, finished_recving=done_recving or None
    )
    scheduler.update_from_output(out, result)


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
