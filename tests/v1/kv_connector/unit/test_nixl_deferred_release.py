# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deferred block release on the D side with a host-pinned KV cache: blocks of a
finished request are reported free only after the guard event recorded when the
release reached the worker has completed. Requests still receiving and CPU staging
keep the upstream path."""

import copy

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker import (
    DeferredBlockRelease,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.connector import (
    NixlBaseConnector,
)
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, KVConnectorOutput
from vllm.v1.request import RequestStatus

from .utils import (
    assert_scheduler_empty,
    create_model_runner_output,
    create_request,
    create_scheduler,
    create_vllm_config,
)


class _FakeEvent:
    recorded: list["_FakeEvent"] = []

    def __init__(self):
        self.done = False

    def record(self):
        _FakeEvent.recorded.append(self)

    def query(self):
        return self.done


@pytest.fixture
def fake_events(monkeypatch):
    _FakeEvent.recorded = []
    monkeypatch.setattr("torch.cuda.Event", _FakeEvent)
    return _FakeEvent


def test_release_waits_for_guard_event(fake_events):
    dbr = DeferredBlockRelease()
    dbr.on_step(["a", "b"])  # the release reaches the worker: one guard, recorded now
    assert len(fake_events.recorded) == 1
    assert dbr.ready() == set()  # guard not complete yet
    dbr.on_step(["c"])  # a later release gets its own guard
    assert len(fake_events.recorded) == 2
    fake_events.recorded[0].done = True
    assert dbr.ready() == {"a", "b"}
    assert dbr.ready() == set()  # released once
    fake_events.recorded[1].done = True
    assert dbr.ready() == {"c"}


def test_no_event_recorded_without_pending(fake_events):
    dbr = DeferredBlockRelease()
    for _ in range(3):
        dbr.on_step([])
    assert fake_events.recorded == [] and dbr.ready() == set()


def _host_pinned_d_scheduler(kv_buffer_device: str = "cuda", kv_role="kv_consumer"):
    """A D-side scheduler with the NIXL connector and a host-pinned KV cache."""
    vllm_config = create_vllm_config(kv_role=kv_role)
    vllm_config.cache_config.kv_cache_host_pinned = True
    assert vllm_config.kv_transfer_config is not None
    vllm_config.kv_transfer_config.kv_buffer_device = kv_buffer_device
    scheduler = create_scheduler(vllm_config)
    connector = scheduler.connector
    assert isinstance(connector, NixlBaseConnector)
    assert connector.connector_scheduler is not None
    return scheduler, connector.connector_scheduler


def _with_kv_output(**kwargs):
    output = copy.deepcopy(EMPTY_MODEL_RUNNER_OUTPUT)
    output.kv_connector_output = KVConnectorOutput(**kwargs)
    return output


def test_abort_during_recv_is_freed_once_on_recv():
    """An abort while the pull is in flight must not be deferred: its blocks stay
    held until finished_recving, which then frees them exactly once."""
    scheduler, connector = _host_pinned_d_scheduler()
    request = create_request(request_id=1, num_tokens=40, do_remote_prefill=True)
    scheduler.add_request(request)
    scheduler.update_from_output(scheduler.schedule(), EMPTY_MODEL_RUNNER_OUTPUT)
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS

    scheduler.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)
    assert connector._reqs_to_release == set()
    assert request.request_id in scheduler.requests  # the NIC may still write
    assert connector._reqs_receiving == set()  # cleared when the request ended

    output = _with_kv_output(finished_recving={request.request_id})
    scheduler.update_from_output(scheduler.schedule(), output)
    assert request.request_id not in scheduler.requests
    assert_scheduler_empty(scheduler)


def test_finished_request_is_released_by_the_worker():
    """A request that finished normally is held until the worker reports its
    deferred release (finished_sending), then freed."""
    scheduler, connector = _host_pinned_d_scheduler()
    request = create_request(
        request_id=2, num_tokens=40, max_tokens=1, do_remote_prefill=True
    )
    scheduler.add_request(request)
    scheduler.update_from_output(scheduler.schedule(), EMPTY_MODEL_RUNNER_OUTPUT)
    output = _with_kv_output(finished_recving={request.request_id})
    scheduler.update_from_output(scheduler.schedule(), output)

    scheduler_output = scheduler.schedule()  # the decode step that finishes it
    scheduler.update_from_output(
        scheduler_output, create_model_runner_output([request], use_eos=True)
    )
    assert request.status == RequestStatus.FINISHED_STOPPED
    assert connector._reqs_to_release == {request.request_id}
    assert request.request_id in scheduler.requests

    scheduler_output = scheduler.schedule()
    assert connector._reqs_to_release == set()  # handed to the worker
    output = _with_kv_output(finished_sending={request.request_id})
    scheduler.update_from_output(scheduler_output, output)
    assert request.request_id not in scheduler.requests
    assert_scheduler_empty(scheduler)


def test_cpu_kv_buffer_is_not_deferred():
    """With a CPU staging buffer the worker has no deferred release, so the
    scheduler must not wait for one (else the blocks would never be freed)."""
    scheduler, connector = _host_pinned_d_scheduler(kv_buffer_device="cpu")
    request = create_request(
        request_id=3, num_tokens=40, max_tokens=1, do_remote_prefill=True
    )
    scheduler.add_request(request)
    scheduler.update_from_output(scheduler.schedule(), EMPTY_MODEL_RUNNER_OUTPUT)
    output = _with_kv_output(finished_recving={request.request_id})
    scheduler.update_from_output(scheduler.schedule(), output)
    scheduler_output = scheduler.schedule()
    scheduler.update_from_output(
        scheduler_output, create_model_runner_output([request], use_eos=True)
    )
    assert connector._reqs_to_release == set()
    assert request.request_id not in scheduler.requests


def _received(scheduler, request):
    """Drive a remote-prefill request until its pull has completed."""
    scheduler.add_request(request)
    scheduler.update_from_output(scheduler.schedule(), EMPTY_MODEL_RUNNER_OUTPUT)
    output = _with_kv_output(finished_recving={request.request_id})
    scheduler.update_from_output(scheduler.schedule(), output)


def _release_by_worker(scheduler, connector, request):
    """The worker reports the deferred release; the blocks are then freed once."""
    assert connector._reqs_to_release == {request.request_id}
    assert request.request_id in scheduler.requests
    scheduler_output = scheduler.schedule()
    assert connector._reqs_to_release == set()
    output = _with_kv_output(finished_sending={request.request_id})
    scheduler.update_from_output(scheduler_output, output)
    assert request.request_id not in scheduler.requests
    assert_scheduler_empty(scheduler)


def test_abort_during_decode_is_deferred():
    """An abort after the pull has landed can race the drafter like a normal
    finish, so it is deferred too."""
    scheduler, connector = _host_pinned_d_scheduler()
    request = create_request(request_id=4, num_tokens=40, do_remote_prefill=True)
    _received(scheduler, request)
    scheduler_output = scheduler.schedule()  # a decode step
    scheduler.update_from_output(
        scheduler_output, create_model_runner_output([request])
    )
    assert request.status == RequestStatus.RUNNING
    scheduler.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)
    _release_by_worker(scheduler, connector, request)


def test_abort_after_recv_before_scheduling_is_deferred():
    """Aborted with its pull complete but not yet scheduled: nothing more will
    arrive for it, so the deferred release is its only completion."""
    scheduler, connector = _host_pinned_d_scheduler()
    request = create_request(request_id=5, num_tokens=40, do_remote_prefill=True)
    scheduler.add_request(request)
    scheduler.update_from_output(scheduler.schedule(), EMPTY_MODEL_RUNNER_OUTPUT)
    output = _with_kv_output(finished_recving={request.request_id})
    scheduler.update_from_output(scheduler.schedule(), output)
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert connector._reqs_receiving == set()
    scheduler.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)
    _release_by_worker(scheduler, connector, request)


def _finish_locally(scheduler, request):
    """Run a request with no transfer params (served locally) to a normal finish."""
    scheduler.add_request(request)
    scheduler_output = scheduler.schedule()
    scheduler.update_from_output(
        scheduler_output, create_model_runner_output([request], use_eos=True)
    )
    assert request.status == RequestStatus.FINISHED_STOPPED


def test_local_request_on_decode_node_is_deferred():
    """A request without transfer params on a decode node frees blocks that a
    later pull can land in, so it waits for the guard like a P/D request."""
    scheduler, connector = _host_pinned_d_scheduler()
    request = create_request(request_id=7, num_tokens=40, max_tokens=1)
    assert request.kv_transfer_params is None
    _finish_locally(scheduler, request)
    _release_by_worker(scheduler, connector, request)


def test_local_request_on_producer_node_is_not_deferred():
    """No pull lands on a producer-only node: nothing to guard against."""
    scheduler, connector = _host_pinned_d_scheduler(kv_role="kv_producer")
    request = create_request(request_id=8, num_tokens=40, max_tokens=1)
    _finish_locally(scheduler, request)
    assert connector._reqs_to_release == set()
    assert request.request_id not in scheduler.requests


def test_bidirectional_kv_xfer_is_rejected():
    """D->P pulls land in a producer's blocks, which the guard does not cover."""
    vllm_config = create_vllm_config(
        kv_role="kv_both", kv_connector_extra_config={"bidirectional_kv_xfer": True}
    )
    vllm_config.cache_config.kv_cache_host_pinned = True
    with pytest.raises(ValueError, match="bidirectional_kv_xfer"):
        create_scheduler(vllm_config)
