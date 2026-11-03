# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deferred block release on the D side with a host-pinned KV cache: blocks of a
finished request are reported free only after the guard event recorded GUARD_AGE
steps later has completed; aborted requests and CPU staging keep the upstream path."""

import copy

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker import (
    DeferredBlockRelease,
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
    dbr.on_step(["a"])  # step 0: request a finished
    assert dbr.ready() == set()
    dbr.on_step([])  # step 1: too early for a guard
    assert dbr.ready() == set() and fake_events.recorded == []
    dbr.on_step(["b"])  # step 2: a gets its guard event, b arrives
    assert len(fake_events.recorded) == 1
    assert dbr.ready() == set()  # guard not complete yet
    fake_events.recorded[0].done = True
    assert dbr.ready() == {"a"}
    assert dbr.ready() == set()  # released once
    dbr.on_step([])
    dbr.on_step([])  # b's guard (one event shared by all requests guarded in a step)
    assert len(fake_events.recorded) == 2
    fake_events.recorded[1].done = True
    assert dbr.ready() == {"b"}


def test_no_event_recorded_without_pending(fake_events):
    dbr = DeferredBlockRelease()
    for _ in range(3):
        dbr.on_step([])
    assert fake_events.recorded == [] and dbr.ready() == set()


def _host_pinned_d_scheduler(kv_buffer_device: str = "cuda"):
    """A D-side scheduler with the NIXL connector and a host-pinned KV cache."""
    vllm_config = create_vllm_config(kv_role="kv_consumer")
    vllm_config.cache_config.kv_cache_host_pinned = True
    vllm_config.kv_transfer_config.kv_buffer_device = kv_buffer_device
    scheduler = create_scheduler(vllm_config)
    return scheduler, scheduler.connector.connector_scheduler


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
    assert connector._reqs_to_release == []
    assert request.request_id in scheduler.requests  # the NIC may still write

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
    assert connector._reqs_to_release == [request.request_id]
    assert request.request_id in scheduler.requests

    scheduler_output = scheduler.schedule()
    assert connector._reqs_to_release == []  # handed to the worker
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
    assert connector._reqs_to_release == []
    assert request.request_id not in scheduler.requests
