# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU state-machine tests; barriers, not sleeps, control native progress.

The service accepts owned jobs and emits terminal/retirement records. These
tests guard against blocking model publication, early completion and freeing
uncertain DMA resources. Loading this dependency-free module directly keeps the
cheapest useful test independent of CUDA, torch, and the NIXL wheel.
"""

import gc
import importlib.util
import sys
import threading
import time
import weakref
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

_SOURCE = (
    Path(__file__).resolve().parents[4]
    / "vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_receiver.py"
)
_SPEC = importlib.util.spec_from_file_location("_tested_nixl_pull_receiver", _SOURCE)
assert _SPEC is not None and _SPEC.loader is not None
receiver_module = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = receiver_module
_SPEC.loader.exec_module(receiver_module)
NixlPullReceiver = receiver_module.NixlPullReceiver
ReadJob = receiver_module.ReadJob
ReceiveTerminal = receiver_module.ReceiveTerminal
ReceiveRetired = receiver_module.ReceiveRetired
NotifyOnlyTerminal = receiver_module.NotifyOnlyTerminal
BackendEvent = receiver_module.BackendEvent
ReceiverFailed = receiver_module.ReceiverFailed


@dataclass
class Handle:
    key: tuple[str, int]
    index: int


class FakeBackend:
    def __init__(self, block=None, failure=None):
        self.block = block
        self.failure = failure
        self.entered = threading.Event()
        self.resume = threading.Event()
        self.complete = threading.Event()
        self.complete.set()
        self.handshake = threading.Event()
        self.handshake.set()
        self.calls = []
        self.controls = []
        self.events: deque[Any] = deque()
        self.handle_refs = []
        self.closed = threading.Event()
        self.owner = None
        self._blocked = False

    def _call(self, method, detail=None):
        assert threading.get_ident() == self.owner
        self.calls.append((method, detail))
        if self.block == method and not self._blocked:
            self._blocked = True
            self.entered.set()
            assert self.resume.wait(5), f"Test failed to release {method} barrier"
        if self.failure == method:
            raise RuntimeError(f"Injected {method} failure")

    def initialize(self):
        self.owner = threading.get_ident()
        self._call("initialize")

    def ready(self, job):
        self._call("ready", job.key)
        return self.handshake.is_set()

    def transfers(self, job):
        for index in range(job.metadata.get("handles", 1)):
            self._call("prepare", (job.key, index))
            handle = Handle(job.key, index)
            self.handle_refs.append(weakref.ref(handle))
            yield handle

    def post(self, handle):
        self._call("post", (handle.key, handle.index))
        return "ERR" if self.failure == "post_status" else "PROC"

    def poll(self, handle):
        self._call("poll", (handle.key, handle.index))
        if self.failure == "poll_status":
            return "ERR"
        return "DONE" if self.complete.is_set() else "PROC"

    def release(self, handle):
        self._call("release", (handle.key, handle.index))

    def notify_without_read(self, job):
        self._call("notify", job.key)

    def control(self, message):
        self._call("control", message)
        self.controls.append(message)
        if isinstance(message, threading.Event):
            message.set()

    def tick(self, active_jobs):
        self._call("tick", tuple(job.key for job in active_jobs))
        while self.events:
            yield self.events.popleft()

    def retire(self, job):
        self._call("retire", job.key)

    def shutdown(self):
        self._call("shutdown")
        self.closed.set()


def job(name="request", generation=0, notify_only=False, handles=1):
    return ReadJob((name, generation), {"handles": handles}, notify_only)


def receive_results(receiver, count=1):
    deadline = time.monotonic() + 5
    results: list[Any] = []
    while len(results) < count:
        results.extend(receiver.drain_results())
        if len(results) >= count:
            return results
        remaining = deadline - time.monotonic()
        assert remaining > 0, "Receiver did not publish expected results"
        assert receiver.results_available.wait(remaining)
    return results


def retire_receive(receiver, request):
    terminal = receive_results(receiver)
    assert len(terminal) == 1
    assert isinstance(terminal[0], ReceiveTerminal)
    assert terminal[0].job == request
    receiver.finalized(request.key)
    retired = receive_results(receiver)
    assert retired == [ReceiveRetired(request, terminal[0].status)]
    return terminal[0]


@pytest.fixture
def service():
    instances = []

    def create(backend=None, start=True, **kwargs):
        backend = backend or FakeBackend()
        receiver = NixlPullReceiver(backend, **kwargs)
        instances.append(receiver)
        if start:
            receiver.start()
            assert receiver.initialized.wait(5)
        return receiver, backend

    yield create
    for receiver in instances:
        receiver.backend.resume.set()
        receiver.backend.handshake.set()
        receiver.backend.complete.set()
        receiver.shutdown()
        if receiver.failure is None:
            # Tests may leave a terminal waiting for model-side finalization.
            deadline = time.monotonic() + 5
            while not receiver.stopped.is_set() and time.monotonic() < deadline:
                for result in receiver.drain_results():
                    if isinstance(result, ReceiveTerminal):
                        receiver.finalized(result.job.key)
                receiver.results_available.wait(0.01)
        receiver.shutdown(wait=True, timeout=5)
        assert receiver.stopped.is_set()


@pytest.mark.parametrize("stage", ["prepare", "post", "poll", "release"])
def test_blocked_native_work_does_not_block_publication_or_result_drain(service, stage):
    receiver, backend = service(FakeBackend(block=stage))
    first, second = job("first"), job("second")
    assert receiver.submit(first)
    assert backend.entered.wait(5)

    # Running these on this thread is the assertion: the backend cannot regain
    # progress until they all return and the explicit barrier is released.
    assert receiver.submit(second)
    assert receiver.publish_control("producer lifecycle")
    receiver.cancel(second.key)
    assert receiver.drain_results() == []
    backend.resume.set()

    terminals = receive_results(receiver, 2)
    assert {result.job.key for result in terminals} == {first.key, second.key}
    for terminal in terminals:
        assert isinstance(terminal, ReceiveTerminal)
        receiver.finalized(terminal.job.key)
    retired = receive_results(receiver, 2)
    assert all(isinstance(result, ReceiveRetired) for result in retired)
    assert backend.owner != threading.get_ident()


def test_transport_completion_waits_for_every_source_and_finalization(service):
    receiver, backend = service(FakeBackend(block="post"))
    request = job(handles=3)
    receiver.submit(request)
    assert backend.entered.wait(5)
    assert receiver.drain_results() == []
    backend.resume.set()
    terminal = receive_results(receiver)
    assert terminal == [ReceiveTerminal(request, "done")]
    assert len([call for call in backend.calls if call[0] == "release"]) == 3
    assert not any(call[0] == "retire" for call in backend.calls)
    assert receiver.drain_results() == []
    receiver.finalized(request.key)
    assert receive_results(receiver) == [ReceiveRetired(request, "done")]
    assert not receiver.submit(request)
    receiver.finalized(request.key)
    assert receiver.drain_results() == []


def test_cancel_before_publication_preserves_notify_obligation(service):
    receiver, backend = service()
    request = job()
    receiver.cancel(request.key)
    receiver.submit(request)
    assert retire_receive(receiver, request).status == "cancelled"
    assert ("notify", request.key) in backend.calls
    assert not any(call[0] == "prepare" for call in backend.calls)


def test_cancel_during_handshake_waits_for_peer_before_notification(service):
    backend = FakeBackend()
    backend.handshake.clear()
    receiver, backend = service(backend)
    request = job()
    receiver.submit(request)
    receiver.cancel(request.key)
    fence = threading.Event()
    receiver.publish_control(fence)
    assert fence.wait(5)
    assert not any(call[0] == "notify" for call in backend.calls)
    backend.handshake.set()
    receiver.signal()
    assert retire_receive(receiver, request).status == "cancelled"


def test_cancel_after_posting_starts_drains_all_sources_before_terminal(service):
    receiver, backend = service(FakeBackend(block="post"))
    request = job(handles=3)
    receiver.submit(request)
    assert backend.entered.wait(5)
    receiver.cancel(request.key)
    backend.resume.set()
    assert retire_receive(receiver, request).status == "cancelled"
    assert len([call for call in backend.calls if call[0] == "post"]) == 3
    assert len([call for call in backend.calls if call[0] == "release"]) == 3
    assert not any(call[0] == "notify" for call in backend.calls)


def test_notify_only_never_emits_receive_completion_even_when_cancelled(service):
    receiver, backend = service()
    request = job(notify_only=True)
    receiver.cancel(request.key)
    receiver.submit(request)
    assert receive_results(receiver) == [NotifyOnlyTerminal(request)]
    assert not any(call[0] == "post" for call in backend.calls)
    assert receiver.drain_results() == []


def test_ready_controls_run_between_individual_source_posts(service):
    receiver, backend = service(FakeBackend(block="post"))
    request = job(handles=3)
    receiver.submit(request)
    assert backend.entered.wait(5)
    receiver.publish_control("urgent heartbeat")
    backend.resume.set()
    retire_receive(receiver, request)
    positions = [i for i, call in enumerate(backend.calls) if call[0] == "post"]
    control = backend.calls.index(("control", "urgent heartbeat"))
    assert positions[0] < control < positions[1]


@pytest.mark.parametrize(
    "failure", ["prepare", "post", "post_status", "poll", "poll_status", "release"]
)
def test_uncertain_native_failure_is_sticky_and_never_cleans_up(service, failure):
    receiver, backend = service(FakeBackend(failure=failure))
    receiver.submit(job(handles=2))
    assert receiver.stopped.wait(5)
    with pytest.raises(ReceiverFailed):
        receiver.drain_results()
    assert not receiver.shutdown(wait=True, timeout=0)
    assert not backend.closed.is_set()
    assert not any(call[0] == "retire" for call in backend.calls)
    if failure != "prepare":
        gc.collect()
        assert backend.handle_refs[0]() is not None


def test_partial_failure_keeps_previously_posted_sibling_pinned(service):
    backend = FakeBackend(block="post")
    backend.complete.clear()
    receiver, backend = service(backend)
    receiver.submit(job(handles=2))
    assert backend.entered.wait(5)
    backend.failure = "prepare"
    backend.resume.set()
    assert receiver.stopped.wait(5)
    assert backend.handle_refs[0]() is not None
    assert not any(
        call[0] in ("release", "retire", "shutdown") for call in backend.calls
    )


@pytest.mark.parametrize("notify_only", [False, True])
def test_admission_class_capacity_exhaustion_is_visible_and_nonblocking(
    service, notify_only
):
    receiver, backend = service(
        FakeBackend(block="prepare"), max_receives=1, max_notify_only=1
    )
    receiver.submit(job("blocking"))
    assert backend.entered.wait(5)
    if notify_only:
        assert receiver.submit(job("first notification", notify_only=True))
    assert not receiver.submit(job("overflow", notify_only=notify_only))
    with pytest.raises(ReceiverFailed):
        receiver.check_health()
    backend.resume.set()
    assert receiver.stopped.wait(5)
    assert not backend.closed.is_set()


def test_control_capacity_has_independent_fatal_latch(service):
    receiver, backend = service(FakeBackend(block="post"), max_controls=1)
    receiver.submit(job())
    assert backend.entered.wait(5)
    assert receiver.publish_control("first")
    assert not receiver.publish_control("overflow")
    with pytest.raises(ReceiverFailed):
        receiver.check_health()
    backend.resume.set()


def test_cancel_and_finalize_do_not_require_control_mailbox_space(service):
    receiver, backend = service(FakeBackend(block="post"), max_controls=1)
    request = job()
    receiver.submit(request)
    assert backend.entered.wait(5)
    assert receiver.publish_control("filled")
    receiver.cancel(request.key)
    backend.resume.set()
    assert retire_receive(receiver, request).status == "cancelled"


def test_result_capacity_is_reserved_through_retirement_consumption(service):
    receiver, backend = service(max_receives=1, max_controls=1)
    request = job()
    receiver.submit(request)
    terminal = receive_results(receiver)
    assert isinstance(terminal[0], ReceiveTerminal)
    receiver.finalized(request.key)
    assert receiver.results_available.wait(5)
    # The owner has retired the resources, but the model still owes consumption
    # of the acknowledgment before it may reuse the worker admission slot.
    assert not receiver.submit(job("too early"))
    with pytest.raises(ReceiverFailed):
        receiver.check_health()


def test_full_backend_result_pool_cannot_block_native_owner(service):
    backend = FakeBackend(block="post")
    receiver, backend = service(backend, max_controls=1)
    receiver.submit(job())
    assert backend.entered.wait(5)
    backend.events.extend(["first", "overflow"])
    backend.resume.set()
    assert receiver.stopped.wait(5)
    with pytest.raises(ReceiverFailed):
        receiver.check_health()
    assert not backend.closed.is_set()


def test_backend_event_capacity_is_reusable_after_consumption(service):
    receiver, backend = service(max_controls=1)
    for payload in ("first", "second"):
        backend.events.append(payload)
        receiver.signal()
        assert receive_results(receiver) == [BackendEvent(payload)]


def test_snapshot_overwrite_preserves_explicit_empty_and_rejects_stale(service):
    receiver, backend = service(FakeBackend(block="post"))
    request = job()
    receiver.submit(request)
    assert backend.entered.wait(5)
    receiver.publish_snapshot(1, ("engine", "request"))
    receiver.publish_snapshot(2, ())
    receiver.publish_snapshot(1, ("stale",))
    backend.resume.set()
    retire_receive(receiver, request)
    assert backend.controls == [()]


def test_idle_wakeup_keeps_queued_work_visible(service):
    # Long idle intervals make loss of an event fail the short barrier timeout.
    receiver, backend = service(idle_interval=60)
    for generation in range(5):
        request = job(generation=generation)
        assert receiver.submit(request)
        retire_receive(receiver, request)


def test_shutdown_while_posting_waits_for_quiescence_and_finalization(service):
    receiver, backend = service(FakeBackend(block="post"))
    request = job()
    receiver.submit(request)
    assert backend.entered.wait(5)
    assert not receiver.shutdown(wait=True, timeout=0)
    assert not backend.closed.is_set()
    backend.resume.set()
    assert retire_receive(receiver, request).status == "cancelled"
    assert receiver.shutdown(wait=True, timeout=5)
    assert backend.closed.is_set()


def test_shutdown_keeps_handshake_callback_lane_open(service):
    backend = FakeBackend()
    backend.handshake.clear()
    receiver, backend = service(backend)
    request = job()
    receiver.submit(request)
    receiver.shutdown()
    callback = threading.Event()
    assert receiver.publish_control(callback)
    assert callback.wait(5)
    backend.handshake.set()
    receiver.signal()
    assert retire_receive(receiver, request).status == "cancelled"
    assert receiver.shutdown(wait=True, timeout=5)


def test_lifetime_history_cannot_evict_duplicate_protection(service):
    receiver, backend = service(max_history=1)
    request = job()
    receiver.submit(request)
    retire_receive(receiver, request)
    assert not receiver.submit(request)
    assert not receiver.submit(job("new"))
    with pytest.raises(ReceiverFailed):
        receiver.check_health()


def test_finalization_before_terminal_consumption_is_fatal(service):
    receiver, backend = service(FakeBackend(block="post"))
    request = job()
    receiver.submit(request)
    assert backend.entered.wait(5)
    receiver.finalized(request.key)
    with pytest.raises(ReceiverFailed):
        receiver.check_health()
    backend.resume.set()


def test_publication_during_clear_is_not_lost_before_owner_sleeps(service):
    class PausedClear:
        def __init__(self):
            self.event = threading.Event()
            self.entered = threading.Event()
            self.resume = threading.Event()
            self.first = True

        def clear(self):
            if self.first:
                self.first = False
                self.entered.set()
                assert self.resume.wait(5)
            self.event.clear()

        def set(self):
            self.event.set()

        def wait(self, timeout):
            return self.event.wait(timeout)

    receiver, backend = service(start=False, idle_interval=60)
    wake = PausedClear()
    receiver._wake = wake
    receiver.start()
    assert wake.entered.wait(5)
    request = job()
    receiver.submit(request)
    # The publisher's set is deliberately erased by the owner's clear. Work
    # remains safe because mailbox inspection follows that clear.
    wake.resume.set()
    retire_receive(receiver, request)


def test_fatal_handler_runs_once_and_quarantine_survives_worker_unwind(service):
    observed: list[BaseException] = []
    receiver, backend = service(
        FakeBackend(failure="post"), fatal_handler=observed.append
    )
    receiver.submit(job())
    assert receiver.stopped.wait(5)
    assert observed == [receiver.failure]
    reference = weakref.ref(receiver)
    assert any(item is receiver for item in receiver_module._FAILED_RECEIVERS)
    receiver._fail(RuntimeError("A second error cannot replace the first"))
    assert observed == [receiver.failure]
    del receiver
    gc.collect()
    assert reference() is not None
    assert backend.handle_refs[0]() is not None


def test_poll_rotation_progresses_all_waiting_receives_without_model_calls(service):
    class ObservedPoll(FakeBackend):
        def __init__(self):
            super().__init__()
            self.complete.clear()
            self.polled = {"first": threading.Event(), "second": threading.Event()}

        def poll(self, handle):
            status = super().poll(handle)
            self.polled[handle.key[0]].set()
            return status

    receiver, backend = service(ObservedPoll(), poll_budget=1)
    first, second = job("first"), job("second")
    receiver.submit(first)
    receiver.submit(second)
    assert backend.polled["first"].wait(5)
    assert backend.polled["second"].wait(5)
    assert receiver.drain_results() == []
    backend.complete.set()
    receiver.signal()
    terminals = receive_results(receiver, 2)
    for terminal in terminals:
        receiver.finalized(terminal.job.key)
    assert len(receive_results(receiver, 2)) == 2
