# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for OffloadingManager.lock() and the tiering control-plane thread.

The control plane of a tier that answers something the engine does not drive --
a remote peer, say -- used to advance only from on_schedule_end(), so a peer's
request waited for a model-step boundary. These tests cover the thread that
services such tiers between steps and the exclusion that makes it safe.
"""

import threading
import time
from collections.abc import Iterable
from typing import ClassVar
from unittest.mock import MagicMock

import pytest

from vllm.v1.kv_offload.base import (
    LookupResult,
    Medium,
    OffloadKey,
    ReqContext,
    RequestOffloadingContext,
    ScheduleEndContext,
)
from vllm.v1.kv_offload.tiering.base import (
    JobResult,
    ParentManager,
    SecondaryTierManager,
    TransferJob,
)
from vllm.v1.kv_offload.tiering.manager import (
    CPUPrimaryTierOffloadingManager,
    TieringOffloadingManager,
)

from .test_tiering_offloading import _mock_mmap_region

_CTX = ReqContext(req_id="test")
_EMPTY_SCHEDULE_END = ScheduleEndContext(new_req_ids=(), preempted_req_ids=())

# Long enough that a 1ms-interval thread gets many rounds, short enough to keep
# the suite quick.
_SETTLE_S = 0.5


class _FakeTier(SecondaryTierManager):
    """Tier that records which thread serviced it, and when."""

    medium: ClassVar[Medium] = Medium.CPU
    needs_control_plane_thread: ClassVar[bool] = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Counters are only ever incremented, and int writes are atomic under
        # the GIL, so the test thread may read them without the manager lock.
        self.serves = 0
        self.polls = 0
        self.serve_threads: set[int] = set()
        self.served = threading.Event()
        self.shutdown_saw_live_thread: bool | None = None
        self.on_serve = None

    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        return LookupResult.MISS

    def submit_store(self, job_metadata: TransferJob) -> None:
        pass

    def submit_load(self, job_metadata: TransferJob) -> None:
        pass

    def get_finished_jobs(self) -> Iterable[JobResult]:
        self.polls += 1
        return ()

    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        return RequestOffloadingContext()

    def drain_jobs(self) -> None:
        pass

    def serve_external_requests(self, parent: ParentManager) -> None:
        self.serves += 1
        self.serve_threads.add(threading.get_ident())
        if self.on_serve is not None:
            self.on_serve(parent)
        self.served.set()

    def shutdown(self) -> None:
        self.shutdown_saw_live_thread = any(
            t.name == "vllm_offload_control_plane" and t.is_alive()
            for t in threading.enumerate()
        )


class _QuietTier(_FakeTier):
    """Tier that does not want servicing between steps."""

    needs_control_plane_thread: ClassVar[bool] = False


def _make_manager(tier_cls=_FakeTier, num_chunks: int = 8, **kwargs):
    primary = CPUPrimaryTierOffloadingManager(
        num_chunks=num_chunks,
        mmap_region=_mock_mmap_region(num_chunks),
    )
    tier = tier_cls(
        offloading_spec=MagicMock(),
        primary_kv_view=primary.get_kv_memoryview(),
        tier_type="fake",
    )
    manager = TieringOffloadingManager(
        primary_tier=primary,
        secondary_tiers=[tier],
        **kwargs,
    )
    return manager, tier


@pytest.fixture
def manager_and_tier(request):
    kwargs = getattr(request, "param", {})
    manager, tier = _make_manager(**kwargs)
    try:
        yield manager, tier
    finally:
        manager.shutdown()


def _live_control_threads() -> list[threading.Thread]:
    return [
        t
        for t in threading.enumerate()
        if t.name == "vllm_offload_control_plane" and t.is_alive()
    ]


def test_control_thread_and_scheduler_never_overlap(manager_and_tier):
    """lock() must keep the two threads out of the manager at the same time.

    Both sides must also make progress: a thread that never runs would pass a
    mutual-exclusion check trivially, and so would one that starves the
    scheduler.
    """
    manager, tier = manager_and_tier
    state = {"inside": False, "overlaps": 0}

    def critical_section():
        if state["inside"]:
            state["overlaps"] += 1
        state["inside"] = True
        time.sleep(0.002)
        state["inside"] = False

    tier.on_serve = lambda parent: critical_section()

    steps = 0
    deadline = time.monotonic() + _SETTLE_S
    while time.monotonic() < deadline:
        manager.lock()
        try:
            critical_section()
        finally:
            manager.unlock()
        steps += 1

    assert state["overlaps"] == 0
    assert steps > 0, "scheduler thread was starved by the control thread"
    assert tier.serves > 0, "control thread never ran"


def test_serves_while_the_scheduler_is_outside_the_lock(manager_and_tier):
    """The point of the thread: serve during the model-execution window.

    The engine releases the manager at the end of a step and then blocks on the
    model future. Here the "scheduler" does the same -- one step, then a sleep
    with the lock free -- and the tier must be serviced during that sleep.
    """
    manager, tier = manager_and_tier

    manager.lock()
    try:
        manager.on_schedule_end(_EMPTY_SCHEDULE_END)
    finally:
        manager.unlock()

    tier.served.clear()
    serves_before = tier.serves

    # Stand in for future.result(): the engine thread is blocked, holding
    # nothing.
    assert tier.served.wait(_SETTLE_S), (
        "tier was not serviced while the scheduler held no lock"
    )
    assert tier.serves > serves_before
    assert tier.polls > 0

    scheduler_thread_id = threading.get_ident()
    assert scheduler_thread_id not in tier.serve_threads or len(tier.serve_threads) > 1


@pytest.mark.parametrize(
    "manager_and_tier", [{"control_plane_interval_s": 0.0}], indirect=True
)
def test_zero_interval_disables_the_thread(manager_and_tier):
    """The escape hatch back to per-step servicing.

    Without the thread the tier is serviced only by on_schedule_end, which is
    the behaviour the control plane had before it existed.
    """
    manager, tier = manager_and_tier

    assert not _live_control_threads()
    assert not tier.served.wait(0.05)
    assert tier.serves == 0

    manager.lock()
    try:
        manager.on_schedule_end(_EMPTY_SCHEDULE_END)
    finally:
        manager.unlock()

    assert tier.serves == 1


def test_no_thread_when_no_tier_asks_for_one():
    """A tier that does not opt in must not cost a thread."""
    manager, tier = _make_manager(tier_cls=_QuietTier)
    try:
        assert not _live_control_threads()
    finally:
        manager.shutdown()


def test_tier_that_did_not_opt_in_is_not_serviced_off_thread():
    """Only opted-in tiers are touched by the control thread.

    A tier that expects the scheduler thread must not be dragged onto another
    one just because it shares a manager with a tier that opted in.
    """
    num_chunks = 8
    primary = CPUPrimaryTierOffloadingManager(
        num_chunks=num_chunks,
        mmap_region=_mock_mmap_region(num_chunks),
    )
    common = {
        "offloading_spec": MagicMock(),
        "primary_kv_view": primary.get_kv_memoryview(),
    }
    opted_in = _FakeTier(tier_type="fake", **common)
    quiet = _QuietTier(tier_type="quiet", **common)
    manager = TieringOffloadingManager(
        primary_tier=primary,
        secondary_tiers=[opted_in, quiet],
    )
    try:
        assert opted_in.served.wait(_SETTLE_S)
        assert quiet.serves == 0
        assert quiet.polls == 0

        manager.lock()
        try:
            manager.on_schedule_end(_EMPTY_SCHEDULE_END)
        finally:
            manager.unlock()

        # on_schedule_end still serves every tier, so the thread dying degrades
        # to per-step servicing rather than to none.
        assert quiet.serves == 1
    finally:
        manager.shutdown()


def test_per_step_gate_is_not_touched_by_the_control_thread(manager_and_tier):
    """_processed_jobs_this_step stays owned by the scheduler thread.

    Sharing it would let a fast step skip a poll it would otherwise do, pushing
    a completion out by a whole step.
    """
    manager, tier = manager_and_tier

    assert tier.served.wait(_SETTLE_S)
    polls_after_rounds = tier.polls
    assert polls_after_rounds > 0
    assert manager._processed_jobs_this_step is False


def test_shutdown_joins_the_control_thread_before_tier_teardown(manager_and_tier):
    """The thread drives tier transports, so it must be stopped first.

    Closing a transport under a running round can take the process down rather
    than raise.
    """
    manager, tier = manager_and_tier
    assert tier.served.wait(_SETTLE_S)
    assert _live_control_threads()

    manager.shutdown()

    assert tier.shutdown_saw_live_thread is False
    assert not _live_control_threads()


def test_locked_helper_releases_on_error(manager_and_tier):
    """locked() must not leak the lock when its body raises."""
    manager, _ = manager_and_tier

    with pytest.raises(RuntimeError), manager.locked():
        raise RuntimeError("boom")

    assert manager.lock(timeout=0.5), "locked() leaked the lock"
    manager.unlock()


def test_lock_timeout_reports_failure(manager_and_tier):
    """A timed lock() returns False rather than blocking forever."""
    manager, _ = manager_and_tier

    manager.lock()
    try:
        acquired: list[bool] = []
        t = threading.Thread(target=lambda: acquired.append(manager.lock(timeout=0.01)))
        t.start()
        t.join(timeout=5.0)
        assert acquired == [False]
    finally:
        manager.unlock()
