# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for OffloadingManager.lock and the tiering tier-poll thread.

The control plane of a tier that answers something the engine does not drive --
a remote peer, say -- used to advance only from on_schedule_end(), so a peer's
request waited for a model-step boundary. These tests cover the thread that
services such tiers between steps and the exclusion that makes it safe.
"""

import threading
import time
from collections.abc import Iterable
from contextlib import nullcontext
from typing import ClassVar
from unittest.mock import MagicMock

import pytest

import vllm.v1.kv_offload.tiering.manager as tiering_manager
from vllm.v1.kv_offload.base import (
    LookupResult,
    Medium,
    OffloadingManager,
    OffloadKey,
    ReqContext,
    RequestOffloadingContext,
    ScheduleEndContext,
)
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
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
    tolerates_off_thread_polling: ClassVar[bool] = True

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
            t.name == "vllm_offload_tier_poll" and t.is_alive()
            for t in threading.enumerate()
        )


class _QuietTier(_FakeTier):
    """Tier that does not want servicing between steps."""

    tolerates_off_thread_polling: ClassVar[bool] = False


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
        if t.name == "vllm_offload_tier_poll" and t.is_alive()
    ]


def test_control_thread_and_scheduler_never_overlap(manager_and_tier):
    """The lock must keep the two threads out of the manager at the same time.

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
        with manager.lock:
            critical_section()
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

    with manager.lock:
        manager.on_schedule_end(_EMPTY_SCHEDULE_END)

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
    "manager_and_tier", [{"tier_poll_interval_s": 0.0}], indirect=True
)
def test_zero_interval_disables_the_thread(manager_and_tier):
    """The escape hatch back to per-step servicing.

    Without the thread the tier is serviced only by on_schedule_end, which is
    the behaviour these tiers had before the thread existed.
    """
    manager, tier = manager_and_tier

    assert not _live_control_threads()
    assert not tier.served.wait(0.05)
    assert tier.serves == 0

    with manager.lock:
        manager.on_schedule_end(_EMPTY_SCHEDULE_END)

    assert tier.serves == 1


def test_no_thread_when_no_tier_asks_for_one():
    """A tier that does not opt in must not cost a thread."""
    manager, tier = _make_manager(tier_cls=_QuietTier)
    try:
        assert not _live_control_threads()
    finally:
        manager.shutdown()


def test_tier_that_did_not_opt_in_is_not_serviced_off_thread():
    """Only opted-in tiers are polled and served by the control thread.

    A tier that expects the scheduler thread must not be dragged onto another
    one just because it shares a manager with a tier that opted in. Fan-out
    through ParentManager (lookup, on_new_request, on_request_finished) can
    still reach it from that thread; that is the documented exception.
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

        with manager.lock:
            manager.on_schedule_end(_EMPTY_SCHEDULE_END)

        # on_schedule_end serves every tier, so the unpolled tier is serviced
        # once per step.
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


class _LookingUpTier(_FakeTier):
    """Opted-in tier that looks keys up through its parent while serving.

    That is what the p2p server role does with a peer's lookup, and the parent
    lookup lands in TieringOffloadingManager.lookup(), which polls tiers for
    finished jobs.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_serve = False
        self.nested_polls = 0
        self.on_serve = self._look_up

    def _look_up(self, parent: ParentManager) -> None:
        self.in_serve = True
        try:
            parent.lookup(OffloadKey(b"\x01" * 8), ReqContext(req_id="peer"))
        finally:
            self.in_serve = False

    def get_finished_jobs(self) -> Iterable[JobResult]:
        if self.in_serve:
            self.nested_polls += 1
        return super().get_finished_jobs()


def test_parent_lookup_while_serving_keeps_the_round_contained():
    """A lookup issued while serving must not escape the control round.

    Before the fix it ran the step's once-per-step poll from the control
    thread: every tier got polled off-thread, the opted-in tier re-entrantly
    from inside its own serve, and the gate was left set, so the next step
    skipped its poll.
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
    opted_in = _LookingUpTier(tier_type="fake", **common)
    quiet = _QuietTier(tier_type="quiet", **common)
    manager = TieringOffloadingManager(
        primary_tier=primary,
        secondary_tiers=[opted_in, quiet],
    )
    try:
        # Several rounds, each issuing a parent lookup between steps.
        deadline = time.monotonic() + _SETTLE_S
        while opted_in.serves < 3 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert opted_in.serves >= 3, "control thread never served"

        with manager.lock:
            assert manager._processed_jobs_this_step is False
            assert opted_in.nested_polls == 0
            assert quiet.polls == 0, "quiet tier was polled off-thread"

            # The next step's first lookup still does its poll.
            manager.lookup(OffloadKey(b"\x02" * 8), _CTX)
            assert quiet.polls == 1
    finally:
        manager.shutdown()


class _HitTier(_FakeTier):
    """Second tier whose lookup HITs, so the parent queues a promotion."""

    tolerates_off_thread_polling: ClassVar[bool] = False

    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        return LookupResult.HIT


class _PromotingTier(_FakeTier):
    """Opted-in tier that serves a peer lookup through its parent.

    The parent lookup excludes this tier and lands on _HitTier, so the
    tiering manager queues a promotion in _pending_load_submissions, which
    only on_schedule_end() flushes.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.on_serve = self._promote
        self.promoted = threading.Event()

    def _promote(self, parent: ParentManager) -> None:
        parent.lookup(OffloadKey(b"\x07" * 8), ReqContext(req_id="peer"))
        self.promoted.set()


def _multi_tier_manager(primary, tier_cls, second_cls, **kwargs):
    common = {
        "offloading_spec": MagicMock(),
        "primary_kv_view": primary.get_kv_memoryview(),
    }
    return TieringOffloadingManager(
        primary_tier=primary,
        secondary_tiers=[
            tier_cls(tier_type="first", **common),
            second_cls(tier_type="second", **common),
        ],
        **kwargs,
    )


def test_multi_tier_manager_keeps_the_engine_stepping_while_idle():
    """With more than one secondary tier, a polled tier must keep the engine
    stepping even when idle.

    A tier-poll round can serve a peer whose lookup promotes a chunk out of
    another secondary tier. poll_tiers() never flushes promotions and only a
    scheduler step collects their completions, so if the engine parked it would
    never take that step and the load would strand. has_pending_work() is
    consulted before the park, so it must stay True here.
    """
    num_chunks = 8
    primary = CPUPrimaryTierOffloadingManager(
        num_chunks=num_chunks,
        mmap_region=_mock_mmap_region(num_chunks),
    )
    manager = _multi_tier_manager(primary, _FakeTier, _HitTier)
    try:
        assert manager._tier_poll_thread is not None
        assert manager.has_pending_work() is True
    finally:
        manager.shutdown()


def test_two_polled_tiers_keep_the_engine_stepping_while_idle():
    """Two polled tiers strand the same way as one polled plus one unpolled.

    A serve on the first can promote out of the second, and poll_tiers() never
    flushes it, so the keep-alive cannot key on the promoted tier being
    unpolled.
    """
    num_chunks = 8
    primary = CPUPrimaryTierOffloadingManager(
        num_chunks=num_chunks,
        mmap_region=_mock_mmap_region(num_chunks),
    )
    manager = _multi_tier_manager(primary, _FakeTier, _FakeTier)
    try:
        assert manager._tier_poll_thread is not None
        assert manager.has_pending_work() is True
    finally:
        manager.shutdown()


def test_off_thread_promotion_is_flushed_by_the_next_step():
    """An off-thread serve queues a promotion that only a step submits.

    Shows the queue actually fills from the control thread, which is why a
    multi-tier manager must keep the engine stepping (see
    test_multi_tier_manager_keeps_the_engine_stepping_while_idle).
    """
    num_chunks = 8
    primary = CPUPrimaryTierOffloadingManager(
        num_chunks=num_chunks,
        mmap_region=_mock_mmap_region(num_chunks),
    )
    manager = _multi_tier_manager(primary, _PromotingTier, _HitTier)
    tier = manager.secondary_tiers[0]
    try:
        assert tier.promoted.wait(_SETTLE_S), "control thread never served"

        with manager.lock:
            assert manager._pending_load_submissions
            manager.on_schedule_end(_EMPTY_SCHEDULE_END)
            assert not manager._pending_load_submissions
    finally:
        manager.shutdown()


def test_dead_poll_thread_makes_the_manager_keep_the_engine_stepping(
    manager_and_tier,
):
    """With no live thread the engine tick is the only driver.

    A failing loop now backs off rather than exiting, so this guards the
    defensive branch: if the field ever holds a finished thread (for example a
    join that failed to clear it), the manager must keep the engine stepping.
    """
    manager, tier = manager_and_tier
    assert tier.served.wait(_SETTLE_S)
    assert manager.has_pending_work() is False

    manager._stop_tier_poll_thread()
    # _stop_tier_poll_thread() clears the field only when the join succeeds.
    # Simulate a finished thread left in the field.
    dead = threading.Thread(target=lambda: None)
    dead.start()
    dead.join()
    manager._tier_poll_thread = dead

    assert not dead.is_alive()
    assert manager.has_pending_work() is True


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


def _wedge_control_thread(manager, monkeypatch) -> threading.Event:
    """Replace the control thread with one that ignores the stop request."""
    manager._stop_tier_poll_thread()
    release = threading.Event()
    stuck = threading.Thread(target=release.wait, daemon=True)
    stuck.start()
    manager._tier_poll_thread = stuck
    monkeypatch.setattr(tiering_manager, "_TIER_POLL_JOIN_TIMEOUT_S", 0.1)
    return release


def test_shutdown_tears_down_under_the_lock_when_the_thread_is_stuck(
    manager_and_tier, monkeypatch
):
    manager, tier = manager_and_tier
    release = _wedge_control_thread(manager, monkeypatch)
    try:
        manager.shutdown()
    finally:
        release.set()
    assert tier.shutdown_saw_live_thread is not None
    assert manager.lock.acquire(timeout=0.5), "lock leaked by shutdown"
    manager.lock.release()


def test_shutdown_raises_when_the_caller_holds_the_lock(manager_and_tier, monkeypatch):
    """Skipping tier teardown must not be silent.

    The lock is not reentrant, so a caller that wraps shutdown() in it can only
    time out; that has to surface instead of leaking the tiers.
    """
    manager, tier = manager_and_tier
    release = _wedge_control_thread(manager, monkeypatch)
    try:
        with manager.lock, pytest.raises(RuntimeError, match="without holding"):
            manager.shutdown()
    finally:
        release.set()
    assert tier.shutdown_saw_live_thread is None


def test_failure_burst_does_not_stop_the_thread(manager_and_tier):
    """The backoff budget is time, not rounds.

    At a 1 ms interval, a few hundred failing rounds take well under a second;
    a fault that short must not turn the tier-poll thread off for good.
    """
    manager, tier = manager_and_tier
    assert tier.served.wait(_SETTLE_S)
    tier.served.clear()
    failures_left = 300

    def flaky(parent):
        nonlocal failures_left
        if failures_left > 0:
            failures_left -= 1
            raise RuntimeError("transient")

    tier.on_serve = flaky

    assert tier.served.wait(_SETTLE_S * 10), "thread did not recover"
    assert failures_left == 0
    assert _live_control_threads()


def test_sustained_failure_backs_off_but_keeps_serving(monkeypatch):
    """The backoff path is non-terminal: an idle engine parks, so a thread that
    exited after sustained failure would leave peers unserved with nothing to
    wake the engine. It must back off and keep retrying, then resume the normal
    interval once a round succeeds."""
    monkeypatch.setattr(tiering_manager, "_TIER_POLL_BACK_OFF_AFTER_S", 0.2)
    monkeypatch.setattr(tiering_manager, "_TIER_POLL_BACKOFF_S", 0.05)
    manager, tier = _make_manager()
    try:
        assert tier.served.wait(_SETTLE_S)

        def always_fails(parent):
            raise RuntimeError("persistent")

        tier.on_serve = always_fails
        time.sleep(0.5)
        assert _live_control_threads(), "thread must survive sustained failure"

        tier.served.clear()
        tier.on_serve = None
        assert tier.served.wait(_SETTLE_S * 2), "thread did not resume serving"
    finally:
        manager.shutdown()


class _RecordingStop:
    """Stand-in for the tier-poll stop Event that records the waits it is given.

    Lets a test drive _tier_poll_loop() on the calling thread and read back the
    interval each round paused for, without real sleeping.
    """

    def __init__(self, rounds: int):
        self.waits: list[float] = []
        self._rounds = rounds
        self._set = False

    def is_set(self) -> bool:
        return self._set or len(self.waits) >= self._rounds

    def set(self) -> None:
        self._set = True

    def wait(self, timeout: float | None = None) -> bool:
        self.waits.append(timeout)
        return False


def _run_failing_rounds(manager, rounds: int, monkeypatch, **constants):
    """Drive _tier_poll_loop() on this thread through `rounds` failing rounds.

    Stops the real control thread first so only this call drives the loop, and
    returns the interval each round paused for.
    """
    for name, value in constants.items():
        monkeypatch.setattr(tiering_manager, name, value)
    manager._stop_tier_poll_thread()
    manager.secondary_tiers[0].on_serve = lambda parent: (_ for _ in ()).throw(
        RuntimeError("persistent")
    )
    stop = _RecordingStop(rounds)
    manager._tier_poll_stop = stop
    manager._tier_poll_loop()
    return stop.waits


def test_backoff_waits_at_least_the_configured_interval(monkeypatch):
    """Backing off must never poll faster than tier_poll_interval_s.

    The wait is max(tier_poll_interval_s, 1 s), so a user who set a slow
    interval does not get a faster loop after a fault.
    """
    manager, _ = _make_manager(tier_poll_interval_s=3.0)
    try:
        waits = _run_failing_rounds(
            manager, 4, monkeypatch, _TIER_POLL_BACK_OFF_AFTER_S=0.0
        )
    finally:
        manager.shutdown()
    assert waits == [3.0, 3.0, 3.0, 3.0]


def test_backoff_floors_at_one_second_for_a_fast_interval(monkeypatch):
    """A sub-second interval still backs off to the 1 s floor."""
    manager, _ = _make_manager(tier_poll_interval_s=0.001)
    try:
        waits = _run_failing_rounds(
            manager, 3, monkeypatch, _TIER_POLL_BACK_OFF_AFTER_S=0.0
        )
    finally:
        manager.shutdown()
    assert waits == [1.0, 1.0, 1.0]


def test_sustained_failure_logs_one_traceback_then_summaries(monkeypatch):
    """A permanent fault must not emit a full traceback every round.

    The first failure of a run logs a traceback; later ones are one-line
    summaries, so an idle rank with a broken peer cannot flood the log.
    """
    manager, _ = _make_manager()
    fake_logger = MagicMock()
    monkeypatch.setattr(tiering_manager, "logger", fake_logger)
    try:
        _run_failing_rounds(
            manager,
            4,
            monkeypatch,
            _TIER_POLL_BACK_OFF_AFTER_S=0.0,
            _TIER_POLL_ERROR_LOG_INTERVAL_S=0.0,
        )
    finally:
        manager.shutdown()
    assert fake_logger.exception.call_count == 1
    assert fake_logger.warning.call_count == 3


def test_lock_releases_on_error(manager_and_tier):
    """Leaving a with-block on an exception must not leak the lock."""
    manager, _ = manager_and_tier

    with pytest.raises(RuntimeError), manager.lock:
        raise RuntimeError("boom")

    assert manager.lock.acquire(timeout=0.5), "lock leaked after an error"
    manager.lock.release()


def test_lock_excludes_other_threads(manager_and_tier):
    """While one thread holds the lock, another cannot take it."""
    manager, _ = manager_and_tier

    with manager.lock:
        acquired: list[bool] = []
        t = threading.Thread(
            target=lambda: acquired.append(manager.lock.acquire(timeout=0.01))
        )
        t.start()
        t.join(timeout=5.0)
        assert acquired == [False]


def test_lock_defaults_to_no_op():
    """Managers entered from one thread only pay nothing for the lock.

    The base default, inherited by the CPU manager, must be a no-op context
    manager that may be entered re-entrantly.
    """
    base_lock = OffloadingManager.lock.fget(MagicMock())  # type: ignore[attr-defined]  # property
    assert isinstance(base_lock, nullcontext)
    with base_lock, base_lock:
        pass

    cpu = CPUOffloadingManager(num_chunks=4)
    with cpu.lock, cpu.lock:
        pass
