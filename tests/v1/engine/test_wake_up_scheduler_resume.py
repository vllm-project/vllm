# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit test for the partial-wake scheduler-resume race (issue #44395).

After a partial ``wake_up(tags=["weights"])`` the KV cache is still released,
yet ``EngineCore.wake_up`` used to resume the scheduler unconditionally. The
engine could then step (including the DP-lockstep ``execute_dummy_batch`` path
on idle ranks) against a released KV cache and crash with an illegal memory
access.

These tests pin the contract that ``EngineCore.wake_up`` only resumes the
scheduler once the executor is *fully* resident (``is_sleeping == False``).
They drive the real ``EngineCore.wake_up`` method against a tiny fake that
faithfully mirrors ``Executor``'s tag bookkeeping, so no GPU / model load is
needed.
"""

from unittest.mock import MagicMock

from vllm.v1.engine.core import EngineCore


class _FakeExecutor:
    """Mirrors Executor.sleep/wake_up tag bookkeeping (abstract.py:318-356).

    is_sleeping flips to False only once every sleeping tag is woken, which is
    exactly the residency signal EngineCore.wake_up must gate the scheduler
    resume on.
    """

    def __init__(self):
        self.is_sleeping = False
        self.sleeping_tags: set[str] = set()

    def sleep(self, level: int = 1):
        self.sleeping_tags = {"weights", "kv_cache"}
        self.is_sleeping = True

    def wake_up(self, tags=None):
        # Mirror Executor.wake_up's early-exit guard (abstract.py:332): a
        # wake_up on an already-resident executor is a no-op and must not
        # disturb the tag bookkeeping.
        if not self.is_sleeping:
            return
        if tags:
            for tag in tags:
                self.sleeping_tags.discard(tag)
        else:
            self.sleeping_tags.clear()
        if not self.sleeping_tags:
            self.is_sleeping = False


def _make_engine_core(executor: _FakeExecutor) -> EngineCore:
    """A bare EngineCore with only the attributes wake_up() touches.

    We avoid EngineCore.__init__ (which builds a real executor + scheduler and
    needs a GPU) and instead attach a recording resume_scheduler plus the fake
    executor, then exercise the real bound wake_up method.
    """
    core = EngineCore.__new__(EngineCore)
    core.model_executor = executor
    core.resume_scheduler = MagicMock(name="resume_scheduler")
    return core


def test_partial_weights_wake_does_not_resume_scheduler():
    """wake_up(tags=["weights"]) leaves KV asleep -> scheduler must stay paused."""
    executor = _FakeExecutor()
    executor.sleep(level=1)
    core = _make_engine_core(executor)

    core.wake_up(tags=["weights"])

    # KV cache tag still present -> executor not fully resident.
    assert executor.is_sleeping is True
    assert executor.sleeping_tags == {"kv_cache"}
    # The race: resuming here would let a forward run against released KV.
    core.resume_scheduler.assert_not_called()


def test_final_kv_wake_resumes_scheduler():
    """The completing wake_up(tags=["kv_cache"]) makes us resident -> resume."""
    executor = _FakeExecutor()
    executor.sleep(level=1)
    core = _make_engine_core(executor)

    core.wake_up(tags=["weights"])
    core.resume_scheduler.assert_not_called()

    core.wake_up(tags=["kv_cache"])

    assert executor.is_sleeping is False
    assert executor.sleeping_tags == set()
    core.resume_scheduler.assert_called_once()


def test_full_wake_resumes_scheduler():
    """A plain full wake_up() is fully resident -> resume immediately."""
    executor = _FakeExecutor()
    executor.sleep(level=1)
    core = _make_engine_core(executor)

    core.wake_up()

    assert executor.is_sleeping is False
    core.resume_scheduler.assert_called_once()


def test_level0_scheduling_wake_resumes_scheduler():
    """Level-0 wake (tags=["scheduling"]) never touched GPU; executor is not
    sleeping, so scheduling must resume."""
    executor = _FakeExecutor()  # never slept the executor (level-0 sleep)
    core = _make_engine_core(executor)

    core.wake_up(tags=["scheduling"])

    # "scheduling" is stripped; executor.wake_up is not called for level 0.
    assert executor.is_sleeping is False
    core.resume_scheduler.assert_called_once()


def test_reversed_tag_order_resume():
    """Order-independence: waking kv_cache first leaves weights asleep (no
    resume), then waking weights empties the tags and resumes exactly once.

    Tags can be woken in either order; residency -- not call order -- gates the
    resume, so the scheduler must stay paused until the last tag is woken.
    """
    executor = _FakeExecutor()
    executor.sleep(level=1)
    core = _make_engine_core(executor)

    # kv_cache first: weights tag still pending -> still partially asleep.
    core.wake_up(tags=["kv_cache"])
    assert executor.is_sleeping is True
    assert executor.sleeping_tags == {"weights"}
    core.resume_scheduler.assert_not_called()

    # weights second: completing wake -> fully resident -> resume once.
    core.wake_up(tags=["weights"])
    assert executor.is_sleeping is False
    assert executor.sleeping_tags == set()
    core.resume_scheduler.assert_called_once()
