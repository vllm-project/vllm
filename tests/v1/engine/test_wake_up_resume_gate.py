# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Regression test for issue #44395 — EngineCore.wake_up() must not resume
scheduling after a partial wake (e.g. wake_up(tags=["weights"])).

Pre-fix, `wake_up` called `resume_scheduler()` unconditionally after
`model_executor.wake_up(tags)`. With a tag set that left the executor
still partially asleep (KV cache offloaded, weights restored), the next
forward pass — either a DP rank's `execute_dummy_batch()` busy-loop or
an externally-arriving request — wrote into released GPU memory and
crashed with "CUDA illegal memory access".

The fix gates `resume_scheduler()` on `not self.model_executor.is_sleeping`,
so scheduling stays paused until the caller's follow-up full wake_up()
(no tags) clears `is_sleeping`.

These are direct mock tests on EngineCore.wake_up — no model load, no
GPU — so they run in seconds and remain green on machines without CUDA.
"""

from unittest.mock import MagicMock

from vllm.v1.engine.core import EngineCore


def _make_core(executor_is_sleeping: bool) -> EngineCore:
    """Build a minimal EngineCore that exercises wake_up()'s gating logic.

    Skips EngineCore.__init__ (heavy: loads model, allocates KV cache, etc.) —
    wake_up() only touches `model_executor` and `resume_scheduler`, so those
    are the only attributes we populate.
    """
    core = EngineCore.__new__(EngineCore)
    core.model_executor = MagicMock()
    core.model_executor.is_sleeping = executor_is_sleeping
    core.resume_scheduler = MagicMock()
    return core


def test_partial_wake_does_not_resume_scheduler():
    """Partial wake — executor still asleep — resume_scheduler must NOT fire.

    This is the regression catcher: pre-fix code called resume_scheduler()
    unconditionally, and this assertion would fail.
    """
    core = _make_core(executor_is_sleeping=True)

    core.wake_up(tags=["weights"])

    core.model_executor.wake_up.assert_called_once_with(["weights"])
    core.resume_scheduler.assert_not_called()


def test_full_wake_resumes_scheduler():
    """Full wake — executor reports awake — resume_scheduler fires exactly once.

    Happy path; guards against an over-correction that would leave scheduling
    permanently paused.
    """
    core = _make_core(executor_is_sleeping=False)

    core.wake_up()

    core.model_executor.wake_up.assert_called_once_with(None)
    core.resume_scheduler.assert_called_once()


def test_full_wake_after_partial_resumes_scheduler():
    """The contract from the PR body: the caller follows a partial wake with
    a full wake. After the full wake clears is_sleeping, scheduling resumes.
    """
    core = _make_core(executor_is_sleeping=True)

    # 1) Partial wake — executor stays asleep — no resume.
    core.wake_up(tags=["weights"])
    assert core.resume_scheduler.call_count == 0

    # 2) Caller follows up with a full wake — executor now reports awake —
    #    resume fires.
    core.model_executor.is_sleeping = False
    core.wake_up()
    assert core.resume_scheduler.call_count == 1


def test_scheduling_only_tag_does_not_invoke_executor_wake_up():
    """tags=["scheduling"] is a level-0 wake that intentionally bypasses the
    executor (the `if tags is None or tags:` branch filters it out after the
    "scheduling" entry is removed). The gating change must not regress this.
    """
    core = _make_core(executor_is_sleeping=False)

    core.wake_up(tags=["scheduling"])

    core.model_executor.wake_up.assert_not_called()
    core.resume_scheduler.assert_called_once()
