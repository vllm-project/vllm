# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ``EngineCore.wake_up`` scheduler-resume gating (issue #44395).

These exercise only the pure branching logic of ``wake_up`` via a mock
``self``, so they need no model, GPU, or engine construction and run on CPU.
"""

from unittest.mock import MagicMock

from vllm.v1.engine.core import EngineCore


def _run_wake_up(tags, executor_is_sleeping):
    """Invoke the unbound ``EngineCore.wake_up`` on a mock self and return it.

    ``model_executor.is_sleeping`` models whether the executor is still
    (partially) asleep *after* ``model_executor.wake_up(tags)`` ran.
    """
    core = MagicMock()
    core.model_executor.is_sleeping = executor_is_sleeping
    EngineCore.wake_up(core, tags)
    return core


class TestWakeUpResumeScheduler:
    def test_partial_weights_wake_does_not_resume_scheduler(self):
        # tags=["weights"] restores weights but leaves the KV cache released,
        # so the executor is still sleeping and the scheduler must stay paused.
        core = _run_wake_up(["weights"], executor_is_sleeping=True)
        core.model_executor.wake_up.assert_called_once_with(["weights"])
        core.resume_scheduler.assert_not_called()

    def test_final_kv_cache_wake_resumes_scheduler(self):
        # The follow-up wake makes the executor fully resident -> resume.
        core = _run_wake_up(["kv_cache"], executor_is_sleeping=False)
        core.model_executor.wake_up.assert_called_once_with(["kv_cache"])
        core.resume_scheduler.assert_called_once_with()

    def test_full_wake_resumes_scheduler(self):
        core = _run_wake_up(None, executor_is_sleeping=False)
        core.model_executor.wake_up.assert_called_once_with(None)
        core.resume_scheduler.assert_called_once_with()

    def test_scheduling_only_wake_resumes_without_touching_executor(self):
        # Level-0 wake: executor memory is untouched (not sleeping) and only
        # scheduling resumes.
        core = _run_wake_up(["scheduling"], executor_is_sleeping=False)
        core.model_executor.wake_up.assert_not_called()
        core.resume_scheduler.assert_called_once_with()

    def test_weights_with_scheduling_tag_still_gated(self):
        # "scheduling" is stripped, ["weights"] is processed; the executor is
        # still sleeping so the scheduler stays paused.
        core = _run_wake_up(["weights", "scheduling"], executor_is_sleeping=True)
        core.model_executor.wake_up.assert_called_once_with(["weights"])
        core.resume_scheduler.assert_not_called()
