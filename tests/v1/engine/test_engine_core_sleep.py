# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engine sleep ordering and failure propagation without allocating a model.

The neighboring test_engine_core.py starts CUDA engines at module scope.
These tests exercise production methods with mocked executor boundaries;
HTTP request recovery belongs to entrypoints/serve/dev/rlhf/state_transitions.
"""

from concurrent.futures import Future
from unittest.mock import Mock

import pytest

from vllm.v1.engine.core import EngineCore

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


@pytest.mark.parametrize("level", [1, 2])
@pytest.mark.parametrize("failure", [False, True])
def test_sleep_waits_for_drain_and_propagates_failure(level, failure):
    engine = EngineCore.__new__(EngineCore)
    drain: Future[None] = Future()
    engine.pause_scheduler = Mock(return_value=drain)
    engine.model_executor = Mock()
    result = engine.sleep(level, mode="wait")
    assert not result.done()
    engine.model_executor.sleep.assert_not_called()
    engine.pause_scheduler.assert_called_once_with(mode="wait", clear_cache=True)
    if failure:
        drain.set_exception(RuntimeError("drain failed"))
        with pytest.raises(RuntimeError, match="drain failed"):
            result.result(timeout=1)
        engine.model_executor.sleep.assert_not_called()
    else:
        drain.set_result(None)
        result.result(timeout=1)
        engine.model_executor.sleep.assert_called_once_with(level)


def test_level_zero_only_waits_for_scheduler():
    engine = EngineCore.__new__(EngineCore)
    drain: Future[None] = Future()
    engine.pause_scheduler = Mock(return_value=drain)
    engine.model_executor = Mock()
    assert engine.sleep(0, mode="wait") is drain
    drain.set_result(None)
    engine.pause_scheduler.assert_called_once_with(mode="wait", clear_cache=False)
    engine.model_executor.sleep.assert_not_called()


@pytest.mark.parametrize("resident", [False, True])
def test_partial_wake_resumes_only_when_all_memory_is_resident(resident):
    engine = EngineCore.__new__(EngineCore)
    engine.model_executor = Mock(is_sleeping=not resident)
    engine.resume_scheduler = Mock()
    engine.wake_up(["weights"])
    engine.model_executor.wake_up.assert_called_once_with(["weights"])
    if resident:
        engine.resume_scheduler.assert_called_once_with()
    else:
        engine.resume_scheduler.assert_not_called()


def test_executor_failure_is_propagated_after_drain():
    engine = EngineCore.__new__(EngineCore)
    drain: Future[None] = Future()
    engine.pause_scheduler = Mock(return_value=drain)
    engine.model_executor = Mock()
    engine.model_executor.sleep.side_effect = RuntimeError("offload failed")
    result = engine.sleep(1, mode="wait")
    drain.set_result(None)
    with pytest.raises(RuntimeError, match="offload failed"):
        result.result(timeout=1)
