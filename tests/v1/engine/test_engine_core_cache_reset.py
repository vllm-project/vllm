# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from concurrent.futures import Future
from unittest.mock import MagicMock

import pytest

from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine.core import EngineCoreProc

pytestmark = pytest.mark.cpu_test


def _pausable_engine_core(
    connector_reset_result: bool, deferred: bool
) -> EngineCoreProc:
    core = object.__new__(EngineCoreProc)
    core.mm_receiver_cache = None
    core.batch_queue = None
    core.engines_running = deferred
    core._idle_state_callbacks = []
    core.model_executor = MagicMock()

    scheduler = MagicMock()
    scheduler.has_requests.return_value = False
    scheduler.has_unfinished_requests.return_value = False
    scheduler.log_stats = False
    scheduler.connector = MagicMock()
    scheduler.connector.reset_cache.return_value = connector_reset_result

    scheduler.reset_prefix_cache.side_effect = lambda *_args, **_kwargs: (
        Scheduler.reset_connector_cache(scheduler)
    )
    core.scheduler = scheduler
    return core


def _sleep(core: EngineCoreProc) -> None:
    result = core.sleep(level=1, mode="keep")
    if isinstance(result, Future):
        core.engines_running = False
        core._notify_idle_state_callbacks()
        result.result(timeout=0)


@pytest.mark.parametrize("deferred", [False, True], ids=["sync", "future"])
def test_sleep_fails_closed_when_connector_cache_reset_fails(deferred: bool):
    core = _pausable_engine_core(False, deferred)

    with pytest.raises(RuntimeError, match="KV connector cache"):
        _sleep(core)

    core.scheduler.reset_prefix_cache.assert_called_once_with(True, True)
    core.model_executor.reset_mm_cache.assert_called_once_with()
    core.scheduler.reset_encoder_cache.assert_called_once_with()
    core.model_executor.reset_encoder_cache.assert_called_once_with()
    core.model_executor.sleep.assert_not_called()


def test_deferred_pause_reports_finish_failure_through_future():
    core = _pausable_engine_core(True, deferred=True)
    result = core.sleep(level=1, mode="keep")
    assert isinstance(result, Future)

    finish_error = RuntimeError("synthetic finish failure")
    core._finish_pause = MagicMock(side_effect=finish_error)
    core.engines_running = False
    core._notify_idle_state_callbacks()

    assert result.done()
    assert result.exception(timeout=0) is finish_error
