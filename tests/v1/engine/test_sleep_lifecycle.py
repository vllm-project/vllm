# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import Mock

from vllm.v1.engine.core import EngineCore


def test_kv_cache_release_waits_for_asynchronous_pause():
    pause_future: Future[None] = Future()
    operation = Mock(return_value={"kv_cache"})
    core = SimpleNamespace(
        pause_scheduler=Mock(return_value=pause_future),
        model_executor=SimpleNamespace(discard=operation),
    )

    result = EngineCore.release_kv_cache_memory(core, "wait")
    assert isinstance(result, Future)
    operation.assert_not_called()

    pause_future.set_result(None)

    assert result.result() == {
        "scheduler": "paused",
        "kv_cache": "discarded",
    }
    operation.assert_called_once_with(("kv_cache",))


def test_sleep_noop_does_not_report_requested_resource_state():
    core = SimpleNamespace(
        pause_scheduler=Mock(return_value=None),
        model_executor=SimpleNamespace(sleep=Mock(return_value=False)),
    )

    assert EngineCore.sleep(core, level=2) == {"scheduler": "paused"}


def test_partial_wake_reports_only_woken_resource():
    core = SimpleNamespace(
        model_executor=SimpleNamespace(
            wake_up=Mock(return_value={"weights"}),
            is_sleeping=True,
        ),
        resume_scheduler=Mock(),
    )

    assert EngineCore.wake_up(core, ["weights"]) == {"weights": "resident"}
    core.resume_scheduler.assert_not_called()
