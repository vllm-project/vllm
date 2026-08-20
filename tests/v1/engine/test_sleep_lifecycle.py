# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm.v1.engine.core import EngineCore


def test_failed_kv_cache_release_keeps_scheduler_paused():
    core = SimpleNamespace(
        pause_scheduler=Mock(return_value=None),
        model_executor=SimpleNamespace(
            discard=Mock(side_effect=RuntimeError("discard failed"))
        ),
        resume_scheduler=Mock(),
    )

    with pytest.raises(RuntimeError, match="discard failed"):
        EngineCore.release_kv_cache_memory(core)

    core.resume_scheduler.assert_not_called()


def test_kv_cache_release_waits_for_asynchronous_pause():
    pause_future: Future[None] = Future()
    operation = Mock()
    core = SimpleNamespace(
        pause_scheduler=Mock(return_value=pause_future),
        model_executor=SimpleNamespace(discard=operation),
    )

    result = EngineCore.release_kv_cache_memory(core, "wait")
    assert isinstance(result, Future)
    operation.assert_not_called()

    pause_future.set_result(None)

    assert result.result() is None
    operation.assert_called_once_with(("kv_cache",))
