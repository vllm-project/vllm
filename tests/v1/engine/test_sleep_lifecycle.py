# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import Mock

from vllm.v1.engine.core import EngineCore


def test_kv_cache_release_waits_for_asynchronous_pause():
    pause_future: Future[None] = Future()
    discard = Mock()
    core = SimpleNamespace(
        pause_scheduler=Mock(return_value=pause_future),
        model_executor=SimpleNamespace(discard=discard),
    )

    result = EngineCore.release_kv_cache_memory(core, "wait")
    assert isinstance(result, Future)
    discard.assert_not_called()

    pause_future.set_result(None)

    assert result.result() is None
    discard.assert_called_once_with(("kv_cache",))
