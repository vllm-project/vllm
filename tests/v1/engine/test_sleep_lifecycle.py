# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm.v1.engine.core import EngineCore


def test_kv_cache_release_requires_completed_pause():
    discard = Mock()
    core = SimpleNamespace(
        is_scheduler_paused=Mock(return_value=False),
        scheduler=SimpleNamespace(has_requests=Mock(return_value=False)),
        batch_queue=[],
        _reset_caches=Mock(),
        model_executor=SimpleNamespace(discard=discard),
    )

    with pytest.raises(RuntimeError, match="completed pause"):
        EngineCore.release_kv_cache_memory(core)
    discard.assert_not_called()

    core.is_scheduler_paused = Mock(return_value=True)
    EngineCore.release_kv_cache_memory(core)
    core._reset_caches.assert_called_once()
    discard.assert_called_once_with(("kv_cache",))
