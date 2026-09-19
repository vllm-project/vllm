# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A failed ``EngineCore.__init__`` must tear down the executor's workers.

``EngineCore.__init__`` constructs the model executor before the scheduler, so
an exception in any later step (e.g. scheduler construction) leaves worker
processes running and holding GPU memory. ``run_engine_core()``'s ``finally``
cannot reach the partially-constructed engine (the ``engine_core = ...``
assignment never completes), so the only remaining cleanup would be the
executor's weakref finalizer -- which is not deterministic. These tests pin the
failure-path teardown behavior.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm.v1.engine.core import EngineCore

pytestmark = pytest.mark.cpu_test


class _FailingExecutor:
    instances: list[_FailingExecutor] = []

    def __init__(self, vllm_config) -> None:
        self.shutdown_called = False
        _FailingExecutor.instances.append(self)

    def shutdown(self) -> None:
        self.shutdown_called = True


def _make_vllm_config() -> SimpleNamespace:
    return SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_rank_local=None),
    )


def test_init_failure_shuts_down_executor():
    """The executor's workers must be torn down when a later init step raises."""
    _FailingExecutor.instances.clear()
    vllm_config = _make_vllm_config()

    with (
        patch.object(
            EngineCore,
            "_initialize_kv_caches",
            side_effect=RuntimeError("boom: simulated KV cache init failure"),
        ),
        pytest.raises(RuntimeError, match="boom"),
    ):
        EngineCore(
            vllm_config=vllm_config,
            executor_class=_FailingExecutor,
            log_stats=False,
        )

    assert len(_FailingExecutor.instances) == 1
    assert _FailingExecutor.instances[0].shutdown_called
