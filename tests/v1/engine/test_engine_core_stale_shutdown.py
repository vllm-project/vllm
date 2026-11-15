# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock

import pytest

import vllm.distributed.parallel_state as parallel_state
from vllm.v1.engine import core as core_module
from vllm.v1.engine.core import EngineCore


@pytest.mark.parametrize("newer_engine_running", [True, False])
def test_late_collected_inproc_engine_shutdown(monkeypatch, newer_engine_running):
    """A late-collected in-process engine must not tear down a newer engine's
    distributed state, but still tears down when no newer engine exists."""
    cleanup = MagicMock()
    monkeypatch.setattr(core_module, "cleanup_dist_env_and_memory", cleanup)
    monkeypatch.setattr(
        parallel_state, "_WORLD", object() if newer_engine_running else None
    )

    engine = object.__new__(EngineCore)
    engine.structured_output_manager = MagicMock()
    engine.model_executor = None
    engine.scheduler = None
    engine._world_group = object()
    engine.shutdown()

    assert cleanup.called != newer_engine_running
