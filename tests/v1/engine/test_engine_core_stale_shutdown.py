# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock

import pytest

import vllm.distributed.parallel_state as parallel_state
from vllm.v1.engine import core as core_module
from vllm.v1.engine.core import EngineCore


@pytest.mark.parametrize(
    "current_world,expect_cleanup",
    [
        ("newer", False),  # A newer engine replaced the world group.
        ("none", True),  # No engine holds a world group.
        ("own", True),  # The world group is still this engine's own.
    ],
)
def test_late_collected_inproc_engine_shutdown(
    monkeypatch, current_world, expect_cleanup
):
    """A late-collected in-process engine must not tear down a newer engine's
    distributed state, but still tears down its own or when none is left."""
    cleanup = MagicMock()
    monkeypatch.setattr(core_module, "cleanup_dist_env_and_memory", cleanup)

    own_group = object()
    world = {"newer": object(), "none": None, "own": own_group}[current_world]
    monkeypatch.setattr(parallel_state, "_WORLD", world)

    engine = object.__new__(EngineCore)
    engine.structured_output_manager = MagicMock()
    engine.model_executor = None
    engine.scheduler = None
    engine._world_group = own_group
    engine.shutdown()

    assert cleanup.called == expect_cleanup
