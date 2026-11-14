# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# An in-process engine (InprocClient) shuts its EngineCore down from a
# weakref finalizer, so the shutdown runs whenever the old engine is garbage
# collected. When that happens while a newer engine in the same process is
# starting up (e.g. at the gc.collect() in its worker's init_device), the old
# engine must not tear down the newer engine's distributed groups, or the newer
# one fails with "pipeline model parallel group is not initialized".

from unittest.mock import MagicMock

import pytest

import vllm.distributed.parallel_state as parallel_state
from vllm.v1.engine import core as core_module
from vllm.v1.engine.core import EngineCore, EngineCoreProc


def _bare_engine(cls: type[EngineCore], world_group: object) -> EngineCore:
    engine = object.__new__(cls)
    engine.structured_output_manager = MagicMock()
    engine.model_executor = None
    engine.scheduler = None
    engine._world_group = world_group
    return engine


@pytest.fixture
def cleanup(monkeypatch):
    mock = MagicMock()
    monkeypatch.setattr(core_module, "cleanup_dist_env_and_memory", mock)
    return mock


def test_stale_inproc_engine_keeps_newer_groups(monkeypatch, cleanup):
    monkeypatch.setattr(parallel_state, "_WORLD", object())
    _bare_engine(EngineCore, world_group=object()).shutdown()
    cleanup.assert_not_called()


def test_inproc_engine_tears_down_its_own_groups(monkeypatch, cleanup):
    world = object()
    monkeypatch.setattr(parallel_state, "_WORLD", world)
    _bare_engine(EngineCore, world_group=world).shutdown()
    cleanup.assert_called_once()


def test_engine_core_proc_always_tears_down(monkeypatch, cleanup):
    monkeypatch.setattr(parallel_state, "_WORLD", object())
    engine = _bare_engine(EngineCoreProc, world_group=object())
    # Call the base teardown directly: EngineCoreProc.shutdown also stops
    # its IO threads, which this bare instance doesn't have.
    EngineCore.shutdown(engine)
    cleanup.assert_called_once()
