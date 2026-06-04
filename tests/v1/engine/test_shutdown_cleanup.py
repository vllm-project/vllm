# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import vllm.v1.engine.core_client as core_client_mod
import vllm.v1.engine.llm_engine as llm_engine_mod
from vllm.v1.engine.core_client import MPClient
from vllm.v1.engine.llm_engine import LLMEngine

pytestmark = pytest.mark.skip_global_cleanup


class DummyResources:

    def __init__(self, *, engine_dead: bool, engine_manager):
        self.engine_dead = engine_dead
        self.engine_manager = engine_manager
        self.cleanup = MagicMock()

    def __call__(self):
        self.cleanup()


def test_mp_client_shutdown_marks_engine_dead_before_manager_shutdown():
    client = object.__new__(MPClient)
    client._finalizer = MagicMock()
    client._finalizer.detach.return_value = object()

    engine_manager = MagicMock()
    client.resources = DummyResources(
        engine_dead=False,
        engine_manager=engine_manager,
    )

    client.shutdown(timeout=3.0)

    assert client.resources.engine_dead is True
    engine_manager.shutdown.assert_called_once_with(timeout=3.0)
    client.resources.cleanup.assert_called_once_with()


def test_mp_client_monitor_ignores_clean_engine_exit(monkeypatch: pytest.MonkeyPatch):
    client = object.__new__(MPClient)
    client._finalizer = SimpleNamespace(alive=True)
    client.resources = SimpleNamespace(
        engine_dead=False,
        engine_manager=SimpleNamespace(
            failed_proc_name=None,
            monitor_engine_liveness=lambda: None,
        ),
    )
    client.shutdown = MagicMock()

    thread_target = None

    class ImmediateThread:
        def __init__(self, *, target, daemon, name):
            nonlocal thread_target
            thread_target = target

        def start(self):
            thread_target()

    monkeypatch.setattr(core_client_mod, "Thread", ImmediateThread)
    logger_error = MagicMock()
    monkeypatch.setattr(core_client_mod.logger, "error", logger_error)

    client.start_engine_core_monitor()

    assert client.resources.engine_dead is False
    client.shutdown.assert_not_called()
    logger_error.assert_not_called()


def test_llm_engine_shutdown_cleans_up_owned_resources(
    monkeypatch: pytest.MonkeyPatch,
):
    llm_engine = object.__new__(LLMEngine)
    renderer = MagicMock()
    engine_core = MagicMock()
    dp_group = object()
    llm_engine.renderer = renderer
    llm_engine.engine_core = engine_core
    llm_engine.dp_group = dp_group
    llm_engine.external_launcher_dp = False

    shutdown_prometheus = MagicMock()
    destroy_dp_group = MagicMock()
    monkeypatch.setattr(llm_engine_mod, "shutdown_prometheus", shutdown_prometheus)
    monkeypatch.setattr(
        llm_engine_mod,
        "stateless_destroy_torch_distributed_process_group",
        destroy_dp_group,
    )

    llm_engine.shutdown(timeout=1.5)

    shutdown_prometheus.assert_called_once_with()
    renderer.shutdown.assert_called_once_with()
    engine_core.shutdown.assert_called_once_with(timeout=1.5)
    destroy_dp_group.assert_called_once_with(dp_group)
    assert llm_engine.renderer is None
    assert llm_engine.engine_core is None
    assert llm_engine.dp_group is None