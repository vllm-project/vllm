# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from vllm.v1.engine import async_llm as async_llm_module
from vllm.v1.engine import core as core_module
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.core import EngineCoreProc, EngineShutdownState
from vllm.v1.engine.exceptions import EngineDeadError

pytestmark = pytest.mark.skip_global_cleanup


def _make_stub_async_llm(get_output_side_effect):
    llm = object.__new__(AsyncLLM)
    llm.output_handler = None
    llm.engine_core = SimpleNamespace(
        get_output_async=AsyncMock(side_effect=get_output_side_effect),
        shutdown=lambda timeout=None: None,
    )
    llm.output_processor = MagicMock()
    llm.log_stats = False
    llm.logger_manager = None
    llm.renderer = SimpleNamespace(
        mm_processor_cache=None,
        shutdown=lambda: None,
    )
    return llm


@pytest.mark.asyncio
async def test_output_handler_logs_engine_dead_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
):
    error = EngineDeadError()
    llm = _make_stub_async_llm(error)

    mock_error = MagicMock()
    mock_exception = MagicMock()
    monkeypatch.setattr(async_llm_module.logger, "error", mock_error)
    monkeypatch.setattr(async_llm_module.logger, "exception", mock_exception)

    llm._run_output_handler()
    assert llm.output_handler is not None
    await llm.output_handler

    mock_error.assert_called_once_with("AsyncLLM output_handler failed: %s", error)
    mock_exception.assert_not_called()
    llm.output_processor.propagate_error.assert_called_once_with(error)


@pytest.mark.asyncio
async def test_output_handler_logs_unexpected_exception_with_traceback(
    monkeypatch: pytest.MonkeyPatch,
):
    error = RuntimeError("unexpected failure")
    llm = _make_stub_async_llm(error)

    mock_error = MagicMock()
    mock_exception = MagicMock()
    monkeypatch.setattr(async_llm_module.logger, "error", mock_error)
    monkeypatch.setattr(async_llm_module.logger, "exception", mock_exception)

    llm._run_output_handler()
    assert llm.output_handler is not None
    await llm.output_handler

    mock_exception.assert_called_once_with("AsyncLLM output_handler failed.")
    mock_error.assert_not_called()
    llm.output_processor.propagate_error.assert_called_once_with(error)


def _patch_run_engine_core_env(monkeypatch: pytest.MonkeyPatch):
    for name in (
        "maybe_register_config_serialize_by_value",
        "set_process_title",
        "maybe_init_worker_tracer",
        "decorate_logs",
    ):
        monkeypatch.setattr(core_module, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(
        core_module,
        "SignalCallback",
        lambda callback: SimpleNamespace(trigger=lambda: None, stop=lambda: None),
    )
    monkeypatch.setattr(core_module.signal, "signal", lambda *args: None)


def _make_vllm_config():
    parallel_config = SimpleNamespace(
        data_parallel_size=1,
        numa_bind=False,
        reconfigure_for_independent_dp_rank=lambda: None,
    )
    return SimpleNamespace(shutdown_timeout=0, parallel_config=parallel_config)


def test_run_engine_core_exits_with_system_exit_on_startup_failure(
    monkeypatch: pytest.MonkeyPatch,
):
    _patch_run_engine_core_env(monkeypatch)
    vllm_config = _make_vllm_config()

    def _fail_init(*args, **kwargs):
        raise RuntimeError("Worker failed with CUDA OOM")

    mock_exception = MagicMock()
    monkeypatch.setattr(core_module, "EngineCoreProc", _fail_init)
    monkeypatch.setattr(core_module.logger, "exception", mock_exception)

    with pytest.raises(SystemExit) as exc_info:
        EngineCoreProc.run_engine_core(vllm_config=vllm_config)

    assert exc_info.value.code == 1
    mock_exception.assert_called_once_with("EngineCore failed to start.")


def test_run_engine_core_exits_with_system_exit_on_fatal_error(
    monkeypatch: pytest.MonkeyPatch,
):
    _patch_run_engine_core_env(monkeypatch)
    vllm_config = _make_vllm_config()
    calls: list[str] = []

    proc = SimpleNamespace(
        shutdown_state=EngineShutdownState.RUNNING,
        has_work=lambda: False,
        vllm_config=vllm_config,
        run_busy_loop=MagicMock(side_effect=RuntimeError("Executor failed.")),
        _send_engine_dead=lambda: calls.append("send_engine_dead"),
        shutdown=lambda: calls.append("shutdown"),
    )

    mock_exception = MagicMock()
    monkeypatch.setattr(core_module, "EngineCoreProc", lambda *args, **kwargs: proc)
    monkeypatch.setattr(core_module.logger, "exception", mock_exception)

    with pytest.raises(SystemExit) as exc_info:
        EngineCoreProc.run_engine_core(vllm_config=vllm_config)

    assert exc_info.value.code == 1
    mock_exception.assert_called_once_with("EngineCore encountered a fatal error.")
    assert calls == ["send_engine_dead", "shutdown"]
