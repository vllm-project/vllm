# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from dummy_stat_logger.dummy_stat_logger import DummyStatLogger

from vllm.config import VllmConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.metrics.loggers import (
    PerEngineStatLoggerAdapter,
    load_stat_logger_plugin_factories,
)


def test_stat_logger_plugin_is_discovered(monkeypatch: pytest.MonkeyPatch):
    with monkeypatch.context() as m:
        m.setenv("VLLM_PLUGINS", "dummy_stat_logger")

        factories = load_stat_logger_plugin_factories()
        assert len(factories) == 1, f"Expected 1 factory, got {len(factories)}"
        assert factories[0] is DummyStatLogger, (
            f"Expected DummyStatLogger class, got {factories[0]}"
        )

        # instantiate and confirm the right type
        vllm_config = VllmConfig()
        instance = factories[0](vllm_config)
        # dummy_stat_logger is loaded dynamically via entry points, so mypy
        # can't statically resolve it; the assert above then narrows
        # DummyStatLogger to the wrong type for the rest of this scope.
        assert isinstance(instance, DummyStatLogger)  # type: ignore[arg-type]


def test_no_plugins_loaded_if_env_empty(monkeypatch: pytest.MonkeyPatch):
    with monkeypatch.context() as m:
        m.setenv("VLLM_PLUGINS", "")

        factories = load_stat_logger_plugin_factories()
        assert factories == []


def test_invalid_stat_logger_plugin_raises(monkeypatch: pytest.MonkeyPatch):
    def fake_plugin_loader(group: str):
        assert group == "vllm.stat_logger_plugins"
        return {"bad": object()}

    with monkeypatch.context() as m:
        m.setattr(
            "vllm.v1.metrics.loggers.load_plugins_by_group",
            fake_plugin_loader,
        )
        with pytest.raises(
            TypeError,
            match="Stat logger plugin 'bad' must be a subclass of StatLoggerBase",
        ):
            load_stat_logger_plugin_factories()


@pytest.mark.asyncio
async def test_stat_logger_plugin_integration_with_engine(
    monkeypatch: pytest.MonkeyPatch,
):
    with monkeypatch.context() as m:
        m.setenv("VLLM_PLUGINS", "dummy_stat_logger")

        engine_args = AsyncEngineArgs(
            model="facebook/opt-125m",
            enforce_eager=True,  # reduce test time
            disable_log_stats=True,  # disable default loggers
        )

        engine = AsyncLLM.from_engine_args(engine_args=engine_args)

        logger_manager = engine.logger_manager
        assert logger_manager is not None
        assert len(logger_manager.stat_loggers) == 2
        per_engine_adapter = logger_manager.stat_loggers[0]
        assert isinstance(per_engine_adapter, PerEngineStatLoggerAdapter)
        assert len(per_engine_adapter.per_engine_stat_loggers) == 1
        assert isinstance(
            per_engine_adapter.per_engine_stat_loggers[0],
            DummyStatLogger,
        )

        engine.shutdown()
