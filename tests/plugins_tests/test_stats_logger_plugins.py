# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from dummy_stat_logger.dummy_stat_logger import DummyStatLogger

from vllm.config import VllmConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.metrics.loggers import load_stat_logger_plugin_factories


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
        assert isinstance(instance, DummyStatLogger)


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

        assert len(engine.logger_manager.stat_loggers) == 2
        assert len(engine.logger_manager.stat_loggers[0].per_engine_stat_loggers) == 1
        assert isinstance(
            engine.logger_manager.stat_loggers[0].per_engine_stat_loggers[0],
            DummyStatLogger,
        )

        engine.shutdown()
