# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from vllm_add_dummy_stat_logger.dummy_stat_logger import DummyStatLogger

from vllm.config import VllmConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.metrics.loggers import load_stat_logger_plugin_factories


def test_stat_logger_plugin_is_discovered(monkeypatch: pytest.MonkeyPatch):
    with monkeypatch.context() as m:
        m.setenv("VLLM_PLUGINS", "dummy_stat_logger")

        factories = load_stat_logger_plugin_factories()
        # there should only be our dummy plugin
        assert len(factories) == 1
        assert factories[0].func.__name__ == "DummyStatLogger"

        # instantiate and confirm the right type
        vllm_config = VllmConfig()
        instance = factories[0](vllm_config)
        assert isinstance(instance, DummyStatLogger)


def test_no_plugins_loaded_if_env_empty(monkeypatch: pytest.MonkeyPatch):
    with monkeypatch.context() as m:
        m.setenv("VLLM_PLUGINS", "")

        factories = load_stat_logger_plugin_factories()
        assert factories == []


def test_stat_logger_plugin_integration_with_engine(
        monkeypatch: pytest.MonkeyPatch):
    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")
        m.setenv("VLLM_PLUGINS", "dummy_stat_logger")
        # Explicitly turn off engine multiprocessing so
        # that the scheduler runs in this process
        m.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

        with pytest.raises(Exception) as exception_info:

            engine_args = AsyncEngineArgs(
                model="facebook/opt-125m",
                enforce_eager=True,  # reduce test time
            )

            engine = AsyncLLM.from_engine_args(engine_args=engine_args)

            sampling_params = SamplingParams(max_tokens=1)
            engine.add_request("0", "foo", sampling_params)
            engine.step()

        assert str(
            exception_info.value) == "Exception raised by DummyStatLogger"
