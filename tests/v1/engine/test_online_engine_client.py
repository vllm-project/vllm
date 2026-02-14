# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for OnlineEngineClient (GPU-less EngineClient)."""

from unittest.mock import MagicMock, patch

import pytest

from vllm.config import VllmConfig
from vllm.engine.protocol import EngineClient
from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingParams
from vllm.v1.engine.online import OnlineEngineClient


@pytest.fixture
def mock_vllm_config():
    config = MagicMock(spec=VllmConfig)
    config.model_config = MagicMock()
    config.model_config.io_processor_plugin = None
    return config


@pytest.fixture
def online_client(mock_vllm_config):
    with (
        patch("vllm.v1.engine.online.renderer_from_config") as mock_renderer,
        patch("vllm.v1.engine.online.get_io_processor") as mock_io,
        patch("vllm.v1.engine.online.InputProcessor") as mock_input,
    ):
        mock_renderer.return_value = MagicMock()
        mock_io.return_value = MagicMock()
        mock_input.return_value = MagicMock()
        yield OnlineEngineClient(mock_vllm_config)


def test_implements_engine_client(online_client):
    assert isinstance(online_client, EngineClient)


def test_init_creates_components(mock_vllm_config):
    with (
        patch("vllm.v1.engine.online.renderer_from_config") as mock_renderer,
        patch("vllm.v1.engine.online.get_io_processor") as mock_io,
        patch("vllm.v1.engine.online.InputProcessor") as mock_input,
    ):
        mock_renderer.return_value = MagicMock()
        mock_io.return_value = MagicMock()
        mock_input.return_value = MagicMock()

        client = OnlineEngineClient(mock_vllm_config)

        mock_renderer.assert_called_once_with(mock_vllm_config)
        mock_io.assert_called_once_with(
            mock_vllm_config,
            mock_vllm_config.model_config.io_processor_plugin,
        )
        mock_input.assert_called_once_with(mock_vllm_config, mock_renderer.return_value)
        assert client.renderer is mock_renderer.return_value
        assert client.io_processor is mock_io.return_value
        assert client.input_processor is mock_input.return_value


def test_from_vllm_config_factory(mock_vllm_config):
    with (
        patch("vllm.v1.engine.online.renderer_from_config"),
        patch("vllm.v1.engine.online.get_io_processor"),
        patch("vllm.v1.engine.online.InputProcessor"),
    ):
        client = OnlineEngineClient.from_vllm_config(mock_vllm_config)
        assert isinstance(client, OnlineEngineClient)
        assert client.vllm_config is mock_vllm_config


def test_model_config_attribute(online_client, mock_vllm_config):
    assert online_client.model_config is mock_vllm_config.model_config


@pytest.mark.asyncio
async def test_get_supported_tasks(online_client):
    tasks = await online_client.get_supported_tasks()
    assert tasks == ("generate",)


@pytest.mark.asyncio
async def test_generate_raises(online_client):
    with pytest.raises(NotImplementedError, match="does not support inference"):
        async for _ in online_client.generate(
            prompt="test",
            sampling_params=SamplingParams(),
            request_id="req-1",
        ):
            pass


@pytest.mark.asyncio
async def test_encode_raises(online_client):
    with pytest.raises(NotImplementedError, match="does not support inference"):
        async for _ in online_client.encode(
            prompt="test",
            pooling_params=MagicMock(),
            request_id="req-1",
        ):
            pass


@pytest.mark.asyncio
async def test_abort_is_noop(online_client):
    await online_client.abort("req-1")


@pytest.mark.asyncio
async def test_pause_generation_is_noop(online_client):
    await online_client.pause_generation()


@pytest.mark.asyncio
async def test_resume_generation_is_noop(online_client):
    await online_client.resume_generation()


@pytest.mark.asyncio
async def test_check_health_is_noop(online_client):
    await online_client.check_health()


@pytest.mark.asyncio
async def test_do_log_stats_is_noop(online_client):
    await online_client.do_log_stats()


@pytest.mark.asyncio
async def test_start_stop_profile_is_noop(online_client):
    await online_client.start_profile()
    await online_client.stop_profile()


@pytest.mark.asyncio
async def test_cache_reset_methods(online_client):
    await online_client.reset_mm_cache()
    await online_client.reset_encoder_cache()
    result = await online_client.reset_prefix_cache()
    assert result is True


@pytest.mark.asyncio
async def test_sleep_wake_methods(online_client):
    await online_client.sleep()
    await online_client.wake_up()
    assert await online_client.is_sleeping() is False


@pytest.mark.asyncio
async def test_is_paused_returns_false(online_client):
    assert await online_client.is_paused() is False


@pytest.mark.asyncio
async def test_is_tracing_enabled_returns_false(online_client):
    assert await online_client.is_tracing_enabled() is False


@pytest.mark.asyncio
async def test_add_lora_returns_false(online_client):
    mock_lora = MagicMock(spec=LoRARequest)
    assert await online_client.add_lora(mock_lora) is False


def test_is_running(online_client):
    assert online_client.is_running is True


def test_is_stopped(online_client):
    assert online_client.is_stopped is False


def test_errored(online_client):
    assert online_client.errored is False


def test_dead_error(online_client):
    error = online_client.dead_error
    assert isinstance(error, RuntimeError)
    assert "does not support inference" in str(error)
