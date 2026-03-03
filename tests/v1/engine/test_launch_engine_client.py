# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for LaunchEngineClient (GPU-less EngineClient)."""

from unittest.mock import MagicMock, patch

import pytest

from vllm.config import VllmConfig
from vllm.engine.protocol import EngineClient
from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingParams
from vllm.v1.engine.launch import LaunchEngineClient


@pytest.fixture
def mock_vllm_config():
    config = MagicMock(spec=VllmConfig)
    config.model_config = MagicMock()
    config.model_config.io_processor_plugin = None
    return config


@pytest.fixture
def launch_client(mock_vllm_config):
    with (
        patch("vllm.v1.engine.launch.renderer_from_config") as mock_renderer,
        patch("vllm.v1.engine.launch.get_io_processor") as mock_io,
        patch("vllm.v1.engine.launch.InputProcessor") as mock_input,
    ):
        mock_renderer.return_value = MagicMock()
        mock_io.return_value = MagicMock()
        mock_input.return_value = MagicMock()
        yield LaunchEngineClient(mock_vllm_config)


def test_implements_engine_client(launch_client):
    assert isinstance(launch_client, EngineClient)


def test_init_creates_components(mock_vllm_config):
    with (
        patch("vllm.v1.engine.launch.renderer_from_config") as mock_renderer,
        patch("vllm.v1.engine.launch.get_io_processor") as mock_io,
        patch("vllm.v1.engine.launch.InputProcessor") as mock_input,
    ):
        mock_renderer.return_value = MagicMock()
        mock_io.return_value = MagicMock()
        mock_input.return_value = MagicMock()

        client = LaunchEngineClient(mock_vllm_config)

        mock_renderer.assert_called_once_with(mock_vllm_config)
        mock_io.assert_called_once_with(
            mock_vllm_config,
            mock_renderer.return_value,
            mock_vllm_config.model_config.io_processor_plugin,
        )
        mock_input.assert_called_once_with(mock_vllm_config, mock_renderer.return_value)
        assert client.renderer is mock_renderer.return_value
        assert client.io_processor is mock_io.return_value
        assert client.input_processor is mock_input.return_value


def test_from_vllm_config_factory(mock_vllm_config):
    with (
        patch("vllm.v1.engine.launch.renderer_from_config"),
        patch("vllm.v1.engine.launch.get_io_processor"),
        patch("vllm.v1.engine.launch.InputProcessor"),
    ):
        client = LaunchEngineClient.from_vllm_config(mock_vllm_config)
        assert isinstance(client, LaunchEngineClient)
        assert client.vllm_config is mock_vllm_config


def test_model_config_attribute(launch_client, mock_vllm_config):
    assert launch_client.model_config is mock_vllm_config.model_config


@pytest.mark.asyncio
async def test_get_supported_tasks(launch_client):
    tasks = await launch_client.get_supported_tasks()
    assert tasks == ("render",)


@pytest.mark.asyncio
async def test_generate_raises(launch_client):
    with pytest.raises(NotImplementedError, match="does not support inference"):
        async for _ in launch_client.generate(
            prompt="test",
            sampling_params=SamplingParams(),
            request_id="req-1",
        ):
            pass


@pytest.mark.asyncio
async def test_encode_raises(launch_client):
    with pytest.raises(NotImplementedError, match="does not support inference"):
        async for _ in launch_client.encode(
            prompt="test",
            pooling_params=MagicMock(),
            request_id="req-1",
        ):
            pass


@pytest.mark.asyncio
async def test_abort_is_noop(launch_client):
    await launch_client.abort("req-1")


@pytest.mark.asyncio
async def test_pause_generation_is_noop(launch_client):
    await launch_client.pause_generation()


@pytest.mark.asyncio
async def test_resume_generation_is_noop(launch_client):
    await launch_client.resume_generation()


@pytest.mark.asyncio
async def test_check_health_is_noop(launch_client):
    await launch_client.check_health()


@pytest.mark.asyncio
async def test_do_log_stats_is_noop(launch_client):
    await launch_client.do_log_stats()


@pytest.mark.asyncio
async def test_start_stop_profile_is_noop(launch_client):
    await launch_client.start_profile()
    await launch_client.stop_profile()


@pytest.mark.asyncio
async def test_cache_reset_methods(launch_client):
    await launch_client.reset_mm_cache()
    await launch_client.reset_encoder_cache()
    result = await launch_client.reset_prefix_cache()
    assert result is True


@pytest.mark.asyncio
async def test_sleep_wake_methods(launch_client):
    await launch_client.sleep()
    await launch_client.wake_up()
    assert await launch_client.is_sleeping() is False


@pytest.mark.asyncio
async def test_is_paused_returns_false(launch_client):
    assert await launch_client.is_paused() is False


@pytest.mark.asyncio
async def test_is_tracing_enabled_returns_false(launch_client):
    assert await launch_client.is_tracing_enabled() is False


@pytest.mark.asyncio
async def test_add_lora_returns_false(launch_client):
    mock_lora = MagicMock(spec=LoRARequest)
    assert await launch_client.add_lora(mock_lora) is False


def test_is_running(launch_client):
    assert launch_client.is_running is True


def test_is_stopped(launch_client):
    assert launch_client.is_stopped is False


def test_errored(launch_client):
    assert launch_client.errored is False


def test_dead_error(launch_client):
    error = launch_client.dead_error
    assert isinstance(error, RuntimeError)
    assert "does not support inference" in str(error)
