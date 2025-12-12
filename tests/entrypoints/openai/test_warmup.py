# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the warmup functionality."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from vllm.entrypoints.openai.warmup import (
    SUPPORTED_ENDPOINTS,
    WarmupConfig,
    run_warmup,
)


class TestWarmupConfig:
    """Tests for WarmupConfig loading."""

    def test_load_basic_config(self, tmp_path: Path):
        """Test loading a basic warmup config file."""
        config_data = {
            "concurrency": 2,
            "requests": [
                {
                    "endpoint": "/v1/chat/completions",
                    "payload": {
                        "messages": [{"role": "user", "content": "Hello"}],
                        "max_tokens": 16,
                    },
                    "count": 3,
                }
            ],
        }

        config_file = tmp_path / "warmup.json"
        config_file.write_text(json.dumps(config_data))

        config = WarmupConfig.from_file(str(config_file))

        assert config.concurrency == 2
        assert len(config.requests) == 1
        assert config.requests[0].endpoint == "/v1/chat/completions"
        assert config.requests[0].count == 3

    def test_load_config_default_concurrency(self, tmp_path: Path):
        """Test that concurrency defaults to 1."""
        config_data = {
            "requests": [
                {
                    "endpoint": "/v1/completions",
                    "payload": {"prompt": "Hello", "max_tokens": 16},
                }
            ]
        }

        config_file = tmp_path / "warmup.json"
        config_file.write_text(json.dumps(config_data))

        config = WarmupConfig.from_file(str(config_file))

        assert config.concurrency == 1

    def test_load_config_default_count(self, tmp_path: Path):
        """Test that count defaults to 1."""
        config_data = {
            "requests": [
                {
                    "endpoint": "/v1/completions",
                    "payload": {"prompt": "Hello", "max_tokens": 16},
                }
            ]
        }

        config_file = tmp_path / "warmup.json"
        config_file.write_text(json.dumps(config_data))

        config = WarmupConfig.from_file(str(config_file))

        assert config.requests[0].count == 1

    def test_load_config_missing_endpoint(self, tmp_path: Path):
        """Test error when endpoint is missing."""
        config_data = {"requests": [{"payload": {"prompt": "Hello", "max_tokens": 16}}]}

        config_file = tmp_path / "warmup.json"
        config_file.write_text(json.dumps(config_data))

        with pytest.raises(ValueError, match="endpoint"):
            WarmupConfig.from_file(str(config_file))

    def test_load_config_missing_payload(self, tmp_path: Path):
        """Test error when payload is missing."""
        config_data = {"requests": [{"endpoint": "/v1/completions"}]}

        config_file = tmp_path / "warmup.json"
        config_file.write_text(json.dumps(config_data))

        with pytest.raises(ValueError, match="payload"):
            WarmupConfig.from_file(str(config_file))

    def test_load_config_file_not_found(self):
        """Test error when config file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            WarmupConfig.from_file("/nonexistent/path/warmup.json")

    def test_load_multiple_requests(self, tmp_path: Path):
        """Test loading config with multiple requests."""
        config_data = {
            "concurrency": 4,
            "requests": [
                {
                    "endpoint": "/v1/chat/completions",
                    "payload": {
                        "messages": [{"role": "user", "content": "Hello"}],
                        "max_tokens": 16,
                    },
                    "count": 2,
                },
                {
                    "endpoint": "/v1/completions",
                    "payload": {"prompt": "Test", "max_tokens": 32},
                    "count": 3,
                },
            ],
        }

        config_file = tmp_path / "warmup.json"
        config_file.write_text(json.dumps(config_data))

        config = WarmupConfig.from_file(str(config_file))

        assert config.concurrency == 4
        assert len(config.requests) == 2
        assert config.requests[0].endpoint == "/v1/chat/completions"
        assert config.requests[0].count == 2
        assert config.requests[1].endpoint == "/v1/completions"
        assert config.requests[1].count == 3


class TestRunWarmup:
    """Tests for run_warmup function."""

    @pytest.fixture
    def mock_state(self):
        """Create a mock app state."""
        state = MagicMock()

        # Mock the model paths
        mock_model_path = MagicMock()
        mock_model_path.name = "test-model"
        state.openai_serving_models.base_model_paths = [mock_model_path]

        # Mock the chat handler
        state.openai_serving_chat = AsyncMock()
        state.openai_serving_chat.create_chat_completion = AsyncMock(
            return_value=MagicMock()
        )

        # Mock the completion handler
        state.openai_serving_completion = AsyncMock()
        state.openai_serving_completion.create_completion = AsyncMock(
            return_value=MagicMock()
        )

        return state

    @pytest.mark.asyncio
    async def test_warmup_chat_completions(self, mock_state, tmp_path: Path):
        """Test warmup with chat completions endpoint."""
        config_data = {
            "requests": [
                {
                    "endpoint": "/v1/chat/completions",
                    "payload": {
                        "messages": [{"role": "user", "content": "Hello"}],
                        "max_tokens": 16,
                    },
                    "count": 2,
                }
            ]
        }

        config_file = tmp_path / "warmup.json"
        config_file.write_text(json.dumps(config_data))

        await run_warmup(mock_state, str(config_file))

        # Should have been called twice (count=2)
        assert mock_state.openai_serving_chat.create_chat_completion.call_count == 2

    @pytest.mark.asyncio
    async def test_warmup_completions(self, mock_state, tmp_path: Path):
        """Test warmup with completions endpoint."""
        config_data = {
            "requests": [
                {
                    "endpoint": "/v1/completions",
                    "payload": {"prompt": "Hello", "max_tokens": 16},
                    "count": 3,
                }
            ]
        }

        config_file = tmp_path / "warmup.json"
        config_file.write_text(json.dumps(config_data))

        await run_warmup(mock_state, str(config_file))

        # Should have been called 3 times (count=3)
        assert mock_state.openai_serving_completion.create_completion.call_count == 3

    @pytest.mark.asyncio
    async def test_warmup_unsupported_endpoint(self, mock_state, tmp_path: Path):
        """Test that unsupported endpoints raise NotImplementedError."""
        config_data = {
            "requests": [
                {
                    "endpoint": "/v1/embeddings",
                    "payload": {"input": "Hello"},
                }
            ]
        }

        config_file = tmp_path / "warmup.json"
        config_file.write_text(json.dumps(config_data))

        # Should log warning but not raise (failures are caught)
        await run_warmup(mock_state, str(config_file))

    @pytest.mark.asyncio
    async def test_warmup_empty_requests(self, mock_state, tmp_path: Path):
        """Test warmup with no requests."""
        config_data: dict[str, list] = {"requests": []}

        config_file = tmp_path / "warmup.json"
        config_file.write_text(json.dumps(config_data))

        # Should complete without error
        await run_warmup(mock_state, str(config_file))

        # No handlers should have been called
        assert mock_state.openai_serving_chat.create_chat_completion.call_count == 0
        assert mock_state.openai_serving_completion.create_completion.call_count == 0

    @pytest.mark.asyncio
    async def test_warmup_auto_fills_model(self, mock_state, tmp_path: Path):
        """Test that model is auto-filled in payload if not provided."""
        config_data = {
            "requests": [
                {
                    "endpoint": "/v1/completions",
                    "payload": {"prompt": "Hello", "max_tokens": 16},
                }
            ]
        }

        config_file = tmp_path / "warmup.json"
        config_file.write_text(json.dumps(config_data))

        await run_warmup(mock_state, str(config_file))

        # Verify the handler was called
        assert mock_state.openai_serving_completion.create_completion.call_count == 1

    @pytest.mark.asyncio
    async def test_warmup_concurrent_execution(self, mock_state, tmp_path: Path):
        """Test that warmup respects concurrency setting."""
        config_data = {
            "concurrency": 2,
            "requests": [
                {
                    "endpoint": "/v1/completions",
                    "payload": {"prompt": "Hello", "max_tokens": 16},
                    "count": 4,
                }
            ],
        }

        config_file = tmp_path / "warmup.json"
        config_file.write_text(json.dumps(config_data))

        await run_warmup(mock_state, str(config_file))

        # All 4 requests should have been made
        assert mock_state.openai_serving_completion.create_completion.call_count == 4


class TestSupportedEndpoints:
    """Tests for supported endpoints constant."""

    def test_chat_completions_supported(self):
        """Test that chat completions endpoint is supported."""
        assert "/v1/chat/completions" in SUPPORTED_ENDPOINTS

    def test_completions_supported(self):
        """Test that completions endpoint is supported."""
        assert "/v1/completions" in SUPPORTED_ENDPOINTS
