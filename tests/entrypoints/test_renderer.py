# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from vllm.entrypoints.renderer import CompletionRenderer


@dataclass
class MockModelConfig:
    max_model_len: int = 100
    encoder_config: Optional[dict] = None


class MockTokenizerResult:

    def __init__(self, input_ids):
        self.input_ids = input_ids


@pytest.fixture
def mock_model_config():
    return MockModelConfig()


@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    return tokenizer


@pytest.fixture
def mock_async_tokenizer():
    async_tokenizer = AsyncMock()
    return async_tokenizer


@pytest.fixture
def renderer(mock_model_config, mock_tokenizer):
    return CompletionRenderer(model_config=mock_model_config,
                              tokenizer=mock_tokenizer,
                              async_tokenizer_pool={})


class TestRenderPrompt:
    """Test Category A: Basic Functionality Tests"""

    @pytest.mark.asyncio
    async def test_token_input(self, renderer):
        tokens = [101, 7592, 2088]
        results = await renderer.render_prompt(prompt_or_prompts=tokens,
                                               max_length=100)

        assert len(results) == 1
        assert results[0]["prompt_token_ids"] == tokens

    @pytest.mark.asyncio
    async def test_token_list_input(self, renderer):
        token_lists = [[101, 7592, 2088], [102, 1234, 5678, 9012], [103, 4567]]
        results = await renderer.render_prompt(prompt_or_prompts=token_lists,
                                               max_length=100)

        assert len(results) == 3
        assert results[0]["prompt_token_ids"] == [101, 7592, 2088]
        assert results[1]["prompt_token_ids"] == [102, 1234, 5678, 9012]
        assert results[2]["prompt_token_ids"] == [103, 4567]

    @pytest.mark.asyncio
    async def test_text_input(self, renderer, mock_async_tokenizer):
        mock_async_tokenizer.return_value = MockTokenizerResult(
            [101, 7592, 2088])
        renderer.async_tokenizer_pool[
            renderer.tokenizer] = mock_async_tokenizer

        results = await renderer.render_prompt(prompt_or_prompts="Hello world",
                                               max_length=100)

        assert len(results) == 1
        assert results[0]["prompt_token_ids"] == [101, 7592, 2088]
        mock_async_tokenizer.assert_called_once()

    @pytest.mark.asyncio
    async def test_text_list_input(self, renderer, mock_async_tokenizer):
        mock_async_tokenizer.return_value = MockTokenizerResult(
            [101, 7592, 2088])
        renderer.async_tokenizer_pool[
            renderer.tokenizer] = mock_async_tokenizer

        text_list_input = ["Hello world", "How are you?", "Good morning"]
        results = await renderer.render_prompt(
            prompt_or_prompts=text_list_input, max_length=100)

        assert len(results) == 3
        for result in results:
            assert result["prompt_token_ids"] == [101, 7592, 2088]
        assert mock_async_tokenizer.call_count == 3

    @pytest.mark.asyncio
    async def test_no_truncation(self, renderer, mock_async_tokenizer):
        mock_async_tokenizer.return_value = MockTokenizerResult(
            [101, 7592, 2088])
        renderer.async_tokenizer_pool[
            renderer.tokenizer] = mock_async_tokenizer

        results = await renderer.render_prompt(prompt_or_prompts="Hello world",
                                               max_length=100)

        assert len(results) == 1
        call_args = mock_async_tokenizer.call_args
        assert "truncation" not in call_args.kwargs or call_args.kwargs[
            "truncation"] is False

    @pytest.mark.asyncio
    async def test_truncation_positive(self, renderer, mock_async_tokenizer):
        mock_async_tokenizer.return_value = MockTokenizerResult(
            [101, 7592, 2088])  # Truncated
        renderer.async_tokenizer_pool[
            renderer.tokenizer] = mock_async_tokenizer

        results = await renderer.render_prompt(prompt_or_prompts="Hello world",
                                               max_length=100,
                                               truncate_prompt_tokens=50)

        assert len(results) == 1
        call_args = mock_async_tokenizer.call_args
        assert call_args.kwargs["truncation"] is True
        assert call_args.kwargs["max_length"] == 50

    @pytest.mark.asyncio
    async def test_token_truncation_last_elements(self, renderer):
        # Test that token truncation keeps the last N elements
        long_tokens = [100, 101, 102, 103, 104, 105, 106, 107, 108,
                       109]  # 10 tokens
        results = await renderer.render_prompt(prompt_or_prompts=long_tokens,
                                               max_length=100,
                                               truncate_prompt_tokens=5)

        assert len(results) == 1
        # Should keep the last 5 tokens: [105, 106, 107, 108, 109]
        assert results[0]["prompt_token_ids"] == [105, 106, 107, 108, 109]

    @pytest.mark.asyncio
    async def test_max_length_exceeded(self, renderer):
        long_tokens = list(range(150))  # Exceeds max_model_len=100

        with pytest.raises(ValueError, match="maximum context length"):
            await renderer.render_prompt(prompt_or_prompts=long_tokens,
                                         max_length=100)

    @pytest.mark.asyncio
    async def test_no_tokenizer_for_text(self, mock_model_config):
        renderer_no_tokenizer = CompletionRenderer(
            model_config=mock_model_config,
            tokenizer=None,
            async_tokenizer_pool={})

        with pytest.raises(ValueError, match="No tokenizer available"):
            await renderer_no_tokenizer.render_prompt(
                prompt_or_prompts="Hello world", max_length=100)
