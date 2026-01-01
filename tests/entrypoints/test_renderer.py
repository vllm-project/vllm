# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import io
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock

import pybase64
import pytest
import torch

from vllm.entrypoints.renderer import CompletionRenderer, RenderConfig
from vllm.inputs.data import is_embeds_prompt


@dataclass
class MockModelConfig:
    max_model_len: int = 100
    encoder_config: dict | None = None
    enable_prompt_embeds: bool = True


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
    return CompletionRenderer(
        model_config=mock_model_config,
        tokenizer=mock_tokenizer,
        async_tokenizer_pool={},
    )


class TestRenderPrompt:
    """Test Category A: Basic Functionality Tests"""

    @pytest.mark.asyncio
    async def test_token_input(self, renderer):
        tokens = [101, 7592, 2088]
        results = await renderer.render_prompt(
            prompt_or_prompts=tokens, config=RenderConfig(max_length=100)
        )

        assert len(results) == 1
        assert results[0]["prompt_token_ids"] == tokens

    @pytest.mark.asyncio
    async def test_token_list_input(self, renderer):
        token_lists = [[101, 7592, 2088], [102, 1234, 5678, 9012], [103, 4567]]
        results = await renderer.render_prompt(
            prompt_or_prompts=token_lists, config=RenderConfig(max_length=100)
        )

        assert len(results) == 3
        assert results[0]["prompt_token_ids"] == [101, 7592, 2088]
        assert results[1]["prompt_token_ids"] == [102, 1234, 5678, 9012]
        assert results[2]["prompt_token_ids"] == [103, 4567]

    @pytest.mark.asyncio
    async def test_text_input(self, renderer, mock_async_tokenizer):
        mock_async_tokenizer.return_value = MockTokenizerResult([101, 7592, 2088])
        renderer.async_tokenizer_pool[renderer.tokenizer] = mock_async_tokenizer

        results = await renderer.render_prompt(
            prompt_or_prompts="Hello world", config=RenderConfig(max_length=100)
        )

        assert len(results) == 1
        assert results[0]["prompt_token_ids"] == [101, 7592, 2088]
        mock_async_tokenizer.assert_called_once()

    @pytest.mark.asyncio
    async def test_text_list_input(self, renderer, mock_async_tokenizer):
        mock_async_tokenizer.return_value = MockTokenizerResult([101, 7592, 2088])
        renderer.async_tokenizer_pool[renderer.tokenizer] = mock_async_tokenizer

        text_list_input = ["Hello world", "How are you?", "Good morning"]
        results = await renderer.render_prompt(
            prompt_or_prompts=text_list_input, config=RenderConfig(max_length=100)
        )

        assert len(results) == 3
        for result in results:
            assert result["prompt_token_ids"] == [101, 7592, 2088]
        assert mock_async_tokenizer.call_count == 3

    @pytest.mark.asyncio
    async def test_no_truncation(self, renderer, mock_async_tokenizer):
        mock_async_tokenizer.return_value = MockTokenizerResult([101, 7592, 2088])
        renderer.async_tokenizer_pool[renderer.tokenizer] = mock_async_tokenizer

        results = await renderer.render_prompt(
            prompt_or_prompts="Hello world", config=RenderConfig(max_length=100)
        )

        assert len(results) == 1
        call_args = mock_async_tokenizer.call_args
        assert (
            "truncation" not in call_args.kwargs
            or call_args.kwargs["truncation"] is False
        )

    @pytest.mark.asyncio
    async def test_truncation_positive(self, renderer, mock_async_tokenizer):
        mock_async_tokenizer.return_value = MockTokenizerResult(
            [101, 7592, 2088]
        )  # Truncated
        renderer.async_tokenizer_pool[renderer.tokenizer] = mock_async_tokenizer

        results = await renderer.render_prompt(
            prompt_or_prompts="Hello world",
            config=RenderConfig(max_length=100, truncate_prompt_tokens=50),
        )

        assert len(results) == 1
        call_args = mock_async_tokenizer.call_args
        assert call_args.kwargs["truncation"] is True
        assert call_args.kwargs["max_length"] == 50

    @pytest.mark.asyncio
    async def test_truncation_negative(self, renderer, mock_async_tokenizer):
        # Test that negative truncation uses model's max_model_len
        mock_async_tokenizer.return_value = MockTokenizerResult(
            [101, 7592, 2088]
        )  # Truncated to max_model_len
        renderer.async_tokenizer_pool[renderer.tokenizer] = mock_async_tokenizer

        results = await renderer.render_prompt(
            prompt_or_prompts="Hello world",
            config=RenderConfig(max_length=200, truncate_prompt_tokens=-1),
        )

        assert len(results) == 1
        call_args = mock_async_tokenizer.call_args
        assert call_args.kwargs["truncation"] is True
        assert call_args.kwargs["max_length"] == 100  # model's max_model_len

    @pytest.mark.asyncio
    async def test_token_truncation_last_elements(self, renderer):
        # Test that token truncation keeps the last N elements
        long_tokens = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]  # 10 tokens
        results = await renderer.render_prompt(
            prompt_or_prompts=long_tokens,
            config=RenderConfig(max_length=100, truncate_prompt_tokens=5),
        )

        assert len(results) == 1
        # Should keep the last 5 tokens: [105, 106, 107, 108, 109]
        assert results[0]["prompt_token_ids"] == [105, 106, 107, 108, 109]

    @pytest.mark.asyncio
    async def test_max_length_exceeded(self, renderer):
        long_tokens = list(range(150))  # Exceeds max_model_len=100

        with pytest.raises(ValueError, match="maximum context length"):
            await renderer.render_prompt(
                prompt_or_prompts=long_tokens, config=RenderConfig(max_length=100)
            )

    @pytest.mark.asyncio
    async def test_no_tokenizer_for_text(self, mock_model_config):
        renderer_no_tokenizer = CompletionRenderer(
            model_config=mock_model_config, tokenizer=None, async_tokenizer_pool={}
        )

        with pytest.raises(ValueError, match="No tokenizer available"):
            await renderer_no_tokenizer.render_prompt(
                prompt_or_prompts="Hello world", config=RenderConfig(max_length=100)
            )

    @pytest.mark.asyncio
    async def test_token_input_with_needs_detokenization(
        self, renderer, mock_async_tokenizer
    ):
        # When needs_detokenization=True for token inputs, renderer should
        # use the async tokenizer to decode and include the original text
        # in the returned prompt object.
        mock_async_tokenizer.decode = AsyncMock(return_value="decoded text")
        renderer.async_tokenizer_pool[renderer.tokenizer] = mock_async_tokenizer

        tokens = [1, 2, 3, 4]
        results = await renderer.render_prompt(
            prompt_or_prompts=tokens,
            config=RenderConfig(needs_detokenization=True),
        )

        assert len(results) == 1
        assert results[0]["prompt_token_ids"] == tokens
        assert results[0]["prompt"] == "decoded text"
        mock_async_tokenizer.decode.assert_awaited_once()


class TestRenderEmbedPrompt:
    def _create_test_embed_bytes(self, tensor: torch.Tensor) -> bytes:
        """Helper to create base64-encoded tensor bytes"""
        buffer = io.BytesIO()
        torch.save(tensor, buffer)
        buffer.seek(0)
        return pybase64.b64encode(buffer.read())

    @pytest.mark.asyncio
    async def test_single_prompt_embed(self, renderer):
        # Create a test tensor
        test_tensor = torch.randn(10, 768, dtype=torch.float32)
        embed_bytes = self._create_test_embed_bytes(test_tensor)

        results = await renderer.render_prompt_and_embeds(
            prompt_embeds=embed_bytes,
            config=RenderConfig(cache_salt="test_salt"),
        )

        assert len(results) == 1
        assert is_embeds_prompt(results[0])
        assert torch.allclose(results[0]["prompt_embeds"], test_tensor)
        assert results[0]["cache_salt"] == "test_salt"

    @pytest.mark.asyncio
    async def test_multiple_prompt_embeds(self, renderer):
        # Create multiple test tensors
        test_tensors = [
            torch.randn(8, 512, dtype=torch.float32),
            torch.randn(12, 512, dtype=torch.float32),
        ]
        embed_bytes_list = [self._create_test_embed_bytes(t) for t in test_tensors]

        results = await renderer.render_prompt_and_embeds(
            prompt_embeds=embed_bytes_list,
            config=RenderConfig(),
        )

        assert len(results) == 2
        for i, result in enumerate(results):
            assert is_embeds_prompt(result)
            assert torch.allclose(result["prompt_embeds"], test_tensors[i])

    @pytest.mark.asyncio
    async def test_prompt_embed_truncation(self, renderer):
        # Create tensor with more tokens than truncation limit
        test_tensor = torch.randn(20, 768, dtype=torch.float32)
        embed_bytes = self._create_test_embed_bytes(test_tensor)

        results = await renderer.render_prompt_and_embeds(
            prompt_embeds=embed_bytes,
            config=RenderConfig(truncate_prompt_tokens=10),
        )

        assert len(results) == 1
        # Should keep last 10 tokens
        expected = test_tensor[-10:]
        assert torch.allclose(results[0]["prompt_embeds"], expected)

    @pytest.mark.asyncio
    async def test_prompt_embed_different_dtypes(self, renderer):
        # Test different supported dtypes
        dtypes = [torch.float32, torch.float16, torch.bfloat16]

        for dtype in dtypes:
            test_tensor = torch.randn(5, 256, dtype=dtype)
            embed_bytes = self._create_test_embed_bytes(test_tensor)

            results = await renderer.render_prompt_and_embeds(
                prompt_embeds=embed_bytes,
                config=RenderConfig(),
            )

            assert len(results) == 1
            assert results[0]["prompt_embeds"].dtype == dtype

    @pytest.mark.asyncio
    async def test_prompt_embed_squeeze_batch_dim(self, renderer):
        # Test tensor with batch dimension gets squeezed
        test_tensor = torch.randn(1, 10, 768, dtype=torch.float32)
        embed_bytes = self._create_test_embed_bytes(test_tensor)

        results = await renderer.render_prompt_and_embeds(
            prompt_embeds=embed_bytes,
            config=RenderConfig(),
        )

        assert len(results) == 1
        # Should be squeezed to 2D
        assert results[0]["prompt_embeds"].shape == (10, 768)

    @pytest.mark.asyncio
    async def test_both_prompts_and_embeds(self, renderer, mock_async_tokenizer):
        # Set up text tokenization
        mock_async_tokenizer.return_value = MockTokenizerResult([101, 102, 103])
        renderer.async_tokenizer_pool[renderer.tokenizer] = mock_async_tokenizer

        # Create embed
        test_tensor = torch.randn(5, 256, dtype=torch.float32)
        embed_bytes = self._create_test_embed_bytes(test_tensor)

        results = await renderer.render_prompt_and_embeds(
            prompt_or_prompts="Hello world",
            prompt_embeds=embed_bytes,
            config=RenderConfig(),
        )

        assert len(results) == 2
        # First should be embed prompt
        assert is_embeds_prompt(results[0])
        # Second should be tokens prompt
        assert "prompt_token_ids" in results[1]
        assert results[1]["prompt_token_ids"] == [101, 102, 103]
