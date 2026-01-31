# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import io
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock

import pybase64
import pytest
import torch

from vllm.inputs.data import is_embeds_prompt
from vllm.renderers import TokenizeParams
from vllm.renderers.hf import HfRenderer
from vllm.tokenizers.registry import tokenizer_args_from_config

MODEL_NAME = "openai-community/gpt2"


@dataclass
class MockHFConfig:
    model_type: str = "any"


@dataclass
class MockModelConfig:
    runner_type = "generate"
    model: str = MODEL_NAME
    tokenizer: str = MODEL_NAME
    trust_remote_code: bool = False
    max_model_len: int = 100
    tokenizer_revision = None
    tokenizer_mode = "auto"
    hf_config = MockHFConfig()
    encoder_config: dict[str, Any] | None = None
    enable_prompt_embeds: bool = True
    skip_tokenizer_init: bool = False


@pytest.fixture
def mock_model_config():
    return MockModelConfig()


@pytest.fixture
def mock_async_tokenizer():
    return AsyncMock()


@pytest.fixture
def renderer(mock_model_config):
    _, tokenizer_name, _, kwargs = tokenizer_args_from_config(mock_model_config)

    return HfRenderer(
        mock_model_config,
        tokenizer_kwargs={**kwargs, "tokenizer_name": tokenizer_name},
    )


class TestValidatePrompt:
    STRING_INPUTS = [
        "",
        "foo",
        "foo bar",
        "foo baz bar",
        "foo bar qux baz",
    ]

    TOKEN_INPUTS = [
        [-1],
        [1],
        [1, 2],
        [1, 3, 4],
        [1, 2, 4, 3],
    ]

    INPUTS_SLICES = [
        slice(None, None, -1),
        slice(None, None, 2),
        slice(None, None, -2),
    ]

    # Test that a nested mixed-type list of lists raises a TypeError.
    def test_empty_input(self, renderer):
        with pytest.raises(ValueError, match="at least one prompt"):
            renderer.render_completions([])

    def test_invalid_type(self, renderer):
        with pytest.raises(TypeError, match="string or an array of tokens"):
            renderer.render_completions([[1, 2], ["foo", "bar"]])

    @pytest.mark.parametrize("string_input", STRING_INPUTS)
    def test_string_consistent(self, renderer, string_input: str):
        assert renderer.render_completions(string_input) == renderer.render_completions(
            [string_input]
        )

    @pytest.mark.parametrize("token_input", TOKEN_INPUTS)
    def test_token_consistent(self, renderer, token_input: list[int]):
        assert renderer.render_completions(token_input) == renderer.render_completions(
            [token_input]
        )

    @pytest.mark.parametrize("inputs_slice", INPUTS_SLICES)
    def test_string_slice(self, renderer, inputs_slice: slice):
        assert renderer.render_completions(self.STRING_INPUTS)[
            inputs_slice
        ] == renderer.render_completions(self.STRING_INPUTS[inputs_slice])


class TestRenderPrompt:
    @pytest.mark.asyncio
    async def test_token_input(self, renderer):
        tokens = [101, 7592, 2088]
        prompts = await renderer.render_completions_async(tokens)
        results = await renderer.tokenize_prompts_async(
            prompts,
            TokenizeParams(max_total_tokens=100),
        )

        assert len(results) == 1
        assert results[0]["prompt_token_ids"] == tokens

    @pytest.mark.asyncio
    async def test_token_list_input(self, renderer):
        token_lists = [[101, 7592, 2088], [102, 1234, 5678, 9012], [103, 4567]]
        prompts = await renderer.render_completions_async(token_lists)
        results = await renderer.tokenize_prompts_async(
            prompts,
            TokenizeParams(max_total_tokens=100),
        )

        assert len(results) == 3
        assert results[0]["prompt_token_ids"] == [101, 7592, 2088]
        assert results[1]["prompt_token_ids"] == [102, 1234, 5678, 9012]
        assert results[2]["prompt_token_ids"] == [103, 4567]

    @pytest.mark.asyncio
    async def test_text_input(self, renderer, mock_async_tokenizer):
        mock_async_tokenizer.encode.return_value = [101, 7592, 2088]
        renderer._async_tokenizer = mock_async_tokenizer

        prompts = await renderer.render_completions_async("Hello world")
        results = await renderer.tokenize_prompts_async(
            prompts,
            TokenizeParams(max_total_tokens=100),
        )

        assert len(results) == 1
        assert results[0]["prompt_token_ids"] == [101, 7592, 2088]
        mock_async_tokenizer.encode.assert_called_once()

    @pytest.mark.asyncio
    async def test_text_list_input(self, renderer, mock_async_tokenizer):
        mock_async_tokenizer.encode.return_value = [101, 7592, 2088]
        renderer._async_tokenizer = mock_async_tokenizer

        text_list_input = ["Hello world", "How are you?", "Good morning"]
        prompts = await renderer.render_completions_async(text_list_input)
        results = await renderer.tokenize_prompts_async(
            prompts,
            TokenizeParams(max_total_tokens=100),
        )

        assert len(results) == 3
        for result in results:
            assert result["prompt_token_ids"] == [101, 7592, 2088]
        assert mock_async_tokenizer.encode.call_count == 3

    @pytest.mark.asyncio
    async def test_no_truncation(self, renderer, mock_async_tokenizer):
        mock_async_tokenizer.encode.return_value = [101, 7592, 2088]
        renderer._async_tokenizer = mock_async_tokenizer

        prompts = await renderer.render_completions_async("Hello world")
        results = await renderer.tokenize_prompts_async(
            prompts,
            TokenizeParams(max_total_tokens=100),
        )

        assert len(results) == 1
        call_args = mock_async_tokenizer.encode.call_args
        assert (
            "truncation" not in call_args.kwargs
            or call_args.kwargs["truncation"] is False
        )

    @pytest.mark.asyncio
    async def test_truncation_positive(self, renderer, mock_async_tokenizer):
        mock_async_tokenizer.encode.return_value = [101, 7592, 2088]  # Truncated
        renderer._async_tokenizer = mock_async_tokenizer

        prompts = await renderer.render_completions_async("Hello world")
        results = await renderer.tokenize_prompts_async(
            prompts,
            TokenizeParams(
                max_total_tokens=200,
                truncate_prompt_tokens=50,
            ),
        )

        assert len(results) == 1
        call_args = mock_async_tokenizer.encode.call_args
        assert call_args.kwargs["truncation"] is True
        assert call_args.kwargs["max_length"] == 50

    @pytest.mark.asyncio
    async def test_truncation_negative(self, renderer, mock_async_tokenizer):
        # Test that negative truncation uses model's max_model_len
        mock_async_tokenizer.encode.return_value = [
            101,
            7592,
            2088,
        ]  # Truncated to max_model_len
        renderer._async_tokenizer = mock_async_tokenizer

        prompts = await renderer.render_completions_async("Hello world")
        results = await renderer.tokenize_prompts_async(
            prompts,
            TokenizeParams(
                max_total_tokens=200,
                truncate_prompt_tokens=-1,
            ),
        )

        assert len(results) == 1
        call_args = mock_async_tokenizer.encode.call_args
        assert call_args.kwargs["truncation"] is True
        assert call_args.kwargs["max_length"] == 200

    @pytest.mark.asyncio
    async def test_token_truncation_last_elements(self, renderer):
        # Test that token truncation keeps the last N elements
        long_tokens = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]  # 10 tokens
        prompts = await renderer.render_completions_async(long_tokens)
        results = await renderer.tokenize_prompts_async(
            prompts,
            TokenizeParams(
                max_total_tokens=100,
                truncate_prompt_tokens=5,
            ),
        )

        assert len(results) == 1
        # Should keep the last 5 tokens: [105, 106, 107, 108, 109]
        assert results[0]["prompt_token_ids"] == [105, 106, 107, 108, 109]

    @pytest.mark.asyncio
    async def test_max_length_exceeded(self, renderer):
        long_tokens = list(range(150))  # Exceeds max_model_len=100

        prompts = await renderer.render_completions_async(long_tokens)

        with pytest.raises(ValueError, match="context length is only"):
            await renderer.tokenize_prompts_async(
                prompts,
                TokenizeParams(max_total_tokens=100),
            )

    @pytest.mark.asyncio
    async def test_no_tokenizer_for_text(self, renderer):
        renderer_no_tokenizer = HfRenderer.from_config(
            MockModelConfig(skip_tokenizer_init=True),
            tokenizer_kwargs={},
        )

        prompts = await renderer_no_tokenizer.render_completions_async("Hello world")

        with pytest.raises(ValueError, match="`skip_tokenizer_init=True`"):
            await renderer_no_tokenizer.tokenize_prompts_async(
                prompts,
                TokenizeParams(max_total_tokens=100),
            )

    @pytest.mark.asyncio
    async def test_token_input_with_needs_detokenization(
        self, renderer, mock_async_tokenizer
    ):
        # When needs_detokenization=True for token inputs, renderer should
        # use the async tokenizer to decode and include the original text
        # in the returned prompt object.
        mock_async_tokenizer.decode = AsyncMock(return_value="decoded text")
        renderer._async_tokenizer = mock_async_tokenizer

        tokens = [1, 2, 3, 4]
        prompts = await renderer.render_completions_async(tokens)
        results = await renderer.tokenize_prompts_async(
            prompts,
            TokenizeParams(
                max_total_tokens=renderer.config.max_model_len,
                needs_detokenization=True,
            ),
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

        prompts = await renderer.render_completions_async(prompt_embeds=embed_bytes)
        results = await renderer.tokenize_prompts_async(
            prompts,
            TokenizeParams(max_total_tokens=renderer.config.max_model_len),
        )

        assert len(results) == 1
        assert is_embeds_prompt(results[0])
        assert torch.allclose(results[0]["prompt_embeds"], test_tensor)

    @pytest.mark.asyncio
    async def test_multiple_prompt_embeds(self, renderer):
        # Create multiple test tensors
        test_tensors = [
            torch.randn(8, 512, dtype=torch.float32),
            torch.randn(12, 512, dtype=torch.float32),
        ]
        embed_bytes_list = [self._create_test_embed_bytes(t) for t in test_tensors]

        prompts = await renderer.render_completions_async(
            prompt_embeds=embed_bytes_list
        )
        results = await renderer.tokenize_prompts_async(
            prompts,
            TokenizeParams(max_total_tokens=renderer.config.max_model_len),
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

        prompts = await renderer.render_completions_async(prompt_embeds=embed_bytes)
        results = await renderer.tokenize_prompts_async(
            prompts,
            TokenizeParams(
                max_total_tokens=renderer.config.max_model_len,
                truncate_prompt_tokens=10,
            ),
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

            prompts = await renderer.render_completions_async(prompt_embeds=embed_bytes)
            results = await renderer.tokenize_prompts_async(
                prompts,
                TokenizeParams(max_total_tokens=renderer.config.max_model_len),
            )

            assert len(results) == 1
            assert results[0]["prompt_embeds"].dtype == dtype

    @pytest.mark.asyncio
    async def test_prompt_embed_squeeze_batch_dim(self, renderer):
        # Test tensor with batch dimension gets squeezed
        test_tensor = torch.randn(1, 10, 768, dtype=torch.float32)
        embed_bytes = self._create_test_embed_bytes(test_tensor)

        prompts = await renderer.render_completions_async(prompt_embeds=embed_bytes)
        results = await renderer.tokenize_prompts_async(
            prompts,
            TokenizeParams(max_total_tokens=renderer.config.max_model_len),
        )

        assert len(results) == 1
        # Should be squeezed to 2D
        assert results[0]["prompt_embeds"].shape == (10, 768)

    @pytest.mark.asyncio
    async def test_both_prompts_and_embeds(self, renderer, mock_async_tokenizer):
        # Set up text tokenization
        mock_async_tokenizer.encode.return_value = [101, 102, 103]
        renderer._async_tokenizer = mock_async_tokenizer

        # Create embed
        test_tensor = torch.randn(5, 256, dtype=torch.float32)
        embed_bytes = self._create_test_embed_bytes(test_tensor)

        prompts = await renderer.render_completions_async(
            "Hello world",
            prompt_embeds=embed_bytes,
        )
        results = await renderer.tokenize_prompts_async(
            prompts,
            TokenizeParams(max_total_tokens=renderer.config.max_model_len),
        )

        assert len(results) == 2
        # First should be embed prompt
        assert is_embeds_prompt(results[0])
        # Second should be tokens prompt
        assert "prompt_token_ids" in results[1]
        assert results[1]["prompt_token_ids"] == [101, 102, 103]
