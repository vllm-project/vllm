# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import io
from dataclasses import dataclass
from typing import Any

import pybase64
import pytest
import torch

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
    tokenizer_revision = None
    tokenizer_mode = "auto"
    hf_config = MockHFConfig()
    encoder_config: dict[str, Any] | None = None
    enable_prompt_embeds: bool = True
    skip_tokenizer_init: bool = False
    is_encoder_decoder: bool = False


@dataclass
class DummyTokenizer:
    truncation_side: str = "left"
    max_chars_per_token: int = 1

    def __post_init__(self) -> None:
        self._captured_encode_kwargs: dict = {}

    def decode(self, tokens: list[int]):
        return str(tokens)

    def encode(self, text: str, **kwargs):
        self._captured_encode_kwargs = kwargs

        in_length = len(text)
        truncation = kwargs.get("truncation")
        max_length = kwargs.get("max_length")
        if truncation and max_length is not None:
            return list(range(min(in_length, max_length)))

        return list(range(in_length))


def _build_renderer(
    model_config: MockModelConfig,
    *,
    truncation_side: str = "left",
    max_chars_per_token: int = 1,
):
    _, tokenizer_name, _, kwargs = tokenizer_args_from_config(model_config)

    renderer = HfRenderer(
        model_config,
        tokenizer_kwargs={**kwargs, "tokenizer_name": tokenizer_name},
    )

    if not model_config.skip_tokenizer_init:
        renderer._tokenizer = DummyTokenizer(
            truncation_side=truncation_side,
            max_chars_per_token=max_chars_per_token,
        )

    return renderer


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
    def test_empty_input(self):
        renderer = _build_renderer(MockModelConfig())

        with pytest.raises(ValueError, match="at least one prompt"):
            renderer.render_completions([])

    def test_invalid_type(self):
        renderer = _build_renderer(MockModelConfig())

        with pytest.raises(TypeError, match="should be a list of integers"):
            renderer.render_completions([[1, 2], ["foo", "bar"]])

    @pytest.mark.parametrize("string_input", STRING_INPUTS)
    def test_string_consistent(self, string_input: str):
        renderer = _build_renderer(MockModelConfig())

        assert [
            renderer.render_completion(string_input)
        ] == renderer.render_completions([string_input])

    @pytest.mark.parametrize("token_input", TOKEN_INPUTS)
    def test_token_consistent(self, token_input: list[int]):
        renderer = _build_renderer(MockModelConfig())

        assert [renderer.render_completion(token_input)] == renderer.render_completions(
            [token_input]
        )

    @pytest.mark.parametrize("inputs_slice", INPUTS_SLICES)
    def test_string_slice(self, inputs_slice: slice):
        renderer = _build_renderer(MockModelConfig())

        assert renderer.render_completions(self.STRING_INPUTS)[
            inputs_slice
        ] == renderer.render_completions(self.STRING_INPUTS[inputs_slice])


class TestRenderPrompt:
    def test_token_input(self):
        renderer = _build_renderer(MockModelConfig())

        tokens = [101, 7592, 2088]
        prompts = renderer.render_completions([tokens])
        results = renderer.tokenize_prompts(
            prompts,
            TokenizeParams(max_total_tokens=100),
        )

        assert len(results) == 1
        assert results[0]["prompt_token_ids"] == tokens

    def test_token_list_input(self):
        renderer = _build_renderer(MockModelConfig())

        token_lists = [[101, 7592, 2088], [102, 1234, 5678, 9012], [103, 4567]]
        prompts = renderer.render_completions(token_lists)
        results = renderer.tokenize_prompts(
            prompts,
            TokenizeParams(max_total_tokens=100),
        )

        assert len(results) == 3
        assert results[0]["prompt_token_ids"] == [101, 7592, 2088]
        assert results[1]["prompt_token_ids"] == [102, 1234, 5678, 9012]
        assert results[2]["prompt_token_ids"] == [103, 4567]

    def test_text_input(self):
        renderer = _build_renderer(MockModelConfig())

        text_input = "x" * 10
        prompts = renderer.render_completions([text_input])
        results = renderer.tokenize_prompts(
            prompts,
            TokenizeParams(max_total_tokens=100),
        )

        assert len(results) == 1
        assert len(results[0]["prompt_token_ids"]) == 10

    def test_text_list_input(self):
        renderer = _build_renderer(MockModelConfig())

        text_list_input = ["x" * 10, "x" * 12, "x" * 14]
        prompts = renderer.render_completions(text_list_input)
        results = renderer.tokenize_prompts(
            prompts,
            TokenizeParams(max_total_tokens=100),
        )

        assert len(results) == 3
        for text_input, result in zip(text_list_input, results):
            assert len(result["prompt_token_ids"]) == len(text_input)

    def test_zero_truncation(self):
        renderer = _build_renderer(MockModelConfig())

        prompts = renderer.render_completions(["x" * 200])
        results = renderer.tokenize_prompts(
            prompts,
            TokenizeParams(max_total_tokens=100, truncate_prompt_tokens=0),
        )

        assert len(results) == 1
        assert len(results[0]["prompt_token_ids"]) == 0

    def test_pos_truncation(self):
        renderer = _build_renderer(MockModelConfig())

        prompts = renderer.render_completions(["x" * 200])
        results = renderer.tokenize_prompts(
            prompts,
            TokenizeParams(max_total_tokens=100, truncate_prompt_tokens=50),
        )

        assert len(results) == 1
        assert len(results[0]["prompt_token_ids"]) == 50

    def test_neg_truncation(self):
        renderer = _build_renderer(MockModelConfig())

        prompts = renderer.render_completions(["x" * 200])
        results = renderer.tokenize_prompts(
            prompts,
            TokenizeParams(max_total_tokens=100, truncate_prompt_tokens=-1),
        )

        assert len(results) == 1
        assert len(results[0]["prompt_token_ids"]) == 100  # max_total_tokens

    def test_truncation_left(self):
        renderer = _build_renderer(MockModelConfig(), truncation_side="left")

        long_tokens = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]  # 10 tokens
        prompts = renderer.render_completions([long_tokens])
        results = renderer.tokenize_prompts(
            prompts,
            TokenizeParams(max_total_tokens=100, truncate_prompt_tokens=5),
        )

        assert len(results) == 1
        # Should keep the last 5 tokens: [105, 106, 107, 108, 109]
        assert results[0]["prompt_token_ids"] == [105, 106, 107, 108, 109]

    def test_truncation_right(self):
        renderer = _build_renderer(MockModelConfig(), truncation_side="right")

        long_tokens = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]  # 10 tokens
        prompts = renderer.render_completions([long_tokens])
        results = renderer.tokenize_prompts(
            prompts,
            TokenizeParams(max_total_tokens=100, truncate_prompt_tokens=5),
        )

        assert len(results) == 1
        # Should keep the first 5 tokens: [100, 101, 102, 103, 104]
        assert results[0]["prompt_token_ids"] == [100, 101, 102, 103, 104]

    def test_text_max_length_exceeded_obvious(self):
        renderer = _build_renderer(MockModelConfig(), max_chars_per_token=1)

        # Exceeds max_total_tokens and max_total_tokens * VLLM_MAX_CHARS_PER_TOKEN
        long_tokens = "x" * 150
        prompts = renderer.render_completions([long_tokens])

        with pytest.raises(
            ValueError,
            match="input characters and requested .* context length is only",
        ):
            renderer.tokenize_prompts(
                prompts,
                TokenizeParams(max_total_tokens=100),
            )

        # Should not even attempt tokenization
        assert renderer._tokenizer._captured_encode_kwargs == {}

    def test_text_max_length_exceeded_nonobvious(self):
        renderer = _build_renderer(MockModelConfig(), max_chars_per_token=2)

        # Exceeds max_total_tokens but not max_total_tokens * VLLM_MAX_CHARS_PER_TOKEN
        long_tokens = "x" * 150
        prompts = renderer.render_completions([long_tokens])

        with pytest.raises(
            ValueError,
            match="input tokens and requested .* context length is only",
        ):
            renderer.tokenize_prompts(
                prompts,
                TokenizeParams(max_total_tokens=100),
            )

        # Should only tokenize the first max_total_tokens + 1 tokens
        assert renderer._tokenizer._captured_encode_kwargs["truncation"] is True
        assert renderer._tokenizer._captured_encode_kwargs["max_length"] == 101

    def test_token_max_length_exceeded(self):
        renderer = _build_renderer(MockModelConfig())

        long_tokens = list(range(150))  # Exceeds max_total_tokens=100
        prompts = renderer.render_completions([long_tokens])

        with pytest.raises(
            ValueError,
            match="input tokens and requested .* context length is only",
        ):
            renderer.tokenize_prompts(
                prompts,
                TokenizeParams(max_total_tokens=100, truncate_prompt_tokens=None),
            )

    def test_no_tokenizer_for_text(self):
        renderer = _build_renderer(MockModelConfig(skip_tokenizer_init=True))

        prompts = renderer.render_completions(["Hello world"])

        with pytest.raises(ValueError, match="`skip_tokenizer_init=True`"):
            renderer.tokenize_prompts(
                prompts,
                TokenizeParams(max_total_tokens=100),
            )

    def test_token_input_with_needs_detokenization(self):
        renderer = _build_renderer(MockModelConfig())

        tokens = [1, 2, 3, 4]
        prompts = renderer.render_completions([tokens])
        results = renderer.tokenize_prompts(
            prompts,
            TokenizeParams(
                max_total_tokens=100,
                needs_detokenization=True,
            ),
        )

        assert len(results) == 1
        assert results[0]["prompt_token_ids"] == tokens
        assert results[0]["prompt"] == "[1, 2, 3, 4]"


class TestRenderEmbedPrompt:
    def _create_test_embed_bytes(self, tensor: torch.Tensor) -> bytes:
        """Helper to create base64-encoded tensor bytes"""
        buffer = io.BytesIO()
        torch.save(tensor, buffer)
        buffer.seek(0)
        return pybase64.b64encode(buffer.read())

    def test_single_prompt_embed(self):
        renderer = _build_renderer(MockModelConfig())

        # Create a test tensor
        tensor_input = torch.randn(10, 768, dtype=torch.float32)
        embed_bytes = self._create_test_embed_bytes(tensor_input)

        prompts = renderer.render_completions([embed_bytes])
        results = renderer.tokenize_prompts(
            prompts,
            TokenizeParams(max_total_tokens=100),
        )

        assert len(results) == 1
        assert torch.equal(results[0]["prompt_embeds"], tensor_input)

    def test_multiple_prompt_embeds(self):
        renderer = _build_renderer(MockModelConfig())

        # Create multiple test tensors
        tensor_inputs = [
            torch.randn(8, 512, dtype=torch.float32),
            torch.randn(12, 512, dtype=torch.float32),
        ]

        prompts = renderer.render_completions(
            [self._create_test_embed_bytes(t) for t in tensor_inputs],
        )
        results = renderer.tokenize_prompts(
            prompts,
            TokenizeParams(max_total_tokens=100),
        )

        assert len(results) == 2
        for i, result in enumerate(results):
            assert torch.allclose(result["prompt_embeds"], tensor_inputs[i])

    def test_prompt_embed_truncation(self):
        renderer = _build_renderer(MockModelConfig())

        # Create tensor with more tokens than truncation limit
        tensor_input = torch.randn(20, 768, dtype=torch.float32)

        prompts = renderer.render_completions(
            [self._create_test_embed_bytes(tensor_input)]
        )
        results = renderer.tokenize_prompts(
            prompts,
            TokenizeParams(
                max_total_tokens=100,
                truncate_prompt_tokens=10,
            ),
        )

        assert len(results) == 1
        # Should keep last 10 tokens
        expected = tensor_input[-10:]
        assert torch.equal(results[0]["prompt_embeds"], expected)

    def test_prompt_embed_different_dtypes(self):
        renderer = _build_renderer(MockModelConfig())

        # Test different supported dtypes
        dtypes = [torch.float32, torch.float16, torch.bfloat16]

        for dtype in dtypes:
            tensor_input = torch.randn(5, 256, dtype=dtype)

            prompts = renderer.render_completions(
                [self._create_test_embed_bytes(tensor_input)]
            )
            results = renderer.tokenize_prompts(
                prompts,
                TokenizeParams(max_total_tokens=100),
            )

            assert len(results) == 1
            assert results[0]["prompt_embeds"].dtype == dtype

    def test_prompt_embed_squeeze_batch_dim(self):
        renderer = _build_renderer(MockModelConfig())

        # Test tensor with batch dimension gets squeezed
        tensor_input = torch.randn(1, 10, 768, dtype=torch.float32)

        prompts = renderer.render_completions(
            [self._create_test_embed_bytes(tensor_input)]
        )
        results = renderer.tokenize_prompts(
            prompts,
            TokenizeParams(max_total_tokens=100),
        )

        assert len(results) == 1
        # Should be squeezed to 2D
        assert results[0]["prompt_embeds"].shape == (10, 768)

    def test_both_prompts_and_embeds(self):
        renderer = _build_renderer(MockModelConfig())

        text_input = "Hello world"
        tensor_input = torch.randn(5, 256, dtype=torch.float32)

        prompts = renderer.render_completions(
            [text_input, self._create_test_embed_bytes(tensor_input)]
        )
        results = renderer.tokenize_prompts(
            prompts,
            TokenizeParams(max_total_tokens=100),
        )

        assert len(results) == 2
        # First should be tokens prompt
        assert "prompt_token_ids" in results[0]
        assert len(results[0]["prompt_token_ids"]) == len(text_input)
        # Second should be embed prompt
        assert torch.equal(results[1]["prompt_embeds"], tensor_input)
