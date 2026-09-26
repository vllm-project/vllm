# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import io
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pybase64
import pytest
import torch

from vllm.config import ModelConfig
from vllm.exceptions import VLLMValidationError
from vllm.inputs import SingletonPrompt
from vllm.renderers import TokenizeParams
from vllm.renderers.hf import HfRenderer
from vllm.renderers.inputs.preprocess import parse_model_prompt, prompt_to_seq

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
    is_multimodal_model: bool = False
    supports_multimodal_inputs: bool = False
    renderer_num_workers: int = 1
    hidden_size: int = 768
    dtype: torch.dtype = torch.float32

    def get_hidden_size(self) -> int:
        return self.hidden_size


@dataclass
class MockParallelConfig:
    _api_process_rank: int = 0


@dataclass
class MockVllmConfig:
    model_config: MockModelConfig
    parallel_config: MockParallelConfig


@dataclass
class DummyTokenizer:
    truncation_side: str = "left"
    max_chars_per_token: int = 1
    # Deliberately outside the range of ids `encode` returns, so a test can
    # tell a pad token apart from a real one.
    pad_token_id: int = 99999

    def __post_init__(self) -> None:
        self._captured_encode_kwargs: dict = {}
        self._captured_text_len: int = 0

    def decode(self, tokens: list[int]):
        return str(tokens)

    def encode(self, text: str, **kwargs):
        self._captured_encode_kwargs = kwargs
        self._captured_text_len = len(text)

        in_length = len(text)
        truncation = kwargs.get("truncation")
        max_length = kwargs.get("max_length")
        if truncation and max_length is not None:
            return list(range(min(in_length, max_length)))

        return list(range(in_length))

    def __call__(self, text: str, **kwargs):
        # BaseRenderer._tokenize_prompt calls the tokenizer via __call__ (to
        # unify the output type), so mirror a real tokenizer's BatchEncoding.
        return {"input_ids": self.encode(text, **kwargs)}


def _build_renderer(
    model_config: MockModelConfig,
    *,
    truncation_side: str = "left",
    max_chars_per_token: int = 1,
):
    renderer = HfRenderer(
        MockVllmConfig(model_config, parallel_config=MockParallelConfig()),
        tokenizer=(
            None
            if model_config.skip_tokenizer_init
            else DummyTokenizer(
                truncation_side=truncation_side,
                max_chars_per_token=max_chars_per_token,
            )
        ),
    )

    return renderer


def _preprocess_prompt(
    model_config: ModelConfig,
    prompt_or_prompts: SingletonPrompt | bytes | Sequence[SingletonPrompt | bytes],
):
    return [
        (
            prompt
            if isinstance(prompt, bytes)
            else parse_model_prompt(model_config, prompt)
        )
        for prompt in prompt_to_seq(prompt_or_prompts)
    ]


class TestValidatePrompt:
    def test_empty_input(self):
        renderer = _build_renderer(MockModelConfig())

        with pytest.raises(ValueError, match="at least one prompt"):
            renderer.render_prompts(_preprocess_prompt(renderer.model_config, []))

    def test_invalid_type(self):
        renderer = _build_renderer(MockModelConfig())

        with pytest.raises(TypeError, match="should be a list of integers"):
            renderer.render_prompts(
                _preprocess_prompt(renderer.model_config, [[1, 2], ["foo", "bar"]])  # type: ignore[arg-type]
            )


class TestRenderPrompt:
    def test_tokens_input(self):
        renderer = _build_renderer(MockModelConfig())

        tokens = [101, 7592, 2088]
        prompts = renderer.render_prompts(
            _preprocess_prompt(renderer.model_config, tokens)
        )
        results = renderer.tokenize_prompts(
            prompts,
            TokenizeParams(max_total_tokens=100),
        )

        assert len(results) == 1
        assert results[0]["prompt_token_ids"] == tokens

    def test_token_list_input(self):
        renderer = _build_renderer(MockModelConfig())

        token_lists = [[101, 7592, 2088], [102, 1234, 5678, 9012], [103, 4567]]
        prompts = renderer.render_prompts(
            _preprocess_prompt(renderer.model_config, token_lists)
        )
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
        prompts = renderer.render_prompts(
            _preprocess_prompt(renderer.model_config, text_input)
        )
        results = renderer.tokenize_prompts(
            prompts,
            TokenizeParams(max_total_tokens=100),
        )

        assert len(results) == 1
        assert len(results[0]["prompt_token_ids"]) == 10

    def test_text_list_input(self):
        renderer = _build_renderer(MockModelConfig())

        text_list_input = ["x" * 10, "x" * 12, "x" * 14]
        prompts = renderer.render_prompts(
            _preprocess_prompt(renderer.model_config, text_list_input)
        )
        results = renderer.tokenize_prompts(
            prompts,
            TokenizeParams(max_total_tokens=100),
        )

        assert len(results) == 3
        for text_input, result in zip(text_list_input, results):
            assert len(result["prompt_token_ids"]) == len(text_input)

    def test_zero_truncation(self):
        renderer = _build_renderer(MockModelConfig())

        prompts = renderer.render_prompts(
            _preprocess_prompt(renderer.model_config, "x" * 200)
        )
        results = renderer.tokenize_prompts(
            prompts,
            TokenizeParams(max_total_tokens=100, truncate_prompt_tokens=0),
        )

        assert len(results) == 1
        assert len(results[0]["prompt_token_ids"]) == 0

    def test_pos_truncation(self):
        renderer = _build_renderer(MockModelConfig())

        prompts = renderer.render_prompts(
            _preprocess_prompt(renderer.model_config, "x" * 200)
        )
        results = renderer.tokenize_prompts(
            prompts,
            TokenizeParams(max_total_tokens=100, truncate_prompt_tokens=50),
        )

        assert len(results) == 1
        assert len(results[0]["prompt_token_ids"]) == 50

    def test_neg_truncation(self):
        renderer = _build_renderer(MockModelConfig())

        prompts = renderer.render_prompts(
            _preprocess_prompt(renderer.model_config, "x" * 200)
        )
        results = renderer.tokenize_prompts(
            prompts,
            TokenizeParams(max_total_tokens=100, truncate_prompt_tokens=-1),
        )

        assert len(results) == 1
        assert len(results[0]["prompt_token_ids"]) == 100  # max_total_tokens

    def test_truncation_left(self):
        renderer = _build_renderer(MockModelConfig(), truncation_side="left")

        long_tokens = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]  # 10 tokens
        prompts = renderer.render_prompts(
            _preprocess_prompt(renderer.model_config, long_tokens)
        )
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
        prompts = renderer.render_prompts(
            _preprocess_prompt(renderer.model_config, long_tokens)
        )
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
        prompts = renderer.render_prompts(
            _preprocess_prompt(renderer.model_config, long_tokens)
        )

        with pytest.raises(
            VLLMValidationError,
            match="maximum context length is",
        ):
            renderer.tokenize_prompts(
                prompts,
                TokenizeParams(max_total_tokens=100),
            )

        # Should not even attempt tokenization
        assert renderer.tokenizer._captured_encode_kwargs == {}

    def test_text_max_length_exceeded_nonobvious(self):
        renderer = _build_renderer(MockModelConfig(), max_chars_per_token=2)

        # Exceeds max_total_tokens but not max_total_tokens * VLLM_MAX_CHARS_PER_TOKEN
        long_tokens = "x" * 150
        prompts = renderer.render_prompts(
            _preprocess_prompt(renderer.model_config, long_tokens)
        )

        with pytest.raises(
            VLLMValidationError,
            match="maximum context length is",
        ):
            renderer.tokenize_prompts(
                prompts,
                TokenizeParams(max_total_tokens=100),
            )

        # Should only tokenize the first max_total_tokens + 1 tokens
        assert renderer.tokenizer._captured_encode_kwargs["truncation"] is True
        assert renderer.tokenizer._captured_encode_kwargs["max_length"] == 101

    def test_token_max_length_exceeded(self):
        renderer = _build_renderer(MockModelConfig())

        long_tokens = list(range(150))  # Exceeds max_total_tokens=100
        prompts = renderer.render_prompts(
            _preprocess_prompt(renderer.model_config, long_tokens)
        )

        with pytest.raises(
            VLLMValidationError,
            match="maximum context length is",
        ):
            renderer.tokenize_prompts(
                prompts,
                TokenizeParams(max_total_tokens=100, truncate_prompt_tokens=None),
            )

    def test_no_tokenizer_for_text(self):
        renderer = _build_renderer(MockModelConfig(skip_tokenizer_init=True))

        prompts = renderer.render_prompts(
            _preprocess_prompt(renderer.model_config, "Hello world")
        )

        with pytest.raises(ValueError, match="`skip_tokenizer_init=True`"):
            renderer.tokenize_prompts(
                prompts,
                TokenizeParams(max_total_tokens=100),
            )

    def test_tokens_input_with_needs_detokenization(self):
        renderer = _build_renderer(MockModelConfig())

        tokens = [1, 2, 3, 4]
        prompts = renderer.render_prompts(
            _preprocess_prompt(renderer.model_config, tokens)
        )
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

    def test_explicit_side_tokenizer_unbounded(self):
        renderer = _build_renderer(MockModelConfig())

        prompts = renderer.render_prompts(
            _preprocess_prompt(renderer.model_config, "x" * 500)
        )
        results = renderer.tokenize_prompts(
            prompts,
            TokenizeParams(
                max_total_tokens=100,
                truncate_prompt_tokens=4,
                truncation_side="left",
            ),
        )

        assert len(results) == 1
        assert len(results[0]["prompt_token_ids"]) == 4

        kwargs = renderer.tokenizer._captured_encode_kwargs
        assert kwargs["truncation"] is False

    def test_explicit_side_left_text(self):
        renderer = _build_renderer(MockModelConfig())

        prompts = renderer.render_prompts(
            _preprocess_prompt(renderer.model_config, "x" * 50)
        )
        results = renderer.tokenize_prompts(
            prompts,
            TokenizeParams(
                max_total_tokens=100,
                truncate_prompt_tokens=5,
                truncation_side="left",
            ),
        )

        assert len(results) == 1
        assert len(results[0]["prompt_token_ids"]) == 5
        assert results[0]["prompt_token_ids"] == list(range(45, 50))

    def test_explicit_side_right_text(self):
        renderer = _build_renderer(MockModelConfig())

        prompts = renderer.render_prompts(
            _preprocess_prompt(renderer.model_config, "x" * 50)
        )
        results = renderer.tokenize_prompts(
            prompts,
            TokenizeParams(
                max_total_tokens=100,
                truncate_prompt_tokens=5,
                truncation_side="right",
            ),
        )

        assert len(results) == 1
        assert len(results[0]["prompt_token_ids"]) == 5
        assert results[0]["prompt_token_ids"] == list(range(5))

    def test_padding_with_left_truncation_keeps_the_prompt(self):
        """Padding must not be truncated away.

        `padding` and `truncate_prompt_tokens`/`truncation_side` are all
        settable on one pooling request. Padding to the full input length
        before truncating from the left leaves a prompt made entirely of pad
        tokens, and the request still succeeds -- so the model embeds nothing
        but padding.
        """
        renderer = _build_renderer(MockModelConfig())
        pad_id = renderer.tokenizer.pad_token_id

        prompts = renderer.render_prompts(
            _preprocess_prompt(renderer.model_config, "x" * 50)
        )
        results = renderer.tokenize_prompts(
            prompts,
            TokenizeParams(
                max_total_tokens=100,
                pad_prompt_tokens=-1,
                truncate_prompt_tokens=5,
                truncation_side="left",
            ),
        )

        assert len(results) == 1
        token_ids = results[0]["prompt_token_ids"]

        # The sentinel: on a padding-first pipeline every surviving id is a
        # pad token, so the prompt reaches the model with no content at all.
        assert set(token_ids) != {pad_id}

        # Transformers semantics: truncate to 5, then pad out to the full
        # input length.
        assert token_ids[:5] == list(range(45, 50))
        assert token_ids[5:] == [pad_id] * 95

    def test_padding_without_truncation_is_unchanged(self):
        renderer = _build_renderer(MockModelConfig())
        pad_id = renderer.tokenizer.pad_token_id

        prompts = renderer.render_prompts(
            _preprocess_prompt(renderer.model_config, "x" * 50)
        )
        results = renderer.tokenize_prompts(
            prompts,
            TokenizeParams(max_total_tokens=100, pad_prompt_tokens=-1),
        )

        assert len(results) == 1
        assert results[0]["prompt_token_ids"] == list(range(50)) + [pad_id] * 50

    def test_explicit_side_text_pretokenization_guard(self):
        renderer = _build_renderer(MockModelConfig(), max_chars_per_token=1)

        prompts = renderer.render_prompts(
            _preprocess_prompt(renderer.model_config, "x" * 500)
        )
        results = renderer.tokenize_prompts(
            prompts,
            TokenizeParams(
                max_total_tokens=100,
                truncate_prompt_tokens=4,
                truncation_side="left",
            ),
        )

        assert len(results) == 1
        assert len(results[0]["prompt_token_ids"]) == 4

        assert renderer.tokenizer._captured_text_len <= 100


class TestMaxTokensNotReservation:
    """Regression tests for GitHub issue #42474.

    ``max_tokens`` must be treated as an upper bound on generation length, not
    as a hard reservation that is subtracted from the available input space.
    """

    def test_large_max_tokens_does_not_eat_prompt_budget(self):
        """Prompt fits in context even when max_tokens is large."""
        renderer = _build_renderer(MockModelConfig())

        # 50-token prompt with a 100-token context window and max_tokens=60.
        # Before the fix: max_input_tokens = 100 - 60 = 40 → 50 > 40 → error.
        # After the fix:  max_input_tokens = 100            → 50 ≤ 100 → OK.
        prompts = renderer.render_prompts(
            _preprocess_prompt(renderer.model_config, "x" * 50)
        )
        results = renderer.tokenize_prompts(
            prompts,
            TokenizeParams(max_total_tokens=100, max_output_tokens=60),
        )
        assert len(results) == 1
        assert len(results[0]["prompt_token_ids"]) == 50

    def test_max_tokens_zero_still_rejects_oversized_prompt(self):
        """Genuine prompt overflow is still caught regardless of max_tokens."""
        renderer = _build_renderer(MockModelConfig())

        # 150-token prompt with 100-token context → should still fail.
        prompts = renderer.render_prompts(
            _preprocess_prompt(renderer.model_config, "x" * 150)
        )
        with pytest.raises(VLLMValidationError, match="maximum context length"):
            renderer.tokenize_prompts(
                prompts,
                TokenizeParams(max_total_tokens=100, max_output_tokens=0),
            )

    def test_negative_truncation_uses_max_total(self):
        """truncate_prompt_tokens=-1 pads to max_total_tokens, not max_input."""
        renderer = _build_renderer(MockModelConfig())

        prompts = renderer.render_prompts(
            _preprocess_prompt(renderer.model_config, "x" * 200)
        )
        results = renderer.tokenize_prompts(
            prompts,
            TokenizeParams(
                max_total_tokens=100,
                max_output_tokens=40,
                truncate_prompt_tokens=-1,
            ),
        )
        # -1 should map to max_total_tokens (100), not max_total - max_output (60).
        assert len(results[0]["prompt_token_ids"]) == 100

    def test_negative_pad_uses_max_total(self):
        """pad_prompt_tokens=-1 pads to max_total_tokens, not max_input."""
        renderer = _build_renderer(MockModelConfig())
        pad_id = renderer.tokenizer.pad_token_id

        prompts = renderer.render_prompts(
            _preprocess_prompt(renderer.model_config, "x" * 30)
        )
        results = renderer.tokenize_prompts(
            prompts,
            TokenizeParams(
                max_total_tokens=100,
                max_output_tokens=40,
                pad_prompt_tokens=-1,
            ),
        )
        # -1 should map to max_total_tokens (100), not max_total - max_output (60).
        assert results[0]["prompt_token_ids"] == list(range(30)) + [pad_id] * 70

    def test_max_tokens_exceeds_context_is_still_clamped_by_api(self):
        """Renderer allows the prompt; API layer clamps max_tokens later."""
        renderer = _build_renderer(MockModelConfig())

        # Simulates the reported scenario: max_tokens > context - prompt_length.
        # Renderer should accept the 40-token prompt; get_max_tokens() clamps
        # sampling max_tokens to 60 at the entrypoint layer.
        prompts = renderer.render_prompts(
            _preprocess_prompt(renderer.model_config, "x" * 40)
        )
        results = renderer.tokenize_prompts(
            prompts,
            TokenizeParams(max_total_tokens=100, max_output_tokens=80),
        )
        assert len(results) == 1
        assert len(results[0]["prompt_token_ids"]) == 40

    def test_with_kwargs_preserves_output_budget(self):
        """with_kwargs without an explicit max_length keeps the output budget.

        Under the old semantics `with_kwargs()` recomputed
        `max_output_tokens = max_total - max_input`, which (after this fix)
        would have silently reset a real output budget to zero.
        """
        tok_params = TokenizeParams(max_total_tokens=100, max_output_tokens=60)
        chained = tok_params.with_kwargs(truncation_side="left")

        assert chained.max_output_tokens == 60
        assert chained.max_input_tokens == 100

    def test_with_kwargs_explicit_max_length_sets_budget(self):
        """An explicit max_length still re-derives the output budget."""
        tok_params = TokenizeParams(max_total_tokens=100, max_output_tokens=60)
        chained = tok_params.with_kwargs(max_length=80)

        assert chained.max_output_tokens == 20
        assert chained.max_input_tokens == 100


class TestRenderEmbedPrompt:
    def _create_test_embed_bytes(self, tensor: torch.Tensor) -> bytes:
        """Helper to create base64-encoded tensor bytes."""
        buffer = io.BytesIO()
        torch.save(tensor, buffer)
        buffer.seek(0)
        return pybase64.b64encode(buffer.read())

    def test_single_prompt_embed(self):
        renderer = _build_renderer(MockModelConfig())

        # Create a test tensor
        tensor_input = torch.randn(10, 768, dtype=torch.float32)
        embed_bytes = self._create_test_embed_bytes(tensor_input)

        prompts = renderer.render_prompts(
            _preprocess_prompt(renderer.model_config, embed_bytes)
        )
        results = renderer.tokenize_prompts(
            prompts,
            TokenizeParams(max_total_tokens=100),
        )

        assert len(results) == 1
        assert torch.equal(results[0]["prompt_embeds"], tensor_input)

    def test_multiple_prompt_embeds(self):
        hidden_size = 512
        renderer = _build_renderer(MockModelConfig(hidden_size=hidden_size))

        # Create multiple test tensors
        tensor_inputs = [
            torch.randn(8, hidden_size, dtype=torch.float32),
            torch.randn(12, hidden_size, dtype=torch.float32),
        ]

        prompts = renderer.render_prompts(
            _preprocess_prompt(
                renderer.model_config,
                [self._create_test_embed_bytes(t) for t in tensor_inputs],
            )
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

        prompts = renderer.render_prompts(
            _preprocess_prompt(
                renderer.model_config, self._create_test_embed_bytes(tensor_input)
            )
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
        hidden_size = 256
        # Test different supported dtypes
        dtypes = [torch.float32, torch.float16, torch.bfloat16]

        for dtype in dtypes:
            renderer = _build_renderer(
                MockModelConfig(hidden_size=hidden_size, dtype=dtype)
            )
            tensor_input = torch.randn(5, hidden_size, dtype=dtype)

            prompts = renderer.render_prompts(
                _preprocess_prompt(
                    renderer.model_config, self._create_test_embed_bytes(tensor_input)
                )
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

        prompts = renderer.render_prompts(
            _preprocess_prompt(
                renderer.model_config, self._create_test_embed_bytes(tensor_input)
            )
        )
        results = renderer.tokenize_prompts(
            prompts,
            TokenizeParams(max_total_tokens=100),
        )

        assert len(results) == 1
        # Should be squeezed to 2D
        assert results[0]["prompt_embeds"].shape == (10, 768)

    def test_both_prompts_and_embeds(self):
        hidden_size = 256
        renderer = _build_renderer(MockModelConfig(hidden_size=hidden_size))

        text_input = "Hello world"
        tensor_input = torch.randn(5, hidden_size, dtype=torch.float32)

        prompts = renderer.render_prompts(
            _preprocess_prompt(
                renderer.model_config,
                [text_input, self._create_test_embed_bytes(tensor_input)],
            )
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
