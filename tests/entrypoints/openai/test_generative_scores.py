# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the Generative Scores API.

These tests verify:
1. Request/Response protocol models
2. Probability computation (apply_softmax=True and apply_softmax=False)
3. Input validation
4. Error handling
"""

import math
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import pytest

from vllm.config.multimodal import MultiModalConfig
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.pooling.score.generative_scores import (
    GenerativeScoreItemResult,
    GenerativeScoreRequest,
    GenerativeScoreResponse,
)
from vllm.entrypoints.pooling.score.generative_scores import (
    OpenAIServingGenerativeScores,
)
from vllm.entrypoints.openai.models.protocol import BaseModelPath
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.logprobs import Logprob
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.tokenizers import get_tokenizer
from vllm.v1.engine.async_llm import AsyncLLM

# Use local model path for testing
MODEL_NAME = "Qwen/Qwen3-0.6B"
MODEL_PATH = "/shared/public/elr-models/Qwen/Qwen3-0.6B/e6de91484c29aa9480d55605af694f39b081c455/"
BASE_MODEL_PATHS = [
    BaseModelPath(name=MODEL_NAME, model_path=MODEL_PATH),
]


@dataclass
class MockHFConfig:
    model_type: str = "any"


@dataclass
class MockModelConfig:
    task = "generate"
    runner_type = "generate"
    tokenizer = MODEL_PATH
    trust_remote_code = False
    tokenizer_mode = "auto"
    max_model_len = 100
    tokenizer_revision = None
    multimodal_config = MultiModalConfig()
    hf_config = MockHFConfig()
    logits_processor_pattern = None
    logits_processors: list[str] | None = None
    diff_sampling_param: dict | None = None
    allowed_local_media_path: str = ""
    allowed_media_domains: list[str] | None = None
    encoder_config = None
    generation_config: str = "auto"
    media_io_kwargs: dict[str, dict[str, Any]] = field(default_factory=dict)
    skip_tokenizer_init = False
    vocab_size = 151936  # Qwen3-0.6B vocab size

    def get_diff_sampling_param(self):
        return self.diff_sampling_param or {}

    def get_vocab_size(self):
        return self.vocab_size


# ============================================================================
# Test Fixtures and Helpers
# ============================================================================


def _create_mock_engine():
    """Create a mock AsyncLLM engine."""
    mock_engine = MagicMock(spec=AsyncLLM)
    mock_engine.get_tokenizer.return_value = get_tokenizer(MODEL_PATH)
    mock_engine.errored = False
    mock_engine.model_config = MockModelConfig()
    mock_engine.input_processor = MagicMock()
    mock_engine.io_processor = MagicMock()
    return mock_engine


def _create_serving(mock_engine) -> OpenAIServingGenerativeScores:
    """Create an OpenAIServingGenerativeScores instance with mocks."""
    models = OpenAIServingModels(
        engine_client=mock_engine,
        base_model_paths=BASE_MODEL_PATHS,
    )
    return OpenAIServingGenerativeScores(
        mock_engine,
        models,
        request_logger=None,
    )


def _create_mock_request_output(
    logprobs_dict: dict[int, float],
    token_id: int = 100,
) -> RequestOutput:
    """Create a mock RequestOutput with specified logprobs."""
    logprobs_with_objs = {
        tid: Logprob(logprob=lp, rank=i + 1)
        for i, (tid, lp) in enumerate(logprobs_dict.items())
    }

    completion_output = CompletionOutput(
        index=0,
        text="",
        token_ids=[token_id],
        cumulative_logprob=-1.0,
        logprobs=[logprobs_with_objs],
        finish_reason="length",
    )

    return RequestOutput(
        request_id="test-request",
        prompt="test prompt",
        prompt_token_ids=[1, 2, 3],
        prompt_logprobs=None,
        outputs=[completion_output],
        finished=True,
    )


# ============================================================================
# Protocol Tests (Parameterized)
# ============================================================================


class TestGenerativeScoreProtocol:
    """Tests for protocol models - parameterized for efficiency."""

    @pytest.mark.parametrize(
        "query,items,label_ids,extra_kwargs,expected_attrs",
        [
            # Basic string input with defaults
            (
                "Is this city the capital?",
                ["Paris", "London"],
                [1234, 5678],
                {},
                {
                    "apply_softmax": True,
                    "item_first": False,
                    "add_special_tokens": True,
                },
            ),
            # Pre-tokenized input
            (
                [100, 200, 300],
                [[400, 500], [600, 700, 800]],
                [1234],
                {},
                {"apply_softmax": True, "item_first": False},
            ),
            # Custom options
            (
                "Test query",
                ["Item1"],
                [100],
                {
                    "apply_softmax": False,
                    "item_first": True,
                    "add_special_tokens": False,
                },
                {
                    "apply_softmax": False,
                    "item_first": True,
                    "add_special_tokens": False,
                },
            ),
        ],
        ids=["basic_defaults", "pretokenized", "custom_options"],
    )
    def test_request_construction(
        self, query, items, label_ids, extra_kwargs, expected_attrs
    ):
        """Test request construction with various inputs and options."""
        request = GenerativeScoreRequest(
            query=query,
            items=items,
            label_token_ids=label_ids,
            **extra_kwargs,
        )
        assert request.query == query
        assert request.items == items
        assert request.label_token_ids == label_ids
        for attr, expected in expected_attrs.items():
            assert getattr(request, attr) == expected, f"{attr} mismatch"

    def test_response_structure(self):
        """Test response model structure."""
        response = GenerativeScoreResponse(
            model="test-model",
            results=[
                GenerativeScoreItemResult(
                    index=0,
                    token_probs={"1234": 0.7, "5678": 0.3},
                )
            ],
            usage={
                "prompt_tokens": 10,
                "total_tokens": 11,
                "completion_tokens": 1,
            },
        )
        assert response.object == "generative_score"
        assert response.model == "test-model"
        assert len(response.results) == 1
        assert response.results[0].token_probs["1234"] == 0.7


# ============================================================================
# Probability Computation Tests (Parameterized - replaces 2 test classes)
# ============================================================================


class TestProbabilityComputation:
    """Unified tests for probability computation - covers both softmax modes."""

    @pytest.mark.parametrize(
        "label_logprobs,apply_softmax,should_sum_to_one",
        [
            # apply_softmax=True cases (subset softmax, sums to 1)
            ({100: -1.0, 200: -2.0}, True, True),
            ({10: -0.5, 20: -1.5}, True, True),
            # Numerical stability with extreme values
            ({100: -100.0, 200: -100.5}, True, True),
            # apply_softmax=False cases (true probs, don't sum to 1)
            ({100: -1.0, 200: -2.0}, False, False),
            ({10: -0.5, 20: -1.5}, False, False),
        ],
        ids=[
            "softmax_basic",
            "softmax_different_values",
            "softmax_numerical_stability",
            "true_probs_basic",
            "true_probs_different_values",
        ],
    )
    def test_compute_probabilities(
        self, label_logprobs, apply_softmax, should_sum_to_one
    ):
        """Test probability computation with various inputs and modes."""
        serving = OpenAIServingGenerativeScores.__new__(
            OpenAIServingGenerativeScores
        )

        probs = serving._compute_probabilities(
            label_logprobs, apply_softmax=apply_softmax
        )

        # Check sum behavior
        total = sum(probs.values())
        if should_sum_to_one:
            assert abs(total - 1.0) < 1e-6, f"Expected sum=1, got {total}"
        else:
            assert total < 1.0, f"True probs should sum <1, got {total}"

        # Verify ordering is preserved (higher logprob = higher prob)
        sorted_logprobs = sorted(
            label_logprobs.items(), key=lambda x: x[1], reverse=True
        )
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        assert [x[0] for x in sorted_logprobs] == [x[0] for x in sorted_probs]

        # Verify math for specific cases
        if apply_softmax:
            # softmax: exp(x_i - max) / sum(exp(x_j - max))
            max_lp = max(label_logprobs.values())
            exp_vals = {k: math.exp(v - max_lp) for k, v in label_logprobs.items()}
            sum_exp = sum(exp_vals.values())
            for token_id, logprob in label_logprobs.items():
                expected = exp_vals[token_id] / sum_exp
                assert abs(probs[token_id] - expected) < 1e-9
        else:
            # true probs: just exp(logprob)
            for token_id, logprob in label_logprobs.items():
                expected = math.exp(logprob)
                assert abs(probs[token_id] - expected) < 1e-9


# ============================================================================
# Validation Tests (Parameterized)
# ============================================================================


class TestValidation:
    """Tests for input validation - parameterized."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "request_kwargs,expected_error_substring",
        [
            # Out of range token ID
            (
                {
                    "query": "test query",
                    "items": ["item1"],
                    "label_token_ids": [999999],
                },
                "out of vocabulary range",
            ),
            # Empty label_token_ids
            (
                {
                    "query": "test query",
                    "items": ["item1"],
                    "label_token_ids": [],
                },
                "at least one token",
            ),
            # Empty items
            (
                {
                    "query": "test query",
                    "items": [],
                    "label_token_ids": [100],
                },
                "at least one item",
            ),
            # Note: mixed_item_types (string and token list) is validated by
            # Pydantic before our code runs, so we test it in e2e tests instead
        ],
        ids=["invalid_token_id", "empty_label_tokens", "empty_items"],
    )
    async def test_validation_errors(self, request_kwargs, expected_error_substring):
        """Test that invalid inputs return appropriate errors."""
        mock_engine = _create_mock_engine()
        serving = _create_serving(mock_engine)

        request = GenerativeScoreRequest(model=MODEL_NAME, **request_kwargs)
        result = await serving.create_generative_score(request, None)

        assert isinstance(result, ErrorResponse)
        assert expected_error_substring in result.error.message.lower()


# ============================================================================
# Integration Tests (with mocked engine)
# ============================================================================


class TestGenerativeScoreGeneration:
    """Tests for the full generation flow with mocked engine."""

    @pytest.mark.asyncio
    async def test_successful_generation(self):
        """Test successful score generation with mocked engine output."""
        mock_engine = _create_mock_engine()
        serving = _create_serving(mock_engine)

        label_token_ids = [1234, 5678]
        mock_logprobs = {
            1234: -0.5,
            5678: -2.0,
            100: -3.0,
            200: -4.0,
        }

        mock_output = _create_mock_request_output(mock_logprobs)

        async def mock_generate(*args, **kwargs):
            yield mock_output

        mock_engine.generate = mock_generate

        request = GenerativeScoreRequest(
            model=MODEL_NAME,
            query="Is Paris the capital of France?",
            items=["Yes", "No"],
            label_token_ids=label_token_ids,
            apply_softmax=True,
        )

        result = await serving.create_generative_score(request, None)

        assert isinstance(result, GenerativeScoreResponse)
        assert len(result.results) == 2

        # Check probabilities are in valid range
        for item_result in result.results:
            for prob in item_result.token_probs.values():
                assert 0.0 <= prob <= 1.0

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "item_first,expected_prompts",
        [
            (
                False,
                [
                    [100, 101, 102, 200, 201],
                    [100, 101, 102, 300, 301],
                ],
            ),
            (
                True,
                [
                    [200, 201, 100, 101, 102],
                    [300, 301, 100, 101, 102],
                ],
            ),
        ],
        ids=["query_first", "item_first"],
    )
    async def test_item_ordering(self, item_first, expected_prompts):
        """Test that item_first flag correctly controls prompt ordering."""
        mock_engine = _create_mock_engine()
        serving = _create_serving(mock_engine)
        tokenizer = MagicMock()

        request = GenerativeScoreRequest(
            query=[100, 101, 102],
            items=[[200, 201], [300, 301]],
            label_token_ids=[500],
            item_first=item_first,
        )

        engine_prompts, _ = await serving._build_prompts(request, tokenizer)

        for i, expected in enumerate(expected_prompts):
            assert engine_prompts[i]["prompt_token_ids"] == expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
