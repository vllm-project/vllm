# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the Generative Scores API.

Tests cover:
1. Protocol models (request/response construction)
2. Probability computation (softmax normalization)
3. Input validation
4. Score formula: P(token[0]) / (P(token[0]) + P(token[1]))
5. Prompt building and item ordering
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
    OpenAIServingGenerativeScores,
)
from vllm.entrypoints.openai.models.protocol import BaseModelPath
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.logprobs import Logprob
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.tokenizers import get_tokenizer
from vllm.v1.engine.async_llm import AsyncLLM

MODEL_NAME = "Qwen/Qwen3-0.6B"
MODEL_PATH = "/shared/public/elr-models/Qwen/Qwen3-0.6B/e6de91484c29aa9480d55605af694f39b081c455/"
BASE_MODEL_PATHS = [BaseModelPath(name=MODEL_NAME, model_path=MODEL_PATH)]


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
    vocab_size = 151936

    def get_diff_sampling_param(self):
        return self.diff_sampling_param or {}

    def get_vocab_size(self):
        return self.vocab_size


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
    return OpenAIServingGenerativeScores(mock_engine, models, request_logger=None)


def _create_mock_request_output(logprobs_dict: dict[int, float]) -> RequestOutput:
    """Create a mock RequestOutput with specified logprobs."""
    logprobs_with_objs = {
        tid: Logprob(logprob=lp, rank=i + 1)
        for i, (tid, lp) in enumerate(logprobs_dict.items())
    }
    completion_output = CompletionOutput(
        index=0,
        text="",
        token_ids=[100],
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


class TestProtocolModels:
    """Tests for GenerativeScoreRequest and GenerativeScoreResponse."""

    def test_request_and_response_all_fields(self):
        """Test request construction with all field types and response structure."""
        # Test request with string inputs
        req_str = GenerativeScoreRequest(
            query="Is this the capital?",
            items=["Paris", "London"],
            label_token_ids=[9454, 2753],
        )
        assert req_str.query == "Is this the capital?"
        assert req_str.items == ["Paris", "London"]
        assert req_str.label_token_ids == [9454, 2753]
        assert req_str.apply_softmax is True  # default
        assert req_str.item_first is False  # default
        assert req_str.add_special_tokens is True  # default

        # Test request with pre-tokenized inputs and custom options
        req_tok = GenerativeScoreRequest(
            query=[100, 200, 300],
            items=[[400, 500], [600, 700]],
            label_token_ids=[1234, 5678],
            apply_softmax=False,
            item_first=True,
            add_special_tokens=False,
        )
        assert req_tok.query == [100, 200, 300]
        assert req_tok.items == [[400, 500], [600, 700]]
        assert req_tok.apply_softmax is False
        assert req_tok.item_first is True
        assert req_tok.add_special_tokens is False

        # Test response structure
        response = GenerativeScoreResponse(
            model="test-model",
            results=[
                GenerativeScoreItemResult(index=0, token_probs={"9454": 0.7, "2753": 0.3}),
                GenerativeScoreItemResult(index=1, token_probs={"9454": 0.4, "2753": 0.6}),
            ],
            usage={"prompt_tokens": 10, "total_tokens": 12, "completion_tokens": 2},
        )
        assert response.object == "generative_score"
        assert response.model == "test-model"
        assert len(response.results) == 2
        assert response.results[0].token_probs["9454"] == 0.7
        assert response.results[1].token_probs["2753"] == 0.6
        assert response.usage.prompt_tokens == 10


class TestProbabilityComputation:
    """Tests for _compute_probabilities with both softmax modes."""

    @pytest.mark.parametrize(
        "label_logprobs,apply_softmax,should_sum_to_one",
        [
            ({100: -1.0, 200: -2.0}, True, True),
            ({100: -100.0, 200: -100.5}, True, True),  # numerical stability
            ({100: -1.0, 200: -2.0}, False, False),
        ],
        ids=["softmax_basic", "softmax_extreme_values", "true_probs"],
    )
    def test_compute_probabilities(self, label_logprobs, apply_softmax, should_sum_to_one):
        """Test probability computation for softmax and true probability modes."""
        serving = OpenAIServingGenerativeScores.__new__(OpenAIServingGenerativeScores)
        probs = serving._compute_probabilities(label_logprobs, apply_softmax=apply_softmax)

        # Verify sum behavior
        total = sum(probs.values())
        if should_sum_to_one:
            assert abs(total - 1.0) < 1e-6
        else:
            assert total < 1.0

        # Verify math
        if apply_softmax:
            max_lp = max(label_logprobs.values())
            exp_vals = {k: math.exp(v - max_lp) for k, v in label_logprobs.items()}
            sum_exp = sum(exp_vals.values())
            for tid, lp in label_logprobs.items():
                assert abs(probs[tid] - exp_vals[tid] / sum_exp) < 1e-9
        else:
            for tid, lp in label_logprobs.items():
                assert abs(probs[tid] - math.exp(lp)) < 1e-9

    def test_score_formula(self):
        """Test the score formula: P(token[0]) / (P(token[0]) + P(token[1]))."""
        serving = OpenAIServingGenerativeScores.__new__(OpenAIServingGenerativeScores)
        
        # With logprobs -0.5 and -2.0, softmax gives higher prob to first token
        logprobs = {9454: -0.5, 2753: -2.0}
        probs = serving._compute_probabilities(logprobs, apply_softmax=True)
        
        # Score = P(9454) / (P(9454) + P(2753)) = P(9454) since they sum to 1
        score = probs[9454]
        
        # Manual calculation
        exp_0 = math.exp(-0.5)
        exp_1 = math.exp(-2.0)
        expected_score = exp_0 / (exp_0 + exp_1)
        
        assert abs(score - expected_score) < 1e-9
        assert score > 0.5  # First token has higher logprob, so higher probability


class TestValidation:
    """Tests for input validation errors."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "request_kwargs,expected_error",
        [
            ({"query": "q", "items": ["i"], "label_token_ids": [999999, 999998]}, "out of vocabulary"),
            ({"query": "q", "items": ["i"], "label_token_ids": [100]}, "at least one token"),
            ({"query": "q", "items": [], "label_token_ids": [100, 200]}, "at least one item"),
        ],
        ids=["invalid_token_id", "single_token", "empty_items"],
    )
    async def test_validation_errors(self, request_kwargs, expected_error):
        """Test that invalid inputs return appropriate errors."""
        mock_engine = _create_mock_engine()
        serving = _create_serving(mock_engine)
        request = GenerativeScoreRequest(model=MODEL_NAME, **request_kwargs)
        result = await serving.create_generative_score(request, None)

        assert isinstance(result, ErrorResponse)
        assert expected_error in result.error.message.lower()


class TestPromptBuilding:
    """Tests for prompt construction and item ordering."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "item_first,expected",
        [
            (False, [[100, 101, 200, 201], [100, 101, 300, 301]]),  # query + item
            (True, [[200, 201, 100, 101], [300, 301, 100, 101]]),   # item + query
        ],
        ids=["query_first", "item_first"],
    )
    async def test_item_ordering(self, item_first, expected):
        """Test that item_first flag controls prompt concatenation order."""
        mock_engine = _create_mock_engine()
        serving = _create_serving(mock_engine)

        request = GenerativeScoreRequest(
            query=[100, 101],
            items=[[200, 201], [300, 301]],
            label_token_ids=[500, 501],
            item_first=item_first,
        )
        engine_prompts, _ = await serving._build_prompts(request, MagicMock())

        for i, exp in enumerate(expected):
            assert engine_prompts[i]["prompt_token_ids"] == exp


class TestGeneration:
    """Tests for the full generation flow with mocked engine."""

    @pytest.mark.asyncio
    async def test_successful_generation(self):
        """Test successful score generation returns valid response."""
        mock_engine = _create_mock_engine()
        serving = _create_serving(mock_engine)

        mock_logprobs = {1234: -0.5, 5678: -2.0, 100: -3.0}
        mock_output = _create_mock_request_output(mock_logprobs)

        async def mock_generate(*args, **kwargs):
            yield mock_output

        mock_engine.generate = mock_generate

        request = GenerativeScoreRequest(
            model=MODEL_NAME,
            query="Is Paris the capital?",
            items=["Yes", "No"],
            label_token_ids=[1234, 5678],
        )
        result = await serving.create_generative_score(request, None)

        assert isinstance(result, GenerativeScoreResponse)
        assert len(result.results) == 2
        for item_result in result.results:
            for prob in item_result.token_probs.values():
                assert 0.0 <= prob <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
