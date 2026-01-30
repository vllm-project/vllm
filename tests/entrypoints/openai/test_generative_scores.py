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
from unittest.mock import AsyncMock, MagicMock

import pytest

from vllm.config.multimodal import MultiModalConfig
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.openai.generative_scores.protocol import (
    GenerativeScoreItemResult,
    GenerativeScoreRequest,
    GenerativeScoreResponse,
)
from vllm.entrypoints.openai.generative_scores.serving import (
    OpenAIServingGenerativeScores,
)
from vllm.entrypoints.openai.models.protocol import BaseModelPath
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.logprobs import Logprob
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.tokenizers import get_tokenizer
from vllm.v1.engine.async_llm import AsyncLLM

MODEL_NAME = "openai-community/gpt2"
BASE_MODEL_PATHS = [
    BaseModelPath(name=MODEL_NAME, model_path=MODEL_NAME),
]


@dataclass
class MockHFConfig:
    model_type: str = "any"


@dataclass
class MockModelConfig:
    task = "generate"
    runner_type = "generate"
    tokenizer = MODEL_NAME
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
    vocab_size = 50257  # GPT-2 vocab size

    def get_diff_sampling_param(self):
        return self.diff_sampling_param or {}

    def get_vocab_size(self):
        return self.vocab_size


# ============================================================================
# Protocol Tests
# ============================================================================


class TestGenerativeScoreProtocol:
    """Tests for protocol models."""

    def test_request_basic_fields(self):
        """Test request with basic required fields."""
        request = GenerativeScoreRequest(
            query="Is this city the capital?",
            items=["Paris", "London"],
            label_token_ids=[1234, 5678],
        )
        assert request.query == "Is this city the capital?"
        assert request.items == ["Paris", "London"]
        assert request.label_token_ids == [1234, 5678]
        assert request.apply_softmax is True  # default
        assert request.item_first is False  # default
        assert request.temperature == 0.0  # default for scoring
        assert request.top_k == 0  # default (disabled)
        assert request.top_p == 1.0  # default (disabled)

    def test_request_with_pretokenized_input(self):
        """Test request with pre-tokenized token IDs."""
        request = GenerativeScoreRequest(
            query=[100, 200, 300],
            items=[[400, 500], [600, 700, 800]],
            label_token_ids=[1234, 5678],
        )
        assert request.query == [100, 200, 300]
        assert request.items == [[400, 500], [600, 700, 800]]

    def test_request_custom_options(self):
        """Test request with custom options."""
        request = GenerativeScoreRequest(
            query="Test query",
            items=["Item1"],
            label_token_ids=[100],
            apply_softmax=False,
            item_first=True,
            temperature=0.5,
            add_special_tokens=False,
        )
        assert request.apply_softmax is False
        assert request.item_first is True
        assert request.temperature == 0.5
        assert request.add_special_tokens is False

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
            usage={"prompt_tokens": 10, "total_tokens": 11, "completion_tokens": 1},
        )
        assert response.object == "generative_score"
        assert response.model == "test-model"
        assert len(response.results) == 1
        assert response.results[0].token_probs["1234"] == 0.7


# ============================================================================
# Probability Computation Tests
# ============================================================================


class TestProbabilityComputation:
    """Tests for probability computation logic."""

    def test_compute_probabilities_with_softmax(self):
        """Test subset softmax normalization (apply_softmax=True).

        When apply_softmax=True, we normalize only over the label tokens.
        softmax([logprob_a, logprob_b]) should sum to 1.
        """
        serving = OpenAIServingGenerativeScores.__new__(
            OpenAIServingGenerativeScores
        )

        # Example logprobs (log probabilities from the model)
        # These are already log(softmax(logits)) values
        label_logprobs = {
            100: -1.0,  # ~0.368 before normalization
            200: -2.0,  # ~0.135 before normalization
        }

        probs = serving._compute_probabilities(label_logprobs, apply_softmax=True)

        # With subset softmax, probs should sum to 1
        total = sum(probs.values())
        assert abs(total - 1.0) < 1e-6, f"Probabilities should sum to 1, got {total}"

        # Check relative ordering is preserved
        assert probs[100] > probs[200], "Higher logprob should have higher probability"

        # Verify the math: softmax([−1, −2]) = [e^−1/(e^−1+e^−2), e^−2/(e^−1+e^−2)]
        exp_neg1 = math.exp(-1)
        exp_neg2 = math.exp(-2)
        expected_prob_100 = exp_neg1 / (exp_neg1 + exp_neg2)
        expected_prob_200 = exp_neg2 / (exp_neg1 + exp_neg2)
        assert abs(probs[100] - expected_prob_100) < 1e-6
        assert abs(probs[200] - expected_prob_200) < 1e-6

    def test_compute_probabilities_without_softmax(self):
        """Test true model probabilities (apply_softmax=False).

        When apply_softmax=False, we return exp(logprob) which gives the
        true model probability for each token over the full vocab.
        """
        serving = OpenAIServingGenerativeScores.__new__(
            OpenAIServingGenerativeScores
        )

        # Example logprobs (already normalized over full vocab by the model)
        label_logprobs = {
            100: -1.0,  # exp(-1) ≈ 0.368
            200: -2.0,  # exp(-2) ≈ 0.135
        }

        probs = serving._compute_probabilities(label_logprobs, apply_softmax=False)

        # These should NOT sum to 1 (they're just exp of the logprobs)
        expected_prob_100 = math.exp(-1.0)
        expected_prob_200 = math.exp(-2.0)

        assert abs(probs[100] - expected_prob_100) < 1e-6
        assert abs(probs[200] - expected_prob_200) < 1e-6

        # These probabilities don't sum to 1 (unless we happened to pick
        # the only tokens with probability mass)
        total = sum(probs.values())
        assert total < 1.0, "True probs over subset shouldn't sum to 1"

    def test_compute_probabilities_numerical_stability(self):
        """Test that computation is numerically stable with extreme values."""
        serving = OpenAIServingGenerativeScores.__new__(
            OpenAIServingGenerativeScores
        )

        # Very negative logprobs (very unlikely tokens)
        label_logprobs = {
            100: -100.0,
            200: -100.5,
        }

        # Should not overflow/underflow with subset softmax
        probs = serving._compute_probabilities(label_logprobs, apply_softmax=True)
        total = sum(probs.values())
        assert abs(total - 1.0) < 1e-6
        assert probs[100] > probs[200]


# ============================================================================
# Mock Engine Tests
# ============================================================================


def _create_mock_engine():
    """Create a mock AsyncLLM engine."""
    mock_engine = MagicMock(spec=AsyncLLM)
    mock_engine.get_tokenizer.return_value = get_tokenizer(MODEL_NAME)
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
    # Convert to Logprob objects
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
        outputs=[completion_output],
        finished=True,
    )


# ============================================================================
# Validation Tests
# ============================================================================


class TestValidation:
    """Tests for input validation."""

    @pytest.mark.asyncio
    async def test_invalid_token_id_out_of_range(self):
        """Test that out-of-range token IDs return an error."""
        mock_engine = _create_mock_engine()
        serving = _create_serving(mock_engine)

        request = GenerativeScoreRequest(
            model=MODEL_NAME,
            query="test query",
            items=["item1"],
            label_token_ids=[999999],  # Way beyond vocab size
        )

        result = await serving.create_generative_score(request, None)

        assert isinstance(result, ErrorResponse)
        assert "out of vocabulary range" in result.error.message.lower()

    @pytest.mark.asyncio
    async def test_empty_label_token_ids(self):
        """Test that empty label_token_ids returns an error."""
        mock_engine = _create_mock_engine()
        serving = _create_serving(mock_engine)

        request = GenerativeScoreRequest(
            model=MODEL_NAME,
            query="test query",
            items=["item1"],
            label_token_ids=[],  # Empty
        )

        result = await serving.create_generative_score(request, None)

        assert isinstance(result, ErrorResponse)
        assert "at least one token" in result.error.message.lower()

    @pytest.mark.asyncio
    async def test_empty_items(self):
        """Test that empty items list returns an error."""
        mock_engine = _create_mock_engine()
        serving = _create_serving(mock_engine)

        request = GenerativeScoreRequest(
            model=MODEL_NAME,
            query="test query",
            items=[],  # Empty
            label_token_ids=[100],
        )

        result = await serving.create_generative_score(request, None)

        assert isinstance(result, ErrorResponse)
        assert "at least one item" in result.error.message.lower()


# ============================================================================
# Integration-style Tests (with mocked engine)
# ============================================================================


class TestGenerativeScoreGeneration:
    """Tests for the full generation flow with mocked engine."""

    @pytest.mark.asyncio
    async def test_successful_generation(self):
        """Test successful score generation with mocked engine output."""
        mock_engine = _create_mock_engine()
        serving = _create_serving(mock_engine)

        # Set up the mock to return logprobs for our label tokens
        label_token_ids = [1234, 5678]
        mock_logprobs = {
            1234: -0.5,  # Higher probability
            5678: -2.0,  # Lower probability
            # Include some other tokens that would be in full vocab
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

        # Should succeed
        assert isinstance(result, GenerativeScoreResponse)
        assert len(result.results) == 2  # One per item

        # Check probabilities are in valid range
        for item_result in result.results:
            for prob in item_result.token_probs.values():
                assert 0.0 <= prob <= 1.0

    @pytest.mark.asyncio
    async def test_item_first_ordering(self):
        """Test that item_first=True prepends item to query."""
        mock_engine = _create_mock_engine()
        serving = _create_serving(mock_engine)

        # Track what prompts are built
        built_prompts = []

        async def mock_tokenizer_call(text, **kwargs):
            result = MagicMock()
            result.input_ids = [ord(c) for c in text[:5]]  # Simple mock
            return result

        # We can verify the prompt building by checking the _build_prompts method
        tokenizer = MagicMock()
        tokenizer.return_value = MagicMock(input_ids=[1, 2, 3])

        request = GenerativeScoreRequest(
            query=[100, 101, 102],  # Pre-tokenized
            items=[[200, 201], [300, 301]],  # Pre-tokenized
            label_token_ids=[500],
            item_first=False,
        )

        # Build prompts and check ordering
        engine_prompts, _ = await serving._build_prompts(request, tokenizer)

        # With item_first=False: query + item
        assert engine_prompts[0]["prompt_token_ids"] == [100, 101, 102, 200, 201]
        assert engine_prompts[1]["prompt_token_ids"] == [100, 101, 102, 300, 301]

        # Now test with item_first=True
        request.item_first = True
        engine_prompts, _ = await serving._build_prompts(request, tokenizer)

        # With item_first=True: item + query
        assert engine_prompts[0]["prompt_token_ids"] == [200, 201, 100, 101, 102]
        assert engine_prompts[1]["prompt_token_ids"] == [300, 301, 100, 101, 102]


# ============================================================================
# Math Verification Tests
# ============================================================================


class TestMathVerification:
    """Detailed tests to verify the probability math is correct."""

    def test_softmax_over_subset(self):
        """Verify: apply_softmax=True gives softmax over subset."""
        serving = OpenAIServingGenerativeScores.__new__(
            OpenAIServingGenerativeScores
        )

        # logits (before log) for full vocab might be [1.0, 2.0, 3.0, ...]
        # After softmax over full vocab, logprobs become log(softmax(logits))
        # For our test, let's say:
        #   - Token A has logprob -0.5 (exp(-0.5) ≈ 0.606 true prob)
        #   - Token B has logprob -1.5 (exp(-1.5) ≈ 0.223 true prob)

        label_logprobs = {10: -0.5, 20: -1.5}

        # With apply_softmax=True, we do softmax over just [10, 20]
        # softmax([-0.5, -1.5]) = [exp(-0.5)/(exp(-0.5)+exp(-1.5)),
        #                          exp(-1.5)/(exp(-0.5)+exp(-1.5))]
        probs = serving._compute_probabilities(label_logprobs, apply_softmax=True)

        exp_a = math.exp(-0.5)
        exp_b = math.exp(-1.5)
        denom = exp_a + exp_b

        expected_a = exp_a / denom
        expected_b = exp_b / denom

        assert abs(probs[10] - expected_a) < 1e-9
        assert abs(probs[20] - expected_b) < 1e-9
        assert abs(sum(probs.values()) - 1.0) < 1e-9

    def test_true_probs_without_softmax(self):
        """Verify: apply_softmax=False gives exp(logprob) = true model prob."""
        serving = OpenAIServingGenerativeScores.__new__(
            OpenAIServingGenerativeScores
        )

        # These logprobs come from log(softmax(logits)) computed by the model
        # So exp(logprob) gives the true probability
        label_logprobs = {10: -0.5, 20: -1.5}

        probs = serving._compute_probabilities(label_logprobs, apply_softmax=False)

        # Just exp of the logprobs
        expected_a = math.exp(-0.5)
        expected_b = math.exp(-1.5)

        assert abs(probs[10] - expected_a) < 1e-9
        assert abs(probs[20] - expected_b) < 1e-9

        # They don't sum to 1 (unless we selected all tokens)
        assert sum(probs.values()) < 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
