# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from vllm.config.multimodal import MultiModalConfig
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.models.protocol import BaseModelPath
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.renderers.hf import HfRenderer
from vllm.tokenizers.registry import tokenizer_args_from_config
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.metrics.stats import RequestStateStats

MODEL_NAME = "openai-community/gpt2"
MODEL_NAME_SHORT = "gpt2"
BASE_MODEL_PATHS = [
    BaseModelPath(name=MODEL_NAME, model_path=MODEL_NAME),
    BaseModelPath(name=MODEL_NAME_SHORT, model_path=MODEL_NAME_SHORT),
]


@dataclass
class MockHFConfig:
    model_type: str = "any"


@dataclass
class MockModelConfig:
    task = "generate"
    runner_type = "generate"
    model = MODEL_NAME
    tokenizer = MODEL_NAME
    trust_remote_code = False
    tokenizer_mode = "auto"
    max_model_len = 100
    tokenizer_revision = None
    multimodal_config = MultiModalConfig()
    hf_config = MockHFConfig()
    hf_text_config = MockHFConfig()
    logits_processors: list[str] | None = None
    diff_sampling_param: dict | None = None
    allowed_local_media_path: str = ""
    allowed_media_domains: list[str] | None = None
    encoder_config = None
    generation_config: str = "auto"
    media_io_kwargs: dict[str, dict[str, Any]] = field(default_factory=dict)
    skip_tokenizer_init = False
    is_encoder_decoder: bool = False
    is_multimodal_model: bool = False

    def get_diff_sampling_param(self):
        return self.diff_sampling_param or {}


@dataclass
class MockParallelConfig:
    _api_process_rank: int = 0


@dataclass
class MockVllmConfig:
    model_config: MockModelConfig
    parallel_config: MockParallelConfig


def _build_renderer(model_config: MockModelConfig):
    _, tokenizer_name, _, kwargs = tokenizer_args_from_config(model_config)
    return HfRenderer.from_config(
        MockVllmConfig(model_config, parallel_config=MockParallelConfig()),
        tokenizer_kwargs={**kwargs, "tokenizer_name": tokenizer_name},
    )


def _build_serving_chat(
    engine: AsyncLLM,
    enable_per_request_metrics: bool = False,
) -> OpenAIServingChat:
    models = OpenAIServingModels(
        engine_client=engine,
        base_model_paths=BASE_MODEL_PATHS,
    )
    serving_chat = OpenAIServingChat(
        engine,
        models,
        response_role="assistant",
        request_logger=None,
        chat_template=None,
        chat_template_content_format="auto",
        enable_per_request_metrics=enable_per_request_metrics,
    )

    async def _fake_preprocess_chat(*args, **kwargs):
        return (
            [{"role": "user", "content": "Test"}],
            [{"prompt_token_ids": [1, 2, 3]}],
        )

    serving_chat._preprocess_chat = AsyncMock(side_effect=_fake_preprocess_chat)
    return serving_chat


def _make_request_output(
    metrics: RequestStateStats | None = None,
    text: str = "Hello",
    token_ids: tuple = (100, 101),
) -> RequestOutput:
    completion_output = CompletionOutput(
        index=0,
        text=text,
        token_ids=list(token_ids),
        cumulative_logprob=None,
        logprobs=None,
        finish_reason="stop",
    )
    return RequestOutput(
        request_id="test-id",
        prompt="Test prompt",
        prompt_token_ids=[1, 2, 3],
        prompt_logprobs=None,
        outputs=[completion_output],
        finished=True,
        metrics=metrics,
        lora_request=None,
        encoder_prompt=None,
        encoder_prompt_token_ids=None,
    )


def _make_mock_engine(metrics: RequestStateStats | None = None) -> MagicMock:
    mock_engine = MagicMock(spec=AsyncLLM)
    mock_engine.errored = False
    mock_engine.model_config = MockModelConfig()
    mock_engine.input_processor = MagicMock()
    mock_engine.io_processor = MagicMock()
    mock_engine.renderer = _build_renderer(mock_engine.model_config)

    request_output = _make_request_output(metrics=metrics)

    async def mock_generate(*args, **kwargs):
        yield request_output

    mock_engine.generate = MagicMock(side_effect=mock_generate)
    return mock_engine


# ---------------------------------------------------------------------------
# enable_per_request_metrics=False (default): metrics never populated
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_metrics_disabled_include_metrics_true_returns_none():
    stats = RequestStateStats(
        queued_ts=1.0,
        scheduled_ts=1.5,
        first_token_ts=2.0,
        last_token_ts=3.0,
        num_generation_tokens=2,
    )
    mock_engine = _make_mock_engine(metrics=stats)
    serving_chat = _build_serving_chat(mock_engine, enable_per_request_metrics=False)

    request = ChatCompletionRequest(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "Test prompt"}],
        max_tokens=10,
        stream=False,
        include_metrics=True,
    )

    response = await serving_chat.create_chat_completion(request)
    assert response.metrics is None


@pytest.mark.asyncio
async def test_metrics_disabled_include_metrics_false_returns_none():
    mock_engine = _make_mock_engine()
    serving_chat = _build_serving_chat(mock_engine, enable_per_request_metrics=False)

    request = ChatCompletionRequest(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "Test prompt"}],
        max_tokens=10,
        stream=False,
        include_metrics=False,
    )

    response = await serving_chat.create_chat_completion(request)
    assert response.metrics is None


# ---------------------------------------------------------------------------
# enable_per_request_metrics=True: gated by request.include_metrics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_metrics_enabled_include_metrics_false_returns_none():
    stats = RequestStateStats(
        queued_ts=1.0,
        scheduled_ts=1.5,
        first_token_ts=2.0,
        last_token_ts=3.0,
        num_generation_tokens=2,
    )
    mock_engine = _make_mock_engine(metrics=stats)
    serving_chat = _build_serving_chat(mock_engine, enable_per_request_metrics=True)

    request = ChatCompletionRequest(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "Test prompt"}],
        max_tokens=10,
        stream=False,
        include_metrics=False,
    )

    response = await serving_chat.create_chat_completion(request)
    assert response.metrics is None


@pytest.mark.asyncio
async def test_metrics_enabled_include_metrics_true_returns_populated():
    stats = RequestStateStats(
        queued_ts=1.0,
        scheduled_ts=1.5,
        first_token_ts=2.0,
        last_token_ts=3.0,
        num_generation_tokens=2,
    )
    mock_engine = _make_mock_engine(metrics=stats)
    serving_chat = _build_serving_chat(mock_engine, enable_per_request_metrics=True)

    request = ChatCompletionRequest(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "Test prompt"}],
        max_tokens=10,
        stream=False,
        include_metrics=True,
    )

    response = await serving_chat.create_chat_completion(request)
    assert response.metrics is not None
    assert response.metrics.time_to_first_token_ms == pytest.approx(500.0)
    assert response.metrics.generation_time_ms == pytest.approx(1500.0)
    assert response.metrics.queue_time_ms == pytest.approx(500.0)


@pytest.mark.asyncio
async def test_metrics_enabled_no_request_output_metrics():
    mock_engine = _make_mock_engine(metrics=None)
    serving_chat = _build_serving_chat(mock_engine, enable_per_request_metrics=True)

    request = ChatCompletionRequest(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "Test prompt"}],
        max_tokens=10,
        stream=False,
        include_metrics=True,
    )

    response = await serving_chat.create_chat_completion(request)
    # metrics object is returned but all fields are None when RequestStateStats is None
    assert response.metrics is not None
    assert response.metrics.time_to_first_token_ms is None
    assert response.metrics.generation_time_ms is None
    assert response.metrics.queue_time_ms is None


# ---------------------------------------------------------------------------
# completion_tokens_details with reasoning tokens
# ---------------------------------------------------------------------------


def test_usage_completion_tokens_details_no_reasoning():
    # When no reasoning parser is present, completion_tokens_details should be None
    from vllm.entrypoints.openai.engine.protocol import (
        UsageInfo,
    )

    usage = UsageInfo(prompt_tokens=3, completion_tokens=5, total_tokens=8)
    assert usage.completion_tokens_details is None


def test_usage_completion_tokens_details_with_reasoning():
    from vllm.entrypoints.openai.engine.protocol import (
        CompletionTokensDetails,
        UsageInfo,
    )

    usage = UsageInfo(prompt_tokens=3, completion_tokens=5, total_tokens=8)
    usage.completion_tokens_details = CompletionTokensDetails(reasoning_tokens=2)
    assert usage.completion_tokens_details is not None
    assert usage.completion_tokens_details.reasoning_tokens == 2


# ---------------------------------------------------------------------------
# RequestStateStats instantiation
# ---------------------------------------------------------------------------


def test_request_state_stats_defaults():
    stats = RequestStateStats()
    assert stats.queued_ts == 0.0
    assert stats.scheduled_ts == 0.0
    assert stats.first_token_ts == 0.0
    assert stats.last_token_ts == 0.0
    assert stats.num_generation_tokens == 0


def test_request_state_stats_with_values():
    stats = RequestStateStats(
        queued_ts=1.0,
        scheduled_ts=2.0,
        first_token_ts=3.0,
        last_token_ts=5.0,
        num_generation_tokens=20,
    )
    assert stats.queued_ts == 1.0
    assert stats.scheduled_ts == 2.0
    assert stats.first_token_ts == 3.0
    assert stats.last_token_ts == 5.0
    assert stats.num_generation_tokens == 20
