# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from vllm.config.multimodal import MultiModalConfig
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion
from vllm.entrypoints.openai.engine.protocol import GenerationError
from vllm.entrypoints.openai.models.protocol import BaseModelPath
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.scale_out.render.serving import ServingRender
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.renderers.hf import HfRenderer
from vllm.renderers.online_renderer import OnlineRenderer
from vllm.tokenizers.registry import cached_tokenizer_from_config
from vllm.v1.engine.async_llm import AsyncLLM

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
    renderer_num_workers: int = 1

    def get_diff_sampling_param(self):
        return self.diff_sampling_param or {}


@dataclass
class MockParallelConfig:
    _api_process_rank: int = 0


@dataclass
class MockVllmConfig:
    model_config: MockModelConfig
    parallel_config: MockParallelConfig


def _build_serving_completion(engine: AsyncLLM) -> OpenAIServingCompletion:
    models = OpenAIServingModels(
        engine_client=engine,
        base_model_paths=BASE_MODEL_PATHS,
    )
    online_renderer = OnlineRenderer(
        model_config=engine.model_config,
        renderer=engine.renderer,
        request_logger=None,
        chat_template=None,
        chat_template_content_format="auto",
    )
    return OpenAIServingCompletion(
        engine,
        models,
        online_renderer=online_renderer,
        request_logger=None,
    )


def _build_renderer(model_config: MockModelConfig):
    return HfRenderer(
        MockVllmConfig(model_config, parallel_config=MockParallelConfig()),
        cached_tokenizer_from_config(model_config),
    )


@pytest.mark.asyncio
async def test_completion_error_non_stream():
    """test finish_reason='error' returns 500 InternalServerError (non-streaming)"""
    mock_engine = MagicMock(spec=AsyncLLM)
    mock_engine.errored = False
    mock_engine.model_config = MockModelConfig()
    mock_engine.input_processor = MagicMock()
    mock_engine.renderer = _build_renderer(mock_engine.model_config)

    serving_completion = _build_serving_completion(mock_engine)

    completion_output = CompletionOutput(
        index=0,
        text="",
        token_ids=[],
        cumulative_logprob=None,
        logprobs=None,
        finish_reason="error",
    )

    request_output = RequestOutput(
        request_id="test-id",
        prompt="Test prompt",
        prompt_token_ids=[1, 2, 3],
        prompt_logprobs=None,
        outputs=[completion_output],
        finished=True,
        metrics=None,
        lora_request=None,
        encoder_prompt=None,
        encoder_prompt_token_ids=None,
    )

    async def mock_generate(*args, **kwargs):
        yield request_output

    mock_engine.generate = MagicMock(side_effect=mock_generate)

    request = CompletionRequest(
        model=MODEL_NAME,
        prompt="Test prompt",
        max_tokens=10,
        stream=False,
    )

    with pytest.raises(GenerationError):
        await serving_completion.create_completion(request)


@pytest.mark.asyncio
async def test_openai_completion_keeps_mm_cache_for_engine_execution():
    mock_engine = MagicMock(spec=AsyncLLM)
    mock_engine.errored = False
    mock_engine.model_config = MockModelConfig()
    mock_engine.input_processor = MagicMock()
    mock_engine.renderer = _build_renderer(mock_engine.model_config)

    serving_completion = _build_serving_completion(mock_engine)
    serving_completion.online_renderer.preprocess_completion = AsyncMock(
        return_value=[{"prompt_token_ids": [1, 2, 3]}]
    )

    request = CompletionRequest(
        model=MODEL_NAME,
        prompt="Test prompt",
    )

    result = await serving_completion.render_completion_request(request)

    assert isinstance(result, list)
    assert (
        serving_completion.online_renderer.preprocess_completion.call_args.kwargs[
            "skip_mm_cache"
        ]
        is False
    )


def _build_serving_render(engine: AsyncLLM) -> ServingRender:
    models = OpenAIServingModels(
        engine_client=engine,
        base_model_paths=BASE_MODEL_PATHS,
    )
    online_renderer = OnlineRenderer(
        model_config=engine.model_config,
        renderer=engine.renderer,
        request_logger=None,
        chat_template=None,
        chat_template_content_format="auto",
    )

    serving_render = ServingRender(models, online_renderer)

    async def _fake_preprocess_chat(*args, **kwargs):
        # return conversation, engine_inputs
        return (
            [{"role": "user", "content": "Test"}],
            [{"prompt_token_ids": [1, 2, 3]}],
        )

    serving_render.online_renderer.preprocess_chat = AsyncMock(
        side_effect=_fake_preprocess_chat
    )
    return serving_render


@pytest.mark.asyncio
async def test_renderer_only_completion_request_skips_mm_cache():
    mock_engine = MagicMock(spec=AsyncLLM)
    mock_engine.errored = False
    mock_engine.model_config = MockModelConfig()
    mock_engine.input_processor = MagicMock()
    mock_engine.renderer = _build_renderer(mock_engine.model_config)

    serving_render = _build_serving_render(mock_engine)

    serving_render.online_renderer.preprocess_completion = AsyncMock(
        return_value=[{"prompt_token_ids": [1, 2, 3]}]
    )

    request = CompletionRequest(
        model=MODEL_NAME,
        prompt="Test prompt",
    )

    result = await serving_render.render_completion_request(request)

    assert isinstance(result, list)
    assert (
        serving_render.online_renderer.preprocess_completion.call_args.kwargs[
            "skip_mm_cache"
        ]
        is True
    )


@pytest.mark.asyncio
async def test_completion_error_stream():
    """test finish_reason='error' returns 500 InternalServerError (streaming)"""
    mock_engine = MagicMock(spec=AsyncLLM)
    mock_engine.errored = False
    mock_engine.model_config = MockModelConfig()
    mock_engine.input_processor = MagicMock()
    mock_engine.renderer = _build_renderer(mock_engine.model_config)

    serving_completion = _build_serving_completion(mock_engine)

    completion_output_1 = CompletionOutput(
        index=0,
        text="Hello",
        token_ids=[100],
        cumulative_logprob=None,
        logprobs=None,
        finish_reason=None,
    )

    request_output_1 = RequestOutput(
        request_id="test-id",
        prompt="Test prompt",
        prompt_token_ids=[1, 2, 3],
        prompt_logprobs=None,
        outputs=[completion_output_1],
        finished=False,
        metrics=None,
        lora_request=None,
        encoder_prompt=None,
        encoder_prompt_token_ids=None,
    )

    completion_output_2 = CompletionOutput(
        index=0,
        text="Hello",
        token_ids=[100],
        cumulative_logprob=None,
        logprobs=None,
        finish_reason="error",
    )

    request_output_2 = RequestOutput(
        request_id="test-id",
        prompt="Test prompt",
        prompt_token_ids=[1, 2, 3],
        prompt_logprobs=None,
        outputs=[completion_output_2],
        finished=True,
        metrics=None,
        lora_request=None,
        encoder_prompt=None,
        encoder_prompt_token_ids=None,
    )

    async def mock_generate(*args, **kwargs):
        yield request_output_1
        yield request_output_2

    mock_engine.generate = MagicMock(side_effect=mock_generate)

    request = CompletionRequest(
        model=MODEL_NAME,
        prompt="Test prompt",
        max_tokens=10,
        stream=True,
    )

    response = await serving_completion.create_completion(request)

    chunks = []
    async for chunk in response:
        chunks.append(chunk)

    assert len(chunks) >= 2
    assert any("Internal server error" in chunk for chunk in chunks), (
        f"Expected error message in chunks: {chunks}"
    )
    assert chunks[-1] == "data: [DONE]\n\n"


def test_json_schema_response_format_missing_schema():
    """When response_format type is 'json_schema' but the json_schema field
    is not provided, request construction should raise a validation error
    so the API returns 400 instead of 500."""
    with pytest.raises(Exception, match="json_schema.*must be provided"):
        CompletionRequest(
            model=MODEL_NAME,
            prompt="Test prompt",
            max_tokens=10,
            response_format={"type": "json_schema"},
        )


@pytest.mark.parametrize("format_value", [None, {}])
def test_structural_tag_response_format_invalid(format_value):
    """Malformed structural tags should be rejected during request validation."""
    with pytest.raises(
        ValidationError,
        match="Invalid response_format structural_tag",
    ):
        CompletionRequest(
            model=MODEL_NAME,
            prompt="Test prompt",
            max_tokens=10,
            response_format={"type": "structural_tag", "format": format_value},
        )


@pytest.mark.parametrize("structural_tag", ["not json", ""])
def test_structured_outputs_structural_tag_invalid(structural_tag):
    """Malformed direct structured_outputs structural tags should be rejected."""
    with pytest.raises(
        ValidationError,
        match="Invalid structured_outputs structural_tag",
    ):
        CompletionRequest(
            model=MODEL_NAME,
            prompt="Test prompt",
            max_tokens=10,
            structured_outputs={"structural_tag": structural_tag},
        )


def test_negative_prompt_token_ids_nested():
    """Negative token IDs in prompt (nested list) should raise validation error."""
    with pytest.raises(Exception, match="greater than or equal to 0"):
        CompletionRequest(
            model=MODEL_NAME,
            prompt=[[-1]],
            max_tokens=10,
        )


def test_negative_prompt_token_ids_flat():
    """Negative token IDs in prompt (flat list) should raise validation error."""
    with pytest.raises(Exception, match="greater than or equal to 0"):
        CompletionRequest(
            model=MODEL_NAME,
            prompt=[-1],
            max_tokens=10,
        )
