# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, field
from http import HTTPStatus
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import TypeAdapter

from vllm.config.multimodal import MultiModalConfig
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.openai.models.protocol import BaseModelPath
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.serve.render.serving import OpenAIServingRender
from vllm.entrypoints.serve.tokenize.api_router import (
    attach_router as attach_tokenize_router,
)
from vllm.entrypoints.serve.tokenize.protocol import (
    TokenizeChatRequest,
    TokenizeCompletionRequest,
    TokenizeRequest,
    TokenizeResponse,
    TokenizeResponsesRequest,
)
from vllm.entrypoints.serve.tokenize.serving import OpenAIServingTokenization
from vllm.v1.engine.async_llm import AsyncLLM

MODEL_NAME = "openai-community/gpt2"
BASE_MODEL_PATHS = [
    BaseModelPath(name=MODEL_NAME, model_path=MODEL_NAME),
]
pytestmark = pytest.mark.skip_global_cleanup


def test_tokenize_request_accepts_responses_input():
    request = TypeAdapter(TokenizeRequest).validate_python(
        {
            "model": MODEL_NAME,
            "input": "Test prompt",
        }
    )

    assert isinstance(request, TokenizeResponsesRequest)


def test_tokenize_request_accepts_responses_list_input():
    request = TypeAdapter(TokenizeRequest).validate_python(
        {
            "model": MODEL_NAME,
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "Test prompt",
                        }
                    ],
                }
            ],
        }
    )

    assert isinstance(request, TokenizeResponsesRequest)


def _make_tokenize_client(handler: Any) -> TestClient:
    app = FastAPI()
    app.state.args = SimpleNamespace(enable_tokenizer_info_endpoint=False)
    app.state.openai_serving_tokenization = handler
    attach_tokenize_router(app)
    return TestClient(app)


@pytest.mark.asyncio
async def test_init_generate_state_wires_responses_handler(monkeypatch):
    class DummyServing:
        def __init__(self, *args, **kwargs):
            pass

        def warmup(self):
            pass

    class DummyTokenization:
        openai_serving_responses = None

        def set_openai_serving_responses(self, openai_serving_responses):
            self.openai_serving_responses = openai_serving_responses

    import vllm.entrypoints.anthropic.serving as anthropic_serving
    import vllm.entrypoints.chat_utils as chat_utils
    import vllm.entrypoints.openai.chat_completion.batch_serving as chat_batch_serving
    import vllm.entrypoints.openai.chat_completion.serving as chat_serving
    import vllm.entrypoints.openai.completion.serving as completion_serving
    import vllm.entrypoints.openai.responses.serving as responses_serving
    import vllm.entrypoints.serve.disagg.serving as disagg_serving
    from vllm.entrypoints.openai.generate.api_router import init_generate_state

    monkeypatch.setattr(chat_utils, "load_chat_template", lambda _: None)
    monkeypatch.setattr(responses_serving, "OpenAIServingResponses", DummyServing)
    monkeypatch.setattr(chat_serving, "OpenAIServingChat", DummyServing)
    monkeypatch.setattr(chat_batch_serving, "OpenAIServingChatBatch", DummyServing)
    monkeypatch.setattr(completion_serving, "OpenAIServingCompletion", DummyServing)
    monkeypatch.setattr(anthropic_serving, "AnthropicServingMessages", DummyServing)
    monkeypatch.setattr(disagg_serving, "ServingTokens", DummyServing)

    state = SimpleNamespace(
        openai_serving_models=MagicMock(),
        openai_serving_render=MagicMock(),
        openai_serving_tokenization=DummyTokenization(),
    )
    args = SimpleNamespace(
        tool_server=None,
        chat_template=None,
        chat_template_content_format="auto",
        fingerprint_mode="full",
        fingerprint_value=None,
        return_tokens_as_token_ids=False,
        enable_auto_tool_choice=False,
        tool_call_parser=None,
        structured_outputs_config=SimpleNamespace(reasoning_parser=None),
        enable_prompt_tokens_details=False,
        enable_force_include_usage=False,
        enable_log_outputs=False,
        response_role="assistant",
        default_chat_template_kwargs=None,
        trust_request_chat_template=False,
        exclude_tools_when_tool_choice_none=False,
        enable_log_deltas=False,
        tokens_only=False,
    )

    await init_generate_state(
        engine_client=MagicMock(),
        state=state,
        args=args,
        request_logger=None,
        supported_tasks=("generate",),
    )

    assert isinstance(state.openai_serving_responses, DummyServing)
    assert state.openai_serving_tokenization.openai_serving_responses is (
        state.openai_serving_responses
    )


def test_tokenize_route_parses_responses_request():
    handler = MagicMock()
    handler.create_tokenize = AsyncMock(
        return_value=TokenizeResponse(
            tokens=[1, 2],
            token_strs=["one", "two"],
            count=2,
            max_model_len=100,
        )
    )

    with _make_tokenize_client(handler) as client:
        response = client.post(
            "/tokenize",
            json={
                "model": MODEL_NAME,
                "input": "Test prompt",
                "return_token_strs": True,
            },
        )

    assert response.status_code == HTTPStatus.OK
    assert response.json() == {
        "tokens": [1, 2],
        "token_strs": ["one", "two"],
        "count": 2,
        "max_model_len": 100,
    }

    parsed_request = handler.create_tokenize.await_args.args[0]
    assert isinstance(parsed_request, TokenizeResponsesRequest)
    assert parsed_request.input == "Test prompt"
    assert parsed_request.return_token_strs is True


def test_tokenize_route_returns_responses_error_status():
    handler = MagicMock()
    handler.create_tokenize = AsyncMock(
        return_value=ErrorResponse(
            error={
                "message": "Responses API tokenization is unavailable.",
                "type": "NotImplementedError",
                "code": HTTPStatus.NOT_IMPLEMENTED,
            }
        )
    )

    with _make_tokenize_client(handler) as client:
        response = client.post(
            "/tokenize",
            json={
                "model": MODEL_NAME,
                "input": "Test prompt",
            },
        )

    assert response.status_code == HTTPStatus.NOT_IMPLEMENTED
    assert response.json()["error"]["code"] == HTTPStatus.NOT_IMPLEMENTED


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
    renderer_num_workers: int = 1

    def get_diff_sampling_param(self):
        return self.diff_sampling_param or {}


def _build_serving_tokenization(engine: AsyncLLM) -> OpenAIServingTokenization:
    models = OpenAIServingModels(
        engine_client=engine,
        base_model_paths=BASE_MODEL_PATHS,
    )
    serving_render = OpenAIServingRender(
        model_config=engine.model_config,
        renderer=engine.renderer,
        model_registry=models.registry,
        request_logger=None,
        chat_template=None,
        chat_template_content_format="auto",
    )
    return OpenAIServingTokenization(
        engine,
        models,
        openai_serving_render=serving_render,
        request_logger=None,
        chat_template=None,
        chat_template_content_format="auto",
    )


@pytest.mark.asyncio
async def test_tokenize_chat_skips_mm_cache_for_renderer_only_path():
    mock_engine = MagicMock(spec=AsyncLLM)
    mock_engine.errored = False
    mock_engine.model_config = MockModelConfig()
    mock_engine.input_processor = MagicMock()
    mock_engine.renderer = MagicMock()

    serving = _build_serving_tokenization(mock_engine)
    serving.openai_serving_render.preprocess_chat = AsyncMock(
        return_value=(
            [{"role": "user", "content": "Test"}],
            [{"prompt_token_ids": [1, 2, 3]}],
        )
    )

    request = TokenizeChatRequest(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "Test prompt"}],
    )

    response = await serving.create_tokenize(request, MagicMock(headers={}))

    assert response.tokens == [1, 2, 3]
    assert (
        serving.openai_serving_render.preprocess_chat.call_args.kwargs["skip_mm_cache"]
        is True
    )


@pytest.mark.asyncio
async def test_tokenize_completion_skips_mm_cache_for_renderer_only_path():
    mock_engine = MagicMock(spec=AsyncLLM)
    mock_engine.errored = False
    mock_engine.model_config = MockModelConfig()
    mock_engine.input_processor = MagicMock()
    mock_engine.renderer = MagicMock()

    serving = _build_serving_tokenization(mock_engine)
    serving.openai_serving_render.preprocess_completion = AsyncMock(
        return_value=[{"prompt_token_ids": [1, 2, 3]}]
    )

    request = TokenizeCompletionRequest(
        model=MODEL_NAME,
        prompt="Test prompt",
    )

    response = await serving.create_tokenize(request, MagicMock(headers={}))

    assert response.tokens == [1, 2, 3]
    assert (
        serving.openai_serving_render.preprocess_completion.call_args.kwargs[
            "skip_mm_cache"
        ]
        is True
    )


@pytest.mark.asyncio
async def test_tokenize_responses_without_handler_returns_501():
    mock_engine = MagicMock(spec=AsyncLLM)
    mock_engine.errored = False
    mock_engine.model_config = MockModelConfig()
    mock_engine.input_processor = MagicMock()
    mock_engine.renderer = MagicMock()

    serving = _build_serving_tokenization(mock_engine)

    request = TokenizeResponsesRequest(
        model=MODEL_NAME,
        input="Test prompt",
    )

    response = await serving.create_tokenize(request, MagicMock(headers={}))

    assert isinstance(response, ErrorResponse)
    assert response.error.code == HTTPStatus.NOT_IMPLEMENTED


@pytest.mark.asyncio
async def test_tokenize_responses_uses_responses_renderer_path():
    mock_engine = MagicMock(spec=AsyncLLM)
    mock_engine.errored = False
    mock_engine.model_config = MockModelConfig()
    mock_engine.input_processor = MagicMock()
    mock_engine.renderer = MagicMock()

    serving = _build_serving_tokenization(mock_engine)
    responses_handler = MagicMock()
    responses_handler.render_response_inputs = AsyncMock(
        return_value=(
            [{"role": "user", "content": "Test prompt"}],
            [{"prompt_token_ids": [1, 2, 3]}],
        )
    )
    serving.set_openai_serving_responses(responses_handler)

    request = TokenizeResponsesRequest(
        model=MODEL_NAME,
        input="Test prompt",
    )

    response = await serving.create_tokenize(request, MagicMock(headers={}))

    assert response.tokens == [1, 2, 3]
    responses_handler.render_response_inputs.assert_awaited_once_with(
        request,
        skip_mm_cache=True,
    )


@pytest.mark.asyncio
async def test_tokenize_responses_return_token_strs():
    mock_engine = MagicMock(spec=AsyncLLM)
    mock_engine.errored = False
    mock_engine.model_config = MockModelConfig()
    mock_engine.input_processor = MagicMock()
    mock_engine.renderer = MagicMock()
    tokenizer = mock_engine.renderer.get_tokenizer.return_value
    tokenizer.convert_ids_to_tokens.return_value = ["one", "two"]

    serving = _build_serving_tokenization(mock_engine)
    responses_handler = MagicMock()
    responses_handler.render_response_inputs = AsyncMock(
        return_value=(
            [{"role": "user", "content": "Test prompt"}],
            [{"prompt_token_ids": [1, 2]}],
        )
    )
    serving.set_openai_serving_responses(responses_handler)

    request = TokenizeResponsesRequest(
        model=MODEL_NAME,
        input="Test prompt",
        return_token_strs=True,
    )

    response = await serving.create_tokenize(request, MagicMock(headers={}))

    assert response.tokens == [1, 2]
    assert response.token_strs == ["one", "two"]
