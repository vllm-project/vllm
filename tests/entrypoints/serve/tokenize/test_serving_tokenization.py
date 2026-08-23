# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from argparse import Namespace
from dataclasses import dataclass, field
from http import HTTPStatus
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm.config.multimodal import MultiModalConfig
from vllm.entrypoints.openai.models.protocol import BaseModelPath
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.serve.exception_handling.register import init_exception_handler
from vllm.entrypoints.serve.tokenize.api_router import attach_router
from vllm.entrypoints.serve.tokenize.protocol import (
    TokenizeChatRequest,
    TokenizeCompletionRequest,
)
from vllm.entrypoints.serve.tokenize.serving import ServingTokenization
from vllm.exceptions import VLLMNotFoundError, VLLMValidationError
from vllm.renderers.online_renderer import OnlineRenderer
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


def _build_serving_tokenization(engine: AsyncLLM) -> ServingTokenization:
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
    return ServingTokenization(
        models,
        online_renderer=online_renderer,
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
    serving.online_renderer.preprocess_chat = AsyncMock(
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
        serving.online_renderer.preprocess_chat.call_args.kwargs["skip_mm_cache"]
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
    serving.online_renderer.preprocess_completion = AsyncMock(
        return_value=[{"prompt_token_ids": [1, 2, 3]}]
    )

    request = TokenizeCompletionRequest(
        model=MODEL_NAME,
        prompt="Test prompt",
    )

    response = await serving.create_tokenize(request, MagicMock(headers={}))

    assert response.tokens == [1, 2, 3]
    assert (
        serving.online_renderer.preprocess_completion.call_args.kwargs["skip_mm_cache"]
        is True
    )


class TestDetokenizeClientErrorResponses:
    """Client-caused errors from /detokenize are 4xx, not 500.

    /detokenize used to wrap ``create_detokenize`` in a blanket
    ``except Exception`` that reported every failure as 500, while its
    sibling /tokenize let exceptions propagate to the global exception
    handlers, which map client-caused errors to 4xx and keep genuine
    server errors at 500. These tests pin the shared contract for both
    endpoints. See #52246 for the same fix in the Anthropic entrypoint.
    """

    @staticmethod
    def _make_api_app(handler: MagicMock) -> FastAPI:
        app = FastAPI()
        app.state.args = Namespace(log_error_stack=False)
        app.state.serving_tokenization = handler
        attach_router(app)
        init_exception_handler(app)
        return app

    def _post(self, handler: MagicMock, path: str = "/detokenize"):
        app = self._make_api_app(handler)
        body: dict[str, Any] = {"model": "test-model"}
        if path == "/detokenize":
            body["tokens"] = [1, 2, 3]
        else:
            body["prompt"] = "Hello"
        with TestClient(app, raise_server_exceptions=False) as client:
            return client.post(path, json=body)

    def test_vllm_validation_error_returns_bad_request(self):
        handler = MagicMock(spec=ServingTokenization)
        handler.create_detokenize.side_effect = VLLMValidationError(
            "invalid token ids", parameter="tokens"
        )

        response = self._post(handler)

        assert response.status_code == HTTPStatus.BAD_REQUEST
        error = response.json()["error"]
        assert error["type"] == "BadRequestError"
        assert error["param"] == "tokens"

    def test_vllm_not_found_error_returns_not_found(self):
        handler = MagicMock(spec=ServingTokenization)
        handler.create_detokenize.side_effect = VLLMNotFoundError(
            "LoRA adapter nonexistent-lora not found"
        )

        response = self._post(handler)

        assert response.status_code == HTTPStatus.NOT_FOUND
        assert response.json()["error"]["type"] == "NotFoundError"

    def test_value_error_returns_bad_request(self):
        handler = MagicMock(spec=ServingTokenization)
        handler.create_detokenize.side_effect = ValueError("bad input")

        response = self._post(handler)

        assert response.status_code == HTTPStatus.BAD_REQUEST
        assert response.json()["error"]["type"] == "BadRequestError"

    def test_overflow_error_returns_bad_request(self):
        """Fast tokenizers raise OverflowError for out-of-range token ids.

        This used to be special-cased into a RequestValidationError; the
        global OverflowError handler keeps the same 400 status code.
        """
        handler = MagicMock(spec=ServingTokenization)
        handler.create_detokenize.side_effect = OverflowError(
            "out of range integral type conversion attempted"
        )

        response = self._post(handler)

        assert response.status_code == HTTPStatus.BAD_REQUEST
        assert response.json()["error"]["type"] == "BadRequestError"

    def test_server_error_still_returns_internal_server_error(self):
        """Genuine server bugs keep the existing 500 behaviour."""
        handler = MagicMock(spec=ServingTokenization)
        handler.create_detokenize.side_effect = RuntimeError("boom")

        response = self._post(handler)

        assert response.status_code == HTTPStatus.INTERNAL_SERVER_ERROR
        assert response.json()["error"]["type"] == "InternalServerError"

    def test_tokenize_and_detokenize_agree_on_client_errors(self):
        """The two sibling endpoints report the same client error alike."""
        statuses = {}
        for path, method in (
            ("/tokenize", "create_tokenize"),
            ("/detokenize", "create_detokenize"),
        ):
            handler = MagicMock(spec=ServingTokenization)
            getattr(handler, method).side_effect = VLLMValidationError(
                "invalid input", parameter="model"
            )
            statuses[path] = self._post(handler, path=path).status_code

        assert statuses["/tokenize"] == HTTPStatus.BAD_REQUEST
        assert statuses["/detokenize"] == HTTPStatus.BAD_REQUEST
