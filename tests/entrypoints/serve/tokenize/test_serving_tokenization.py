# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from vllm.config.multimodal import MultiModalConfig
from vllm.entrypoints.openai.models.protocol import BaseModelPath
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.serve.render.serving import OpenAIServingRender
from vllm.entrypoints.serve.tokenize.protocol import (
    TokenizeChatRequest,
    TokenizeCompletionRequest,
)
from vllm.entrypoints.serve.tokenize.serving import OpenAIServingTokenization
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
