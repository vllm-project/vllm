# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import suppress
from dataclasses import dataclass, field
from http import HTTPStatus
from unittest.mock import AsyncMock, MagicMock

import pytest

from vllm.config.multimodal import MultiModalConfig
from vllm.entrypoints.openai.protocol import CompletionRequest, ErrorResponse
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.entrypoints.openai.serving_models import BaseModelPath, OpenAIServingModels
from vllm.lora.request import LoRARequest
from vllm.lora.resolver import LoRAResolver, LoRAResolverRegistry
from vllm.tokenizers import get_tokenizer
from vllm.v1.engine.async_llm import AsyncLLM

MODEL_NAME = "openai-community/gpt2"
BASE_MODEL_PATHS = [BaseModelPath(name=MODEL_NAME, model_path=MODEL_NAME)]

MOCK_RESOLVER_NAME = "mock_test_resolver"


@dataclass
class MockHFConfig:
    model_type: str = "any"


@dataclass
class MockModelConfig:
    """Minimal mock ModelConfig for testing."""

    model: str = MODEL_NAME
    tokenizer: str = MODEL_NAME
    trust_remote_code: bool = False
    tokenizer_mode: str = "auto"
    max_model_len: int = 100
    tokenizer_revision: str | None = None
    multimodal_config: MultiModalConfig = field(default_factory=MultiModalConfig)
    hf_config: MockHFConfig = field(default_factory=MockHFConfig)
    logits_processors: list[str] | None = None
    logits_processor_pattern: str | None = None
    diff_sampling_param: dict | None = None
    allowed_local_media_path: str = ""
    allowed_media_domains: list[str] | None = None
    encoder_config = None
    generation_config: str = "auto"
    skip_tokenizer_init: bool = False

    def get_diff_sampling_param(self):
        return self.diff_sampling_param or {}


class MockLoRAResolver(LoRAResolver):
    async def resolve_lora(
        self, base_model_name: str, lora_name: str
    ) -> LoRARequest | None:
        if lora_name == "test-lora":
            return LoRARequest(
                lora_name="test-lora",
                lora_int_id=1,
                lora_path="/fake/path/test-lora",
            )
        elif lora_name == "invalid-lora":
            return LoRARequest(
                lora_name="invalid-lora",
                lora_int_id=2,
                lora_path="/fake/path/invalid-lora",
            )
        return None


@pytest.fixture(autouse=True)
def register_mock_resolver():
    """Fixture to register and unregister the mock LoRA resolver."""
    resolver = MockLoRAResolver()
    LoRAResolverRegistry.register_resolver(MOCK_RESOLVER_NAME, resolver)
    yield
    # Cleanup: remove the resolver after the test runs
    if MOCK_RESOLVER_NAME in LoRAResolverRegistry.resolvers:
        del LoRAResolverRegistry.resolvers[MOCK_RESOLVER_NAME]


@pytest.fixture
def mock_serving_setup():
    """Provides a mocked engine and serving completion instance."""
    mock_engine = MagicMock(spec=AsyncLLM)
    mock_engine.errored = False

    tokenizer = get_tokenizer(MODEL_NAME)
    mock_engine.get_tokenizer = AsyncMock(return_value=tokenizer)

    async def mock_add_lora_side_effect(lora_request: LoRARequest):
        """Simulate engine behavior when adding LoRAs."""
        if lora_request.lora_name == "test-lora":
            # Simulate successful addition
            return True
        if lora_request.lora_name == "invalid-lora":
            # Simulate failure during addition (e.g. invalid format)
            raise ValueError(f"Simulated failure adding LoRA: {lora_request.lora_name}")
        return True

    mock_engine.add_lora = AsyncMock(side_effect=mock_add_lora_side_effect)

    async def mock_generate(*args, **kwargs):
        for _ in []:
            yield _

    mock_engine.generate = MagicMock(spec=AsyncLLM.generate, side_effect=mock_generate)

    mock_engine.generate.reset_mock()
    mock_engine.add_lora.reset_mock()

    mock_engine.model_config = MockModelConfig()
    mock_engine.input_processor = MagicMock()
    mock_engine.io_processor = MagicMock()

    models = OpenAIServingModels(
        engine_client=mock_engine,
        base_model_paths=BASE_MODEL_PATHS,
    )

    serving_completion = OpenAIServingCompletion(
        mock_engine, models, request_logger=None
    )

    serving_completion._process_inputs = AsyncMock(
        return_value=(MagicMock(name="engine_request"), {})
    )

    return mock_engine, serving_completion


@pytest.mark.asyncio
async def test_serving_completion_with_lora_resolver(mock_serving_setup, monkeypatch):
    monkeypatch.setenv("VLLM_ALLOW_RUNTIME_LORA_UPDATING", "true")

    mock_engine, serving_completion = mock_serving_setup

    lora_model_name = "test-lora"
    req_found = CompletionRequest(
        model=lora_model_name,
        prompt="Generate with LoRA",
    )

    # Suppress potential errors during the mocked generate call,
    # as we are primarily checking for add_lora and generate calls
    with suppress(Exception):
        await serving_completion.create_completion(req_found)

    mock_engine.add_lora.assert_awaited_once()
    called_lora_request = mock_engine.add_lora.call_args[0][0]
    assert isinstance(called_lora_request, LoRARequest)
    assert called_lora_request.lora_name == lora_model_name

    mock_engine.generate.assert_called_once()
    called_lora_request = mock_engine.generate.call_args[1]["lora_request"]
    assert isinstance(called_lora_request, LoRARequest)
    assert called_lora_request.lora_name == lora_model_name


@pytest.mark.asyncio
async def test_serving_completion_resolver_not_found(mock_serving_setup, monkeypatch):
    monkeypatch.setenv("VLLM_ALLOW_RUNTIME_LORA_UPDATING", "true")

    mock_engine, serving_completion = mock_serving_setup

    non_existent_model = "non-existent-lora-adapter"
    req = CompletionRequest(
        model=non_existent_model,
        prompt="what is 1+1?",
    )

    response = await serving_completion.create_completion(req)

    mock_engine.add_lora.assert_not_awaited()
    mock_engine.generate.assert_not_called()

    assert isinstance(response, ErrorResponse)
    assert response.error.code == HTTPStatus.NOT_FOUND.value
    assert non_existent_model in response.error.message


@pytest.mark.asyncio
async def test_serving_completion_resolver_add_lora_fails(
    mock_serving_setup, monkeypatch
):
    monkeypatch.setenv("VLLM_ALLOW_RUNTIME_LORA_UPDATING", "true")

    mock_engine, serving_completion = mock_serving_setup

    invalid_model = "invalid-lora"
    req = CompletionRequest(
        model=invalid_model,
        prompt="what is 1+1?",
    )

    response = await serving_completion.create_completion(req)

    # Assert add_lora was called before the failure
    mock_engine.add_lora.assert_awaited_once()
    called_lora_request = mock_engine.add_lora.call_args[0][0]
    assert isinstance(called_lora_request, LoRARequest)
    assert called_lora_request.lora_name == invalid_model

    # Assert generate was *not* called due to the failure
    mock_engine.generate.assert_not_called()

    # Assert the correct error response
    assert isinstance(response, ErrorResponse)
    assert response.error.code == HTTPStatus.BAD_REQUEST.value
    assert invalid_model in response.error.message


@pytest.mark.asyncio
async def test_serving_completion_flag_not_set(mock_serving_setup):
    mock_engine, serving_completion = mock_serving_setup

    lora_model_name = "test-lora"
    req_found = CompletionRequest(
        model=lora_model_name,
        prompt="Generate with LoRA",
    )

    await serving_completion.create_completion(req_found)

    mock_engine.add_lora.assert_not_called()
    mock_engine.generate.assert_not_called()
