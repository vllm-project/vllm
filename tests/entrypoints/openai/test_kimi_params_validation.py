# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Kimi model parameter validation in OpenAIServingChat."""

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import pytest

from vllm.config import MultiModalConfig
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.openai.models.serving import BaseModelPath, OpenAIServingModels
from vllm.renderers.hf import HfRenderer
from vllm.tokenizers.registry import tokenizer_args_from_config
from vllm.v1.engine.async_llm import AsyncLLM

MODEL_NAME = "openai-community/gpt2"
CHAT_TEMPLATE = "Dummy chat template for testing {}"
BASE_MODEL_PATHS = [BaseModelPath(name=MODEL_NAME, model_path=MODEL_NAME)]


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
    hf_config: MockHFConfig = field(default_factory=MockHFConfig)
    hf_text_config: MockHFConfig = field(default_factory=MockHFConfig)
    logits_processors: list[str] | None = None
    diff_sampling_param: dict | None = None
    allowed_local_media_path: str = ""
    allowed_media_domains: list[str] | None = None
    encoder_config = None
    generation_config: str = "auto"
    override_generation_config: dict[str, Any] = field(default_factory=dict)
    media_io_kwargs: dict[str, dict[str, Any]] = field(default_factory=dict)
    skip_tokenizer_init: bool = False
    is_encoder_decoder: bool = False
    is_multimodal_model: bool = False

    def get_diff_sampling_param(self):
        return self.diff_sampling_param or {}


@dataclass
class MockVllmConfig:
    model_config: MockModelConfig


def _build_renderer(model_config: MockModelConfig):
    _, tokenizer_name, _, kwargs = tokenizer_args_from_config(model_config)
    return HfRenderer.from_config(
        MockVllmConfig(model_config),
        tokenizer_kwargs={**kwargs, "tokenizer_name": tokenizer_name},
    )


def _build_kimi_serving_chat() -> OpenAIServingChat:
    """Build an OpenAIServingChat with Kimi K2 model type."""
    model_config = MockModelConfig(
        hf_config=MockHFConfig(model_type="kimi_k2"),
        hf_text_config=MockHFConfig(model_type="kimi_k2"),
    )
    mock_engine = MagicMock(spec=AsyncLLM)
    mock_engine.errored = False
    mock_engine.model_config = model_config
    mock_engine.input_processor = MagicMock()
    mock_engine.io_processor = MagicMock()
    mock_engine.renderer = _build_renderer(model_config)

    models = OpenAIServingModels(
        engine_client=mock_engine,
        base_model_paths=BASE_MODEL_PATHS,
    )
    serving_chat = OpenAIServingChat(
        mock_engine,
        models,
        response_role="assistant",
        chat_template=CHAT_TEMPLATE,
        chat_template_content_format="auto",
        request_logger=None,
    )
    return serving_chat


def _build_non_kimi_serving_chat() -> OpenAIServingChat:
    """Build an OpenAIServingChat with a non-Kimi model type."""
    model_config = MockModelConfig()
    mock_engine = MagicMock(spec=AsyncLLM)
    mock_engine.errored = False
    mock_engine.model_config = model_config
    mock_engine.input_processor = MagicMock()
    mock_engine.io_processor = MagicMock()
    mock_engine.renderer = _build_renderer(model_config)

    models = OpenAIServingModels(
        engine_client=mock_engine,
        base_model_paths=BASE_MODEL_PATHS,
    )
    serving_chat = OpenAIServingChat(
        mock_engine,
        models,
        response_role="assistant",
        chat_template=CHAT_TEMPLATE,
        chat_template_content_format="auto",
        request_logger=None,
    )
    return serving_chat


@pytest.fixture
def kimi_chat():
    return _build_kimi_serving_chat()


@pytest.fixture
def non_kimi_chat():
    return _build_non_kimi_serving_chat()


class TestKimiIsKimiFlag:
    """Test that is_kimi flag is set correctly."""

    def test_kimi_k2_model_type(self, kimi_chat: OpenAIServingChat):
        assert kimi_chat.is_kimi is True
        assert kimi_chat.tool_call_id_type == "kimi_k2"

    def test_non_kimi_model_type(self, non_kimi_chat: OpenAIServingChat):
        assert non_kimi_chat.is_kimi is False
        assert non_kimi_chat.tool_call_id_type == "random"


class TestKimiThinkModeDefaults:
    """Test thinking mode (default): temperature=1.0, top_p=0.95."""

    def test_think_mode_defaults_applied(self, kimi_chat: OpenAIServingChat):
        """When no params set, defaults for think mode are applied."""
        req = ChatCompletionRequest(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "hello"}],
        )
        result = kimi_chat._validate_kimi_params(req)
        assert result is None
        assert req.temperature == 1.0
        assert req.top_p == 0.95

    def test_think_mode_correct_temperature(self, kimi_chat: OpenAIServingChat):
        """Explicit temperature=1.0 in think mode passes validation."""
        req = ChatCompletionRequest(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "hello"}],
            temperature=1.0,
            top_p=0.95,
        )
        result = kimi_chat._validate_kimi_params(req)
        assert result is None

    def test_think_mode_temperature_in_range(self, kimi_chat: OpenAIServingChat):
        """temperature within [0, 1] in think mode passes validation."""
        for temp in [0.0, 0.3, 0.6, 0.8, 1.0]:
            req = ChatCompletionRequest(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": "hello"}],
                temperature=temp,
            )
            result = kimi_chat._validate_kimi_params(req)
            assert result is None, f"temperature={temp} should pass"

    def test_think_mode_temperature_out_of_range(self, kimi_chat: OpenAIServingChat):
        """temperature > 1 in think mode is rejected."""
        req = ChatCompletionRequest(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "hello"}],
            temperature=1.5,
        )
        result = kimi_chat._validate_kimi_params(req)
        assert isinstance(result, ErrorResponse)
        assert "temperature" in result.error.message

    def test_think_mode_enable_thinking_set(self, kimi_chat: OpenAIServingChat):
        """Validation sets enable_thinking=True in chat_template_kwargs."""
        req = ChatCompletionRequest(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "hello"}],
        )
        kimi_chat._validate_kimi_params(req)
        assert req.chat_template_kwargs is not None
        assert req.chat_template_kwargs["enable_thinking"] is True


class TestKimiChatTemplateKwargsThinking:
    """Test thinking mode detection from chat_template_kwargs."""

    def test_chat_template_kwargs_thinking_false(self, kimi_chat: OpenAIServingChat):
        """chat_template_kwargs.thinking=false -> non-think mode."""
        req = ChatCompletionRequest(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "hello"}],
            chat_template_kwargs={"thinking": False},
        )
        result = kimi_chat._validate_kimi_params(req)
        assert result is None
        assert req.temperature == 0.6
        assert req.top_p == 0.95
        assert req.chat_template_kwargs["enable_thinking"] is False

    def test_chat_template_kwargs_thinking_true(self, kimi_chat: OpenAIServingChat):
        """chat_template_kwargs.thinking=true -> think mode."""
        req = ChatCompletionRequest(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "hello"}],
            chat_template_kwargs={"thinking": True},
        )
        result = kimi_chat._validate_kimi_params(req)
        assert result is None
        assert req.temperature == 1.0
        assert req.top_p == 0.95
        assert req.chat_template_kwargs["enable_thinking"] is True

    def test_chat_template_kwargs_enable_thinking_false(
        self, kimi_chat: OpenAIServingChat
    ):
        """chat_template_kwargs.enable_thinking=false -> non-think mode."""
        req = ChatCompletionRequest(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "hello"}],
            chat_template_kwargs={"enable_thinking": False},
        )
        result = kimi_chat._validate_kimi_params(req)
        assert result is None
        assert req.temperature == 0.6
        assert req.top_p == 0.95

    def test_chat_template_kwargs_enable_thinking_true(
        self, kimi_chat: OpenAIServingChat
    ):
        """chat_template_kwargs.enable_thinking=true -> think mode."""
        req = ChatCompletionRequest(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "hello"}],
            chat_template_kwargs={"enable_thinking": True},
        )
        result = kimi_chat._validate_kimi_params(req)
        assert result is None
        assert req.temperature == 1.0
        assert req.top_p == 0.95

    def test_chat_template_kwargs_thinking_false_with_temp_zero(
        self, kimi_chat: OpenAIServingChat
    ):
        """thinking=false + temperature=0 should pass (within [0, 1])."""
        req = ChatCompletionRequest(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "hello"}],
            temperature=0,
            chat_template_kwargs={"thinking": False},
        )
        result = kimi_chat._validate_kimi_params(req)
        assert result is None

    def test_non_think_temperature_in_range(self, kimi_chat: OpenAIServingChat):
        """Any temperature within [0, 1] in non-think mode passes."""
        for temp in [0.0, 0.3, 0.6, 0.8, 1.0]:
            req = ChatCompletionRequest(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": "hello"}],
                temperature=temp,
                top_p=0.95,
                chat_template_kwargs={"thinking": False},
            )
            result = kimi_chat._validate_kimi_params(req)
            assert result is None, f"temperature={temp} should pass"

    def test_non_think_temperature_out_of_range(self, kimi_chat: OpenAIServingChat):
        """temperature > 1 in non-think mode is rejected."""
        req = ChatCompletionRequest(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "hello"}],
            temperature=1.5,
            chat_template_kwargs={"thinking": False},
        )
        result = kimi_chat._validate_kimi_params(req)
        assert isinstance(result, ErrorResponse)
        assert "temperature" in result.error.message

    def test_non_think_enable_thinking_false_sets_kwarg(
        self, kimi_chat: OpenAIServingChat
    ):
        """Validation sets enable_thinking=False in chat_template_kwargs."""
        req = ChatCompletionRequest(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "hello"}],
            chat_template_kwargs={"thinking": False},
        )
        kimi_chat._validate_kimi_params(req)
        assert req.chat_template_kwargs["enable_thinking"] is False


class TestKimiCommonParamValidation:
    """Test common parameter constraints for both modes."""

    def test_wrong_top_p_rejected(self, kimi_chat: OpenAIServingChat):
        req = ChatCompletionRequest(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "hello"}],
            top_p=0.8,
        )
        result = kimi_chat._validate_kimi_params(req)
        assert isinstance(result, ErrorResponse)
        assert "top_p" in result.error.message

    def test_wrong_presence_penalty_rejected(self, kimi_chat: OpenAIServingChat):
        req = ChatCompletionRequest(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "hello"}],
            presence_penalty=1.0,
        )
        result = kimi_chat._validate_kimi_params(req)
        assert isinstance(result, ErrorResponse)
        assert "presence_penalty" in result.error.message

    def test_wrong_frequency_penalty_rejected(self, kimi_chat: OpenAIServingChat):
        req = ChatCompletionRequest(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "hello"}],
            frequency_penalty=0.5,
        )
        result = kimi_chat._validate_kimi_params(req)
        assert isinstance(result, ErrorResponse)
        assert "frequency_penalty" in result.error.message

    def test_wrong_n_rejected(self, kimi_chat: OpenAIServingChat):
        req = ChatCompletionRequest(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "hello"}],
            n=2,
        )
        result = kimi_chat._validate_kimi_params(req)
        assert isinstance(result, ErrorResponse)
        assert "n must be 1" in result.error.message

    def test_multiple_errors_reported(self, kimi_chat: OpenAIServingChat):
        """Multiple invalid params are all reported in one error."""
        req = ChatCompletionRequest(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "hello"}],
            temperature=1.5,
            top_p=0.8,
            n=3,
        )
        result = kimi_chat._validate_kimi_params(req)
        assert isinstance(result, ErrorResponse)
        assert "temperature" in result.error.message
        assert "top_p" in result.error.message
        assert "n must be 1" in result.error.message

    def test_correct_default_params_pass(self, kimi_chat: OpenAIServingChat):
        """Default values for presence_penalty, frequency_penalty, n pass."""
        req = ChatCompletionRequest(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "hello"}],
            temperature=1.0,
            top_p=0.95,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            n=1,
        )
        result = kimi_chat._validate_kimi_params(req)
        assert result is None


class TestNonKimiModelSkipsValidation:
    """Test that non-Kimi models are not affected by Kimi validation."""

    def test_non_kimi_no_validation(self, non_kimi_chat: OpenAIServingChat):
        assert non_kimi_chat.is_kimi is False
        # Non-kimi models don't have _validate_kimi_params called
        # in create_chat_completion, so arbitrary params are fine.
        # Non-kimi models don't call _validate_kimi_params in the flow.
        # The important thing is is_kimi is False.
        assert non_kimi_chat.is_kimi is False
