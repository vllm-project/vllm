# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for token offsets surfacing via render endpoints."""

from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from vllm.config import ModelConfig
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.openai.models.protocol import BaseModelPath
from vllm.entrypoints.openai.models.serving import OpenAIModelRegistry
from vllm.entrypoints.serve.disagg.protocol import GenerateRequest
from vllm.entrypoints.serve.render.serving import OpenAIServingRender
from vllm.sampling_params import SamplingParams


class TestCompletionRequestField:
    def test_default_is_false(self):
        """return_token_offsets must default to False so existing
        callers see zero behavioral change."""
        req = CompletionRequest(model="m", prompt="hi")
        assert req.return_token_offsets is False

    def test_accepts_true(self):
        req = CompletionRequest(model="m", prompt="hi", return_token_offsets=True)
        assert req.return_token_offsets is True

    def test_none_coerces_to_false_in_tok_params(self):
        """JSON null must coerce to False when forwarded into TokenizeParams."""
        req = CompletionRequest(model="m", prompt="hi", return_token_offsets=None)
        model_config = Mock(spec=ModelConfig)
        model_config.max_model_len = 128
        params = req.build_tok_params(model_config)
        assert params.return_token_offsets is False

    def test_build_tok_params_forwards_true(self):
        req = CompletionRequest(model="m", prompt="hi", return_token_offsets=True)
        model_config = Mock(spec=ModelConfig)
        model_config.max_model_len = 128
        params = req.build_tok_params(model_config)
        assert params.return_token_offsets is True

    def test_build_tok_params_default_is_false(self):
        req = CompletionRequest(model="m", prompt="hi")
        model_config = Mock(spec=ModelConfig)
        model_config.max_model_len = 128
        params = req.build_tok_params(model_config)
        assert params.return_token_offsets is False


class TestChatCompletionRequestField:
    def test_default_is_false(self):
        req = ChatCompletionRequest(
            model="m", messages=[{"role": "user", "content": "hi"}]
        )
        assert req.return_token_offsets is False

    def test_accepts_true(self):
        req = ChatCompletionRequest(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            return_token_offsets=True,
        )
        assert req.return_token_offsets is True

    def test_none_coerces_to_false_in_tok_params(self):
        req = ChatCompletionRequest(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            return_token_offsets=None,
        )
        model_config = Mock(spec=ModelConfig)
        model_config.max_model_len = 128
        params = req.build_tok_params(model_config)
        assert params.return_token_offsets is False

    def test_build_tok_params_forwards_true(self):
        req = ChatCompletionRequest(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            return_token_offsets=True,
        )
        model_config = Mock(spec=ModelConfig)
        model_config.max_model_len = 128
        params = req.build_tok_params(model_config)
        assert params.return_token_offsets is True

    def test_build_tok_params_default_is_false(self):
        req = ChatCompletionRequest(
            model="m", messages=[{"role": "user", "content": "hi"}]
        )
        model_config = Mock(spec=ModelConfig)
        model_config.max_model_len = 128
        params = req.build_tok_params(model_config)
        assert params.return_token_offsets is False


class TestGenerateRequestField:
    def test_default_is_none(self):
        """token_offsets must default to None so existing /v1/.../render
        responses are byte-identical (modulo new key emission)."""
        req = GenerateRequest(
            token_ids=[1, 2, 3],
            sampling_params=SamplingParams(),
        )
        assert req.token_offsets is None

    def test_accepts_offsets_list(self):
        req = GenerateRequest(
            token_ids=[10, 20],
            sampling_params=SamplingParams(),
            token_offsets=[(0, 1), (1, 3)],
        )
        assert req.token_offsets == [(0, 1), (1, 3)]

    def test_offsets_serialize_to_json(self):
        """Pydantic v2 round-trip: tuple[int, int] elements survive
        model_dump and re-validate."""
        req = GenerateRequest(
            token_ids=[10, 20],
            sampling_params=SamplingParams(),
            token_offsets=[(0, 1), (1, 3)],
        )
        dumped = req.model_dump()
        assert dumped["token_offsets"] == [(0, 1), (1, 3)]
        # Re-validate from the dumped dict (excluding sampling_params
        # which doesn't round-trip cleanly via dump).
        again = GenerateRequest.model_validate(
            {
                **dumped,
                "sampling_params": SamplingParams(),
            }
        )
        assert again.token_offsets == [(0, 1), (1, 3)]


@pytest.fixture
def mock_model_config():
    """Reusable lightweight ModelConfig stub."""
    mc = MagicMock()
    mc.max_model_len = 1024
    mc.multimodal_config = None
    mc.is_multimodal_model = False
    mc.get_diff_sampling_param.return_value = {}
    return mc


@pytest.fixture
def render_handler(mock_model_config):
    """Construct an OpenAIServingRender with mocked dependencies."""
    renderer = MagicMock()
    model_registry = MagicMock(spec=OpenAIModelRegistry)
    model_registry.base_model_paths = [
        BaseModelPath(name="test-model", model_path="test-model"),
    ]
    model_registry.lora_requests = {}
    model_registry.prompt_adapter_requests = {}

    handler = OpenAIServingRender(
        model_config=mock_model_config,
        renderer=renderer,
        model_registry=model_registry,
        request_logger=None,
        chat_template=None,
        chat_template_content_format="auto",
        trust_request_chat_template=False,
        enable_auto_tools=False,
        exclude_tools_when_tool_choice_none=False,
        tool_parser=None,
        reasoning_parser=None,
        default_chat_template_kwargs=None,
        log_error_stack=False,
    )
    # _check_model passes through (no actual model lookup needed)
    handler._check_model = AsyncMock(return_value=None)
    return handler


def _make_engine_input(token_ids, *, offsets=None, include_offsets_key=True):
    """Build a TokensInput-shaped dict for mocking.

    include_offsets_key=False simulates the renderer's behavior when offsets
    were not computed (key is absent, not None).
    """
    ei = {
        "type": "token",
        "prompt_token_ids": list(token_ids),
        "prompt": "ignored",
    }
    if include_offsets_key:
        ei["prompt_token_offsets"] = offsets
    return ei


class TestRenderCompletionSurfacesOffsets:
    @pytest.mark.asyncio
    async def test_flag_with_offsets_surfaces_in_response(
        self, render_handler, mock_model_config
    ):
        offsets = [(0, 5), (5, 6), (6, 12), (12, 13)]
        engine_input = _make_engine_input([15496, 11, 995, 13], offsets=offsets)
        render_handler.render_completion = AsyncMock(return_value=[engine_input])

        req = CompletionRequest(
            model="test-model",
            prompt="Hello, world.",
            return_token_offsets=True,
        )
        result = await render_handler.render_completion_request(req)

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].token_offsets == offsets
        assert len(result[0].token_offsets) == len(result[0].token_ids)

    @pytest.mark.asyncio
    async def test_default_flag_yields_null_offsets(
        self, render_handler, mock_model_config
    ):
        # Even when engine_input contains offsets, response.token_offsets
        # depends ONLY on what engine_input.get returns; the renderer
        # would not have populated it when the flag is off. Simulate
        # that by setting include_offsets_key=False.
        engine_input = _make_engine_input(
            [15496, 11, 995, 13], include_offsets_key=False
        )
        render_handler.render_completion = AsyncMock(return_value=[engine_input])

        req = CompletionRequest(model="test-model", prompt="Hello, world.")
        result = await render_handler.render_completion_request(req)

        assert result[0].token_offsets is None

    @pytest.mark.asyncio
    async def test_flag_with_missing_offsets_yields_null(
        self, render_handler, mock_model_config
    ):
        """Simulates renderer's slow-tokenizer / MM / pre-tokenized path
        where _wants_offsets returned False so the key is absent."""
        engine_input = _make_engine_input(
            [15496, 11, 995, 13], include_offsets_key=False
        )
        render_handler.render_completion = AsyncMock(return_value=[engine_input])

        req = CompletionRequest(
            model="test-model",
            prompt="Hello, world.",
            return_token_offsets=True,
        )
        result = await render_handler.render_completion_request(req)

        assert result[0].token_offsets is None

    @pytest.mark.asyncio
    async def test_multi_prompt_batch_surfaces_per_prompt_offsets(
        self, render_handler, mock_model_config
    ):
        ei1 = _make_engine_input([15496, 11], offsets=[(0, 5), (5, 6)])
        ei2 = _make_engine_input([4944, 13], offsets=[(0, 4), (4, 5)])
        render_handler.render_completion = AsyncMock(return_value=[ei1, ei2])

        req = CompletionRequest(
            model="test-model",
            prompt=["Hello,", "Sure."],
            return_token_offsets=True,
        )
        result = await render_handler.render_completion_request(req)

        assert len(result) == 2
        assert result[0].token_offsets == [(0, 5), (5, 6)]
        assert result[1].token_offsets == [(0, 4), (4, 5)]
