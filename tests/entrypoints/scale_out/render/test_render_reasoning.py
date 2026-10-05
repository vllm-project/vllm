# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for render → generate reasoning field wiring (#60059)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.scale_out.render.serving import (
    ServingRender,
    resolve_generate_reasoning_fields,
)
from vllm.entrypoints.scale_out.token_in_token_out.protocol import GenerateRequest
from vllm.sampling_params import SamplingParams


def _chat_request(**kwargs) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
        **kwargs,
    )


def _online_renderer(*, parser_cls=None, chat_template_kwargs=None):
    online = MagicMock()
    online.chat_template = None
    online.chat_template_content_format = "auto"
    online.default_chat_template_kwargs = chat_template_kwargs or {}
    online.parser = parser_cls
    online.renderer.tokenizer = object()
    return online


@pytest.mark.skip_global_cleanup
def test_include_reasoning_false_sets_reasoning_ended_true():
    request = _chat_request(include_reasoning=False)
    ended, kwargs = resolve_generate_reasoning_fields(
        request,
        [1, 2, 3],
        online_renderer=_online_renderer(),
        model_config=MagicMock(),
    )
    assert ended is True
    assert kwargs is None


@pytest.mark.skip_global_cleanup
def test_grammar_from_parser_sets_reasoning_ended_true():
    request = _chat_request(include_reasoning=True)
    request._grammar_from_parser = True
    ended, kwargs = resolve_generate_reasoning_fields(
        request,
        [1, 2, 3],
        online_renderer=_online_renderer(),
        model_config=MagicMock(),
    )
    assert ended is True
    assert kwargs is None


@pytest.mark.skip_global_cleanup
def test_no_parser_leaves_reasoning_fields_unset():
    request = _chat_request(include_reasoning=True)
    ended, kwargs = resolve_generate_reasoning_fields(
        request,
        [1, 2, 3],
        online_renderer=_online_renderer(parser_cls=None),
        model_config=MagicMock(),
    )
    assert ended is None
    assert kwargs is None


@pytest.mark.skip_global_cleanup
def test_reasoning_parser_checks_prompt_and_forwards_template_kwargs():
    request = _chat_request(
        include_reasoning=True,
        chat_template_kwargs={"enable_thinking": False},
    )
    parser_instance = MagicMock()
    parser_instance.reasoning_parser = object()
    parser_instance.is_reasoning_end.return_value = False

    parser_cls = MagicMock(return_value=parser_instance)
    online = _online_renderer(
        parser_cls=parser_cls,
        chat_template_kwargs={"enable_thinking": True},
    )
    model_config = MagicMock()

    ended, kwargs = resolve_generate_reasoning_fields(
        request,
        [10, 20],
        online_renderer=online,
        model_config=model_config,
    )

    assert ended is False
    assert kwargs is not None
    assert kwargs["chat_template_kwargs"]["enable_thinking"] is False
    parser_cls.assert_called_once()
    parser_instance.is_reasoning_end.assert_called_once_with([10, 20])


@pytest.mark.asyncio
@pytest.mark.skip_global_cleanup
async def test_render_chat_request_populates_reasoning_fields():
    serving = ServingRender.__new__(ServingRender)
    serving.model_config = SimpleNamespace(
        max_model_len=100,
        is_encoder_decoder=False,
    )
    serving.default_sampling_params = {}
    serving.override_max_tokens = None
    serving.tool_server = None
    serving.online_renderer = _online_renderer()
    serving.online_renderer.render_chat = AsyncMock(
        return_value=([], [{"prompt_token_ids": [1, 2, 3]}])
    )
    serving._check_model = AsyncMock(return_value=None)
    serving._extract_mm_features = MagicMock(return_value=None)

    request = _chat_request(include_reasoning=False, max_tokens=8)
    response = await serving.render_chat_request(request)

    assert isinstance(response, GenerateRequest)
    assert response.token_ids == [1, 2, 3]
    assert response.reasoning_ended is True
    assert response.reasoning_parser_kwargs is None


@pytest.mark.asyncio
@pytest.mark.skip_global_cleanup
async def test_serve_tokens_forwards_reasoning_fields_to_engine():
    from vllm.entrypoints.scale_out.token_in_token_out.serving import ServingTokens

    serving = ServingTokens.__new__(ServingTokens)
    serving.engine_client = MagicMock()
    serving.engine_client.vllm_config.scheduler_config.max_num_seqs = 16
    serving.engine_client.generate = MagicMock(return_value=AsyncMock())
    serving.models = MagicMock()
    serving.models.model_name.return_value = "test-model"
    serving.model_config = SimpleNamespace(
        max_model_len=100,
        is_encoder_decoder=False,
    )
    serving.default_sampling_params = {}
    serving.override_max_tokens = None
    serving.force_no_detokenize = False
    serving.has_tokenizer = True
    serving.online_renderer = MagicMock()
    serving.online_renderer.preprocess_completion = AsyncMock(
        return_value=[{"prompt_token_ids": [1, 2, 3]}]
    )
    serving._check_model = AsyncMock(return_value=None)
    serving._preflight = MagicMock()
    serving._maybe_get_adapters = MagicMock(return_value=None)
    serving._base_request_id = MagicMock(return_value="abc")
    serving._log_inputs = MagicMock()
    serving._get_trace_headers = AsyncMock(return_value=None)
    serving._get_data_parallel_rank = MagicMock(return_value=None)
    serving._get_session_id_from_headers = MagicMock(return_value=None)
    serving._extract_prompt_len = MagicMock(return_value=3)
    serving.serve_tokens_full_generator = AsyncMock(return_value=MagicMock())

    request = GenerateRequest(
        token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=4),
        reasoning_ended=True,
        reasoning_parser_kwargs={"chat_template_kwargs": {"enable_thinking": False}},
    )

    await serving.serve_tokens(request)

    serving.engine_client.generate.assert_called_once()
    kwargs = serving.engine_client.generate.call_args.kwargs
    assert kwargs["reasoning_ended"] is True
    assert kwargs["reasoning_parser_kwargs"] == {
        "chat_template_kwargs": {"enable_thinking": False},
    }
