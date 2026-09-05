# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Report-mode propagation through the serving APIs, without model execution."""

import asyncio
from io import BytesIO
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import UploadFile, WebSocket
from starlette.requests import Request

from vllm.entrypoints.anthropic.protocol import AnthropicMessagesRequest
from vllm.entrypoints.anthropic.serving import AnthropicServingMessages
from vllm.entrypoints.generate.generative_scoring.serving import (
    GenerativeScoringRequest,
    ServingGenerativeScoring,
)
from vllm.entrypoints.openai.chat_completion.batch_serving import OpenAIServingChatBatch
from vllm.entrypoints.openai.chat_completion.protocol import (
    BatchChatCompletionRequest,
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.entrypoints.openai.responses.serving import OpenAIServingResponses
from vllm.entrypoints.scale_out.render.serving import ServingRender
from vllm.entrypoints.scale_out.token_in_token_out.protocol import GenerateRequest
from vllm.entrypoints.scale_out.token_in_token_out.serving import ServingTokens
from vllm.entrypoints.speech_to_text.base.serving import SpeechToTextBaseServing
from vllm.entrypoints.speech_to_text.realtime.connection import RealtimeConnection
from vllm.entrypoints.speech_to_text.transcription.protocol import (
    TranscriptionRequest,
    TranscriptionResponse,
)
from vllm.entrypoints.speech_to_text.translation.protocol import TranslationRequest
from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import SamplingParams

_MESSAGES = [{"role": "user", "content": "hi"}]
_PROMPT = {"type": "token", "prompt_token_ids": [1, 2]}


class _EngineReached(Exception):
    def __init__(self, params):
        self.params = params


def _capture_engine(prompt=None, sampling_params=None, *args, **kwargs):
    raise _EngineReached(sampling_params)


def _serving(cls):
    serving = cls.__new__(cls)
    serving.engine_client = SimpleNamespace(
        generate=_capture_engine, errored=False, check_admission=Mock()
    )
    serving.has_kv_connector = False
    serving.model_config = SimpleNamespace(
        max_model_len=100, is_encoder_decoder=False, is_multimodal_model=False
    )
    serving.models = SimpleNamespace(model_name=lambda *args: "test-model")
    serving.renderer = Mock()
    serving._preflight = Mock()
    serving._check_model = AsyncMock(return_value=None)
    serving._maybe_get_adapters = Mock(return_value=None)
    serving._extract_prompt_len = Mock(return_value=2)
    serving._extract_prompt_components = Mock(
        return_value=SimpleNamespace(token_ids=[1, 2])
    )
    serving._log_inputs = Mock()
    serving._get_trace_headers = AsyncMock(return_value=None)
    serving.default_sampling_params = {}
    serving.override_max_tokens = None
    serving.parser_cls = None
    serving._effective_chat_template_kwargs = Mock(return_value={})
    serving.render_chat_request = AsyncMock(return_value=([], [_PROMPT]))
    serving.render_batch_chat_request = AsyncMock(return_value=([[]], [_PROMPT]))
    serving.render_completion_request = AsyncMock(return_value=[_PROMPT])
    return serving


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "api",
    [
        "chat",
        "completion",
        "responses",
        "batch",
        "anthropic",
        "cohere",
        "generate",
        "transcription",
        "translation",
        "scoring",
    ],
)
@pytest.mark.parametrize("mode", ["full", "incremental", "invalid"])
async def test_kv_cache_report_header_reaches_sampling_params(api, mode):
    raw_request = Request(
        {
            "type": "http",
            "headers": [
                (b"x-kv-cache-report-mode", mode.encode()),
            ],
        }
    )
    if api == "chat":
        serving = _serving(OpenAIServingChat)
        request = ChatCompletionRequest(messages=_MESSAGES)
        call = serving.create_chat_completion(request, raw_request)
    elif api == "completion":
        serving = _serving(OpenAIServingCompletion)
        call = serving.create_completion(CompletionRequest(prompt="hi"), raw_request)
    elif api == "responses":
        serving = _serving(OpenAIServingResponses)
        serving._validate_create_responses_input = Mock(return_value=None)
        serving._validate_generator_input = Mock(return_value=None)
        serving.enable_store = False
        serving.use_harmony = False
        serving.tool_server = None
        serving.parser = None
        serving._make_request = AsyncMock(return_value=([], [_PROMPT]))
        serving._generate_with_builtin_tools = Mock(side_effect=_capture_engine)
        call = serving.create_responses(ResponsesRequest(input="hi"), raw_request)
    elif api == "batch":
        serving = _serving(OpenAIServingChatBatch)
        call = serving.create_batch_chat_completion(
            BatchChatCompletionRequest(messages=[_MESSAGES]), raw_request
        )
    elif api == "anthropic":
        serving = _serving(AnthropicServingMessages)
        serving._merge_inline_system = False
        call = serving.create_messages(
            AnthropicMessagesRequest(
                model="test-model", messages=_MESSAGES, max_tokens=1
            ),
            raw_request,
        )
    elif api == "cohere":
        pytest.importorskip("cohere")
        from vllm.entrypoints.cohere.protocol import CohereChatV2Request
        from vllm.entrypoints.cohere.serving import CohereServingChatV2

        serving = _serving(CohereServingChatV2)
        call = serving.create_chat_v2(
            CohereChatV2Request(model="test-model", messages=_MESSAGES), raw_request
        )
    elif api in ("transcription", "translation"):
        serving = _serving(SpeechToTextBaseServing)
        serving.task_type = "transcribe" if api == "transcription" else "translate"
        serving._preprocess_speech_to_text = AsyncMock(
            return_value=([_PROMPT], 1.0, [0.0])
        )
        request_cls = (
            TranscriptionRequest if api == "transcription" else TranslationRequest
        )
        call = serving._create_speech_to_text(
            b"audio",
            request_cls(file=UploadFile(BytesIO()), model="test-model"),
            raw_request,
            TranscriptionResponse,
            Mock(),
        )
    elif api == "scoring":
        serving = _serving(ServingGenerativeScoring)
        serving.model_config.get_vocab_size = lambda: 16
        serving._build_prompts = AsyncMock(return_value=([_PROMPT], [2]))
        call = serving.create_generative_scoring(
            GenerativeScoringRequest(
                model="test-model", query="hi", items=["hello"], label_token_ids=[3]
            ),
            raw_request,
        )
    else:
        serving = _serving(ServingTokens)
        serving.engine_client.vllm_config = SimpleNamespace(
            scheduler_config=SimpleNamespace(max_num_seqs=16)
        )
        serving.online_renderer = SimpleNamespace(
            preprocess_completion=AsyncMock(return_value=[_PROMPT])
        )
        serving.force_no_detokenize = False
        call = serving.serve_tokens(
            GenerateRequest(
                token_ids=[1, 2], sampling_params=SamplingParams(max_tokens=1)
            ),
            raw_request,
        )

    if mode == "invalid":
        with pytest.raises(VLLMValidationError):
            await call
    else:
        with pytest.raises(_EngineReached) as reached:
            await call
        assert reached.value.params.extra_args == {"kv_cache_report_mode": mode}


@pytest.mark.asyncio
@pytest.mark.parametrize("api", ["chat", "completion", "anthropic"])
async def test_render_preserves_kv_cache_report_header(api):
    serving: Any = ServingRender.__new__(ServingRender)
    serving._check_model = AsyncMock(return_value=None)
    serving.model_config = SimpleNamespace(
        max_model_len=100, is_encoder_decoder=False, is_multimodal_model=False
    )
    serving.online_renderer = SimpleNamespace(
        render_chat=AsyncMock(return_value=([], [_PROMPT])),
        render_completion=AsyncMock(return_value=[_PROMPT]),
    )
    serving._merge_inline_system = False
    serving.default_sampling_params = {}
    serving.override_max_tokens = None
    raw_request = Request(
        {
            "type": "http",
            "headers": [
                (b"x-kv-cache-report-mode", b"full"),
            ],
        }
    )
    if api == "chat":
        rendered = await serving.render_chat_request(
            ChatCompletionRequest(messages=_MESSAGES), raw_request
        )
    elif api == "completion":
        (rendered,) = await serving.render_completion_request(
            CompletionRequest(prompt="hi"), raw_request
        )
    else:
        rendered = await serving.render_messages_request(
            AnthropicMessagesRequest(
                model="test-model", messages=_MESSAGES, max_tokens=1
            ),
            raw_request,
        )
    assert rendered.sampling_params.extra_args == {"kv_cache_report_mode": "full"}


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["full", "invalid"])
async def test_realtime_applies_report_header_to_each_utterance(mode, monkeypatch):
    generate = Mock(return_value=AsyncMock())
    serving = Mock(
        engine_client=SimpleNamespace(generate=generate),
        model_cls=SimpleNamespace(realtime_max_tokens=10),
    )
    websocket = WebSocket(
        {
            "type": "websocket",
            "headers": [
                (b"x-kv-cache-report-mode", mode.encode()),
            ],
        },
        AsyncMock(),
        AsyncMock(),
    )
    connection = RealtimeConnection(websocket, serving)
    send_error = AsyncMock()
    monkeypatch.setattr(connection, "send", AsyncMock())
    monkeypatch.setattr(connection, "send_error", send_error)

    for _ in range(2):
        await connection._run_generation(AsyncMock(), asyncio.Queue())

    if mode == "invalid":
        generate.assert_not_called()
        assert send_error.await_count == 2
    else:
        assert generate.call_count == 2
        for call in generate.call_args_list:
            assert call.kwargs["sampling_params"].extra_args == {
                "kv_cache_report_mode": mode
            }
