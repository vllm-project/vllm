# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cross-API HF render-input parity tests.

Drives the real Chat Completions and Responses prep paths and captures the
arguments each passes to ``BaseRenderer.render_chat_async`` (via shared
``OnlineRenderer.preprocess_chat``):

  - conversation messages
  - ChatParams fields that affect the HF template / media path
    (template, content format, template kwargs including tools,
     media_io_kwargs, mm_processor_kwargs)

``TokenizeParams``, ``prompt_extras``, Harmony / GPT-OSS, prefill / continue,
prompt cache salt, and truncation are out of scope for this file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import pytest
from openai.types.shared import Reasoning

from vllm.config.multimodal import MultiModalConfig
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.entrypoints.openai.responses.serving import OpenAIServingResponses
from vllm.inputs import tokens_input
from vllm.renderers.online_renderer import OnlineRenderer
from vllm.renderers.params import ChatParams

_MODEL = "test-model"
_USER = [{"role": "user", "content": "Hello"}]

_WEATHER_PARAMETERS = {
    "type": "object",
    "properties": {"location": {"type": "string"}},
    "required": ["location"],
}

_CHAT_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather",
        "parameters": _WEATHER_PARAMETERS,
    },
}

_RESPONSES_WEATHER_TOOL = {
    "type": "function",
    "name": "get_weather",
    "description": "Get the weather",
    "parameters": _WEATHER_PARAMETERS,
}


@dataclass
class MockHFConfig:
    model_type: str = "llama"


@dataclass
class MockModelConfig:
    task = "generate"
    runner_type = "generate"
    model = _MODEL
    tokenizer = _MODEL
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
    override_generation_config: dict[str, Any] = field(default_factory=dict)
    media_io_kwargs: dict[str, dict[str, Any]] = field(default_factory=dict)
    skip_tokenizer_init = False
    is_encoder_decoder: bool = False
    is_multimodal_model: bool = False
    renderer_num_workers: int = 1
    enable_prompt_embeds: bool = False

    def get_diff_sampling_param(self):
        return self.diff_sampling_param or {}


@dataclass(frozen=True)
class CapturedRenderInputs:
    """Args observed at ``render_chat_async`` (HF render boundary)."""

    messages: list[Any]
    chat_params: ChatParams


class RenderCapture:
    """Install a ``render_chat_async`` stub and record its HF-bound inputs."""

    def __init__(self, online_renderer: OnlineRenderer) -> None:
        self.online_renderer = online_renderer
        self.captured: CapturedRenderInputs | None = None

        async def fake_render_chat_async(
            conversations,
            chat_params,
            tok_params=None,
            *,
            prompt_extras=None,
            skip_mm_cache=False,
        ):
            assert len(conversations) == 1
            self.captured = CapturedRenderInputs(
                messages=list(conversations[0]),
                chat_params=chat_params,
            )
            return [list(conversations[0])], [
                tokens_input(prompt_token_ids=[0]),
            ]

        online_renderer.renderer.render_chat_async = fake_render_chat_async

    def take(self) -> CapturedRenderInputs:
        assert self.captured is not None
        captured = self.captured
        self.captured = None
        return captured


async def _capture_chat(
    online_renderer: OnlineRenderer,
    request: ChatCompletionRequest,
) -> CapturedRenderInputs:
    capture = RenderCapture(online_renderer)
    result = await online_renderer.render_chat(request)
    assert not isinstance(result, ErrorResponse), result
    return capture.take()


async def _capture_responses(
    serving: OpenAIServingResponses,
    request: ResponsesRequest,
) -> CapturedRenderInputs:
    capture = RenderCapture(serving.online_renderer)
    await serving._make_request(request, prev_response=None)
    return capture.take()


async def _assert_parity(
    online_renderer: OnlineRenderer,
    serving: OpenAIServingResponses,
    *,
    chat_kwargs: dict[str, Any],
    responses_kwargs: dict[str, Any],
) -> None:
    """Build paired requests, capture HF render inputs, and assert equality."""
    chat_req = ChatCompletionRequest(model=_MODEL, **chat_kwargs)
    responses_req = ResponsesRequest(model=_MODEL, **responses_kwargs)
    chat = await _capture_chat(online_renderer, chat_req)
    responses = await _capture_responses(serving, responses_req)

    assert chat.messages == responses.messages

    chat_params = chat.chat_params
    responses_params = responses.chat_params
    assert chat_params.chat_template == responses_params.chat_template
    assert (
        chat_params.chat_template_content_format
        == responses_params.chat_template_content_format
    )
    assert chat_params.media_io_kwargs == responses_params.media_io_kwargs
    assert chat_params.mm_processor_kwargs == responses_params.mm_processor_kwargs
    assert dict(chat_params.chat_template_kwargs) == dict(
        responses_params.chat_template_kwargs
    )


def _weather_tools(
    *, overrides: dict[str, Any] | None = None
) -> tuple[list[dict], list[dict]]:
    """Return (chat_tools, responses_tools) for the shared weather function."""
    overrides = overrides or {}
    chat_tool = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather",
            "parameters": _WEATHER_PARAMETERS,
            **overrides,
        },
    }
    responses_tool = {
        "type": "function",
        "name": "get_weather",
        "description": "Get the weather",
        "parameters": _WEATHER_PARAMETERS,
        **overrides,
    }
    return [chat_tool], [responses_tool]


@pytest.fixture
def model_config() -> MockModelConfig:
    return MockModelConfig()


@pytest.fixture
def online_renderer(model_config: MockModelConfig, request) -> OnlineRenderer:
    exclude_tools_when_tool_choice_none = getattr(request, "param", False)
    renderer = MagicMock()
    # Non-Mistral stub; only needed so render_chat / preprocess_chat can
    # inspect tokenizer type before render_chat_async is mocked.
    renderer.tokenizer = MagicMock()

    return OnlineRenderer(
        model_config=model_config,  # type: ignore[arg-type]
        renderer=renderer,
        request_logger=None,
        chat_template=None,
        chat_template_content_format="auto",
        enable_auto_tools=True,
        tool_parser="openai",
        exclude_tools_when_tool_choice_none=exclude_tools_when_tool_choice_none,
    )


@pytest.fixture
def serving_responses(
    model_config: MockModelConfig,
    online_renderer: OnlineRenderer,
) -> OpenAIServingResponses:
    engine_client = MagicMock()
    engine_client.model_config = model_config
    engine_client.renderer = online_renderer.renderer
    engine_client.input_processor = MagicMock()
    engine_client.vllm_config = MagicMock()

    return OpenAIServingResponses(
        engine_client=engine_client,
        models=MagicMock(),
        online_renderer=online_renderer,
        request_logger=None,
        chat_template=online_renderer.chat_template,
        chat_template_content_format=online_renderer.chat_template_content_format,
        enable_auto_tools=True,
        tool_parser="openai",
    )


@pytest.mark.asyncio
class TestConversationRenderParity:
    async def test_multiturn_tool_calling(self, online_renderer, serving_responses):
        """System/instructions, tool-call turn, and a follow-up user message."""
        await _assert_parity(
            online_renderer,
            serving_responses,
            chat_kwargs={
                "messages": [
                    {"role": "system", "content": "Be helpful."},
                    {"role": "user", "content": "Weather in NYC?"},
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location":"NYC"}',
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_1",
                        "content": "72F",
                    },
                    {"role": "user", "content": "Thanks"},
                ],
                "tools": [_CHAT_WEATHER_TOOL],
                "tool_choice": "auto",
            },
            responses_kwargs={
                "instructions": "Be helpful.",
                "input": [
                    {"role": "user", "content": "Weather in NYC?"},
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "get_weather",
                        "arguments": '{"location":"NYC"}',
                    },
                    {
                        "type": "function_call_output",
                        "call_id": "call_1",
                        "output": "72F",
                    },
                    {"role": "user", "content": "Thanks"},
                ],
                "tools": [_RESPONSES_WEATHER_TOOL],
                "tool_choice": "auto",
            },
        )


@pytest.mark.asyncio
class TestToolsRenderParity:
    @pytest.mark.parametrize(
        "chat_tool_choice,responses_tool_choice",
        [
            ("auto", "auto"),
            ("required", "required"),
            (
                {"type": "function", "function": {"name": "get_weather"}},
                {"type": "function", "name": "get_weather"},
            ),
        ],
        ids=["auto", "required", "named"],
    )
    async def test_tools_with_tool_choice(
        self,
        online_renderer,
        serving_responses,
        chat_tool_choice,
        responses_tool_choice,
    ):
        chat_tools, responses_tools = _weather_tools()
        await _assert_parity(
            online_renderer,
            serving_responses,
            chat_kwargs={
                "messages": _USER,
                "tools": chat_tools,
                "tool_choice": chat_tool_choice,
            },
            responses_kwargs={
                "input": _USER,
                "tools": responses_tools,
                "tool_choice": responses_tool_choice,
            },
        )

    @pytest.mark.parametrize(
        "online_renderer",
        [False, True],
        indirect=True,
        ids=["include_tools", "exclude_tools"],
    )
    async def test_tools_with_tool_choice_none(
        self, online_renderer, serving_responses
    ):
        chat_tools, responses_tools = _weather_tools()
        await _assert_parity(
            online_renderer,
            serving_responses,
            chat_kwargs={
                "messages": _USER,
                "tools": chat_tools,
                "tool_choice": "none",
            },
            responses_kwargs={
                "input": _USER,
                "tools": responses_tools,
                "tool_choice": "none",
            },
        )

    async def test_function_tool_optional_fields(
        self, online_renderer, serving_responses
    ):
        """Optional FunctionDefinition fields dump the same on both APIs."""
        chat_tools, responses_tools = _weather_tools(
            overrides={
                "strict": True,
                "defer_loading": False,
                "unrelated_extra": "should_be_ignored",
            }
        )
        await _assert_parity(
            online_renderer,
            serving_responses,
            chat_kwargs={
                "messages": _USER,
                "tools": chat_tools,
                "tool_choice": "auto",
            },
            responses_kwargs={
                "input": _USER,
                "tools": responses_tools,
                "tool_choice": "auto",
            },
        )


@pytest.mark.asyncio
class TestReasoningRenderParity:
    @pytest.mark.parametrize(
        "effort",
        ["none", "minimal", "low", "medium", "high", "xhigh"],
    )
    async def test_reasoning_effort(
        self, online_renderer, serving_responses, effort: str
    ):
        await _assert_parity(
            online_renderer,
            serving_responses,
            chat_kwargs={
                "messages": _USER,
                "reasoning_effort": effort,
                "tool_choice": "none",
            },
            responses_kwargs={
                "input": _USER,
                "reasoning": Reasoning(effort=effort),
                "tool_choice": "none",
            },
        )

    async def test_explicit_enable_thinking_not_overridden(
        self, online_renderer, serving_responses
    ):
        await _assert_parity(
            online_renderer,
            serving_responses,
            chat_kwargs={
                "messages": _USER,
                "reasoning_effort": "high",
                "chat_template_kwargs": {"enable_thinking": False},
                "tool_choice": "none",
            },
            responses_kwargs={
                "input": _USER,
                "reasoning": Reasoning(effort="high"),
                "chat_template_kwargs": {"enable_thinking": False},
                "tool_choice": "none",
            },
        )


@pytest.mark.asyncio
class TestTemplateKwargsRenderParity:
    async def test_passthrough_fields(self, online_renderer, serving_responses):
        """chat_template_kwargs / media_io_kwargs / mm_processor_kwargs parity."""
        await _assert_parity(
            online_renderer,
            serving_responses,
            chat_kwargs={
                "messages": [
                    {"role": "system", "content": "Be helpful."},
                    {"role": "user", "content": "Hello"},
                ],
                "tools": [_CHAT_WEATHER_TOOL],
                "tool_choice": "auto",
                "reasoning_effort": "medium",
                "chat_template_kwargs": {"custom_flag": True},
                "media_io_kwargs": {"image": {"max_pixels": 512}},
                "mm_processor_kwargs": {"num_crops": 2},
            },
            responses_kwargs={
                "instructions": "Be helpful.",
                "input": _USER,
                "tools": [_RESPONSES_WEATHER_TOOL],
                "tool_choice": "auto",
                "reasoning": Reasoning(effort="medium"),
                "chat_template_kwargs": {"custom_flag": True},
                "media_io_kwargs": {"image": {"max_pixels": 512}},
                "mm_processor_kwargs": {"num_crops": 2},
            },
        )
