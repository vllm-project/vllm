# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
import types
from unittest.mock import MagicMock

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.engine.protocol import FunctionCall
from vllm.entrypoints.openai.models.serving import (
    BaseModelPath,
    OpenAIServingModels,
)
from vllm.inputs import TokensPrompt
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.v1.engine.async_llm import AsyncLLM

MODEL_NAME = "test-model"
BASE_MODEL_PATHS = [BaseModelPath(name=MODEL_NAME, model_path=MODEL_NAME)]


class MockModelConfig:
    task = "generate"
    runner_type = "generate"
    model = MODEL_NAME
    tokenizer = MODEL_NAME
    trust_remote_code = False
    tokenizer_mode = "auto"
    max_model_len = 100
    tokenizer_revision = None
    generation_config = "auto"
    override_generation_config: dict[str, int] = {}
    hf_config = type("HFConfig", (), {"model_type": "any"})()
    hf_text_config = type("HFTextConfig", (), {"model_type": "any"})()
    is_encoder_decoder = False
    is_multimodal_model = False

    def get_diff_sampling_param(self):
        return {}


def _build_serving_chat(engine: AsyncLLM) -> OpenAIServingChat:
    models = OpenAIServingModels(
        engine_client=engine,
        base_model_paths=BASE_MODEL_PATHS,
    )
    return OpenAIServingChat(
        engine,
        models,
        response_role="assistant",
        openai_serving_render=MagicMock(),
        chat_template="unused",
        chat_template_content_format="auto",
        request_logger=None,
    )


def _stub_mistral_tokenizer(monkeypatch):
    mistral_module = types.ModuleType("vllm.tokenizers.mistral")
    mistral_module.__dict__["MistralTokenizer"] = type("MistralTokenizer", (), {})
    monkeypatch.setitem(sys.modules, "vllm.tokenizers.mistral", mistral_module)


@pytest.mark.asyncio
async def test_non_streaming_chat_uses_unified_parser(monkeypatch):
    tokenizer = MagicMock()
    tokenizer.get_vocab.return_value = {}

    mock_engine = MagicMock(spec=AsyncLLM)
    mock_engine.errored = False
    mock_engine.model_config = MockModelConfig()
    mock_engine.input_processor = MagicMock()
    mock_engine.io_processor = MagicMock()
    mock_engine.renderer = MagicMock(tokenizer=tokenizer)

    async def mock_generate(*args, **kwargs):
        yield RequestOutput(
            request_id="test-request",
            prompt="test prompt",
            prompt_token_ids=[1, 2, 3],
            prompt_logprobs=None,
            outputs=[
                CompletionOutput(
                    index=0,
                    text="ignored by parser",
                    token_ids=[4, 5, 6],
                    cumulative_logprob=0.0,
                    logprobs=None,
                    finish_reason="stop",
                    stop_reason=None,
                )
            ],
            finished=True,
        )

    mock_engine.generate = mock_generate

    serving_chat = _build_serving_chat(mock_engine)
    serving_chat.enable_auto_tools = True
    serving_chat.tool_parser = MagicMock()

    async def render_chat_request(_request):
        return ([], [TokensPrompt(prompt_token_ids=[1, 2, 3], prompt="test prompt")])

    serving_chat.render_chat_request = render_chat_request

    parser_init_kwargs = {}

    class SpyReasoningParser:
        def __init__(self, tokenizer, *args, **kwargs):
            parser_init_kwargs["reasoning_parser_kwargs"] = kwargs

        def is_reasoning_end(self, input_ids):
            return False

        def extract_reasoning(self, model_output, request):
            raise AssertionError(
                "non-streaming chat should use parser.extract_chat_completion_parts"
            )

    class SpyParser:
        def __init__(self, tokenizer, *args, **kwargs):
            parser_init_kwargs["parser_kwargs"] = kwargs

        def extract_chat_completion_parts(
            self,
            *,
            model_output: str,
            request: ChatCompletionRequest,
            enable_auto_tools: bool = False,
        ):
            assert model_output == "ignored by parser"
            assert enable_auto_tools is True
            return (
                "thinking",
                [
                    FunctionCall(
                        name="get_weather",
                        arguments='{"location": "Rome"}',
                    )
                ],
                None,
            )

    serving_chat.reasoning_parser_cls = SpyReasoningParser
    serving_chat.parser_cls = SpyParser

    _stub_mistral_tokenizer(monkeypatch)

    req = ChatCompletionRequest(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "What's the weather?"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                    },
                },
            }
        ],
        tool_choice="auto",
        chat_template_kwargs={"foo": "bar"},
    )

    response = await serving_chat.create_chat_completion(req)

    assert isinstance(response, ChatCompletionResponse)
    assert parser_init_kwargs["reasoning_parser_kwargs"] == {
        "chat_template_kwargs": {"foo": "bar"}
    }
    assert parser_init_kwargs["parser_kwargs"] == {
        "chat_template_kwargs": {"foo": "bar"}
    }

    choice = response.choices[0]
    assert choice.finish_reason == "tool_calls"
    assert choice.index == 0
    assert choice.message.role == "assistant"
    assert choice.message.reasoning == "thinking"
    assert choice.message.content is None
    assert len(choice.message.tool_calls) == 1
    assert choice.message.tool_calls[0].function.name == "get_weather"
    assert choice.message.tool_calls[0].function.arguments == '{"location": "Rome"}'
    assert choice.message.tool_calls[0].type == "function"
    assert choice.message.tool_calls[0].id is not None
    assert response.id.startswith("chatcmpl-")
    assert response.model == MODEL_NAME
    assert response.usage.prompt_tokens == 3
    assert response.usage.completion_tokens == 3
    assert response.usage.total_tokens == 6
    assert response.prompt_token_ids is None
    assert choice.token_ids is None
    assert response.prompt_logprobs is None


@pytest.mark.asyncio
async def test_non_streaming_chat_named_tool_choice_uses_parser_output(monkeypatch):
    tokenizer = MagicMock()
    tokenizer.get_vocab.return_value = {}

    mock_engine = MagicMock(spec=AsyncLLM)
    mock_engine.errored = False
    mock_engine.model_config = MockModelConfig()
    mock_engine.input_processor = MagicMock()
    mock_engine.io_processor = MagicMock()
    mock_engine.renderer = MagicMock(tokenizer=tokenizer)

    async def mock_generate(*args, **kwargs):
        yield RequestOutput(
            request_id="test-request",
            prompt="test prompt",
            prompt_token_ids=[1, 2, 3],
            prompt_logprobs=None,
            outputs=[
                CompletionOutput(
                    index=0,
                    text="ignored by parser",
                    token_ids=[4, 5, 6],
                    cumulative_logprob=0.0,
                    logprobs=None,
                    finish_reason="stop",
                    stop_reason=None,
                )
            ],
            finished=True,
        )

    mock_engine.generate = mock_generate
    serving_chat = _build_serving_chat(mock_engine)

    async def render_chat_request(_request):
        return ([], [TokensPrompt(prompt_token_ids=[1, 2, 3], prompt="test prompt")])

    class SpyParser:
        def __init__(self, tokenizer, *args, **kwargs):
            pass

        def extract_chat_completion_parts(
            self,
            *,
            model_output: str,
            request: ChatCompletionRequest,
            enable_auto_tools: bool = False,
        ):
            return (
                None,
                [FunctionCall(name="get_weather", arguments='{"location": "Rome"}')],
                None,
            )

    serving_chat.render_chat_request = render_chat_request
    serving_chat.parser_cls = SpyParser
    _stub_mistral_tokenizer(monkeypatch)

    req = ChatCompletionRequest(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "What's the weather?"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                    },
                },
            }
        ],
        tool_choice={"type": "function", "function": {"name": "get_weather"}},
    )

    response = await serving_chat.create_chat_completion(req)

    assert isinstance(response, ChatCompletionResponse)
    choice = response.choices[0]
    assert choice.finish_reason == "stop"
    assert choice.message.content == ""
    assert len(choice.message.tool_calls) == 1
    assert choice.message.tool_calls[0].function.name == "get_weather"
    assert choice.message.tool_calls[0].function.arguments == '{"location": "Rome"}'
    assert choice.message.tool_calls[0].id is not None


@pytest.mark.asyncio
async def test_non_streaming_chat_required_tool_choice_uses_parser_output(
    monkeypatch,
):
    tokenizer = MagicMock()
    tokenizer.get_vocab.return_value = {}

    mock_engine = MagicMock(spec=AsyncLLM)
    mock_engine.errored = False
    mock_engine.model_config = MockModelConfig()
    mock_engine.input_processor = MagicMock()
    mock_engine.io_processor = MagicMock()
    mock_engine.renderer = MagicMock(tokenizer=tokenizer)

    async def mock_generate(*args, **kwargs):
        yield RequestOutput(
            request_id="test-request",
            prompt="test prompt",
            prompt_token_ids=[1, 2, 3],
            prompt_logprobs=None,
            outputs=[
                CompletionOutput(
                    index=0,
                    text="ignored by parser",
                    token_ids=[4, 5, 6],
                    cumulative_logprob=0.0,
                    logprobs=None,
                    finish_reason="stop",
                    stop_reason=None,
                )
            ],
            finished=True,
        )

    mock_engine.generate = mock_generate
    serving_chat = _build_serving_chat(mock_engine)

    async def render_chat_request(_request):
        return ([], [TokensPrompt(prompt_token_ids=[1, 2, 3], prompt="test prompt")])

    class SpyParser:
        def __init__(self, tokenizer, *args, **kwargs):
            pass

        def extract_chat_completion_parts(
            self,
            *,
            model_output: str,
            request: ChatCompletionRequest,
            enable_auto_tools: bool = False,
        ):
            return (
                None,
                [FunctionCall(name="get_weather", arguments='{"location": "Rome"}')],
                None,
            )

    serving_chat.render_chat_request = render_chat_request
    serving_chat.parser_cls = SpyParser
    _stub_mistral_tokenizer(monkeypatch)

    req = ChatCompletionRequest(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "What's the weather?"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                    },
                },
            }
        ],
        tool_choice="required",
    )

    response = await serving_chat.create_chat_completion(req)

    assert isinstance(response, ChatCompletionResponse)
    choice = response.choices[0]
    assert choice.finish_reason == "tool_calls"
    assert choice.message.content == ""
    assert len(choice.message.tool_calls) == 1
    assert choice.message.tool_calls[0].function.name == "get_weather"
    assert choice.message.tool_calls[0].function.arguments == '{"location": "Rome"}'
    assert choice.message.tool_calls[0].id is not None


@pytest.mark.asyncio
async def test_non_streaming_chat_named_tool_choice_without_parser_still_wraps_tool(
    monkeypatch,
):
    tokenizer = MagicMock()
    tokenizer.get_vocab.return_value = {}

    mock_engine = MagicMock(spec=AsyncLLM)
    mock_engine.errored = False
    mock_engine.model_config = MockModelConfig()
    mock_engine.input_processor = MagicMock()
    mock_engine.io_processor = MagicMock()
    mock_engine.renderer = MagicMock(tokenizer=tokenizer)

    async def mock_generate(*args, **kwargs):
        yield RequestOutput(
            request_id="test-request",
            prompt="test prompt",
            prompt_token_ids=[1, 2, 3],
            prompt_logprobs=None,
            outputs=[
                CompletionOutput(
                    index=0,
                    text='{"location": "Rome"}',
                    token_ids=[4, 5, 6],
                    cumulative_logprob=0.0,
                    logprobs=None,
                    finish_reason="stop",
                    stop_reason=None,
                )
            ],
            finished=True,
        )

    mock_engine.generate = mock_generate
    serving_chat = _build_serving_chat(mock_engine)
    serving_chat.reasoning_parser_cls = None
    serving_chat.parser_cls = None

    async def render_chat_request(_request):
        return ([], [TokensPrompt(prompt_token_ids=[1, 2, 3], prompt="test prompt")])

    serving_chat.render_chat_request = render_chat_request
    _stub_mistral_tokenizer(monkeypatch)

    req = ChatCompletionRequest(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "What's the weather?"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                    },
                },
            }
        ],
        tool_choice={"type": "function", "function": {"name": "get_weather"}},
    )

    response = await serving_chat.create_chat_completion(req)

    assert isinstance(response, ChatCompletionResponse)
    choice = response.choices[0]
    assert choice.finish_reason == "stop"
    assert choice.message.content == ""
    assert len(choice.message.tool_calls) == 1
    assert choice.message.tool_calls[0].function.name == "get_weather"
    assert choice.message.tool_calls[0].function.arguments == '{"location": "Rome"}'
    assert choice.message.tool_calls[0].id is not None


@pytest.mark.asyncio
async def test_required_tool_choice_without_parser_still_parses_tools(
    monkeypatch,
):
    tokenizer = MagicMock()
    tokenizer.get_vocab.return_value = {}

    mock_engine = MagicMock(spec=AsyncLLM)
    mock_engine.errored = False
    mock_engine.model_config = MockModelConfig()
    mock_engine.input_processor = MagicMock()
    mock_engine.io_processor = MagicMock()
    mock_engine.renderer = MagicMock(tokenizer=tokenizer)

    async def mock_generate(*args, **kwargs):
        yield RequestOutput(
            request_id="test-request",
            prompt="test prompt",
            prompt_token_ids=[1, 2, 3],
            prompt_logprobs=None,
            outputs=[
                CompletionOutput(
                    index=0,
                    text=('[{"name":"get_weather","parameters":{"location":"Rome"}}]'),
                    token_ids=[4, 5, 6],
                    cumulative_logprob=0.0,
                    logprobs=None,
                    finish_reason="stop",
                    stop_reason=None,
                )
            ],
            finished=True,
        )

    mock_engine.generate = mock_generate
    serving_chat = _build_serving_chat(mock_engine)
    serving_chat.reasoning_parser_cls = None
    serving_chat.parser_cls = None

    async def render_chat_request(_request):
        return ([], [TokensPrompt(prompt_token_ids=[1, 2, 3], prompt="test prompt")])

    serving_chat.render_chat_request = render_chat_request
    _stub_mistral_tokenizer(monkeypatch)

    req = ChatCompletionRequest(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "What's the weather?"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                    },
                },
            }
        ],
        tool_choice="required",
    )

    response = await serving_chat.create_chat_completion(req)

    assert isinstance(response, ChatCompletionResponse)
    choice = response.choices[0]
    assert choice.finish_reason == "tool_calls"
    assert choice.message.content == ""
    assert len(choice.message.tool_calls) == 1
    assert choice.message.tool_calls[0].function.name == "get_weather"
    assert choice.message.tool_calls[0].function.arguments == '{"location": "Rome"}'
    assert choice.message.tool_calls[0].id is not None
