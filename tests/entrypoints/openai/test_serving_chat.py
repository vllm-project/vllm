# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from openai import OpenAI

from vllm.config.multimodal import MultiModalConfig
from vllm.entrypoints.openai.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_models import BaseModelPath, OpenAIServingModels
from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm.v1.engine.async_llm import AsyncLLM

from ...utils import RemoteOpenAIServer

GPT_OSS_MODEL_NAME = "openai/gpt-oss-20b"


@pytest.fixture(scope="module")
def monkeypatch_module():
    from _pytest.monkeypatch import MonkeyPatch

    mpatch = MonkeyPatch()
    yield mpatch
    mpatch.undo()


@pytest.fixture(
    scope="module",
    params=[True, False],
    ids=["with_tool_parser", "without_tool_parser"],
)
def with_tool_parser(request) -> bool:
    return request.param


@pytest.fixture(scope="module")
def default_server_args(with_tool_parser: bool):
    args = [
        # use half precision for speed and memory savings in CI environment
        "--enforce-eager",
        "--max-model-len",
        "4096",
        "--reasoning-parser",
        "openai_gptoss",
        "--gpu-memory-utilization",
        "0.8",
    ]
    if with_tool_parser:
        args.extend(
            [
                "--tool-call-parser",
                "openai",
                "--enable-auto-tool-choice",
            ]
        )
    return args


@pytest.fixture(scope="module")
def gptoss_server(
    monkeypatch_module: pytest.MonkeyPatch, default_server_args: list[str]
):
    with monkeypatch_module.context() as m:
        m.setenv("VLLM_ATTENTION_BACKEND", "TRITON_ATTN")
        with RemoteOpenAIServer(
            GPT_OSS_MODEL_NAME, default_server_args
        ) as remote_server:
            yield remote_server


@pytest_asyncio.fixture
async def gptoss_client(gptoss_server):
    async with gptoss_server.get_async_client() as async_client:
        yield async_client


@pytest.mark.asyncio
async def test_gpt_oss_chat_tool_call_streaming(
    gptoss_client: OpenAI, with_tool_parser: bool
):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "state": {"type": "string"},
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                        },
                    },
                    "required": ["city", "state", "unit"],
                },
            },
        }
    ]

    messages = [
        {"role": "user", "content": "What is the weather in Dallas, TX?"},
    ]

    stream = await gptoss_client.chat.completions.create(
        model=GPT_OSS_MODEL_NAME,
        messages=messages,
        tools=tools if with_tool_parser else None,
        stream=True,
    )

    name = None
    args_buf = ""
    content_buf = ""
    async for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.tool_calls:
            tc = delta.tool_calls[0]
            if tc.function and tc.function.name:
                name = tc.function.name
            if tc.function and tc.function.arguments:
                args_buf += tc.function.arguments
        if getattr(delta, "content", None):
            content_buf += delta.content
    if with_tool_parser:
        assert name is not None
        assert len(args_buf) > 0
    else:
        assert name is None
        assert len(args_buf) == 0
        assert len(content_buf) > 0


@pytest.mark.asyncio
async def test_gpt_oss_multi_turn_chat(gptoss_client: OpenAI, with_tool_parser: bool):
    if not with_tool_parser:
        pytest.skip("skip non-tool for multi-turn tests")
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "state": {"type": "string"},
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                        },
                    },
                    "required": ["city", "state", "unit"],
                },
            },
        }
    ]

    messages = [
        {"role": "system", "content": "you are a helpful assistant"},
        {"role": "user", "content": "What is the weather in Dallas, TX with celsius?"},
    ]

    first = await gptoss_client.chat.completions.create(
        model=GPT_OSS_MODEL_NAME,
        messages=messages,
        tools=tools,
        temperature=0.0,
    )
    first_msg = first.choices[0].message
    assert first_msg.tool_calls is not None and len(first_msg.tool_calls) > 0
    tc = first_msg.tool_calls[0]
    assert tc.function is not None and tc.function.name == "get_current_weather"
    args1 = tc.function.arguments
    assert args1 is not None and len(args1) > 0
    assert not first_msg.content

    messages.append({"role": "assistant", "content": args1})
    messages.append(
        {"role": "user", "content": "Now convert to celsius and return JSON only"}
    )

    second = await gptoss_client.chat.completions.create(
        model=GPT_OSS_MODEL_NAME,
        messages=messages,
        tools=tools,
        temperature=0.0,
    )
    second_msg = second.choices[0].message
    assert (second_msg.content is not None and len(second_msg.content) > 0) or (
        second_msg.tool_calls is not None and len(second_msg.tool_calls) > 0
    )


@pytest.mark.asyncio
async def test_gpt_oss_tool_message_array_content(
    gptoss_client: OpenAI, with_tool_parser: bool
):
    """Test that tool messages support both string and array content formats."""
    if not with_tool_parser:
        pytest.skip("skip non-tool for array content tests")

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "state": {"type": "string"},
                    },
                    "required": ["city", "state"],
                },
            },
        }
    ]

    # Test 1: Tool message with string content
    messages_string = [
        {"role": "user", "content": "What's the weather in Paris?"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city": "Paris", "state": "TX"}',
                    },
                }
            ],
        },
        {"role": "tool", "content": "The weather in Paris, TX is sunny, 22°C"},
    ]

    response_string = await gptoss_client.chat.completions.create(
        model=GPT_OSS_MODEL_NAME,
        messages=messages_string,
        tools=tools,
        temperature=0.0,
    )

    assert response_string is not None
    assert response_string.choices[0].message is not None

    # Test 2: Tool message with array content
    messages_array = [
        {"role": "user", "content": "What's the weather in Dallas?"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_456",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city": "Dallas", "state": "TX"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": [
                {"type": "text", "text": "f2e897a7-2705-4337-8193-2a8f57b81618"}
            ],
        },
    ]

    response_array = await gptoss_client.chat.completions.create(
        model=GPT_OSS_MODEL_NAME,
        messages=messages_array,
        tools=tools,
        temperature=0.0,
    )

    assert response_array is not None
    assert response_array.choices[0].message is not None

    # Test 3: Tool message with multiple array content items
    messages_multi_array = [
        {"role": "user", "content": "Search for information"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_789",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city": "Austin", "state": "TX"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": [
                {"type": "text", "text": "Weather data: "},
                {"type": "text", "text": "Austin, TX - Partly cloudy, 25°C"},
                {"type": "text", "text": " with 60% humidity"},
            ],
        },
    ]

    response_multi_array = await gptoss_client.chat.completions.create(
        model=GPT_OSS_MODEL_NAME,
        messages=messages_multi_array,
        tools=tools,
        temperature=0.0,
    )

    assert response_multi_array is not None
    assert response_multi_array.choices[0].message is not None


MODEL_NAME = "openai-community/gpt2"
MODEL_NAME_SHORT = "gpt2"
CHAT_TEMPLATE = "Dummy chat template for testing {}"
BASE_MODEL_PATHS = [
    BaseModelPath(name=MODEL_NAME, model_path=MODEL_NAME),
    BaseModelPath(name=MODEL_NAME_SHORT, model_path=MODEL_NAME_SHORT),
]


@dataclass
class MockHFConfig:
    model_type: str = "any"


@dataclass
class MockModelConfig:
    task = "generate"
    runner_type = "generate"
    tokenizer = MODEL_NAME
    trust_remote_code = False
    tokenizer_mode = "auto"
    max_model_len = 100
    tokenizer_revision = None
    multimodal_config = MultiModalConfig()
    hf_config = MockHFConfig()
    logits_processor_pattern = None
    diff_sampling_param: dict | None = None
    allowed_local_media_path: str = ""
    allowed_media_domains: list[str] | None = None
    encoder_config = None
    generation_config: str = "auto"
    media_io_kwargs: dict[str, dict[str, Any]] = field(default_factory=dict)
    skip_tokenizer_init = False

    def get_diff_sampling_param(self):
        return self.diff_sampling_param or {}


def _build_serving_chat(engine: AsyncLLM) -> OpenAIServingChat:
    models = OpenAIServingModels(
        engine_client=engine,
        base_model_paths=BASE_MODEL_PATHS,
    )
    serving_chat = OpenAIServingChat(
        engine,
        models,
        response_role="assistant",
        chat_template=CHAT_TEMPLATE,
        chat_template_content_format="auto",
        request_logger=None,
    )

    async def _fake_process_inputs(
        request_id,
        engine_prompt,
        sampling_params,
        *,
        lora_request,
        trace_headers,
        priority,
    ):
        return dict(engine_prompt), {}

    serving_chat._process_inputs = AsyncMock(side_effect=_fake_process_inputs)
    return serving_chat


@dataclass
class MockEngine:
    model_config: MockModelConfig = field(default_factory=MockModelConfig)
    processor: MagicMock = field(default_factory=MagicMock)
    io_processor: MagicMock = field(default_factory=MagicMock)


async def _async_serving_chat_init():
    engine = MockEngine()

    models = OpenAIServingModels(engine, BASE_MODEL_PATHS)
    serving_completion = OpenAIServingChat(
        engine,
        models,
        response_role="assistant",
        chat_template=CHAT_TEMPLATE,
        chat_template_content_format="auto",
        request_logger=None,
    )
    return serving_completion


def test_async_serving_chat_init():
    serving_completion = asyncio.run(_async_serving_chat_init())
    assert serving_completion.chat_template == CHAT_TEMPLATE


@pytest.mark.asyncio
async def test_serving_chat_returns_correct_model_name():
    mock_engine = MagicMock(spec=AsyncLLM)
    mock_engine.get_tokenizer.return_value = get_tokenizer(MODEL_NAME)
    mock_engine.errored = False
    mock_engine.model_config = MockModelConfig()
    mock_engine.processor = MagicMock()
    mock_engine.io_processor = MagicMock()

    serving_chat = _build_serving_chat(mock_engine)
    messages = [{"role": "user", "content": "what is 1+1?"}]

    async def return_model_name(*args):
        return args[3]

    serving_chat.chat_completion_full_generator = return_model_name

    # Test that full name is returned when short name is requested
    req = ChatCompletionRequest(model=MODEL_NAME_SHORT, messages=messages)
    assert await serving_chat.create_chat_completion(req) == MODEL_NAME

    # Test that full name is returned when empty string is specified
    req = ChatCompletionRequest(model="", messages=messages)
    assert await serving_chat.create_chat_completion(req) == MODEL_NAME

    # Test that full name is returned when no model is specified
    req = ChatCompletionRequest(messages=messages)
    assert await serving_chat.create_chat_completion(req) == MODEL_NAME


@pytest.mark.asyncio
async def test_serving_chat_should_set_correct_max_tokens():
    mock_engine = MagicMock(spec=AsyncLLM)
    mock_engine.get_tokenizer.return_value = get_tokenizer(MODEL_NAME)
    mock_engine.errored = False
    mock_engine.model_config = MockModelConfig()
    mock_engine.processor = MagicMock()
    mock_engine.io_processor = MagicMock()

    serving_chat = _build_serving_chat(mock_engine)

    req = ChatCompletionRequest(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "what is 1+1?"}],
    )

    with suppress(Exception):
        await serving_chat.create_chat_completion(req)

    assert mock_engine.generate.call_args.args[1].max_tokens == 93

    req.max_tokens = 10
    with suppress(Exception):
        await serving_chat.create_chat_completion(req)

    assert mock_engine.generate.call_args.args[1].max_tokens == 10

    # Setting server's max_tokens in the generation_config.json
    # lower than context_window - prompt_tokens
    mock_model_config = MockModelConfig()
    mock_model_config.diff_sampling_param = {
        "max_tokens": 10  # Setting server-side max_tokens limit
    }

    # Reinitialize the engine with new settings
    mock_engine = MagicMock(spec=AsyncLLM)
    mock_engine.get_tokenizer.return_value = get_tokenizer(MODEL_NAME)
    mock_engine.errored = False
    mock_engine.model_config = mock_model_config
    mock_engine.processor = MagicMock()
    mock_engine.io_processor = MagicMock()

    # Initialize the serving chat
    serving_chat = _build_serving_chat(mock_engine)

    # Test Case 1: No max_tokens specified in request
    req = ChatCompletionRequest(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "what is 1+1?"}],
    )

    with suppress(Exception):
        await serving_chat.create_chat_completion(req)

    assert mock_engine.generate.call_args.args[1].max_tokens == 10

    # Test Case 2: Request's max_tokens set higher than server accepts
    req.max_tokens = 15

    with suppress(Exception):
        await serving_chat.create_chat_completion(req)

    assert mock_engine.generate.call_args.args[1].max_tokens == 10

    # Test Case 3: Request's max_tokens set lower than server accepts
    req.max_tokens = 5

    with suppress(Exception):
        await serving_chat.create_chat_completion(req)

    assert mock_engine.generate.call_args.args[1].max_tokens == 5

    # Setting server's max_tokens in the generation_config.json
    # higher than context_window - prompt_tokens
    mock_model_config = MockModelConfig()
    mock_model_config.diff_sampling_param = {
        "max_tokens": 200  # Setting server-side max_tokens limit
    }

    # Reinitialize the engine with new settings
    mock_engine = MagicMock(spec=AsyncLLM)
    mock_engine.get_tokenizer.return_value = get_tokenizer(MODEL_NAME)
    mock_engine.errored = False
    mock_engine.model_config = mock_model_config
    mock_engine.processor = MagicMock()
    mock_engine.io_processor = MagicMock()

    # Initialize the serving chat
    serving_chat = _build_serving_chat(mock_engine)

    # Test case 1: No max_tokens specified, defaults to context_window
    req = ChatCompletionRequest(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "what is 1+1?"}],
    )

    with suppress(Exception):
        await serving_chat.create_chat_completion(req)

    assert mock_engine.generate.call_args.args[1].max_tokens == 93

    # Test Case 2: Request's max_tokens set higher than server accepts
    req.max_tokens = 100

    with suppress(Exception):
        await serving_chat.create_chat_completion(req)

    assert mock_engine.generate.call_args.args[1].max_tokens == 93

    # Test Case 3: Request's max_tokens set lower than server accepts
    req.max_tokens = 5

    with suppress(Exception):
        await serving_chat.create_chat_completion(req)

    assert mock_engine.generate.call_args.args[1].max_tokens == 5


@pytest.mark.asyncio
async def test_serving_chat_could_load_correct_generation_config():
    mock_model_config = MockModelConfig()
    mock_model_config.diff_sampling_param = {
        "temperature": 0.5,
        "repetition_penalty": 1.05,
    }

    mock_engine = MagicMock(spec=AsyncLLM)
    mock_engine.get_tokenizer.return_value = get_tokenizer(MODEL_NAME)
    mock_engine.errored = False
    mock_engine.model_config = mock_model_config
    mock_engine.processor = MagicMock()
    mock_engine.io_processor = MagicMock()

    # Initialize the serving chat
    serving_chat = _build_serving_chat(mock_engine)

    req = ChatCompletionRequest(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "what is 1+1?"}],
    )

    with suppress(Exception):
        await serving_chat.create_chat_completion(req)

    assert mock_engine.generate.call_args.args[1].temperature == 0.5
    assert mock_engine.generate.call_args.args[1].repetition_penalty == 1.05

    # Test the param when user set it
    req.temperature = 0.1

    with suppress(Exception):
        await serving_chat.create_chat_completion(req)

    assert mock_engine.generate.call_args.args[1].temperature == 0.1
    assert mock_engine.generate.call_args.args[1].repetition_penalty == 1.05

    # Test When temperature==0.0
    req.temperature = 0.0

    with suppress(Exception):
        await serving_chat.create_chat_completion(req)

    assert mock_engine.generate.call_args.args[1].temperature == 0.0
    assert mock_engine.generate.call_args.args[1].repetition_penalty == 1.05


@pytest.mark.parametrize("model_type", ["gpt_oss", "any"])
@pytest.mark.asyncio
async def test_serving_chat_did_set_correct_cache_salt(model_type):
    mock_model_config = MockModelConfig()
    mock_model_config.hf_config.model_type = model_type

    mock_engine = MagicMock(spec=AsyncLLM)
    mock_engine.get_tokenizer.return_value = get_tokenizer(MODEL_NAME)
    mock_engine.errored = False
    mock_engine.model_config = mock_model_config
    mock_engine.processor = MagicMock()
    mock_engine.io_processor = MagicMock()

    serving_chat = _build_serving_chat(mock_engine)

    # Test cache_salt
    req = ChatCompletionRequest(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "what is 1+1?"}],
    )

    # By default, cache_salt in the engine prompt is not set
    with suppress(Exception):
        await serving_chat.create_chat_completion(req)
    engine_prompt = serving_chat._process_inputs.await_args_list[0].args[1]
    assert "cache_salt" not in engine_prompt

    # Test with certain cache_salt
    req.cache_salt = "test_salt"
    with suppress(Exception):
        await serving_chat.create_chat_completion(req)
    engine_prompt = serving_chat._process_inputs.await_args_list[1].args[1]
    assert engine_prompt.get("cache_salt") == "test_salt"
