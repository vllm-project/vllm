# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional
from unittest.mock import MagicMock

import pytest
import pytest_asyncio

from vllm.config.multimodal import MultiModalConfig
from vllm.entrypoints.openai.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_models import (BaseModelPath,
                                                    OpenAIServingModels)
from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm.v1.engine.async_llm import AsyncLLM

from ...utils import RemoteOpenAIServer

if TYPE_CHECKING:
    from openai import OpenAI

GPT_OSS_MODEL_NAME = "openai/gpt-oss-20b"


@pytest.fixture(scope="module")
def monkeypatch_module():
    from _pytest.monkeypatch import MonkeyPatch
    mpatch = MonkeyPatch()
    yield mpatch
    mpatch.undo()


@pytest.fixture(scope="module",
                params=[True, False],
                ids=["with_tool_parser", "without_tool_parser"])
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
        args.extend([
            "--tool-call-parser",
            "openai",
            "--enable-auto-tool-choice",
        ])
    return args


@pytest.fixture(scope="module")
def gptoss_server(monkeypatch_module: pytest.MonkeyPatch,
                  default_server_args: list[str]):
    with monkeypatch_module.context() as m:
        m.setenv("VLLM_ATTENTION_BACKEND", "TRITON_ATTN_VLLM_V1")
        with RemoteOpenAIServer(GPT_OSS_MODEL_NAME,
                                default_server_args) as remote_server:
            yield remote_server


@pytest_asyncio.fixture
async def gptoss_client(gptoss_server):
    async with gptoss_server.get_async_client() as async_client:
        yield async_client


@pytest.mark.asyncio
async def test_gpt_oss_chat_tool_call_streaming(gptoss_client: OpenAI,
                                                with_tool_parser: bool):
    tools = [{
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string"
                    },
                    "state": {
                        "type": "string"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                    },
                },
                "required": ["city", "state", "unit"],
            },
        },
    }]

    messages = [
        {
            "role": "user",
            "content": "What is the weather in Dallas, TX?"
        },
    ]

    stream = await gptoss_client.chat.completions.create(
        model=GPT_OSS_MODEL_NAME,
        messages=messages,
        tools=tools if with_tool_parser else None,
        stream=True)

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
async def test_gpt_oss_multi_turn_chat(gptoss_client: OpenAI,
                                       with_tool_parser: bool):
    if not with_tool_parser:
        pytest.skip("skip non-tool for multi-turn tests")
    tools = [{
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string"
                    },
                    "state": {
                        "type": "string"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                    },
                },
                "required": ["city", "state", "unit"],
            },
        },
    }]

    messages = [
        {
            "role": "system",
            "content": "you are a helpful assistant"
        },
        {
            "role": "user",
            "content": "What is the weather in Dallas, TX with celsius?"
        },
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

    messages.append({"role": "assistant", "content": args1})
    messages.append({
        "role": "user",
        "content": "Now convert to celsius and return JSON only"
    })

    second = await gptoss_client.chat.completions.create(
        model=GPT_OSS_MODEL_NAME,
        messages=messages,
        tools=tools,
        temperature=0.0,
    )
    second_msg = second.choices[0].message
    assert (second_msg.content is not None and len(second_msg.content) > 0) or \
        (second_msg.tool_calls is not None and len(second_msg.tool_calls) > 0)


MODEL_NAME = "openai-community/gpt2"
MODEL_NAME_SHORT = "gpt2"
CHAT_TEMPLATE = "Dummy chat template for testing {}"
BASE_MODEL_PATHS = [
    BaseModelPath(name=MODEL_NAME, model_path=MODEL_NAME),
    BaseModelPath(name=MODEL_NAME_SHORT, model_path=MODEL_NAME_SHORT)
]


@dataclass
class MockHFConfig:
    model_type: str = "any"


@dataclass
class MockModelConfig:
    task = "generate"
    tokenizer = MODEL_NAME
    trust_remote_code = False
    tokenizer_mode = "auto"
    max_model_len = 100
    tokenizer_revision = None
    multimodal_config = MultiModalConfig()
    hf_config = MockHFConfig()
    logits_processor_pattern = None
    diff_sampling_param: Optional[dict] = None
    allowed_local_media_path: str = ""
    encoder_config = None
    generation_config: str = "auto"
    media_io_kwargs: dict[str, dict[str, Any]] = field(default_factory=dict)

    def get_diff_sampling_param(self):
        return self.diff_sampling_param or {}


@dataclass
class MockEngine:

    async def get_model_config(self):
        return MockModelConfig()


async def _async_serving_chat_init():
    engine = MockEngine()
    model_config = await engine.get_model_config()

    models = OpenAIServingModels(engine, model_config, BASE_MODEL_PATHS)
    serving_completion = OpenAIServingChat(engine,
                                           model_config,
                                           models,
                                           response_role="assistant",
                                           chat_template=CHAT_TEMPLATE,
                                           chat_template_content_format="auto",
                                           request_logger=None)
    return serving_completion


def test_async_serving_chat_init():
    serving_completion = asyncio.run(_async_serving_chat_init())
    assert serving_completion.chat_template == CHAT_TEMPLATE


@pytest.mark.asyncio
async def test_serving_chat_returns_correct_model_name():
    mock_engine = MagicMock(spec=AsyncLLM)
    mock_engine.get_tokenizer.return_value = get_tokenizer(MODEL_NAME)
    mock_engine.errored = False

    models = OpenAIServingModels(engine_client=mock_engine,
                                 base_model_paths=BASE_MODEL_PATHS,
                                 model_config=MockModelConfig())
    serving_chat = OpenAIServingChat(mock_engine,
                                     MockModelConfig(),
                                     models,
                                     response_role="assistant",
                                     chat_template=CHAT_TEMPLATE,
                                     chat_template_content_format="auto",
                                     request_logger=None)
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

    models = OpenAIServingModels(engine_client=mock_engine,
                                 base_model_paths=BASE_MODEL_PATHS,
                                 model_config=MockModelConfig())
    serving_chat = OpenAIServingChat(mock_engine,
                                     MockModelConfig(),
                                     models,
                                     response_role="assistant",
                                     chat_template=CHAT_TEMPLATE,
                                     chat_template_content_format="auto",
                                     request_logger=None)

    req = ChatCompletionRequest(
        model=MODEL_NAME,
        messages=[{
            "role": "user",
            "content": "what is 1+1?"
        }],
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

    # Initialize the serving chat
    models = OpenAIServingModels(engine_client=mock_engine,
                                 base_model_paths=BASE_MODEL_PATHS,
                                 model_config=mock_model_config)
    serving_chat = OpenAIServingChat(mock_engine,
                                     mock_model_config,
                                     models,
                                     response_role="assistant",
                                     chat_template=CHAT_TEMPLATE,
                                     chat_template_content_format="auto",
                                     request_logger=None)

    # Test Case 1: No max_tokens specified in request
    req = ChatCompletionRequest(
        model=MODEL_NAME,
        messages=[{
            "role": "user",
            "content": "what is 1+1?"
        }],
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

    # Initialize the serving chat
    models = OpenAIServingModels(engine_client=mock_engine,
                                 base_model_paths=BASE_MODEL_PATHS,
                                 model_config=mock_model_config)
    serving_chat = OpenAIServingChat(mock_engine,
                                     mock_model_config,
                                     models,
                                     response_role="assistant",
                                     chat_template=CHAT_TEMPLATE,
                                     chat_template_content_format="auto",
                                     request_logger=None)

    # Test case 1: No max_tokens specified, defaults to context_window
    req = ChatCompletionRequest(
        model=MODEL_NAME,
        messages=[{
            "role": "user",
            "content": "what is 1+1?"
        }],
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
        "repetition_penalty": 1.05
    }

    mock_engine = MagicMock(spec=AsyncLLM)
    mock_engine.get_tokenizer.return_value = get_tokenizer(MODEL_NAME)
    mock_engine.errored = False

    # Initialize the serving chat
    models = OpenAIServingModels(engine_client=mock_engine,
                                 base_model_paths=BASE_MODEL_PATHS,
                                 model_config=mock_model_config)
    serving_chat = OpenAIServingChat(mock_engine,
                                     mock_model_config,
                                     models,
                                     response_role="assistant",
                                     chat_template=CHAT_TEMPLATE,
                                     chat_template_content_format="auto",
                                     request_logger=None)

    req = ChatCompletionRequest(
        model=MODEL_NAME,
        messages=[{
            "role": "user",
            "content": "what is 1+1?"
        }],
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

    # Initialize the serving chat
    models = OpenAIServingModels(engine_client=mock_engine,
                                 base_model_paths=BASE_MODEL_PATHS,
                                 model_config=mock_model_config)
    serving_chat = OpenAIServingChat(mock_engine,
                                     mock_model_config,
                                     models,
                                     response_role="assistant",
                                     chat_template=CHAT_TEMPLATE,
                                     chat_template_content_format="auto",
                                     request_logger=None)

    # Test cache_salt
    req = ChatCompletionRequest(
        model=MODEL_NAME,
        messages=[{
            "role": "user",
            "content": "what is 1+1?"
        }],
    )

    # By default, cache_salt in the engine prompt is not set
    with suppress(Exception):
        await serving_chat.create_chat_completion(req)
    assert "cache_salt" not in mock_engine.generate.call_args.args[0]

    # Test with certain cache_salt
    req.cache_salt = "test_salt"
    with suppress(Exception):
        await serving_chat.create_chat_completion(req)
    assert mock_engine.generate.call_args.args[0]["cache_salt"] == "test_salt"
