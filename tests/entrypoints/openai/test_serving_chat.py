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

from vllm._aiter_ops import is_aiter_found_and_supported
from vllm.config import MultiModalConfig
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.engine.protocol import (
    ErrorResponse,
    RequestResponseMetadata,
)
from vllm.entrypoints.openai.models.serving import BaseModelPath, OpenAIServingModels
from vllm.entrypoints.openai.parser.harmony_utils import get_encoding
from vllm.inputs import TokensPrompt
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.renderers.hf import HfRenderer
from vllm.renderers.mistral import MistralRenderer
from vllm.tokenizers import get_tokenizer
from vllm.tokenizers.mistral import MistralTokenizer
from vllm.tokenizers.registry import tokenizer_args_from_config
from vllm.tool_parsers import ToolParserManager
from vllm.v1.engine.async_llm import AsyncLLM

from ...utils import RemoteOpenAIServer
from .utils import (
    accumulate_streaming_response,
    verify_chat_response,
    verify_harmony_messages,
)

GPT_OSS_MODEL_NAME = "openai/gpt-oss-20b"
GPT_OSS_SPECULATOR_NAME = "RedHatAI/gpt-oss-20b-speculator.eagle3"


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


@pytest.fixture(
    scope="module",
    params=[True],
    ids=["exclude_tools_when_tool_choice_none"],
)
def exclude_tools_when_tool_choice_none(request) -> bool:
    return request.param


@pytest.fixture(scope="module")
def default_server_args(
    with_tool_parser: bool,
    exclude_tools_when_tool_choice_none: bool,
):
    args = [
        # use half precision for speed and memory savings in CI environment
        "--enforce-eager",
        "--max-model-len",
        "4096",
        "--reasoning-parser",
        "openai_gptoss",
        "--gpu-memory-utilization",
        "0.85",
    ]
    if with_tool_parser:
        args.extend(
            [
                "--tool-call-parser",
                "openai",
                "--enable-auto-tool-choice",
            ]
        )
    if exclude_tools_when_tool_choice_none:
        args.append("--exclude-tools-when-tool-choice-none")
    return args


@pytest.fixture(scope="class")
def gptoss_server(default_server_args: list[str]):
    server_args = default_server_args + ["--attention-backend=TRITON_ATTN"]
    with RemoteOpenAIServer(GPT_OSS_MODEL_NAME, server_args) as remote_server:
        yield remote_server


@pytest.fixture(scope="class")
def gptoss_speculative_server(default_server_args: list[str]):
    attention_backend = (
        "TRITON_ATTN"
        if not is_aiter_found_and_supported()
        else "ROCM_AITER_UNIFIED_ATTN"
    )
    server_args = default_server_args + [
        "--speculative-config",
        f'{{"model": "{GPT_OSS_SPECULATOR_NAME}", '
        f'"method": "eagle3", "num_speculative_tokens": 3}}',
        f"--attention-backend={attention_backend}",
    ]
    # gpt-oss requires AITER unified attention on ROCm
    # TODO: Remove after fixing TRITON_ATTN issue on ROCm
    # https://github.com/vllm-project/vllm/issues/32434
    env_dict = None
    if is_aiter_found_and_supported():
        env_dict = {"VLLM_ROCM_USE_AITER": "1"}
    with RemoteOpenAIServer(
        GPT_OSS_MODEL_NAME, server_args, env_dict=env_dict
    ) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def gptoss_client(gptoss_server):
    async with gptoss_server.get_async_client() as async_client:
        yield async_client


@pytest_asyncio.fixture
async def gptoss_speculative_client(gptoss_speculative_server):
    async with gptoss_speculative_server.get_async_client() as async_client:
        yield async_client


class TestGPTOSSChat:
    @pytest.mark.asyncio
    async def test_gpt_oss_chat_tool_call_streaming(
        self, gptoss_client: OpenAI, with_tool_parser: bool
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
    async def test_gpt_oss_multi_turn_chat(
        self, gptoss_client: OpenAI, with_tool_parser: bool
    ):
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
            {
                "role": "user",
                "content": "What is the weather in Dallas, TX with celsius?",
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
        self, gptoss_client: OpenAI, with_tool_parser: bool
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

    @pytest.mark.asyncio
    async def test_gpt_oss_tool_choice_none(
        self,
        gptoss_client: OpenAI,
        with_tool_parser: bool,
        exclude_tools_when_tool_choice_none: bool,
    ):
        if not (with_tool_parser and exclude_tools_when_tool_choice_none):
            pytest.skip(
                "skip tool_choice tests when non-tool or "
                "--exclude-tools-when-tool-choice-none not set"
            )

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
            {
                "role": "user",
                "content": "What's the temperature(in degrees Celsius) in Dallas?",
            },
        ]

        tool_choice_auto = await gptoss_client.chat.completions.create(
            model=GPT_OSS_MODEL_NAME,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.0,
        )
        msg = tool_choice_auto.choices[0].message
        assert len(msg.tool_calls) == 1

        tool_choice_none = await gptoss_client.chat.completions.create(
            model=GPT_OSS_MODEL_NAME,
            messages=messages,
            tools=tools,
            tool_choice="none",
            temperature=0.0,
        )

        msg = tool_choice_none.choices[0].message
        assert len(msg.tool_calls) == 0


class TestGPTOSSSpeculativeChat:
    @pytest.mark.asyncio
    async def test_gpt_oss_speculative_reasoning_leakage(
        self,
        gptoss_speculative_client: OpenAI,
        with_tool_parser: bool,
    ):
        if not with_tool_parser:
            pytest.skip("skip non-tool for array content tests")

        messages = [
            {"role": "user", "content": "Calculate 2+2. Return the answer 4 only."},
        ]

        stream = await gptoss_speculative_client.chat.completions.create(
            model=GPT_OSS_MODEL_NAME,
            messages=messages,
            stream=True,
            temperature=0.0,
        )

        content = ""
        reasoning_content = ""
        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                content += delta.content

            chunk_reasoning = getattr(delta, "reasoning", None)
            if chunk_reasoning:
                reasoning_content += delta.reasoning

        assert len(reasoning_content) > 0, "No reasoning was generated."
        assert content.strip() == "4"


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
    hf_text_config = MockHFConfig()
    logits_processors: list[str] | None = None
    logits_processor_pattern = None
    diff_sampling_param: dict | None = None
    allowed_local_media_path: str = ""
    allowed_media_domains: list[str] | None = None
    encoder_config = None
    generation_config: str = "auto"
    media_io_kwargs: dict[str, dict[str, Any]] = field(default_factory=dict)
    skip_tokenizer_init: bool = False

    def get_diff_sampling_param(self):
        return self.diff_sampling_param or {}


def _build_renderer(model_config: MockModelConfig):
    _, tokenizer_name, _, kwargs = tokenizer_args_from_config(model_config)

    return HfRenderer(
        model_config,
        tokenizer_kwargs={**kwargs, "tokenizer_name": tokenizer_name},
    )


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

    return serving_chat


@dataclass
class MockEngine:
    model_config: MockModelConfig = field(default_factory=MockModelConfig)
    input_processor: MagicMock = field(default_factory=MagicMock)
    io_processor: MagicMock = field(default_factory=MagicMock)
    renderer: MagicMock = field(default_factory=MagicMock)


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
    mock_engine.errored = False
    mock_engine.model_config = MockModelConfig()
    mock_engine.input_processor = MagicMock()
    mock_engine.io_processor = MagicMock()
    mock_engine.renderer = _build_renderer(mock_engine.model_config)

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
    mock_engine.errored = False
    mock_engine.model_config = MockModelConfig()
    mock_engine.input_processor = MagicMock()
    mock_engine.io_processor = MagicMock()
    mock_engine.renderer = _build_renderer(mock_engine.model_config)

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
    mock_engine.errored = False
    mock_engine.model_config = mock_model_config
    mock_engine.input_processor = MagicMock()
    mock_engine.io_processor = MagicMock()
    mock_engine.renderer = _build_renderer(mock_engine.model_config)

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
    mock_engine.errored = False
    mock_engine.model_config = mock_model_config
    mock_engine.input_processor = MagicMock()
    mock_engine.io_processor = MagicMock()
    mock_engine.renderer = _build_renderer(mock_engine.model_config)

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
async def test_serving_chat_mistral_token_ids_prompt_is_validated():
    """Regression test: when the Mistral tokenizer path returns token IDs
    directly, we must still apply input length + max_tokens validation.
    """

    mock_engine = MagicMock(spec=AsyncLLM)
    mock_engine.errored = False
    mock_engine.model_config = MockModelConfig(skip_tokenizer_init=True)
    mock_engine.input_processor = MagicMock()
    mock_engine.io_processor = MagicMock()

    mock_tokenizer = MagicMock(spec=MistralTokenizer)
    mock_renderer = MistralRenderer(mock_engine.model_config, tokenizer_kwargs={})
    mock_renderer._tokenizer = mock_tokenizer
    # Force the Mistral chat template renderer to return token IDs.
    # Choose a prompt length that is < max_model_len, but large enough that
    # adding max_tokens should exceed the model context window.
    mock_renderer.render_messages_async = AsyncMock(
        return_value=(
            [],
            TokensPrompt(prompt_token_ids=list(range(95))),
        )
    )
    mock_engine.renderer = mock_renderer

    serving_chat = _build_serving_chat(mock_engine)

    req = ChatCompletionRequest(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "what is 1+1?"}],
        max_tokens=10,
    )

    resp = await serving_chat.create_chat_completion(req)
    assert isinstance(resp, ErrorResponse)
    assert "context length is only" in resp.error.message


@pytest.mark.asyncio
async def test_serving_chat_mistral_token_ids_prompt_too_long_is_rejected():
    """Regression test: MistralTokenizer token-id prompts must still enforce
    the max context length for the input itself (token_num >= max_model_len).
    """

    mock_engine = MagicMock(spec=AsyncLLM)
    mock_engine.errored = False
    mock_engine.model_config = MockModelConfig(skip_tokenizer_init=True)
    mock_engine.input_processor = MagicMock()
    mock_engine.io_processor = MagicMock()

    mock_tokenizer = MagicMock(spec=MistralTokenizer)
    mock_renderer = MistralRenderer(mock_engine.model_config, tokenizer_kwargs={})
    mock_renderer._tokenizer = mock_tokenizer
    # prompt_token_ids length == max_model_len should be rejected for
    # completion-like requests (ChatCompletionRequest).
    mock_renderer.render_messages_async = AsyncMock(
        return_value=(
            [],
            TokensPrompt(
                prompt_token_ids=list(range(mock_engine.model_config.max_model_len))
            ),
        )
    )
    mock_engine.renderer = mock_renderer

    serving_chat = _build_serving_chat(mock_engine)

    req = ChatCompletionRequest(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "what is 1+1?"}],
        max_tokens=1,
    )

    resp = await serving_chat.create_chat_completion(req)
    assert isinstance(resp, ErrorResponse)
    assert "context length is only" in resp.error.message


@pytest.mark.asyncio
async def test_serving_chat_could_load_correct_generation_config():
    mock_model_config = MockModelConfig()
    mock_model_config.diff_sampling_param = {
        "temperature": 0.5,
        "repetition_penalty": 1.05,
    }

    mock_engine = MagicMock(spec=AsyncLLM)
    mock_engine.errored = False
    mock_engine.model_config = mock_model_config
    mock_engine.input_processor = MagicMock()
    mock_engine.io_processor = MagicMock()
    mock_engine.renderer = _build_renderer(mock_engine.model_config)

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
    mock_engine.errored = False
    mock_engine.model_config = mock_model_config
    mock_engine.input_processor = MagicMock()
    mock_engine.io_processor = MagicMock()
    mock_engine.renderer = _build_renderer(mock_engine.model_config)

    orig_tokenize_prompt_async = mock_engine.renderer.tokenize_prompt_async

    captured_prompts = []

    async def tokenize_prompt_async(prompt, **kwargs):
        captured_prompts.append(prompt)
        return await orig_tokenize_prompt_async(prompt, **kwargs)

    serving_chat = _build_serving_chat(mock_engine)

    # Test cache_salt
    req = ChatCompletionRequest(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "what is 1+1?"}],
    )

    # By default, cache_salt in the engine prompt is not set
    with suppress(Exception):
        await serving_chat.create_chat_completion(req)

    engine_prompt = captured_prompts[0]
    assert "cache_salt" not in engine_prompt

    # Test with certain cache_salt
    req.cache_salt = "test_salt"
    with suppress(Exception):
        await serving_chat.create_chat_completion(req)

    engine_prompt = captured_prompts[1]
    assert engine_prompt.get("cache_salt") == "test_salt"


@pytest.mark.asyncio
async def test_serving_chat_data_parallel_rank_extraction():
    """Test that data_parallel_rank is properly extracted from header and
    passed to engine."""
    mock_engine = MagicMock(spec=AsyncLLM)
    mock_engine.errored = False
    mock_engine.model_config = MockModelConfig()
    mock_engine.input_processor = MagicMock()
    mock_engine.io_processor = MagicMock()
    mock_engine.renderer = _build_renderer(mock_engine.model_config)

    # Mock the generate method to return an async generator
    async def mock_generate(*args, **kwargs):
        # Yield a fake RequestOutput
        from vllm.outputs import CompletionOutput, RequestOutput

        yield RequestOutput(
            request_id="test-request",
            prompt="test prompt",
            prompt_token_ids=[1, 2, 3],
            prompt_logprobs=None,
            outputs=[
                CompletionOutput(
                    index=0,
                    text="test response",
                    token_ids=[4, 5, 6],
                    cumulative_logprob=0.0,
                    logprobs=None,
                    finish_reason="stop",
                    stop_reason=None,
                )
            ],
            finished=True,
        )

    mock_engine.generate = AsyncMock(side_effect=mock_generate)

    serving_chat = _build_serving_chat(mock_engine)

    # Test when data_parallel_rank is present in header
    req = ChatCompletionRequest(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "what is 1+1?"}],
    )

    # Mock request with X-data-parallel-rank header
    mock_raw_request = MagicMock()
    mock_raw_request.headers = {"X-data-parallel-rank": "2"}
    mock_raw_request.state = MagicMock()

    with suppress(Exception):
        await serving_chat.create_chat_completion(req, mock_raw_request)

    # Verify that data_parallel_rank was passed to engine.generate
    assert "data_parallel_rank" in mock_engine.generate.call_args.kwargs
    assert mock_engine.generate.call_args.kwargs["data_parallel_rank"] == 2

    # Test when data_parallel_rank is not present (defaults to None)
    req_no_dp = ChatCompletionRequest(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "what is 2+2?"}],
    )

    # Mock request with no header
    mock_raw_request_no_dp = MagicMock()
    mock_raw_request_no_dp.headers = {}
    mock_raw_request_no_dp.state = MagicMock()

    with suppress(Exception):
        await serving_chat.create_chat_completion(req_no_dp, mock_raw_request_no_dp)

    # Verify that data_parallel_rank defaults to None
    assert "data_parallel_rank" in mock_engine.generate.call_args.kwargs
    assert mock_engine.generate.call_args.kwargs["data_parallel_rank"] is None


class TestServingChatWithHarmony:
    """
    These tests ensure Chat Completion requests are being properly converted into
    Harmony messages and Harmony response messages back into Chat Completion responses.
    These tests are not exhaustive, but each one was created to cover a specific case
    that we got wrong but is now fixed.

    Any changes to the tests and their expectations may result in changes to the
    accuracy of model prompting and responses generated. It is suggested to run
    an evaluation or benchmarking suite (such as bfcl multi_turn) to understand
    any impact of changes in how we prompt Harmony models.
    """

    @pytest.fixture(params=[False, True], ids=["non_streaming", "streaming"])
    def stream(self, request) -> bool:
        """Parameterize tests to run in both non-streaming and streaming modes."""
        return request.param

    @pytest.fixture()
    def mock_engine(self) -> AsyncLLM:
        mock_engine = MagicMock(spec=AsyncLLM)
        mock_engine.get_tokenizer.return_value = get_tokenizer(MODEL_NAME)
        mock_engine.errored = False
        mock_engine.model_config = MockModelConfig()
        mock_engine.input_processor = MagicMock()
        mock_engine.io_processor = MagicMock()
        return mock_engine

    @pytest.fixture()
    def serving_chat(self, mock_engine) -> OpenAIServingChat:
        chat = _build_serving_chat(mock_engine)
        chat.use_harmony = True
        chat.tool_parser = ToolParserManager.get_tool_parser("openai")
        return chat

    def mock_request_output_from_req_and_token_ids(
        self, req: ChatCompletionRequest, token_ids: list[int], finished: bool = False
    ) -> RequestOutput:
        # Our tests don't use most fields, so just get the token ids correct
        completion_output = CompletionOutput(
            index=0,
            text="",
            token_ids=token_ids,
            cumulative_logprob=0.0,
            logprobs=None,
        )
        return RequestOutput(
            request_id=req.request_id,
            prompt=[],
            prompt_token_ids=[],
            prompt_logprobs=None,
            outputs=[completion_output],
            finished=finished,
        )

    @pytest.fixture
    def weather_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"},
                        },
                        "required": ["location"],
                    },
                },
            },
        ]

    @pytest.fixture
    def weather_messages_start(self) -> list[dict[str, Any]]:
        return [
            {
                "role": "user",
                "content": "What's the weather like in Paris today?",
            },
        ]

    async def generate_response_from_harmony_str(
        self,
        serving_chat: OpenAIServingChat,
        req: ChatCompletionRequest,
        harmony_str: str,
        stream: bool = False,
    ) -> ChatCompletionResponse:
        harmony_token_ids = get_encoding().encode(harmony_str, allowed_special="all")

        async def result_generator():
            if stream:
                for token_id in harmony_token_ids:
                    yield self.mock_request_output_from_req_and_token_ids(
                        req, [token_id]
                    )
                yield self.mock_request_output_from_req_and_token_ids(
                    req, [], finished=True
                )
            else:
                yield self.mock_request_output_from_req_and_token_ids(
                    req, harmony_token_ids, finished=True
                )

        generator_func = (
            serving_chat.chat_completion_stream_generator
            if stream
            else serving_chat.chat_completion_full_generator
        )

        result = generator_func(
            request=req,
            result_generator=result_generator(),
            request_id=req.request_id,
            model_name=req.model,
            conversation=[],
            tokenizer=get_tokenizer(req.model),
            request_metadata=RequestResponseMetadata(
                request_id=req.request_id,
                model_name=req.model,
            ),
        )

        if stream:
            return await accumulate_streaming_response(result)
        return await result

    @pytest.mark.asyncio
    async def test_simple_chat(self, serving_chat, stream):
        messages = [{"role": "user", "content": "what is 1+1?"}]

        # Test the Harmony messages for the first turn's input
        req = ChatCompletionRequest(model=MODEL_NAME, messages=messages)
        input_messages, _ = serving_chat._make_request_with_harmony(req)
        verify_harmony_messages(
            input_messages,
            [
                {"role": "system"},
                {"role": "user", "content": messages[0]["content"]},
            ],
        )

        # Test the Chat Completion response for the first turn's output
        reasoning_str = "We need to think really hard about this."
        final_str = "The answer is 2."
        response_str = (
            f"<|channel|>analysis<|message|>{reasoning_str}<|end|>"
            f"<|start|>assistant<|channel|>final<|message|>{final_str}<|end|>"
        )
        response = await self.generate_response_from_harmony_str(
            serving_chat, req, response_str, stream=stream
        )
        verify_chat_response(response, content=final_str, reasoning=reasoning_str)

        # Add the output messages from the first turn as input to the second turn
        for choice in response.choices:
            messages.append(choice.message.model_dump(exclude_none=True))

        # Test the Harmony messages for the second turn's input
        req_2 = ChatCompletionRequest(model=MODEL_NAME, messages=messages)
        input_messages_2, _ = serving_chat._make_request_with_harmony(req_2)
        verify_harmony_messages(
            input_messages_2,
            [
                {"role": "system"},
                {"role": "user"},
                # The analysis message should be dropped on subsequent inputs because
                # of the subsequent assistant message to the final channel.
                {"role": "assistant", "channel": "final", "content": final_str},
            ],
        )

    @pytest.mark.asyncio
    async def test_tool_call_response_with_content(
        self, serving_chat, stream, weather_tools, weather_messages_start
    ):
        tools = weather_tools
        messages = list(weather_messages_start)

        # Test the Harmony messages for the first turn's input
        req = ChatCompletionRequest(model=MODEL_NAME, messages=messages, tools=tools)
        input_messages, _ = serving_chat._make_request_with_harmony(req)
        verify_harmony_messages(
            input_messages,
            [
                {"role": "system"},
                {"role": "developer", "tool_definitions": ["get_weather"]},
                {"role": "user", "content": messages[0]["content"]},
            ],
        )

        # Test the Chat Completion response for the first turn's output
        commentary_str = "We'll call get_weather."
        tool_args_str = '{"location": "Paris"}'
        response_str = (
            f"<|channel|>commentary<|message|>{commentary_str}<|end|>"
            "<|start|>assistant to=functions.get_weather<|channel|>commentary"
            f"<|constrain|>json<|message|>{tool_args_str}<|call|>"
        )
        response = await self.generate_response_from_harmony_str(
            serving_chat, req, response_str, stream=stream
        )
        verify_chat_response(
            response,
            content=commentary_str,
            tool_calls=[("get_weather", tool_args_str)],
        )

        tool_call = response.choices[0].message.tool_calls[0]

        # Add the output messages from the first turn as input to the second turn
        for choice in response.choices:
            messages.append(choice.message.model_dump(exclude_none=True))

        # Add our tool output message
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": "20 degrees Celsius",
            },
        )

        # Test the Harmony messages for the second turn's input
        req_2 = ChatCompletionRequest(model=MODEL_NAME, messages=messages, tools=tools)
        input_messages_2, _ = serving_chat._make_request_with_harmony(req_2)
        verify_harmony_messages(
            input_messages_2,
            [
                {"role": "system"},
                {"role": "developer"},
                {"role": "user"},
                {
                    "role": "assistant",
                    "channel": "commentary",
                    "content": commentary_str,
                },
                {
                    "role": "assistant",
                    "channel": "commentary",
                    "recipient": "functions.get_weather",
                    "content": tool_args_str,
                },
                {
                    "role": "tool",
                    "author_name": "functions.get_weather",
                    "channel": "commentary",
                    "recipient": "assistant",
                    "content": "20 degrees Celsius",
                },
            ],
        )

    @pytest.mark.asyncio
    async def test_tools_and_reasoning(
        self, serving_chat, stream, weather_tools, weather_messages_start
    ):
        tools = weather_tools
        messages = list(weather_messages_start)

        # Test the Harmony messages for the first turn's input
        req = ChatCompletionRequest(model=MODEL_NAME, messages=messages, tools=tools)
        input_messages, _ = serving_chat._make_request_with_harmony(req)
        verify_harmony_messages(
            input_messages,
            [
                {"role": "system"},
                {"role": "developer", "tool_definitions": ["get_weather"]},
                {"role": "user", "content": messages[0]["content"]},
            ],
        )

        # Test the Chat Completion response for the first turn's output
        reasoning_str = "I'll call get_weather."
        tool_args_str = '{"location": "Paris"}'
        response_str = (
            f"<|channel|>analysis<|message|>{reasoning_str}<|end|>"
            "<|start|>assistant to=functions.get_weather<|channel|>commentary"
            f"<|constrain|>json<|message|>{tool_args_str}<|call|>"
        )
        response = await self.generate_response_from_harmony_str(
            serving_chat, req, response_str, stream=stream
        )
        verify_chat_response(
            response,
            reasoning=reasoning_str,
            tool_calls=[("get_weather", tool_args_str)],
        )

        tool_call = response.choices[0].message.tool_calls[0]

        # Add the output messages from the first turn as input to the second turn
        for choice in response.choices:
            messages.append(choice.message.model_dump(exclude_none=True))

        # Add our tool output message
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": "20 degrees Celsius",
            },
        )

        # Test the Harmony messages for the second turn's input
        req_2 = ChatCompletionRequest(model=MODEL_NAME, messages=messages, tools=tools)
        input_messages_2, _ = serving_chat._make_request_with_harmony(req_2)
        verify_harmony_messages(
            input_messages_2,
            [
                {"role": "system"},
                {"role": "developer"},
                {"role": "user"},
                {
                    "role": "assistant",
                    "channel": "analysis",
                    "content": reasoning_str,
                },
                {
                    "role": "assistant",
                    "channel": "commentary",
                    "recipient": "functions.get_weather",
                    "content": tool_args_str,
                },
                {
                    "role": "tool",
                    "author_name": "functions.get_weather",
                    "channel": "commentary",
                    "recipient": "assistant",
                    "content": "20 degrees Celsius",
                },
            ],
        )

    @pytest.mark.asyncio
    async def test_multi_turn_tools_and_reasoning(
        self, serving_chat, stream, weather_tools, weather_messages_start
    ):
        tools = weather_tools
        messages = list(weather_messages_start)

        # Test the Harmony messages for the first turn's input
        req = ChatCompletionRequest(model=MODEL_NAME, messages=messages, tools=tools)
        input_messages, _ = serving_chat._make_request_with_harmony(req)
        verify_harmony_messages(
            input_messages,
            [
                {"role": "system"},
                {"role": "developer", "tool_definitions": ["get_weather"]},
                {"role": "user", "content": messages[0]["content"]},
            ],
        )

        # Test the Chat Completion response for the first turn's output
        reasoning_str = "I'll call get_weather."
        paris_tool_args_str = '{"location": "Paris"}'
        response_str = (
            f"<|channel|>analysis<|message|>{reasoning_str}<|end|>"
            "<|start|>assistant to=functions.get_weather<|channel|>commentary"
            f"<|constrain|>json<|message|>{paris_tool_args_str}<|call|>"
        )
        response = await self.generate_response_from_harmony_str(
            serving_chat, req, response_str, stream=stream
        )
        verify_chat_response(
            response,
            reasoning=reasoning_str,
            tool_calls=[("get_weather", paris_tool_args_str)],
        )

        tool_call = response.choices[0].message.tool_calls[0]

        # Add the output messages from the first turn as input to the second turn
        for choice in response.choices:
            messages.append(choice.message.model_dump(exclude_none=True))

        # Add our tool output message
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": "20 degrees Celsius",
            },
        )

        # Test the Harmony messages for the second turn's input
        req_2 = ChatCompletionRequest(model=MODEL_NAME, messages=messages, tools=tools)
        input_messages_2, _ = serving_chat._make_request_with_harmony(req_2)
        verify_harmony_messages(
            input_messages_2,
            [
                {"role": "system"},
                {"role": "developer"},
                {"role": "user"},
                {
                    "role": "assistant",
                    "channel": "analysis",
                    "content": reasoning_str,
                },
                {
                    "role": "assistant",
                    "channel": "commentary",
                    "recipient": "functions.get_weather",
                    "content": paris_tool_args_str,
                },
                {
                    "role": "tool",
                    "author_name": "functions.get_weather",
                    "channel": "commentary",
                    "recipient": "assistant",
                    "content": "20 degrees Celsius",
                },
            ],
        )

        # Test the Chat Completion response for the second turn's output
        paris_weather_str = "The weather in Paris today is 20 degrees Celsius."
        response_str = f"<|channel|>final<|message|>{paris_weather_str}<|end|>"
        response_2 = await self.generate_response_from_harmony_str(
            serving_chat, req_2, response_str, stream=stream
        )
        verify_chat_response(response_2, content=paris_weather_str)

        # Add the output messages from the second turn as input to the third turn
        for choice in response_2.choices:
            messages.append(choice.message.model_dump(exclude_none=True))

        # Add a new user message for the third turn
        messages.append(
            {
                "role": "user",
                "content": "What's the weather like in Boston today?",
            },
        )

        # Test the Harmony messages for the third turn's input
        req_3 = ChatCompletionRequest(model=MODEL_NAME, messages=messages, tools=tools)
        input_messages_3, _ = serving_chat._make_request_with_harmony(req_3)
        verify_harmony_messages(
            input_messages_3,
            [
                {"role": "system"},
                {"role": "developer"},
                {"role": "user"},
                {
                    "role": "assistant",
                    "channel": "commentary",
                    "recipient": "functions.get_weather",
                    "content": paris_tool_args_str,
                },
                {
                    "role": "tool",
                    "author_name": "functions.get_weather",
                    "channel": "commentary",
                    "recipient": "assistant",
                    "content": "20 degrees Celsius",
                },
                {
                    "role": "assistant",
                    "channel": "final",
                    "content": paris_weather_str,
                },
                {"role": "user", "content": messages[-1]["content"]},
            ],
        )

        # Test the Chat Completion response for the third turn's output
        reasoning_str = "I'll call get_weather."
        boston_tool_args_str = '{"location": "Boston"}'
        response_str = (
            f"<|channel|>analysis<|message|>{reasoning_str}<|end|>"
            "<|start|>assistant to=functions.get_weather<|channel|>commentary"
            f"<|constrain|>json<|message|>{boston_tool_args_str}<|call|>"
        )
        response_3 = await self.generate_response_from_harmony_str(
            serving_chat, req, response_str, stream=stream
        )
        verify_chat_response(
            response_3,
            reasoning=reasoning_str,
            tool_calls=[("get_weather", boston_tool_args_str)],
        )

        tool_call = response_3.choices[0].message.tool_calls[0]

        # Add the output messages from the third turn as input to the fourth turn
        for choice in response_3.choices:
            messages.append(choice.message.model_dump(exclude_none=True))

        # Add our tool output message
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": "10 degrees Celsius",
            },
        )

        # Test the Harmony messages for the fourth turn's input
        req_4 = ChatCompletionRequest(model=MODEL_NAME, messages=messages, tools=tools)
        input_messages_4, _ = serving_chat._make_request_with_harmony(req_4)
        verify_harmony_messages(
            input_messages_4,
            [
                {"role": "system"},
                {"role": "developer"},
                {"role": "user"},
                {"role": "assistant"},
                {"role": "tool"},
                {
                    "role": "assistant",
                    "channel": "final",
                },
                {"role": "user"},
                {
                    "role": "assistant",
                    "channel": "analysis",
                    "content": reasoning_str,
                },
                {
                    "role": "assistant",
                    "channel": "commentary",
                    "recipient": "functions.get_weather",
                    "content": boston_tool_args_str,
                },
                {
                    "role": "tool",
                    "author_name": "functions.get_weather",
                    "channel": "commentary",
                    "recipient": "assistant",
                    "content": "10 degrees Celsius",
                },
            ],
        )

    @pytest.mark.asyncio
    async def test_non_tool_reasoning(self, serving_chat):
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": "What's 2+2?",
            },
            {
                "role": "assistant",
                "reasoning": "Adding 2 and 2 is easy. The result is 4.",
                "content": "4",
            },
        ]
        req = ChatCompletionRequest(model=MODEL_NAME, messages=messages)
        input_messages, _ = serving_chat._make_request_with_harmony(req)

        verify_harmony_messages(
            input_messages,
            [
                {"role": "system"},
                {"role": "user", "content": messages[0]["content"]},
                # The reasoning that would have resulted in an analysis message is
                # dropped because of a later assistant message to the final channel.
                {
                    "role": "assistant",
                    "channel": "final",
                    "content": messages[1]["content"],
                },
            ],
        )

    @pytest.mark.asyncio
    async def test_non_tool_reasoning_empty_content(self, serving_chat):
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": "What's 2+2?",
            },
            {
                "role": "assistant",
                "reasoning": "Adding 2 and 2 is easy. The result is 4.",
                "content": "",
            },
        ]
        req = ChatCompletionRequest(model=MODEL_NAME, messages=messages)
        input_messages, _ = serving_chat._make_request_with_harmony(req)

        verify_harmony_messages(
            input_messages,
            [
                {"role": "system"},
                {"role": "user", "content": messages[0]["content"]},
                {
                    "role": "assistant",
                    "channel": "analysis",
                    "content": messages[1]["reasoning"],
                },
            ],
        )

    @pytest.mark.asyncio
    async def test_non_tool_reasoning_empty_content_list(self, serving_chat):
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": "What's 2+2?",
            },
            {
                "role": "assistant",
                "reasoning": "Adding 2 and 2 is easy. The result is 4.",
                "content": [],
            },
        ]
        req = ChatCompletionRequest(model=MODEL_NAME, messages=messages)
        input_messages, _ = serving_chat._make_request_with_harmony(req)

        verify_harmony_messages(
            input_messages,
            [
                {"role": "system"},
                {"role": "user", "content": messages[0]["content"]},
                {
                    "role": "assistant",
                    "channel": "analysis",
                    "content": messages[1]["reasoning"],
                },
            ],
        )


@pytest.mark.asyncio
async def test_tool_choice_validation_without_parser():
    """Test that tool_choice='required' or named tool without tool_parser
    returns an appropriate error message."""
    mock_engine = MagicMock(spec=AsyncLLM)
    mock_engine.get_tokenizer.return_value = get_tokenizer(MODEL_NAME)
    mock_engine.errored = False
    mock_engine.model_config = MockModelConfig()
    mock_engine.input_processor = MagicMock()
    mock_engine.io_processor = MagicMock()

    models = OpenAIServingModels(
        engine_client=mock_engine,
        base_model_paths=BASE_MODEL_PATHS,
    )
    # Create serving_chat without tool_parser (enable_auto_tools=False)
    serving_chat = OpenAIServingChat(
        mock_engine,
        models,
        response_role="assistant",
        chat_template=CHAT_TEMPLATE,
        chat_template_content_format="auto",
        request_logger=None,
        enable_auto_tools=False,  # No tool parser
    )

    tools = [
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
    ]

    # Test tool_choice="required" without tool_parser
    req_required = ChatCompletionRequest(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "What's the weather?"}],
        tools=tools,
        tool_choice="required",
    )
    response_required = await serving_chat.create_chat_completion(req_required)
    assert isinstance(response_required, ErrorResponse)
    assert "tool_choice" in response_required.error.message
    assert "--tool-call-parser" in response_required.error.message

    # Test named tool_choice without tool_parser
    req_named = ChatCompletionRequest(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "What's the weather?"}],
        tools=tools,
        tool_choice={"type": "function", "function": {"name": "get_weather"}},
    )
    response_named = await serving_chat.create_chat_completion(req_named)
    assert isinstance(response_named, ErrorResponse)
    assert "tool_choice" in response_named.error.message
    assert "--tool-call-parser" in response_named.error.message


class TestCreateRemainingArgsDelta:
    """Tests for _create_remaining_args_delta helper function.

    This helper is used when streaming tool calls to preserve id/type/name
    fields in the finish chunk, which would otherwise be lost.
    """

    def test_preserves_id_type_name(self):
        """Test that id, type, and name are preserved from original delta."""
        from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
        from vllm.entrypoints.openai.engine.protocol import (
            DeltaFunctionCall,
            DeltaMessage,
            DeltaToolCall,
        )

        original_delta = DeltaMessage(
            tool_calls=[
                DeltaToolCall(
                    index=0,
                    id="call_abc123",
                    type="function",
                    function=DeltaFunctionCall(
                        name="get_weather",
                        arguments='{"location": "Paris"}',
                    ),
                )
            ]
        )

        result = OpenAIServingChat._create_remaining_args_delta(
            original_delta, '", "unit": "celsius"}', 0
        )

        assert len(result.tool_calls) == 1
        tc = result.tool_calls[0]
        assert tc.index == 0
        assert tc.id == "call_abc123"
        assert tc.type == "function"
        assert tc.function.name == "get_weather"
        assert tc.function.arguments == '", "unit": "celsius"}'

    def test_matches_by_index(self):
        """Test that the correct tool call is matched by index."""
        from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
        from vllm.entrypoints.openai.engine.protocol import (
            DeltaFunctionCall,
            DeltaMessage,
            DeltaToolCall,
        )

        original_delta = DeltaMessage(
            tool_calls=[
                DeltaToolCall(
                    index=0,
                    id="call_first",
                    type="function",
                    function=DeltaFunctionCall(name="func_a", arguments="{}"),
                ),
                DeltaToolCall(
                    index=1,
                    id="call_second",
                    type="function",
                    function=DeltaFunctionCall(name="func_b", arguments="{}"),
                ),
            ]
        )

        result = OpenAIServingChat._create_remaining_args_delta(
            original_delta, '{"extra": true}', 1
        )

        assert len(result.tool_calls) == 1
        tc = result.tool_calls[0]
        assert tc.index == 1
        assert tc.id == "call_second"
        assert tc.function.name == "func_b"

    def test_no_matching_tool_call(self):
        """Test graceful handling when no matching tool call is found."""
        from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
        from vllm.entrypoints.openai.engine.protocol import (
            DeltaFunctionCall,
            DeltaMessage,
            DeltaToolCall,
        )

        original_delta = DeltaMessage(
            tool_calls=[
                DeltaToolCall(
                    index=0,
                    id="call_zero",
                    type="function",
                    function=DeltaFunctionCall(name="func", arguments="{}"),
                )
            ]
        )

        result = OpenAIServingChat._create_remaining_args_delta(
            original_delta, '{"arg": 1}', 5
        )

        assert len(result.tool_calls) == 1
        tc = result.tool_calls[0]
        assert tc.index == 5
        assert tc.id is None
        assert tc.type is None
        assert tc.function.name is None
        assert tc.function.arguments == '{"arg": 1}'

    def test_function_is_none(self):
        """Test handling when original tool call has no function."""
        from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
        from vllm.entrypoints.openai.engine.protocol import DeltaMessage, DeltaToolCall

        original_delta = DeltaMessage(
            tool_calls=[
                DeltaToolCall(
                    index=0,
                    id="call_nofunc",
                    type="function",
                    function=None,
                )
            ]
        )

        result = OpenAIServingChat._create_remaining_args_delta(
            original_delta, '{"data": "value"}', 0
        )

        assert len(result.tool_calls) == 1
        tc = result.tool_calls[0]
        assert tc.index == 0
        assert tc.id == "call_nofunc"
        assert tc.type == "function"
        assert tc.function.name is None
        assert tc.function.arguments == '{"data": "value"}'
