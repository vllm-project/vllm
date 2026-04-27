# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from dataclasses import dataclass, field

import openai
import pytest

from tests.tool_use.utils import (
    MESSAGES_ASKING_FOR_PARALLEL_TOOLS,
    MESSAGES_ASKING_FOR_TOOLS,
    MESSAGES_WITH_TOOL_RESPONSE,
    MESSAGES_WITHOUT_TOOLS,
    SEARCH_TOOL,
    SEED,
    WEATHER_TOOL,
    ensure_system_prompt,
)

from .utils import ServerConfig


def _requires_tool_parser(server_config: ServerConfig) -> None:
    r"""Skip test if server was not started with --tool-call-parser."""
    if "--tool-call-parser" not in server_config.get("arguments", []):
        pytest.skip(
            f"Skipping: {server_config['model']} not configured with --tool-call-parser"
        )


def _is_pre_v11(server_config: ServerConfig) -> bool:
    r"""Pre-v11 Mistral models lack grammar-based tool call enforcement."""
    return "7B" in server_config.get("model", "")


@dataclass
class StreamedToolCallResult:
    r"""Accumulated result from streaming a single tool call."""

    function_name: str | None = None
    function_args_str: str = ""
    tool_call_id: str | None = None
    role_name: str | None = None
    finish_reason_count: int = 0
    finish_reason: str | None = None


async def _collect_streamed_tool_call(
    stream: openai.AsyncStream,
    *,
    expected_finish_reason: str = "tool_calls",
) -> StreamedToolCallResult:
    result = StreamedToolCallResult()

    async for chunk in stream:
        if chunk.choices[0].finish_reason:
            result.finish_reason_count += 1
            result.finish_reason = chunk.choices[0].finish_reason
            assert chunk.choices[0].finish_reason == expected_finish_reason

        if chunk.choices[0].delta.role:
            assert not result.role_name or result.role_name == "assistant"
            result.role_name = "assistant"

        streamed_tool_calls = chunk.choices[0].delta.tool_calls
        if streamed_tool_calls and len(streamed_tool_calls) > 0:
            assert len(streamed_tool_calls) == 1
            tool_call = streamed_tool_calls[0]

            if tool_call.id:
                assert not result.tool_call_id
                result.tool_call_id = tool_call.id

            if tool_call.function:
                if tool_call.function.name:
                    assert result.function_name is None
                    result.function_name = tool_call.function.name
                if tool_call.function.arguments:
                    result.function_args_str += tool_call.function.arguments

    return result


@dataclass
class StreamedContentResult:
    r"""Accumulated result from streaming a content-only response."""

    chunks: list[str] = field(default_factory=list)
    finish_reason_count: int = 0
    finish_reason: str | None = None
    role_sent: bool = False


async def _collect_streamed_content(
    stream: openai.AsyncStream,
    *,
    expected_finish_reason: str | None = None,
    no_tool_calls: bool = True,
) -> StreamedContentResult:
    r"""Consume a streaming response and collect text content."""
    result = StreamedContentResult()

    async for chunk in stream:
        delta = chunk.choices[0].delta

        if delta.role:
            assert not result.role_sent
            assert delta.role == "assistant"
            result.role_sent = True

        if delta.content:
            result.chunks.append(delta.content)

        if chunk.choices[0].finish_reason is not None:
            result.finish_reason_count += 1
            result.finish_reason = chunk.choices[0].finish_reason
            if expected_finish_reason is not None:
                assert result.finish_reason == expected_finish_reason

        if no_tool_calls:
            assert not delta.tool_calls or len(delta.tool_calls) == 0

    return result


@dataclass
class StreamedParallelToolCallResult:
    r"""Accumulated result from streaming parallel tool calls."""

    function_names: list[str] = field(default_factory=list)
    function_args_strs: list[str] = field(default_factory=list)
    tool_call_ids: list[str] = field(default_factory=list)
    role_name: str | None = None
    finish_reason_count: int = 0


async def _collect_streamed_parallel_tool_calls(
    stream: openai.AsyncStream,
) -> StreamedParallelToolCallResult:
    r"""Consume a streaming response and collect parallel tool calls."""
    result = StreamedParallelToolCallResult()
    tool_call_idx: int = -1

    async for chunk in stream:
        if chunk.choices[0].finish_reason:
            result.finish_reason_count += 1
            assert chunk.choices[0].finish_reason == "tool_calls"

        if chunk.choices[0].delta.role:
            assert not result.role_name or result.role_name == "assistant"
            result.role_name = "assistant"

        streamed_tool_calls = chunk.choices[0].delta.tool_calls
        if streamed_tool_calls and len(streamed_tool_calls) > 0:
            assert len(streamed_tool_calls) == 1
            tool_call = streamed_tool_calls[0]

            if tool_call.index != tool_call_idx:
                tool_call_idx = tool_call.index
                result.function_args_strs.append("")
                result.tool_call_ids.append("")

            if tool_call.id:
                result.tool_call_ids[tool_call.index] = tool_call.id

            if tool_call.function:
                if tool_call.function.name:
                    result.function_names.append(tool_call.function.name)
                if tool_call.function.arguments:
                    result.function_args_strs[tool_call.index] += (
                        tool_call.function.arguments
                    )

    return result


# test: a tool_choice with mistral-tokenizer results in an ID of length 9
@pytest.mark.asyncio
async def test_tool_call_with_tool_choice(
    client: openai.AsyncOpenAI, server_config: ServerConfig
) -> None:
    _requires_tool_parser(server_config)

    models = await client.models.list()
    model_name: str = models.data[0].id
    chat_completion = await client.chat.completions.create(
        messages=ensure_system_prompt(MESSAGES_ASKING_FOR_TOOLS, server_config),
        temperature=0,
        max_completion_tokens=100,
        model=model_name,
        tools=[WEATHER_TOOL],
        tool_choice=WEATHER_TOOL,
        logprobs=False,
        seed=SEED,
    )

    choice = chat_completion.choices[0]

    assert choice.finish_reason != "tool_calls"  # "stop" or "length"
    assert choice.message.role == "assistant"
    assert choice.message.tool_calls is None or len(choice.message.tool_calls) == 1
    assert len(choice.message.tool_calls[0].id) == 9  # length of 9 for mistral


_NOT_SET = object()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tools, tool_choice, streaming_id_len_pre_v11",
    [
        pytest.param(
            [WEATHER_TOOL, SEARCH_TOOL],
            _NOT_SET,
            9,
            id="auto",
        ),
        pytest.param(
            [WEATHER_TOOL],
            "required",
            30,
            id="required",
        ),
    ],
)
async def test_tool_call_auto_or_required(
    client: openai.AsyncOpenAI,
    server_config: ServerConfig,
    tools: list,
    tool_choice: object,
    streaming_id_len_pre_v11: int,
) -> None:
    _requires_tool_parser(server_config)

    models = await client.models.list()
    model_name: str = models.data[0].id

    create_kwargs: dict = {
        "messages": ensure_system_prompt(MESSAGES_ASKING_FOR_TOOLS, server_config),
        "temperature": 0,
        "max_completion_tokens": 100,
        "model": model_name,
        "tools": tools,
        "logprobs": False,
        "seed": SEED,
    }
    if tool_choice is not _NOT_SET:
        create_kwargs["tool_choice"] = tool_choice

    # --- non-streaming ---
    chat_completion = await client.chat.completions.create(**create_kwargs)

    choice = chat_completion.choices[0]
    tool_calls = choice.message.tool_calls

    assert choice.finish_reason == "tool_calls"
    assert tool_calls is not None and len(tool_calls) >= 1
    assert tool_calls[0].function.name == "get_current_weather"
    parsed_arguments = json.loads(tool_calls[0].function.arguments)
    assert "city" in parsed_arguments
    assert len(tool_calls[0].id) == 9

    # --- streaming ---
    stream = await client.chat.completions.create(**create_kwargs, stream=True)

    result = await _collect_streamed_tool_call(stream)

    assert result.finish_reason_count == 1
    assert result.role_name == "assistant"
    assert result.function_name == "get_current_weather"
    streamed_args = json.loads(result.function_args_str)
    assert isinstance(result.tool_call_id, str)
    if _is_pre_v11(server_config):
        assert len(result.tool_call_id) == streaming_id_len_pre_v11
    else:
        assert len(result.tool_call_id) == 9
    assert parsed_arguments == streamed_args


@pytest.mark.asyncio
async def test_tool_call_none_with_tools(
    client: openai.AsyncOpenAI, server_config: ServerConfig
) -> None:
    _requires_tool_parser(server_config)

    models = await client.models.list()
    model_name: str = models.data[0].id

    # --- non-streaming ---
    chat_completion = await client.chat.completions.create(
        messages=ensure_system_prompt(MESSAGES_ASKING_FOR_TOOLS, server_config),
        temperature=0,
        max_completion_tokens=100,
        model=model_name,
        tools=[WEATHER_TOOL],
        tool_choice="none",
        logprobs=False,
        seed=SEED,
    )

    choice = chat_completion.choices[0]

    assert choice.finish_reason != "tool_calls"
    assert choice.message.tool_calls is None or len(choice.message.tool_calls) == 0
    assert choice.message.content is not None
    # Without grammar enforcement, pre-v11 models may still emit [TOOL_CALLS]
    if not _is_pre_v11(server_config):
        assert "[TOOL_CALLS]" not in choice.message.content

    non_streaming_content = choice.message.content

    # --- streaming ---
    stream = await client.chat.completions.create(
        messages=ensure_system_prompt(MESSAGES_ASKING_FOR_TOOLS, server_config),
        temperature=0,
        max_completion_tokens=100,
        model=model_name,
        tools=[WEATHER_TOOL],
        tool_choice="none",
        logprobs=False,
        seed=SEED,
        stream=True,
    )

    # Pre-v11 models lack grammar enforcement, so the model may still
    # emit tool calls even with tool_choice="none".
    pre_v11 = _is_pre_v11(server_config)
    result = await _collect_streamed_content(stream, no_tool_calls=not pre_v11)

    assert result.finish_reason_count == 1
    if not pre_v11:
        assert result.finish_reason != "tool_calls"
    streamed_content = "".join(result.chunks)
    if not pre_v11:
        assert "[TOOL_CALLS]" not in streamed_content
        assert streamed_content == non_streaming_content


@pytest.mark.asyncio
async def test_chat_without_tools(
    client: openai.AsyncOpenAI, server_config: ServerConfig
) -> None:
    models = await client.models.list()
    model_name: str = models.data[0].id

    # --- non-streaming ---
    chat_completion = await client.chat.completions.create(
        messages=ensure_system_prompt(MESSAGES_WITHOUT_TOOLS, server_config),
        temperature=0,
        max_completion_tokens=150,
        model=model_name,
        logprobs=False,
        seed=SEED,
    )

    choice = chat_completion.choices[0]
    output_text = choice.message.content

    assert output_text is not None and len(output_text) > 0
    assert choice.finish_reason != "tool_calls"
    assert choice.message.tool_calls is None or len(choice.message.tool_calls) == 0

    # --- streaming ---
    stream = await client.chat.completions.create(
        messages=ensure_system_prompt(MESSAGES_WITHOUT_TOOLS, server_config),
        temperature=0,
        max_completion_tokens=150,
        model=model_name,
        logprobs=False,
        seed=SEED,
        stream=True,
    )

    result = await _collect_streamed_content(
        stream, expected_finish_reason=choice.finish_reason
    )

    assert result.role_sent
    assert result.finish_reason_count == 1
    assert len(result.chunks)
    assert "".join(result.chunks) == output_text


@pytest.mark.asyncio
async def test_tool_call_with_results(
    client: openai.AsyncOpenAI, server_config: ServerConfig
) -> None:
    _requires_tool_parser(server_config)

    models = await client.models.list()
    model_name: str = models.data[0].id

    # --- non-streaming ---
    chat_completion = await client.chat.completions.create(
        messages=ensure_system_prompt(MESSAGES_WITH_TOOL_RESPONSE, server_config),
        temperature=0,
        max_completion_tokens=100,
        model=model_name,
        tools=[WEATHER_TOOL, SEARCH_TOOL],
        logprobs=False,
        seed=SEED,
    )

    choice = chat_completion.choices[0]

    assert choice.finish_reason != "tool_calls"
    assert choice.message.tool_calls is None or len(choice.message.tool_calls) == 0
    assert choice.message.content is not None
    assert "98" in choice.message.content

    # --- streaming ---
    stream = await client.chat.completions.create(
        messages=ensure_system_prompt(MESSAGES_WITH_TOOL_RESPONSE, server_config),
        temperature=0,
        max_completion_tokens=100,
        model=model_name,
        tools=[WEATHER_TOOL, SEARCH_TOOL],
        logprobs=False,
        seed=SEED,
        stream=True,
    )

    result = await _collect_streamed_content(
        stream, expected_finish_reason=choice.finish_reason
    )

    assert result.role_sent
    assert result.finish_reason_count == 1
    assert len(result.chunks)
    assert "".join(result.chunks) == choice.message.content


def _requires_parallel(server_config: ServerConfig) -> None:
    r"""Skip test if the model does not support parallel tool calls."""
    if not server_config.get("supports_parallel"):
        pytest.skip(
            f"Skipping: {server_config['model']} does not support parallel tool calls"
        )


@pytest.mark.asyncio
async def test_tool_call_parallel(
    client: openai.AsyncOpenAI, server_config: ServerConfig
) -> None:
    _requires_tool_parser(server_config)
    _requires_parallel(server_config)

    models = await client.models.list()
    model_name: str = models.data[0].id

    # --- non-streaming ---
    chat_completion = await client.chat.completions.create(
        messages=ensure_system_prompt(
            MESSAGES_ASKING_FOR_PARALLEL_TOOLS, server_config
        ),
        temperature=0,
        max_completion_tokens=200,
        model=model_name,
        tools=[WEATHER_TOOL],
        logprobs=False,
        seed=SEED,
    )

    choice = chat_completion.choices[0]
    tool_calls = choice.message.tool_calls

    assert choice.finish_reason == "tool_calls"
    assert tool_calls is not None and len(tool_calls) >= 2
    for tc in tool_calls:
        assert tc.type == "function"
        assert tc.function.name == "get_current_weather"
        assert isinstance(tc.function.arguments, str)
        parsed = json.loads(tc.function.arguments)
        assert "city" in parsed
        assert len(tc.id) == 9

    non_streaming_tool_calls = tool_calls

    # --- streaming ---
    stream = await client.chat.completions.create(
        messages=ensure_system_prompt(
            MESSAGES_ASKING_FOR_PARALLEL_TOOLS, server_config
        ),
        temperature=0,
        max_completion_tokens=200,
        model=model_name,
        tools=[WEATHER_TOOL],
        logprobs=False,
        seed=SEED,
        stream=True,
    )

    result = await _collect_streamed_parallel_tool_calls(stream)

    assert result.finish_reason_count == 1
    assert result.role_name == "assistant"
    assert len(result.function_names) >= 2
    assert all(name == "get_current_weather" for name in result.function_names)
    assert len(result.tool_call_ids) >= 2
    assert all(isinstance(tid, str) and len(tid) == 9 for tid in result.tool_call_ids)

    for args_str in result.function_args_strs:
        streamed_args = json.loads(args_str)
        assert "city" in streamed_args

    assert len(result.function_names) == len(non_streaming_tool_calls)
