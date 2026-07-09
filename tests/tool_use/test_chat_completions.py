# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import openai
import pytest

from .utils import (
    MESSAGES_WITHOUT_TOOLS,
    SEED,
    WEATHER_TOOL,
    ServerConfig,
    ensure_system_prompt,
)


# test: make sure chat completions without tools provided work even when tools
# are enabled. This makes sure tool call chat templates work, AND that the tool
# parser stream processing doesn't change the output of the model.
@pytest.mark.asyncio
async def test_chat_completion_without_tools(
    client: openai.AsyncOpenAI, server_config: ServerConfig
):
    models = await client.models.list()
    model_name: str = models.data[0].id
    chat_completion = await client.chat.completions.create(
        messages=ensure_system_prompt(MESSAGES_WITHOUT_TOOLS, server_config),
        temperature=0,
        max_completion_tokens=150,
        model=model_name,
        logprobs=False,
        seed=SEED,
    )
    choice = chat_completion.choices[0]
    stop_reason = chat_completion.choices[0].finish_reason
    output_text = chat_completion.choices[0].message.content

    # check to make sure we got text
    assert output_text is not None
    assert len(output_text) > 0
    assert stop_reason != "tool_calls"

    # check to make sure no tool calls were returned
    assert choice.message.tool_calls is None or len(choice.message.tool_calls) == 0

    # make the same request, streaming
    stream = await client.chat.completions.create(
        messages=ensure_system_prompt(MESSAGES_WITHOUT_TOOLS, server_config),
        temperature=0,
        max_completion_tokens=150,
        model=model_name,
        logprobs=False,
        seed=SEED,
        stream=True,
    )
    chunks: list[str] = []
    finish_reason_count = 0
    role_sent: bool = False

    # assemble streamed chunks
    async for chunk in stream:
        delta = chunk.choices[0].delta

        # make sure the role is assistant
        if delta.role:
            assert not role_sent
            assert delta.role == "assistant"
            role_sent = True

        if delta.content:
            chunks.append(delta.content)

        if chunk.choices[0].finish_reason is not None:
            finish_reason_count += 1
            assert chunk.choices[0].finish_reason == choice.finish_reason

        # make sure tool call chunks aren't being streamed
        assert not delta.tool_calls or len(delta.tool_calls) == 0

    # make sure the role was sent, only 1 finish reason was sent, that chunks
    # were in fact sent, and that the chunks match non-streaming
    assert role_sent
    assert finish_reason_count == 1
    assert len(chunks)
    assert "".join(chunks) == output_text


# test: conversation with tools enabled and provided that should not invoke
# tools, to make sure we can still get normal chat completion responses
# and that they won't be parsed as tools
@pytest.mark.asyncio
async def test_chat_completion_with_tools(
    client: openai.AsyncOpenAI, server_config: ServerConfig
):
    models = await client.models.list()
    model_name: str = models.data[0].id
    chat_completion = await client.chat.completions.create(
        messages=ensure_system_prompt(MESSAGES_WITHOUT_TOOLS, server_config),
        temperature=0,
        max_completion_tokens=150,
        model=model_name,
        tools=[WEATHER_TOOL],
        logprobs=False,
        seed=SEED,
    )
    choice = chat_completion.choices[0]
    stop_reason = chat_completion.choices[0].finish_reason
    output_text = chat_completion.choices[0].message.content

    # check to make sure we got text
    assert output_text is not None
    assert stop_reason != "tool_calls"
    assert len(output_text) > 0

    # check to make sure no tool calls were returned
    assert choice.message.tool_calls is None or len(choice.message.tool_calls) == 0

    # make the same request, streaming
    stream = await client.chat.completions.create(
        messages=ensure_system_prompt(MESSAGES_WITHOUT_TOOLS, server_config),
        temperature=0,
        max_completion_tokens=150,
        model=model_name,
        logprobs=False,
        tools=[WEATHER_TOOL],
        seed=SEED,
        stream=True,
    )

    chunks: list[str] = []
    finish_reason_count = 0
    role_sent: bool = False

    # assemble streamed chunks
    async for chunk in stream:
        delta = chunk.choices[0].delta

        # make sure the role is assistant
        if delta.role:
            assert delta.role == "assistant"
            role_sent = True

        if delta.content:
            chunks.append(delta.content)

        if chunk.choices[0].finish_reason is not None:
            finish_reason_count += 1

        # make sure tool call chunks aren't being streamed
        assert not delta.tool_calls or len(delta.tool_calls) == 0

    # make sure the role was sent, only 1 finish reason was sent, that chunks
    # were in fact sent, and that the chunks match non-streaming
    assert role_sent
    assert finish_reason_count == 1
    assert chunk.choices[0].finish_reason == stop_reason
    assert chunk.choices[0].finish_reason != "tool_calls"
    assert len(chunks)
    assert "".join(chunks) == output_text


# Regression test for https://github.com/vllm-project/vllm/issues/32006
# Engine crash when combining response_format: json_object with
# tool_choice: required
@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_response_format_with_tool_choice_required(
    client: openai.AsyncOpenAI, server_config: ServerConfig
):
    """
    Test that combining response_format: json_object with tool_choice: required
    doesn't crash the engine.

    Before the fix, this would cause a validation error:
    "You can only use one kind of structured outputs constraint but multiple
    are specified" because both json_object and json (from tool schema) would
    be set in StructuredOutputsParams.
    """
    models = await client.models.list()
    model_name: str = models.data[0].id

    # This combination previously crashed the engine
    chat_completion = await client.chat.completions.create(
        messages=ensure_system_prompt(
            [{"role": "user", "content": "What is the weather in Dallas, Texas?"}],
            server_config,
        ),
        temperature=0,
        max_completion_tokens=150,
        model=model_name,
        tools=[WEATHER_TOOL],
        tool_choice="required",
        response_format={"type": "json_object"},
    )

    # The fix clears response_format when tool_choice forces tool calling,
    # so the request should complete successfully with tool calls
    choice = chat_completion.choices[0]
    assert choice.finish_reason == "tool_calls"
    assert choice.message.tool_calls is not None
    assert len(choice.message.tool_calls) > 0


def _tool_call_parser(server_config: ServerConfig) -> str | None:
    arguments = server_config.get("arguments", [])
    if "--tool-call-parser" not in arguments:
        return None
    return arguments[arguments.index("--tool-call-parser") + 1]


# Composed tool-call-or-schema grammar for tool_choice="auto" + response_format.
# See https://github.com/vllm-project/vllm/issues/39929.
@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_response_format_with_tool_choice_auto(
    client: openai.AsyncOpenAI, server_config: ServerConfig
):
    """
    Test that combining response_format: json_schema with tool_choice: auto
    doesn't crash the engine and, for models with a composable structural tag
    (hermes, minimax), still calls the tool or answers within the schema
    instead of silently dropping the response_format.
    """
    models = await client.models.list()
    model_name: str = models.data[0].id

    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }

    chat_completion = await client.chat.completions.create(
        messages=ensure_system_prompt(
            [{"role": "user", "content": "What is the weather in Dallas, Texas?"}],
            server_config,
        ),
        temperature=0,
        max_completion_tokens=150,
        model=model_name,
        tools=[WEATHER_TOOL],
        tool_choice="auto",
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "final_answer", "schema": schema},
        },
    )

    choice = chat_completion.choices[0]
    if _tool_call_parser(server_config) in ("hermes", "minimax_m2"):
        # Either branch of the composed grammar is valid: a tool call, or a
        # final answer constrained to the schema.
        if choice.finish_reason == "tool_calls":
            assert choice.message.tool_calls
        else:
            json.loads(choice.message.content)
    else:
        # Non-composable models fall back to unchanged behavior and must not
        # crash the engine.
        assert choice.finish_reason is not None
