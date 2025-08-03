# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import openai
import pytest

from .utils import (MESSAGES_WITHOUT_TOOLS, WEATHER_TOOL, ServerConfig,
                    ensure_system_prompt)


# test: make sure chat completions without tools provided work even when tools
# are enabled. This makes sure tool call chat templates work, AND that the tool
# parser stream processing doesn't change the output of the model.
@pytest.mark.asyncio
async def test_chat_completion_without_tools(client: openai.AsyncOpenAI,
                                             server_config: ServerConfig):
    models = await client.models.list()
    model_name: str = models.data[0].id
    chat_completion = await client.chat.completions.create(
        messages=ensure_system_prompt(MESSAGES_WITHOUT_TOOLS, server_config),
        temperature=0,
        max_completion_tokens=150,
        model=model_name,
        logprobs=False)
    choice = chat_completion.choices[0]
    stop_reason = chat_completion.choices[0].finish_reason
    output_text = chat_completion.choices[0].message.content

    # check to make sure we got text
    assert output_text is not None
    assert len(output_text) > 0
    assert stop_reason != "tool_calls"

    # check to make sure no tool calls were returned
    assert (choice.message.tool_calls is None
            or len(choice.message.tool_calls) == 0)

    # make the same request, streaming
    stream = await client.chat.completions.create(
        messages=ensure_system_prompt(MESSAGES_WITHOUT_TOOLS, server_config),
        temperature=0,
        max_completion_tokens=150,
        model=model_name,
        logprobs=False,
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
            assert delta.role == 'assistant'
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
async def test_chat_completion_with_tools(client: openai.AsyncOpenAI,
                                          server_config: ServerConfig):
    models = await client.models.list()
    model_name: str = models.data[0].id
    chat_completion = await client.chat.completions.create(
        messages=ensure_system_prompt(MESSAGES_WITHOUT_TOOLS, server_config),
        temperature=0,
        max_completion_tokens=150,
        model=model_name,
        tools=[WEATHER_TOOL],
        logprobs=False)
    choice = chat_completion.choices[0]
    stop_reason = chat_completion.choices[0].finish_reason
    output_text = chat_completion.choices[0].message.content

    # check to make sure we got text
    assert output_text is not None
    assert stop_reason != 'tool_calls'
    assert len(output_text) > 0

    # check to make sure no tool calls were returned
    assert (choice.message.tool_calls is None
            or len(choice.message.tool_calls) == 0)

    # make the same request, streaming
    stream = await client.chat.completions.create(
        messages=ensure_system_prompt(MESSAGES_WITHOUT_TOOLS, server_config),
        temperature=0,
        max_completion_tokens=150,
        model=model_name,
        logprobs=False,
        tools=[WEATHER_TOOL],
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
            assert delta.role == 'assistant'
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
    assert chunk.choices[0].finish_reason != 'tool_calls'
    assert len(chunks)
    assert "".join(chunks) == output_text
