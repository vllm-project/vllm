# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import openai  # use the official client for correctness check
import pytest
import pytest_asyncio

from ...utils import RemoteOpenAIServer

# a reasoning and tool calling model
MODEL_NAME = "Qwen/QwQ-32B"


@pytest.fixture(scope="module")
def server():
    args = [
        "--max-model-len",
        "8192",
        "--enforce-eager",
        "--reasoning-parser",
        "deepseek_r1",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "hermes",
    ]

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city to find the weather for, e.g. "
                        "'San Francisco'",
                    },
                    "state": {
                        "type": "string",
                        "description": "the two-letter abbreviation for the state that "
                        "the city is in, e.g. 'CA' which would mean 'California'",
                    },
                    "unit": {
                        "type": "string",
                        "description": "The unit to fetch the temperature in",
                        "enum": ["celsius", "fahrenheit"],
                    },
                },
                "required": ["city", "state", "unit"],
            },
        },
    }
]

MESSAGES = [
    {"role": "user", "content": "Hi! How are you doing today?"},
    {"role": "assistant", "content": "I'm doing well! How can I help you?"},
    {
        "role": "user",
        "content": "Can you tell me what the temperate will be in Dallas, "
        "in fahrenheit?",
    },
]

FUNC_NAME = "get_current_weather"
FUNC_ARGS = """{"city": "Dallas", "state": "TX", "unit": "fahrenheit"}"""


def extract_reasoning_and_calls(chunks: list):
    reasoning = ""
    tool_call_idx = -1
    arguments = []
    function_names = []
    for chunk in chunks:
        if chunk.choices[0].delta.tool_calls:
            tool_call = chunk.choices[0].delta.tool_calls[0]
            if tool_call.index != tool_call_idx:
                tool_call_idx = chunk.choices[0].delta.tool_calls[0].index
                arguments.append("")
                function_names.append("")

            if tool_call.function:
                if tool_call.function.name:
                    function_names[tool_call_idx] = tool_call.function.name

                if tool_call.function.arguments:
                    arguments[tool_call_idx] += tool_call.function.arguments
        else:
            if hasattr(chunk.choices[0].delta, "reasoning"):
                reasoning += chunk.choices[0].delta.reasoning
    return reasoning, arguments, function_names


# test streaming
@pytest.mark.asyncio
async def test_chat_streaming_of_tool_and_reasoning(client: openai.AsyncOpenAI):
    stream = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=MESSAGES,
        tools=TOOLS,
        temperature=0.0,
        stream=True,
    )

    chunks = []
    async for chunk in stream:
        chunks.append(chunk)

    reasoning, arguments, function_names = extract_reasoning_and_calls(chunks)
    assert len(reasoning) > 0
    assert len(function_names) > 0 and function_names[0] == FUNC_NAME
    assert len(arguments) > 0 and arguments[0] == FUNC_ARGS


# test full generate
@pytest.mark.asyncio
async def test_chat_full_of_tool_and_reasoning(client: openai.AsyncOpenAI):
    tool_calls = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=MESSAGES,
        tools=TOOLS,
        temperature=0.0,
        stream=False,
    )

    assert len(tool_calls.choices[0].message.reasoning) > 0
    assert tool_calls.choices[0].message.tool_calls[0].function.name == FUNC_NAME
    assert tool_calls.choices[0].message.tool_calls[0].function.arguments == FUNC_ARGS


# test that content does not leak into final chunk when finish_reason=tool_calls
@pytest.mark.asyncio
async def test_no_content_leak_when_finish_reason_tool_calls(
    client: openai.AsyncOpenAI,
):
    """
    Test that when finish_reason='tool_calls', the final chunk does not
    contain any content field. This prevents reasoning_content from leaking
    into content, which violates OpenAI's schema contract.

    This test specifically targets the bug where leftover reasoning buffers
    (especially from speculative decoding) were incorrectly flushed into
    the content field in the final streamed chunk.
    """
    stream = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=MESSAGES,
        tools=TOOLS,
        temperature=0.0,
        stream=True,
        tool_choice="auto",
        include_reasoning=True,
    )

    chunks = []
    final_chunk = None
    async for chunk in stream:
        chunks.append(chunk)
        # Track the final chunk with finish_reason
        if chunk.choices and chunk.choices[0].finish_reason:
            final_chunk = chunk

    # Ensure we got a final chunk with tool_calls
    assert final_chunk is not None, "Expected a final chunk with finish_reason"
    assert final_chunk.choices[0].finish_reason == "tool_calls", (
        "Expected finish_reason to be 'tool_calls'"
    )

    delta = final_chunk.choices[0].delta

    # Per OpenAI spec, when finish_reason='tool_calls', content must be null/absent
    # This is the core fix: prevent reasoning_content from leaking into content
    assert delta.content is None or delta.content == "", (
        f"Final chunk with finish_reason='tool_calls' must not have content. "
        f"Got content='{delta.content}'. This indicates reasoning_content leaked "
        f"into content field."
    )

    # Also ensure reasoning fields are not present in final chunk
    # (they should only appear in earlier chunks)
    reasoning = getattr(delta, "reasoning", None)
    reasoning_content = getattr(delta, "reasoning_content", None)
    assert reasoning is None or reasoning == "", (
        "Final chunk with tool_calls should not have reasoning field"
    )
    assert reasoning_content is None or reasoning_content == "", (
        "Final chunk with tool_calls should not have reasoning_content field"
    )

    # Verify tool_calls are present (the expected behavior)
    assert delta.tool_calls is not None and len(delta.tool_calls) > 0, (
        "Final chunk with finish_reason='tool_calls' must have tool_calls"
    )

    # Verify reasoning was streamed in earlier chunks (not in final)
    reasoning_found_in_earlier_chunks = False
    for chunk in chunks[:-1]:  # All chunks except the final one
        if chunk.choices:
            delta = chunk.choices[0].delta
            if hasattr(delta, "reasoning") and delta.reasoning:
                reasoning_found_in_earlier_chunks = True
                break
            if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                reasoning_found_in_earlier_chunks = True
                break

    assert reasoning_found_in_earlier_chunks, (
        "Reasoning should be streamed in earlier chunks, not in final chunk"
    )
