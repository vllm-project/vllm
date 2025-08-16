# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest

from ....utils import RemoteOpenAIServer

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
LORA_MODEL = "minpeter/LoRA-Llama-3.2-1B-tool-vllm-ci"

SERVER_ARGS = [
    "--enforce-eager",
    "--enable-auto-tool-choice",
    "--tool-call-parser",
    "hermes",
    "--enable-lora",
    "--lora-modules",
    f"{LORA_MODEL}={LORA_MODEL}",
]

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description":
                    "The city and state, e.g. San Francisco, CA",
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"]
                },
            },
            "required": ["location"],
        },
    },
}]

MESSAGES = [{"role": "user", "content": "What's the weather like in Boston?"}]


@pytest.mark.asyncio
async def test_non_streaming_tool_call():
    """Test tool call in non-streaming mode."""
    with RemoteOpenAIServer(MODEL_NAME, SERVER_ARGS) as server:
        client = server.get_async_client()

        response = await client.chat.completions.create(
            model=LORA_MODEL,
            messages=MESSAGES,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.0,
        )

        assert response.choices
        choice = response.choices[0]
        message = choice.message

        assert choice.finish_reason == "tool_calls"
        assert message.tool_calls is not None

        tool_call = message.tool_calls[0]
        assert tool_call.type == "function"
        assert tool_call.function.name == "get_current_weather"

        arguments = json.loads(tool_call.function.arguments)
        assert "location" in arguments
        assert "Boston" in arguments["location"]
        print("\n[Non-Streaming Test Passed]")
        print(f"Tool Call: {tool_call.function.name}")
        print(f"Arguments: {arguments}")


@pytest.mark.asyncio
async def test_streaming_tool_call():
    """Test tool call in streaming mode."""
    with RemoteOpenAIServer(MODEL_NAME, SERVER_ARGS) as server:
        client = server.get_async_client()

        stream = await client.chat.completions.create(
            model=LORA_MODEL,
            messages=MESSAGES,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.0,
            stream=True,
        )

        tool_call_chunks = {}
        async for chunk in stream:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta
            if not delta or not delta.tool_calls:
                continue

            for tool_chunk in delta.tool_calls:
                index = tool_chunk.index
                if index not in tool_call_chunks:
                    tool_call_chunks[index] = {"name": "", "arguments": ""}

                if tool_chunk.function.name:
                    tool_call_chunks[index]["name"] += tool_chunk.function.name
                if tool_chunk.function.arguments:
                    tool_call_chunks[index][
                        "arguments"] += tool_chunk.function.arguments

        assert len(tool_call_chunks) == 1
        reconstructed_tool_call = tool_call_chunks[0]

        assert reconstructed_tool_call["name"] == "get_current_weather"

        arguments = json.loads(reconstructed_tool_call["arguments"])
        assert "location" in arguments
        assert "Boston" in arguments["location"]
        print("\n[Streaming Test Passed]")
        print(f"Reconstructed Tool Call: {reconstructed_tool_call['name']}")
        print(f"Reconstructed Arguments: {arguments}")
