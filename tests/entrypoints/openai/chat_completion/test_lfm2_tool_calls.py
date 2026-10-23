# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest

from vllm.tokenizers import get_tokenizer

from ....utils import RemoteOpenAIServer

MODEL_NAME = "LiquidAI/LFM2.5-2.6B"

# OpenAI tool names may contain hyphens; LFM2 emits them verbatim.
WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get-weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}
MESSAGES = [{"role": "user", "content": "What is the weather in Paris?"}]


@pytest.fixture(scope="module")
def server():
    args = [
        "--max-model-len",
        "4096",
        "--max-num-seqs",
        "8",
        "--gpu-memory-utilization",
        "0.7",
        "--enforce-eager",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "lfm2",
    ]
    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_hyphenated_tool_name_is_parsed(server, stream: bool):
    async with server.get_async_client() as client:
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=MESSAGES,
            tools=[WEATHER_TOOL],
            temperature=0,
            stream=stream,
        )
        if stream:
            names: dict[int, str] = {}
            arguments: dict[int, str] = {}
            async for chunk in response:
                if not chunk.choices:
                    continue
                for call in chunk.choices[0].delta.tool_calls or []:
                    names[call.index] = names.get(call.index, "") + (
                        call.function.name or ""
                    )
                    arguments[call.index] = arguments.get(call.index, "") + (
                        call.function.arguments or ""
                    )
            calls = [(names[i], arguments[i]) for i in sorted(names)]
        else:
            calls = [
                (call.function.name, call.function.arguments)
                for call in response.choices[0].message.tool_calls or []
            ]

    assert [name for name, _ in calls] == ["get-weather"]
    assert "Paris" in json.loads(calls[0][1])["city"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool_choice",
    ["required", {"type": "function", "function": {"name": "get-weather"}}],
)
async def test_forced_tool_choice_emits_native_tool_call(server, tool_choice):
    """Forced calls must be decoded as LFM2's own tool-call block, not as the
    generic JSON tool-call list, and parsed back from it."""
    async with server.get_async_client() as client:
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=MESSAGES,
            tools=[WEATHER_TOOL],
            tool_choice=tool_choice,
            temperature=0,
            extra_body={"return_token_ids": True},
        )

    choice = response.choices[0]
    tool_call_start = get_tokenizer(MODEL_NAME).convert_tokens_to_ids(
        "<|tool_call_start|>"
    )
    assert tool_call_start in choice.token_ids
    assert [call.function.name for call in choice.message.tool_calls] == ["get-weather"]
    assert (
        "Paris" in json.loads(choice.message.tool_calls[0].function.arguments)["city"]
    )
