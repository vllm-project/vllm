# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock

import pytest
import pytest_asyncio
from openai.types.responses import (
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseOutputItemAddedEvent,
)
from tests.utils import RemoteOpenAIServer

MODEL_NAME = "Qwen/Qwen3-0.6B"

INTEGRATION_TOOLS = [
    {
        "type": "function",
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "country": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city", "country", "unit"],
        },
    }
]

@pytest.fixture(scope="module")
def responses_server():
    args = [
        "--dtype",
        "half",
        "--enable-auto-tool-choice",
        "--structured-outputs-config.backend",
        "xgrammar",
        "--tool-call-parser",
        "hermes",
        "--reasoning-parser",
        "qwen3",
        "--gpu-memory-utilization",
        "0.3",
        "--max-model-len",
        "2048",
    ]
    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def responses_client(responses_server):
    async with responses_server.get_async_client() as async_client:
        yield async_client


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool_choice",
    [
        "auto",
    ],
)
async def test_responses_streaming_tool_calls_e2e(responses_client, tool_choice):
    stream = await responses_client.responses.create(
        model=MODEL_NAME,
        input=(
            "Use the weather tool to get the temperature for Berlin, Germany in "
            "fahrenheit"
        ),
        tools=INTEGRATION_TOOLS,
        tool_choice=tool_choice,
        stream=True,
        temperature=0,
    )

    added_event = None
    arg_chunks: list[str] = []
    final_arguments: str | None = None
    async for event in stream:
        if (
            isinstance(event, ResponseOutputItemAddedEvent)
            and getattr(event.item, "type", None) == "function_call"
        ):
            added_event = event
        if isinstance(event, ResponseFunctionCallArgumentsDeltaEvent):
            arg_chunks.append(event.delta)
        if isinstance(event, ResponseFunctionCallArgumentsDoneEvent):
            final_arguments = event.arguments

    assert added_event is not None
    assert final_arguments is not None
    assert final_arguments == "".join(arg_chunks)
    args_text = final_arguments.lower()
    assert "berlin" in args_text
    assert "germany" in args_text
    assert "fahrenheit" in args_text
