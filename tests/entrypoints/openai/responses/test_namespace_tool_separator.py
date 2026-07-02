# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import openai  # use the official client for correctness check
import pytest

MODEL_NAME = "Qwen/Qwen3-1.7B"
NAMESPACE = "mcp__computer_use"
TOOL_NAME = "get_app_state"
FLAT_TOOL_NAME = f"{NAMESPACE}__{TOOL_NAME}"

tools = [
    {
        "type": "namespace",
        "name": NAMESPACE,
        "description": "Computer control tools.",
        "tools": [
            {
                "type": "function",
                "name": TOOL_NAME,
                "description": "Get the current state of a desktop application.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "app": {
                            "type": "string",
                            "description": "Application name, for example Chrome.",
                        }
                    },
                    "required": ["app"],
                    "additionalProperties": False,
                },
            }
        ],
    }
]

prompt = [
    {
        "role": "user",
        "content": "Use the computer app state tool to inspect Google Chrome.",
    },
]


def _assert_namespace_tool_call(tool_call) -> None:
    assert tool_call.type == "function_call"
    assert tool_call.name == TOOL_NAME
    assert tool_call.namespace == NAMESPACE
    assert tool_call.name != FLAT_TOOL_NAME

    args = json.loads(tool_call.arguments)
    assert args["app"]


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_namespace_tool_separator(client: openai.AsyncOpenAI, model_name: str):
    response = await client.responses.create(
        model=model_name,
        input=prompt,
        tools=tools,
        tool_choice={"type": "function", "name": FLAT_TOOL_NAME},
        temperature=0.0,
    )

    assert len(response.output) >= 1
    tool_call = next(
        (out for out in response.output if out.type == "function_call"), None
    )
    assert tool_call is not None
    _assert_namespace_tool_call(tool_call)


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_namespace_tool_separator_streaming(
    client: openai.AsyncOpenAI, model_name: str
):
    stream = await client.responses.create(
        model=model_name,
        input=prompt,
        tools=tools,
        tool_choice={"type": "function", "name": FLAT_TOOL_NAME},
        temperature=0.0,
        stream=True,
    )
    events = [event async for event in stream]

    added_call = next(
        (
            event.item
            for event in events
            if event.type == "response.output_item.added"
            and getattr(event.item, "type", None) == "function_call"
        ),
        None,
    )
    done_call = next(
        (
            event.item
            for event in events
            if event.type == "response.output_item.done"
            and getattr(event.item, "type", None) == "function_call"
        ),
        None,
    )

    assert added_call is not None
    assert added_call.name == TOOL_NAME
    assert added_call.namespace == NAMESPACE

    assert done_call is not None
    _assert_namespace_tool_call(done_call)
