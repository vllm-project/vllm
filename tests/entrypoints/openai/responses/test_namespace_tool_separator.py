# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import logging

import openai  # use the official client for correctness check
import pytest

logger = logging.getLogger(__name__)

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


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("tool_choice", ["required", "named"])
async def test_namespaced_custom_tool_round_trip(
    client: openai.AsyncOpenAI, stream: bool, tool_choice: str
):
    """Preserve generation and match the existing function-history contract."""
    custom_tools = [
        {
            "type": "namespace",
            "name": "sandbox",
            "description": "Sandbox text utilities.",
            "tools": [
                {
                    "type": "custom",
                    "name": "emit",
                    "description": "Emit the requested text unchanged.",
                    "format": {"type": "text"},
                }
            ],
        }
    ]
    prompt = [{"role": "user", "content": "Use sandbox.emit to emit hello."}]
    choice = (
        {"type": "custom", "name": "sandbox__emit"}
        if tool_choice == "named"
        else "required"
    )
    result = await client.responses.create(
        model=MODEL_NAME,
        input=prompt,
        tools=custom_tools,
        tool_choice=choice,
        temperature=0,
        max_output_tokens=256,
        stream=stream,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    if stream:
        events = [event async for event in result]
        assert events[-1].type == "response.completed"
        response = events[-1].response
    else:
        response = result

    assert response.status == "completed"
    calls = [item for item in response.output if item.type == "custom_tool_call"]
    assert len(calls) == 1, response.model_dump_json()
    call = calls[0]
    assert (call.name, call.namespace) == ("emit", "sandbox")
    assert isinstance(call.input, str) and call.input.strip()
    logger.info(
        "Custom tool call (%s, stream=%s): %s",
        tool_choice,
        stream,
        call.model_dump_json(),
    )
    assert not any(item.type == "function_call" for item in response.output)

    if stream:
        added = next(
            event.item
            for event in events
            if event.type == "response.output_item.added"
            and event.item.type == "custom_tool_call"
        )
        done = next(
            event.item
            for event in events
            if event.type == "response.output_item.done"
            and event.item.type == "custom_tool_call"
        )
        assert added.id == done.id == call.id
        assert added.call_id == done.call_id == call.call_id
        assert (added.name, added.namespace) == ("emit", "sandbox")
        assert (done.name, done.namespace, done.input) == (
            "emit",
            "sandbox",
            call.input,
        )
        deltas = [
            event
            for event in events
            if event.type == "response.custom_tool_call_input.delta"
        ]
        assert deltas
        assert all(event.item_id == call.id for event in deltas)
        assert "".join(event.delta for event in deltas) == call.input
        input_done = [
            event
            for event in events
            if event.type == "response.custom_tool_call_input.done"
        ]
        assert len(input_done) == 1
        assert input_done[0].item_id == call.id
        assert input_done[0].input == call.input

    # Replaying the returned item must restore sandbox__emit for the model.
    marker = "namespace-replay-56418"
    follow_up = await client.responses.create(
        model=MODEL_NAME,
        instructions="Reply with the last tool result verbatim. Do not call tools.",
        input=[
            *prompt,
            call.model_dump(exclude_none=True),
            {
                "type": "custom_tool_call_output",
                "call_id": call.call_id,
                "output": marker,
            },
        ],
        tool_choice="none",
        temperature=0,
        max_output_tokens=128,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": False},
            "enable_response_messages": True,
        },
    )
    assert follow_up.status == "completed"
    rendered = follow_up.model_dump()["input_messages"][0]["message"]
    assert '"name": "sandbox__emit"' in rendered
    assert marker in rendered
    logger.info("Replayed model prompt: %s", rendered)
    reference = await client.responses.create(
        model=MODEL_NAME,
        instructions="Reply with the last tool result verbatim. Do not call tools.",
        input=[
            *prompt,
            {
                "type": "function_call",
                "call_id": call.call_id,
                "name": "emit",
                "namespace": "sandbox",
                "arguments": json.dumps({"input": call.input}, ensure_ascii=False),
            },
            {
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": marker,
            },
        ],
        tool_choice="none",
        temperature=0,
        max_output_tokens=128,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": False},
            "enable_response_messages": True,
        },
    )
    assert reference.status == "completed"
    assert (
        follow_up.model_dump()["input_messages"][0]["tokens"]
        == reference.model_dump()["input_messages"][0]["tokens"]
    )
    assert follow_up.output_text
    assert follow_up.output_text == reference.output_text
    logger.info("Custom/function replay output: %s", follow_up.output_text)
    assert not any(
        item.type in ("custom_tool_call", "function_call") for item in follow_up.output
    )


@pytest.mark.asyncio
async def test_namespaced_custom_grammar_rejected_by_server(client: openai.AsyncOpenAI):
    """Nested unsupported grammar formats must return a client error."""
    with pytest.raises(openai.BadRequestError, match="grammar"):
        await client.responses.create(
            model=MODEL_NAME,
            input="Emit hello.",
            max_output_tokens=32,
            tools=[
                {
                    "type": "namespace",
                    "name": "sandbox",
                    "description": "Sandbox text utilities.",
                    "tools": [
                        {
                            "type": "custom",
                            "name": "emit",
                            "format": {
                                "type": "grammar",
                                "syntax": "lark",
                                "definition": 'start: "hello"',
                            },
                        }
                    ],
                }
            ],
        )
