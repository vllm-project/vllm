# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Integration tests for the Harmony-based Responses API."""

from __future__ import annotations

import importlib.util
import json
import logging
import time
from typing import Any

import pytest
import pytest_asyncio
import requests
from openai import BadRequestError, NotFoundError, OpenAI
from openai_harmony import Message

from ....utils import RemoteOpenAIServer
from .conftest import (
    BASE_TEST_ENV,
    events_contain_type,
    has_output_type,
    retry_for_tool_call,
    retry_streaming_for,
    validate_streaming_event_stack,
)

logger = logging.getLogger(__name__)

MODEL_NAME = "openai/gpt-oss-20b"

GET_WEATHER_SCHEMA = {
    "type": "function",
    "name": "get_weather",
    "description": "Get current temperature for provided coordinates in celsius.",  # noqa
    "parameters": {
        "type": "object",
        "properties": {
            "latitude": {"type": "number"},
            "longitude": {"type": "number"},
        },
        "required": ["latitude", "longitude"],
        "additionalProperties": False,
    },
    "strict": True,
}


def get_weather(latitude, longitude):
    try:
        response = requests.get(
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={latitude}&longitude={longitude}"
            f"&current=temperature_2m,wind_speed_10m"
            f"&hourly=temperature_2m,relative_humidity_2m,"
            f"wind_speed_10m",
            timeout=10,
        )
        data = response.json()
        return data["current"]["temperature_2m"]
    except (requests.RequestException, KeyError) as e:
        logger.warning(
            "External weather API call failed (%s), "
            "returning fake value. This does not affect "
            "test correctness — only the tool-calling "
            "protocol is under test.",
            e,
        )
        return 15.0


def get_place_to_travel():
    return "Paris"


def get_horoscope(sign):
    return f"{sign}: Next Tuesday you will befriend a baby otter."


def call_function(name, args):
    logger.info("Calling function %s with args %s", name, args)
    dispatch = {
        "get_weather": lambda: get_weather(**args),
        "get_place_to_travel": lambda: get_place_to_travel(),
        "get_horoscope": lambda: get_horoscope(**args),
    }
    if name not in dispatch:
        raise ValueError(f"Unknown function: {name}")
    result = dispatch[name]()
    logger.info("Function %s returned: %s", name, result)
    return result


@pytest.fixture(scope="module")
def server():
    assert importlib.util.find_spec("gpt_oss") is not None, (
        "Harmony tests require gpt_oss package to be installed"
    )
    args = [
        "--enforce-eager",
        "--tool-server",
        "demo",
        "--max_model_len",
        "5000",
    ]
    env_dict = {
        **BASE_TEST_ENV,
        "VLLM_ENABLE_RESPONSES_API_STORE": "1",
        "PYTHON_EXECUTION_BACKEND": "dangerously_use_uv",
        "VLLM_GPT_OSS_SYSTEM_TOOL_MCP_LABELS": (
            "code_interpreter,container,web_search_preview"
        ),
        "VLLM_GPT_OSS_HARMONY_SYSTEM_INSTRUCTIONS": "1",
    }
    with RemoteOpenAIServer(MODEL_NAME, args, env_dict=env_dict) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_basic(client: OpenAI, model_name: str):
    response = await client.responses.create(
        model=model_name,
        input="What is 123 * 456?",
    )
    assert response is not None
    print("response: ", response)
    assert response.status == "completed"


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_basic_with_instructions(client: OpenAI, model_name: str):
    response = await client.responses.create(
        model=model_name,
        input="What is 123 * 456?",
        instructions="Respond in Korean.",
    )
    assert response is not None
    assert response.status == "completed"


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_basic_with_reasoning_effort(client: OpenAI, model_name: str):
    response = await client.responses.create(
        model=model_name,
        input="What is the capital of South Korea?",
        reasoning={"effort": "low"},
    )
    assert response is not None
    assert response.status == "completed"


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_max_tokens(client: OpenAI, model_name: str):
    response = await client.responses.create(
        model=model_name,
        input="What is the first paragraph of Moby Dick?",
        reasoning={"effort": "low"},
        max_output_tokens=30,
    )
    assert response is not None
    assert response.status == "incomplete"
    assert response.incomplete_details.reason == "max_output_tokens"


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_chat(client: OpenAI, model_name: str):
    response = await client.responses.create(
        model=model_name,
        input=[
            {"role": "system", "content": "Respond in Korean."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hello! How can I help you today?"},
            {"role": "user", "content": "What is 123 * 456? Explain your answer."},
        ],
    )
    assert response is not None
    assert response.status == "completed"


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_chat_with_input_type(client: OpenAI, model_name: str):
    response = await client.responses.create(
        model=model_name,
        input=[
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "What is 123 * 456?"}],
            },
        ],
    )
    assert response is not None
    assert response.status == "completed"


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_structured_output(client: OpenAI, model_name: str):
    response = await client.responses.create(
        model=model_name,
        input=[
            {"role": "system", "content": "Extract the event information."},
            {
                "role": "user",
                "content": "Alice and Bob are going to a science fair on Friday.",
            },
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "calendar_event",
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "date": {"type": "string"},
                        "participants": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["name", "date", "participants"],
                    "additionalProperties": False,
                },
                "description": "A calendar event.",
                "strict": True,
            }
        },
    )
    assert response is not None
    assert response.status == "completed"


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_structured_output_with_parse(client: OpenAI, model_name: str):
    from pydantic import BaseModel

    class CalendarEvent(BaseModel):
        name: str
        date: str
        participants: list[str]

    response = await client.responses.parse(
        model=model_name,
        input="Alice and Bob are going to a science fair on Friday",
        instructions="Extract the event information",
        text_format=CalendarEvent,
    )
    assert response is not None
    assert response.status == "completed"


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_store(client: OpenAI, model_name: str):
    for store in [True, False]:
        response = await client.responses.create(
            model=model_name,
            input="What is 123 * 456?",
            store=store,
        )
        assert response is not None

        try:
            _retrieved_response = await client.responses.retrieve(response.id)
            is_not_found = False
        except NotFoundError:
            is_not_found = True

        assert is_not_found == (not store), (
            f"store={store}: expected not_found={not store}, got {is_not_found}"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_background(client: OpenAI, model_name: str):
    response = await client.responses.create(
        model=model_name,
        input="What is 123 * 456?",
        background=True,
    )
    assert response is not None

    retries = 0
    max_retries = 30
    while retries < max_retries:
        response = await client.responses.retrieve(response.id)
        if response.status == "completed":
            break
        time.sleep(1)
        retries += 1

    assert response.status == "completed"


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_background_cancel(client: OpenAI, model_name: str):
    response = await client.responses.create(
        model=model_name,
        input="Write a long story about a cat.",
        background=True,
    )
    assert response is not None
    time.sleep(1)

    cancelled_response = await client.responses.cancel(response.id)
    assert cancelled_response is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_stateful_multi_turn(client: OpenAI, model_name: str):
    response1 = await client.responses.create(
        model=model_name, input="What is 123 * 456?"
    )
    assert response1.status == "completed"

    response2 = await client.responses.create(
        model=model_name,
        input="What if I increase both numbers by 1?",
        previous_response_id=response1.id,
    )
    assert response2.status == "completed"

    response3 = await client.responses.create(
        model=model_name,
        input="Divide the result by 2.",
        previous_response_id=response2.id,
    )
    assert response3.status == "completed"


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_streaming_types(
    pairs_of_event_types: dict[str, str], client: OpenAI, model_name: str
):
    stream = await client.responses.create(
        model=model_name,
        input="tell me a story about a cat in 20 words",
        reasoning={"effort": "low"},
        tools=[],
        stream=True,
        background=False,
    )
    events = []
    async for event in stream:
        events.append(event)

    validate_streaming_event_stack(events, pairs_of_event_types)


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_function_calling_with_streaming_types(
    pairs_of_event_types: dict[str, str], client: OpenAI, model_name: str
):
    """Streaming event nesting for function-calling responses."""

    def _has_function_events(evts: list) -> bool:
        return events_contain_type(evts, "function_call_arguments")

    events = await retry_streaming_for(
        client,
        model=model_name,
        validate_events=_has_function_events,
        input=[{"role": "user", "content": "What's the weather like in Paris today?"}],
        tools=[GET_WEATHER_SCHEMA],
        temperature=0.0,
    )

    validate_streaming_event_stack(events, pairs_of_event_types)


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
@pytest.mark.parametrize("background", [True, False])
async def test_streaming(client: OpenAI, model_name: str, background: bool):
    # TODO: Add back when web search and code interpreter are available in CI
    prompts = [
        "tell me a story about a cat in 20 words",
        "What is 123 * 456? Use python to calculate the result.",
        # "When did Jensen found NVIDIA? Search it and answer the year only.",
    ]

    for prompt in prompts:
        stream = await client.responses.create(
            model=model_name,
            input=prompt,
            reasoning={"effort": "low"},
            tools=[
                # {
                #     "type": "web_search_preview"
                # },
                {"type": "code_interpreter", "container": {"type": "auto"}},
            ],
            stream=True,
            background=background,
            extra_body={"enable_response_messages": True},
        )

        current_item_id = ""
        current_content_index = -1

        events = []
        current_event_mode = None
        resp_id = None
        checked_response_completed = False

        async for event in stream:
            if event.type == "response.created":
                resp_id = event.response.id

            # Validate custom fields on response-level events
            if event.type in [
                "response.completed",
                "response.in_progress",
                "response.created",
            ]:
                assert "input_messages" in event.response.model_extra
                assert "output_messages" in event.response.model_extra
                if event.type == "response.completed":
                    # make sure the serialization of content works
                    for msg in event.response.model_extra["output_messages"]:
                        # make sure we can convert the messages back into harmony
                        Message.from_dict(msg)

                    for msg in event.response.model_extra["input_messages"]:
                        # make sure we can convert the messages back into harmony
                        Message.from_dict(msg)
                    checked_response_completed = True

            if current_event_mode != event.type:
                current_event_mode = event.type
                logger.debug("[%s] ", event.type)

            # Verify item IDs
            if event.type == "response.output_item.added":
                assert event.item.id != current_item_id
                current_item_id = event.item.id
            elif event.type in [
                "response.output_text.delta",
                "response.reasoning_text.delta",
            ]:
                assert event.item_id == current_item_id

            # Verify content indices
            if event.type in [
                "response.content_part.added",
                "response.reasoning_part.added",
            ]:
                assert event.content_index != current_content_index
                current_content_index = event.content_index
            elif event.type in [
                "response.output_text.delta",
                "response.reasoning_text.delta",
            ]:
                assert event.content_index == current_content_index

            events.append(event)

        assert len(events) > 0
        assert events[-1].response.output, "Final response should have output"
        assert checked_response_completed

        if background:
            starting_after = 5
            async with await client.responses.retrieve(
                response_id=resp_id, stream=True, starting_after=starting_after
            ) as replay_stream:
                counter = starting_after
                async for event in replay_stream:
                    counter += 1
                    assert event == events[counter]
            assert counter == len(events) - 1


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
@pytest.mark.skip(reason="Web search tool is not available in CI yet.")
async def test_web_search(client: OpenAI, model_name: str):
    response = await client.responses.create(
        model=model_name,
        input="Who is the president of South Korea as of now?",
        tools=[{"type": "web_search_preview"}],
    )
    assert response is not None
    assert response.status == "completed"


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_code_interpreter(client: OpenAI, model_name: str):
    timeout_value = client.timeout * 3
    client_with_timeout = client.with_options(timeout=timeout_value)

    response = await client_with_timeout.responses.create(
        model=model_name,
        input=(
            "What's the first 4 digits after the decimal point of "
            "cube root of `19910212 * 20250910`? "
            "Show only the digits. The python interpreter is not stateful "
            "and you must print to see the output."
        ),
        tools=[{"type": "code_interpreter", "container": {"type": "auto"}}],
        temperature=0.0,
    )
    assert response is not None
    assert response.status == "completed"
    assert response.usage.output_tokens_details.tool_output_tokens > 0

    for item in response.output:
        if item.type == "message":
            output_string = item.content[0].text
            assert "5846" in output_string, (
                f"Expected '5846' in output, got: {output_string}"
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_reasoning_item(client: OpenAI, model_name: str):
    response = await client.responses.create(
        model=model_name,
        input=[
            {"type": "message", "content": "Hello.", "role": "user"},
            {
                "type": "reasoning",
                "id": "lol",
                "content": [
                    {"type": "reasoning_text", "text": "We need to respond: greeting."}
                ],
                "summary": [],
            },
        ],
        temperature=0.0,
    )
    assert response is not None
    assert response.status == "completed"


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_function_calling(client: OpenAI, model_name: str):
    tools = [GET_WEATHER_SCHEMA]

    response = await retry_for_tool_call(
        client,
        model=model_name,
        expected_tool_type="function_call",
        input="What's the weather like in Paris today?",
        tools=tools,
        temperature=0.0,
        extra_body={"request_id": "test_function_calling_non_resp"},
    )
    assert response.status == "completed"
    assert has_output_type(response, "function_call"), (
        f"Expected function_call in output, got: "
        f"{[getattr(o, 'type', None) for o in response.output]}"
    )

    tool_call = next(o for o in response.output if o.type == "function_call")
    args = json.loads(tool_call.arguments)
    result = call_function(tool_call.name, args)

    response_2 = await client.responses.create(
        model=model_name,
        input=[
            {
                "type": "function_call_output",
                "call_id": tool_call.call_id,
                "output": str(result),
            }
        ],
        tools=tools,
        previous_response_id=response.id,
        temperature=0.0,
    )
    assert response_2.status == "completed"
    assert response_2.output_text is not None

    # NOTE: chain-of-thought should be removed.
    response_3 = await client.responses.create(
        model=model_name,
        input="What's the weather like in Paris today?",
        tools=tools,
        previous_response_id=response_2.id,
        temperature=0.0,
    )
    assert response_3.status == "completed"
    assert response_3.output_text is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_function_calling_multi_turn(client: OpenAI, model_name: str):
    """Multi-tool, multi-turn function calling with retry at API level."""
    tools = [
        {
            "type": "function",
            "name": "get_place_to_travel",
            "description": "Get a random place to travel",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
            "strict": True,
        },
        GET_WEATHER_SCHEMA,
    ]

    # Turn 1: model should call one of the tools
    response = await retry_for_tool_call(
        client,
        model=model_name,
        expected_tool_type="function_call",
        input="Help me plan a trip to a random place. And tell me the weather there.",
        tools=tools,
        temperature=0.0,
    )
    assert response.status == "completed"
    assert has_output_type(response, "function_call"), (
        f"Turn 1: expected function_call, got: "
        f"{[getattr(o, 'type', None) for o in response.output]}"
    )

    tool_call = next(o for o in response.output if o.type == "function_call")
    result = call_function(tool_call.name, json.loads(tool_call.arguments))

    # Turn 2
    response_2 = await retry_for_tool_call(
        client,
        model=model_name,
        expected_tool_type="function_call",
        input=[
            {
                "type": "function_call_output",
                "call_id": tool_call.call_id,
                "output": str(result),
            }
        ],
        tools=tools,
        previous_response_id=response.id,
        temperature=0.0,
    )
    assert response_2.status == "completed"

    # If model produced another tool call, execute it
    if has_output_type(response_2, "function_call"):
        tool_call_2 = next(o for o in response_2.output if o.type == "function_call")
        result_2 = call_function(tool_call_2.name, json.loads(tool_call_2.arguments))
        response_3 = await client.responses.create(
            model=model_name,
            input=[
                {
                    "type": "function_call_output",
                    "call_id": tool_call_2.call_id,
                    "output": str(result_2),
                }
            ],
            tools=tools,
            previous_response_id=response_2.id,
            temperature=0.0,
        )
        assert response_3.status == "completed"
        assert response_3.output_text is not None
    else:
        # Model went straight to answering - acceptable but unexpected.
        # Log as warning so it shows up in CI without failing the test.
        assert response_2.output_text is not None
        pytest.xfail(
            "Model went straight to answering instead of calling a "
            "second tool. Valid behaviour but not the expected path."
            "If this happens consistently, the prompt or model may have "
            "changed behaviour."
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_function_calling_required(client: OpenAI, model_name: str):
    tools = [GET_WEATHER_SCHEMA]

    with pytest.raises(BadRequestError):
        await client.responses.create(
            model=model_name,
            input="What's the weather like in Paris today?",
            tools=tools,
            tool_choice="required",
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_system_message_with_tools(client: OpenAI, model_name: str):
    from vllm.entrypoints.openai.parser.harmony_utils import get_system_message

    # Test with custom tools enabled - commentary channel should be available
    sys_msg = get_system_message(with_custom_tools=True)
    valid_channels = sys_msg.content[0].channel_config.valid_channels
    assert "commentary" in valid_channels

    # Test with custom tools disabled - commentary channel should be removed
    sys_msg = get_system_message(with_custom_tools=False)
    valid_channels = sys_msg.content[0].channel_config.valid_channels
    assert "commentary" not in valid_channels


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_function_calling_full_history(client: OpenAI, model_name: str):
    tools = [GET_WEATHER_SCHEMA]

    input_messages = [
        {"role": "user", "content": "What's the weather like in Paris today?"}
    ]

    response = await retry_for_tool_call(
        client,
        model=model_name,
        expected_tool_type="function_call",
        input=input_messages,
        tools=tools,
        temperature=0.0,
    )
    assert response.status == "completed"

    tool_call = next((o for o in response.output if o.type == "function_call"), None)
    assert tool_call is not None, (
        f"Expected function_call in output, got: "
        f"{[getattr(o, 'type', None) for o in response.output]}"
    )

    result = call_function(tool_call.name, json.loads(tool_call.arguments))

    input_messages.extend(response.output)
    input_messages.append(
        {  # append result message
            "type": "function_call_output",
            "call_id": tool_call.call_id,
            "output": str(result),
        }
    )

    response_2 = await client.responses.create(
        model=model_name,
        input=input_messages,
        tools=tools,
        temperature=0.0,
    )
    assert response_2.status == "completed"
    assert response_2.output_text is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_function_calling_with_stream(client: OpenAI, model_name: str):
    """Function calling via streaming, with retry for non-determinism."""
    tools = [GET_WEATHER_SCHEMA]
    input_list = [
        {"role": "user", "content": "What's the weather like in Paris today?"},
    ]

    def _has_function_call(evts: list) -> bool:
        return any(
            getattr(e, "type", "") == "response.output_item.added"
            and getattr(getattr(e, "item", None), "type", None) == "function_call"
            for e in evts
        )

    events = await retry_streaming_for(
        client,
        model=model_name,
        validate_events=_has_function_call,
        input=input_list,
        tools=tools,
        temperature=0.0,
    )

    # Parse tool calls from events
    final_tool_calls: dict[int, Any] = {}
    for event in events:
        if event.type == "response.output_item.added":
            if getattr(event.item, "type", None) == "function_call":
                final_tool_calls[event.output_index] = event.item
        elif event.type == "response.function_call_arguments.delta":
            tc = final_tool_calls.get(event.output_index)
            if tc:
                tc.arguments += event.delta
        elif event.type == "response.function_call_arguments.done":
            tc = final_tool_calls.get(event.output_index)
            if tc:
                assert event.arguments == tc.arguments

    # Find get_weather call
    tool_call = None
    result = None
    for tc in final_tool_calls.values():
        if getattr(tc, "type", None) == "function_call" and tc.name == "get_weather":
            args = json.loads(tc.arguments)
            result = call_function(tc.name, args)
            tool_call = tc
            input_list.append(tc)
            break

    assert tool_call is not None, (
        "Expected model to call 'get_weather', "
        f"but got: {[getattr(tc, 'name', None) for tc in final_tool_calls.values()]}"
    )

    # Second turn with the tool result
    response = await client.responses.create(
        model=model_name,
        input=input_list
        + [
            {
                "type": "function_call_output",
                "call_id": tool_call.call_id,
                "output": str(result),
            }
        ],
        tools=tools,
        stream=True,
        temperature=0.0,
    )
    async for event in response:
        # check that no function call events in the stream
        assert event.type != "response.function_call_arguments.delta"
        assert event.type != "response.function_call_arguments.done"
        # check that the response contains output text
        if event.type == "response.completed":
            assert len(event.response.output) > 0
            assert event.response.output_text is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_function_calling_no_code_interpreter_events(
    client: OpenAI, model_name: str
):
    """Verify that function calls don't trigger code_interpreter events.

    Uses retry_streaming_for to handle non-determinism: the model might not
    always produce a function_call, but if it does, code_interpreter events
    should NEVER appear.
    """
    tools = [GET_WEATHER_SCHEMA]
    input_list = [
        {"role": "user", "content": "What's the weather like in Paris today?"},
    ]

    def _has_function_call(evts: list) -> bool:
        return any(
            getattr(e, "type", "") == "response.output_item.added"
            and getattr(getattr(e, "item", None), "type", None) == "function_call"
            for e in evts
        )

    events = await retry_streaming_for(
        client,
        model=model_name,
        validate_events=_has_function_call,
        input=input_list,
        tools=tools,
        temperature=0.0,
    )

    event_types_seen = {e.type for e in events}
    function_call_found = _has_function_call(events)

    assert function_call_found, (
        f"Expected to see a function_call after retries. "
        f"Event types: {sorted(event_types_seen)}"
    )

    # The actual invariant under test
    for event in events:
        assert "code_interpreter" not in event.type, (
            f"Found code_interpreter event '{event.type}' during function call. "
            "Function calls should only emit function_call events."
        )

    # Verify we saw the correct function call event types
    assert (
        "response.function_call_arguments.delta" in event_types_seen
        or "response.function_call_arguments.done" in event_types_seen
    ), "Expected to see function_call_arguments events"


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_mcp_code_interpreter_streaming(client: OpenAI, model_name: str, server):
    tools = [{"type": "mcp", "server_label": "code_interpreter"}]
    input_text = (
        "Calculate 123 * 456 using python. "
        "The python interpreter is not stateful and you must "
        "print to see the output."
    )

    def _has_mcp_call(evts: list) -> bool:
        return events_contain_type(evts, "mcp_call")

    events = await retry_streaming_for(
        client,
        model=model_name,
        validate_events=_has_mcp_call,
        input=input_text,
        tools=tools,
        temperature=0.0,
        instructions=(
            "You must use the Python tool to execute code. Never simulate execution."
        ),
    )

    event_types = [e.type for e in events]
    event_types_set = set(event_types)
    logger.info(
        "\n====== MCP Streaming Diagnostics ======\n"
        "Event count: %d\n"
        "Event types (in order): %s\n"
        "Unique event types: %s\n"
        "=======================================",
        len(events),
        event_types,
        sorted(event_types_set),
    )

    # Verify the full MCP streaming lifecycle
    assert "response.output_item.added" in event_types_set, (
        f"MCP call was not added. Events: {sorted(event_types_set)}"
    )
    assert "response.mcp_call.in_progress" in event_types_set, (
        f"MCP call in_progress not seen. Events: {sorted(event_types_set)}"
    )
    assert "response.mcp_call_arguments.delta" in event_types_set, (
        f"MCP arguments delta not seen. Events: {sorted(event_types_set)}"
    )
    assert "response.mcp_call_arguments.done" in event_types_set, (
        f"MCP arguments done not seen. Events: {sorted(event_types_set)}"
    )
    assert "response.mcp_call.completed" in event_types_set, (
        f"MCP call completed not seen. Events: {sorted(event_types_set)}"
    )
    assert "response.output_item.done" in event_types_set, (
        f"MCP item done not seen. Events: {sorted(event_types_set)}"
    )

    # Validate specific MCP event details
    for event in events:
        if event.type == "response.output_item.added":
            if hasattr(event.item, "type") and event.item.type == "mcp_call":
                assert event.item.name == "python"
                assert event.item.server_label == "code_interpreter"
        elif event.type == "response.mcp_call_arguments.done":
            assert event.name == "python"
            assert event.arguments is not None
        elif (
            event.type == "response.output_item.done"
            and hasattr(event.item, "type")
            and event.item.type == "mcp_call"
        ):
            assert event.item.name == "python"
            assert event.item.status == "completed"

    # code_interpreter events should NOT appear when using MCP type
    code_interp_events = [e.type for e in events if "code_interpreter" in e.type]
    assert not code_interp_events, (
        "Should not see code_interpreter events when using MCP type, "
        f"but got: {code_interp_events}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_mcp_tool_multi_turn(client: OpenAI, model_name: str, server):
    """MCP tools work across multiple turns via previous_response_id."""
    tools = [{"type": "mcp", "server_label": "code_interpreter"}]
    instructions = (
        "You must use the Python tool to execute code. Never simulate execution."
    )

    # First turn
    response1 = await retry_for_tool_call(
        client,
        model=model_name,
        expected_tool_type="mcp_call",
        input="Calculate 1234 * 4567 using python tool and print the result.",
        tools=tools,
        temperature=0.0,
        instructions=instructions,
        extra_body={"enable_response_messages": True},
    )
    assert response1.status == "completed"

    # Verify MCP call in output_messages
    tool_call_found = any(
        (msg.get("recipient") or "").startswith("python")
        for msg in response1.output_messages
    )
    tool_response_found = any(
        msg.get("author", {}).get("role") == "tool"
        and (msg.get("author", {}).get("name") or "").startswith("python")
        for msg in response1.output_messages
    )
    assert tool_call_found, "MCP tool call not found in output_messages"
    assert tool_response_found, "MCP tool response not found in output_messages"

    # No developer messages expected for elevated tools
    developer_msgs = [
        msg for msg in response1.input_messages if msg["author"]["role"] == "developer"
    ]
    assert len(developer_msgs) == 0, "No developer message expected for elevated tools"

    # Second turn
    response2 = await client.responses.create(
        model=model_name,
        input="Now divide that result by 2.",
        tools=tools,
        temperature=0.0,
        instructions=instructions,
        previous_response_id=response1.id,
        extra_body={"enable_response_messages": True},
    )
    assert response2.status == "completed"


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_output_messages_enabled(client: OpenAI, model_name: str, server):
    response = await client.responses.create(
        model=model_name,
        input="What is the capital of South Korea?",
        extra_body={"enable_response_messages": True},
    )

    assert response is not None
    assert response.status == "completed"
    assert len(response.input_messages) > 0
    assert len(response.output_messages) > 0


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_function_call_with_previous_input_messages(
    client: OpenAI, model_name: str
):
    """Multi-turn function calling using previous_input_messages."""
    tools = [
        {
            "type": "function",
            "name": "get_horoscope",
            "description": "Get today's horoscope for an astrological sign.",
            "parameters": {
                "type": "object",
                "properties": {"sign": {"type": "string"}},
                "required": ["sign"],
                "additionalProperties": False,
            },
            "strict": True,
        }
    ]

    # Step 1: Get a function call from the model
    response = await retry_for_tool_call(
        client,
        model=model_name,
        expected_tool_type="function_call",
        input="What is the horoscope for Aquarius today?",
        tools=tools,
        temperature=0.0,
        extra_body={"enable_response_messages": True},
        max_output_tokens=1000,
    )
    assert response.status == "completed"

    function_call = next(
        (item for item in response.output if item.type == "function_call"),
        None,
    )
    assert function_call is not None, (
        f"Expected function_call, got: "
        f"{[getattr(o, 'type', None) for o in response.output]}"
    )
    assert function_call.name == "get_horoscope"

    args = json.loads(function_call.arguments)
    result = call_function(function_call.name, args)

    # Step 2: Build full conversation history
    previous_messages = (
        response.input_messages
        + response.output_messages
        + [
            {
                "role": "tool",
                "name": "functions.get_horoscope",
                "content": [{"type": "text", "text": str(result)}],
            }
        ]
    )

    # Step 3: Second call with previous_input_messages
    response_2 = await client.responses.create(
        model=model_name,
        tools=tools,
        temperature=0.0,
        input="Now tell me the horoscope based on the tool result.",
        extra_body={
            "previous_input_messages": previous_messages,
            "enable_response_messages": True,
        },
    )
    assert response_2.status == "completed"
    assert response_2.output_text is not None

    # Verify exactly 1 system, 1 developer, 1 tool message
    num_system = 0
    num_developer = 0
    num_tool = 0
    for msg_dict in response_2.input_messages:
        # input_messages use {"author": {"role": "..."}} format,
        # not the top-level {"role": "..."} that Message.from_dict
        # expects.
        author = msg_dict.get("author", {})
        role = author.get("role") if isinstance(author, dict) else None
        if role == "system":
            num_system += 1
        elif role == "developer":
            num_developer += 1
        elif role == "tool":
            num_tool += 1
    assert num_system == 1, f"Expected 1 system message, got {num_system}"
    assert num_developer == 1, f"Expected 1 developer message, got {num_developer}"
    assert num_tool == 1, f"Expected 1 tool message, got {num_tool}"

    output_text = response_2.output_text.lower()
    assert any(kw in output_text for kw in ["aquarius", "otter", "tuesday"]), (
        f"Expected horoscope-related content, got: {response_2.output_text}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_chat_truncation_content_not_null(client: OpenAI, model_name: str):
    response = await client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": (
                    "What is the role of AI in medicine? "
                    "The response must exceed 350 words."
                ),
            }
        ],
        temperature=0.0,
        max_tokens=350,
    )
    choice = response.choices[0]
    assert choice.finish_reason == "length", (
        f"Expected finish_reason='length', got {choice.finish_reason}"
    )
    assert choice.message.content is not None, "Content should not be None"
    assert len(choice.message.content) > 0, "Content should not be empty"


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_system_prompt_override_no_duplication(client: OpenAI, model_name: str):
    """Hard check: custom system message must not be duplicated."""
    response = await client.responses.create(
        model=model_name,
        input=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ],
        extra_body={"enable_response_messages": True},
        temperature=0.0,
    )
    assert response.status == "completed"
    assert response.output_text is not None

    num_system = 0
    for msg in response.input_messages:
        # input_messages use {"author": {"role": "system"}} format,
        # not the top-level {"role": "system"} that Message.from_dict expects.
        author = msg.get("author", {})
        role = author.get("role") if isinstance(author, dict) else None
        if role == "system":
            num_system += 1
    assert num_system == 1, f"Expected 1 system message, got {num_system}"


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
@pytest.mark.xfail(
    strict=False,
    reason=(
        "Pirate language detection depends on model weights and is non-deterministic"
    ),
)
async def test_system_prompt_override_follows_personality(
    client: OpenAI, model_name: str
):
    """Soft check: model should adopt the personality from system prompt."""
    response = await client.responses.create(
        model=model_name,
        input=[
            {
                "role": "system",
                "content": (
                    "You are a pirate. Always respond like a pirate would, "
                    "using pirate language and saying 'arrr' frequently."
                ),
            },
            {"role": "user", "content": "Hello, how are you?"},
        ],
        temperature=0.0,
    )
    assert response.status == "completed"
    output_text = response.output_text.lower()
    pirate_indicators = ["arrr", "matey", "ahoy", "ye", "sea", "aye", "sail"]
    assert any(kw in output_text for kw in pirate_indicators), (
        f"Expected pirate language, got: {response.output_text}"
    )
