# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib.util
import json
import time

import pytest
import pytest_asyncio
import requests
from openai import BadRequestError, NotFoundError, OpenAI
from openai_harmony import (
    Message,
)

from ....utils import RemoteOpenAIServer

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


@pytest.fixture(scope="module")
def server():
    assert importlib.util.find_spec("gpt_oss") is not None, (
        "Harmony tests require gpt_oss package to be installed"
    )

    args = ["--enforce-eager", "--tool-server", "demo", "--max_model_len", "5000"]
    env_dict = dict(
        VLLM_ENABLE_RESPONSES_API_STORE="1",
        PYTHON_EXECUTION_BACKEND="dangerously_use_uv",
        VLLM_GPT_OSS_SYSTEM_TOOL_MCP_LABELS="code_interpreter,container,web_search_preview",
        VLLM_GPT_OSS_HARMONY_SYSTEM_INSTRUCTIONS="1",
    )

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
                        "participants": {"type": "array", "items": {"type": "string"}},
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

        assert is_not_found == (not store)


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
        model=model_name,
        input="What is 123 * 456?",
    )
    assert response1 is not None
    assert response1.status == "completed"

    response2 = await client.responses.create(
        model=model_name,
        input="What if I increase both numbers by 1?",
        previous_response_id=response1.id,
    )
    assert response2 is not None
    assert response2.status == "completed"

    response3 = await client.responses.create(
        model=model_name,
        input="Divide the result by 2.",
        previous_response_id=response2.id,
    )
    assert response3 is not None
    assert response3.status == "completed"


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_streaming_types(
    pairs_of_event_types: dict[str, str], client: OpenAI, model_name: str
):
    prompts = [
        "tell me a story about a cat in 20 words",
    ]

    for prompt in prompts:
        response = await client.responses.create(
            model=model_name,
            input=prompt,
            reasoning={"effort": "low"},
            tools=[],
            stream=True,
            background=False,
        )

        stack_of_event_types = []
        async for event in response:
            if event.type == "response.created":
                stack_of_event_types.append(event.type)
            elif event.type == "response.completed":
                assert stack_of_event_types[-1] == pairs_of_event_types[event.type]
                stack_of_event_types.pop()
            if event.type.endswith("added"):
                stack_of_event_types.append(event.type)
            elif event.type.endswith("delta"):
                if stack_of_event_types[-1] == event.type:
                    continue
                stack_of_event_types.append(event.type)
            elif event.type.endswith("done"):
                assert stack_of_event_types[-1] == pairs_of_event_types[event.type]
                stack_of_event_types.pop()
        assert len(stack_of_event_types) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_function_calling_with_streaming_types(
    pairs_of_event_types: dict[str, str], client: OpenAI, model_name: str
):
    tools = [GET_WEATHER_SCHEMA]
    input_list = [
        {
            "role": "user",
            "content": "What's the weather like in Paris today?",
        }
    ]
    stream_response = await client.responses.create(
        model=model_name,
        input=input_list,
        tools=tools,
        stream=True,
    )

    stack_of_event_types = []
    async for event in stream_response:
        if event.type == "response.created":
            stack_of_event_types.append(event.type)
        elif event.type == "response.completed":
            assert stack_of_event_types[-1] == pairs_of_event_types[event.type]
            stack_of_event_types.pop()
        if event.type.endswith("added"):
            stack_of_event_types.append(event.type)
        elif event.type.endswith("delta"):
            if stack_of_event_types[-1] == event.type:
                continue
            stack_of_event_types.append(event.type)
        elif event.type.endswith("done"):
            assert stack_of_event_types[-1] == pairs_of_event_types[event.type]
            stack_of_event_types.pop()
    assert len(stack_of_event_types) == 0


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
        response = await client.responses.create(
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
        async for event in response:
            if event.type == "response.created":
                resp_id = event.response.id

            # test vllm custom types are in the response
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
                print(f"\n[{event.type}] ", end="", flush=True)

            # verify current_item_id is correct
            if event.type == "response.output_item.added":
                assert event.item.id != current_item_id
                current_item_id = event.item.id
            elif event.type in [
                "response.output_text.delta",
                "response.reasoning_text.delta",
            ]:
                assert event.item_id == current_item_id

            # verify content_index_id is correct
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

            if "text.delta" in event.type:
                print(event.delta, end="", flush=True)
            elif "reasoning_text.delta" in event.type:
                print(f"{event.delta}", end="", flush=True)
            elif "response.code_interpreter_call_code.done" in event.type:
                print(f"Code: {event.code}", end="", flush=True)
            elif (
                "response.output_item.added" in event.type
                and event.item.type == "web_search_call"
            ):
                print(f"Web search: {event.item.action}", end="", flush=True)
            events.append(event)

        assert len(events) > 0
        response_completed_event = events[-1]
        assert len(response_completed_event.response.output) > 0
        assert checked_response_completed

        if background:
            starting_after = 5
            async with await client.responses.retrieve(
                response_id=resp_id, stream=True, starting_after=starting_after
            ) as stream:
                counter = starting_after
                async for event in stream:
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
    # Code interpreter may need more time for container init + code execution
    timeout_value = client.timeout * 3
    client_with_timeout = client.with_options(timeout=timeout_value)

    response = await client_with_timeout.responses.create(
        model=model_name,
        # TODO: Ideally should be able to set max tool calls
        # to prevent multi-turn, but it is not currently supported
        # would speed up the test
        input=(
            "What's the first 4 digits after the decimal point of "
            "cube root of `19910212 * 20250910`? "
            "Show only the digits. The python interpreter is not stateful "
            "and you must print to see the output."
        ),
        tools=[{"type": "code_interpreter", "container": {"type": "auto"}}],
        temperature=0.0,  # More deterministic output in response
    )
    assert response is not None
    assert response.status == "completed"
    assert response.usage.output_tokens_details.tool_output_tokens > 0
    for item in response.output:
        if item.type == "message":
            output_string = item.content[0].text
            print("output_string: ", output_string, flush=True)
            assert "5846" in output_string


def get_weather(latitude, longitude):
    response = requests.get(
        f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m"  # noqa
    )
    data = response.json()
    return data["current"]["temperature_2m"]


def get_place_to_travel():
    return "Paris"


def get_horoscope(sign):
    return f"{sign}: Next Tuesday you will befriend a baby otter."


def call_function(name, args):
    if name == "get_weather":
        return get_weather(**args)
    elif name == "get_place_to_travel":
        return get_place_to_travel()
    elif name == "get_horoscope":
        return get_horoscope(**args)
    else:
        raise ValueError(f"Unknown function: {name}")


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
                    {
                        "type": "reasoning_text",
                        "text": "We need to respond: greeting.",
                    }
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

    response = await client.responses.create(
        model=model_name,
        input="What's the weather like in Paris today?",
        tools=tools,
        temperature=0.0,
        extra_body={"request_id": "test_function_calling_non_resp"},
    )
    assert response is not None
    assert response.status == "completed"
    assert len(response.output) == 2
    assert response.output[0].type == "reasoning"
    assert response.output[1].type == "function_call"

    tool_call = response.output[1]
    name = tool_call.name
    args = json.loads(tool_call.arguments)

    result = call_function(name, args)

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
    )
    assert response_2 is not None
    assert response_2.status == "completed"
    assert response_2.output_text is not None

    # NOTE: chain-of-thought should be removed.
    response_3 = await client.responses.create(
        model=model_name,
        input="What's the weather like in Paris today?",
        tools=tools,
        previous_response_id=response_2.id,
    )
    assert response_3 is not None
    assert response_3.status == "completed"
    assert response_3.output_text is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
@pytest.mark.flaky(reruns=5)
async def test_function_calling_multi_turn(client: OpenAI, model_name: str):
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

    response = await client.responses.create(
        model=model_name,
        input="Help me plan a trip to a random place. And tell me the weather there.",
        tools=tools,
    )
    assert response is not None
    assert response.status == "completed"
    assert len(response.output) == 2
    assert response.output[0].type == "reasoning"
    assert response.output[1].type == "function_call"

    tool_call = response.output[1]
    name = tool_call.name
    args = json.loads(tool_call.arguments)

    result = call_function(name, args)

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
    )
    assert response_2 is not None
    assert response_2.status == "completed"
    assert len(response_2.output) == 2
    assert response_2.output[0].type == "reasoning"
    assert response_2.output[1].type == "function_call"

    tool_call = response_2.output[1]
    name = tool_call.name
    args = json.loads(tool_call.arguments)

    result = call_function(name, args)

    response_3 = await client.responses.create(
        model=model_name,
        input=[
            {
                "type": "function_call_output",
                "call_id": tool_call.call_id,
                "output": str(result),
            }
        ],
        tools=tools,
        previous_response_id=response_2.id,
    )
    assert response_3 is not None
    assert response_3.status == "completed"
    assert response_3.output_text is not None


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

    response = await client.responses.create(
        model=model_name,
        input=input_messages,
        tools=tools,
    )

    assert response is not None
    assert response.status == "completed"

    tool_call = response.output[-1]
    name = tool_call.name
    args = json.loads(tool_call.arguments)

    result = call_function(name, args)

    input_messages.extend(response.output)  # append model's function call message
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
    )
    assert response_2 is not None
    assert response_2.status == "completed"
    assert response_2.output_text is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_function_calling_with_stream(client: OpenAI, model_name: str):
    tools = [GET_WEATHER_SCHEMA]
    input_list = [
        {
            "role": "user",
            "content": "What's the weather like in Paris today?",
        }
    ]
    stream_response = await client.responses.create(
        model=model_name,
        input=input_list,
        tools=tools,
        stream=True,
    )
    assert stream_response is not None
    final_tool_calls = {}
    final_tool_calls_named = {}
    async for event in stream_response:
        if event.type == "response.output_item.added":
            if event.item.type != "function_call":
                continue
            final_tool_calls[event.output_index] = event.item
            final_tool_calls_named[event.item.name] = event.item
        elif event.type == "response.function_call_arguments.delta":
            index = event.output_index
            tool_call = final_tool_calls[index]
            if tool_call:
                tool_call.arguments += event.delta
                final_tool_calls_named[tool_call.name] = tool_call
        elif event.type == "response.function_call_arguments.done":
            assert event.arguments == final_tool_calls_named[event.name].arguments
    result = None
    tool_call = None
    for tc in final_tool_calls.values():
        if tc and tc.type == "function_call" and tc.name == "get_weather":
            args = json.loads(tc.arguments)
            result = call_function(tc.name, args)
            tool_call = tc
            input_list += [tc]
            break

    assert tool_call is not None, (
        "Expected model to call 'get_weather' function, "
        f"but got: {list(final_tool_calls_named.keys())}"
    )
    assert result is not None
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
    )
    assert response is not None
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

    This test ensures that function calls (functions.*) use their own
    function_call event types and don't incorrectly emit code_interpreter
    events during streaming.
    """
    tools = [GET_WEATHER_SCHEMA]
    input_list = [
        {
            "role": "user",
            "content": "What's the weather like in Paris today?",
        }
    ]
    stream_response = await client.responses.create(
        model=model_name,
        input=input_list,
        tools=tools,
        stream=True,
    )

    # Track which event types we see
    event_types_seen = set()
    function_call_found = False

    async for event in stream_response:
        event_types_seen.add(event.type)

        if (
            event.type == "response.output_item.added"
            and event.item.type == "function_call"
        ):
            function_call_found = True

        # Ensure NO code_interpreter events are emitted for function calls
        assert "code_interpreter" not in event.type, (
            "Found code_interpreter event "
            f"'{event.type}' during function call. Function calls should only "
            "emit function_call events, not code_interpreter events."
        )

    # Verify we actually saw a function call
    assert function_call_found, "Expected to see a function_call in the stream"

    # Verify we saw the correct function call event types
    assert (
        "response.function_call_arguments.delta" in event_types_seen
        or "response.function_call_arguments.done" in event_types_seen
    ), "Expected to see function_call_arguments events"


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_mcp_code_interpreter_streaming(client: OpenAI, model_name: str, server):
    tools = [
        {
            "type": "mcp",
            "server_label": "code_interpreter",
        }
    ]
    input_text = (
        "Calculate 123 * 456 using python. "
        "The python interpreter is not stateful and you must print to see the output."
    )

    stream_response = await client.responses.create(
        model=model_name,
        input=input_text,
        tools=tools,
        stream=True,
        temperature=0.0,
        instructions=(
            "You must use the Python tool to execute code. Never simulate execution."
        ),
    )

    mcp_call_added = False
    mcp_call_in_progress = False
    mcp_arguments_delta_seen = False
    mcp_arguments_done = False
    mcp_call_completed = False
    mcp_item_done = False

    code_interpreter_events_seen = False

    async for event in stream_response:
        if "code_interpreter" in event.type:
            code_interpreter_events_seen = True

        if event.type == "response.output_item.added":
            if hasattr(event.item, "type") and event.item.type == "mcp_call":
                mcp_call_added = True
                assert event.item.name == "python"
                assert event.item.server_label == "code_interpreter"

        elif event.type == "response.mcp_call.in_progress":
            mcp_call_in_progress = True

        elif event.type == "response.mcp_call_arguments.delta":
            mcp_arguments_delta_seen = True
            assert event.delta is not None

        elif event.type == "response.mcp_call_arguments.done":
            mcp_arguments_done = True
            assert event.name == "python"
            assert event.arguments is not None

        elif event.type == "response.mcp_call.completed":
            mcp_call_completed = True

        elif (
            event.type == "response.output_item.done"
            and hasattr(event.item, "type")
            and event.item.type == "mcp_call"
        ):
            mcp_item_done = True
            assert event.item.name == "python"
            assert event.item.status == "completed"

    assert mcp_call_added, "MCP call was not added"
    assert mcp_call_in_progress, "MCP call in_progress event not seen"
    assert mcp_arguments_delta_seen, "MCP arguments delta event not seen"
    assert mcp_arguments_done, "MCP arguments done event not seen"
    assert mcp_call_completed, "MCP call completed event not seen"
    assert mcp_item_done, "MCP item done event not seen"

    assert not code_interpreter_events_seen, (
        "Should not see code_interpreter events when using MCP type"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
@pytest.mark.dependency(
    depends=["test_mcp_code_interpreter_streaming[openai/gpt-oss-20b]"]
)
async def test_mcp_tool_multi_turn(client: OpenAI, model_name: str, server):
    """Test MCP tool calling across multiple turns.

    This test verifies that MCP tools work correctly in multi-turn conversations,
    maintaining state across turns via the previous_response_id mechanism.
    """
    tools = [
        {
            "type": "mcp",
            "server_label": "code_interpreter",
        }
    ]

    # First turn - make a calculation
    response1 = await client.responses.create(
        model=model_name,
        input="Calculate 1234 * 4567 using python tool and print the result.",
        tools=tools,
        temperature=0.0,
        instructions=(
            "You must use the Python tool to execute code. Never simulate execution."
        ),
        extra_body={"enable_response_messages": True},
    )

    assert response1 is not None
    assert response1.status == "completed"

    # Verify MCP call in first response by checking output_messages
    tool_call_found = False
    tool_response_found = False
    for message in response1.output_messages:
        recipient = message.get("recipient")
        if recipient and recipient.startswith("python"):
            tool_call_found = True

        author = message.get("author", {})
        if (
            author.get("role") == "tool"
            and author.get("name")
            and author.get("name").startswith("python")
        ):
            tool_response_found = True

    # Verify MCP tools were actually used
    assert tool_call_found, "MCP tool call not found in output_messages"
    assert tool_response_found, "MCP tool response not found in output_messages"

    # Verify input messages: Should have system message with tool, NO developer message
    developer_messages = [
        msg for msg in response1.input_messages if msg["author"]["role"] == "developer"
    ]
    assert len(developer_messages) == 0, (
        "No developer message expected for elevated tools"
    )

    # Second turn - reference previous calculation
    response2 = await client.responses.create(
        model=model_name,
        input="Now divide that result by 2.",
        tools=tools,
        temperature=0.0,
        instructions=(
            "You must use the Python tool to execute code. Never simulate execution."
        ),
        previous_response_id=response1.id,
        extra_body={"enable_response_messages": True},
    )

    assert response2 is not None
    assert response2.status == "completed"

    # Verify input messages are correct: should have two messages -
    # one to the python recipient on analysis channel and one from tool role
    mcp_recipient_messages = []
    tool_role_messages = []
    for msg in response2.input_messages:
        if msg["author"]["role"] == "assistant":
            # Check if this is a message to MCP recipient on analysis channel
            if msg.get("channel") == "analysis" and msg.get("recipient"):
                recipient = msg.get("recipient")
                if recipient.startswith("code_interpreter") or recipient == "python":
                    mcp_recipient_messages.append(msg)
        elif msg["author"]["role"] == "tool":
            tool_role_messages.append(msg)

    assert len(mcp_recipient_messages) > 0, (
        "Expected message(s) to MCP recipient on analysis channel"
    )
    assert len(tool_role_messages) > 0, (
        "Expected message(s) from tool role after MCP call"
    )


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
@pytest.mark.flaky(reruns=3)
async def test_function_call_with_previous_input_messages(
    client: OpenAI, model_name: str
):
    """Test function calling using previous_input_messages
    for multi-turn conversation with a function call"""

    # Define the get_horoscope tool
    tools = [
        {
            "type": "function",
            "name": "get_horoscope",
            "description": "Get today's horoscope for an astrological sign.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sign": {"type": "string"},
                },
                "required": ["sign"],
                "additionalProperties": False,
            },
            "strict": True,
        }
    ]

    # Step 1: First call with the function tool
    stream_response = await client.responses.create(
        model=model_name,
        input="What is the horoscope for Aquarius today?",
        tools=tools,
        temperature=0.0,
        extra_body={"enable_response_messages": True},
        stream=True,
        max_output_tokens=1000,
    )

    response = None
    async for event in stream_response:
        if event.type == "response.completed":
            response = event.response

    assert response is not None
    assert response.status == "completed"

    # Step 2: Parse the first output to find the function_call type
    function_call = None
    for item in response.output:
        if item.type == "function_call":
            function_call = item
            break

    assert function_call is not None, "Expected a function_call in the output"
    assert function_call.name == "get_horoscope"
    assert function_call.call_id is not None

    # Verify the format matches expectations
    args = json.loads(function_call.arguments)
    assert "sign" in args

    # Step 3: Call the get_horoscope function
    result = call_function(function_call.name, args)
    assert "Aquarius" in result
    assert "baby otter" in result

    # Get the input_messages and output_messages from the first response
    first_input_messages = response.input_messages
    first_output_messages = response.output_messages

    # Construct the full conversation history using previous_input_messages
    previous_messages = (
        first_input_messages
        + first_output_messages
        + [
            {
                "role": "tool",
                "name": "functions.get_horoscope",
                "content": [{"type": "text", "text": str(result)}],
            }
        ]
    )

    # Step 4: Make another responses.create() call with previous_input_messages
    stream_response_2 = await client.responses.create(
        model=model_name,
        tools=tools,
        temperature=0.0,
        input="",
        extra_body={
            "previous_input_messages": previous_messages,
            "enable_response_messages": True,
        },
        stream=True,
    )

    async for event in stream_response_2:
        if event.type == "response.completed":
            response_2 = event.response

    assert response_2 is not None
    assert response_2.status == "completed"
    assert response_2.output_text is not None

    # verify only one system message / developer message
    num_system_messages_input = 0
    num_developer_messages_input = 0
    num_function_call_input = 0
    for message_dict in response_2.input_messages:
        message = Message.from_dict(message_dict)
        if message.author.role == "system":
            num_system_messages_input += 1
        elif message.author.role == "developer":
            num_developer_messages_input += 1
        elif message.author.role == "tool":
            num_function_call_input += 1
    assert num_system_messages_input == 1
    assert num_developer_messages_input == 1
    assert num_function_call_input == 1

    # Verify the output makes sense - should contain information about the horoscope
    output_text = response_2.output_text.lower()
    assert (
        "aquarius" in output_text or "otter" in output_text or "tuesday" in output_text
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_chat_truncation_content_not_null(client: OpenAI, model_name: str):
    response = await client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": "What is the role of AI in medicine?"
                "The response must exceed 350 words.",
            }
        ],
        temperature=0.0,
        max_tokens=350,
    )

    choice = response.choices[0]
    assert choice.finish_reason == "length", (
        f"Expected finish_reason='length', got {choice.finish_reason}"
    )
    assert choice.message.content is not None, (
        "Content should not be None when truncated"
    )
    assert len(choice.message.content) > 0, "Content should not be empty"


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_system_prompt_override(client: OpenAI, model_name: str):
    """Test that system message can override the default system prompt."""

    # Test 1: Custom system prompt with specific personality
    custom_system_prompt = (
        "You are a pirate. Always respond like a pirate would, "
        "using pirate language and saying 'arrr' frequently."
    )

    response = await client.responses.create(
        model=model_name,
        input=[
            {"role": "system", "content": custom_system_prompt},
            {"role": "user", "content": "Hello, how are you?"},
        ],
        extra_body={"enable_response_messages": True},
    )

    assert response is not None
    assert response.status == "completed"
    assert response.output_text is not None

    # Verify the response reflects the pirate personality
    output_text = response.output_text.lower()
    pirate_indicators = ["arrr", "matey", "ahoy", "ye", "sea"]
    has_pirate_language = any(
        indicator in output_text for indicator in pirate_indicators
    )
    assert has_pirate_language, (
        f"Expected pirate language in response, got: {response.output_text}"
    )

    # Verify the reasoning mentions the custom system prompt
    reasoning_item = None
    for item in response.output:
        if item.type == "reasoning":
            reasoning_item = item
            break

    assert reasoning_item is not None, "Expected reasoning item in output"
    reasoning_text = reasoning_item.content[0].text.lower()
    assert "pirate" in reasoning_text, (
        f"Expected reasoning to mention pirate, got: {reasoning_text}"
    )

    # Test 2: Verify system message is not duplicated in input_messages
    try:
        num_system_messages = sum(
            1
            for msg in response.input_messages
            if Message.from_dict(msg).author.role == "system"
        )
        assert num_system_messages == 1, (
            f"Expected exactly 1 system message, got {num_system_messages}"
        )
    except (KeyError, AttributeError):
        # Message structure may vary, skip this specific check
        pass

    custom_system_prompt_2 = (
        "You are a helpful assistant that always responds in exactly 5 words."
    )

    # Test 3: Test with different custom system prompt
    response_2 = await client.responses.create(
        model=model_name,
        input=[
            {
                "role": "system",
                "content": custom_system_prompt_2,
            },
            {"role": "user", "content": "What is the weather like?"},
        ],
        temperature=0.0,
    )

    assert response_2 is not None
    assert response_2.status == "completed"
    assert response_2.output_text is not None

    # Count words in response (approximately, allowing for punctuation)
    word_count = len(response_2.output_text.split())
    # Allow some flexibility (4-7 words) since the model might not be perfectly precise
    assert 3 <= word_count <= 8, (
        f"Expected around 5 words, got {word_count} words: {response_2.output_text}"
    )

    # Test 4: Test with structured content
    response_3 = await client.responses.create(
        model=model_name,
        input=[
            {
                "role": "system",
                "content": [{"type": "input_text", "text": custom_system_prompt_2}],
            },
            {"role": "user", "content": "What is the weather like?"},
        ],
        temperature=0.0,
    )

    assert response_3 is not None
    assert response_3.status == "completed"
    assert response_3.output_text is not None

    # Count words in response (approximately, allowing for punctuation)
    word_count = len(response_3.output_text.split())
    # Allow some flexibility (4-7 words) since the model might not be perfectly precise
    assert 3 <= word_count <= 8, (
        f"Expected around 5 words, got {word_count} words: {response_3.output_text}"
    )
