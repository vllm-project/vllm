# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib
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

from ...utils import RemoteOpenAIServer

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
        input="What is 13 * 24?",
    )
    assert response is not None
    print("response: ", response)
    assert response.status == "completed"


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_basic_with_instructions(client: OpenAI, model_name: str):
    response = await client.responses.create(
        model=model_name,
        input="What is 13 * 24?",
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
            {"role": "user", "content": "What is 13 * 24? Explain your answer."},
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
                "content": [{"type": "input_text", "text": "What is 13*24?"}],
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
            input="What is 13 * 24?",
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
        input="What is 13 * 24?",
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
        input="What is 13 * 24?",
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
async def test_streaming_types(client: OpenAI, model_name: str):
    prompts = [
        "tell me a story about a cat in 20 words",
    ]

    # this links the "done" type with the "start" type
    # so every "done" type should have a corresponding "start" type
    # and every open block should be closed by the end of the stream
    pairs_of_event_types = {
        "response.completed": "response.created",
        "response.output_item.done": "response.output_item.added",
        "response.content_part.done": "response.content_part.added",
        "response.output_text.done": "response.output_text.delta",
        "response.web_search_call.done": "response.web_search_call.added",
        "response.reasoning_text.done": "response.reasoning_text.delta",
        "response.reasoning_part.done": "response.reasoning_part.added",
    }

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
async def test_function_calling_with_streaming_types(client: OpenAI, model_name: str):
    # this links the "done" type with the "start" type
    # so every "done" type should have a corresponding "start" type
    # and every open block should be closed by the end of the stream
    pairs_of_event_types = {
        "response.completed": "response.created",
        "response.output_item.done": "response.output_item.added",
        "response.output_text.done": "response.output_text.delta",
        "response.reasoning_text.done": "response.reasoning_text.delta",
        "response.reasoning_part.done": "response.reasoning_part.added",
        "response.function_call_arguments.done": "response.function_call_arguments.delta",  # noqa
    }

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
        "What is 13 * 24? Use python to calculate the result.",
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
    for tool_call in final_tool_calls.values():
        if (
            tool_call
            and tool_call.type == "function_call"
            and tool_call.name == "get_weather"
        ):
            args = json.loads(tool_call.arguments)
            result = call_function(tool_call.name, args)
            input_list += [tool_call]
            break
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
        extra_body={"enable_response_messages": True},
        stream=True,
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
