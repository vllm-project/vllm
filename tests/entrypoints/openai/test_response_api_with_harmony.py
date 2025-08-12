# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import time

import pytest
import pytest_asyncio
import requests
from openai import BadRequestError, NotFoundError, OpenAI

from ...utils import RemoteOpenAIServer

pytest.skip(allow_module_level=True, reason="gpt-oss can't run on CI yet.")

MODEL_NAME = "openai/gpt-oss-20b"
DTYPE = "bfloat16"


@pytest.fixture(scope="module")
def server():
    args = ["--enforce-eager", "--tool-server", "demo"]

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
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
async def test_chat(client: OpenAI, model_name: str):
    response = await client.responses.create(
        model=model_name,
        input=[
            {
                "role": "system",
                "content": "Respond in Korean."
            },
            {
                "role": "user",
                "content": "Hello!"
            },
            {
                "role": "assistant",
                "content": "Hello! How can I help you today?"
            },
            {
                "role": "user",
                "content": "What is 13 * 24? Explain your answer."
            },
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
                "content": [{
                    "type": "input_text",
                    "text": "What is 13*24?"
                }],
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
            {
                "role": "system",
                "content": "Extract the event information."
            },
            {
                "role": "user",
                "content":
                "Alice and Bob are going to a science fair on Friday.",
            },
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "calendar_event",
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string"
                        },
                        "date": {
                            "type": "string"
                        },
                        "participants": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            }
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
async def test_streaming(client: OpenAI, model_name: str):
    prompts = [
        "tell me a story about a cat in 20 words",
        "What is 13 * 24? Use python to calculate the result.",
        "When did Jensen found NVIDIA? Search it and answer the year only.",
    ]

    for prompt in prompts:
        response = await client.responses.create(
            model=model_name,
            input=prompt,
            reasoning={"effort": "low"},
            tools=[
                {
                    "type": "web_search_preview"
                },
                {
                    "type": "code_interpreter",
                    "container": {
                        "type": "auto"
                    }
                },
            ],
            stream=True,
        )

        events = []
        current_event_mode = None
        async for event in response:
            if current_event_mode != event.type:
                current_event_mode = event.type
                print(f"\n[{event.type}] ", end="", flush=True)

            if "text.delta" in event.type:
                print(event.delta, end="", flush=True)
            elif "reasoning_text.delta" in event.type:
                print(f"{event.delta}", end="", flush=True)
            elif "response.code_interpreter_call_code.done" in event.type:
                print(f"Code: {event.code}", end="", flush=True)
            elif ("response.output_item.added" in event.type
                  and event.item.type == "web_search_call"):
                print(f"Web search: {event.item.action}", end="", flush=True)
            events.append(event)

        assert len(events) > 0


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_web_search(client: OpenAI, model_name: str):
    response = await client.responses.create(
        model=model_name,
        input="Who is the president of South Korea as of now?",
        tools=[{
            "type": "web_search_preview"
        }],
    )
    assert response is not None
    assert response.status == "completed"


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_code_interpreter(client: OpenAI, model_name: str):
    response = await client.responses.create(
        model=model_name,
        input="Multiply 64548*15151 using builtin python interpreter.",
        tools=[{
            "type": "code_interpreter",
            "container": {
                "type": "auto"
            }
        }],
    )
    assert response is not None
    assert response.status == "completed"


def get_weather(latitude, longitude):
    response = requests.get(
        f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m"  # noqa
    )
    data = response.json()
    return data["current"]["temperature_2m"]


def get_place_to_travel():
    return "Paris"


def call_function(name, args):
    if name == "get_weather":
        return get_weather(**args)
    elif name == "get_place_to_travel":
        return get_place_to_travel()
    else:
        raise ValueError(f"Unknown function: {name}")


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_function_calling(client: OpenAI, model_name: str):
    tools = [{
        "type": "function",
        "name": "get_weather",
        "description":
        "Get current temperature for provided coordinates in celsius.",  # noqa
        "parameters": {
            "type": "object",
            "properties": {
                "latitude": {
                    "type": "number"
                },
                "longitude": {
                    "type": "number"
                },
            },
            "required": ["latitude", "longitude"],
            "additionalProperties": False,
        },
        "strict": True,
    }]

    response = await client.responses.create(
        model=model_name,
        input="What's the weather like in Paris today?",
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
        input=[{
            "type": "function_call_output",
            "call_id": tool_call.call_id,
            "output": str(result),
        }],
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
        {
            "type": "function",
            "name": "get_weather",
            "description":
            "Get current temperature for provided coordinates in celsius.",  # noqa
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {
                        "type": "number"
                    },
                    "longitude": {
                        "type": "number"
                    },
                },
                "required": ["latitude", "longitude"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    ]

    response = await client.responses.create(
        model=model_name,
        input=
        "Help me plan a trip to a random place. And tell me the weather there.",
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
        input=[{
            "type": "function_call_output",
            "call_id": tool_call.call_id,
            "output": str(result),
        }],
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
        input=[{
            "type": "function_call_output",
            "call_id": tool_call.call_id,
            "output": str(result),
        }],
        tools=tools,
        previous_response_id=response_2.id,
    )
    assert response_3 is not None
    assert response_3.status == "completed"
    assert response_3.output_text is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_function_calling_required(client: OpenAI, model_name: str):
    tools = [{
        "type": "function",
        "name": "get_weather",
        "description":
        "Get current temperature for provided coordinates in celsius.",  # noqa
        "parameters": {
            "type": "object",
            "properties": {
                "latitude": {
                    "type": "number"
                },
                "longitude": {
                    "type": "number"
                },
            },
            "required": ["latitude", "longitude"],
            "additionalProperties": False,
        },
        "strict": True,
    }]

    with pytest.raises(BadRequestError):
        await client.responses.create(
            model=model_name,
            input="What's the weather like in Paris today?",
            tools=tools,
            tool_choice="required",
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_function_calling_full_history(client: OpenAI, model_name: str):
    tools = [{
        "type": "function",
        "name": "get_weather",
        "description":
        "Get current temperature for provided coordinates in celsius.",  # noqa
        "parameters": {
            "type": "object",
            "properties": {
                "latitude": {
                    "type": "number"
                },
                "longitude": {
                    "type": "number"
                },
            },
            "required": ["latitude", "longitude"],
            "additionalProperties": False,
        },
        "strict": True,
    }]

    input_messages = [{
        "role": "user",
        "content": "What's the weather like in Paris today?"
    }]

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

    input_messages.extend(
        response.output)  # append model's function call message
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
