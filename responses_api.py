"""
vllm serve /data/woosuk/os-mini-weights/pytorch-rc-20b --tokenizer /data/xmo/os-mini/models/hf-converted --enforce-eager
"""
import json
import time
from openai import BadRequestError, NotFoundError, OpenAI
import argparse

import ast
import requests

parser = argparse.ArgumentParser()
parser.add_argument("--model",
                    type=str,
                    required=False,
                    choices=["gpt-4.1", "o4-mini"])
parser.add_argument("--port", type=int, required=False, default=8000)
args = parser.parse_args()

MODEL = args.model
if MODEL is None:
    openai_api_key = "EMPTY"
    openai_api_base = f"http://localhost:{args.port}/v1"
else:
    openai_api_key = None
    openai_api_base = None

client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)


def test_basic():
    response = client.responses.create(
        model=MODEL,
        input="What is 13 * 24?",
        # max_output_tokens=10,
    )
    print(response)


def test_basic_with_instructions():
    response = client.responses.create(
        model=MODEL,
        input="What is 13 * 24?",
        instructions="Respond in Korean.",
    )
    print(response)


def test_basic_with_reasoning_effort():
    response = client.responses.create(
        model=MODEL,
        input="What is the capital of South Korea?",
        reasoning={"effort": "low"},
    )
    print(response)


def test_chat():
    response = client.responses.create(
        model=MODEL,
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
    print(response)


def test_chat_with_input_type():
    response = client.responses.create(
        model=MODEL,
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
    print(response)


def test_structured_output():
    response = client.responses.create(
        model=MODEL,
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
    print(response)


def test_structured_output_with_parse():

    from pydantic import BaseModel

    class CalendarEvent(BaseModel):
        name: str
        date: str
        participants: list[str]

    response = client.responses.parse(
        model=MODEL,
        input="Alice and Bob are going to a science fair on Friday",
        instructions="Extract the event information",
        text_format=CalendarEvent,
    )
    print(response)


def test_store():
    for store in [True, False]:
        response = client.responses.create(
            model=MODEL,
            input="What is 13 * 24?",
            store=store,
        )
        print(response)

        try:
            response = client.responses.retrieve(response.id)
            print(response)
        except NotFoundError:
            is_not_found = True
        else:
            is_not_found = False
        assert is_not_found == (not store)


def test_background():
    response = client.responses.create(
        model=MODEL,
        input="What is 13 * 24?",
        background=True,
    )
    print(response)

    while True:
        response = client.responses.retrieve(response.id)
        if response.status == "completed":
            break
        time.sleep(1)
    print(response)


def test_background_cancel():
    response = client.responses.create(
        model=MODEL,
        input="Write a long story about a cat.",
        background=True,
    )
    print(response)
    time.sleep(1)
    response = client.responses.cancel(response.id)
    print(response)


def test_stateful_multi_turn():
    response1 = client.responses.create(
        model=MODEL,
        input="What is 13 * 24?",
    )
    print(response1)

    response2 = client.responses.create(
        model=MODEL,
        input="What if I increase both numbers by 1?",
        previous_response_id=response1.id,
    )
    print(response2)

    response3 = client.responses.create(
        model=MODEL,
        input="Divide the result by 2.",
        previous_response_id=response2.id,
    )
    print(response3)


def test_streaming():
    response = client.responses.create(
        model=MODEL,
        input="What is 13 * 24? Explain your answer.",
        stream=True,
    )

    for event in response:
        if "text.delta" in event.type:
            print(event.delta, end="", flush=True)
    print()


def test_web_search():
    response = client.responses.create(
        model=MODEL,
        input="Who is the president of South Korea as of now?",
        tools=[{
            "type": "web_search_preview"
        }],
    )
    print(response)


def test_code_interpreter():
    response = client.responses.create(
        model=MODEL,
        input="Multiply 643258029438.6132 * 23516705917230.84279 using Python.",
        tools=[{
            "type": "code_interpreter",
            "container": {
                "type": "auto"
            }
        }],
    )
    print(response)


def get_weather(latitude, longitude):
    response = requests.get(
        f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m"
    )
    data = response.json()
    return data['current']['temperature_2m']


def get_place_to_travel():
    return "Paris"


def call_function(name, args):
    if name == "get_weather":
        return get_weather(**args)
    elif name == "get_place_to_travel":
        return get_place_to_travel()
    else:
        raise ValueError(f"Unknown function: {name}")


def test_function_calling():
    tools = [{
        "type": "function",
        "name": "get_weather",
        "description":
        "Get current temperature for provided coordinates in celsius.",
        "parameters": {
            "type": "object",
            "properties": {
                "latitude": {
                    "type": "number"
                },
                "longitude": {
                    "type": "number"
                }
            },
            "required": ["latitude", "longitude"],
            "additionalProperties": False
        },
        "strict": True
    }]

    response = client.responses.create(
        model=MODEL,
        input="What's the weather like in Paris today?",
        tools=tools,
    )
    print("The first response:")
    print(response)
    print("output:")
    for out in response.output:
        print(out)
    print("--------------------------------")

    assert len(response.output) == 2
    assert response.output[0].type == "reasoning"
    assert response.output[1].type == "function_call"

    tool_call = response.output[1]

    name = tool_call.name
    args = json.loads(tool_call.arguments)

    result = call_function(name, args)
    print("tool call result: ", result, type(result))

    response_2 = client.responses.create(
        model=MODEL,
        input=[{
            "type": "function_call_output",
            "call_id": tool_call.call_id,
            "output": str(result)
        }],
        tools=tools,
        previous_response_id=response.id,
    )
    print("The second response:")
    print(response_2)
    print("output:")
    for out in response_2.output:
        print(out)
    print("--------------------------------")
    print(response_2.output_text)

    # NOTE: chain-of-thought should be removed.
    response_3 = client.responses.create(
        model=MODEL,
        input="What's the weather like in Paris today?",
        tools=tools,
        previous_response_id=response_2.id,
    )
    print("The third response:")
    print(response_3)
    print("output:")
    for out in response_3.output:
        print(out)
    print("--------------------------------")
    print(response_3.output_text)


def test_function_calling_multi_turn():
    tools = [{
        "type": "function",
        "name": "get_place_to_travel",
        "description": "Get a random place to travel",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False
        },
        "strict": True
    }, {
        "type": "function",
        "name": "get_weather",
        "description":
        "Get current temperature for provided coordinates in celsius.",
        "parameters": {
            "type": "object",
            "properties": {
                "latitude": {
                    "type": "number"
                },
                "longitude": {
                    "type": "number"
                }
            },
            "required": ["latitude", "longitude"],
            "additionalProperties": False
        },
        "strict": True
    }]

    response = client.responses.create(
        model=MODEL,
        input=
        "Help me plan a trip to a random place. And tell me the weather there.",
        tools=tools,
    )
    print("The first response:")
    print(response)
    print("output:")
    for out in response.output:
        print(out)
    print("--------------------------------")

    assert len(response.output) == 2
    assert response.output[0].type == "reasoning"
    assert response.output[1].type == "function_call"

    tool_call = response.output[1]

    name = tool_call.name
    args = json.loads(tool_call.arguments)

    result = call_function(name, args)
    print("tool call result: ", result, type(result))

    response_2 = client.responses.create(
        model=MODEL,
        input=[{
            "type": "function_call_output",
            "call_id": tool_call.call_id,
            "output": str(result)
        }],
        tools=tools,
        previous_response_id=response.id,
    )
    print("The second response:")
    print(response_2)
    print("output:")
    for out in response_2.output:
        print(out)
    print("--------------------------------")
    assert len(response_2.output) == 2
    assert response_2.output[0].type == "reasoning"
    assert response_2.output[1].type == "function_call"

    tool_call = response_2.output[1]

    name = tool_call.name
    args = json.loads(tool_call.arguments)

    result = call_function(name, args)
    print("tool call result: ", result, type(result))

    response_3 = client.responses.create(
        model=MODEL,
        input=[{
            "type": "function_call_output",
            "call_id": tool_call.call_id,
            "output": str(result)
        }],
        tools=tools,
        previous_response_id=response_2.id,
    )
    print("The third response:")
    print(response_3)
    print("output:")
    for out in response_3.output:
        print(out)
    print("--------------------------------")
    print(response_3.output_text)


def test_function_calling_required():
    tools = [{
        "type": "function",
        "name": "get_weather",
        "description":
        "Get current temperature for provided coordinates in celsius.",
        "parameters": {
            "type": "object",
            "properties": {
                "latitude": {
                    "type": "number"
                },
                "longitude": {
                    "type": "number"
                }
            },
            "required": ["latitude", "longitude"],
            "additionalProperties": False
        },
        "strict": True
    }]
    try:
        response = client.responses.create(
            model=MODEL,
            input="What's the weather like in Paris today?",
            tools=tools,
            tool_choice="required",
        )
    except BadRequestError as e:
        print(e)
        return
    else:
        raise ValueError("Should raise BadRequestError")


def test_function_calling_full_history():
    tools = [{
        "type": "function",
        "name": "get_weather",
        "description":
        "Get current temperature for provided coordinates in celsius.",
        "parameters": {
            "type": "object",
            "properties": {
                "latitude": {
                    "type": "number"
                },
                "longitude": {
                    "type": "number"
                }
            },
            "required": ["latitude", "longitude"],
            "additionalProperties": False
        },
        "strict": True
    }]

    input_messages = [{
        "role": "user",
        "content": "What's the weather like in Paris today?"
    }]

    response = client.responses.create(
        model=MODEL,
        input=input_messages,
        tools=tools,
    )

    print(response)
    print("output:")
    for out in response.output:
        print(out)
    print("--------------------------------")

    tool_call = response.output[-1]
    name = tool_call.name
    args = json.loads(tool_call.arguments)

    result = call_function(name, args)
    print("tool call result: ", result, type(result))

    input_messages.extend(
        response.output)  # append model's function call message
    input_messages.append({  # append result message
        "type": "function_call_output",
        "call_id": tool_call.call_id,
        "output": str(result)
    })

    print("input_messages: ", input_messages)

    response_2 = client.responses.create(
        model=MODEL,
        input=input_messages,
        tools=tools,
    )
    print(response_2.output_text)


if __name__ == "__main__":
    # 1. Stateless & Non-streaming tests:
    print("===test_basic:")
    test_basic()
    print("===test_basic_with_instructions:")
    test_basic_with_instructions()
    print("===test_basic_with_reasoning_effort:")
    test_basic_with_reasoning_effort()
    print("===test_chat:")
    test_chat()  # should we overwrite system message?
    print("===test_chat_with_input_type:")
    test_chat_with_input_type()
    print("===test_structured_output:")
    test_structured_output()
    print("===test_structured_output_with_parse:")
    test_structured_output_with_parse()

    # 2. Stateful & Non-streaming tests:
    print("===test_store:")
    test_store()
    print("===test_background:")
    test_background()
    print("===test_background_cancel:")
    test_background_cancel()
    print("===test_stateful_multi_turn:")
    test_stateful_multi_turn()

    # 3. Streaming tests:
    # print("===test_streaming:")
    # test_streaming()

    # 4. Tool tests:
    print("===test_web_search:")
    test_web_search()  # can crash occasionally
    print("===test_code_interpreter:")
    test_code_interpreter()
    print("===test_function_calling:")
    test_function_calling()
    print("===test_function_calling_multi_turn:")
    test_function_calling_multi_turn()  # can crash occasionally
    print("===test_function_calling_required:")
    test_function_calling_required()
    print("===test_function_calling_full_history:")
    test_function_calling_full_history()
