# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import openai  # use the official client for correctness check
import pytest

MODEL_NAME = "Qwen/Qwen3-1.7B"
tools = [
    {
        "type": "function",
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city to find the weather for, e.g. 'Vienna'",
                    "default": "Vienna",
                },
                "country": {
                    "type": "string",
                    "description": "The country that the city is in, e.g. 'Austria'",
                },
                "unit": {
                    "type": "string",
                    "description": "The unit to fetch the temperature in",
                    "enum": ["celsius", "fahrenheit"],
                },
                "options": {
                    "$ref": "#/$defs/WeatherOptions",
                    "description": "Optional parameters for weather query",
                },
            },
            "required": ["country", "unit"],
            "$defs": {
                "WeatherOptions": {
                    "title": "WeatherOptions",
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "default": "celsius",
                            "description": "Temperature unit",
                            "title": "Temperature Unit",
                        },
                        "include_forecast": {
                            "type": "boolean",
                            "default": False,
                            "description": "Whether to include a 24-hour forecast",
                            "title": "Include Forecast",
                        },
                        "language": {
                            "type": "string",
                            "default": "zh-CN",
                            "description": "Language of the response",
                            "title": "Language",
                            "enum": ["zh-CN", "en-US", "ja-JP"],
                        },
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "name": "get_forecast",
        "description": "Get the weather forecast for a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city to get the forecast for, e.g. 'Vienna'",
                    "default": "Vienna",
                },
                "country": {
                    "type": "string",
                    "description": "The country that the city is in, e.g. 'Austria'",
                },
                "days": {
                    "type": "integer",
                    "description": "Number of days to get the forecast for (1-7)",
                },
                "unit": {
                    "type": "string",
                    "description": "The unit to fetch the temperature in",
                    "enum": ["celsius", "fahrenheit"],
                },
            },
            "required": ["country", "days", "unit"],
        },
    },
]


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
@pytest.mark.parametrize("tool_choice", ["auto", "required"])
async def test_function_tool_use(
    client: openai.AsyncOpenAI, model_name: str, tool_choice: str
):
    prompt = [
        {
            "role": "user",
            "content": "Can you tell me what the current weather is in Berlin and the "
            "forecast for the next 5 days, in fahrenheit?",
        },
    ]
    response = await client.responses.create(
        model=model_name,
        input=prompt,
        tools=tools,
        tool_choice=tool_choice,
    )

    assert len(response.output) >= 1
    tool_call = None
    reasoning = None
    for out in response.output:
        if out.type == "function_call":
            tool_call = out
        if out.type == "reasoning":
            reasoning = out
    assert tool_call is not None
    assert tool_call.type == "function_call"
    assert json.loads(tool_call.arguments) is not None
    assert reasoning is not None
    assert reasoning.type == "reasoning"


@pytest.mark.asyncio
async def test_named_tool_use(client: openai.AsyncOpenAI):
    def get_weather(latitude: float, longitude: float) -> str:
        """
        Mock function to simulate getting weather data.
        In a real application, this would call an external weather API.
        """
        return f"Current temperature at ({latitude}, {longitude}) is 20°C."

    tools = [
        {
            "type": "function",
            "name": "get_weather",
            "description": (
                "Get current temperature for provided coordinates in celsius."
            ),
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
    ]

    input_messages = [
        {"role": "user", "content": "What's the weather like in Paris today?"}
    ]

    response = await client.responses.create(
        model=MODEL_NAME,
        input=input_messages,
        tools=tools,
        tool_choice={"type": "function", "name": "get_weather"},
    )
    assert len(response.output) >= 1
    for out in response.output:
        if out.type == "function_call":
            tool_call = out
    assert tool_call is not None
    assert tool_call.type == "function_call"
    assert tool_call.name == "get_weather"
    args = json.loads(tool_call.arguments)
    assert args["latitude"] is not None
    assert args["longitude"] is not None
    # call the tool
    result = get_weather(args["latitude"], args["longitude"])
    input_messages.append(tool_call)  # append model's function call message
    input_messages.append(
        {  # append result message
            "type": "function_call_output",
            "call_id": tool_call.call_id,
            "output": str(result),
        }
    )
    # create a new response with the tool call result
    response_2 = await client.responses.create(model=MODEL_NAME, input=input_messages)
    # check the output
    assert len(response_2.output_text) > 0
