# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import openai  # use the official client for correctness check
import pytest


MODEL_NAME = "Qwen/Qwen3-0.6B"

@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
@pytest.mark.parametrize("tool_choice", ["auto", "required"])
async def test_function_tool_use(client: openai.AsyncOpenAI, model_name: str,
                                 tool_choice: str):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description":
                            "The city to find the weather for, e.g. 'Vienna'",
                            "default": "Vienna",
                        },
                        "country": {
                            "type":
                            "string",
                            "description":
                            "The country that the city is in, e.g. 'Austria'",
                        },
                        "unit": {
                            "type": "string",
                            "description":
                            "The unit to fetch the temperature in",
                            "enum": ["celsius", "fahrenheit"],
                        },
                        "options": {
                            "$ref": "#/$defs/WeatherOptions",
                            "description":
                            "Optional parameters for weather query",
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
                                    "description":
                                    "Whether to include a 24-hour forecast",
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
        },
        {
            "type": "function",
            "function": {
                "name": "get_forecast",
                "description": "Get the weather forecast for a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description":
                            "The city to get the forecast for, e.g. 'Vienna'",
                            "default": "Vienna",
                        },
                        "country": {
                            "type":
                            "string",
                            "description":
                            "The country that the city is in, e.g. 'Austria'",
                        },
                        "days": {
                            "type":
                            "integer",
                            "description":
                            "Number of days to get the forecast for (1-7)",
                        },
                        "unit": {
                            "type": "string",
                            "description":
                            "The unit to fetch the temperature in",
                            "enum": ["celsius", "fahrenheit"],
                        },
                    },
                    "required": ["country", "days", "unit"],
                },
            },
        },
    ]

    prompt = [{
            "role":
            "user",
            "content":
            "Can you tell me what the current weather is in Berlin and the "\
            "forecast for the next 5 days, in fahrenheit?",
        },]
    response = client.responses.create(
        model=model_name,
        input=prompt,
        tools=tools,
        tool_choice=tool_choice,
    )
    
    assert len(response.output) >= 1
    tool_call = response.output[0]
    
    assert tool_call.type == "function_call"
    assert json.loads(tool_call.arguments) is not None

@pytest.mark.asyncio
async def test_named_tool_use(client: openai.AsyncOpenAI, sample_json_schema):
    pass
