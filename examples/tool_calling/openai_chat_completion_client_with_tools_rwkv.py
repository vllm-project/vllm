# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Start an RWKV7 OpenAI-compatible server with tool calls enabled:

VLLM_RWKV7_WKV_MODE=fp16 VLLM_RWKV7_EMB_DEVICE=gpu \
vllm serve /path/to/rwkv7-g1g-7.2b-20260523-ctx8192.pth \
    --enable-auto-tool-choice
"""

import json
from typing import Any

from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")

tools: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name, e.g. Paris.",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                    },
                },
                "required": ["city", "unit"],
            },
        },
    }
]


def get_weather(city: str, unit: str) -> str:
    return f"The weather in {city} is 21 degrees {unit}."


messages: list[dict[str, Any]] = [
    {
        "role": "user",
        "content": "Use the weather tool for Paris in celsius.",
    }
]

model = client.models.list().data[0].id
response = client.chat.completions.create(
    model=model,
    messages=messages,
    tools=tools,
    tool_choice="auto",
)

message = response.choices[0].message
print(message)

if message.tool_calls:
    messages.append(
        {
            "role": "assistant",
            "content": message.content,
            "tool_calls": message.tool_calls,
        }
    )
    available_tools = {"get_weather": get_weather}
    for tool_call in message.tool_calls:
        function = tool_call.function
        result = available_tools[function.name](**json.loads(function.arguments))
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": function.name,
                "content": result,
            }
        )

    final_response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
    )
    print(final_response.choices[0].message)
