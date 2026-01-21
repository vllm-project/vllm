# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Set up this example by starting a vLLM OpenAI-compatible server with tool call
options enabled.
Reasoning models can be used through the Responses API as seen here
https://platform.openai.com/docs/api-reference/responses
For example:
vllm serve Qwen/Qwen3-1.7B --reasoning-parser qwen3 \
      --structured-outputs-config.backend xgrammar \
      --enable-auto-tool-choice --tool-call-parser hermes
"""

import json

from openai import OpenAI
from utils import get_first_model


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
        "description": "Get current temperature for provided coordinates in celsius.",
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


def main():
    base_url = "http://0.0.0.0:8000/v1"
    client = OpenAI(base_url=base_url, api_key="empty")
    model = get_first_model(client)
    response = client.responses.create(
        model=model, input=input_messages, tools=tools, tool_choice="required"
    )

    for out in response.output:
        if out.type == "function_call":
            print("Function call:", out.name, out.arguments)
            tool_call = out
    args = json.loads(tool_call.arguments)
    result = get_weather(args["latitude"], args["longitude"])

    input_messages.append(tool_call)  # append model's function call message
    input_messages.append(
        {  # append result message
            "type": "function_call_output",
            "call_id": tool_call.call_id,
            "output": str(result),
        }
    )
    response_2 = client.responses.create(
        model=model,
        input=input_messages,
        tools=tools,
    )
    print(response_2.output_text)


if __name__ == "__main__":
    main()
