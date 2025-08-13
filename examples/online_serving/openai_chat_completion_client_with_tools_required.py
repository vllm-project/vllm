# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
To run this example, you can start the vLLM server
without any specific flags:

```bash
VLLM_USE_V1=0 vllm serve unsloth/Llama-3.2-1B-Instruct \
    --structured-outputs-config.backend outlines
```

This example demonstrates how to generate chat completions
using the OpenAI Python client library.
"""

from openai import OpenAI

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

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
                        "description": "The city to find the weather for"
                        ", e.g. 'San Francisco'",
                    },
                    "state": {
                        "type": "string",
                        "description": (
                            "the two-letter abbreviation for the state that the "
                            "city is in, e.g. 'CA' which would mean 'California'"
                        ),
                    },
                    "unit": {
                        "type": "string",
                        "description": "The unit to fetch the temperature in",
                        "enum": ["celsius", "fahrenheit"],
                    },
                },
                "required": ["city", "state", "unit"],
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
                        "description": (
                            "The city to get the forecast for, e.g. 'New York'"
                        ),
                    },
                    "state": {
                        "type": "string",
                        "description": (
                            "The two-letter abbreviation for the state, e.g. 'NY'"
                        ),
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
                "required": ["city", "state", "days", "unit"],
            },
        },
    },
]

messages = [
    {"role": "user", "content": "Hi! How are you doing today?"},
    {"role": "assistant", "content": "I'm doing well! How can I help you?"},
    {
        "role": "user",
        "content": "Can you tell me what the current weather is in Dallas \
            and the forecast for the next 5 days, in fahrenheit?",
    },
]


def main():
    client = OpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    models = client.models.list()
    model = models.data[0].id

    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model,
        tools=tools,
        tool_choice="required",
        stream=True,  # Enable streaming response
    )

    for chunk in chat_completion:
        if chunk.choices and chunk.choices[0].delta.tool_calls:
            print(chunk.choices[0].delta.tool_calls)

    chat_completion = client.chat.completions.create(
        messages=messages, model=model, tools=tools, tool_choice="required"
    )

    print(chat_completion.choices[0].message.tool_calls)


if __name__ == "__main__":
    main()
