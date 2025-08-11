# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
An example demonstrates how to use tool calling with reasoning models 
like QwQ-32B. The reasoning_content will not be parsed by the tool 
calling process; only the final output will be parsed.

To run this example, you need to start the vLLM server with both 
the reasoning parser and tool calling enabled.

```bash
vllm serve Qwen/QwQ-32B \
     --reasoning-parser deepseek_r1 \
     --enable-auto-tool-choice --tool-call-parser hermes
     
```

"""

from openai import OpenAI


# Now, simulate a tool call
def get_current_weather(city: str, state: str, unit: "str"):
    return (
        "The weather in Dallas, Texas is 85 degrees fahrenheit. It is "
        "partly cloudly, with highs in the 90's."
    )


available_tools = {"get_current_weather": get_current_weather}

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

properties = {
    "city": {
        "type": "string",
        "description": "The city to find the weather for, e.g. 'San Francisco'",
    },
    "state": {
        "type": "string",
        "description": "the two-letter abbreviation for the state that the city is"
        " in, e.g. 'CA' which would mean 'California'",
    },
    "unit": {
        "type": "string",
        "description": "The unit to fetch the temperature in",
        "enum": ["celsius", "fahrenheit"],
    },
}

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": ["city", "state", "unit"],
            },
        },
    }
]
messages = [
    {"role": "user", "content": "Hi! How are you doing today?"},
    {"role": "assistant", "content": "I'm doing well! How can I help you?"},
    {
        "role": "user",
        "content": (
            "Can you tell me what the temperate will be in Dallas, in fahrenheit?"
        ),
    },
]


def extract_reasoning_and_calls(chunks: list):
    reasoning_content = ""
    tool_call_idx = -1
    arguments = []
    function_names = []
    for chunk in chunks:
        if chunk.choices[0].delta.tool_calls:
            tool_call = chunk.choices[0].delta.tool_calls[0]
            if tool_call.index != tool_call_idx:
                tool_call_idx = chunk.choices[0].delta.tool_calls[0].index
                arguments.append("")
                function_names.append("")

            if tool_call.function:
                if tool_call.function.name:
                    function_names[tool_call_idx] = tool_call.function.name

                if tool_call.function.arguments:
                    arguments[tool_call_idx] += tool_call.function.arguments
        else:
            if hasattr(chunk.choices[0].delta, "reasoning_content"):
                reasoning_content += chunk.choices[0].delta.reasoning_content
    return reasoning_content, arguments, function_names


def main():
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    models = client.models.list()
    model = models.data[0].id

    print("---------Full Generate With Automatic Function Calling-------------")
    tool_calls = client.chat.completions.create(
        messages=messages, model=model, tools=tools
    )
    print(f"reasoning_content: {tool_calls.choices[0].message.reasoning_content}")
    print(f"function name: {tool_calls.choices[0].message.tool_calls[0].function.name}")
    print(
        f"function arguments: "
        f"{tool_calls.choices[0].message.tool_calls[0].function.arguments}"
    )

    print("----------Stream Generate With Automatic Function Calling-----------")
    tool_calls_stream = client.chat.completions.create(
        messages=messages, model=model, tools=tools, stream=True
    )

    chunks = list(tool_calls_stream)

    reasoning_content, arguments, function_names = extract_reasoning_and_calls(chunks)

    print(f"reasoning_content: {reasoning_content}")
    print(f"function name: {function_names[0]}")
    print(f"function arguments: {arguments[0]}")

    print("----------Full Generate With Named Function Calling-----------------")
    tool_calls = client.chat.completions.create(
        messages=messages,
        model=model,
        tools=tools,
        tool_choice={"type": "function", "function": {"name": "get_current_weather"}},
    )

    tool_call = tool_calls.choices[0].message.tool_calls[0].function
    print(f"reasoning_content: {tool_calls.choices[0].message.reasoning_content}")
    print(f"function name: {tool_call.name}")
    print(f"function arguments: {tool_call.arguments}")
    print("----------Stream Generate With Named Function Calling--------------")

    tool_calls_stream = client.chat.completions.create(
        messages=messages,
        model=model,
        tools=tools,
        tool_choice={"type": "function", "function": {"name": "get_current_weather"}},
        stream=True,
    )

    chunks = list(tool_calls_stream)

    reasoning_content, arguments, function_names = extract_reasoning_and_calls(chunks)
    print(f"reasoning_content: {reasoning_content}")
    print(f"function name: {function_names[0]}")
    print(f"function arguments: {arguments[0]}")
    print("\n\n")


if __name__ == "__main__":
    main()
