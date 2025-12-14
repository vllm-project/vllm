# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Example demonstrating tool calling + structured output issue in vLLM.
When both are used together, tools are ignored even with tool_choice='auto'.
"""

from openai import OpenAI

# Initialize client (assumes vLLM server running on localhost:8000)
client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

# Define a simple tool
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "The city name"}
                },
                "required": ["location"],
            },
        },
    }
]

# Define a response format (structured output)
response_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "response",
        "schema": {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        },
    },
}

# Test: tool should be called for this query
messages = [{"role": "user", "content": "What's the weather in San Francisco?"}]

print("Testing tool calling + structured output together...\n")

# Test WITH structured output (should fail to call tool)
response = client.chat.completions.create(
    model="Qwen/Qwen2.5-1.5B-Instruct",  # Replace with your model
    messages=messages,
    tools=tools,
    tool_choice="auto",
    response_format=response_format,
)

print("Response with both tools and response_format:")
print(f"Tool calls: {response.choices[0].message.tool_calls}")
print(f"Content: {response.choices[0].message.content}")

# Check if tool was called (it should be, but currently isn't due to the bug)
if response.choices[0].message.tool_calls:
    print("✅ GOOD: Tool was called (expected behavior)")
else:
    print(
        "❌ BAD: Tool was NOT called (bug - should call tool even with response_format)"
    )
print()

# Test WITHOUT structured output (should work)
response2 = client.chat.completions.create(
    model="Qwen/Qwen2.5-1.5B-Instruct",  # Replace with your model
    messages=messages,
    tools=tools,
    tool_choice="auto",
)

print("Response with only tools (no response_format):")
print(f"Tool calls: {response2.choices[0].message.tool_calls}")
print(f"Content: {response2.choices[0].message.content}")

# Check if tool was called
if response2.choices[0].message.tool_calls:
    print("✅ GOOD: Tool was called (expected behavior)")
else:
    print(
        "❌ BAD: Tool was NOT called (unexpected - should work without response_format)"
    )
