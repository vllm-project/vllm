# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""
Set up this example by starting a vLLM OpenAI-compatible server with tool call
options enabled for xLAM-2 models:

vllm serve --model Salesforce/Llama-xLAM-2-8b-fc-r --enable-auto-tool-choice --tool-call-parser xlam

OR

vllm serve --model Salesforce/xLAM-2-3b-fc-r --enable-auto-tool-choice --tool-call-parser xlam

This example demonstrates streaming tool calls with xLAM models.
"""

import json
import time

from openai import OpenAI

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "empty"
openai_api_base = "http://localhost:8000/v1"


# Define tool functions
def get_weather(location: str, unit: str):
    return f"Weather in {location} is 22 degrees {unit}."


def calculate_expression(expression: str):
    try:
        result = eval(expression)
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Could not calculate {expression}: {e}"


def translate_text(text: str, target_language: str):
    return f"Translation of '{text}' to {target_language}: [translated content]"


# Define tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City and state, e.g., 'San Francisco, CA'",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location", "unit"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_expression",
            "description": "Calculate a mathematical expression",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate, needs to be a valid Python expression",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "translate_text",
            "description": "Translate text to another language",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to translate"},
                    "target_language": {
                        "type": "string",
                        "description": "Target language for translation",
                    },
                },
                "required": ["text", "target_language"],
            },
        },
    },
]

# Map of function names to implementations
tool_functions = {
    "get_weather": get_weather,
    "calculate_expression": calculate_expression,
    "translate_text": translate_text,
}


def process_stream(response, tool_functions, original_query):
    """Process a streaming response with possible tool calls"""
    # Track multiple tool calls
    tool_calls = {}  # Dictionary to store tool calls by ID

    current_id = None

    print("\n--- Stream Output ---")
    for chunk in response:
        # Handle tool calls in the stream
        if chunk.choices[0].delta.tool_calls:
            for tool_call_chunk in chunk.choices[0].delta.tool_calls:
                # Get the tool call ID
                if hasattr(tool_call_chunk, "id") and tool_call_chunk.id:
                    current_id = tool_call_chunk.id
                    if current_id not in tool_calls:
                        tool_calls[current_id] = {
                            "function_name": None,
                            "function_args": "",
                            "function_id": current_id,
                        }

                # Extract function information as it comes in chunks
                if (
                    hasattr(tool_call_chunk, "function")
                    and current_id
                    and current_id in tool_calls
                ):
                    if (
                        hasattr(tool_call_chunk.function, "name")
                        and tool_call_chunk.function.name
                    ):
                        tool_calls[current_id]["function_name"] = (
                            tool_call_chunk.function.name
                        )
                        print(f"Function called: {tool_call_chunk.function.name}")

                    if (
                        hasattr(tool_call_chunk.function, "arguments")
                        and tool_call_chunk.function.arguments
                    ):
                        tool_calls[current_id]["function_args"] += (
                            tool_call_chunk.function.arguments
                        )
                        print(f"Arguments chunk: {tool_call_chunk.function.arguments}")

        # Handle regular content in the stream
        elif chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="")

    print("\n--- End Stream ---\n")

    # Execute each function call and build messages for follow-up
    follow_up_messages = [{"role": "user", "content": original_query}]

    for tool_id, tool_data in tool_calls.items():
        function_name = tool_data["function_name"]
        function_args = tool_data["function_args"]
        function_id = tool_data["function_id"]

        if function_name and function_args:
            try:
                # Parse the JSON arguments
                args = json.loads(function_args)

                # Call the function with the arguments
                function_result = tool_functions[function_name](**args)
                print(
                    f"\n--- Function Result ({function_name}) ---\n{function_result}\n"
                )

                # Add the assistant message with tool call
                follow_up_messages.append(
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": function_id,
                                "type": "function",
                                "function": {
                                    "name": function_name,
                                    "arguments": function_args,
                                },
                            }
                        ],
                    }
                )

                # Add the tool message with function result
                follow_up_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": function_id,
                        "content": function_result,
                    }
                )

            except Exception as e:
                print(f"Error executing function: {e}")

    # Only send follow-up if we have results to process
    if len(follow_up_messages) > 1:
        # Create a follow-up message with all the function results
        follow_up_response = client.chat.completions.create(
            model=client.models.list().data[0].id,
            messages=follow_up_messages,
            stream=True,
        )

        print("\n--- Follow-up Response ---")
        for chunk in follow_up_response:
            if chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="")
        print("\n--- End Follow-up ---\n")


def run_test_case(query, test_name):
    """Run a single test case with the given query"""
    print(f"\n{'=' * 50}\nTEST CASE: {test_name}\n{'=' * 50}")
    print(f"Query: '{query}'")

    start_time = time.time()

    # Create streaming chat completion request
    response = client.chat.completions.create(
        model=client.models.list().data[0].id,
        messages=[{"role": "user", "content": query}],
        tools=tools,
        tool_choice="auto",
        stream=True,
    )

    # Process the streaming response
    process_stream(response, tool_functions, query)

    end_time = time.time()
    print(f"Test completed in {end_time - start_time:.2f} seconds")


def main():
    # Initialize OpenAI client
    global client
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    # Run test cases
    test_cases = [
        ("I want to know the weather in San Francisco", "Weather Information"),
        ("Calculate 25 * 17 + 31", "Math Calculation"),
        ("Translate 'Hello world' to Spanish", "Text Translation"),
        ("What is the weather in Tokyo and New York in celsius", "Multiple Tool Usage"),
    ]

    # Execute all test cases
    for query, test_name in test_cases:
        run_test_case(query, test_name)
        time.sleep(1)  # Small delay between tests

    print("\nAll tests completed.")


if __name__ == "__main__":
    main()
