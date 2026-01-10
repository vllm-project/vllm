# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""
Set up this example by starting a vLLM OpenAI-compatible server with tool call
options enabled for xLAM-2 models:

vllm serve --model Salesforce/Llama-xLAM-2-8b-fc-r --enable-auto-tool-choice --tool-call-parser xlam

OR

vllm serve --model Salesforce/xLAM-2-3b-fc-r --enable-auto-tool-choice --tool-call-parser xlam
"""

import ast
import json
import operator
import time

from openai import OpenAI

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "empty"
openai_api_base = "http://localhost:8000/v1"


# Define tool functions
def get_weather(location: str, unit: str):
    return f"Weather in {location} is 22 degrees {unit}."


def calculate_expression(expression: str):
    """Safely evaluate a mathematical expression.

    Uses ast.parse to ensure only safe numeric operations are performed,
    preventing arbitrary code execution.
    """
    # Define allowed operators for safe math evaluation
    allowed_operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def safe_eval(node):
        if isinstance(node, ast.Constant):  # Numbers
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"Unsupported constant: {node.value}")
        elif isinstance(node, ast.BinOp):  # Binary operations
            if type(node.op) not in allowed_operators:
                raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
            left = safe_eval(node.left)
            right = safe_eval(node.right)
            return allowed_operators[type(node.op)](left, right)
        elif isinstance(node, ast.UnaryOp):  # Unary operations (-x, +x)
            if type(node.op) not in allowed_operators:
                raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
            operand = safe_eval(node.operand)
            return allowed_operators[type(node.op)](operand)
        elif isinstance(node, ast.Expression):
            return safe_eval(node.body)
        else:
            raise ValueError(f"Unsupported expression type: {type(node).__name__}")

    try:
        tree = ast.parse(expression, mode="eval")
        result = safe_eval(tree)
        return f"The result of {expression} is {result}"
    except (ValueError, SyntaxError) as e:
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
                        "description": "Mathematical expression to evaluate, needs to be a valid python expression",
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


def process_response(response, tool_functions, original_query):
    """Process a non-streaming response with possible tool calls"""

    print("\n--- Response Output ---")

    # Check if the response has content
    if response.choices[0].message.content:
        print(f"Content: {response.choices[0].message.content}")

    # Check if the response has tool calls
    if response.choices[0].message.tool_calls:
        print("--------------------------------")
        print(f"Tool calls: {response.choices[0].message.tool_calls}")
        print("--------------------------------")

        # Collect all tool calls and results before making follow-up request
        tool_results = []
        assistant_message = {"role": "assistant"}

        if response.choices[0].message.content:
            assistant_message["content"] = response.choices[0].message.content

        assistant_tool_calls = []

        # Process each tool call
        for tool_call in response.choices[0].message.tool_calls:
            function_name = tool_call.function.name
            function_args = tool_call.function.arguments
            function_id = tool_call.id

            print(f"Function called: {function_name}")
            print(f"Arguments: {function_args}")
            print(f"Function ID: {function_id}")

            # Execute the function
            try:
                # Parse the JSON arguments
                args = json.loads(function_args)

                # Call the function with the arguments
                function_result = tool_functions[function_name](**args)
                print(f"\n--- Function Result ---\n{function_result}\n")

                # Add tool call to assistant message
                assistant_tool_calls.append(
                    {
                        "id": function_id,
                        "type": "function",
                        "function": {"name": function_name, "arguments": function_args},
                    }
                )

                # Add tool result to tool_results
                tool_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": function_id,
                        "content": function_result,
                    }
                )

            except Exception as e:
                print(f"Error executing function: {e}")

        # Add tool_calls to assistant message
        assistant_message["tool_calls"] = assistant_tool_calls

        # Create a follow-up message with all function results
        follow_up_messages = [
            {"role": "user", "content": original_query},
            assistant_message,
        ]

        # Add all tool results to the messages
        follow_up_messages.extend(tool_results)

        # Get completion with all tool results in a single follow-up
        follow_up_response = client.chat.completions.create(
            model=client.models.list().data[0].id,
            messages=follow_up_messages,
            stream=False,
        )

        print("\n--- Follow-up Response ---")
        print(follow_up_response.choices[0].message.content)
        print("--- End Follow-up ---\n")

    print("--- End Response ---\n")


def run_test_case(query, test_name):
    """Run a single test case with the given query"""
    print(f"\n{'=' * 50}\nTEST CASE: {test_name}\n{'=' * 50}")
    print(f"Query: '{query}'")

    start_time = time.time()

    # Create non-streaming chat completion request
    response = client.chat.completions.create(
        model=client.models.list().data[0].id,
        messages=[{"role": "user", "content": query}],
        tools=tools,
        tool_choice="auto",
        stream=False,
    )

    # Process the non-streaming response, passing the original query
    process_response(response, tool_functions, query)

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
