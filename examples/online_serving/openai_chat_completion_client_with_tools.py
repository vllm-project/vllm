# SPDX-License-Identifier: Apache-2.0
"""
Set up this example by starting a vLLM OpenAI-compatible server with tool call
options enabled. For example:

IMPORTANT: for mistral, you must use one of the provided mistral tool call
templates, or your own - the model default doesn't work for tool calls with vLLM
See the vLLM docs on OpenAI server & tool calling for more details.

vllm serve mistralai/Mistral-7B-Instruct-v0.3 \
            --chat-template examples/tool_chat_template_mistral.jinja \
            --enable-auto-tool-choice --tool-call-parser mistral

OR
vllm serve NousResearch/Hermes-2-Pro-Llama-3-8B \
            --chat-template examples/tool_chat_template_hermes.jinja \
            --enable-auto-tool-choice --tool-call-parser hermes
"""
import json
from typing import Any

from openai import OpenAI

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

tools = [{
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type":
                    "string",
                    "description":
                    "The city to find the weather for, e.g. 'San Francisco'"
                },
                "state": {
                    "type":
                    "string",
                    "description":
                    "the two-letter abbreviation for the state that the city is"
                    " in, e.g. 'CA' which would mean 'California'"
                },
                "unit": {
                    "type": "string",
                    "description": "The unit to fetch the temperature in",
                    "enum": ["celsius", "fahrenheit"]
                }
            },
            "required": ["city", "state", "unit"]
        }
    }
}]

messages = [{
    "role": "user",
    "content": "Hi! How are you doing today?"
}, {
    "role": "assistant",
    "content": "I'm doing well! How can I help you?"
}, {
    "role":
    "user",
    "content":
    "Can you tell me what the temperate will be in Dallas, in fahrenheit?"
}]


def get_current_weather(city: str, state: str, unit: 'str'):
    return ("The weather in Dallas, Texas is 85 degrees fahrenheit. It is "
            "partly cloudly, with highs in the 90's.")


def handle_tool_calls_stream(
    client: OpenAI,
    messages: list[dict[str, str]],
    model: str,
    tools: list[dict[str, Any]],
) -> list[Any]:
    tool_calls_stream = client.chat.completions.create(messages=messages,
                                                       model=model,
                                                       tools=tools,
                                                       stream=True)
    chunks = []
    print("chunks: ")
    for chunk in tool_calls_stream:
        chunks.append(chunk)
        if chunk.choices[0].delta.tool_calls:
            print(chunk.choices[0].delta.tool_calls[0])
        else:
            print(chunk.choices[0].delta)
    return chunks


def handle_tool_calls_arguments(chunks: list[Any]) -> list[str]:
    arguments = []
    tool_call_idx = -1
    print("arguments: ")
    for chunk in chunks:
        if chunk.choices[0].delta.tool_calls:
            tool_call = chunk.choices[0].delta.tool_calls[0]
            if tool_call.index != tool_call_idx:
                if tool_call_idx >= 0:
                    print(f"streamed tool call arguments: "
                          f"{arguments[tool_call_idx]}")
                tool_call_idx = chunk.choices[0].delta.tool_calls[0].index
                arguments.append("")
            if tool_call.id:
                print(f"streamed tool call id: {tool_call.id} ")

            if tool_call.function:
                if tool_call.function.name:
                    print(
                        f"streamed tool call name: {tool_call.function.name}")

                if tool_call.function.arguments:
                    arguments[tool_call_idx] += tool_call.function.arguments

    return arguments


def main():
    # Initialize OpenAI client
    client = OpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    # Get available models and select one
    models = client.models.list()
    model = models.data[0].id

    chat_completion = client.chat.completions.create(messages=messages,
                                                     model=model,
                                                     tools=tools)

    print("-" * 70)
    print("Chat completion results:")
    print(chat_completion)
    print("-" * 70)

    # Stream tool calls
    chunks = handle_tool_calls_stream(client, messages, model, tools)
    print("-" * 70)

    # Handle arguments from streamed tool calls
    arguments = handle_tool_calls_arguments(chunks)

    if len(arguments):
        print(f"streamed tool call arguments: {arguments[-1]}\n")

    print("-" * 70)

    # Add tool call results to the conversation
    messages.append({
        "role": "assistant",
        "tool_calls": chat_completion.choices[0].message.tool_calls
    })

    # Now, simulate a tool call
    available_tools = {"get_current_weather": get_current_weather}

    completion_tool_calls = chat_completion.choices[0].message.tool_calls
    for call in completion_tool_calls:
        tool_to_call = available_tools[call.function.name]
        args = json.loads(call.function.arguments)
        result = tool_to_call(**args)
        print("tool_to_call result: ", result)
        messages.append({
            "role": "tool",
            "content": result,
            "tool_call_id": call.id,
            "name": call.function.name
        })

    chat_completion_2 = client.chat.completions.create(messages=messages,
                                                       model=model,
                                                       tools=tools,
                                                       stream=False)
    print("Chat completion2 results:")
    print(chat_completion_2)
    print("-" * 70)


if __name__ == "__main__":
    main()
