# Interleaved Thinking

## Introduction

Interleaved thinking allows models to reason between tool calls, enabling more sophisticated decision-making after receiving tool results. This feature helps models chain multiple tool calls with reasoning steps in between and make nuanced decisions based on intermediate results.

Important: Interleaved thinking increases token usage and response latency. Consider your budget and performance requirements when enabling this feature.

## How Interleaved Thinking Works

With interleaved thinking, the model can:

- Reason about the results of a tool call before deciding what to do next
- Chain multiple tool calls with reasoning steps in between
- Make more nuanced decisions based on intermediate results
- Provide transparent reasoning for its tool selection process

## Supported Models

vLLM currently supports the following interleaved thinking models:

| Model Series | Reasoning Parser Name |
|--------------|-----------------------|
| moonshotai/Kimi-K2-Thinking    |  kimi_k2  |
| MiniMaxAI/MiniMax-M2           |  minimax_m2  |

## Example Usage

To use interleaved thinking with tool calls, specify a model that supports this feature and enable tool calls in your chat completion request. Here's an example:

??? code

    ```python
    """
    vllm serve MiniMaxAI/MiniMax-M2 \
      --tensor-parallel-size 4 \
      --tool-call-parser minimax_m2 \
      --reasoning-parser minimax_m2 \
      --enable-auto-tool-choice
    """
    import json
    
    from openai import OpenAI
    
    client = OpenAI(base_url="http://localhost:8000/v1",     api_key="dummy")
    
    
    def get_current_weather(location: str, unit: "str"):
        """Get the current weather in a given location"""
        if unit == "celsius":
            return f"The current temperature in {location} is 22°C."
        else:
            return f"The current temperature in {location} is 72°F."
    
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather in a given     location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City and state, e.g.,     'San Francisco, CA'",
                        },
                        "unit": {"type": "string", "enum":     ["celsius", "fahrenheit"]},
                    },
                    "required": ["location", "unit"],
                },
            },
        }
    ]
    messages = [{"role": "user", "content": "What's the weather in Fahrenheit like in San Francisco?"}]
    response = client.chat.completions.create(
        model=client.models.list().data[0].id,
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    
    tool_call = response.choices[0].message.tool_calls[0].function
    
    messages.append(
        {
            "role": "assistant",
            "tool_calls": response.choices[0].message.tool_calls,
            "reasoning": response.choices[0].message.reasoning, # append reasoning
        }
    )
    
    # Simulate tool execution
    available_tools = {"get_weather": get_current_weather}
    
    completion_tool_calls = response.choices[0].message.tool_calls
    for call in completion_tool_calls:
        tool_to_call = available_tools[call.function.name]
        args = json.loads(call.function.arguments)
        result = tool_to_call(**args)
        messages.append(
            {
                "role": "tool",
                "content": result,
                "tool_call_id": call.id,
                "name": call.function.name,
            }
        )
    response_2 = client.chat.completions.create(
        model=client.models.list().data[0].id,
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    print(response_2.choices[0].message.content)
    ```
This example demonstrates how to set up interleaved thinking with tool calls using a weather retrieval function. The model reasons about the tool results before generating the final response.
