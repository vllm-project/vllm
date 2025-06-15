# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from mistral_common.protocol.instruct.messages import (AssistantMessage,
                                                       ToolMessage,
                                                       UserMessage)
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.protocol.instruct.tool_calls import (Function,
                                                         FunctionCall, Tool,
                                                         ToolCall)

from vllm.transformers_utils.tokenizers.mistral import (
    make_mistral_chat_completion_request)


@pytest.mark.parametrize(
    "openai_request,expected_mistral_request",
    [(
        {
            "messages": [{
                "role": "user",
                "content": "What is the current local date and time?",
            }],
            "tools": [{
                "type": "function",
                "function": {
                    "description": "Fetch the current local date and time.",
                    "name": "get_current_time",
                },
            }],
        },
        ChatCompletionRequest(
            messages=[
                UserMessage(content="What is the current local date and time?")
            ],
            tools=[
                Tool(
                    type="function",
                    function=Function(
                        name="get_current_time",
                        description="Fetch the current local date and time.",
                        parameters={},
                    ),
                )
            ],
        ),
    ),
     (
         {
             "messages":
             [{
                 "role": "user",
                 "content": "What is the current local date and time?",
             }],
             "tools": [{
                 "type": "function",
                 "function": {
                     "description": "Fetch the current local date and time.",
                     "name": "get_current_time",
                     "parameters": None,
                 },
             }],
         },
         ChatCompletionRequest(
             messages=[
                 UserMessage(
                     content="What is the current local date and time?")
             ],
             tools=[
                 Tool(
                     type="function",
                     function=Function(
                         name="get_current_time",
                         description="Fetch the current local date and time.",
                         parameters={},
                     ),
                 )
             ],
         ),
     )],
)
def test_make_mistral_chat_completion_request(openai_request,
                                              expected_mistral_request):
    actual_request = make_mistral_chat_completion_request(
        openai_request["messages"], openai_request["tools"])
    assert actual_request == expected_mistral_request


# Tool use with list content and reasoning_content
@pytest.mark.parametrize("openai_request,expected_mistral_request", [(
    {
        "messages": [
            {
                "role": "user",
                "content": "What's the weather in Paris?",
            },
            {
                "role":
                "assistant",
                "reasoning_content":
                None,
                "content":
                None,
                "tool_calls": [{
                    "id": "call123",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city": "Paris"}',
                    },
                }],
            },
            {
                "role": "tool",
                "content": [{
                    "type": "text",
                    "text": "Rainy"
                }],
                "name": "get_weather",
                "tool_call_id": "call123",
            },
        ],
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Gets the current weather in a city.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "The city name"
                        }
                    },
                    "required": ["city"],
                },
            },
        }],
    },
    ChatCompletionRequest(
        messages=[
            UserMessage(content="What's the weather in Paris?"),
            AssistantMessage(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="call123",
                        function=FunctionCall(
                            name="get_weather",
                            arguments='{"city": "Paris"}',
                        ),
                    )
                ],
            ),
            ToolMessage(
                content="Rainy",
                tool_call_id="call123",
                name="get_weather",
            ),
        ],
        tools=[
            Tool(
                type="function",
                function=Function(
                    name="get_weather",
                    description="Gets the current weather in a city.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "The city name"
                            }
                        },
                        "required": ["city"],
                    },
                ),
            )
        ],
    ),
)])
def test_make_mistral_chat_completion_request_list_content(
        openai_request, expected_mistral_request):
    actual_request = make_mistral_chat_completion_request(
        openai_request["messages"], openai_request["tools"])
    assert actual_request == expected_mistral_request
