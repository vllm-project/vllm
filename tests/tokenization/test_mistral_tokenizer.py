# SPDX-License-Identifier: Apache-2.0

import pytest
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.protocol.instruct.tool_calls import Function, Tool

from vllm.transformers_utils.tokenizers.mistral import (
    make_mistral_chat_completion_request)


# yapf: enable
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
    )],
)
def test_make_mistral_chat_completion_request(openai_request,
                                              expected_mistral_request):
    assert (make_mistral_chat_completion_request(
        openai_request["messages"],
        openai_request["tools"]) == expected_mistral_request)
