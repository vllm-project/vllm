# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.sampling_params import StructuredOutputsParams

pytestmark = pytest.mark.skip_global_cleanup


def test_chat_completion_request_with_no_tools():
    # tools key is not present
    request = ChatCompletionRequest.model_validate(
        {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "facebook/opt-125m",
        }
    )
    assert request.tool_choice == "none"

    # tools key is None
    request = ChatCompletionRequest.model_validate(
        {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "facebook/opt-125m",
            "tools": None,
        }
    )
    assert request.tool_choice == "none"

    # tools key present but empty -- should be rejected
    with pytest.raises(ValueError, match="must not be an empty array"):
        ChatCompletionRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "Hello"}],
                "model": "facebook/opt-125m",
                "tools": [],
            }
        )


@pytest.mark.parametrize("tool_choice", ["auto", "required"])
def test_chat_completion_request_with_tool_choice_but_no_tools(tool_choice):
    with pytest.raises(
        ValueError, match="When using `tool_choice`, `tools` must be set."
    ):
        ChatCompletionRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "Hello"}],
                "model": "facebook/opt-125m",
                "tool_choice": tool_choice,
            }
        )

    with pytest.raises(
        ValueError, match="When using `tool_choice`, `tools` must be set."
    ):
        ChatCompletionRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "Hello"}],
                "model": "facebook/opt-125m",
                "tool_choice": tool_choice,
                "tools": None,
            }
        )


def test_chat_completion_request_materializes_tool_call_generators():
    def tool_call_iter():
        yield {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"city": "Paris"}',
            },
        }

    request = ChatCompletionRequest.model_validate(
        {
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": tool_call_iter(),
                }
            ],
            "model": "facebook/opt-125m",
        }
    )

    tool_calls = request.messages[0]["tool_calls"]
    assert isinstance(tool_calls, list)
    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "get_weather"


def test_chat_completion_request_caches_tool_dicts():
    request = ChatCompletionRequest.model_validate(
        {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "facebook/opt-125m",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "city": {"type": "string"},
                            },
                            "required": ["city"],
                        },
                    },
                }
            ],
        }
    )

    tool_dicts = request.get_tool_dicts()
    assert tool_dicts is not None
    assert request.get_tool_dicts() is tool_dicts
    assert tool_dicts[0]["function"]["name"] == "get_weather"


def test_chat_completion_request_reuses_resolved_structured_outputs():
    request = ChatCompletionRequest.model_validate(
        {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "facebook/opt-125m",
            "response_format": {"type": "json_object"},
        }
    )

    first_sampling_params = request.to_sampling_params(
        max_tokens=10,
        default_sampling_params={},
    )
    first_structured_outputs = request.structured_outputs

    assert isinstance(first_structured_outputs, StructuredOutputsParams)
    assert first_sampling_params.structured_outputs is first_structured_outputs

    second_sampling_params = request.to_sampling_params(
        max_tokens=10,
        default_sampling_params={},
    )

    assert request.structured_outputs is first_structured_outputs
    assert second_sampling_params.structured_outputs is first_structured_outputs
