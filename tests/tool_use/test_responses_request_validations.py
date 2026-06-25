# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from pydantic import ValidationError

from vllm.entrypoints.openai.responses.protocol import ResponsesRequest

SAMPLE_TOOL = {
    "type": "function",
    "name": "get_weather",
    "description": "Get current weather",
    "parameters": {
        "type": "object",
        "properties": {"location": {"type": "string", "description": "City name"}},
        "required": ["location"],
    },
}

NAMED_TOOL_CHOICE = {
    "type": "function",
    "name": "get_weather",
}


def test_responses_request_with_no_tools():
    # tools key is not present — defaults tool_choice to "none"
    request = ResponsesRequest.model_validate({"input": "Hello", "model": "test-model"})
    assert request.tool_choice == "none"

    # tools key present but empty
    request = ResponsesRequest.model_validate(
        {"input": "Hello", "model": "test-model", "tools": []}
    )
    assert request.tool_choice == "none"


def test_responses_request_no_tools_tool_choice_none():
    request = ResponsesRequest.model_validate(
        {"input": "Hello", "model": "test-model", "tool_choice": "none"}
    )
    assert request.tool_choice == "none"


def test_responses_request_no_tools_tool_choice_auto():
    request = ResponsesRequest.model_validate(
        {"input": "Hello", "model": "test-model", "tool_choice": "auto"}
    )
    assert request.tool_choice == "none"


@pytest.mark.parametrize("tools", [None, []])
def test_responses_request_required_without_tools(tools):
    kwargs = {"input": "Hello", "model": "test-model", "tool_choice": "required"}
    if tools is not None:
        kwargs["tools"] = tools
    with pytest.raises(
        ValidationError, match="Tool choice 'required' must be specified"
    ):
        ResponsesRequest.model_validate(kwargs)


def test_responses_request_named_tool_choice_without_tools():
    with pytest.raises(ValidationError, match="not found in 'tools' parameter"):
        ResponsesRequest.model_validate(
            {
                "input": "Hello",
                "model": "test-model",
                "tool_choice": NAMED_TOOL_CHOICE,
            }
        )


def test_responses_request_with_tools_default_tool_choice():
    request = ResponsesRequest.model_validate(
        {"input": "Hello", "model": "test-model", "tools": [SAMPLE_TOOL]}
    )
    assert request.tool_choice == "auto"


def test_responses_request_with_tools_tool_choice_none():
    request = ResponsesRequest.model_validate(
        {
            "input": "Hello",
            "model": "test-model",
            "tools": [SAMPLE_TOOL],
            "tool_choice": "none",
        }
    )
    assert request.tool_choice == "none"


def test_responses_request_named_tool_choice_matching():
    request = ResponsesRequest.model_validate(
        {
            "input": "Hello",
            "model": "test-model",
            "tools": [SAMPLE_TOOL],
            "tool_choice": NAMED_TOOL_CHOICE,
        }
    )
    assert request.tool_choice.type == "function"
    assert request.tool_choice.name == "get_weather"


def test_responses_request_named_tool_choice_not_matching():
    with pytest.raises(ValidationError, match="not found in 'tools' parameter"):
        ResponsesRequest.model_validate(
            {
                "input": "Hello",
                "model": "test-model",
                "tools": [SAMPLE_TOOL],
                "tool_choice": {"type": "function", "name": "nonexistent"},
            }
        )


def test_responses_request_with_tools_tool_choice_auto():
    request = ResponsesRequest.model_validate(
        {
            "input": "Hello",
            "model": "test-model",
            "tools": [SAMPLE_TOOL],
            "tool_choice": "auto",
        }
    )
    assert request.tool_choice == "auto"


def test_responses_request_with_tools_tool_choice_required():
    request = ResponsesRequest.model_validate(
        {
            "input": "Hello",
            "model": "test-model",
            "tools": [SAMPLE_TOOL],
            "tool_choice": "required",
        }
    )
    assert request.tool_choice == "required"


def test_responses_request_empty_tools_tool_choice_none():
    request = ResponsesRequest.model_validate(
        {"input": "Hello", "model": "test-model", "tools": [], "tool_choice": "none"}
    )
    assert request.tool_choice == "none"


def test_responses_request_empty_tools_tool_choice_auto():
    request = ResponsesRequest.model_validate(
        {"input": "Hello", "model": "test-model", "tools": [], "tool_choice": "auto"}
    )
    assert request.tool_choice == "none"


@pytest.mark.parametrize(
    "tool_choice",
    [
        {"type": "function"},
        {"type": "function", "name": ""},
    ],
)
def test_responses_request_named_tool_choice_missing_name(tool_choice):
    with pytest.raises(ValidationError, match="not found in 'tools' parameter"):
        ResponsesRequest.model_validate(
            {
                "input": "Hello",
                "model": "test-model",
                "tools": [SAMPLE_TOOL],
                "tool_choice": tool_choice,
            }
        )


def test_responses_request_empty_tools_named_tool_choice():
    with pytest.raises(ValidationError, match="not found in 'tools' parameter"):
        ResponsesRequest.model_validate(
            {
                "input": "Hello",
                "model": "test-model",
                "tools": [],
                "tool_choice": NAMED_TOOL_CHOICE,
            }
        )


# https://github.com/vllm-project/vllm/issues/46631
IMAGE_URL = "https://example.com/image.png"


def _input_image_message(image_url):
    return {
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "What is this?"},
                    {"type": "input_image", "image_url": image_url, "detail": "auto"},
                ],
            }
        ],
        "model": "test-model",
    }


def _image_url_of(request):
    # The validated input_image content part is a TypedDict (plain dict).
    part = request.input[0]["content"][1]
    return part["image_url"]


def test_responses_request_input_image_flat_url():
    # The Responses-native flat string form is accepted unchanged.
    request = ResponsesRequest.model_validate(_input_image_message(IMAGE_URL))
    assert _image_url_of(request) == IMAGE_URL


def test_responses_request_input_image_nested_url_coerced():
    # chat-completions-style nested {"url": ...} is coerced to the flat string
    # instead of failing strict validation with an opaque error (#46631).
    request = ResponsesRequest.model_validate(_input_image_message({"url": IMAGE_URL}))
    assert _image_url_of(request) == IMAGE_URL
