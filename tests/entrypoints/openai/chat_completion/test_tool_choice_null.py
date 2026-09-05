# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""An explicit `"tool_choice": null` has to mean the same as an absent key.

Clients that serialise their unset optional fields still send the key, and the
OpenAI API treats a null the same as an omission: `auto` when tools are present.
Left as a None it matches no branch downstream, so tool calls are never parsed
and a structured-outputs request is rejected as if it had asked for tools.
"""

import pytest
from pydantic import ValidationError

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

MESSAGES = [{"role": "user", "content": "what is the weather in Paris?"}]
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {"type": "object", "properties": {}},
        },
    }
]


def test_null_tool_choice_defaults_to_auto_like_an_absent_one():
    omitted = ChatCompletionRequest.model_validate(
        {"model": "m", "messages": MESSAGES, "tools": TOOLS}
    )
    explicit_null = ChatCompletionRequest.model_validate(
        {"model": "m", "messages": MESSAGES, "tools": TOOLS, "tool_choice": None}
    )

    assert omitted.tool_choice == "auto"
    assert explicit_null.tool_choice == omitted.tool_choice


def test_null_tool_choice_without_tools_matches_an_absent_one():
    omitted = ChatCompletionRequest.model_validate({"model": "m", "messages": MESSAGES})
    explicit_null = ChatCompletionRequest.model_validate(
        {"model": "m", "messages": MESSAGES, "tool_choice": None}
    )

    assert explicit_null.tool_choice == omitted.tool_choice


def test_null_tool_choice_does_not_conflict_with_structured_outputs():
    """The conflict check reads `tool_choice`, so a null was taken for a tool
    request and the caller was told it had asked for both."""
    request = ChatCompletionRequest.model_validate(
        {
            "model": "m",
            "messages": MESSAGES,
            "structured_outputs": {"json": {"type": "object"}},
            "tool_choice": None,
        }
    )

    assert request.structured_outputs is not None


@pytest.mark.parametrize(
    "tool_choice,tools",
    [
        ("bogus", TOOLS),
        ("auto", None),
    ],
)
def test_invalid_tool_choice_is_still_rejected(tool_choice, tools):
    payload = {"model": "m", "messages": MESSAGES, "tool_choice": tool_choice}
    if tools is not None:
        payload["tools"] = tools

    with pytest.raises(ValidationError):
        ChatCompletionRequest.model_validate(payload)


def test_explicit_none_tool_choice_is_untouched():
    request = ChatCompletionRequest.model_validate(
        {"model": "m", "messages": MESSAGES, "tools": TOOLS, "tool_choice": "none"}
    )

    assert request.tool_choice == "none"
