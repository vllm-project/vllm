# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Duplicate tool names are rejected, as in the Rust frontend (#68579)."""

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.exceptions import VLLMValidationError


def _tool(name: str, prop: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "parameters": {"type": "object", "properties": {prop: {"type": "string"}}},
        },
    }


def _request(tools: list[dict], **kwargs) -> ChatCompletionRequest:
    return ChatCompletionRequest.model_validate(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": tools,
            **kwargs,
        }
    )


@pytest.mark.parametrize("tool_choice", [None, "auto", "none", "required"])
def test_duplicate_tool_names_are_rejected(tool_choice):
    tools = [_tool("lookup", "city"), _tool("other", "x"), _tool("lookup", "id")]
    kwargs = {} if tool_choice is None else {"tool_choice": tool_choice}
    with pytest.raises(VLLMValidationError, match="Duplicate tool name `lookup`"):
        _request(tools, **kwargs)


def test_unique_tool_names_are_accepted():
    request = _request([_tool("lookup", "city"), _tool("other", "x")])
    assert [t.function.name for t in request.tools] == ["lookup", "other"]
