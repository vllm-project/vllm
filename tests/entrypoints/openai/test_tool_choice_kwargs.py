# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for tool_choice forwarding in build_chat_params.

An absent tool_choice renders nothing in the prompt; only an explicitly
provided one reaches the renderer. The request field's default "none" is
serving-layer state and must not be rendered.
"""

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)

_TOOL = {
    "type": "function",
    "function": {"name": "f", "parameters": {"type": "object", "properties": {}}},
}


def _request(**kwargs) -> ChatCompletionRequest:
    defaults = {"model": "test-model", "messages": [{"role": "user", "content": "hi"}]}
    defaults.update(kwargs)
    return ChatCompletionRequest.model_validate(defaults)


def _message_level_tools_request(**kwargs) -> ChatCompletionRequest:
    messages = [
        {"role": "system", "content": "sys", "tools": [_TOOL]},
        {"role": "user", "content": "hi"},
    ]
    defaults = {"model": "test-model", "messages": messages}
    defaults.update(kwargs)
    return ChatCompletionRequest.model_validate(defaults)


def test_absent_tool_choice_not_forwarded():
    request = _request()
    params = request.build_chat_params(None, "auto")
    assert params.tool_choice is None


def test_explicit_none_tool_choice_forwarded():
    request = _request(tool_choice="none")
    params = request.build_chat_params(None, "auto")
    assert params.tool_choice == "none"


def test_explicit_null_tool_choice_not_forwarded():
    # Explicit null parses to None; the request must not resurrect the
    # field default either.
    request = _request(tool_choice=None)
    params = request.build_chat_params(None, "auto")
    assert params.tool_choice is None


def test_auto_default_from_tools_forwarded():
    request = _request(tools=[_TOOL])
    params = request.build_chat_params(None, "auto")
    assert params.tool_choice == "auto"


def test_user_chat_template_kwarg_kept():
    request = _request(chat_template_kwargs={"tool_choice": "none"})
    params = request.build_chat_params(None, "auto")
    assert params.chat_template_kwargs["tool_choice"] == "none"


def test_message_level_tools_default_forwarded():
    # The "auto" default triggered by message-level tools is forwarded, so
    # the renderer does not treat the request as tool-less.
    request = _message_level_tools_request()
    assert request.tool_choice == "auto"
    params = request.build_chat_params(None, "auto")
    assert params.tool_choice == "auto"


def test_message_level_tools_explicit_choice_forwarded():
    request = _message_level_tools_request(tool_choice="required")
    params = request.build_chat_params(None, "auto")
    assert params.tool_choice == "required"
