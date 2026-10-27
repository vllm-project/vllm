# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Named tool choice for Kimi K3: allowed when the XTML structural tag is
attached (strict tool calling), rejected otherwise."""

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import StructuredOutputsParams


class _DummyTokenizer:
    def get_vocab(self):
        return {}

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


def _request(with_tag: bool) -> ChatCompletionRequest:
    req = ChatCompletionRequest(
        model="k3",
        messages=[{"role": "user", "content": "hi"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ],
        tool_choice={"type": "function", "function": {"name": "get_weather"}},
    )
    if with_tag:
        req.structured_outputs = StructuredOutputsParams(
            structural_tag='{"type": "structural_tag", "format": {}}'
        )
    return req


def _parser():
    from vllm.tool_parsers.kimi_k3_tool_parser import KimiK3ToolParser

    return KimiK3ToolParser(_DummyTokenizer())


def test_named_choice_allowed_with_structural_tag():
    req = _parser().adjust_request(_request(with_tag=True))
    assert req.skip_special_tokens is False


def test_named_choice_rejected_without_structural_tag():
    with pytest.raises(VLLMValidationError):
        _parser().adjust_request(_request(with_tag=False))


def _custom_request(with_tag: bool) -> ResponsesRequest:
    req = ResponsesRequest(
        input="hi",
        tools=[{"type": "custom", "name": "emit_command"}],
        tool_choice={"type": "custom", "name": "emit_command"},
    )
    if with_tag:
        req.structured_outputs = StructuredOutputsParams(
            structural_tag='{"type": "structural_tag", "format": {}}'
        )
    return req


def test_named_custom_choice_allowed_with_structural_tag():
    req = _parser().adjust_request(_custom_request(with_tag=True))
    assert req.skip_special_tokens is False


def test_named_custom_choice_rejected_without_structural_tag():
    with pytest.raises(VLLMValidationError):
        _parser().adjust_request(_custom_request(with_tag=False))
