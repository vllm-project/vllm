# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Named tool choice for Gemma 4 must be rejected: the parser cannot force a
specific function without JSON guided decoding, which conflicts with native
``<|tool_call>`` syntax.
"""

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.exceptions import VLLMValidationError
from vllm.tool_parsers.gemma4_engine_tool_parser import Gemma4EngineToolParser


class _DummyTokenizer:
    _VOCAB = {
        "<|tool_call>": 256_000,
        "<tool_call|>": 256_001,
        "<|channel>": 256_002,
        "<channel|>": 256_003,
    }

    def get_vocab(self):
        return dict(self._VOCAB)

    @property
    def all_special_tokens(self) -> list[str]:
        return list(self._VOCAB.keys())

    @property
    def all_special_ids(self) -> list[int]:
        return list(self._VOCAB.values())


def _named_request() -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="gemma4",
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


def _required_request() -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="gemma4",
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
        tool_choice="required",
    )


def test_named_choice_rejected():
    parser = Gemma4EngineToolParser(_DummyTokenizer())
    with pytest.raises(VLLMValidationError, match="Named tool choice") as exc_info:
        parser.adjust_request(_named_request())
    assert exc_info.value.parameter == "tool_choice"


def test_required_choice_still_skips_structured_outputs():
    parser = Gemma4EngineToolParser(_DummyTokenizer())
    req = parser.adjust_request(_required_request())
    assert req.structured_outputs is None
    assert req.skip_special_tokens is False
