# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.serving import OpenAIServing
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.parser.abstract_parser import DelegatingParser

pytestmark = pytest.mark.skip_global_cleanup


class _DummyDelegatingParser(DelegatingParser):
    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        return False

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        return input_ids

    def extract_reasoning(self, model_output: str, request):
        return None, model_output

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: list[int],
        current_token_ids: list[int],
        delta_token_ids: list[int],
    ):
        return None

    def extract_tool_calls(self, model_output: str, request):
        return None


def test_parse_tool_calls_from_content_allows_named_tool_choice_with_none_content():
    request = ChatCompletionRequest.model_validate(
        {
            "model": "test-model",
            "messages": [{"role": "user", "content": "test"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
        }
    )

    tool_calls, content = OpenAIServing._parse_tool_calls_from_content(
        request=request,
        tokenizer=None,
        enable_auto_tools=True,
        tool_parser_cls=None,
        content=None,
    )

    assert content is None
    assert tool_calls is not None
    assert tool_calls == []


def test_responses_parser_allows_named_tool_choice_with_none_content():
    request = ResponsesRequest.model_validate(
        {
            "model": "test-model",
            "input": "test",
            "tools": [
                {
                    "type": "function",
                    "name": "get_weather",
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
            "tool_choice": {"type": "function", "name": "get_weather"},
        }
    )
    parser = _DummyDelegatingParser(tokenizer=None)

    tool_calls, content = parser._parse_tool_calls(
        request=request,
        content=None,
        enable_auto_tools=False,
    )

    assert content is None
    assert tool_calls == []
