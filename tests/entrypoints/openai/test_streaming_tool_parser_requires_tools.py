# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The streaming tool parser must stay out of the way when a request has no tools.

A server started with --tool-call-parser builds a parser for every request,
so the parser's existence says nothing about whether a given request offered
tools. Without an explicit check, the streaming path treats a plain
`response_format` answer as a candidate tool call: json-style parsers claim
any generation beginning with "{", which structured output guarantees, and
the caller receives a tool call it never asked for and no content.

The non-streaming path already guards this in the serving layer, via the
`not request.tool_choice or request.tool_choice == "none"` branch.
"""

import pytest

from vllm.parser.abstract_parser import DelegatingParser, StreamState
from vllm.tool_parsers.abstract_tool_parser import ToolParser

pytestmark = pytest.mark.skip_global_cleanup


class _DummyToolParser(ToolParser):
    def __init__(self, tokenizer=None, tools=None):
        pass


class _DummyDelegatingParser(DelegatingParser):
    tool_parser_cls = _DummyToolParser

    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        return True

    def extract_reasoning(self, model_output: str, request):
        return None, model_output

    def extract_reasoning_streaming(self, *args, **kwargs):
        return None


def _phase(tools) -> bool:
    parser = _DummyDelegatingParser(tokenizer=None, tools=tools)
    state = StreamState(tool_call_id_type="random", engine_based=True)
    state.reasoning_ended = True
    return parser._in_tool_call_phase(state)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {"type": "object", "properties": {}},
        },
    }
]


@pytest.mark.parametrize("tools", [None, []])
def test_no_tools_means_no_tool_call_phase(tools):
    """A response_format-only request must never enter the tool-call phase."""
    assert _phase(tools) is False


def test_tools_present_still_enter_tool_call_phase():
    """The guard must not disable tool calling for requests that do use tools."""
    assert _phase(TOOLS) is True
