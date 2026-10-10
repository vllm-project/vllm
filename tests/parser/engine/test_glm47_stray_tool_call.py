# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for issue #60703.

A stray <tool_call> in prose used to swallow the stream.
The fix adds a (TOOL_NAME, TOOL_START) transition.
"""

import pytest

from tests.parser.engine.conftest import make_mock_tokenizer
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest, ChatCompletionToolsParam, FunctionDefinition)
from vllm.parser.glm47_moe import (
    THINK_END, THINK_START, TOOL_CALL_END, TOOL_CALL_START,
    Glm47MoeParser, glm47_moe_config)
from vllm.parser.engine.parser_engine_config import ParserState

TOOL_S, TOOL_E = 1002, 1003
VOCAB = {
    THINK_START: 1000, THINK_END: 1001,
    TOOL_CALL_START: TOOL_S, TOOL_CALL_END: TOOL_E,
    "<tool_call>": TOOL_S, "</tool_call>": TOOL_E,
    "<arg_key>": 1010, "</arg_key>": 1011,
    "<arg_value>": 1012, "</arg_value>": 1013}

TOOLS = [ChatCompletionToolsParam(
    function=FunctionDefinition(name="get_weather", parameters={}))]

@pytest.fixture
def parser():
    return Glm47MoeParser(make_mock_tokenizer(VOCAB))

@pytest.fixture
def mock_request():
    return ChatCompletionRequest(model="test",
        messages=[{"role": "user", "content": "hi"}],
        tools=TOOLS)

class TestStrayToolCallTransition:
    """Verify (TOOL_NAME, TOOL_START) transition exists."""

    def test_transition_exists(self):
        cfg = glm47_moe_config(thinking=True)
        key = (ParserState.TOOL_NAME, "TOOL_START")
        assert key in cfg.transitions
        t = cfg.transitions[key]
        assert t.next_state == ParserState.TOOL_NAME

class TestStrayToolCallRecovery:
    def test_stray_then_real_call(self, parser, mock_request):
        """Stray <tool_call> then real call: tool must be recovered."""
        txt = "Fetching. I will issue a <tool_call>" + "get_weather</tool_call>"
        r = parser.extract_tool_calls(txt, mock_request)
        assert r.tools_called
        assert len(r.tool_calls) == 1
        assert r.tool_calls[0].function.name == "get_weather"

    def test_two_stray_then_real(self, parser, mock_request):
        """Two stray <tool_call> then real: both phantoms dropped."""
        txt = "text <tool_call><tool_call>get_weather</tool_call>"
        r = parser.extract_tool_calls(txt, mock_request)
        assert r.tools_called
        assert len(r.tool_calls) == 1
        assert r.tool_calls[0].function.name == "get_weather"

    def test_lone_stray_tag(self, parser, mock_request):
        """Lone stray <tool_call> no real call: no tools_called."""
        txt = "text <tool_call>"
        r = parser.extract_tool_calls(txt, mock_request)
        assert not r.tools_called

    def test_clean_call_unchanged(self, parser, mock_request):
        """Clean call: no regression."""
        txt = "<tool_call>get_weather</tool_call>"
        r = parser.extract_tool_calls(txt, mock_request)
        assert r.tools_called
        assert r.tool_calls[0].function.name == "get_weather"

    def test_stray_with_args(self, parser, mock_request):
        """Stray <tool_call> then real call with args."""
        txt = "prose <tool_call><tool_call>get_weather<arg_key>city</arg_key><arg_value>BJ</arg_value></tool_call>"
        r = parser.extract_tool_calls(txt, mock_request)
        assert r.tools_called
        assert len(r.tool_calls) == 1
        assert r.tool_calls[0].function.name == "get_weather"
        assert "city" in r.tool_calls[0].function.arguments
