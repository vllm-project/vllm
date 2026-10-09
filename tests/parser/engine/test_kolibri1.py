# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the Kolibri 1 parser.

Kolibri 1 uses the Qwen3 reasoning grammar, which ``test_qwen3_reasoning.py``
covers, and Hermes tool calls. These tests cover the starting state, which
follows the Kolibri 1 chat template (``reasoning_effort`` takes precedence over
``enable_thinking``), and the reasoning adapter paired with the Hermes tool
parser the way the serving layer pairs them.
"""

import json

import pytest

from tests.parser.engine.conftest import make_mock_tokenizer
from vllm.parser.abstract_parser import DelegatingParser
from vllm.parser.kolibri1 import thinking_enabled
from vllm.reasoning import ReasoningParserManager
from vllm.tool_parsers import ToolParserManager

VOCAB = {"<think>": 50, "</think>": 51, "<tool_call>": 60, "</tool_call>": 61}

THINKING_OFF_KWARGS = [
    {"enable_thinking": False},
    {"reasoning_effort": None, "enable_thinking": False},
    {"reasoning_effort": "none"},
    {"reasoning_effort": "none", "enable_thinking": True},
]

THINKING_ON_KWARGS = [
    {},
    {"reasoning_effort": None},
    {"enable_thinking": True},
    {"enable_thinking": None},
    {"reasoning_effort": "high"},
    {"reasoning_effort": "low", "enable_thinking": False},
]


def _kwargs_id(kwargs: dict) -> str:
    return ",".join(f"{k}={v}" for k, v in kwargs.items()) or "default"


@pytest.fixture
def mock_tokenizer():
    return make_mock_tokenizer(VOCAB)


def _make_adapter(tokenizer, **chat_template_kwargs):
    adapter_cls = ReasoningParserManager.get_reasoning_parser("kolibri1")
    return adapter_cls(tokenizer, chat_template_kwargs=chat_template_kwargs)


def _make_parser(tokenizer, **chat_template_kwargs) -> DelegatingParser:
    class _Kolibri1Parser(DelegatingParser):
        reasoning_parser_cls = ReasoningParserManager.get_reasoning_parser("kolibri1")
        tool_parser_cls = ToolParserManager.get_tool_parser("kolibri1")

    return _Kolibri1Parser(tokenizer, chat_template_kwargs=chat_template_kwargs)


@pytest.mark.parametrize("kwargs", THINKING_OFF_KWARGS, ids=_kwargs_id)
def test_thinking_off_kwargs(kwargs):
    assert not thinking_enabled(kwargs)


@pytest.mark.parametrize("kwargs", THINKING_ON_KWARGS, ids=_kwargs_id)
def test_thinking_on_kwargs(kwargs):
    assert thinking_enabled(kwargs)


def test_thinking_off_non_streaming_is_content(mock_tokenizer, mock_request):
    """Non-streaming never sees the prompt, so only the kwargs tell the parser
    that the template already closed the think block. Qwen3Parser reads
    enable_thinking alone and would file the answer as reasoning."""
    adapter = _make_adapter(
        mock_tokenizer, reasoning_effort="none", enable_thinking=True
    )
    assert adapter.extract_reasoning("The answer.", mock_request) == (
        None,
        "The answer.",
    )


def test_thinking_on_non_streaming_splits_reasoning(mock_tokenizer, mock_request):
    adapter = _make_adapter(
        mock_tokenizer, reasoning_effort="low", enable_thinking=False
    )
    reasoning, content = adapter.extract_reasoning(
        "<think>\nlet me see\n</think>\n\nThe answer.", mock_request
    )
    assert (reasoning.strip(), content.strip()) == ("let me see", "The answer.")


def test_reasoning_then_hermes_tool_call(mock_tokenizer, mock_request):
    """Kolibri 1 emits Hermes JSON tool calls, not Qwen3 XML ones."""
    reasoning, content, tool_calls = _make_parser(mock_tokenizer).parse(
        "<think>\nI should look it up.\n</think>\n\n"
        '<tool_call>\n{"name": "lookup", "arguments": {"city": "Paris"}}\n'
        "</tool_call>",
        mock_request,
        enable_auto_tools=True,
    )
    assert reasoning.strip() == "I should look it up."
    assert not (content or "").strip()
    assert [(tc.name, json.loads(tc.arguments)) for tc in tool_calls] == [
        ("lookup", {"city": "Paris"})
    ]
