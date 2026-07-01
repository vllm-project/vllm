# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the engine-based seed_oss parser.

seed_oss is Qwen3 with four overridden wrapper tokens, so the shared grammar
(arg types, multiline values, parallel calls, streaming mechanics, …) is
already covered by ``test_qwen3.py``/``test_qwen3_reasoning.py``. These tests
cover only what is seed_oss-specific: that the ``seed:`` token overrides are
wired through, the reasoning→tool boundary holds with them, the malformed
header from #46314 no longer drops sibling calls, and the registered adapters
resolve. Seed-specific budget-reflect tags inside reasoning are also covered
here because the old dedicated parser tests exercised them.
"""

import json

import pytest

from tests.parser.engine.conftest import make_mock_tokenizer
from tests.parser.engine.streaming_helpers import (
    collect_function_name,
    collect_tool_arguments,
    simulate_reasoning_streaming,
    simulate_tool_streaming,
)
from vllm.parser.engine.registered_adapters import (
    SeedOssParserReasoningAdapter,
    SeedOssParserToolAdapter,
)
from vllm.parser.seed_oss import SeedOssParser

TOOL_CALL_START = "<seed:tool_call>"
TOOL_CALL_END = "</seed:tool_call>"
THINK_START = "<seed:think>"
THINK_END = "</seed:think>"

_THINK_END_ID = 51
_TOOL_CALL_ID = 60

_SEED_OSS_VOCAB = {
    THINK_START: 50,
    THINK_END: _THINK_END_ID,
    TOOL_CALL_START: _TOOL_CALL_ID,
    TOOL_CALL_END: 61,
}


@pytest.fixture
def mock_tokenizer():
    return make_mock_tokenizer(_SEED_OSS_VOCAB)


@pytest.fixture
def tool_parser(mock_tokenizer):
    return SeedOssParser(
        mock_tokenizer, chat_template_kwargs={"enable_thinking": False}
    )


@pytest.fixture
def parser(mock_tokenizer):
    return SeedOssParser(mock_tokenizer)


def test_token_overrides_wired(parser):
    assert parser.parser_engine_config.name == "seed_oss"
    assert parser.reasoning_start_str == THINK_START
    assert parser.reasoning_end_str == THINK_END


def test_single_tool_call(tool_parser, mock_request):
    text = (
        f"{TOOL_CALL_START}\n<function=get_weather>\n"
        "<parameter=city>Tokyo</parameter>\n"
        f"</function>\n{TOOL_CALL_END}"
    )
    result = tool_parser.extract_tool_calls(text, mock_request)

    assert result.tools_called is True
    assert result.tool_calls[0].function.name == "get_weather"
    assert json.loads(result.tool_calls[0].function.arguments) == {"city": "Tokyo"}


def test_malformed_function_end_does_not_drop_siblings(tool_parser, mock_request):
    """Regression for #46314: a malformed ``</function>`` with no closing ``>``
    on the header must not discard the other, well-formed calls."""
    text = (
        f"{TOOL_CALL_START}\n<function=broken</function>\n{TOOL_CALL_END}"
        f"{TOOL_CALL_START}\n<function=get_weather>\n"
        "<parameter=city>Tokyo</parameter>\n"
        f"</function>\n{TOOL_CALL_END}"
    )
    result = tool_parser.extract_tool_calls(text, mock_request)

    weather = next(tc for tc in result.tool_calls if tc.function.name == "get_weather")
    assert json.loads(weather.function.arguments) == {"city": "Tokyo"}


def test_basic_streaming(tool_parser, mock_request):
    chunks = [
        f"{TOOL_CALL_START}\n",
        "<function=get_weather>\n",
        "<parameter=city>Tokyo",
        "</parameter>\n",
        "</function>\n",
        f"{TOOL_CALL_END}",
    ]
    results = simulate_tool_streaming(tool_parser, mock_request, chunks)

    assert collect_function_name(results) == "get_weather"
    assert json.loads(collect_tool_arguments(results)) == {"city": "Tokyo"}


def test_reasoning_then_tool_call(parser):
    text = (
        f"{THINK_START}I need to read the file.{THINK_END}"
        f"{TOOL_CALL_START}\n<function=read>\n"
        "<parameter=path>/tmp/x</parameter>\n"
        f"</function>\n{TOOL_CALL_END}"
    )
    reasoning, _ = parser.extract_reasoning(text, None)
    assert reasoning == "I need to read the file."
    assert TOOL_CALL_START not in reasoning


def test_streaming_think_end_and_tool_call_same_delta(parser):
    """``</seed:think>`` and ``<seed:tool_call>`` arriving in one delta must
    not leak the terminal tokens into the reasoning text."""
    reasoning, content = simulate_reasoning_streaming(
        parser,
        [
            "Let me list the directory.",
            f"{THINK_END}{TOOL_CALL_START}",
            "<function=read>",
        ],
        [(1,), (_THINK_END_ID, _TOOL_CALL_ID), (2,)],
    )
    assert reasoning == "Let me list the directory."
    assert THINK_END not in reasoning
    assert TOOL_CALL_START not in reasoning
    assert content is not None


def test_end_to_end_through_registered_adapters(mock_tokenizer, mock_request):
    reasoning_parser = SeedOssParserReasoningAdapter(mock_tokenizer)
    tool_parser = SeedOssParserToolAdapter(mock_tokenizer)
    text = (
        f"{THINK_START}Plan the call.{THINK_END}"
        f"{TOOL_CALL_START}\n<function=get_weather>\n"
        "<parameter=city>Tokyo</parameter>\n"
        f"</function>\n{TOOL_CALL_END}"
    )
    reasoning, remaining = reasoning_parser.extract_reasoning(text, mock_request)
    assert reasoning == "Plan the call."

    tool_result = tool_parser.extract_tool_calls(remaining, mock_request)
    assert tool_result.tool_calls[0].function.name == "get_weather"
    assert json.loads(tool_result.tool_calls[0].function.arguments) == {"city": "Tokyo"}


def test_budget_reflect_tags_do_not_break_adapter_pipeline(
    mock_tokenizer,
    mock_request,
):
    reasoning_parser = SeedOssParserReasoningAdapter(mock_tokenizer)
    tool_parser = SeedOssParserToolAdapter(mock_tokenizer)
    text = (
        f"{THINK_START}"
        "The user's current thinking budget is 512.</seed:cot_budget_reflect>\n"
        "I need the weather.\n"
        "<seed:cot_budget_reflect>I have used 131 tokens."
        "</seed:cot_budget_reflect>\n"
        f"{THINK_END}"
        f"{TOOL_CALL_START}\n<function=get_weather>\n"
        "<parameter=city>Barcelona</parameter>\n"
        f"</function>\n{TOOL_CALL_END}"
    )

    reasoning, remaining = reasoning_parser.extract_reasoning(text, mock_request)
    assert reasoning is not None
    assert "current thinking budget is 512" in reasoning
    assert "<seed:cot_budget_reflect>" in reasoning
    assert "</seed:cot_budget_reflect>" in reasoning

    tool_result = tool_parser.extract_tool_calls(remaining, mock_request)
    assert tool_result.tool_calls[0].function.name == "get_weather"
    assert json.loads(tool_result.tool_calls[0].function.arguments) == {
        "city": "Barcelona"
    }
