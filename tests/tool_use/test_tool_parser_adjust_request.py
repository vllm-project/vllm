# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request-level wiring for composing an auto tool-call grammar with a
response_format schema (https://github.com/vllm-project/vllm/issues/39929).

``DelegatingParser._apply_structural_tag`` decides whether a tool_choice="auto"
request that also sets a response_format gets its schema composed with the
tool-call grammar, via ``get_composed_structural_tag``. These tests drive that
decision through the real parser registry (``ParserManager.get_parser``), not
through the composition helpers directly, to prove the wiring itself.
"""

from unittest.mock import MagicMock

import pytest
import xgrammar as xgr
from openai.types.responses import (
    FunctionTool,
    ResponseFormatTextJSONSchemaConfig,
    ResponseTextConfig,
)
from xgrammar.testing import _is_grammar_accept_string

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.parser.parser_manager import ParserManager

pytestmark = pytest.mark.cpu_test

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}

NON_STRICT_WEATHER_TOOL = ChatCompletionToolsParam(
    type="function",
    function={
        "name": "get_weather",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
)
NON_STRICT_WEATHER_RESPONSES_TOOL = FunctionTool(
    type="function",
    name="get_weather",
    parameters={
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
)

HERMES_TOOL_CALL = (
    '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Dallas"}}\n'
    "</tool_call>"
)
SCHEMA_VALID_JSON = '{"answer": "sunny in Dallas"}'
FREE_TEXT = "The weather is sunny."


def _parser(tool_parser_name: str):
    parser_cls = ParserManager.get_parser(
        tool_parser_name=tool_parser_name, enable_auto_tools=True
    )
    assert parser_cls is not None
    return parser_cls(MagicMock(), tools=[NON_STRICT_WEATHER_TOOL])


def _chat_request(**overrides) -> ChatCompletionRequest:
    fields = dict(
        messages=[{"role": "user", "content": "What is the weather in Dallas?"}],
        model="m",
        tools=[NON_STRICT_WEATHER_TOOL],
        tool_choice="auto",
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "final_answer", "schema": RESPONSE_SCHEMA},
        },
    )
    fields.update(overrides)
    return ChatCompletionRequest(**fields)


def test_composable_model_composes_tag_and_clears_response_format():
    """hermes is a composable model: an auto tool_choice request that also
    sets response_format should compose the two into a single structural_tag
    instead of dropping the schema."""
    parser = _parser("hermes")
    request = _chat_request()

    out = parser.adjust_request(request)

    assert out.response_format is None
    assert out.structured_outputs is not None
    assert out.structured_outputs.structural_tag is not None

    grammar = xgr.Grammar.from_structural_tag(out.structured_outputs.structural_tag)
    assert _is_grammar_accept_string(grammar, HERMES_TOOL_CALL)
    assert _is_grammar_accept_string(grammar, SCHEMA_VALID_JSON)
    assert not _is_grammar_accept_string(grammar, FREE_TEXT)


def test_composable_model_composes_responses_request_text_format():
    """The Responses API expresses response_format as text.format rather than
    response_format, but the composition applies there too."""
    parser = _parser("hermes")
    request = ResponsesRequest(
        input="What is the weather in Dallas?",
        model="m",
        tools=[NON_STRICT_WEATHER_RESPONSES_TOOL],
        tool_choice="auto",
        text=ResponseTextConfig(
            format=ResponseFormatTextJSONSchemaConfig(
                type="json_schema",
                name="final_answer",
                schema=RESPONSE_SCHEMA,
            )
        ),
    )

    out = parser.adjust_request(request)

    assert out.text is None
    assert out.structured_outputs is not None
    assert out.structured_outputs.structural_tag is not None

    grammar = xgr.Grammar.from_structural_tag(out.structured_outputs.structural_tag)
    assert _is_grammar_accept_string(grammar, HERMES_TOOL_CALL)
    assert _is_grammar_accept_string(grammar, SCHEMA_VALID_JSON)
    assert not _is_grammar_accept_string(grammar, FREE_TEXT)


def test_non_composable_model_leaves_response_format_intact():
    """qwen3_coder only has an xgrammar builtin template, not a vLLM-owned
    registry builder, so it is out of scope for this PR and falls back
    unchanged rather than dropping the schema or crashing."""
    parser = _parser("qwen3_coder")
    request = _chat_request()

    out = parser.adjust_request(request)

    assert out.response_format is not None
    assert out.structured_outputs is None


def test_reasoning_active_falls_back_to_unchanged_behavior():
    """A think preamble would not match the schema answer branch, so an
    active reasoning parser must skip composition entirely."""
    parser = _parser("hermes")
    parser.reasoning_parser = MagicMock(adjust_request=lambda request: request)
    request = _chat_request()

    out = parser.adjust_request(request)

    assert out.response_format is not None
    assert out.structured_outputs is None
