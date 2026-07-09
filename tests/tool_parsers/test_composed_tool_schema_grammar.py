# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Composed tool-call-or-schema grammar for tool_choice="auto" + response_format.

Proves that composing a model's auto tool-call structural tag with a
response_format schema (https://github.com/vllm-project/vllm/issues/39929)
yields a grammar that xgrammar can compile and that both calls a tool and
constrains a non-tool final answer to the schema.
"""

import json

import pytest
import xgrammar as xgr
from xgrammar import StructuralTag
from xgrammar.testing import _is_grammar_accept_string

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionToolsParam,
)
from vllm.tool_parsers.structural_tag_registry import (
    compose_tool_call_or_schema,
    get_composed_structural_tag,
    get_model_structural_tag,
)

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}

HERMES_TOOL_CALL = (
    '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Dallas"}}\n'
    "</tool_call>"
)
MINIMAX_TOOL_CALL = (
    "<minimax:tool_call>\n"
    '<invoke name="get_weather">\n<parameter name="city">Dallas</parameter>\n'
    "</invoke>\n</minimax:tool_call>"
)
SCHEMA_VALID_JSON = '{"answer": "sunny in Dallas"}'
SCHEMA_INVALID_KEY = '{"wrong": 1}'
SCHEMA_INVALID_TYPE = '{"answer": 5}'
FREE_TEXT = "The weather is sunny."
ANY_JSON = '{"anything": [1, 2, 3]}'


@pytest.fixture
def strict_tools() -> list[ChatCompletionToolsParam]:
    return [
        ChatCompletionToolsParam(
            type="function",
            function={
                "name": "get_weather",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        )
    ]


@pytest.fixture
def non_strict_tools() -> list[ChatCompletionToolsParam]:
    return [
        ChatCompletionToolsParam(
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
    ]


def _grammar(tag: StructuralTag) -> xgr.Grammar:
    return xgr.Grammar.from_structural_tag(json.dumps(tag.model_dump()))


def _auto_tag(model: str, tools: list[ChatCompletionToolsParam]) -> StructuralTag:
    tag = get_model_structural_tag(
        model=model,
        tools=tools,
        tool_choice="auto",
        reasoning=False,
    )
    assert isinstance(tag, StructuralTag)
    return tag


def test_compose_hermes_accepts_tool_call_and_schema(
    strict_tools: list[ChatCompletionToolsParam],
):
    composed = compose_tool_call_or_schema(
        _auto_tag("hermes", strict_tools), RESPONSE_SCHEMA
    )
    assert composed is not None
    grammar = _grammar(composed)

    assert _is_grammar_accept_string(grammar, HERMES_TOOL_CALL)
    assert _is_grammar_accept_string(grammar, SCHEMA_VALID_JSON)
    assert not _is_grammar_accept_string(grammar, SCHEMA_INVALID_KEY)
    assert not _is_grammar_accept_string(grammar, SCHEMA_INVALID_TYPE)
    assert not _is_grammar_accept_string(grammar, FREE_TEXT)


def test_compose_minimax_accepts_tool_call_and_schema(
    strict_tools: list[ChatCompletionToolsParam],
):
    composed = compose_tool_call_or_schema(
        _auto_tag("minimax", strict_tools), RESPONSE_SCHEMA
    )
    assert composed is not None
    grammar = _grammar(composed)

    assert _is_grammar_accept_string(grammar, MINIMAX_TOOL_CALL)
    assert _is_grammar_accept_string(grammar, SCHEMA_VALID_JSON)
    assert not _is_grammar_accept_string(grammar, SCHEMA_INVALID_KEY)


def test_compose_requires_tool_call_so_schema_branch_is_reachable(
    strict_tools: list[ChatCompletionToolsParam],
):
    """The schema branch is only enforced because the tool branch is forced to
    require a tool call. A bare auto tag would match free text and make the
    schema constraint vacuous."""
    auto_tag = _auto_tag("hermes", strict_tools)

    vacuous = _grammar(auto_tag)
    assert _is_grammar_accept_string(vacuous, FREE_TEXT)
    assert _is_grammar_accept_string(vacuous, SCHEMA_INVALID_KEY)

    composed = compose_tool_call_or_schema(auto_tag, RESPONSE_SCHEMA)
    assert composed is not None
    constrained = _grammar(composed)
    assert not _is_grammar_accept_string(constrained, FREE_TEXT)
    assert not _is_grammar_accept_string(constrained, SCHEMA_INVALID_KEY)


def test_compose_returns_none_for_non_triggered_format():
    """Formats that cannot require a tool call (e.g. a forced/required tag)
    return None so callers fall back to dropping the response_format."""
    forced = get_model_structural_tag(
        model="hermes",
        tools=[
            ChatCompletionToolsParam(
                type="function",
                function={
                    "name": "get_weather",
                    "parameters": {"type": "object", "properties": {}},
                },
            )
        ],
        tool_choice="required",
        reasoning=False,
    )
    assert isinstance(forced, StructuralTag)
    assert compose_tool_call_or_schema(forced, RESPONSE_SCHEMA) is None


def test_compose_json_object_accepts_any_json_answer(
    strict_tools: list[ChatCompletionToolsParam],
):
    """response_format: {"type": "json_object"} maps to json_schema=True, so
    the answer branch accepts any well-formed JSON rather than a fixed shape.
    """
    composed = compose_tool_call_or_schema(_auto_tag("hermes", strict_tools), True)
    assert composed is not None
    grammar = _grammar(composed)

    assert _is_grammar_accept_string(grammar, HERMES_TOOL_CALL)
    assert _is_grammar_accept_string(grammar, SCHEMA_VALID_JSON)
    assert _is_grammar_accept_string(grammar, ANY_JSON)
    assert not _is_grammar_accept_string(grammar, FREE_TEXT)


@pytest.mark.parametrize("model", ["hermes", "minimax"])
def test_get_composed_structural_tag_bypasses_strict_short_circuit(
    model: str, non_strict_tools: list[ChatCompletionToolsParam]
):
    """get_composed_structural_tag must compose even for non-strict tools:
    get_model_structural_tag drops an auto request with non-strict tools
    (see test_auto_tool_choice_skips_structural_tag_without_strict in
    test_structural_tag_registry.py), but the whole point of this entry
    point is to make non-strict auto tool calling work alongside a schema.
    """
    bare_auto = get_model_structural_tag(
        model=model, tools=non_strict_tools, tool_choice="auto", reasoning=False
    )
    assert bare_auto is None

    composed = get_composed_structural_tag(
        model=model,
        tools=non_strict_tools,
        tool_choice="auto",
        response_format_schema=RESPONSE_SCHEMA,
    )
    assert composed is not None
    grammar = _grammar(composed)
    assert _is_grammar_accept_string(grammar, SCHEMA_VALID_JSON)
    assert not _is_grammar_accept_string(grammar, FREE_TEXT)


def test_get_composed_structural_tag_returns_none_for_non_composable_model(
    non_strict_tools: list[ChatCompletionToolsParam],
):
    """Models without a vLLM-owned registry builder (e.g. the opaque xgrammar
    builtins) are out of scope for this PR; callers fall back unchanged."""
    assert (
        get_composed_structural_tag(
            model="llama",
            tools=non_strict_tools,
            tool_choice="auto",
            response_format_schema=RESPONSE_SCHEMA,
        )
        is None
    )


def test_get_composed_structural_tag_returns_none_for_non_auto_tool_choice(
    non_strict_tools: list[ChatCompletionToolsParam],
):
    assert (
        get_composed_structural_tag(
            model="hermes",
            tools=non_strict_tools,
            tool_choice="required",
            response_format_schema=RESPONSE_SCHEMA,
        )
        is None
    )


def test_get_composed_structural_tag_returns_none_without_tools():
    assert (
        get_composed_structural_tag(
            model="hermes",
            tools=None,
            tool_choice="auto",
            response_format_schema=RESPONSE_SCHEMA,
        )
        is None
    )
