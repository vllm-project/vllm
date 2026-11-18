# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from copy import deepcopy
from typing import Any

import pytest
from xgrammar import Grammar
from xgrammar.testing import _is_grammar_accept_string

from vllm.parser.chat_parsing.response_templates import load_response_template
from vllm.parser.response_template_grammar import (
    answer_rest_structural_tag,
    tool_call_structural_tag,
)

TEMPLATE: dict[str, Any] = {
    "start_anchor": "<assistant>",
    "fields": {
        "thinking": {"open": "<think>", "close": "</think>", "content": "text"},
        "content": {"content": "text"},
        "tool_calls": {
            "open_pattern": r'<call name="(?P<name>[\w.-]+)">',
            "close": "</call>",
            "repeats": True,
            "content": "xml-inline",
            "content_args": {
                "tag_pattern": r'<arg name="(?P<key>[^"]+)">(?P<value>.*?)</arg>'
            },
            "transform": {
                "type": "function",
                "function": {"name": "{name}", "arguments": "{content}"},
            },
        },
    },
}
ANSWER_TEMPLATE = deepcopy(TEMPLATE)
ANSWER_TEMPLATE["fields"]["content"] = {
    "open": "<answer>",
    "close": "</answer>",
    "content": "text",
}


def tool(parameters, name="f", strict=True) -> dict[str, Any]:
    return {"name": name, "parameters": parameters, "strict": strict}


WEATHER = tool(
    {
        "type": "object",
        "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
        "required": ["city"],
    },
    name="get.weather",
)
PING = {"name": "ping"}
CALL = '<call name="get.weather"><arg name="city">Paris</arg></call>'


def build(
    template=TEMPLATE,
    functions=(WEATHER, PING),
    *,
    parallel=True,
    thinking=True,
    answer=False,
    answer_format=None,
    sorted_arguments=False,
):
    fields = load_response_template(template).fields
    return tool_call_structural_tag(
        fields["tool_calls"],
        list(functions),
        parallel_tool_calls=parallel,
        thinking=fields["thinking"] if thinking else None,
        answer=fields["content"] if answer else None,
        answer_format=answer_format,
        sorted_arguments=sorted_arguments,
    )


def accepts(tag, text):
    return _is_grammar_accept_string(Grammar.from_structural_tag(json.dumps(tag)), text)


def call(*arguments):
    body = "".join(f'<arg name="{key}">{value}</arg>' for key, value in arguments)
    return f'<call name="f">{body}</call>'


@pytest.mark.parametrize(
    ("text", "accepted"),
    [
        ("<think>plan</think>" + CALL, True),
        (CALL, True),
        (CALL.removesuffix("</call>"), True),
        (
            (
                '<call name="get.weather"><arg name="city">Paris</arg>\n'
                '<arg name="days">3</arg></call>'
            ),
            True,
        ),
        (
            (
                '<call name="get.weather"><arg name="city">Paris</arg>'
                '<arg name="days">soon</arg></call>'
            ),
            False,
        ),
        ('<call name="get.weather"><arg name="days">3</arg></call>', False),
        (CALL + '<call name="ping"><arg name="any">x</arg></call>', True),
        ('<call name="other"></call>', False),
        ("<think>plan</think>The answer.", False),
    ],
)
def test_forced_calls_follow_the_template_format(text, accepted):
    assert accepts(build(), text) is accepted


def test_single_call_when_parallel_calls_are_disabled():
    tag = build(parallel=False)

    assert accepts(tag, CALL)
    assert not accepts(tag, CALL + CALL)


def test_thinking_is_rejected_when_not_requested():
    assert not accepts(build(thinking=False), "<think>plan</think>" + CALL)


@pytest.mark.parametrize(
    ("text", "accepted"),
    [
        ("<think>plan</think><answer>Sunny.", True),
        ("<answer>Sunny.</answer>", True),
        ("<think>plan</think>" + CALL, True),
        ("<answer>Sunny.</answer>" + CALL, False),
        ('<call name="get.weather"><arg name="days">soon</arg></call>', False),
    ],
)
def test_answer_field_lets_the_model_reply_instead(text, accepted):
    assert accepts(build(ANSWER_TEMPLATE, answer=True), text) is accepted


@pytest.mark.parametrize("required", [False, True])
@pytest.mark.parametrize(
    ("text", "optional_only"),
    [
        ("<think>plan</think>" + CALL, False),
        ("<think>\nplan</think><answer>Sunny.", False),
        ("<think> \n</think><answer>Sunny.", False),
        (CALL, True),
        ("<answer>Sunny.", True),
        ("<think></think><answer>Sunny.", True),
        ("<think><</think><answer>Sunny.", True),
    ],
)
def test_required_reasoning_opens_the_message_with_nonempty_reasoning(
    required, text, optional_only
):
    fields = load_response_template(ANSWER_TEMPLATE).fields
    tag = tool_call_structural_tag(
        fields["tool_calls"],
        [WEATHER],
        parallel_tool_calls=True,
        thinking=fields["thinking"],
        answer=fields["content"],
        message_start=["<assistant>"],
        requires_reasoning=required,
    )

    assert accepts(tag, text) is (not required or not optional_only)


def test_answer_rest_finishes_the_open_answer():
    answer = load_response_template(ANSWER_TEMPLATE).fields["content"]
    tag = answer_rest_structural_tag(answer)

    assert accepts(tag, "Sunny.</answer>")
    assert accepts(tag, "Sunny.")
    assert not accepts(tag, "Sunny.</answer>" + CALL)


def test_answer_without_literal_opener_is_not_derived():
    assert build(answer=True) is None


def test_answer_holds_the_answer_format():
    schema = {"type": "json_schema", "json_schema": {"type": "object"}}
    with_tools = build(ANSWER_TEMPLATE, answer=True, answer_format=schema)
    answer_only = build(ANSWER_TEMPLATE, (), answer=True, answer_format=schema)

    for tag in (with_tools, answer_only):
        assert accepts(tag, '<think>plan</think><answer>{"a": 1}</answer>')
        assert not accepts(tag, "<answer>Sunny.</answer>")
    assert accepts(with_tools, CALL)
    assert not accepts(answer_only, CALL)


def test_json_arguments_follow_the_schema():
    template = deepcopy(TEMPLATE)
    template["fields"]["tool_calls"].update(content="json", content_args={})
    tag = build(template, [WEATHER])

    assert accepts(tag, '<call name="get.weather">{"city": "Paris"}</call>')
    assert not accepts(tag, '<call name="get.weather">{"days": 3}</call>')


END_X = {"tag_pattern": r'<arg name="(?P<key>[^"]+)">(?P<value>.*?)X'}


@pytest.mark.parametrize(
    "change",
    [
        {"content": "json", "content_args": {"unquoted_keys": True}},
        {
            "transform": {
                "type": "function",
                "function": {"name": "fixed", "arguments": "{content}"},
            }
        },
        {"content": "kv-lines", "content_args": {}},
        # Framing in which argument text can spell a closer, where the parser
        # would end the call.
        {"close": ["}X", "</call>"], "content_args": END_X},
        {
            "close": ["}X", "</call>"],
            "content_args": {**END_X, "value_parser": {"name": "json"}},
        },
        {"close": ["7X", "</call>"], "content_args": END_X},
        {"close": [">X", "</call>"]},
        {"close": ["></call>", "</call>"]},
        {"close": ["><a", "</call>"]},
        {"close": ["}</call>", "</call>"], "content": "json", "content_args": {}},
    ],
)
def test_underivable_tool_formats_are_not_derived(change):
    template = deepcopy(TEMPLATE)
    template["fields"]["tool_calls"].update(change)

    assert build(template) is None


def test_non_literal_thinking_delimiters_are_not_derived():
    template = deepcopy(TEMPLATE)
    del template["fields"]["thinking"]["open"]
    template["fields"]["thinking"]["open_pattern"] = r"<think(?: mode=\w+)?>"

    assert build(template) is None
    assert build(template, thinking=False) is not None


LEVEL = {"type": "object", "properties": {"level": {"$ref": "#/$defs/level"}}}


@pytest.mark.parametrize(
    ("schema", "value", "accepted"),
    [
        ({"type": "integer", "nullable": True}, "null", True),
        ({"type": "integer", "nullable": True}, "soon", False),
        ({"type": ["integer", "null"]}, "null", True),
        ({"anyOf": [{"type": "integer"}, {"type": "boolean"}]}, "true", True),
        ({"anyOf": [{"type": "integer"}, {"type": "boolean"}]}, "soon", False),
        ({"type": ["string", "integer"]}, "soon", True),
        ({"enum": ["celsius", "fahrenheit"]}, "celsius", True),
        ({"enum": ["celsius", "fahrenheit"]}, "kelvin", False),
        ({"type": "string", "const": "on"}, "off", False),
        (LEVEL, '{"level": 2}', True),
        (LEVEL, '{"level": "high"}', False),
    ],
)
def test_argument_values_follow_how_the_parser_reads_them(schema, value, accepted):
    parameters = {
        "type": "object",
        "properties": {"v": schema},
        "required": ["v"],
        "$defs": {"level": {"type": "integer"}},
    }

    assert accepts(build(functions=[tool(parameters)]), call(("v", value))) is accepted


@pytest.mark.parametrize(
    ("schema", "strict", "derivable"),
    [
        # The parser reads values without a type as text.
        ({"enum": [1, 2]}, True, False),
        ({"$ref": "#/$defs/level"}, True, False),
        ({"$ref": "#/$defs/level"}, False, True),
        # Raw text cannot hold its closer.
        ({"enum": ["a</arg>b"]}, True, False),
        ({"type": "object", "properties": {"x": {"$ref": "#/x"}}}, True, False),
        (False, True, False),
        # A match could write the start of the value's end.
        ({"type": "string", "pattern": "^[a-z<]+$"}, True, False),
        ({"type": "string", "pattern": "^[a-z<]+$"}, False, True),
        ({"type": "string", "pattern": "^a.b$"}, True, False),
        # XGrammar holds length bounds only in JSON.
        ({"type": "string", "minLength": 3}, True, False),
        ({"type": "string", "format": "ipv6"}, True, False),
    ],
)
def test_schemas_text_values_cannot_hold_are_not_derived(schema, strict, derivable):
    parameters = {
        "type": "object",
        "properties": {"v": schema},
        "$defs": {"level": {"type": "integer"}},
    }

    assert (build(functions=[tool(parameters, strict=strict)]) is not None) is derivable


@pytest.mark.parametrize(
    ("schema", "value", "accepted"),
    [
        ({"type": "string", "format": "email"}, "a.b@x.co", True),
        ({"type": "string", "format": "email"}, "ab", False),
        ({"type": "string", "format": "date-time"}, "2026-10-06T12:00:00Z", True),
        ({"type": "string", "format": "date-time"}, "2026-13-06T12:00:00Z", False),
        ({"type": "string", "pattern": "^[A-Z]{3}$"}, "ABC", True),
        ({"type": "string", "pattern": "^[A-Z]{3}$"}, "ABCD", False),
        # Like XGrammar, the format overrides the pattern and the length bounds.
        (
            {"type": "string", "format": "uuid", "pattern": "^x$", "maxLength": 1},
            "123e4567-e89b-12d3-a456-426614174000",
            True,
        ),
    ],
)
def test_text_values_hold_string_formats_and_patterns(schema, value, accepted):
    parameters = {"type": "object", "properties": {"v": schema}, "required": ["v"]}

    assert accepts(build(functions=[tool(parameters)]), call(("v", value))) is accepted


def test_values_cannot_hold_a_closer():
    template = deepcopy(TEMPLATE)
    template["fields"]["tool_calls"]["close"] = ["</call>", "STOP"]
    text = {"type": "object", "properties": {"v": {"type": "string"}}}
    strings = {"type": "array", "items": {"type": "string"}}
    json_text = {"type": "object", "properties": {"v": strings}}

    assert accepts(build(template, [tool(text)]), call(("v", "ab")))
    assert not accepts(build(template, [tool(text)]), call(("v", "aSTOPb")))
    assert not accepts(
        build(template, [tool(None, strict=False)]), call(("v", "aSTOPb"))
    )
    assert accepts(build(template, [tool(json_text)]), call(("v", '["ab"]')))
    assert not accepts(build(template, [tool(json_text)]), call(("v", '["aSTOPb"]')))


def test_extra_arguments_need_additional_properties():
    parameters = {"type": "object", "properties": {"city": {"type": "string"}}}
    extra = call(("city", "Paris"), ("zip", "75001"))

    assert not accepts(build(functions=[tool(parameters)]), extra)
    # Next to declared keys, an extra one could repeat them unchecked.
    allowed = {**parameters, "additionalProperties": True}
    assert build(functions=[tool(allowed)]) is None
    free_form = {"type": "object", "additionalProperties": True}
    assert accepts(build(functions=[tool(free_form)]), extra)


def test_only_strict_tools_follow_their_schema():
    off_schema = call(("days", "soon"))
    parameters = WEATHER["parameters"]

    assert accepts(build(functions=[tool(parameters, strict=False)]), off_schema)
    assert not accepts(build(functions=[tool(parameters)]), off_schema)
    assert accepts(build(functions=[tool(None)]), call())
    assert not accepts(build(functions=[tool(None)]), call(("a", "x")))


def test_annotations_keep_strict_arguments_to_their_schema():
    parameters = {
        **WEATHER["parameters"],
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "weather",
    }
    tag = build(functions=[tool(parameters)])

    assert accepts(tag, call(("city", "Paris"), ("days", "3")))
    assert not accepts(tag, call(("city", "Paris"), ("days", "soon")))


@pytest.mark.parametrize(
    ("arguments", "accepted", "accepted_sorted"),
    [
        ((("path", "a"), ("content", "b")), True, True),
        ((("content", "b"), ("path", "a")), False, True),
        ((("path", "a"),), True, True),
        ((("content", "b"),), False, False),
        ((("path", "a"), ("path", "c")), False, False),
        ((("path", "a"), ("mode", "c")), False, False),
    ],
)
def test_strict_arguments_follow_declared_or_sorted_order(
    arguments, accepted, accepted_sorted
):
    parameters = {
        "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
        "required": ["path"],
    }
    tags = [
        build(functions=[tool(parameters)], sorted_arguments=sorted_arguments)
        for sorted_arguments in (False, True)
    ]

    assert [accepts(tag, call(*arguments)) for tag in tags] == [
        accepted,
        accepted_sorted,
    ]


LOOKAHEAD_TEMPLATE = deepcopy(TEMPLATE)
LOOKAHEAD_TEMPLATE["fields"]["tool_calls"]["content_args"]["tag_pattern"] = (
    r'<arg name="(?P<key>[^"]+)">(?P<value>.*?)\n(?:</arg>|(?=<arg name="))'
)


@pytest.mark.parametrize(
    ("body", "accepted_loose", "accepted_strict"),
    [
        ('<arg name="city">Paris\n</arg>\n<arg name="days">3\n</arg>', True, True),
        ('<arg name="city">Paris\n<arg name="days">3\n</arg>', True, True),
        ('<arg name="days">3\n<arg name="city">Paris\n</arg>', True, False),
        ('<arg name="city">Paris\n<arg name="city">Lyon\n</arg>', True, False),
        ('<arg name="city">Paris\n<arg name="hours">3\n</arg>', True, False),
        ('<arg name="city">Paris\n<arg name="', False, False),
        ('<arg name="city">Paris\n<arg name="days\n</arg>', False, False),
        ('<arg name="city">a <arg name="b"> c\n</arg>', True, True),
    ],
)
def test_values_end_where_a_lookahead_end_does(body, accepted_loose, accepted_strict):
    parameters = WEATHER["parameters"]
    tags = [
        build(LOOKAHEAD_TEMPLATE, [tool(parameters, strict=strict)])
        for strict in (False, True)
    ]
    text = f'<call name="f">{body}</call>'

    assert [accepts(tag, text) for tag in tags] == [accepted_loose, accepted_strict]


@pytest.mark.parametrize("lookahead", [False, True])
@pytest.mark.parametrize(
    ("keys", "accepted"),
    [
        ("b", True),
        ("ab", True),
        ("bc", True),
        ("bd", True),
        ("be", True),
        ("abde", True),
        ("bcde", True),
        ("", False),
        ("a", False),
        ("acd", False),
        ("bdc", False),
        ("bcc", False),
    ],
)
def test_last_parameter_follows_every_required_one(keys, accepted, lookahead):
    parameters = {
        "properties": {key: {"type": "string"} for key in "abcde"},
        "required": ["b"],
    }
    tag = build(LOOKAHEAD_TEMPLATE, [tool(parameters)])
    ends = ["\n" if lookahead else "\n</arg>"] * (len(keys) - 1) + ["\n</arg>"]
    body = "".join(f'<arg name="{k}">x{e}' for k, e in zip(keys, ends))

    assert accepts(tag, f'<call name="f">{body}</call>') is accepted


def test_required_argument_without_schema_is_still_required():
    parameters = {"type": "object", "properties": {}, "required": ["a"]}
    tag = build(functions=[tool(parameters)])

    assert accepts(tag, call(("a", "x")))
    assert not accepts(tag, call())
    assert build(functions=[tool({**parameters, "required": ["a", 1]})]) is None


@pytest.mark.parametrize(
    ("function", "match"),
    [
        (tool({}, name="get weather"), "tool name"),
        (tool({"properties": {'a"b': {"type": "string"}}}), "parameter name"),
    ],
)
def test_unencodable_names_are_rejected(function, match):
    with pytest.raises(ValueError, match=match):
        build(functions=[function])
