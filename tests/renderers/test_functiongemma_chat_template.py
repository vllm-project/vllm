# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for the FunctionGemma chat template rendering tool call arguments."""

import json
from pathlib import Path

import jinja2.sandbox
import pytest

TEMPLATE_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "examples"
    / "tool_chat_template_functiongemma.jinja"
)


@pytest.fixture(scope="module")
def functiongemma_template():
    """Load and compile the FunctionGemma chat template."""
    template_str = TEMPLATE_PATH.read_text()
    env = jinja2.sandbox.ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True
    )
    env.filters["tojson"] = lambda value, **kwargs: json.dumps(
        value, ensure_ascii=False, **kwargs
    )
    env.filters["fromjson"] = json.loads
    return env.from_string(template_str)


def _render(template, arguments, arguments_as_json=False):
    """Render one assistant tool call built from `arguments`."""
    if arguments_as_json:
        arguments = json.dumps(arguments)
    messages = [
        {"role": "user", "content": "weather in Paris?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": "get_weather", "arguments": arguments}}
            ],
        },
    ]
    return template.render(messages=messages, tools=[], add_generation_prompt=False)


def _call_body(rendered):
    """Return the text between `call:get_weather{` and its closing `}`."""
    start = rendered.index("call:get_weather{") + len("call:get_weather{")
    end = rendered.index("}<end_function_call>", start)
    return rendered[start:end]


def _split_arguments(body):
    """Split a rendered argument body on the commas that separate arguments."""
    parts, current, depth, in_escape = [], "", 0, False
    for char in body:
        current += char
        if in_escape:
            if current.endswith("<escape>") and not current.endswith(
                "<escape><escape>"
            ):
                in_escape = False
            continue
        if current.endswith("<escape>"):
            in_escape = True
        elif char in "[{":
            depth += 1
        elif char in "]}":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(current[:-1])
            current = ""
    if current:
        parts.append(current)
    return parts


def _parse_arguments(body):
    """Parse a rendered argument body the way the FunctionGemma parser does.

    Strings are `<escape>` delimited, everything else is JSON.
    """
    arguments = {}
    for part in _split_arguments(body):
        key, _, raw = part.partition(":")
        if raw.startswith("<escape>") and raw.endswith("<escape>"):
            arguments[key] = raw[len("<escape>") : -len("<escape>")]
        else:
            arguments[key] = json.loads(raw)
    return arguments


class TestFunctionGemmaChatTemplateArguments:
    def test_strings_are_escape_delimited(self, functiongemma_template):
        rendered = _render(functiongemma_template, {"city": "Paris"})
        assert "city:<escape>Paris<escape>" in rendered

    def test_non_string_values_are_written_bare(self, functiongemma_template):
        arguments = {"enabled": True, "off": False, "hour": 7}
        body = _call_body(_render(functiongemma_template, arguments))
        assert body == "enabled:true,off:false,hour:7"
        assert "<escape>" not in body

    def test_nested_object_and_array_are_written_as_json(self, functiongemma_template):
        arguments = {
            "tags": ["a", "b"],
            "opts": {"city": "SF", "zip": 94103},
        }
        body = _call_body(_render(functiongemma_template, arguments))
        assert "<escape>" not in body
        assert _parse_arguments(body) == arguments

    def test_arguments_round_trip_through_the_parser(self, functiongemma_template):
        arguments = {
            "city": "Paris",
            "dry": False,
            "limit": 3,
            "unit": None,
            "addr": {"city": "SF", "zip": 94103},
            "tags": ["a", "b"],
        }
        body = _call_body(_render(functiongemma_template, arguments))
        assert _parse_arguments(body) == arguments

    def test_arguments_given_as_a_json_string(self, functiongemma_template):
        arguments = {"city": "Paris", "dry": False}
        rendered = _render(functiongemma_template, arguments, arguments_as_json=True)
        assert _call_body(rendered) == "city:<escape>Paris<escape>,dry:false"
