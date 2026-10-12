# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from unittest.mock import MagicMock

import pytest

from vllm.entrypoints.generate.base.protocol import DeltaMessage
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.parser.parser_manager import ParserManager
from vllm.tool_parsers import ToolParserManager
from vllm.tool_parsers.k2_horizon_tool_parser import K2HorizonToolParser

pytestmark = pytest.mark.skip_global_cleanup

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer"},
                    "enabled": {"type": "boolean"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ping",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


@pytest.fixture
def tokenizer():
    tokenizer = MagicMock()
    tokenizer.get_vocab.return_value = {}
    return tokenizer


def _request(
    tool_format: str | None = None,
    tool_choice="auto",
) -> ChatCompletionRequest:
    chat_template_kwargs = (
        {"tool_call_format": tool_format} if tool_format is not None else None
    )
    return ChatCompletionRequest(
        model="k2-horizon",
        messages=[{"role": "user", "content": "test"}],
        tools=TOOLS,
        tool_choice=tool_choice,
        chat_template_kwargs=chat_template_kwargs,
    )


def _group(*calls: str, prefix: str = "", suffix: str = "") -> str:
    return prefix + "<ifm|tool_calls>" + "\n".join(calls) + "</ifm|tool_calls>" + suffix


def _json_call(name: str, arguments: dict) -> str:
    payload = json.dumps({"name": name, "arguments": arguments})
    return f"<ifm|tool_call>{payload}</ifm|tool_call>"


def _xml_call(
    name: str,
    arguments: list[tuple[str, str]],
    *,
    typed: bool = False,
) -> str:
    parts = [f"<ifm|tool_call>{name}"]
    for key, value in arguments:
        parts.append(f"<ifm|arg_key>{key}</ifm|arg_key>")
        if typed:
            arg_type = "integer" if key == "limit" else "string"
            parts.append(f"<ifm|arg_type>{arg_type}</ifm|arg_type>")
        parts.append(f"<ifm|arg_value>{value}</ifm|arg_value>")
    parts.append("</ifm|tool_call>")
    return "".join(parts)


def _collect_stream(
    parser: K2HorizonToolParser,
    request: ChatCompletionRequest,
    output: str,
) -> tuple[str, list[tuple[str, dict]]]:
    messages: list[DeltaMessage | None] = []
    for char in output:
        messages.append(
            parser.extract_tool_calls_streaming(
                previous_text="",
                current_text="",
                delta_text=char,
                previous_token_ids=[],
                current_token_ids=[],
                delta_token_ids=[],
                request=request,
            )
        )
    messages.append(parser.finish_streaming())

    content: list[str] = []
    calls: list[tuple[str, dict]] = []
    for message in messages:
        if message is None:
            continue
        if message.content is not None:
            content.append(message.content)
        for tool_call in message.tool_calls:
            assert tool_call.function is not None
            assert tool_call.function.name is not None
            assert tool_call.function.arguments is not None
            calls.append(
                (
                    tool_call.function.name,
                    json.loads(tool_call.function.arguments),
                )
            )
    return "".join(content), calls


def test_xml_default_parses_multiple_calls_and_coerces_schema(tokenizer):
    parser = K2HorizonToolParser(tokenizer, _request().tools)
    output = _group(
        _xml_call("lookup", [("query", "123"), ("limit", "3")]),
        _xml_call("ping", []),
        prefix="Before ",
        suffix=" after.",
    )

    result = parser.extract_tool_calls(output, _request())

    assert parser.tool_format == "xml"
    assert result.tools_called
    assert result.content == "Before  after."
    assert [call.function.name for call in result.tool_calls] == ["lookup", "ping"]
    assert json.loads(result.tool_calls[0].function.arguments) == {
        "query": "123",
        "limit": 3,
    }


@pytest.mark.parametrize(
    ("tool_format", "call", "expected"),
    [
        ("json", _json_call("lookup", {"limit": "2"}), {"limit": 2}),
        (
            "xml_typed",
            _xml_call("lookup", [("query", "weather"), ("limit", "2")], typed=True),
            {"query": "weather", "limit": 2},
        ),
    ],
)
def test_request_level_format_overrides(tokenizer, tool_format, call, expected):
    request = _request(tool_format)
    parser = K2HorizonToolParser(tokenizer, request.tools)

    result = parser.extract_tool_calls(_group(call), request)

    assert result.tools_called
    assert json.loads(result.tool_calls[0].function.arguments) == expected


@pytest.mark.parametrize(
    "tool_format",
    ["yaml", "python", "xml_untyped", "", None, 1],
)
def test_invalid_request_format_rejected_before_generation(tokenizer, tool_format):
    parser = K2HorizonToolParser(tokenizer, _request().tools)
    request = _request().model_copy(
        update={"chat_template_kwargs": {"tool_call_format": tool_format}}
    )

    with pytest.raises(ValueError, match="Unsupported tool_call_format"):
        parser.adjust_request(request)


@pytest.mark.parametrize(
    ("tool_request", "call"),
    [
        (_request(), _json_call("ping", {})),
        (_request("json"), _xml_call("ping", [])),
        (_request("xml"), _xml_call("lookup", [("query", "x")], typed=True)),
        (_request("xml_typed"), _xml_call("lookup", [("query", "x")])),
    ],
)
def test_mismatched_formats_fail_closed(tokenizer, tool_request, call):
    parser = K2HorizonToolParser(tokenizer, tool_request.tools)
    output = _group(call)

    result = parser.extract_tool_calls(output, tool_request)

    assert not result.tools_called
    assert result.content == output


@pytest.mark.parametrize(
    "tool_choice",
    [
        "auto",
        "required",
        {"type": "function", "function": {"name": "lookup"}},
    ],
)
def test_auto_required_and_named_use_ifm_parser(tokenizer, tool_choice):
    request = _request("json", tool_choice)
    parser_cls = ParserManager.get_parser(
        tool_parser_name="k2_horizon",
        enable_auto_tools=True,
    )
    assert parser_cls is not None
    parser = parser_cls(tokenizer, request.tools)

    assert parser.adjust_request(request) is request
    assert request.structured_outputs is None
    _, content, tool_calls = parser.parse(
        _group(_json_call("lookup", {"enabled": "true"})),
        request,
        enable_auto_tools=True,
    )

    assert content is None
    assert tool_calls is not None
    assert json.loads(tool_calls[0].arguments) == {"enabled": True}


def test_unknown_or_wrong_named_tool_fails_closed(tokenizer):
    named_request = _request(
        "json", {"type": "function", "function": {"name": "lookup"}}
    )
    parser = K2HorizonToolParser(tokenizer, named_request.tools)

    unknown = parser.extract_tool_calls(
        _group(_json_call("missing", {})), _request("json")
    )
    wrong_named = parser.extract_tool_calls(
        _group(_json_call("ping", {})), named_request
    )

    assert not unknown.tools_called
    assert not wrong_named.tools_called


@pytest.mark.parametrize(
    "output",
    [
        "<ifm|tool_calls><ifm|tool_call>ping",
        "<ifm|tool_calls><ifm|tool_call>ping</ifm|tool_call>",
        _group("junk"),
    ],
)
def test_malformed_and_incomplete_input_is_content(tokenizer, output):
    request = _request()
    parser = K2HorizonToolParser(tokenizer, request.tools)

    result = parser.extract_tool_calls(output, request)

    assert not result.tools_called
    assert result.content == output


@pytest.mark.parametrize(
    ("tool_request", "call", "expected"),
    [
        (_request(), _xml_call("lookup", [("limit", "2")]), {"limit": 2}),
        (_request("json"), _json_call("lookup", {"limit": "2"}), {"limit": 2}),
        (
            _request("xml_typed"),
            _xml_call("lookup", [("limit", "2")], typed=True),
            {"limit": 2},
        ),
    ],
)
def test_character_split_streaming(tokenizer, tool_request, call, expected):
    parser = K2HorizonToolParser(tokenizer, tool_request.tools)
    output = _group(call, prefix="Before", suffix="after")

    content, calls = _collect_stream(parser, tool_request, output)

    assert content == "Beforeafter"
    assert calls == [("lookup", expected)]


def test_streaming_flushes_incomplete_markup_as_content(tokenizer):
    request = _request()
    parser = K2HorizonToolParser(tokenizer, request.tools)
    output = "Before<ifm|tool_calls><ifm|tool_call>ping"

    content, calls = _collect_stream(parser, request, output)

    assert content == output
    assert calls == []


def test_whitespace_only_surrounding_content_is_not_preserved(tokenizer):
    request = _request()
    parser = K2HorizonToolParser(tokenizer, request.tools)

    result = parser.extract_tool_calls(
        _group(_xml_call("ping", []), prefix=" \n", suffix="\t "), request
    )

    assert result.tools_called
    assert result.content is None


def test_tool_parser_registered():
    assert ToolParserManager.get_tool_parser("k2_horizon") is K2HorizonToolParser
    assert K2HorizonToolParser.supports_required_and_named is False


class TestCoerceJsonValue:
    """Note(arpera):
    Unit tests coverage for K2HorizonToolParser._coerce_json_value
    Tests are organized in subclasses following the same approach
    that tests/tool_parsers/test_utils.py uses for testing coerce_to_schema_type.

    Once you identify a new regression in K2._coerce_json_value
    please implement a regression test as a subclass here.
    """

    class TestDecodedValuesArePreserved:
        @pytest.mark.parametrize(
            ("value", "schema"),
            [
                ([1, 2], {"type": "array"}),
                ({"a": 1}, {"type": "object"}),
                (3, {"type": "integer"}),
                (True, {"type": "boolean"}),
            ],
        )
        def test_value_matching_its_schema_is_untouched(self, value, schema):
            assert K2HorizonToolParser._coerce_json_value(value, schema) == value

        def test_schema_without_types_returns_value(self):
            assert K2HorizonToolParser._coerce_json_value([1, 2], None) == [1, 2]

    class TestStringOrientedCoercionStillApplies:
        def test_double_encoded_object_is_decoded(self):
            """The helper exists for this: a model sending an object as text."""
            result = K2HorizonToolParser._coerce_json_value(
                '{"a": 1}', {"type": "object"}
            )
            assert result == {"a": 1}

        def test_numeric_string_is_decoded(self):
            assert K2HorizonToolParser._coerce_json_value("3", {"type": "integer"}) == 3

        @pytest.mark.parametrize(
            ("value", "expected"),
            [([1, 2], "[1, 2]"), (3, "3"), (True, "true")],
        )
        def test_declared_string_encodes_a_decoded_value(self, value, expected):
            """A declared string still wins: the encoding is what was asked for."""
            result = K2HorizonToolParser._coerce_json_value(value, {"type": "string"})
            assert result == expected
            assert isinstance(result, str)

    class TestPR57005Regressions:
        """Note(arpera):
        List of regressions that were caught during review of PR #57005
        """

        @pytest.mark.parametrize(
            ("value", "schema"),
            [
                ([1, 2], {"type": "object"}),
                ({"a": 1}, {"type": "array"}),
                (3, {"type": "object"}),
                (True, {"type": "object"}),
                (None, {"type": "object"}),
            ],
        )
        def test_unmatched_schema_keeps_the_decoded_value(self, value, schema):
            result = K2HorizonToolParser._coerce_json_value(value, schema)
            assert result == value
            assert not isinstance(result, str)

        @pytest.mark.parametrize("alias", ["str", "text", "varchar"])
        def test_string_alias_still_encodes(self, alias):
            """The keep-decoded guard must use the same alias table as coercion."""
            result = K2HorizonToolParser._coerce_json_value(3, {"type": alias})
            assert result == "3"
            assert isinstance(result, str)

        def test_zero_fraction_number_reaches_the_integer_branch(self):
            """Shares ``coerce_to_schema_type``'s integer fix: 3.0 is an integer."""
            result = K2HorizonToolParser._coerce_json_value(3.0, {"type": "integer"})
            assert result == 3
            assert isinstance(result, int)


class TestCoerceXmlValue:
    """Unit coverage for ``K2HorizonToolParser._coerce_xml_value``.

    XML tool calls carry every argument as tag text, so the value always
    arrives as a string and the schema is the only source of its type. An
    ``<ifm|arg_type>`` tag, when present, narrows that choice further.
    """

    class TestSchemaDrivenCoercion:
        def test_numeric_string_is_coerced(self):
            assert (
                K2HorizonToolParser._coerce_xml_value("3", {"type": "integer"}, None)
                == 3
            )

        def test_padding_is_stripped_for_non_string_types(self):
            assert (
                K2HorizonToolParser._coerce_xml_value(" 3 ", {"type": "integer"}, None)
                == 3
            )

        def test_padding_is_kept_for_string_types(self):
            """Whitespace can be meaningful once the schema asks for a string."""
            assert (
                K2HorizonToolParser._coerce_xml_value(" x ", {"type": "string"}, None)
                == " x "
            )

        def test_array_is_not_accepted_for_an_object_schema(self):
            assert (
                K2HorizonToolParser._coerce_xml_value("[1,2]", {"type": "object"}, None)
                == "[1,2]"
            )

    class TestExplicitArgType:
        def test_explicit_type_narrows_a_union_schema(self):
            """``<ifm|arg_type>string</ifm|arg_type>`` wins over the priority order."""
            schema = {"type": ["string", "integer"]}
            assert K2HorizonToolParser._coerce_xml_value("42", schema, None) == 42
            assert K2HorizonToolParser._coerce_xml_value("42", schema, "string") == "42"

        def test_explicit_type_drives_coercion_without_a_schema(self):
            assert K2HorizonToolParser._coerce_xml_value("42", None, "integer") == 42

        def test_explicit_any_keeps_the_raw_text(self):
            assert K2HorizonToolParser._coerce_xml_value("42", None, "any") == "42"

    class TestPR57005Regressions:
        """The tail type-mask must not hold back a decoded scalar.

        Rejecting a parse whose type is not declared is right for containers --
        an array must not stand in for an object -- but for a scalar it hands
        the client a four-character ``"null"`` where the model meant a JSON
        null, and the tool sees a string instead of a value.
        """

        @pytest.mark.parametrize(
            ("value", "expected"),
            [("null", None), ("true", True), ("3.5", 3.5)],
        )
        def test_scalar_survives_a_type_the_schema_omits(self, value, expected):
            result = K2HorizonToolParser._coerce_xml_value(
                value, {"type": "integer"}, None
            )
            assert result == expected
            assert not isinstance(result, str)
