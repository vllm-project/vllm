# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Shared regression tests for the Qwen3 XML and Coder tool parsers.

These tests cover behaviour that BOTH parsers must implement identically.
Each test runs twice — once against ``Qwen3XMLToolParser`` and once against
``Qwen3CoderToolParser`` — via the ``parser_cls`` fixture.  Tests that
target streaming-mode-specific quirks of one parser only stay in their
parser-specific file (``test_qwen3xml_tool_parser.py`` or
``test_qwen3coder_tool_parser.py``).
"""
import json

import pytest

from tests.tool_parsers.utils import run_tool_extraction_streaming
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
)
from vllm.tokenizers import get_tokenizer
from vllm.tool_parsers.qwen3coder_tool_parser import Qwen3CoderToolParser
from vllm.tool_parsers.qwen3xml_tool_parser import Qwen3XMLToolParser

MODEL = "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"


@pytest.fixture(scope="module")
def qwen3_tokenizer():
    return get_tokenizer(tokenizer_name=MODEL)


@pytest.fixture(
    params=[Qwen3XMLToolParser, Qwen3CoderToolParser],
    ids=["xml", "coder"],
)
def parser_cls(request):
    return request.param


# ---------------------------------------------------------------------------
# Value conversion: string "null" must NOT become JSON null
# ---------------------------------------------------------------------------


def test_string_null_value_preserved(qwen3_tokenizer, parser_cls):
    """A string-typed parameter with literal value "null" must be preserved
    as the string "null" (not converted to Python None / JSON null).

    Root cause: _convert_param_value must check the schema's ``string``
    type BEFORE the "null" shortcut — otherwise any param whose raw text
    is "null" becomes None regardless of declared type.
    """
    tools = [
        ChatCompletionToolsParam(
            type="function",
            function={
                "name": "search",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
            },
        )
    ]
    parser = parser_cls(qwen3_tokenizer, tools=tools)
    model_output = (
        "<tool_call>\n"
        "<function=search>\n"
        "<parameter=query>null</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )
    request = ChatCompletionRequest(model=MODEL, messages=[], tools=tools)
    result = parser.extract_tool_calls(model_output, request=request)

    assert result.tools_called
    args = json.loads(result.tool_calls[0].function.arguments)
    assert args["query"] == "null", (
        f"String parameter 'null' was converted incorrectly. "
        f"Got: {args.get('query')!r}"
    )


# ---------------------------------------------------------------------------
# anyOf nullable schema — type detection
# ---------------------------------------------------------------------------


def test_anyof_string_null_keeps_value_as_string(qwen3_tokenizer, parser_cls):
    """anyOf [{type: string}, {type: null}] with a numeric-looking value
    must keep the value as a string (the schema declares ``string``).

    Root cause: anyOf was previously treated as ``object`` (for the Coder
    parser) or fell back to ``string`` only when no object/array option
    was present (for the XML parser).  The correct behaviour is to pick
    the FIRST non-null type from the anyOf list.
    """
    tools = [
        ChatCompletionToolsParam(
            type="function",
            function={
                "name": "set_code",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "anyOf": [{"type": "string"}, {"type": "null"}],
                        },
                    },
                },
            },
        )
    ]
    parser = parser_cls(qwen3_tokenizer, tools=tools)
    model_output = (
        "<tool_call>\n"
        "<function=set_code>\n"
        "<parameter=code>42</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )
    request = ChatCompletionRequest(model=MODEL, messages=[], tools=tools)
    result = parser.extract_tool_calls(model_output, request=request)

    assert result.tools_called
    args = json.loads(result.tool_calls[0].function.arguments)
    assert args["code"] == "42", (
        f"anyOf string|null param '42' was parsed as "
        f"{type(args['code']).__name__}: {args['code']!r}"
    )


def test_anyof_integer_null_parses_as_int(qwen3_tokenizer, parser_cls):
    """anyOf [{type: integer}, {type: null}] must parse a numeric value as
    an int.  Previously the XML parser ignored anyOf for non-container
    types and silently treated the param as ``string``.
    """
    tools = [
        ChatCompletionToolsParam(
            type="function",
            function={
                "name": "set_count",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "count": {
                            "anyOf": [{"type": "integer"}, {"type": "null"}],
                        },
                    },
                },
            },
        )
    ]
    parser = parser_cls(qwen3_tokenizer, tools=tools)
    model_output = (
        "<tool_call>\n"
        "<function=set_count>\n"
        "<parameter=count>42</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )
    request = ChatCompletionRequest(model=MODEL, messages=[], tools=tools)
    result = parser.extract_tool_calls(model_output, request=request)

    assert result.tools_called
    args = json.loads(result.tool_calls[0].function.arguments)
    assert args["count"] == 42, (
        f"anyOf integer|null: expected int 42, got {args['count']!r}"
    )


# ---------------------------------------------------------------------------
# anyOf object schema — value not double-encoded
# ---------------------------------------------------------------------------

_ANYOF_OBJECT_TOOLS = [
    ChatCompletionToolsParam(
        type="function",
        function={
            "name": "update_record",
            "parameters": {
                "type": "object",
                "properties": {
                    "data": {
                        "anyOf": [{"type": "object"}, {"type": "null"}],
                    },
                },
            },
        },
    )
]

_ANYOF_OBJECT_OUTPUT = (
    "<tool_call>\n"
    "<function=update_record>\n"
    '<parameter=data>{"key": "value", "count": 42}</parameter>\n'
    "</function>\n"
    "</tool_call>"
)


def test_anyof_object_param_not_double_encoded_nonstreaming(
    qwen3_tokenizer, parser_cls
):
    parser = parser_cls(qwen3_tokenizer, tools=_ANYOF_OBJECT_TOOLS)
    request = ChatCompletionRequest(
        model=MODEL, messages=[], tools=_ANYOF_OBJECT_TOOLS
    )
    result = parser.extract_tool_calls(_ANYOF_OBJECT_OUTPUT, request=request)

    assert result.tools_called
    args = json.loads(result.tool_calls[0].function.arguments)
    assert isinstance(args["data"], dict), (
        f"anyOf object param was double-encoded: data={args['data']!r}"
    )
    assert args["data"] == {"key": "value", "count": 42}


def test_anyof_object_param_not_double_encoded_streaming(
    qwen3_tokenizer, parser_cls
):
    parser = parser_cls(qwen3_tokenizer, tools=_ANYOF_OBJECT_TOOLS)
    request = ChatCompletionRequest(
        model=MODEL, messages=[], tools=_ANYOF_OBJECT_TOOLS
    )
    deltas = [
        "<tool_call>",
        "\n<function=update_record>",
        '\n<parameter=data>{"key": "value", "count": 42}</parameter>',
        "\n</function>",
        "\n</tool_call>",
    ]
    reconstructor = run_tool_extraction_streaming(
        parser, deltas, request, assert_one_tool_per_delta=False
    )
    assert len(reconstructor.tool_calls) == 1
    args = json.loads(reconstructor.tool_calls[0].function.arguments)
    assert isinstance(args["data"], dict), (
        f"anyOf object param was double-encoded in streaming: "
        f"data={args['data']!r}"
    )


# ---------------------------------------------------------------------------
# Object param double-encoded as JSON-encoded Python repr
# ---------------------------------------------------------------------------

_DOUBLE_ENCODED_TOOLS = [
    ChatCompletionToolsParam(
        type="function",
        function={
            "name": "process",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "data": {"type": "object"},
                },
            },
        },
    )
]

_DOUBLE_ENCODED_OUTPUT = (
    "<tool_call>\n"
    "<function=process>\n"
    "<parameter=name>\nhello\n</parameter>\n"
    "<parameter=data>\n\"{'key': 'value', 'n': 1}\"\n</parameter>\n"
    "</function>\n"
    "</tool_call>\n"
)


def test_double_encoded_object_param_nonstreaming(qwen3_tokenizer, parser_cls):
    """A model trained with a buggy template (json.dumps(str(dict))) emits
    object args as a JSON-encoded Python repr string.  The parser must
    double-decode it back to a dict.
    """
    parser = parser_cls(qwen3_tokenizer, tools=_DOUBLE_ENCODED_TOOLS)
    request = ChatCompletionRequest(
        model=MODEL, messages=[], tools=_DOUBLE_ENCODED_TOOLS
    )
    result = parser.extract_tool_calls(_DOUBLE_ENCODED_OUTPUT, request=request)

    assert result.tools_called
    args = json.loads(result.tool_calls[0].function.arguments)
    assert args["name"] == "hello"
    assert isinstance(args["data"], dict), (
        f"Expected dict, got {type(args['data'])}: {args['data']!r}"
    )
    assert args["data"] == {"key": "value", "n": 1}


def test_double_encoded_object_param_streaming(qwen3_tokenizer, parser_cls):
    parser = parser_cls(qwen3_tokenizer, tools=_DOUBLE_ENCODED_TOOLS)
    request = ChatCompletionRequest(
        model=MODEL, messages=[], tools=_DOUBLE_ENCODED_TOOLS
    )
    reconstructor = run_tool_extraction_streaming(
        parser, _DOUBLE_ENCODED_OUTPUT, request, assert_one_tool_per_delta=False
    )
    assert len(reconstructor.tool_calls) == 1
    args = json.loads(reconstructor.tool_calls[0].function.arguments)
    assert args["name"] == "hello"
    assert isinstance(args["data"], dict), (
        f"Expected dict, got {type(args['data'])}: {args['data']!r}"
    )
    assert args["data"] == {"key": "value", "n": 1}


# ---------------------------------------------------------------------------
# Parameter value containing XML structural tags as literal text.
# Expected: the value is preserved intact, no spurious extra parameters
# are created from the embedded tags.
# ---------------------------------------------------------------------------

_WRITE_FILE_TOOLS = [
    ChatCompletionToolsParam(
        type="function",
        function={
            "name": "write_file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
            },
        },
    )
]

# Content with all four structural tags as literal strings (a Python file
# that documents the tool-call format).
_XML_TAGS_IN_CONTENT = (
    'char_deltas = [\n'
    '    "<tool_call>\\n",\n'
    '    "<parameter=query>\\n",\n'
    '    "\\n</parameter>\\n",\n'
    '    "</function>\\n",\n'
    ']\n'
)

_WRITE_FILE_XML_TAGS_OUTPUT = (
    "<tool_call>\n"
    "<function=write_file>\n"
    "<parameter=path>\ntest.py\n</parameter>\n"
    f"<parameter=content>\n{_XML_TAGS_IN_CONTENT}</parameter>\n"
    "</function>\n"
    "</tool_call>\n"
)


def test_content_with_xml_structural_tags_nonstreaming(
    qwen3_tokenizer, parser_cls
):
    """Non-streaming: a string param whose value embeds <tool_call>,
    <parameter=...>, </parameter>, </function> as literal text must be
    extracted intact, with no spurious extra params being created from
    the embedded tags.
    """
    parser = parser_cls(qwen3_tokenizer, tools=_WRITE_FILE_TOOLS)
    request = ChatCompletionRequest(
        model=MODEL, messages=[], tools=_WRITE_FILE_TOOLS
    )
    result = parser.extract_tool_calls(
        _WRITE_FILE_XML_TAGS_OUTPUT, request=request
    )

    assert result.tools_called
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].function.name == "write_file"
    args = json.loads(result.tool_calls[0].function.arguments)
    assert list(args.keys()) == ["path", "content"], (
        f"Spurious params from embedded tags: {list(args.keys())}"
    )
    assert args["path"] == "test.py"
    expected = _XML_TAGS_IN_CONTENT.rstrip("\n")
    assert args["content"] == expected, (
        f"content was truncated/corrupted. Got: {args.get('content')!r}"
    )


def test_content_with_xml_structural_tags_streaming(
    qwen3_tokenizer, parser_cls
):
    """Streaming variant: pre-formed chunks, full content in one delta."""
    parser = parser_cls(qwen3_tokenizer, tools=_WRITE_FILE_TOOLS)
    request = ChatCompletionRequest(
        model=MODEL, messages=[], tools=_WRITE_FILE_TOOLS
    )
    char_deltas = [
        "<tool_call>\n",
        "<function=write_file>\n",
        "<parameter=path>\ntest.py\n</parameter>\n",
        f"<parameter=content>\n{_XML_TAGS_IN_CONTENT}</parameter>\n",
        "</function>\n",
        "</tool_call>\n",
    ]
    reconstructor = run_tool_extraction_streaming(
        parser, char_deltas, request, assert_one_tool_per_delta=False
    )
    assert len(reconstructor.tool_calls) == 1
    assert reconstructor.tool_calls[0].function.name == "write_file"
    args = json.loads(reconstructor.tool_calls[0].function.arguments)
    assert list(args.keys()) == ["path", "content"], (
        f"Spurious params from embedded tags: {list(args.keys())}"
    )
    assert args["path"] == "test.py"
    expected = _XML_TAGS_IN_CONTENT.rstrip("\n")
    assert args["content"] == expected


# ---------------------------------------------------------------------------
# Parameter value containing </parameter> and <parameter=NAME> on their
# OWN lines (Jinja2 templates, parser fixtures, etc.).  Schema filtering
# must prevent the unknown name from being treated as structural.
# ---------------------------------------------------------------------------

_CONTENT_WITH_PARAM_LIKE_LINES = (
    'TOOL_CALL_TEMPLATE = """\n'
    "</parameter>\n"
    "<parameter=new_string>\n"
    "#!/usr/bin/env python3\n"
    "</parameter>\n"
    '"""\n'
)

_WRITE_FILE_PARAM_LIKE_LINES_OUTPUT = (
    "<tool_call>\n"
    "<function=write_file>\n"
    "<parameter=path>\ntest_template.py\n</parameter>\n"
    f"<parameter=content>\n{_CONTENT_WITH_PARAM_LIKE_LINES}</parameter>\n"
    "</function>\n"
    "</tool_call>\n"
)


def test_content_with_param_like_lines_nonstreaming(
    qwen3_tokenizer, parser_cls
):
    """Non-streaming: ``</parameter>`` and ``<parameter=NAME>`` on their
    own lines inside a string value must not terminate the parameter
    early.  Requires schema-based filtering so that ``new_string`` (not a
    real parameter of write_file) is treated as literal text.
    """
    parser = parser_cls(qwen3_tokenizer, tools=_WRITE_FILE_TOOLS)
    request = ChatCompletionRequest(
        model=MODEL, messages=[], tools=_WRITE_FILE_TOOLS
    )
    result = parser.extract_tool_calls(
        _WRITE_FILE_PARAM_LIKE_LINES_OUTPUT, request=request
    )

    assert result.tools_called
    args = json.loads(result.tool_calls[0].function.arguments)
    assert list(args.keys()) == ["path", "content"], (
        f"Spurious params: {list(args.keys())}"
    )
    assert args["path"] == "test_template.py"
    expected = _CONTENT_WITH_PARAM_LIKE_LINES.rstrip("\n")
    assert args["content"] == expected, (
        f"content truncated/wrong: {args.get('content')!r}"
    )


def test_content_with_param_like_lines_streaming(qwen3_tokenizer, parser_cls):
    """Streaming variant: each structural-looking literal line arrives in
    its own delta — the critical case is when ``</parameter>\\n`` appears
    alone with empty lookahead, which must NOT be treated as a real
    structural close.
    """
    parser = parser_cls(qwen3_tokenizer, tools=_WRITE_FILE_TOOLS)
    request = ChatCompletionRequest(
        model=MODEL, messages=[], tools=_WRITE_FILE_TOOLS
    )
    char_deltas = [
        "<tool_call>\n",
        "<function=write_file>\n",
        "<parameter=path>\ntest_template.py\n</parameter>\n",
        '<parameter=content>\nTOOL_CALL_TEMPLATE = """\n',
        "</parameter>\n",            # literal close — alone in its delta
        "<parameter=new_string>\n",  # literal new-param line
        "#!/usr/bin/env python3\n",
        "</parameter>\n",            # second literal close
        '"""\n',
        "</parameter>\n",            # REAL close of content
        "</function>\n",
        "</tool_call>\n",
    ]
    reconstructor = run_tool_extraction_streaming(
        parser, char_deltas, request, assert_one_tool_per_delta=False
    )
    assert len(reconstructor.tool_calls) == 1
    args = json.loads(reconstructor.tool_calls[0].function.arguments)
    assert list(args.keys()) == ["path", "content"], (
        f"Spurious params: {list(args.keys())}"
    )
    assert args["path"] == "test_template.py"
    expected = _CONTENT_WITH_PARAM_LIKE_LINES.rstrip("\n")
    assert args["content"] == expected


# ---------------------------------------------------------------------------
# Array param containing JSON true/false/null
# ---------------------------------------------------------------------------

_ARRAY_TOOLS = [
    ChatCompletionToolsParam(
        type="function",
        function={
            "name": "pick",
            "parameters": {
                "type": "object",
                "properties": {"items": {"type": "array"}},
            },
        },
    )
]

_ARRAY_WITH_JSON_BOOL_OUTPUT = (
    "<tool_call>\n<function=pick>\n"
    '<parameter=items>\n["a", "b", 1, true]\n</parameter>\n'
    "</function>\n</tool_call>"
)


def test_array_with_json_bool(qwen3_tokenizer, parser_cls):
    """An array param containing a JSON literal (``true``/``false``/``null``)
    must be parsed as a real Python list, not wrapped as a string.

    Root cause for the XML parser: the deferred path used
    ``ast.literal_eval`` first, which doesn't understand JSON tokens.
    Both parsers must try ``json.loads`` before falling back to
    ``ast.literal_eval``.
    """
    parser = parser_cls(qwen3_tokenizer, tools=_ARRAY_TOOLS)
    request = ChatCompletionRequest(
        model=MODEL, messages=[], tools=_ARRAY_TOOLS
    )
    result = parser.extract_tool_calls(
        _ARRAY_WITH_JSON_BOOL_OUTPUT, request=request
    )

    assert result.tools_called
    args = json.loads(result.tool_calls[0].function.arguments)
    assert isinstance(args["items"], list), (
        f"Array with JSON bool was not parsed as list: "
        f"{type(args['items']).__name__} = {args['items']!r}"
    )
    assert args["items"] == ["a", "b", 1, True]


# ---------------------------------------------------------------------------
# Speculative decoding: two complete tool calls in a single streaming delta.
# Both parsers must emit both tool calls, not drop the second.
# ---------------------------------------------------------------------------

_WEATHER_TOOLS = [
    ChatCompletionToolsParam(
        type="function",
        function={
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        },
    )
]

_TWO_TOOL_CALLS_IN_ONE_CHUNK = (
    "<tool_call>\n<function=get_weather>\n"
    "<parameter=city>\nParis\n</parameter>\n"
    "</function>\n</tool_call>\n"
    "<tool_call>\n<function=get_weather>\n"
    "<parameter=city>\nLondon\n</parameter>\n"
    "</function>\n</tool_call>"
)


def test_two_tool_calls_in_one_streaming_chunk(qwen3_tokenizer, parser_cls):
    """Speculative decoding flushes can deliver several full
    ``<tool_call>...</tool_call>`` blocks in a single delta. Both must be
    emitted; dropping the second one is a regression.
    """
    parser = parser_cls(qwen3_tokenizer, tools=_WEATHER_TOOLS)
    request = ChatCompletionRequest(
        model=MODEL, messages=[], tools=_WEATHER_TOOLS
    )
    reconstructor = run_tool_extraction_streaming(
        parser,
        [_TWO_TOOL_CALLS_IN_ONE_CHUNK],
        request,
        assert_one_tool_per_delta=False,
    )
    assert len(reconstructor.tool_calls) == 2, (
        f"Expected 2 tool calls in one delta, got "
        f"{len(reconstructor.tool_calls)}"
    )
    args0 = json.loads(reconstructor.tool_calls[0].function.arguments)
    args1 = json.loads(reconstructor.tool_calls[1].function.arguments)
    assert args0 == {"city": "Paris"}
    assert args1 == {"city": "London"}
