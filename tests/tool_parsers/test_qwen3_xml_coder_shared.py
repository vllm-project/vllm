# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Shared tests for the Qwen3 XML and Coder tool parsers.

These tests cover behaviour that BOTH parsers must implement identically.
Each test runs twice — once against ``Qwen3XMLToolParser`` and once against
``Qwen3CoderToolParser`` — via the ``parser_cls`` fixture.  Tests that
target streaming-mode-specific quirks of one parser only stay in their
parser-specific file (``test_qwen3xml_tool_parser.py`` or
``test_qwen3coder_tool_parser.py``).
"""

import json
from collections.abc import Generator

import pytest
from openai.types.responses.function_tool import FunctionTool
from xgrammar import StructuralTag

from tests.tool_parsers.utils import run_tool_extraction_streaming
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedFunction,
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
    ChatCompletionToolsParam,
)
from vllm.entrypoints.openai.engine.protocol import (
    DeltaMessage,
    FunctionCall,
    ToolCall,
)
from vllm.tokenizers import TokenizerLike, get_tokenizer
from vllm.tokenizers.detokenizer_utils import detokenize_incrementally
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


WEATHER_PARAMS = {
    "type": "object",
    "properties": {
        "city": {"type": "string", "description": "The city name"},
        "state": {"type": "string", "description": "The state code"},
        "unit": {"type": "string", "enum": ["fahrenheit", "celsius"]},
    },
    "required": ["city", "state"],
}

AREA_PARAMS = {
    "type": "object",
    "properties": {
        "shape": {"type": "string"},
        "dimensions": {"type": "object"},
        "precision": {"type": "integer"},
    },
}


@pytest.fixture(params=["chat_completion", "responses_api"])
def sample_tools(request):
    if request.param == "chat_completion":
        return [
            ChatCompletionToolsParam(
                type="function",
                function={
                    "name": "get_current_weather",
                    "description": "Get the current weather",
                    "parameters": WEATHER_PARAMS,
                },
            ),
            ChatCompletionToolsParam(
                type="function",
                function={
                    "name": "calculate_area",
                    "description": "Calculate area of a shape",
                    "parameters": AREA_PARAMS,
                },
            ),
        ]
    else:
        return [
            FunctionTool(
                type="function",
                name="get_current_weather",
                description="Get the current weather",
                parameters=WEATHER_PARAMS,
            ),
            FunctionTool(
                type="function",
                name="calculate_area",
                description="Calculate area of a shape",
                parameters=AREA_PARAMS,
            ),
        ]


@pytest.fixture
def parser(parser_cls, qwen3_tokenizer, sample_tools):
    return parser_cls(qwen3_tokenizer, tools=sample_tools)


def _as_chat_completion_tools(
    tools: list[ChatCompletionToolsParam | FunctionTool],
) -> list[ChatCompletionToolsParam]:
    normalized: list[ChatCompletionToolsParam] = []
    for tool in tools:
        if isinstance(tool, ChatCompletionToolsParam):
            normalized.append(tool)
        else:
            normalized.append(
                ChatCompletionToolsParam(
                    type="function",
                    function={
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                )
            )
    return normalized


def assert_tool_calls(
    actual_tool_calls: list[ToolCall], expected_tool_calls: list[ToolCall]
):
    assert len(actual_tool_calls) == len(expected_tool_calls)
    for actual_tool_call, expected_tool_call in zip(
        actual_tool_calls, expected_tool_calls
    ):
        assert actual_tool_call.type == "function"
        assert actual_tool_call.function.name == expected_tool_call.function.name
        assert json.loads(actual_tool_call.function.arguments) == json.loads(
            expected_tool_call.function.arguments
        )


def stream_delta_message_generator(
    parser,
    tokenizer: TokenizerLike,
    model_output: str,
    request: ChatCompletionRequest | None = None,
) -> Generator[DeltaMessage, None, None]:
    all_token_ids = tokenizer.encode(model_output, add_special_tokens=False)

    previous_text = ""
    previous_tokens = None
    prefix_offset = 0
    read_offset = 0
    for i, delta_token in enumerate(all_token_ids):
        delta_token_ids = [delta_token]
        previous_token_ids = all_token_ids[:i]
        current_token_ids = all_token_ids[: i + 1]

        (new_tokens, delta_text, new_prefix_offset, new_read_offset) = (
            detokenize_incrementally(
                tokenizer=tokenizer,
                all_input_ids=current_token_ids,
                prev_tokens=previous_tokens,
                prefix_offset=prefix_offset,
                read_offset=read_offset,
                skip_special_tokens=False,
                spaces_between_special_tokens=True,
            )
        )

        current_text = previous_text + delta_text

        delta_message = parser.extract_tool_calls_streaming(
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
            request=request,
        )
        if delta_message:
            yield delta_message

        previous_text = current_text
        previous_tokens = (
            previous_tokens + new_tokens if previous_tokens else new_tokens
        )
        prefix_offset = new_prefix_offset
        read_offset = new_read_offset


# ---------------------------------------------------------------------------
# Basic extraction
# ---------------------------------------------------------------------------


def test_extract_tool_calls_no_tools(parser):
    model_output = "This is a test response without any tool calls"
    extracted_tool_calls = parser.extract_tool_calls(model_output, request=None)
    assert not extracted_tool_calls.tools_called
    assert extracted_tool_calls.tool_calls == []
    assert extracted_tool_calls.content == model_output


_EXTRACT_CASES = [
    (
        """<tool_call>
<function=get_current_weather>
<parameter=city>
Dallas
</parameter>
<parameter=state>
TX
</parameter>
<parameter=unit>
fahrenheit
</parameter>
</function>
</tool_call>""",
        [
            ToolCall(
                function=FunctionCall(
                    name="get_current_weather",
                    arguments=json.dumps(
                        {"city": "Dallas", "state": "TX", "unit": "fahrenheit"}
                    ),
                )
            )
        ],
        None,
    ),
    (
        """Sure! Let me check the weather for you.<tool_call>
<function=get_current_weather>
<parameter=city>
Dallas
</parameter>
<parameter=state>
TX
</parameter>
<parameter=unit>
fahrenheit
</parameter>
</function>
</tool_call>""",
        [
            ToolCall(
                function=FunctionCall(
                    name="get_current_weather",
                    arguments=json.dumps(
                        {"city": "Dallas", "state": "TX", "unit": "fahrenheit"}
                    ),
                )
            )
        ],
        "Sure! Let me check the weather for you.",
    ),
    (
        """<tool_call>
<function=calculate_area>
<parameter=shape>
rectangle
</parameter>
<parameter=dimensions>
{"width": 10,
 "height": 20}
</parameter>
<parameter=precision>
2
</parameter>
</function>
</tool_call>""",
        [
            ToolCall(
                function=FunctionCall(
                    name="calculate_area",
                    arguments=json.dumps(
                        {
                            "shape": "rectangle",
                            "dimensions": {"width": 10, "height": 20},
                            "precision": 2,
                        }
                    ),
                )
            )
        ],
        None,
    ),
    (
        """<tool_call>
<function=get_current_weather>
<parameter=city>
Dallas
</parameter>
<parameter=state>
TX
</parameter>
<parameter=unit>
fahrenheit
</parameter>
</function>
</tool_call>
<tool_call>
<function=get_current_weather>
<parameter=city>
Orlando
</parameter>
<parameter=state>
FL
</parameter>
<parameter=unit>
fahrenheit
</parameter>
</function>
</tool_call>""",
        [
            ToolCall(
                function=FunctionCall(
                    name="get_current_weather",
                    arguments=json.dumps(
                        {"city": "Dallas", "state": "TX", "unit": "fahrenheit"}
                    ),
                )
            ),
            ToolCall(
                function=FunctionCall(
                    name="get_current_weather",
                    arguments=json.dumps(
                        {"city": "Orlando", "state": "FL", "unit": "fahrenheit"}
                    ),
                )
            ),
        ],
        "\n",
    ),
    (
        """Let me calculate that area for you.<tool_call>
<function=calculate_area>
<parameter=shape>
circle
</parameter>
<parameter=dimensions>
{"radius": 15.5}
</parameter>
<parameter=precision>
3
</parameter>
</function>
</tool_call>""",
        [
            ToolCall(
                function=FunctionCall(
                    name="calculate_area",
                    arguments=json.dumps(
                        {
                            "shape": "circle",
                            "dimensions": {"radius": 15.5},
                            "precision": 3,
                        }
                    ),
                )
            )
        ],
        "Let me calculate that area for you.",
    ),
]

_EXTRACT_IDS = [
    "single_tool",
    "single_tool_with_content",
    "single_tool_multiline_param",
    "parallel_tools",
    "tool_with_typed_params",
]


@pytest.mark.parametrize(
    ids=_EXTRACT_IDS,
    argnames=["model_output", "expected_tool_calls", "expected_content"],
    argvalues=_EXTRACT_CASES,
)
def test_extract_tool_calls(
    parser, model_output, expected_tool_calls, expected_content
):
    request = ChatCompletionRequest(model=MODEL, messages=[])
    extracted_tool_calls = parser.extract_tool_calls(model_output, request=request)
    assert extracted_tool_calls.tools_called
    assert_tool_calls(extracted_tool_calls.tool_calls, expected_tool_calls)
    # Both ``None`` and ``""`` are acceptable when the expected content is
    # only whitespace — the two parsers differ on whether they preserve the
    # newline that separates parallel tool-call blocks.
    actual_content = extracted_tool_calls.content
    if expected_content and expected_content.strip():
        assert actual_content == expected_content
    else:
        assert (actual_content or "").strip() == (expected_content or "").strip()


def test_extract_tool_calls_fallback_no_tags(parser):
    """Test fallback parsing when XML tags are missing."""
    model_output = """<function=get_current_weather>
<parameter=city>
Dallas
</parameter>
<parameter=state>
TX
</parameter>
</function>"""
    request = ChatCompletionRequest(model=MODEL, messages=[])
    extracted_tool_calls = parser.extract_tool_calls(model_output, request=request)
    assert extracted_tool_calls.tools_called
    assert len(extracted_tool_calls.tool_calls) == 1
    assert extracted_tool_calls.tool_calls[0].function.name == "get_current_weather"


# ---------------------------------------------------------------------------
# Type conversion
# ---------------------------------------------------------------------------


def test_extract_tool_calls_type_conversion(qwen3_tokenizer, parser_cls):
    """Test parameter type conversion based on tool schema."""
    tools = [
        ChatCompletionToolsParam(
            type="function",
            function={
                "name": "test_types",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "int_param": {"type": "integer"},
                        "float_param": {"type": "float"},
                        "bool_param": {"type": "boolean"},
                        "str_param": {"type": "string"},
                        "obj_param": {"type": "object"},
                    },
                },
            },
        )
    ]

    model_output = """<tool_call>
<function=test_types>
<parameter=int_param>
42
</parameter>
<parameter=float_param>
3.14
</parameter>
<parameter=bool_param>
true
</parameter>
<parameter=str_param>
hello world
</parameter>
<parameter=obj_param>
{"key": "value"}
</parameter>
</function>
</tool_call>"""

    parser_inst = parser_cls(qwen3_tokenizer, tools=tools)
    request = ChatCompletionRequest(model=MODEL, messages=[], tools=tools)
    extracted_tool_calls = parser_inst.extract_tool_calls(model_output, request=request)

    args = json.loads(extracted_tool_calls.tool_calls[0].function.arguments)
    assert args["int_param"] == 42
    assert args["float_param"] == 3.14
    assert args["bool_param"] is True
    assert args["str_param"] == "hello world"
    assert args["obj_param"] == {"key": "value"}


def test_extract_tool_calls_complex_type_with_single_quote(qwen3_tokenizer, parser_cls):
    """Object parameter expressed as a Python repr (single quotes)."""
    tools = [
        ChatCompletionToolsParam(
            type="function",
            function={
                "name": "test_types",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "int_param": {"type": "integer"},
                        "float_param": {"type": "float"},
                        "bool_param": {"type": "boolean"},
                        "str_param": {"type": "string"},
                        "obj_param": {"type": "object"},
                    },
                },
            },
        )
    ]

    model_output = """<tool_call>
<function=test_types>
<parameter=obj_param>
{'key': 'value'}
</parameter>
</function>
</tool_call>"""

    parser_inst = parser_cls(qwen3_tokenizer, tools=tools)
    request = ChatCompletionRequest(model=MODEL, messages=[], tools=tools)
    extracted_tool_calls = parser_inst.extract_tool_calls(model_output, request=request)

    args = json.loads(extracted_tool_calls.tool_calls[0].function.arguments)
    assert args["obj_param"] == {"key": "value"}


# ---------------------------------------------------------------------------
# Streaming extraction
# ---------------------------------------------------------------------------


_STREAMING_CASES = [
    ("This is a test without tools", [], "This is a test without tools"),
] + _EXTRACT_CASES

_STREAMING_IDS = ["no_tools"] + _EXTRACT_IDS


@pytest.mark.parametrize(
    ids=_STREAMING_IDS,
    argnames=["model_output", "expected_tool_calls", "expected_content"],
    argvalues=_STREAMING_CASES,
)
def test_extract_tool_calls_streaming(
    parser,
    qwen3_tokenizer,
    model_output,
    expected_tool_calls,
    expected_content,
):
    """Test incremental streaming behavior including typed parameters."""
    request = ChatCompletionRequest(model=MODEL, messages=[])

    other_content = ""
    tool_states = {}

    for delta_message in stream_delta_message_generator(
        parser, qwen3_tokenizer, model_output, request
    ):
        assert not delta_message.role

        if delta_message.content:
            other_content += delta_message.content

        if delta_message.tool_calls:
            for tool_call in delta_message.tool_calls:
                idx = tool_call.index

                if idx not in tool_states:
                    tool_states[idx] = {
                        "id": None,
                        "name": None,
                        "arguments": "",
                        "type": None,
                    }

                if tool_call.id:
                    tool_states[idx]["id"] = tool_call.id

                if tool_call.type:
                    assert tool_call.type == "function"
                    tool_states[idx]["type"] = tool_call.type

                if tool_call.function:
                    if tool_call.function.name:
                        assert tool_states[idx]["name"] is None
                        tool_states[idx]["name"] = tool_call.function.name

                    if tool_call.function.arguments is not None:
                        tool_states[idx]["arguments"] += tool_call.function.arguments

    # Be tolerant about whitespace-only deltas between parallel tool calls;
    # see ``test_extract_tool_calls`` for the same reasoning.
    if expected_content and expected_content.strip():
        assert other_content == expected_content
    else:
        assert other_content.strip() == (expected_content or "").strip()
    assert len(tool_states) == len(expected_tool_calls)
    assert len(parser.prev_tool_call_arr) == len(expected_tool_calls)

    for idx, expected_tool in enumerate(expected_tool_calls):
        state = tool_states[idx]
        assert state["id"] is not None
        assert state["type"] == "function"
        assert state["name"] == expected_tool.function.name

        arguments_str = state["arguments"]
        assert arguments_str is not None
        actual_args = json.loads(arguments_str)
        expected_args = json.loads(expected_tool.function.arguments)
        assert actual_args == expected_args


def test_extract_tool_calls_missing_closing_parameter_tag(parser):
    """Test handling of missing closing </parameter> tag."""
    model_output = """Let me check the weather for you:
<tool_call>
<function=get_current_weather>
<parameter=city>
Dallas
<parameter=state>
TX
</parameter>
<parameter=unit>
fahrenheit
</parameter>
</function>
</tool_call>"""

    request = ChatCompletionRequest(model=MODEL, messages=[])
    extracted_tool_calls = parser.extract_tool_calls(model_output, request=request)

    assert extracted_tool_calls.tools_called
    assert len(extracted_tool_calls.tool_calls) == 1
    assert extracted_tool_calls.tool_calls[0].function.name == "get_current_weather"
    args = json.loads(extracted_tool_calls.tool_calls[0].function.arguments)
    assert "city" in args
    assert args["city"] == "Dallas"
    assert args["state"] == "TX"
    assert args["unit"] == "fahrenheit"
    assert "Let me check the weather for you:" in extracted_tool_calls.content


def test_extract_tool_calls_streaming_missing_closing_tag(parser, qwen3_tokenizer):
    """Streaming with missing closing </parameter> tag."""
    model_output = """Let me check the weather for you:
<tool_call>
<function=get_current_weather>
<parameter=city>
Dallas
<parameter=state>
TX
</parameter>
<parameter=unit>
fahrenheit
</parameter>
</function>
</tool_call>"""

    request = ChatCompletionRequest(model=MODEL, messages=[])
    other_content = ""
    tool_states = {}

    for delta_message in stream_delta_message_generator(
        parser, qwen3_tokenizer, model_output, request
    ):
        if delta_message.content:
            other_content += delta_message.content

        if delta_message.tool_calls:
            for tool_call in delta_message.tool_calls:
                idx = tool_call.index
                if idx not in tool_states:
                    tool_states[idx] = {
                        "id": None,
                        "name": None,
                        "arguments": "",
                        "type": None,
                    }
                if tool_call.id:
                    tool_states[idx]["id"] = tool_call.id
                if tool_call.type:
                    assert tool_call.type == "function"
                    tool_states[idx]["type"] = tool_call.type
                if tool_call.function:
                    if tool_call.function.name:
                        tool_states[idx]["name"] = tool_call.function.name
                    if tool_call.function.arguments is not None:
                        tool_states[idx]["arguments"] += tool_call.function.arguments

    assert "Let me check the weather for you:" in other_content
    assert len(tool_states) == 1
    assert len(parser.prev_tool_call_arr) == 1

    state = tool_states[0]
    assert state["id"] is not None
    assert state["type"] == "function"
    assert state["name"] == "get_current_weather"
    args = json.loads(state["arguments"])
    assert args["city"] == "Dallas"
    assert args["state"] == "TX"
    assert args["unit"] == "fahrenheit"


def test_extract_tool_calls_streaming_incremental(parser, qwen3_tokenizer):
    """Test that streaming is truly incremental."""
    model_output = """I'll check the weather.<tool_call>
<function=get_current_weather>
<parameter=city>
Dallas
</parameter>
<parameter=state>
TX
</parameter>
</function>
</tool_call>"""

    request = ChatCompletionRequest(model=MODEL, messages=[])
    chunks = []
    for delta_message in stream_delta_message_generator(
        parser, qwen3_tokenizer, model_output, request
    ):
        chunks.append(delta_message)

    assert len(chunks) > 3
    assert chunks[0].content is not None
    assert chunks[0].tool_calls is None or chunks[0].tool_calls == []

    header_found = False
    for chunk in chunks:
        if chunk.tool_calls and chunk.tool_calls[0].id:
            header_found = True
            assert chunk.tool_calls[0].function.name == "get_current_weather"
            assert chunk.tool_calls[0].type == "function"
            # XML emits an empty arguments string with the header; Coder
            # emits the opening "{" with the header.  Both are valid.
            assert chunk.tool_calls[0].function.arguments in ("", "{")
            break
    assert header_found

    arg_chunks = []
    for chunk in chunks:
        if chunk.tool_calls and chunk.tool_calls[0].function.arguments:
            arg_chunks.append(chunk.tool_calls[0].function.arguments)

    assert len(arg_chunks) > 1
    full_args = "".join(arg_chunks)
    parsed_args = json.loads(full_args)
    assert parsed_args["city"] == "Dallas"
    assert parsed_args["state"] == "TX"


# ---------------------------------------------------------------------------
# Robustness regressions
# ---------------------------------------------------------------------------


def test_malformed_xml_no_gt_delimiter(parser):
    """Regression: malformed XML without '>' must not crash (PR #36774)."""
    model_output = (
        "<tool_call>\n"
        "<function=get_current_weather\n"
        "<parameter=city>Dallas</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )
    request = ChatCompletionRequest(model=MODEL, messages=[])
    result = parser.extract_tool_calls(model_output, request=request)
    assert result is not None
    assert isinstance(result.tool_calls, list)
    assert all(tc is not None for tc in result.tool_calls)


def test_none_tool_calls_filtered(parser):
    """Regression: None tool calls filtered from output (PR #36774)."""
    model_output = (
        "<tool_call>\n"
        "<function=bad_func_no_gt\n"
        "</function>\n"
        "</tool_call>\n"
        "<tool_call>\n"
        "<function=get_current_weather>\n"
        "<parameter=city>Dallas</parameter>\n"
        "<parameter=state>TX</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )
    request = ChatCompletionRequest(model=MODEL, messages=[])
    result = parser.extract_tool_calls(model_output, request=request)
    assert all(tc is not None for tc in result.tool_calls)
    assert result.tools_called
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].function.name == "get_current_weather"
    args = json.loads(result.tool_calls[0].function.arguments)
    assert args["city"] == "Dallas"
    assert args["state"] == "TX"


def test_streaming_multi_param_single_chunk(parser):
    """Regression: speculative decode delivering multiple params at once
    (PR #35615)."""
    request = ChatCompletionRequest(model=MODEL, messages=[])

    deltas = [
        "<tool_call>",
        "\n<function=get_current_weather>",
        "\n",
        # This single delta delivers all three parameters at once
        "<parameter=city>\nDallas\n</parameter>"
        "\n<parameter=state>\nTX\n</parameter>"
        "\n<parameter=unit>\nfahrenheit\n</parameter>",
        "\n</function>",
        "\n</tool_call>",
    ]

    reconstructor = run_tool_extraction_streaming(
        parser,
        deltas,
        request,
        assert_one_tool_per_delta=False,
    )

    assert len(reconstructor.tool_calls) == 1
    args = json.loads(reconstructor.tool_calls[0].function.arguments)
    assert args["city"] == "Dallas"
    assert args["state"] == "TX"
    assert args["unit"] == "fahrenheit"


def test_no_double_serialization_string_args(qwen3_tokenizer, parser_cls):
    """Regression: string arguments must not be double-serialized
    (PR #35615)."""
    tools = [
        ChatCompletionToolsParam(
            type="function",
            function={
                "name": "greet",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {"type": "string"},
                    },
                },
            },
        )
    ]

    model_output = (
        "<tool_call>\n"
        "<function=greet>\n"
        "<parameter=message>hello world</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )

    parser_inst = parser_cls(qwen3_tokenizer, tools=tools)
    request = ChatCompletionRequest(model=MODEL, messages=[], tools=tools)
    result = parser_inst.extract_tool_calls(model_output, request=request)

    assert result.tools_called
    assert len(result.tool_calls) == 1
    raw_arguments = result.tool_calls[0].function.arguments
    args = json.loads(raw_arguments)
    assert args["message"] == "hello world"
    assert '\\"hello world\\"' not in raw_arguments


def test_extract_tool_calls_streaming_speculative_decode_loss(parser):
    """If the parser hasn't started JSON yet and the delta contains the
    parameters AND the end of the tool call, the parser should not just
    return '{' and lose the parameters.
    """
    request = ChatCompletionRequest(model="test", messages=[])

    text1 = "<tool_call>\n<function=test>\n"
    parser.extract_tool_calls_streaming("", text1, text1, [], [1], [1], request)

    delta_str = "<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>"
    text2 = text1 + delta_str
    delta2 = parser.extract_tool_calls_streaming(
        text1, text2, delta_str, [1], [1, 2], [2], request
    )

    assert delta2 is not None
    assert delta2.tool_calls is not None
    assert len(delta2.tool_calls) == 1
    args = delta2.tool_calls[0].function.arguments
    assert "Paris" in args, f"Arguments lost! Got: {args}"


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
        f"String parameter 'null' was converted incorrectly. Got: {args.get('query')!r}"
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
    request = ChatCompletionRequest(model=MODEL, messages=[], tools=_ANYOF_OBJECT_TOOLS)
    result = parser.extract_tool_calls(_ANYOF_OBJECT_OUTPUT, request=request)

    assert result.tools_called
    args = json.loads(result.tool_calls[0].function.arguments)
    assert isinstance(args["data"], dict), (
        f"anyOf object param was double-encoded: data={args['data']!r}"
    )
    assert args["data"] == {"key": "value", "count": 42}


def test_anyof_object_param_not_double_encoded_streaming(qwen3_tokenizer, parser_cls):
    parser = parser_cls(qwen3_tokenizer, tools=_ANYOF_OBJECT_TOOLS)
    request = ChatCompletionRequest(model=MODEL, messages=[], tools=_ANYOF_OBJECT_TOOLS)
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
        f"anyOf object param was double-encoded in streaming: data={args['data']!r}"
    )


# ---------------------------------------------------------------------------
# anyOf / nullable (Pydantic v2 Optional[T]) type resolution.
# Both parsers extract the first non-null type from the anyOf union.
# ---------------------------------------------------------------------------

_ANYOF_TYPES_TOOLS = [
    ChatCompletionToolsParam(
        type="function",
        function={
            "name": "test_anyof",
            "parameters": {
                "type": "object",
                "properties": {
                    "anyof_int": {
                        "anyOf": [{"type": "integer"}, {"type": "null"}],
                        "default": 5,
                    },
                    "anyof_str": {
                        "anyOf": [{"type": "string"}, {"type": "null"}],
                    },
                    "anyof_array": {
                        "anyOf": [
                            {"type": "array", "items": {"type": "string"}},
                            {"type": "null"},
                        ],
                    },
                    "anyof_obj": {
                        "anyOf": [{"type": "object"}, {"type": "null"}],
                    },
                    "type_as_array": {
                        "type": ["integer", "null"],
                    },
                    "multi_non_null": {
                        "anyOf": [
                            {"type": "string"},
                            {"type": "integer"},
                            {"type": "null"},
                        ],
                    },
                },
            },
        },
    )
]

_ANYOF_TYPES_OUTPUT = (
    "<tool_call>\n"
    "<function=test_anyof>\n"
    "<parameter=anyof_int>5</parameter>\n"
    "<parameter=anyof_str>hello</parameter>\n"
    '<parameter=anyof_array>["a", "b", "c"]</parameter>\n'
    '<parameter=anyof_obj>{"key": "value"}</parameter>\n'
    "<parameter=type_as_array>42</parameter>\n"
    "<parameter=multi_non_null>some text</parameter>\n"
    "</function>\n"
    "</tool_call>"
)


def test_extract_tool_calls_anyof_type_conversion(qwen3_tokenizer, parser_cls):
    """anyOf nullable schemas (Pydantic v2 ``Optional[T]``) must resolve to
    the first non-null type and apply the matching conversion: int(),
    list/dict via json, string passthrough.
    """
    parser = parser_cls(qwen3_tokenizer, tools=_ANYOF_TYPES_TOOLS)
    request = ChatCompletionRequest(model=MODEL, messages=[], tools=_ANYOF_TYPES_TOOLS)
    result = parser.extract_tool_calls(_ANYOF_TYPES_OUTPUT, request=request)

    assert result.tools_called
    args = json.loads(result.tool_calls[0].function.arguments)
    assert args["anyof_int"] == 5
    assert isinstance(args["anyof_int"], int)
    assert args["anyof_str"] == "hello"
    assert isinstance(args["anyof_str"], str)
    assert args["anyof_array"] == ["a", "b", "c"]
    assert isinstance(args["anyof_array"], list)
    assert args["anyof_obj"] == {"key": "value"}
    assert isinstance(args["anyof_obj"], dict)
    # JSON-Schema list-form type {"type": ["integer", "null"]} → int
    assert args["type_as_array"] == 42
    assert isinstance(args["type_as_array"], int)
    # anyOf[string, integer, null] → first non-null type is string
    assert args["multi_non_null"] == "some text"
    assert isinstance(args["multi_non_null"], str)


_ANYOF_STREAMING_TOOLS = [
    ChatCompletionToolsParam(
        type="function",
        function={
            "name": "search_web",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "anyOf": [{"type": "string"}, {"type": "null"}],
                    },
                    "count": {
                        "anyOf": [{"type": "integer"}, {"type": "null"}],
                        "default": 5,
                    },
                    "verbose": {
                        "anyOf": [{"type": "boolean"}, {"type": "null"}],
                    },
                },
            },
        },
    )
]

_ANYOF_STREAMING_OUTPUT = (
    "<tool_call>\n"
    "<function=search_web>\n"
    "<parameter=query>vllm tool parser</parameter>\n"
    "<parameter=count>10</parameter>\n"
    "<parameter=verbose>true</parameter>\n"
    "</function>\n"
    "</tool_call>"
)


def test_extract_tool_calls_anyof_type_conversion_streaming(
    qwen3_tokenizer, parser_cls
):
    """Streaming e2e for anyOf nullable schemas: string/int/bool types must
    be resolved through the incremental pipeline for both parsers.
    """
    parser = parser_cls(qwen3_tokenizer, tools=_ANYOF_STREAMING_TOOLS)
    request = ChatCompletionRequest(
        model=MODEL, messages=[], tools=_ANYOF_STREAMING_TOOLS
    )
    reconstructor = run_tool_extraction_streaming(
        parser,
        _ANYOF_STREAMING_OUTPUT,
        request,
        assert_one_tool_per_delta=False,
    )
    assert len(reconstructor.tool_calls) == 1
    assert reconstructor.tool_calls[0].function.name == "search_web"
    args = json.loads(reconstructor.tool_calls[0].function.arguments)
    assert args["query"] == "vllm tool parser"
    assert isinstance(args["query"], str)
    assert args["count"] == 10
    assert isinstance(args["count"], int)
    assert args["verbose"] is True
    assert isinstance(args["verbose"], bool)


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

_XML_TAGS_IN_CONTENT = (
    "char_deltas = [\n"
    '    "<tool_call>\\n",\n'
    '    "<parameter=query>\\n",\n'
    '    "\\n</parameter>\\n",\n'
    '    "</function>\\n",\n'
    "]\n"
)

_WRITE_FILE_XML_TAGS_OUTPUT = (
    "<tool_call>\n"
    "<function=write_file>\n"
    "<parameter=path>\ntest.py\n</parameter>\n"
    f"<parameter=content>\n{_XML_TAGS_IN_CONTENT}</parameter>\n"
    "</function>\n"
    "</tool_call>\n"
)


def test_content_with_xml_structural_tags_nonstreaming(qwen3_tokenizer, parser_cls):
    """Non-streaming: a string param whose value embeds <tool_call>,
    <parameter=...>, </parameter>, </function> as literal text must be
    extracted intact, with no spurious extra params being created from
    the embedded tags.
    """
    parser = parser_cls(qwen3_tokenizer, tools=_WRITE_FILE_TOOLS)
    request = ChatCompletionRequest(model=MODEL, messages=[], tools=_WRITE_FILE_TOOLS)
    result = parser.extract_tool_calls(_WRITE_FILE_XML_TAGS_OUTPUT, request=request)

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


def test_content_with_xml_structural_tags_streaming(qwen3_tokenizer, parser_cls):
    """Streaming variant: pre-formed chunks, full content in one delta."""
    parser = parser_cls(qwen3_tokenizer, tools=_WRITE_FILE_TOOLS)
    request = ChatCompletionRequest(model=MODEL, messages=[], tools=_WRITE_FILE_TOOLS)
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


def test_content_with_param_like_lines_nonstreaming(qwen3_tokenizer, parser_cls):
    """Non-streaming: ``</parameter>`` and ``<parameter=NAME>`` on their
    own lines inside a string value must not terminate the parameter
    early.  Requires schema-based filtering so that ``new_string`` (not a
    real parameter of write_file) is treated as literal text.
    """
    parser = parser_cls(qwen3_tokenizer, tools=_WRITE_FILE_TOOLS)
    request = ChatCompletionRequest(model=MODEL, messages=[], tools=_WRITE_FILE_TOOLS)
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
    request = ChatCompletionRequest(model=MODEL, messages=[], tools=_WRITE_FILE_TOOLS)
    char_deltas = [
        "<tool_call>\n",
        "<function=write_file>\n",
        "<parameter=path>\ntest_template.py\n</parameter>\n",
        '<parameter=content>\nTOOL_CALL_TEMPLATE = """\n',
        "</parameter>\n",  # literal close — alone in its delta
        "<parameter=new_string>\n",  # literal new-param line
        "#!/usr/bin/env python3\n",
        "</parameter>\n",  # second literal close
        '"""\n',
        "</parameter>\n",  # REAL close of content
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
    request = ChatCompletionRequest(model=MODEL, messages=[], tools=_ARRAY_TOOLS)
    result = parser.extract_tool_calls(_ARRAY_WITH_JSON_BOOL_OUTPUT, request=request)

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
    request = ChatCompletionRequest(model=MODEL, messages=[], tools=_WEATHER_TOOLS)
    reconstructor = run_tool_extraction_streaming(
        parser,
        [_TWO_TOOL_CALLS_IN_ONE_CHUNK],
        request,
        assert_one_tool_per_delta=False,
    )
    assert len(reconstructor.tool_calls) == 2, (
        f"Expected 2 tool calls in one delta, got {len(reconstructor.tool_calls)}"
    )
    args0 = json.loads(reconstructor.tool_calls[0].function.arguments)
    args1 = json.loads(reconstructor.tool_calls[1].function.arguments)
    assert args0 == {"city": "Paris"}
    assert args1 == {"city": "London"}


# ---------------------------------------------------------------------------
# Trailing free text after the LAST </tool_call> in the SAME delta (MTP /
# speculative decoding). The text must be emitted as content; dropping it
# silently is a regression.
# ---------------------------------------------------------------------------


def test_python_none_value_for_nullable_int(qwen3_tokenizer, parser_cls):
    """A Qwen3.5-trained model emits Python ``None`` (not ``null``) for a
    nullable non-string parameter, because the Qwen3.5 chat template
    renders ``args_value | string`` for non-container types — turning a
    null arg from a previous tool call into the literal "None" in the
    prompt. The model then learns to generate the same "None" verbatim.

    The parser must recognise this and convert "None" to JSON null,
    just like it already does for the literal "null" emitted by
    Qwen3.6-trained models.
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
                            "anyOf": [
                                {"type": "integer"},
                                {"type": "null"},
                            ],
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
        "<parameter=count>None</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )
    request = ChatCompletionRequest(model=MODEL, messages=[], tools=tools)
    result = parser.extract_tool_calls(model_output, request=request)

    assert result.tools_called
    args = json.loads(result.tool_calls[0].function.arguments)
    assert args["count"] is None, (
        f"Python repr None was not converted to JSON null. Got: {args['count']!r}"
    )


def test_streaming_two_tool_calls_plus_trailing_text_one_delta(
    qwen3_tokenizer, parser_cls
):
    """MTP: a single delta delivers tool 1 + tool 2 + trailing free text.
    Both tool calls must be emitted AND the trailing text must surface as
    content in the same delta — not be silently dropped.
    """
    parser = parser_cls(qwen3_tokenizer, tools=_WEATHER_TOOLS)
    request = ChatCompletionRequest(model=MODEL, messages=[], tools=_WEATHER_TOOLS)
    deltas = [
        _TWO_TOOL_CALLS_IN_ONE_CHUNK + "\nAll done!",
    ]
    reconstructor = run_tool_extraction_streaming(
        parser, deltas, request, assert_one_tool_per_delta=False
    )
    assert len(reconstructor.tool_calls) == 2, (
        f"Expected 2 tool calls, got {len(reconstructor.tool_calls)}"
    )
    assert "All done!" in reconstructor.other_content, (
        f"Trailing text after the second tool call was dropped. "
        f"Got content: {reconstructor.other_content!r}"
    )


def test_streaming_trailing_text_with_final_close_in_same_delta(
    qwen3_tokenizer, parser_cls
):
    """MTP / speculative decoding can deliver the closing ``</tool_call>``
    together with trailing free text in a single delta.  The text after
    the close must be emitted as content rather than being silently
    consumed by the parser's "advance to next tool" logic.
    """
    parser = parser_cls(qwen3_tokenizer, tools=_WEATHER_TOOLS)
    request = ChatCompletionRequest(model=MODEL, messages=[], tools=_WEATHER_TOOLS)
    deltas = [
        # Build up the tool call up to and including </function>.
        "<tool_call>\n<function=get_weather>\n"
        "<parameter=city>Paris</parameter>\n</function>",
        # Then deliver </tool_call> + trailing text in ONE delta.
        "\n</tool_call>\nI hope this helps!",
    ]
    reconstructor = run_tool_extraction_streaming(
        parser, deltas, request, assert_one_tool_per_delta=False
    )
    assert len(reconstructor.tool_calls) == 1
    assert "I hope this helps!" in reconstructor.other_content, (
        f"Trailing text after </tool_call> was dropped. "
        f"Got content: {reconstructor.other_content!r}"
    )


# ---------------------------------------------------------------------------
# Parameter value containing a literal ``<parameter=NAME>`` whose NAME IS
# itself a real parameter of the same tool.  The schema-based filter cannot
# rule the literal out by name, so a stronger heuristic is required (e.g.
# the literal does not pair with a structural ``</parameter>`` followed by
# another structural delimiter).  This is the exact pattern that breaks
# qwen-code WriteFile when the file being written is itself a parser test
# fixture.
# ---------------------------------------------------------------------------

_CONTENT_WITH_REAL_PARAM_NAME_LITERAL = (
    'doc = """\n<parameter=path>\nliteral/value\n</parameter>\n"""\n'
)

_REAL_PARAM_NAME_LITERAL_OUTPUT = (
    "<tool_call>\n"
    "<function=write_file>\n"
    "<parameter=path>\nfixture.py\n</parameter>\n"
    f"<parameter=content>\n{_CONTENT_WITH_REAL_PARAM_NAME_LITERAL}</parameter>\n"
    "</function>\n"
    "</tool_call>"
)


def test_content_with_real_param_name_literal_nonstreaming(qwen3_tokenizer, parser_cls):
    """Non-streaming: parameter ``content`` value embeds
    ``<parameter=path>...</parameter>`` where ``path`` IS the other real
    parameter of the same ``write_file`` tool.  Schema name filtering alone
    cannot disambiguate — the parser must use a stronger rule (e.g. the
    embedded ``</parameter>`` must be followed by a structural delimiter
    that closes the OUTER param, not the inner literal).
    """
    parser = parser_cls(qwen3_tokenizer, tools=_WRITE_FILE_TOOLS)
    request = ChatCompletionRequest(model=MODEL, messages=[], tools=_WRITE_FILE_TOOLS)
    result = parser.extract_tool_calls(_REAL_PARAM_NAME_LITERAL_OUTPUT, request=request)

    assert result.tools_called
    assert len(result.tool_calls) == 1
    args = json.loads(result.tool_calls[0].function.arguments)
    assert list(args.keys()) == ["path", "content"], (
        f"Spurious params from embedded same-name literal: {list(args.keys())}"
    )
    assert args["path"] == "fixture.py", (
        f"Outer ``path`` was overwritten by embedded literal: {args.get('path')!r}"
    )
    expected = _CONTENT_WITH_REAL_PARAM_NAME_LITERAL.rstrip("\n")
    assert args["content"] == expected, (
        f"content was truncated at the embedded <parameter=path>. "
        f"Got: {args.get('content')!r}"
    )


def test_content_with_real_param_name_literal_streaming(qwen3_tokenizer, parser_cls):
    """Streaming variant of the same case.  Each meaningful structural-
    looking line arrives in its own delta — the parser cannot wait for the
    full text to disambiguate.
    """
    parser = parser_cls(qwen3_tokenizer, tools=_WRITE_FILE_TOOLS)
    request = ChatCompletionRequest(model=MODEL, messages=[], tools=_WRITE_FILE_TOOLS)
    char_deltas = [
        "<tool_call>\n",
        "<function=write_file>\n",
        "<parameter=path>\nfixture.py\n</parameter>\n",
        '<parameter=content>\ndoc = """\n',
        "<parameter=path>\n",
        "literal/value\n",
        "</parameter>\n",
        '"""\n',
        "</parameter>\n",
        "</function>\n",
        "</tool_call>",
    ]
    reconstructor = run_tool_extraction_streaming(
        parser, char_deltas, request, assert_one_tool_per_delta=False
    )
    assert len(reconstructor.tool_calls) == 1
    args = json.loads(reconstructor.tool_calls[0].function.arguments)
    assert list(args.keys()) == ["path", "content"], (
        f"Spurious params from embedded same-name literal: {list(args.keys())}"
    )
    assert args["path"] == "fixture.py"
    expected = _CONTENT_WITH_REAL_PARAM_NAME_LITERAL.rstrip("\n")
    assert args["content"] == expected, (
        f"content was truncated at the embedded <parameter=path>. "
        f"Got: {args.get('content')!r}"
    )


# ---------------------------------------------------------------------------
# Parameter value containing a COMPLETE nested tool_call (all four balise
# types: <tool_call>, <function=...>, <parameter=...>, </parameter>,
# </function>, </tool_call>) — the qwen-code WriteFile pattern when the
# file being written is itself a parser fixture or a chat-template
# example. Every literal must stay inside the value; no spurious extra
# tool calls or params should be generated.
# ---------------------------------------------------------------------------

_CONTENT_WITH_FULL_NESTED_CALL = (
    'doc = """\n'
    "<tool_call>\n"
    "<function=write_file>\n"
    "<parameter=path>\n"
    "literal/value.txt\n"
    "</parameter>\n"
    "<parameter=content>\n"
    "hello\n"
    "</parameter>\n"
    "</function>\n"
    "</tool_call>\n"
    '"""\n'
)

_FULL_NESTED_CALL_OUTPUT = (
    "<tool_call>\n"
    "<function=write_file>\n"
    "<parameter=path>\nfixture.py\n</parameter>\n"
    f"<parameter=content>\n{_CONTENT_WITH_FULL_NESTED_CALL}</parameter>\n"
    "</function>\n"
    "</tool_call>"
)


def test_content_with_full_nested_tool_call_nonstreaming(qwen3_tokenizer, parser_cls):
    """Non-streaming: parameter ``content`` contains a complete literal
    ``<tool_call>...</tool_call>`` whose function/parameter names match
    the OUTER tool's schema.  Every literal must stay inside the value;
    no extra tool call must be generated.
    """
    parser = parser_cls(qwen3_tokenizer, tools=_WRITE_FILE_TOOLS)
    request = ChatCompletionRequest(model=MODEL, messages=[], tools=_WRITE_FILE_TOOLS)
    result = parser.extract_tool_calls(_FULL_NESTED_CALL_OUTPUT, request=request)

    assert result.tools_called
    assert len(result.tool_calls) == 1, (
        f"Expected 1 tool call (the outer one), got "
        f"{len(result.tool_calls)} — embedded literal tool_call was "
        f"incorrectly promoted to a real call."
    )
    args = json.loads(result.tool_calls[0].function.arguments)
    assert list(args.keys()) == ["path", "content"]
    assert args["path"] == "fixture.py"
    expected = _CONTENT_WITH_FULL_NESTED_CALL.rstrip("\n")
    assert args["content"] == expected, (
        f"content truncated/corrupted: {args.get('content')!r}"
    )


def test_content_with_full_nested_tool_call_streaming(qwen3_tokenizer, parser_cls):
    """Streaming variant: the literal nested ``<tool_call>...</tool_call>``
    crosses many delta boundaries; the parser must not start a second
    tool call.
    """
    parser = parser_cls(qwen3_tokenizer, tools=_WRITE_FILE_TOOLS)
    request = ChatCompletionRequest(model=MODEL, messages=[], tools=_WRITE_FILE_TOOLS)
    char_deltas = [
        "<tool_call>\n",
        "<function=write_file>\n",
        "<parameter=path>\nfixture.py\n</parameter>\n",
        '<parameter=content>\ndoc = """\n',
        "<tool_call>\n",
        "<function=write_file>\n",
        "<parameter=path>\n",
        "literal/value.txt\n",
        "</parameter>\n",
        "<parameter=content>\n",
        "hello\n",
        "</parameter>\n",
        "</function>\n",
        "</tool_call>\n",
        '"""\n',
        "</parameter>\n",
        "</function>\n",
        "</tool_call>",
    ]
    reconstructor = run_tool_extraction_streaming(
        parser, char_deltas, request, assert_one_tool_per_delta=False
    )
    assert len(reconstructor.tool_calls) == 1, (
        f"Expected 1 tool call, got {len(reconstructor.tool_calls)} — "
        f"a literal nested <tool_call> was promoted to a real call."
    )
    args = json.loads(reconstructor.tool_calls[0].function.arguments)
    assert list(args.keys()) == ["path", "content"]
    assert args["path"] == "fixture.py"
    expected = _CONTENT_WITH_FULL_NESTED_CALL.rstrip("\n")
    assert args["content"] == expected, (
        f"content truncated/corrupted: {args.get('content')!r}"
    )


# ---------------------------------------------------------------------------
# Two consecutive tool calls, where the SECOND embeds a literal nested
# tool_call whose ``<parameter=NAME>`` uses a NAME that is NOT in the
# OUTER tool's schema (e.g. a description of a different tool's format).
# Reproduces the qwen-code Qwen 3.6 freeze scenario: the depth tracker
# in ``_find_true_param_end`` filters opens by schema, so the literal
# ``</parameter>`` that closes the unknown-NAME literal open appears
# unmatched and matches the structural lookahead of the trailing
# ``</function>``, truncating the OUTER content value.
# ---------------------------------------------------------------------------

_OUT_OF_SCHEMA_NESTED_CONTENT = (
    'template = """\n'
    "<tool_call>\n<function=foo>\n"
    "<parameter=bar>baz</parameter>\n"
    "</function>\n</tool_call>\n"
    '"""\n'
)

_TWO_TOOLS_OUT_OF_SCHEMA_NESTED_OUTPUT = (
    "<tool_call>\n<function=foo>\n"
    "<parameter=bar>baz</parameter>\n"
    "</function>\n</tool_call>"
    "\n\n"
    "<tool_call>\n<function=write_file>\n"
    "<parameter=path>\nfixture.py\n</parameter>\n"
    f"<parameter=content>\n{_OUT_OF_SCHEMA_NESTED_CONTENT}</parameter>\n"
    "</function>\n</tool_call>"
)


def test_two_tools_second_with_out_of_schema_nested_literal_nonstreaming(
    qwen3_tokenizer, parser_cls
):
    """Two structural tool calls; the second's ``content`` value embeds a
    literal nested ``<tool_call>`` block whose inner ``<parameter=bar>``
    uses a NAME not in the outer tool's schema (``write_file`` only knows
    ``path`` and ``content``).

    The walker must still match the outer ``</parameter>`` of ``content``,
    not the literal ``</parameter>`` of the unknown-NAME nested open.
    """
    parser = parser_cls(qwen3_tokenizer, tools=_WRITE_FILE_TOOLS)
    request = ChatCompletionRequest(model=MODEL, messages=[], tools=_WRITE_FILE_TOOLS)
    result = parser.extract_tool_calls(
        _TWO_TOOLS_OUT_OF_SCHEMA_NESTED_OUTPUT, request=request
    )
    assert result.tools_called
    assert len(result.tool_calls) == 2, (
        f"Expected 2 tool calls, got {len(result.tool_calls)}: "
        f"{[tc.function.name for tc in result.tool_calls]}"
    )
    args0 = json.loads(result.tool_calls[0].function.arguments)
    args1 = json.loads(result.tool_calls[1].function.arguments)
    assert args0 == {"bar": "baz"}, f"first tool args wrong: {args0!r}"
    assert result.tool_calls[1].function.name == "write_file"
    assert list(args1.keys()) == ["path", "content"], (
        f"Spurious params on outer tool: {list(args1.keys())}"
    )
    assert args1["path"] == "fixture.py"
    expected = _OUT_OF_SCHEMA_NESTED_CONTENT.rstrip("\n")
    assert args1["content"] == expected, (
        f"outer content truncated at literal </parameter>: {args1.get('content')!r}"
    )


def test_two_tools_second_with_out_of_schema_nested_literal_streaming(
    qwen3_tokenizer, parser_cls
):
    """Streaming variant of the same scenario."""
    parser = parser_cls(qwen3_tokenizer, tools=_WRITE_FILE_TOOLS)
    request = ChatCompletionRequest(model=MODEL, messages=[], tools=_WRITE_FILE_TOOLS)
    char_deltas = [
        "<tool_call>\n<function=foo>\n",
        "<parameter=bar>baz</parameter>\n",
        "</function>\n</tool_call>",
        "\n\n",
        "<tool_call>\n<function=write_file>\n",
        "<parameter=path>\nfixture.py\n</parameter>\n",
        '<parameter=content>\ntemplate = """\n',
        "<tool_call>\n<function=foo>\n",
        "<parameter=bar>baz</parameter>\n",
        "</function>\n</tool_call>\n",
        '"""\n',
        "</parameter>\n",
        "</function>\n",
        "</tool_call>",
    ]
    reconstructor = run_tool_extraction_streaming(
        parser, char_deltas, request, assert_one_tool_per_delta=False
    )
    assert len(reconstructor.tool_calls) == 2, (
        f"Expected 2 tool calls, got {len(reconstructor.tool_calls)}"
    )
    args0 = json.loads(reconstructor.tool_calls[0].function.arguments)
    args1 = json.loads(reconstructor.tool_calls[1].function.arguments)
    assert args0 == {"bar": "baz"}
    assert reconstructor.tool_calls[1].function.name == "write_file"
    assert list(args1.keys()) == ["path", "content"]
    assert args1["path"] == "fixture.py"
    expected = _OUT_OF_SCHEMA_NESTED_CONTENT.rstrip("\n")
    assert args1["content"] == expected, (
        f"outer content truncated/corrupted: {args1.get('content')!r}"
    )


# ---------------------------------------------------------------------------
# Phantom tool calls produced when the model writes an UNRENDERED Jinja
# template literally in its response: ``<tool_call>\n<function={{ x }}>\n
# <parameter={{ k }}>...``.  The function name ``{{ x }}`` contains
# template-syntax characters and CANNOT be a real function — the parser
# must reject these tool calls (or render them as content) rather than
# emit them as real ones, since the client will then raise "tool not
# found" errors and cause the agent to loop.
# ---------------------------------------------------------------------------

_JINJA_PHANTOM_OUTPUT = (
    "<tool_call>\n<function={{ tc.name }}>\n"
    "<parameter={{ k }}>\n{{ v }}\n</parameter>\n"
    "</function>\n</tool_call>"
    "\n\n"
    "<tool_call>\n<function=write_file>\n"
    "<parameter=path>\nout.txt\n</parameter>\n"
    "<parameter=content>\nhello\n</parameter>\n"
    "</function>\n</tool_call>"
)


def test_jinja_template_phantom_tool_call_is_rejected_nonstreaming(
    qwen3_tokenizer, parser_cls
):
    """A ``<function={{ tc.name }}>`` block (unrendered Jinja) emits a
    function name that is not a valid identifier.  It must NOT be
    surfaced as a real tool call — the client would fail with "tool not
    found" and the agent would loop.
    """
    tools = [
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
    parser = parser_cls(qwen3_tokenizer, tools=tools)
    request = ChatCompletionRequest(model=MODEL, messages=[], tools=tools)
    result = parser.extract_tool_calls(_JINJA_PHANTOM_OUTPUT, request=request)
    assert result.tools_called
    names = [tc.function.name for tc in result.tool_calls]
    assert "{{ tc.name }}" not in names, (
        f"Phantom Jinja-template tool call surfaced as real: {names}"
    )
    assert names == ["write_file"], (
        f"Expected only the real ``write_file`` tool call, got: {names}"
    )


# NOTE: a streaming counterpart of the above test is intentionally not
# added.  Filtering phantoms in streaming requires a separate
# "client-visible index" counter (the existing ``current_tool_index`` is
# also used for internal position bookkeeping).  Until that refactor
# lands, the streaming path may still surface phantoms and the client
# is expected to drop unknown function names.  The non-streaming path
# is the one consumed by the offline tools-extraction code and by the
# ``_parse_xml_function_call`` helper invoked at function-end during
# streaming, so production users still see the filtered result for
# completed tool calls.


# ---------------------------------------------------------------------------
# Inline empty ``<tool_call>...</tool_call>`` (no ``<function=>``) before a
# real tool call: the content text BETWEEN the inline literal and the real
# tool call must be preserved.  Previously the content was truncated at the
# position of the FIRST ``<tool_call>`` token regardless of whether that
# block contained a real ``<function=>``.
# ---------------------------------------------------------------------------


def test_inline_empty_tool_call_preserves_content_before_real_call(
    qwen3_tokenizer, parser_cls
):
    """A bare ``<tool_call>example</tool_call>`` in the model's narrative
    text (no ``<function=>`` inside) must NOT consume the surrounding
    content; only the real ``<tool_call>`` block that contains a valid
    function call should anchor ``content_index``.

    The XML parser's SAX-based pipeline consumes the inline empty
    block's body as XML text (so ``example`` is dropped), but the
    surrounding narrative ("I'll show:" and "Now real:") must still be
    preserved — both parsers are checked.
    """
    tools = [
        ChatCompletionToolsParam(
            type="function",
            function={
                "name": "log",
                "parameters": {
                    "type": "object",
                    "properties": {"msg": {"type": "string"}},
                },
            },
        )
    ]
    parser = parser_cls(qwen3_tokenizer, tools=tools)
    request = ChatCompletionRequest(model=MODEL, messages=[], tools=tools)

    text = (
        "I'll show: <tool_call>example</tool_call>. Now real:\n"
        "<tool_call>\n<function=log>\n<parameter=msg>\nhi\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    result = parser.extract_tool_calls(text, request=request)
    assert result.tools_called
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].function.name == "log"
    # Content between the inline empty tool_call and the real one MUST be
    # preserved — dropping it loses the model's contextual narrative.
    assert result.content is not None
    assert "I'll show:" in result.content, (
        f"Pre-inline narrative lost from content: {result.content!r}"
    )
    assert "Now real:" in result.content, (
        f"Content between inline literal and real tool_call lost: {result.content!r}"
    )


# ---------------------------------------------------------------------------
# anyOf [{type: string}, {type: null}] with the literal "null" or "None"
# value must convert to JSON null, NOT preserve as the string "null"/"None".
# Observed against a real Qwen 3.6 server: the model emits ``None`` for a
# nullable optional parameter and the parser kept it as the string "None",
# breaking nullable-typed clients.
# ---------------------------------------------------------------------------


def test_anyof_string_null_with_null_literal_returns_none(qwen3_tokenizer, parser_cls):
    """anyOf [{type: string}, {type: null}] with value "null" or "None"
    must convert to JSON null.  String-typed paths preserve the literal,
    but a nullable schema MUST recognise the null sentinel — otherwise
    the client receives the literal "null" / "None" string and downstream
    type checks fail.
    """
    tools = [
        ChatCompletionToolsParam(
            type="function",
            function={
                "name": "set_value",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "optional": {
                            "anyOf": [{"type": "string"}, {"type": "null"}],
                        },
                    },
                },
            },
        )
    ]
    request = ChatCompletionRequest(model=MODEL, messages=[], tools=tools)
    for literal in ("null", "None"):
        parser = parser_cls(qwen3_tokenizer, tools=tools)
        model_output = (
            "<tool_call>\n"
            "<function=set_value>\n"
            f"<parameter=optional>{literal}</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = parser.extract_tool_calls(model_output, request=request)
        assert result.tools_called
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["optional"] is None, (
            f"anyOf string|null with value {literal!r} was kept as "
            f"{type(args['optional']).__name__}: {args['optional']!r}"
        )


def test_get_vllm_registry_structural_tag_returns_structural_tag(
    parser,
    sample_tools: list[ChatCompletionToolsParam],
) -> None:
    request_tools = _as_chat_completion_tools(sample_tools)
    req = ChatCompletionRequest(
        messages=[],
        model="m",
        tools=request_tools,
        tool_choice="auto",
    )
    tag = parser.get_structural_tag(req)
    assert isinstance(tag, StructuralTag)

    req = ChatCompletionRequest(
        messages=[],
        model="m",
        tools=request_tools,
        tool_choice="required",
    )
    tag = parser.get_structural_tag(req)
    assert isinstance(tag, StructuralTag)

    if request_tools:
        tool = request_tools[0]
        req = ChatCompletionRequest(
            messages=[],
            model="m",
            tools=request_tools,
        )
        req.tool_choice = ChatCompletionNamedToolChoiceParam(
            function=ChatCompletionNamedFunction(name=tool.function.name)
        )
        tag = parser.get_structural_tag(req)
        assert isinstance(tag, StructuralTag)


@pytest.mark.parametrize("include_reasoning", [True, False])
def test_adjust_request_auto_uses_vllm_registry_structural_tag(
    monkeypatch: pytest.MonkeyPatch,
    parser,
    sample_tools: list[ChatCompletionToolsParam],
    include_reasoning: bool,
) -> None:
    monkeypatch.setattr(
        "vllm.tool_parsers.abstract_tool_parser.VLLM_ENFORCE_STRICT_TOOL_CALLING",
        True,
    )
    request_tools = _as_chat_completion_tools(sample_tools)
    req = ChatCompletionRequest(
        messages=[],
        model="m",
        tools=request_tools,
        tool_choice="auto",
        include_reasoning=include_reasoning,
    )
    out = parser.adjust_request(req)
    assert out.structured_outputs is not None
    assert out.structured_outputs.structural_tag is not None
    assert isinstance(out.structured_outputs.structural_tag, str)
    loaded = json.loads(out.structured_outputs.structural_tag)
    assert isinstance(loaded, dict)


def test_adjust_request_required_prefers_structural_tag(
    monkeypatch: pytest.MonkeyPatch,
    parser,
    sample_tools: list[ChatCompletionToolsParam],
) -> None:
    monkeypatch.setattr(
        "vllm.tool_parsers.abstract_tool_parser.VLLM_ENFORCE_STRICT_TOOL_CALLING",
        True,
    )
    request_tools = _as_chat_completion_tools(sample_tools)
    req = ChatCompletionRequest(
        messages=[],
        model="m",
        tools=request_tools,
        tool_choice="required",
    )
    out = parser.adjust_request(req)
    assert out.structured_outputs is not None
    assert out.structured_outputs.structural_tag is not None
