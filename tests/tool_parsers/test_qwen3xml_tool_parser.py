# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest

from tests.tool_parsers.common_tests import (
    ToolParserTestConfig,
    ToolParserTests,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
)
from vllm.tokenizers import get_tokenizer
from vllm.tool_parsers.qwen3xml_tool_parser import Qwen3XMLToolParser
from tests.tool_parsers.utils import run_tool_extraction_streaming

MODEL = "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"


@pytest.fixture(scope="module")
def qwen3_tokenizer():
    return get_tokenizer(tokenizer_name=MODEL)

class TestQwen3xmlToolParser(ToolParserTests):
    @pytest.fixture
    def test_config(self) -> ToolParserTestConfig:
        return ToolParserTestConfig(
            parser_name="qwen3_xml",
            # Test data
            no_tool_calls_output="This is a regular response without any tool calls.",
            single_tool_call_output="<tool_call>\n<function=get_weather>\n<parameter=city>Tokyo</parameter>\n</function>\n</tool_call>",
            parallel_tool_calls_output="<tool_call>\n<function=get_weather>\n<parameter=city>Tokyo</parameter>\n</function>\n</tool_call><tool_call>\n<function=get_time>\n<parameter=timezone>Asia/Tokyo</parameter>\n</function>\n</tool_call>",
            various_data_types_output=(
                "<tool_call>\n<function=test_function>\n"
                "<parameter=string_field>hello</parameter>\n"
                "<parameter=int_field>42</parameter>\n"
                "<parameter=float_field>3.14</parameter>\n"
                "<parameter=bool_field>true</parameter>\n"
                "<parameter=null_field>null</parameter>\n"
                '<parameter=array_field>["a", "b", "c"]</parameter>\n'
                '<parameter=object_field>{"nested": "value"}</parameter>\n'
                "</function>\n</tool_call>"
            ),
            empty_arguments_output="<tool_call>\n<function=refresh>\n</function>\n</tool_call>",
            surrounding_text_output=(
                "Let me check the weather for you.\n\n"
                "<tool_call>\n<function=get_weather>\n"
                "<parameter=city>Tokyo</parameter>\n"
                "</function>\n</tool_call>\n\n"
                "I will get that information."
            ),
            escaped_strings_output=(
                "<tool_call>\n<function=test_function>\n"
                '<parameter=quoted>He said "hello"</parameter>\n'
                "<parameter=path>C:\\Users\\file.txt</parameter>\n"
                "<parameter=newline>line1\nline2</parameter>\n"
                "</function>\n</tool_call>"
            ),
            malformed_input_outputs=[
                "<tool_call><function=func>",
                "<tool_call><function=></function></tool_call>",
            ],
            # Expected results
            single_tool_call_expected_name="get_weather",
            single_tool_call_expected_args={"city": "Tokyo"},
            parallel_tool_calls_count=2,
            parallel_tool_calls_names=["get_weather", "get_time"],
            supports_typed_arguments=False,
        )

    def test_qwen3xml_async_streaming_free_text(self, qwen3_tokenizer):
        parser = Qwen3XMLToolParser(qwen3_tokenizer)
        
        # 1. First tool call
        # 2. Free text
        # 3. Second tool call
        text_to_stream = (
            "<tool_call>\n<function=get_weather>\n<parameter=city>Paris</parameter>\n</function>\n</tool_call>"
            "\nNext, I will check the weather for London:\n"
            "<tool_call>\n<function=get_weather>\n<parameter=city>London</parameter>\n</function>\n</tool_call>"
        )
        
        request = ChatCompletionRequest(messages=[], model="test")
        emitted_messages = []
        previous_text = ""
        previous_tokens = []
        token_ids = qwen3_tokenizer.encode(text_to_stream, add_special_tokens=False)
        
        for i in range(1, len(token_ids) + 1):
            current_token_ids = token_ids[:i]
            current_text = qwen3_tokenizer.decode(current_token_ids)
            delta_text = current_text[len(previous_text):]
            token_delta = current_token_ids[len(previous_tokens):]
            
            delta = parser.extract_tool_calls_streaming(
                previous_text,
                current_text,
                delta_text,
                previous_tokens,
                current_token_ids,
                token_delta,
                request
            )
            if delta is not None:
                emitted_messages.append(delta)
                
            previous_text = current_text
            previous_tokens = current_token_ids

        # Check that the free text is emitted BEFORE London's arguments are emitted.
        found_early = False
        accumulated_content = ""
        for i, msg in enumerate(emitted_messages):
            if msg.content:
                accumulated_content += msg.content
            
            if "Next, I will check the weather for London" in accumulated_content:
                # Check if we already saw "London" in any previous or current tool call arguments
                is_london_emitted = any(
                    tc.function.arguments and "London" in tc.function.arguments 
                    for m in emitted_messages[:i+1] if m.tool_calls 
                    for tc in m.tool_calls
                )
                if not is_london_emitted:
                    found_early = True
                break
        
        assert found_early, "Free text between tool calls should be emitted as soon as the second tool call starts, not delayed."

    def test_qwen3xml_streaming_text_after_tool_call(self, qwen3_tokenizer):
        parser = Qwen3XMLToolParser(qwen3_tokenizer)
        
        # Tool call followed by free text
        text_to_stream = (
            "<tool_call>\n<function=get_weather>\n<parameter=city>Paris</parameter>\n</function>\n</tool_call>"
            "\nI hope this helps!"
        )
        
        request = ChatCompletionRequest(messages=[], model="test")
        emitted_messages = []
        previous_text = ""
        previous_tokens = []
        token_ids = qwen3_tokenizer.encode(text_to_stream, add_special_tokens=False)
        
        for i in range(1, len(token_ids) + 1):
            current_token_ids = token_ids[:i]
            current_text = qwen3_tokenizer.decode(current_token_ids)
            delta_text = current_text[len(previous_text):]
            token_delta = current_token_ids[len(previous_tokens):]
            
            delta = parser.extract_tool_calls_streaming(
                previous_text,
                current_text,
                delta_text,
                previous_tokens,
                current_token_ids,
                token_delta,
                request
            )
            if delta is not None:
                emitted_messages.append(delta)
                
            previous_text = current_text
            previous_tokens = current_token_ids

        # Aggregate all emitted content
        all_content = "".join([m.content for m in emitted_messages if m.content])
        
        assert "I hope this helps!" in all_content, "Free text after the last tool call should be emitted."


def test_qwen36_anyof_parameter_xml_not_double_encoded(qwen3_tokenizer):
    tools = [
        ChatCompletionToolsParam(
            type="function",
            function={
                "name": "update_record",
                "parameters": {
                    "type": "object",
                    "properties": {
                        # anyOf schema — no top-level "type" key
                        "data": {
                            "anyOf": [{"type": "object"}, {"type": "null"}],
                        },
                    },
                },
            },
        )
    ]

    parser = Qwen3XMLToolParser(qwen3_tokenizer, tools=tools)
    model_output = (
        "<tool_call>\n"
        "<function=update_record>\n"
        '<parameter=data>{"key": "value", "count": 42}</parameter>\n'
        "</function>\n"
        "</tool_call>"
    )
    request = ChatCompletionRequest(model=MODEL, messages=[], tools=tools)
    result = parser.extract_tool_calls(model_output, request=request)

    assert result.tools_called
    assert len(result.tool_calls) == 1
    args = json.loads(result.tool_calls[0].function.arguments)

    assert isinstance(args["data"], dict), (
        f"anyOf parameter was double-encoded: data={args['data']!r}. "
        "StreamingXMLToolCallParser._get_param_type ignores anyOf schemas."
    )
    assert args["data"] == {"key": "value", "count": 42}


def test_qwen36_anyof_parameter_xml_streaming_not_double_encoded(qwen3_tokenizer):
  
    tools = [
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

    parser = Qwen3XMLToolParser(qwen3_tokenizer, tools=tools)
    request = ChatCompletionRequest(model=MODEL, messages=[], tools=tools)

    # Deltas are pre-formed XML element chunks (one element per delta),
    # which is the same pattern used by speculative decoding.
    deltas = [
        "<tool_call>",
        "\n<function=update_record>",
        '\n<parameter=data>{"key": "value", "count": 42}</parameter>',
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
    assert isinstance(args["data"], dict), (
        f"anyOf parameter was double-encoded in streaming: data={args['data']!r}"
    )


def test_qwen36_xml_streaming_double_close_brace(qwen3_tokenizer):
    tools = [
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

    parser = Qwen3XMLToolParser(qwen3_tokenizer, tools=tools)
    request = ChatCompletionRequest(model=MODEL, messages=[], tools=tools)

    deltas = [
        "<tool_call>",
        "\n<function=get_weather>",
        "\n<parameter=city>\nDallas\n</parameter>",
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
    full_args = reconstructor.tool_calls[0].function.arguments

    assert not full_args.endswith("}}"), (
        f"XML streaming parser emitted double closing brace: {full_args!r}. "
        "parse_single_streaming_chunks fallback called _end_element('function') twice."
    )
    args = json.loads(full_args)
    assert args == {"city": "Dallas"}


def test_xml_streaming_parallel_tool_calls_preformed_chunks(qwen3_tokenizer):
    """
    Note: in normal token-by-token streaming this rarely triggers because
    the tokenizer splits XML tags across multiple tokens.  It CAN trigger with
    speculative decoding multi-token flushes.
    """
    

    tools = [
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

    parser = Qwen3XMLToolParser(qwen3_tokenizer, tools=tools)
    request = ChatCompletionRequest(model=MODEL, messages=[], tools=tools)

    deltas = [
        "<tool_call>",
        "\n<function=get_weather>",
        "\n<parameter=city>Paris</parameter>",
        "\n</function>",
        "\n</tool_call>",
        "<tool_call>",
        "\n<function=get_weather>",
        "\n<parameter=city>London</parameter>",
        "\n</function>",
        "\n</tool_call>",
    ]

    reconstructor = run_tool_extraction_streaming(
        parser,
        deltas,
        request,
        assert_one_tool_per_delta=False,
    )

    assert len(reconstructor.tool_calls) == 2, (
        f"Expected 2 tool calls, got {len(reconstructor.tool_calls)}"
    )

    args0 = json.loads(reconstructor.tool_calls[0].function.arguments)
    args1 = json.loads(reconstructor.tool_calls[1].function.arguments)

    assert reconstructor.tool_calls[0].function.name == "get_weather"
    assert reconstructor.tool_calls[1].function.name == "get_weather"
    assert args0 == {"city": "Paris"}, f"First call args wrong: {args0!r}"
    assert args1 == {"city": "London"}, f"Second call args wrong: {args1!r}"


# ---------------------------------------------------------------------------
# Bug-confirmation tests (regressions to FIX)
# ---------------------------------------------------------------------------


def test_xml_string_null_value_not_emptied(qwen3_tokenizer):
    """
    Bug A: _convert_param_value intercepts "null" before the type check.
    For a STRING parameter with value "null", the parser should output
    the JSON string "null", not an empty string "".

    Root cause: `if param_value.lower() == "null": return None` runs first,
    then _convert_for_json_streaming(None, "string") returns "", so the
    closing-quote _end_element emits "" instead of "null".
    """
    tools = [
        ChatCompletionToolsParam(
            type="function",
            function={
                "name": "search",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                    },
                },
            },
        )
    ]

    parser = Qwen3XMLToolParser(qwen3_tokenizer, tools=tools)
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
    assert len(result.tool_calls) == 1
    args = json.loads(result.tool_calls[0].function.arguments)

    assert "query" in args, f"Parameter 'query' missing from args: {args!r}"
    assert args["query"] == "null", (
        f"String parameter with literal value 'null' was incorrectly converted. "
        f"Got: {args['query']!r}. "
        f"Expected: 'null' (the string). "
        f"_convert_param_value returns None before checking type, "
        f"then _convert_for_json_streaming(None, 'string') returns ''."
    )


def test_xml_streaming_boolean_true_not_false(qwen3_tokenizer):
    """
    Bug B: In streaming mode, a boolean parameter with value "true" is
    streamed as "false".

    Root cause: When "true" arrives character by character:
      - 't' → _convert_param_value("t", "boolean") = False → emits "false"
      - 'r','u','e' → no new delta (output_data[len("false"):] = "")
    Final accumulated arguments contain "false" instead of "true".
    """
    tools = [
        ChatCompletionToolsParam(
            type="function",
            function={
                "name": "set_flag",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "enabled": {"type": "boolean"},
                    },
                },
            },
        )
    ]

    parser = Qwen3XMLToolParser(qwen3_tokenizer, tools=tools)
    request = ChatCompletionRequest(model=MODEL, messages=[], tools=tools)

    # Feed character-by-character to trigger the streaming accumulation bug.
    # Each chunk simulates a single-character token arriving in streaming.
    char_deltas = [
        "<tool_call>",
        "\n<function=set_flag>",
        "\n<parameter=enabled>",
        "t",   # ← first char triggers False → emits "false"
        "r",
        "u",
        "e",   # ← full "true" but delta = "true"[5:] = ""
        "</parameter>",
        "\n</function>",
        "\n</tool_call>",
    ]

    reconstructor = run_tool_extraction_streaming(
        parser,
        char_deltas,
        request,
        assert_one_tool_per_delta=False,
    )

    assert len(reconstructor.tool_calls) == 1
    args = json.loads(reconstructor.tool_calls[0].function.arguments)

    assert args["enabled"] is True, (
        f"Boolean streaming bug: expected True, got {args['enabled']!r}. "
        f"First char 't' emits 'false'; subsequent chars emit nothing; "
        f"final value is 'false' even though the model said 'true'."
    )


def test_xml_streaming_string_null_last_char_not_dropped(qwen3_tokenizer):
    """
    Bug A (streaming variant): String parameter with value "null" loses
    the last character 'l' when tokens arrive one by one.

    Root cause: Accumulating 'n','u','l' emits correctly, but on the
    fourth char 'l' the full value is "null" →
    _convert_param_value("null", "string") → None →
    _convert_for_json_streaming(None, "string") → "" → delta = ""[3:] = "".
    The closing quote is then emitted, yielding "nul" not "null".
    """
    tools = [
        ChatCompletionToolsParam(
            type="function",
            function={
                "name": "search",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                    },
                },
            },
        )
    ]

    parser = Qwen3XMLToolParser(qwen3_tokenizer, tools=tools)
    request = ChatCompletionRequest(model=MODEL, messages=[], tools=tools)

    char_deltas = [
        "<tool_call>",
        "\n<function=search>",
        "\n<parameter=query>",
        "n",
        "u",
        "l",
        "l",   # ← triggers _convert_param_value("null",…) = None → nothing emitted
        "</parameter>",
        "\n</function>",
        "\n</tool_call>",
    ]

    reconstructor = run_tool_extraction_streaming(
        parser,
        char_deltas,
        request,
        assert_one_tool_per_delta=False,
    )

    assert len(reconstructor.tool_calls) == 1
    args = json.loads(reconstructor.tool_calls[0].function.arguments)

    assert "query" in args
    assert args["query"] == "null", (
        f"String 'null' streaming bug: last 'l' was dropped. "
        f"Got: {args['query']!r}. "
        f"When full value reaches 'null', _convert_param_value returns None "
        f"and _convert_for_json_streaming(None, 'string') returns '', "
        f"so the final delta is empty and the 'l' is never emitted."
    )


def test_xml_anyof_integer_null_type_detected(qwen3_tokenizer):
    """
    Bug C: _get_param_type only returns non-string for anyOf schemas that
    contain "object" or "array". For anyOf: [{type: "integer"}, {type: "null"}]
    it falls through and returns "string", so integer parameters with
    nullable schemas are incorrectly quoted and not converted.
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

    parser = Qwen3XMLToolParser(qwen3_tokenizer, tools=tools)
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
    assert len(result.tool_calls) == 1
    args = json.loads(result.tool_calls[0].function.arguments)

    assert args["count"] == 42, (
        f"anyOf integer+null: expected int 42, got {args['count']!r}. "
        f"_get_param_type only checks for object/array in anyOf schemas, "
        f"so integer anyOf schemas fall back to 'string', causing '42' "
        f"to be returned as the JSON string '\"42\"' instead of the number 42."
    )


# ---------------------------------------------------------------------------
# Regression: XML structural tags as literal text inside string parameters
# ---------------------------------------------------------------------------

_WRITE_FILE_TOOLS_XML = [
    ChatCompletionToolsParam(
        type="function",
        function={
            "name": "write_file",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "content": {"type": "string"},
                },
            },
        },
    )
]

# Python content that contains all four XML structural tags as literal strings.
# When qwen3xml encounters "</parameter>" inside the content value it
# currently treats it as the structural end of the <parameter=content> element,
# truncating the value and creating a spurious "query" parameter from the text
# that follows the fake </parameter>.
_XML_TAGS_IN_CONTENT_XML = (
    'char_deltas = [\n'
    '    "<tool_call>\\n",\n'
    '    "<parameter=query>\\n",\n'
    '    "\\n</parameter>\\n",\n'
    '    "</function>\\n",\n'
    ']\n'
)

_WRITE_FILE_XML_TAGS_OUTPUT_XML = (
    "<tool_call>\n"
    "<function=write_file>\n"
    "<parameter=file_path>\ntest.py\n</parameter>\n"
    f"<parameter=content>\n{_XML_TAGS_IN_CONTENT_XML}</parameter>\n"
    "</function>\n"
    "</tool_call>\n"
)


def test_xml_streaming_content_with_structural_xml_tags(qwen3_tokenizer):
    """Streaming variant: pre-formed chunks, full content in one delta."""
    parser = Qwen3XMLToolParser(qwen3_tokenizer, tools=_WRITE_FILE_TOOLS_XML)
    request = ChatCompletionRequest(
        model=MODEL, messages=[], tools=_WRITE_FILE_TOOLS_XML
    )

    char_deltas = [
        "<tool_call>\n",
        "<function=write_file>\n",
        "<parameter=file_path>\ntest.py\n</parameter>\n",
        f"<parameter=content>\n{_XML_TAGS_IN_CONTENT_XML}</parameter>\n",
        "</function>\n",
        "</tool_call>\n",
    ]

    reconstructor = run_tool_extraction_streaming(
        parser,
        char_deltas,
        request,
        assert_one_tool_per_delta=False,
    )

    assert len(reconstructor.tool_calls) == 1
    assert reconstructor.tool_calls[0].function.name == "write_file"

    args = json.loads(reconstructor.tool_calls[0].function.arguments)

    assert list(args.keys()) == ["file_path", "content"], (
        f"Unexpected parameter keys (spurious params from embedded tags?): "
        f"{list(args.keys())}"
    )
    assert args["file_path"] == "test.py"
    expected_content = _XML_TAGS_IN_CONTENT_XML.rstrip("\n")
    assert args["content"] == expected_content, (
        f"content was truncated or corrupted by embedded XML tags.\n"
        f"Got:      {args.get('content')!r}\n"
        f"Expected: {expected_content!r}"
    )


def test_xml_nonstreaming_content_with_structural_xml_tags(qwen3_tokenizer):
    """Regression: string parameter containing </parameter>, </function>,
    </tool_call> as literal text must be extracted intact.

    Bug: the SAX pre-processor (_preprocess_xml_chunk) returns
    ``safe_text + "</parameter>"`` when it sees ``</parameter>`` inside the
    accumulated parameter buffer, terminating the current parameter too early.
    The text that follows the spurious closing tag is then misinterpreted as a
    new parameter named "query", creating a ghost parameter and truncating
    the real "content" value.

    Expected: exactly two parameters -- file_path and content -- with content
    equal to the full Python snippet including the embedded XML tags.
    """
    parser = Qwen3XMLToolParser(qwen3_tokenizer, tools=_WRITE_FILE_TOOLS_XML)
    request = ChatCompletionRequest(
        model=MODEL, messages=[], tools=_WRITE_FILE_TOOLS_XML
    )
    result = parser.extract_tool_calls(_WRITE_FILE_XML_TAGS_OUTPUT_XML, request=request)

    assert result.tools_called
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].function.name == "write_file"

    args = json.loads(result.tool_calls[0].function.arguments)

    assert list(args.keys()) == ["file_path", "content"], (
        f"Unexpected parameter keys (spurious params created from embedded tags?): "
        f"{list(args.keys())}. "
        f"_preprocess_xml_chunk sees '</parameter>' inside the accumulated "
        f"_pre_param_buffer and terminates the parameter early; the text after "
        f"'<parameter=query>' becomes a ghost 'query' parameter."
    )
    assert args["file_path"] == "test.py"
    expected_content = _XML_TAGS_IN_CONTENT_XML.rstrip("\n")
    assert args["content"] == expected_content, (
        f"content was truncated or corrupted by embedded XML tags. "
        f"Got:      {args.get('content')!r}\n"
        f"Expected: {expected_content!r}"
    )


# File content whose lines ARE standalone </parameter> and <parameter=NAME>
# tokens (preceded by \n). This simulates writing a Jinja2 template, a test
# fixture for the parser, or any file that references the tool-call format.
# "new_string" is intentionally NOT a parameter of write_file (schema has
# "file_path" and "content"), so the schema filter must prevent it from being
# treated as a structural boundary.
_CONTENT_WITH_PARAM_LIKE_LINES_XML = (
    'TOOL_CALL_TEMPLATE = """\n'
    "</parameter>\n"
    "<parameter=new_string>\n"
    "#!/usr/bin/env python3\n"
    "</parameter>\n"
    '"""\n'
)

_WRITE_FILE_PARAM_LIKE_LINES_OUTPUT_XML = (
    "<tool_call>\n"
    "<function=write_file>\n"
    "<parameter=file_path>\ntest_template.py\n</parameter>\n"
    f"<parameter=content>\n{_CONTENT_WITH_PARAM_LIKE_LINES_XML}</parameter>\n"
    "</function>\n"
    "</tool_call>\n"
)


def test_xml_nonstreaming_content_with_param_like_lines(qwen3_tokenizer):
    """Non-streaming: file content containing </parameter> and <parameter=NAME>
    on their own lines must not be truncated at the first </parameter> or
    create spurious extra parameters.  Requires schema-based filtering so that
    "new_string" (not a real parameter of write_file) is ignored.
    """
    parser = Qwen3XMLToolParser(qwen3_tokenizer, tools=_WRITE_FILE_TOOLS_XML)
    request = ChatCompletionRequest(
        model=MODEL, messages=[], tools=_WRITE_FILE_TOOLS_XML
    )
    result = parser.extract_tool_calls(_WRITE_FILE_PARAM_LIKE_LINES_OUTPUT_XML, request=request)

    assert result.tools_called
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].function.name == "write_file"

    args = json.loads(result.tool_calls[0].function.arguments)
    assert list(args.keys()) == ["file_path", "content"], (
        f"Spurious parameters created: {list(args.keys())}"
    )
    assert args["file_path"] == "test_template.py"
    expected = _CONTENT_WITH_PARAM_LIKE_LINES_XML.rstrip("\n")
    assert args["content"] == expected, (
        f"content truncated or wrong: {args.get('content')!r}"
    )


def test_xml_streaming_content_with_param_like_lines(qwen3_tokenizer):
    """Streaming: file content containing </parameter> and <parameter=NAME> on
    their own lines — split into one chunk per structural token — must not
    cause spurious extra parameters.

    The critical scenario: chunk 5 is '</parameter>\\n' arriving ALONE so
    the streaming buffer has nothing after it (rest='') which previously
    triggered the 'not rest → structural' fallback, ending the 'content'
    parameter prematurely.  After the schema fix, the subsequent
    '<parameter=new_string>' is recognised as non-structural and the full
    content is preserved.
    """
    parser = Qwen3XMLToolParser(qwen3_tokenizer, tools=_WRITE_FILE_TOOLS_XML)
    request = ChatCompletionRequest(
        model=MODEL, messages=[], tools=_WRITE_FILE_TOOLS_XML
    )

    char_deltas = [
        "<tool_call>\n",
        "<function=write_file>\n",
        "<parameter=file_path>\ntest_template.py\n</parameter>\n",
        '<parameter=content>\nTOOL_CALL_TEMPLATE = """\n',
        "</parameter>\n",               # first literal close — alone in its delta
        "<parameter=new_string>\n",     # literal new-param line
        "#!/usr/bin/env python3\n",
        "</parameter>\n",               # second literal close
        '"""\n',
        "</parameter>\n",               # REAL close of content
        "</function>\n",
        "</tool_call>\n",
    ]

    reconstructor = run_tool_extraction_streaming(
        parser,
        char_deltas,
        request,
        assert_one_tool_per_delta=False,
    )

    assert len(reconstructor.tool_calls) == 1, (
        f"Expected 1 tool call, got {len(reconstructor.tool_calls)}: "
        f"{[tc.function.name for tc in reconstructor.tool_calls]}"
    )
    assert reconstructor.tool_calls[0].function.name == "write_file"
    args = json.loads(reconstructor.tool_calls[0].function.arguments)
    assert list(args.keys()) == ["file_path", "content"], (
        f"Spurious parameters created: {list(args.keys())}"
    )
    assert args["file_path"] == "test_template.py"
    expected = _CONTENT_WITH_PARAM_LIKE_LINES_XML.rstrip("\n")
    assert args["content"] == expected, (
        f"content truncated or wrong: {args.get('content')!r}"
    )


_OBJECT_PARAM_TOOLS_XML = [
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

_DOUBLE_ENCODED_OBJECT_OUTPUT_XML = (
    "<tool_call>\n"
    "<function=process>\n"
    "<parameter=name>\nhello\n</parameter>\n"
    "<parameter=data>\n\"{'key': 'value', 'n': 1}\"\n</parameter>\n"
    "</function>\n"
    "</tool_call>\n"
)


def test_xml_nonstreaming_double_encoded_object_param(qwen3_tokenizer):
    """Non-streaming: model trained with buggy template (json.dumps(str(dict)))
    outputs object args as a JSON-encoded Python repr.  Parser must recover
    the real dict via double-decode.
    """
    parser = Qwen3XMLToolParser(qwen3_tokenizer, tools=_OBJECT_PARAM_TOOLS_XML)
    request = ChatCompletionRequest(
        model=MODEL, messages=[], tools=_OBJECT_PARAM_TOOLS_XML
    )
    result = parser.extract_tool_calls(
        _DOUBLE_ENCODED_OBJECT_OUTPUT_XML, request=request
    )

    assert result.tools_called
    assert len(result.tool_calls) == 1
    args = json.loads(result.tool_calls[0].function.arguments)
    assert args["name"] == "hello"
    assert isinstance(args["data"], dict), (
        f"Expected dict, got {type(args['data'])}: {args['data']!r}"
    )
    assert args["data"] == {"key": "value", "n": 1}


def test_xml_streaming_double_encoded_object_param(qwen3_tokenizer):
    """Streaming: same double-encoded object parameter scenario."""
    parser = Qwen3XMLToolParser(qwen3_tokenizer, tools=_OBJECT_PARAM_TOOLS_XML)
    request = ChatCompletionRequest(
        model=MODEL, messages=[], tools=_OBJECT_PARAM_TOOLS_XML
    )
    reconstructor = run_tool_extraction_streaming(
        parser,
        _DOUBLE_ENCODED_OBJECT_OUTPUT_XML,
        request,
        assert_one_tool_per_delta=False,
    )
    assert len(reconstructor.tool_calls) == 1
    args = json.loads(reconstructor.tool_calls[0].function.arguments)
    assert args["name"] == "hello"
    assert isinstance(args["data"], dict), (
        f"Expected dict, got {type(args['data'])}: {args['data']!r}"
    )
    assert args["data"] == {"key": "value", "n": 1}


# ============================================================================
# Qwen 3.6 Bug Confirmations (placeholder, truncated test removed)
# ============================================================================

