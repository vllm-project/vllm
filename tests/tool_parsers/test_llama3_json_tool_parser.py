# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from unittest.mock import MagicMock, patch

import pytest
from transformers import AutoTokenizer

from vllm.entrypoints.generate.base.protocol import ExtractedToolCallInformation
from vllm.tool_parsers.llama_tool_parser import Llama3JsonToolParser

LLAMA_MODEL = "meta-llama/Llama-3.2-1B-Instruct"


@pytest.fixture(scope="module")
def llama_tokenizer():
    return AutoTokenizer.from_pretrained(LLAMA_MODEL)


@pytest.fixture
def parser(llama_tokenizer):
    return Llama3JsonToolParser(llama_tokenizer)


def test_extract_tool_calls_simple(parser):
    # Test with a simple tool call
    model_output = (
        'Here is the result: {"name": "getOpenIncidentsTool", '
        '"parameters": {}} Would you like to know more?'
    )
    result = parser.extract_tool_calls(model_output, None)

    assert isinstance(result, ExtractedToolCallInformation)
    assert result.tools_called is True
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].type == "function"
    assert result.tool_calls[0].function.name == "getOpenIncidentsTool"
    assert result.tool_calls[0].function.arguments == "{}"
    assert result.content is None


def test_extract_tool_calls_with_arguments(parser):
    # Test with a tool call that has arguments
    model_output = (
        '{"name": "searchTool", "parameters": {"query": "test query", "limit": 10}}'
    )
    result = parser.extract_tool_calls(model_output, None)

    assert result.tools_called is True
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].function.name == "searchTool"
    assert '"query": "test query"' in result.tool_calls[0].function.arguments
    assert '"limit": 10' in result.tool_calls[0].function.arguments


def test_extract_tool_calls_no_json(parser):
    # Test with text that doesn't contain a JSON object
    model_output = "This is just some text without any tool calls"
    result = parser.extract_tool_calls(model_output, None)

    assert result.tools_called is False
    assert len(result.tool_calls) == 0
    assert result.content == model_output


def test_extract_tool_calls_invalid_json(parser):
    # Test with invalid JSON
    model_output = '{"name": "invalidTool", "parameters": {invalid json}'
    result = parser.extract_tool_calls(model_output, None)

    assert result.tools_called is False
    assert len(result.tool_calls) == 0
    assert result.content == model_output


def test_extract_tool_calls_with_arguments_key(parser):
    # Test with a tool call that uses "arguments" instead of "parameters"
    model_output = '{"name": "searchTool", "arguments": {"query": "test"}}'
    result = parser.extract_tool_calls(model_output, None)

    assert result.tools_called is True
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].function.name == "searchTool"
    assert '"query": "test"' in result.tool_calls[0].function.arguments


def test_extract_tool_calls_multiple_json(parser):
    # Test with multiple JSONs separated by semicolons
    model_output = (
        '{"name": "searchTool", "parameters": {"query": "test1"}}; '
        '{"name": "getOpenIncidentsTool", "parameters": {}}; '
        '{"name": "searchTool", "parameters": {"query": "test2"}}'
    )
    result = parser.extract_tool_calls(model_output, None)

    assert result.tools_called is True
    assert len(result.tool_calls) == 3

    # Check first tool call
    assert result.tool_calls[0].function.name == "searchTool"
    assert '"query": "test1"' in result.tool_calls[0].function.arguments

    # Check second tool call
    assert result.tool_calls[1].function.name == "getOpenIncidentsTool"
    assert result.tool_calls[1].function.arguments == "{}"

    # Check third tool call
    assert result.tool_calls[2].function.name == "searchTool"
    assert '"query": "test2"' in result.tool_calls[2].function.arguments


def test_extract_tool_calls_multiple_json_with_whitespace(parser):
    # Test with multiple JSONs separated by semicolons and extra whitespace
    model_output = (
        '{"name": "searchTool", "parameters": {"query": "test1"}} ; '
        '{"name": "getOpenIncidentsTool", "parameters": {}} ; '
        '{"name": "searchTool", "parameters": {"query": "test2"}}'
    )
    result = parser.extract_tool_calls(model_output, None)

    assert result.tools_called is True
    assert len(result.tool_calls) == 3
    assert result.tool_calls[0].function.name == "searchTool"
    assert result.tool_calls[1].function.name == "getOpenIncidentsTool"
    assert result.tool_calls[2].function.name == "searchTool"


def test_extract_tool_calls_multiple_json_with_surrounding_text(parser):
    # Test with multiple JSONs and surrounding text
    model_output = (
        "Here are the results: "
        '{"name": "searchTool", "parameters": {"query": "test1"}}; '
        '{"name": "getOpenIncidentsTool", "parameters": {}}; '
        '{"name": "searchTool", "parameters": {"query": "test2"}} '
        "Would you like to know more?"
    )
    result = parser.extract_tool_calls(model_output, None)

    assert result.tools_called is True
    assert len(result.tool_calls) == 3
    assert result.tool_calls[0].function.name == "searchTool"
    assert result.tool_calls[1].function.name == "getOpenIncidentsTool"
    assert result.tool_calls[2].function.name == "searchTool"


def test_extract_tool_calls_deeply_nested_json(parser):
    # Test with deeply nested JSON parameters (5 levels)
    model_output = (
        '{"name": "complexTool", '
        '"parameters": {'
        '"level1": {'
        '"level2": {'
        '"level3": {'
        '"level4": {'
        '"value": "deep"'
        "}}}}}}"
    )
    result = parser.extract_tool_calls(model_output, None)

    assert result.tools_called is True
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].function.name == "complexTool"
    # Verify the nested structure is preserved in the arguments
    import json

    args = json.loads(result.tool_calls[0].function.arguments)
    assert args["level1"]["level2"]["level3"]["level4"]["value"] == "deep"


def test_extract_tool_calls_multiple_with_deep_nesting(parser):
    # Test with multiple tool calls where some have deeply nested parameters
    model_output = (
        '{"name": "simpleTool", "parameters": {"value": "test"}}; '
        '{"name": "complexTool", "parameters": '
        '{"config": {"database": {"connection": {"pool": {"size": 10}}}}}}'
    )
    result = parser.extract_tool_calls(model_output, None)

    assert result.tools_called is True
    assert len(result.tool_calls) == 2

    # Check first tool call
    assert result.tool_calls[0].function.name == "simpleTool"
    import json

    args0 = json.loads(result.tool_calls[0].function.arguments)
    assert args0["value"] == "test"

    # Check second tool call with deep nesting
    assert result.tool_calls[1].function.name == "complexTool"
    args1 = json.loads(result.tool_calls[1].function.arguments)
    assert args1["config"]["database"]["connection"]["pool"]["size"] == 10


def test_extract_tool_calls_with_quotes_and_brackets_in_string(parser):
    # Test with quotes and brackets inside quoted string values
    model_output = (
        '{"name": "searchTool", '
        '"parameters": {'
        '"query": "test {value} [complex]",'
        '"nested": {"inner": "more {brackets}"}'
        "}}"
    )
    result = parser.extract_tool_calls(model_output, None)

    assert result.tools_called is True
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].function.name == "searchTool"
    # Verify the string values are preserved including brackets and quotes
    import json

    args = json.loads(result.tool_calls[0].function.arguments)
    assert args["query"] == "test {value} [complex]"
    assert args["nested"]["inner"] == "more {brackets}"


def test_extract_tool_calls_with_escaped_quotes_in_nested_json(parser):
    # Test with escaped quotes in deeply nested JSON
    model_output = (
        '{"name": "parserTool", "parameters": {"text": "He said \\"Hello {world}\\""}}'
    )
    result = parser.extract_tool_calls(model_output, None)

    assert result.tools_called is True
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].function.name == "parserTool"
    # Verify escaped quotes are preserved
    import json

    args = json.loads(result.tool_calls[0].function.arguments)
    assert args["text"] == 'He said "Hello {world}"'


def test_extract_tool_calls_missing_name_key(parser):
    # Test that missing "name" key returns content
    model_output = '{"parameters": {}}'
    result = parser.extract_tool_calls(model_output, None)

    assert result.tools_called is False
    assert len(result.tool_calls) == 0
    assert result.content == model_output


def test_extract_tool_calls_missing_parameters_and_arguments_key(parser):
    # Test that missing both "parameters" and "arguments" keys returns content
    model_output = '{"name": "toolWithoutParams"}'
    result = parser.extract_tool_calls(model_output, None)

    assert result.tools_called is False
    assert len(result.tool_calls) == 0
    assert result.content == model_output


def _run_streaming(parser, full_text, chunk_size=1):
    """Feed `full_text` through extract_tool_calls_streaming in fixed-size
    chunks and reassemble the resulting content and tool calls, mirroring
    how an OpenAI-compatible streaming client accumulates deltas. Token ids
    are unused by this parser's streaming logic, so empty lists stand in
    for them.
    """
    previous_text = ""
    content = ""
    tool_calls: dict[int, list[str]] = {}
    for i in range(0, len(full_text), chunk_size):
        current_text = full_text[: i + chunk_size]
        delta_text = current_text[len(previous_text) :]
        delta = parser.extract_tool_calls_streaming(
            previous_text, current_text, delta_text, [], [], [], None
        )
        if delta is not None:
            if delta.content:
                content += delta.content
            for tc in delta.tool_calls or []:
                slot = tool_calls.setdefault(tc.index, ["", ""])
                if tc.function:
                    slot[0] += tc.function.get("name") or ""
                    slot[1] += tc.function.get("arguments") or ""
        previous_text = current_text
    return (content or None), [tuple(v) for v in tool_calls.values()]


def test_streaming_plain_json_answer_not_a_call(parser):
    # Regression test for #58824 case 1: a plain answer that happens to
    # start with '{' but never produces a "name" key must be streamed as
    # content, matching non-streaming behavior, instead of the whole reply
    # silently disappearing.
    model_output = '{"a": 1} is a dict'

    non_streaming = parser.extract_tool_calls(model_output, None)
    assert non_streaming.tools_called is False
    assert non_streaming.content == model_output

    content, tool_calls = _run_streaming(parser, model_output)
    assert content == model_output
    assert tool_calls == []


def test_streaming_empty_object_not_a_call(parser):
    # '{}' is valid, complete JSON but has no "name" key, so it must also
    # be flushed as content rather than buffered forever.
    model_output = "{}"

    content, tool_calls = _run_streaming(parser, model_output)
    assert content == model_output
    assert tool_calls == []


def test_streaming_malformed_interim_no_premature_flush(parser):
    # While the buffered object is still incomplete, extract_tool_calls_
    # streaming must keep buffering (return None) rather than guessing that
    # it isn't a tool call. The flush to content may only happen once the
    # object is a complete, valid JSON value with no "name" key - never
    # while it's still malformed/partial.
    model_output = '{"a": 1} is a dict'
    json_object = '{"a": 1}'  # the buffered object; everything after is trailing text

    parser_state = parser
    previous_text = ""
    for i in range(len(json_object) - 1):
        # Every prefix up to (but not including) the closing '}' is
        # incomplete JSON; none of these deltas may surface content yet.
        current_text = model_output[: i + 1]
        delta_text = current_text[len(previous_text) :]
        delta = parser_state.extract_tool_calls_streaming(
            previous_text, current_text, delta_text, [], [], [], None
        )
        assert delta is None, (
            f"expected no premature flush while buffering {current_text!r}, "
            f"got {delta!r}"
        )
        previous_text = current_text

    # Once the object completes ('{"a": 1}'), it must flush as content.
    current_text = model_output[: len(json_object)]
    delta_text = current_text[len(previous_text) :]
    delta = parser_state.extract_tool_calls_streaming(
        previous_text, current_text, delta_text, [], [], [], None
    )
    assert delta is not None
    assert delta.content == current_text
    assert delta.tool_calls == []


def test_streaming_real_tool_call_unaffected(parser):
    # Sanity check that the content-passthrough fix doesn't regress the
    # ordinary streaming tool-call path.
    model_output = '{"name": "get_weather", "parameters": {"city": "Tokyo"}}'

    content, tool_calls = _run_streaming(parser, model_output)
    assert content is None
    assert len(tool_calls) == 1
    name, arguments = tool_calls[0]
    assert name == "get_weather"
    assert json.loads(arguments) == {"city": "Tokyo"}


def test_streaming_known_limitation_call_after_plain_json_answer(parser):
    # Known limitation, discussed on PR #58829: once the content-passthrough
    # latch trips for a leading plain JSON answer, it never resets. A
    # genuine tool call appearing later in the same stream is therefore also
    # streamed as content instead of being parsed. This is the same class of
    # mismatch as "text before a call" (case 3 in #58824), which is left to
    # the broader ParserEngine port in #51577. This test pins the current,
    # accepted behavior so a future change to it is deliberate.
    model_output = (
        '{"a": 1} is a dict, then: '
        '{"name": "get_weather", "parameters": {"city": "Tokyo"}}'
    )

    content, tool_calls = _run_streaming(parser, model_output)
    assert content == model_output
    assert tool_calls == []


def test_regex_timeout_handling(parser):
    """Test regex timeout is handled gracefully."""
    fake_problematic_input = "{hello world[A(A=" + "\t)A(A=,\t" * 2

    # create a mock regex that raises TimeoutError
    mock_regex = MagicMock()
    mock_regex.finditer.side_effect = TimeoutError("Regex timeout")

    with patch.object(parser, "tool_call_start_regex", mock_regex):
        result = parser.extract_tool_calls(fake_problematic_input, None)

        # should treat as regular text when regex times out
        assert result.content == fake_problematic_input
        assert result.tools_called is False
        assert len(result.tool_calls) == 0
        mock_regex.finditer.assert_called_once()
